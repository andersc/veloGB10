//! DSpark drafter, CPU reference (S8): the `Naive-N0.5-Flash-FP8-Draft` network op for op after the
//! shipped `dflash.py` / `dspark.py` and vLLM's dspark speculator (PLAN §1.2), f32.
//!
//! One speculation round: the target's hidden states after the 8 tap layers (concatenated,
//! `[T][8*H]`) become the context through `hidden_norm(fc(.))`; the query block is the anchor token
//! (real embedding) plus `block_size - 1` mask slots (the learned `mask_embedding`) at positions
//! T..T+B-1; every draft layer attends NON-causally over context ++ block (window 1024 is never
//! binding at these sizes — honoured for T+B > 1024 by the mask); the drafter's final-norm hidden
//! goes through the target `lm_head`; the Markov head adds `markov_w2(markov_w1(prev))` per position
//! sequentially (prev = anchor, then the drafted token); the confidence head scores each position.
//!
//! Gate: `tests/fixtures/naive/drafter` (scripts/naive/drafter_oracle.py) — hidden, logits, tokens,
//! confidence must match the HF modules.

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

use super::cpu::{rel_l2, rms_norm, rope_apply, rope_tables, Mat};

#[derive(Clone, Debug)]
pub struct DrafterConfig {
    pub hidden: usize,      // 4096
    pub inter: usize,       // 4096
    pub layers: usize,      // 5
    pub heads: usize,       // 32
    pub kv_heads: usize,    // 4
    pub head_dim: usize,    // 128 (full rotary)
    pub eps: f32,
    pub rope_theta: f32,
    pub sliding_window: usize, // 1024
    pub block: usize,       // 7
    pub mask_token: u32,
    pub taps: Vec<usize>,   // target layer ids (hidden_states[l+1])
    pub markov_rank: usize,
    pub confidence: bool,
    pub vocab: usize,
}

impl DrafterConfig {
    pub fn from_json(raw: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(raw)?;
        let u = |k: &str| -> Result<usize> { v[k].as_u64().map(|x| x as usize).with_context(|| format!("drafter config: {k}")) };
        let dc = &v["dflash_config"];
        let theta = v["rope_parameters"]["rope_theta"].as_f64().or(v["rope_theta"].as_f64()).unwrap_or(10000.0) as f32;
        ensure!(dc["causal"].as_bool() != Some(true), "the drafter must run non-causal within the block (dflash_config.causal=false)");
        ensure!(dc["use_mask_embedding"].as_bool() == Some(true), "the Naive drafter uses a learned mask embedding");
        Ok(DrafterConfig {
            hidden: u("hidden_size")?, inter: u("intermediate_size")?, layers: u("num_hidden_layers")?,
            heads: u("num_attention_heads")?, kv_heads: u("num_key_value_heads")?, head_dim: u("head_dim")?,
            eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32, rope_theta: theta,
            sliding_window: v["sliding_window"].as_u64().unwrap_or(1 << 30) as usize,
            block: u("block_size")?,
            mask_token: dc["mask_token_id"].as_u64().context("mask_token_id")? as u32,
            taps: dc["target_layer_ids"].as_array().context("target_layer_ids")?.iter().map(|x| x.as_u64().unwrap() as usize).collect(),
            markov_rank: v["markov_rank"].as_u64().unwrap_or(0) as usize,
            confidence: v["enable_confidence_head"].as_bool().unwrap_or(false),
            vocab: u("vocab_size")?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct DrafterLayer {
    pub in_norm: Vec<f32>, pub post_norm: Vec<f32>,
    pub q: Mat, pub k: Mat, pub v: Mat, pub o: Mat,
    pub q_norm: Vec<f32>, pub k_norm: Vec<f32>, // per-head RMSNorm weights [hd]
    pub gate: Mat, pub up: Mat, pub down: Mat,
}

#[derive(Clone, Debug)]
pub struct CpuDrafter {
    pub cfg: DrafterConfig,
    pub fc: Mat,              // [H, taps*H]
    pub hidden_norm: Vec<f32>,
    pub mask_embedding: Vec<f32>,
    pub layers: Vec<DrafterLayer>,
    pub norm: Vec<f32>,
    pub markov_w1: Vec<f32>,  // [V][r]
    pub markov_w2: Mat,       // [V, r]  (bias = markov_w2(markov_w1[prev]))
    pub conf_w: Option<(Vec<f32>, f32)>, // ([H + r], bias)
}

#[derive(Clone, Debug)]
pub struct DraftRound {
    pub hidden: Vec<f32>,        // [B][H] final-norm hidden
    pub base_logits: Vec<f32>,   // [B][V]
    pub corrected: Vec<f32>,     // [B][V]
    pub tokens: Vec<u32>,        // [B] greedy drafts (position T+i+1)
    pub confidence: Vec<f32>,    // [B]
}

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

impl CpuDrafter {
    /// One round. `taps_concat` = `[T][taps*H]` target hidden states (the reference's
    /// `extract_context_feature`), `embed` the target embedding table `[V][H]`, `lm_head` `[V, H]`.
    pub fn round(&self, taps_concat: &[f32], t: usize, anchor: u32, embed: &[f32], lm_head: &Mat) -> Result<DraftRound> {
        let c = &self.cfg;
        let (h, b, nh, kv, hd) = (c.hidden, c.block, c.heads, c.kv_heads, c.head_dim);
        ensure!(taps_concat.len() == t * c.taps.len() * h, "taps_concat size");
        let gqa = nh / kv;
        let n = t + b;
        // context feature: hidden_norm(fc(concat))  [T][H]
        let ctx = rms_norm(&self.fc.apply(taps_concat, t), &self.hidden_norm, t, c.eps);
        // block input: anchor embedding, then the mask embedding
        let mut x = vec![0f32; b * h];
        x[..h].copy_from_slice(&embed[anchor as usize * h..(anchor as usize + 1) * h]);
        for i in 1..b { x[i * h..(i + 1) * h].copy_from_slice(&self.mask_embedding); }
        let (cos, sin) = rope_tables(c.rope_theta, hd, n); // full-dim rotary (no partial factor)
        let scale = (hd as f32).powf(-0.5);
        for l in &self.layers {
            let hn = rms_norm(&x, &l.in_norm, b, c.eps);
            // q for the block rows; k/v for context ++ block
            let mut q = l.q.apply(&hn, b);                      // [b][nh*hd]
            let mut kc = l.k.apply(&ctx, t); kc.extend(l.k.apply(&hn, b)); // [n][kv*hd]
            let mut vc = l.v.apply(&ctx, t); vc.extend(l.v.apply(&hn, b));
            // per-head RMSNorm on q and k, then rope (q at positions t..t+b-1, k at 0..n-1)
            for i in 0..b { for hh in 0..nh {
                let s = &mut q[i * nh * hd + hh * hd..i * nh * hd + (hh + 1) * hd];
                let y = rms_norm(s, &l.q_norm, 1, c.eps); s.copy_from_slice(&y);
                rope_apply(s, &cos, &sin, t + i, hd);
            } }
            for p in 0..n { for g in 0..kv {
                let s = &mut kc[p * kv * hd + g * hd..p * kv * hd + (g + 1) * hd];
                let y = rms_norm(s, &l.k_norm, 1, c.eps); s.copy_from_slice(&y);
                rope_apply(s, &cos, &sin, p, hd);
            } }
            // non-causal attention over all n keys (window honoured only when it binds)
            let mut out = vec![0f32; b * nh * hd];
            for hh in 0..nh {
                let g = hh / gqa;
                for i in 0..b {
                    let qi = t + i;
                    let qa = &q[i * nh * hd + hh * hd..i * nh * hd + (hh + 1) * hd];
                    let mut lg = vec![f32::NEG_INFINITY; n];
                    let mut mx = f32::NEG_INFINITY;
                    for p in 0..n {
                        if qi >= c.sliding_window && p + c.sliding_window <= qi { continue; } // |q - k| >= window: masked
                        let ka = &kc[p * kv * hd + g * hd..p * kv * hd + (g + 1) * hd];
                        let mut d = 0f32; for j in 0..hd { d += qa[j] * ka[j]; }
                        lg[p] = d * scale; mx = mx.max(lg[p]);
                    }
                    let mut den = 0f32; for p in 0..n { if lg[p] > f32::NEG_INFINITY { den += (lg[p] - mx).exp(); } }
                    let o = &mut out[i * nh * hd + hh * hd..i * nh * hd + (hh + 1) * hd];
                    for p in 0..n {
                        if lg[p] == f32::NEG_INFINITY { continue; }
                        let w = (lg[p] - mx).exp() / den;
                        let va = &vc[p * kv * hd + g * hd..p * kv * hd + (g + 1) * hd];
                        for j in 0..hd { o[j] += w * va[j]; }
                    }
                }
            }
            let a = l.o.apply(&out, b);
            for i in 0..b * h { x[i] += a[i]; }
            let hn2 = rms_norm(&x, &l.post_norm, b, c.eps);
            let gt = l.gate.apply(&hn2, b); let up = l.up.apply(&hn2, b);
            let act: Vec<f32> = gt.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
            let d = l.down.apply(&act, b);
            for i in 0..b * h { x[i] += d[i]; }
        }
        let hidden = rms_norm(&x, &self.norm, b, c.eps);
        let base = lm_head.apply(&hidden, b); // [b][V]
        let v = c.vocab;
        let r = c.markov_rank;
        let mut corrected = base.clone();
        let mut tokens = vec![0u32; b];
        let mut conf = vec![0f32; b];
        let mut prev = anchor as usize;
        for i in 0..b {
            let me = &self.markov_w1[prev * r..(prev + 1) * r];
            let bias = self.markov_w2.apply(me, 1); // [V]
            let row = &mut corrected[i * v..(i + 1) * v];
            for j in 0..v { row[j] += bias[j]; }
            let am = row.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b2, (j, &y)| if y > b2.1 { (j, y) } else { b2 }).0;
            tokens[i] = am as u32;
            if let Some((w, bias_c)) = &self.conf_w {
                let mut s = *bias_c;
                for j in 0..h { s += w[j] * hidden[i * h + j]; }
                for j in 0..r { s += w[h + j] * me[j]; }
                conf[i] = sigmoid(s);
            }
            prev = am;
        }
        Ok(DraftRound { hidden, base_logits: base, corrected, tokens, confidence: conf })
    }

    /// Plain f32/bf16 safetensors in the shipped naming (+ optional `target.*` tensors ignored here).
    pub fn load_plain(dir: &str, file: &str) -> Result<CpuDrafter> {
        let cfg = DrafterConfig::from_json(&std::fs::read_to_string(format!("{}/config.json", dir.trim_end_matches('/')))?)?;
        let bytes = std::fs::read(format!("{}/{file}", dir.trim_end_matches('/'))).context("read drafter safetensors")?;
        let st = safetensors::SafeTensors::deserialize(&bytes)?;
        let f32s = |name: &str| -> Result<Vec<f32>> {
            let t = st.tensor(name).with_context(|| format!("drafter tensor {name} missing"))?;
            Ok(match t.dtype() {
                safetensors::Dtype::F32 => t.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                safetensors::Dtype::BF16 => t.data().chunks_exact(2).map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
                safetensors::Dtype::F16 => t.data().chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
                other => bail!("{name}: dtype {other:?}"),
            })
        };
        let mat = |name: &str, out: usize, inp: usize| -> Result<Mat> { Mat::new(out, inp, f32s(name)?) };
        let (h, nh, kv, hd) = (cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim);
        let mut layers = Vec::new();
        for l in 0..cfg.layers {
            let p = format!("layers.{l}.");
            layers.push(DrafterLayer {
                in_norm: f32s(&format!("{p}input_layernorm.weight"))?, post_norm: f32s(&format!("{p}post_attention_layernorm.weight"))?,
                q: mat(&format!("{p}self_attn.q_proj.weight"), nh * hd, h)?, k: mat(&format!("{p}self_attn.k_proj.weight"), kv * hd, h)?,
                v: mat(&format!("{p}self_attn.v_proj.weight"), kv * hd, h)?, o: mat(&format!("{p}self_attn.o_proj.weight"), h, nh * hd)?,
                q_norm: f32s(&format!("{p}self_attn.q_norm.weight"))?, k_norm: f32s(&format!("{p}self_attn.k_norm.weight"))?,
                gate: mat(&format!("{p}mlp.gate_proj.weight"), cfg.inter, h)?, up: mat(&format!("{p}mlp.up_proj.weight"), cfg.inter, h)?,
                down: mat(&format!("{p}mlp.down_proj.weight"), h, cfg.inter)?,
            });
        }
        let r = cfg.markov_rank;
        let conf_w = if cfg.confidence {
            let w = f32s("confidence_head.proj.weight")?; ensure!(w.len() == h + r, "confidence proj width");
            Some((w, f32s("confidence_head.proj.bias")?[0]))
        } else { None };
        Ok(CpuDrafter {
            fc: mat("fc.weight", h, cfg.taps.len() * h)?, hidden_norm: f32s("hidden_norm.weight")?,
            mask_embedding: f32s("mask_embedding")?, layers, norm: f32s("norm.weight")?,
            markov_w1: f32s("markov_head.markov_w1.weight")?, markov_w2: mat("markov_head.markov_w2.weight", cfg.vocab, r)?,
            conf_w, cfg,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv4_load::{read_npz_key, NpyData};
    use std::path::Path;

    #[test]
    fn tiny_drafter_round_matches_the_hf_modules() {
        let dir = "tests/fixtures/naive/drafter";
        // oracle.npz is committed; drafter.safetensors (7.9 MB, *.safetensors is gitignored) is generated on the box:
        //   python3 scripts/naive/drafter_oracle.py tests/fixtures/naive/drafter   (needs torch + safetensors)
        if !Path::new(dir).join("oracle.npz").is_file() || !Path::new(dir).join("drafter.safetensors").is_file() {
            eprintln!("skip: drafter fixture missing (oracle.npz is committed; generate drafter.safetensors with: python3 scripts/naive/drafter_oracle.py tests/fixtures/naive/drafter)"); return;
        }
        let d = CpuDrafter::load_plain(dir, "drafter.safetensors").unwrap();
        let bytes = std::fs::read(Path::new(dir).join("drafter.safetensors")).unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let f = |n: &str| -> Vec<f32> { st.tensor(n).unwrap().data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
        let embed = f("target.embed_tokens.weight");
        let lm = Mat::new(d.cfg.vocab, d.cfg.hidden, f("target.lm_head.weight")).unwrap();
        let npz = Path::new(dir).join("oracle.npz");
        let g = |k: &str| match read_npz_key(&npz, k).unwrap().1 { NpyData::F32(v) => v, NpyData::I64(v) => v.iter().map(|x| *x as f32).collect() };
        let th = g("target_hidden");
        let t = th.len() / (d.cfg.taps.len() * d.cfg.hidden);
        let anchor = g("anchor")[0] as u32;
        let r = d.round(&th, t, anchor, &embed, &lm).unwrap();
        let e = rel_l2(&r.hidden, &g("hidden")); assert!(e < 2e-4, "hidden rel-L2 {e}");
        let e = rel_l2(&r.base_logits, &g("base_logits")); assert!(e < 2e-4, "base logits rel-L2 {e}");
        let e = rel_l2(&r.corrected, &g("corrected_logits")); assert!(e < 2e-4, "corrected logits rel-L2 {e}");
        let want: Vec<u32> = g("tokens").iter().map(|x| *x as u32).collect();
        assert_eq!(r.tokens, want, "drafted tokens");
        let wc = g("confidence");
        for i in 0..r.confidence.len() { assert!((r.confidence[i] - wc[i]).abs() < 1e-4, "confidence {i}: {} vs {}", r.confidence[i], wc[i]); }
    }
}
