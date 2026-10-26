//! Naive-N0.5-Flash CPU reference forward (S2): plain-Rust f32 reimplementation of the HF
//! reference `modeling_naive_n05_flash.py`, op for op, in the order the reference runs them.
//!
//! Purpose: the ground truth every GPU kernel and the GPU model are checked against. It is slow
//! and simple on purpose. Two weight sources feed it:
//!   * a plain bf16/f32 safetensors checkpoint in the HF naming (the tiny replica of
//!     `scripts/naive/tiny_replica.py`, and the taps it dumps) — [`CpuModel::load_plain`];
//!   * the EXL3 pack, dequantized on the host (S2b: `pack_dequant`), where `v_proj` is per-head
//!     padded and `attention_value_scale` is already folded into `o_proj` (`value_scale_folded`).
//!
//! Numerics policy: f32 everywhere (the reference runs in f32 for the oracle; the real model's
//! bf16/FP8 storage is a quantization question, not a semantics one). Dot products accumulate
//! sequentially, so a comparison against torch is tolerance-level (~1e-6 rel) except where the
//! reference rounds explicitly: the indexer's fp8 e4m3 round-trip is reproduced bit for bit
//! (RNE encoder from `dsv4_cpu`), the top-k is a STABLE descending sort (ties -> lower index), the
//! router's selection is on `sigmoid + bias` and its weights are the un-biased sigmoid scores.
//!
//! Reference facts this file encodes (modeling file line refs in comments): RMSNorm `w * x`
//! (Llama); GPT-NeoX partial rotary on the FIRST `rotary_dim` dims (`rotate_half`); SWA mask
//! `q - k < window`; the sink is an extra logit column per head that only widens the softmax
//! denominator; V is scaled by `attention_value_scale` before attention; the DSA indexer ropes q
//! and k with the ATTENTION layer's cos/sin, rounds both to fp8, scores `relu(q.k)` summed over
//! heads with `weights_proj(x) * heads^-0.5`, masks, and keeps the top `index_top_k` keys.

use anyhow::{bail, ensure, Context, Result};

use super::config::{AttnKind, NaiveConfig};

// ---------------------------------------------------------------------------------------------
// weights
// ---------------------------------------------------------------------------------------------

/// Row-major `[out, in]` matrix (nn.Linear layout: y = x @ W^T).
#[derive(Clone, Debug, Default)]
pub struct Mat {
    pub out: usize,
    pub inp: usize,
    pub w: Vec<f32>,
}

impl Mat {
    pub fn new(out: usize, inp: usize, w: Vec<f32>) -> Result<Mat> {
        ensure!(w.len() == out * inp, "Mat {out}x{inp}: {} values", w.len());
        Ok(Mat { out, inp, w })
    }
    /// `y[t, o] = sum_i x[t, i] * W[o, i]` for `t` rows of `x`.
    pub fn apply(&self, x: &[f32], rows: usize) -> Vec<f32> {
        debug_assert_eq!(x.len(), rows * self.inp);
        let mut y = vec![0f32; rows * self.out];
        for t in 0..rows {
            let xr = &x[t * self.inp..(t + 1) * self.inp];
            for o in 0..self.out {
                let wr = &self.w[o * self.inp..(o + 1) * self.inp];
                let mut acc = 0f32;
                for i in 0..self.inp {
                    acc += xr[i] * wr[i];
                }
                y[t * self.out + o] = acc;
            }
        }
        y
    }
}

#[derive(Clone, Debug)]
pub struct CpuIndexer {
    pub wq: Mat,           // [heads*hd, H]
    pub wk: Mat,           // [hd, H]
    pub k_norm_w: Vec<f32>, // LayerNorm weight [hd]
    pub k_norm_b: Vec<f32>, // LayerNorm bias [hd]
    pub weights_proj: Mat, // [heads, H] (the pack pads out to 128; the CPU model holds the true 16)
}

#[derive(Clone, Debug)]
pub enum CpuMlp {
    Dense { gate: Mat, up: Mat, down: Mat },
    Moe {
        gate_w: Mat,           // router [E, H] (f32)
        bias: Vec<f32>,        // e_score_correction_bias [E]
        experts: Vec<(Mat, Mat, Mat)>, // (gate [mi,H], up [mi,H], down [H,mi]) per expert
    },
    /// the real pack's experts, dequantized on demand
    PackMoe(PackMoe),
}

#[derive(Clone, Debug)]
pub struct CpuLayer {
    pub kind: AttnKind,
    pub in_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub q: Mat,  // [nh*hd, H]
    pub k: Mat,  // [kv*hd, H]
    pub v: Mat,  // [kv*vd, H] (compact: vd rows per kv head)
    pub o: Mat,  // [H, nh*vd]
    pub sink: Option<Vec<f32>>, // [nh]
    pub indexer: Option<CpuIndexer>,
    pub mlp: CpuMlp,
    /// routing-weight scale: routed_scaling_factor, times the converter's up_proj divisor when the
    /// experts come from the PACK (`NaiveConfig::pack_router_scale`); bf16 weights are undivided
    pub router_scale: f32,
}

#[derive(Clone, Debug)]
pub struct CpuModel {
    pub cfg: NaiveConfig,
    pub embed: Vec<f32>, // [V, H]
    pub norm: Vec<f32>,
    pub lm_head: Mat,    // [V, H]
    pub layers: Vec<CpuLayer>,
    /// true when `o_proj` already carries `attention_value_scale` (the EXL3 pack); the plain HF
    /// checkpoint keeps the scale on V.
    pub value_scale_folded: bool,
}

/// Every intermediate the oracle dump carries, for bisecting.
#[derive(Clone, Debug, Default)]
pub struct Taps {
    pub embed_out: Vec<f32>,
    pub layer_out: Vec<Vec<f32>>,          // per layer [T*H]
    pub router_ids: Vec<Option<Vec<usize>>>, // per layer, [T*K] (None on dense layers)
    pub router_wts: Vec<Option<Vec<f32>>>,
    pub indexer_scores: Vec<Option<Vec<f32>>>, // per layer, [T*T] before the mask
    pub dsa_selected: Vec<Option<Vec<bool>>>,  // per layer, [T*T]
    pub final_norm_out: Vec<f32>,
}

// ---------------------------------------------------------------------------------------------
// ops (reference order)
// ---------------------------------------------------------------------------------------------

/// LlamaRMSNorm: `x * rsqrt(mean(x^2) + eps) * w` (modeling: NaiveN05FlashRMSNorm = LlamaRMSNorm).
pub fn rms_norm(x: &[f32], w: &[f32], rows: usize, eps: f32) -> Vec<f32> {
    let h = w.len();
    let mut y = vec![0f32; rows * h];
    for t in 0..rows {
        let xr = &x[t * h..(t + 1) * h];
        let mut ss = 0f32;
        for v in xr { ss += v * v; }
        let inv = 1.0 / (ss / h as f32 + eps).sqrt();
        for i in 0..h { y[t * h + i] = xr[i] * inv * w[i]; }
    }
    y
}

/// torch.nn.LayerNorm (biased variance, eps inside the sqrt), weight and bias.
pub fn layer_norm(x: &[f32], w: &[f32], b: &[f32], rows: usize, eps: f32) -> Vec<f32> {
    let h = w.len();
    let mut y = vec![0f32; rows * h];
    for t in 0..rows {
        let xr = &x[t * h..(t + 1) * h];
        let mean = xr.iter().sum::<f32>() / h as f32;
        let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / h as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..h { y[t * h + i] = (xr[i] - mean) * inv * w[i] + b[i]; }
    }
    y
}

/// GPT-NeoX default rope tables for positions `0..t`: `cos/sin [t][rotary_dim]` with the two halves
/// duplicated (`emb = cat(freqs, freqs)`), `inv_freq = 1 / theta^(2i/rotary_dim)` in f32.
pub fn rope_tables(theta: f32, rotary_dim: usize, t: usize) -> (Vec<f32>, Vec<f32>) {
    let half = rotary_dim / 2;
    let inv: Vec<f32> = (0..half).map(|i| 1.0f32 / theta.powf((2 * i) as f32 / rotary_dim as f32)).collect();
    let mut cos = vec![0f32; t * rotary_dim];
    let mut sin = vec![0f32; t * rotary_dim];
    for p in 0..t {
        for i in 0..half {
            let f = p as f32 * inv[i];
            let (s, c) = f.sin_cos();
            cos[p * rotary_dim + i] = c; cos[p * rotary_dim + half + i] = c;
            sin[p * rotary_dim + i] = s; sin[p * rotary_dim + half + i] = s;
        }
    }
    (cos, sin)
}

/// `apply_rotary_pos_emb` on one head vector at position `p`: rotate the first `rotary_dim` dims
/// (`x_rot * cos + rotate_half(x_rot) * sin`, rotate_half = cat(-x2, x1)), pass the rest through.
pub fn rope_apply(x: &mut [f32], cos: &[f32], sin: &[f32], p: usize, rotary_dim: usize) {
    let half = rotary_dim / 2;
    let c = &cos[p * rotary_dim..(p + 1) * rotary_dim];
    let s = &sin[p * rotary_dim..(p + 1) * rotary_dim];
    let mut r = vec![0f32; rotary_dim];
    for i in 0..rotary_dim {
        let rot = if i < half { -x[i + half] } else { x[i - half] };
        r[i] = x[i] * c[i] + rot * s[i];
    }
    x[..rotary_dim].copy_from_slice(&r);
}

/// `round_indexer_fp8`: per-row `scale = max(|x|).clamp_min(1e-4) / 448`, `(x / scale).clamp(-448, 448)`
/// -> fp8 e4m3 (RNE) -> f32 `* scale`.
pub fn round_indexer_fp8(x: &mut [f32], rows: usize, n: usize) {
    for t in 0..rows {
        let r = &mut x[t * n..(t + 1) * n];
        let amax = r.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-4);
        let scale = amax / 448.0;
        for v in r.iter_mut() {
            let q = (*v / scale).clamp(-448.0, 448.0);
            let code = crate::dsv4_cpu::f32_to_e4m3_rne(q);
            *v = crate::quant::e4m3_to_f32(code) * scale;
        }
    }
}

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

/// Stable descending argsort of one row (ties -> lower index), first `k` entries.
pub fn topk_stable_desc(scores: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    idx
}

// ---------------------------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------------------------

impl CpuModel {
    /// Full prefill of `ids`; returns logits `[T, V]` and the taps.
    pub fn forward(&self, ids: &[u32]) -> Result<(Vec<f32>, Taps)> {
        let c = &self.cfg;
        let (t, h) = (ids.len(), c.hidden_size);
        ensure!(t > 0, "empty prompt");
        let mut taps = Taps::default();
        let mut x = vec![0f32; t * h];
        for (i, &id) in ids.iter().enumerate() {
            ensure!((id as usize) < c.vocab_size, "token {id} out of vocab");
            x[i * h..(i + 1) * h].copy_from_slice(&self.embed[id as usize * h..(id as usize + 1) * h]);
        }
        taps.embed_out = x.clone();
        for (li, layer) in self.layers.iter().enumerate() {
            let hn = rms_norm(&x, &layer.in_norm, t, c.rms_eps);
            let (attn, scores, selected) = self.attention(li, layer, &hn, t)?;
            for i in 0..t * h { x[i] += attn[i]; }
            let hn2 = rms_norm(&x, &layer.post_norm, t, c.rms_eps);
            let (mlp, rid, rw) = self.mlp(layer, &hn2, t)?;
            for i in 0..t * h { x[i] += mlp[i]; }
            taps.layer_out.push(x.clone());
            taps.router_ids.push(rid);
            taps.router_wts.push(rw);
            taps.indexer_scores.push(scores);
            taps.dsa_selected.push(selected);
        }
        let xn = rms_norm(&x, &self.norm, t, c.rms_eps);
        taps.final_norm_out = xn.clone();
        let logits = self.lm_head.apply(&xn, t);
        Ok((logits, taps))
    }

    /// One attention layer on the normed input. Returns (output [T*H], indexer scores, selection).
    #[allow(clippy::type_complexity)]
    fn attention(&self, li: usize, l: &CpuLayer, hn: &[f32], t: usize) -> Result<(Vec<f32>, Option<Vec<f32>>, Option<Vec<bool>>)> {
        let c = &self.cfg;
        let (nh, hd, vd) = (c.num_heads, c.head_dim, c.v_head_dim);
        let kv = c.kv_heads(li);
        let gqa = nh / kv;
        let rd = c.rotary_dim;
        let (cos, sin) = rope_tables(c.rope_theta_of(li), rd, t);

        let mut q = l.q.apply(hn, t); // [t][nh*hd]
        let mut k = l.k.apply(hn, t); // [t][kv*hd]
        let v = l.v.apply(hn, t);     // [t][kv*vd]
        for p in 0..t {
            for hh in 0..nh { rope_apply(&mut q[p * nh * hd + hh * hd..p * nh * hd + (hh + 1) * hd], &cos, &sin, p, rd); }
            for g in 0..kv { rope_apply(&mut k[p * kv * hd + g * hd..p * kv * hd + (g + 1) * hd], &cos, &sin, p, rd); }
        }

        // allowed[q][k]: causal, then the window or the indexer selection
        let mut allowed = vec![false; t * t];
        for qi in 0..t { for ki in 0..=qi { allowed[qi * t + ki] = true; } }
        let mut scores_tap = None;
        let mut selected_tap = None;
        match l.kind {
            AttnKind::Swa => {
                for qi in 0..t { for ki in 0..=qi { if qi - ki >= c.sliding_window { allowed[qi * t + ki] = false; } } }
            }
            AttnKind::Dsa => {
                let ix = l.indexer.as_ref().context("DSA layer without an indexer")?;
                let (ih, ihd) = (c.index_n_heads, c.index_head_dim);
                let mut iq = ix.wq.apply(hn, t); // [t][ih*ihd]
                let ik0 = ix.wk.apply(hn, t);     // [t][ihd]
                let mut ik = layer_norm(&ik0, &ix.k_norm_w, &ix.k_norm_b, t, 1e-5);
                for p in 0..t {
                    for hh in 0..ih { rope_apply(&mut iq[p * ih * ihd + hh * ihd..p * ih * ihd + (hh + 1) * ihd], &cos, &sin, p, rd); }
                    rope_apply(&mut ik[p * ihd..(p + 1) * ihd], &cos, &sin, p, rd);
                }
                if c.indexer_fp8 {
                    round_indexer_fp8(&mut iq, t * ih, ihd); // per (token, head) row: amax over the last dim
                    round_indexer_fp8(&mut ik, t, ihd);
                }
                let w = ix.weights_proj.apply(hn, t); // [t][ih]
                let wscale = (ih as f32).powf(-0.5);
                let mut sc = vec![0f32; t * t];
                for qi in 0..t {
                    for ki in 0..t {
                        let mut s = 0f32;
                        for hh in 0..ih {
                            let qa = &iq[qi * ih * ihd + hh * ihd..qi * ih * ihd + (hh + 1) * ihd];
                            let ka = &ik[ki * ihd..(ki + 1) * ihd];
                            let mut dot = 0f32;
                            for d in 0..ihd { dot += qa[d] * ka[d]; }
                            s += dot.max(0.0) * (w[qi * ih + hh] * wscale);
                        }
                        sc[qi * t + ki] = s;
                    }
                }
                scores_tap = Some(sc.clone());
                let mut sel = vec![false; t * t];
                for qi in 0..t {
                    let row: Vec<f32> = (0..t).map(|ki| if allowed[qi * t + ki] { sc[qi * t + ki] } else { f32::NEG_INFINITY }).collect();
                    for ki in topk_stable_desc(&row, c.index_top_k) { sel[qi * t + ki] = true; }
                }
                for i in 0..t * t { allowed[i] &= sel[i]; }
                selected_tap = Some(allowed.clone());
            }
        }

        let scale = (hd as f32).powf(-0.5);
        let vscale = if self.value_scale_folded { 1.0 } else { c.value_scale };
        let mut out = vec![0f32; t * nh * vd];
        for hh in 0..nh {
            let g = hh / gqa;
            for qi in 0..t {
                let qa = &q[qi * nh * hd + hh * hd..qi * nh * hd + (hh + 1) * hd];
                // logits over keys (masked -> -inf) plus the sink column
                let mut lg = vec![f32::NEG_INFINITY; t + 1];
                let mut mx = f32::NEG_INFINITY;
                for ki in 0..t {
                    if !allowed[qi * t + ki] { continue; }
                    let ka = &k[ki * kv * hd + g * hd..ki * kv * hd + (g + 1) * hd];
                    let mut dot = 0f32;
                    for d in 0..hd { dot += qa[d] * ka[d]; }
                    lg[ki] = dot * scale;
                    mx = mx.max(lg[ki]);
                }
                let has_sink = l.sink.is_some();
                if let Some(s) = &l.sink { lg[t] = s[hh]; mx = mx.max(lg[t]); }
                let mut den = 0f32;
                let n = if has_sink { t + 1 } else { t };
                for ki in 0..n { if lg[ki] > f32::NEG_INFINITY { den += (lg[ki] - mx).exp(); } }
                let o = &mut out[qi * nh * vd + hh * vd..qi * nh * vd + (hh + 1) * vd];
                for ki in 0..t {
                    if lg[ki] == f32::NEG_INFINITY { continue; }
                    let p = (lg[ki] - mx).exp() / den;
                    let va = &v[ki * kv * vd + g * vd..ki * kv * vd + (g + 1) * vd];
                    for d in 0..vd { o[d] += p * va[d] * vscale; }
                }
            }
        }
        Ok((l.o.apply(&out, t), scores_tap, selected_tap))
    }

    #[allow(clippy::type_complexity)]
    fn mlp(&self, l: &CpuLayer, hn: &[f32], t: usize) -> Result<(Vec<f32>, Option<Vec<usize>>, Option<Vec<f32>>)> {
        let c = &self.cfg;
        let h = c.hidden_size;
        match &l.mlp {
            CpuMlp::Dense { gate, up, down } => {
                let g = gate.apply(hn, t);
                let u = up.apply(hn, t);
                let inter = gate.out;
                let mut a = vec![0f32; t * inter];
                for i in 0..t * inter { a[i] = silu(g[i]) * u[i]; }
                Ok((down.apply(&a, t), None, None))
            }
            CpuMlp::Moe { .. } | CpuMlp::PackMoe(_) => {
                let (gate_w, bias): (&Mat, &Vec<f32>) = match &l.mlp {
                    CpuMlp::Moe { gate_w, bias, .. } => (gate_w, bias),
                    CpuMlp::PackMoe(pm) => (&pm.gate_w, &pm.bias),
                    _ => unreachable!(),
                };
                let expert = |ex: usize| -> Result<(Mat, Mat, Mat)> {
                    match &l.mlp {
                        CpuMlp::Moe { experts, .. } => Ok(experts[ex].clone()),
                        CpuMlp::PackMoe(pm) => pm.experts.get(ex),
                        _ => unreachable!(),
                    }
                };
                let (e, k) = (c.num_experts, c.num_experts_per_tok);
                let logits = gate_w.apply(hn, t); // f32 router
                let mut ids = vec![0usize; t * k];
                let mut wts = vec![0f32; t * k];
                let mut out = vec![0f32; t * h];
                for ti in 0..t {
                    let sc: Vec<f32> = (0..e).map(|j| sigmoid(logits[ti * e + j])).collect();
                    let choice: Vec<f32> = (0..e).map(|j| sc[j] + bias[j]).collect();
                    let top = topk_stable_desc(&choice, k); // n_group = topk_group = 1: plain top-k
                    let mut sum = 0f32;
                    for (j, &ex) in top.iter().enumerate() { ids[ti * k + j] = ex; wts[ti * k + j] = sc[ex]; sum += sc[ex]; }
                    if c.norm_topk_prob { for j in 0..k { wts[ti * k + j] /= sum + 1e-20; } }
                    for j in 0..k { wts[ti * k + j] *= l.router_scale; }
                    let xr = &hn[ti * h..(ti + 1) * h];
                    for j in 0..k {
                        let (gm, um, dm) = expert(ids[ti * k + j])?;
                        let g = gm.apply(xr, 1);
                        let u = um.apply(xr, 1);
                        let a: Vec<f32> = g.iter().zip(&u).map(|(g, u)| silu(*g) * u).collect();
                        let d = dm.apply(&a, 1);
                        let w = wts[ti * k + j];
                        for i in 0..h { out[ti * h + i] += w * d[i]; }
                    }
                }
                // a layer's dequantized experts are only reused within this call: drop them (a 48-layer
                // run would otherwise hold ~100 MB per routed expert for every layer)
                if let CpuMlp::PackMoe(pm) = &l.mlp { pm.experts.cache.borrow_mut().clear(); }
                Ok((out, Some(ids), Some(wts)))
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// plain checkpoint loader (the tiny replica; HF naming, bf16 / f32 tensors)
// ---------------------------------------------------------------------------------------------

fn st_f32(st: &safetensors::SafeTensors, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let t = st.tensor(name).with_context(|| format!("tensor {name} missing"))?;
    let data = t.data();
    let v = match t.dtype() {
        safetensors::Dtype::F32 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        safetensors::Dtype::BF16 => data.chunks_exact(2).map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
        safetensors::Dtype::F16 => data.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
        other => bail!("{name}: dtype {other:?} unsupported"),
    };
    Ok((t.shape().to_vec(), v))
}

fn st_mat(st: &safetensors::SafeTensors, name: &str, out: usize, inp: usize) -> Result<Mat> {
    let (shape, v) = st_f32(st, name)?;
    ensure!(shape == vec![out, inp], "{name}: shape {shape:?} (wanted [{out}, {inp}])");
    Mat::new(out, inp, v)
}

fn st_vec(st: &safetensors::SafeTensors, name: &str, n: usize) -> Result<Vec<f32>> {
    let (shape, v) = st_f32(st, name)?;
    ensure!(shape == vec![n], "{name}: shape {shape:?} (wanted [{n}])");
    Ok(v)
}

impl CpuModel {
    /// A plain checkpoint directory: `config.json` + `model.safetensors` in the HF naming.
    pub fn load_plain(dir: &str) -> Result<CpuModel> {
        let cfg = NaiveConfig::from_dir(dir)?;
        let bytes = std::fs::read(format!("{}/model.safetensors", dir.trim_end_matches('/'))).context("read model.safetensors")?;
        let st = safetensors::SafeTensors::deserialize(&bytes).context("parse model.safetensors")?;
        let c = &cfg;
        let (h, nh, hd, vd) = (c.hidden_size, c.num_heads, c.head_dim, c.v_head_dim);
        let (_, embed) = st_f32(&st, "model.embed_tokens.weight")?;
        ensure!(embed.len() == c.vocab_size * h, "embed size");
        let norm = st_vec(&st, "model.norm.weight", h)?;
        let lm_head = if c.tie_word_embeddings { Mat::new(c.vocab_size, h, embed.clone())? } else { st_mat(&st, "lm_head.weight", c.vocab_size, h)? };
        let mut layers = Vec::with_capacity(c.num_layers);
        for l in 0..c.num_layers {
            let p = format!("model.layers.{l}.");
            let kv = c.kv_heads(l);
            let indexer = if c.attn[l] == AttnKind::Dsa {
                let (ih, ihd) = (c.index_n_heads, c.index_head_dim);
                Some(CpuIndexer {
                    wq: st_mat(&st, &format!("{p}self_attn.indexer.wq.weight"), ih * ihd, h)?,
                    wk: st_mat(&st, &format!("{p}self_attn.indexer.wk.weight"), ihd, h)?,
                    k_norm_w: st_vec(&st, &format!("{p}self_attn.indexer.k_norm.weight"), ihd)?,
                    k_norm_b: st_vec(&st, &format!("{p}self_attn.indexer.k_norm.bias"), ihd)?,
                    weights_proj: st_mat(&st, &format!("{p}self_attn.indexer.weights_proj.weight"), ih, h)?,
                })
            } else { None };
            let sink = if c.has_sink(l) { Some(st_vec(&st, &format!("{p}self_attn.attention_sink_bias"), nh)?) } else { None };
            let mlp = if c.moe[l] {
                let (e, mi) = (c.num_experts, c.moe_intermediate_size);
                let mut experts = Vec::with_capacity(e);
                for ex in 0..e {
                    experts.push((
                        st_mat(&st, &format!("{p}mlp.experts.{ex}.gate_proj.weight"), mi, h)?,
                        st_mat(&st, &format!("{p}mlp.experts.{ex}.up_proj.weight"), mi, h)?,
                        st_mat(&st, &format!("{p}mlp.experts.{ex}.down_proj.weight"), h, mi)?,
                    ));
                }
                CpuMlp::Moe {
                    gate_w: st_mat(&st, &format!("{p}mlp.gate.weight"), e, h)?,
                    bias: st_vec(&st, &format!("{p}mlp.gate.e_score_correction_bias"), e)?,
                    experts,
                }
            } else {
                CpuMlp::Dense {
                    gate: st_mat(&st, &format!("{p}mlp.gate_proj.weight"), c.intermediate_size, h)?,
                    up: st_mat(&st, &format!("{p}mlp.up_proj.weight"), c.intermediate_size, h)?,
                    down: st_mat(&st, &format!("{p}mlp.down_proj.weight"), h, c.intermediate_size)?,
                }
            };
            layers.push(CpuLayer {
                kind: c.attn[l],
                router_scale: c.routed_scaling_factor,
                in_norm: st_vec(&st, &format!("{p}input_layernorm.weight"), h)?,
                post_norm: st_vec(&st, &format!("{p}post_attention_layernorm.weight"), h)?,
                q: st_mat(&st, &format!("{p}self_attn.q_proj.weight"), nh * hd, h)?,
                k: st_mat(&st, &format!("{p}self_attn.k_proj.weight"), kv * hd, h)?,
                v: st_mat(&st, &format!("{p}self_attn.v_proj.weight"), kv * vd, h)?,
                o: st_mat(&st, &format!("{p}self_attn.o_proj.weight"), h, nh * vd)?,
                sink, indexer, mlp,
            });
        }
        Ok(CpuModel { cfg, embed, norm, lm_head, layers, value_scale_folded: false })
    }
}

// ---------------------------------------------------------------------------------------------
// comparison helpers
// ---------------------------------------------------------------------------------------------

/// Relative L2 distance `|a - b| / |b|`.
pub fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut d = 0f64; let mut n = 0f64;
    for i in 0..a.len() { let e = (a[i] - b[i]) as f64; d += e * e; n += (b[i] as f64) * (b[i] as f64); }
    if n == 0.0 { return if d == 0.0 { 0.0 } else { f32::INFINITY }; }
    (d / n).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsv4_load::{read_npz_key, NpyData};
    use std::path::Path;

    fn fixture() -> Option<String> {
        let d = concat!(".", "/tests/fixtures/naive/tiny");
        // oracle.npz is committed; model.safetensors (12.5 MB, *.safetensors is gitignored) is generated on the box:
        //   python3 scripts/naive/tiny_replica.py build tests/fixtures/naive/tiny   (needs torch + safetensors)
        if Path::new(d).join("oracle.npz").is_file() && Path::new(d).join("model.safetensors").is_file() { Some(d.to_string()) }
        else { eprintln!("skip: tiny replica fixture missing (oracle.npz is committed; generate model.safetensors with: python3 scripts/naive/tiny_replica.py build tests/fixtures/naive/tiny)"); None }
    }
    fn f32s(npz: &Path, key: &str) -> Vec<f32> {
        match read_npz_key(npz, key).unwrap().1 { NpyData::F32(v) => v, NpyData::I64(v) => v.iter().map(|x| *x as f32).collect() }
    }
    fn i64s(npz: &Path, key: &str) -> Vec<i64> {
        match read_npz_key(npz, key).unwrap().1 { NpyData::I64(v) => v, NpyData::F32(v) => v.iter().map(|x| *x as i64).collect() }
    }

    #[test]
    fn rope_and_norm_sanity() {
        let (cos, sin) = rope_tables(10000.0, 16, 4);
        assert_eq!(cos.len(), 64);
        for i in 0..16 { assert!((cos[i] - 1.0).abs() < 1e-7 && sin[i].abs() < 1e-7, "position 0 is the identity"); }
        let mut x = vec![1.0f32; 48];
        rope_apply(&mut x, &cos, &sin, 0, 16);
        assert!(x.iter().all(|v| (v - 1.0).abs() < 1e-6));
        let y = rms_norm(&[3.0, 4.0], &[1.0, 1.0], 1, 0.0);
        assert!((y[0] - 3.0 / 3.5355339).abs() < 1e-6);
        assert_eq!(topk_stable_desc(&[1.0, 3.0, 3.0, 2.0], 2), vec![1, 2], "ties -> lower index");
    }

    #[test]
    fn fp8_round_trip_is_idempotent_and_bounded() {
        let mut x = vec![0.1, -2.5, 300.0, 1e-7, 0.0, 7.0, -0.3, 448.0];
        let n = x.len();
        let before = x.clone();
        round_indexer_fp8(&mut x, 1, n);
        let once = x.clone();
        round_indexer_fp8(&mut x, 1, n);
        for i in 0..n {
            assert!((x[i] - once[i]).abs() <= 1e-6 * once[i].abs().max(1.0), "idempotent at {i}");
            assert!((once[i] - before[i]).abs() <= before[i].abs() * 0.0625 + 1e-6, "e4m3 relative step at {i}: {} vs {}", once[i], before[i]);
        }
    }

    /// THE S2 gate: the Rust forward reproduces the HF reference on the tiny replica, tap by tap.
    #[test]
    fn tiny_replica_matches_the_hf_reference() {
        let Some(dir) = fixture() else { return };
        let npz = Path::new(&dir).join("oracle.npz");
        let m = CpuModel::load_plain(&dir).unwrap();
        let ids: Vec<u32> = i64s(&npz, "ids").iter().map(|x| *x as u32).collect();
        let t = ids.len();
        let (logits, taps) = m.forward(&ids).unwrap();
        let tol = 2e-4f32;
        let e = rel_l2(&taps.embed_out, &f32s(&npz, "embed_out"));
        assert!(e < 1e-6, "embed_out rel-L2 {e}");
        for l in 0..m.cfg.num_layers {
            let r = rel_l2(&taps.layer_out[l], &f32s(&npz, &format!("layer{l}_out")));
            assert!(r < tol, "layer {l} output rel-L2 {r}");
            if let Some(ids_l) = &taps.router_ids[l] {
                let want = i64s(&npz, &format!("layer{l}_router_ids"));
                let k = m.cfg.num_experts_per_tok;
                for ti in 0..t {
                    let mut a: Vec<i64> = ids_l[ti * k..(ti + 1) * k].iter().map(|x| *x as i64).collect(); a.sort();
                    let mut b: Vec<i64> = want[ti * k..(ti + 1) * k].to_vec(); b.sort();
                    assert_eq!(a, b, "layer {l} token {ti}: router selection");
                }
                // weights: compare as (expert -> weight) maps since torch's top-k order is unsorted
                let wwant = f32s(&npz, &format!("layer{l}_router_wts"));
                let wgot = taps.router_wts[l].as_ref().unwrap();
                for ti in 0..t {
                    for j in 0..k {
                        let ex = ids_l[ti * k + j] as i64;
                        let pos = (0..k).find(|&q| want[ti * k + q] == ex).unwrap();
                        let (a, b) = (wgot[ti * k + j], wwant[ti * k + pos]);
                        assert!((a - b).abs() < 1e-5, "layer {l} token {ti} expert {ex}: weight {a} vs {b}");
                    }
                }
            }
            if let Some(sc) = &taps.indexer_scores[l] {
                let r = rel_l2(sc, &f32s(&npz, &format!("layer{l}_indexer_scores")));
                assert!(r < 1e-4, "layer {l} indexer scores rel-L2 {r}");
                let sel = taps.dsa_selected[l].as_ref().unwrap();
                let want: Vec<bool> = i64s(&npz, &format!("layer{l}_dsa_selected")).iter().map(|x| *x != 0).collect();
                assert_eq!(sel, &want, "layer {l}: DSA selection differs");
                let per_row: Vec<usize> = (0..t).map(|q| (0..t).filter(|&kk| sel[q * t + kk]).count()).collect();
                assert!(per_row.iter().any(|&n| n == m.cfg.index_top_k), "the prompt must exceed top-k so the sparse path is exercised: {per_row:?}");
            }
        }
        let r = rel_l2(&taps.final_norm_out, &f32s(&npz, "final_norm_out"));
        assert!(r < tol, "final norm rel-L2 {r}");
        let want = f32s(&npz, "logits");
        let r = rel_l2(&logits, &want);
        assert!(r < tol, "logits rel-L2 {r}");
        let v = m.cfg.vocab_size;
        for ti in 0..t {
            let am = |x: &[f32]| x.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
            assert_eq!(am(&logits[ti * v..(ti + 1) * v]), am(&want[ti * v..(ti + 1) * v]), "argmax at token {ti}");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// pack-backed CPU model (S2b/S4 oracle): the first `n_layers` layers from dequantized EXL3 quads.
// Experts are dequantized lazily per routed expert (a layer's 256 experts are 25 GB in f32).
// ---------------------------------------------------------------------------------------------

use std::cell::RefCell;
use std::collections::HashMap;

/// Lazily dequantized routed experts of one layer (gate, up, down in nn.Linear layout).
pub struct LazyExperts {
    pack: std::sync::Arc<super::pack::NaivePack>,
    layer: usize,
    cache: RefCell<HashMap<usize, (Mat, Mat, Mat)>>,
}

impl LazyExperts {
    pub fn get(&self, e: usize) -> Result<(Mat, Mat, Mat)> {
        if let Some(v) = self.cache.borrow().get(&e) { return Ok(v.clone()); }
        let c = &self.pack.cfg;
        let (h, mi) = (c.hidden_size, c.moe_intermediate_size);
        let q = |w: &str| self.pack.quad(&format!("model.layers.{}.mlp.experts.{e}.{w}", self.layer));
        let g = Mat::new(mi, h, super::dequant::dequant_linear(&self.pack, q("gate_proj")?)?)?;
        let u = Mat::new(mi, h, super::dequant::dequant_linear(&self.pack, q("up_proj")?)?)?;
        let d = Mat::new(h, mi, super::dequant::dequant_linear(&self.pack, q("down_proj")?)?)?;
        self.cache.borrow_mut().insert(e, (g.clone(), u.clone(), d.clone()));
        Ok((g, u, d))
    }
}

impl std::fmt::Debug for LazyExperts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "LazyExperts(layer {})", self.layer) }
}

impl Clone for LazyExperts {
    fn clone(&self) -> Self { LazyExperts { pack: self.pack.clone(), layer: self.layer, cache: RefCell::new(self.cache.borrow().clone()) } }
}

/// The experts of a pack-backed layer live here instead of `CpuMlp::Moe`.
#[derive(Clone, Debug)]
pub struct PackMoe {
    pub gate_w: Mat,
    pub bias: Vec<f32>,
    pub experts: LazyExperts,
}

impl CpuModel {
    /// The first `n_layers` layers of the real pack on the CPU (+ final norm and lm_head). V is
    /// compacted from the per-head padded `v_proj`, `weights_proj` keeps its true 16 outputs, and
    /// `attention_value_scale` is already folded into `o_proj` (`value_scale_folded = true`).
    pub fn load_pack_slice(pack: std::sync::Arc<super::pack::NaivePack>, n_layers: usize) -> Result<CpuModel> {
        use super::dequant::dequant_linear;
        let c = pack.cfg.clone();
        ensure!(n_layers >= 1 && n_layers <= c.num_layers, "n_layers {n_layers}");
        let (h, nh, hd, vd) = (c.hidden_size, c.num_heads, c.head_dim, c.v_head_dim);
        let plain_f32 = |name: &str| -> Result<Vec<f32>> {
            let m = pack.plain_tensor(name)?;
            let raw = m.read_bytes(&pack.dir)?;
            Ok(match m.dtype.as_str() {
                "BF16" => raw.chunks_exact(2).map(|x| half::bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32()).collect(),
                "F16" => raw.chunks_exact(2).map(|x| half::f16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32()).collect(),
                "F32" => raw.chunks_exact(4).map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]])).collect(),
                other => bail!("{name}: dtype {other}"),
            })
        };
        let embed = plain_f32("model.embed_tokens.weight")?;
        let norm = plain_f32("model.norm.weight")?;
        let lm_head = Mat::new(c.vocab_size, h, dequant_linear(&pack, pack.quad("lm_head")?)?)?;
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let p = format!("model.layers.{l}.");
            let kv = c.kv_heads(l);
            let lin = |name: &str, out: usize, inp: usize| -> Result<Mat> {
                let q = pack.quad(&format!("{p}{name}"))?;
                ensure!(q.k == inp && q.n == out, "{p}{name}: stored {}x{} vs wanted in {inp} out {out}", q.k, q.n);
                Mat::new(out, inp, dequant_linear(&pack, q)?)
            };
            // v_proj: stored [kv*192 (padded), H] -> compact [kv*128, H]
            let vq = pack.quad(&format!("{p}self_attn.v_proj"))?;
            let vfull = dequant_linear(&pack, vq)?; // [kv*hd][H]
            let mut vcomp = vec![0f32; kv * vd * h];
            for g in 0..kv { for d in 0..vd { vcomp[(g * vd + d) * h..(g * vd + d + 1) * h].copy_from_slice(&vfull[(g * hd + d) * h..(g * hd + d + 1) * h]); } }
            let indexer = if c.attn[l] == AttnKind::Dsa {
                let (ih, ihd) = (c.index_n_heads, c.index_head_dim);
                let wq = pack.quad(&format!("{p}self_attn.indexer.weights_proj"))?;
                let wfull = dequant_linear(&pack, wq)?; // [128 (padded)][H]
                Some(CpuIndexer {
                    wq: lin("self_attn.indexer.wq", ih * ihd, h)?,
                    wk: lin("self_attn.indexer.wk", ihd, h)?,
                    k_norm_w: plain_f32(&format!("{p}self_attn.indexer.k_norm.weight"))?,
                    k_norm_b: plain_f32(&format!("{p}self_attn.indexer.k_norm.bias"))?,
                    weights_proj: Mat::new(ih, h, wfull[..ih * h].to_vec())?,
                })
            } else { None };
            let sink = if c.has_sink(l) { Some(plain_f32(&format!("{p}self_attn.attention_sink_bias"))?) } else { None };
            let mlp = if c.moe[l] {
                CpuMlp::PackMoe(PackMoe {
                    gate_w: Mat::new(c.num_experts, h, plain_f32(&format!("{p}mlp.gate.weight"))?)?,
                    bias: plain_f32(&format!("{p}mlp.gate.e_score_correction_bias"))?,
                    experts: LazyExperts { pack: pack.clone(), layer: l, cache: RefCell::new(HashMap::new()) },
                })
            } else {
                CpuMlp::Dense {
                    gate: lin("mlp.gate_proj", c.intermediate_size, h)?,
                    up: lin("mlp.up_proj", c.intermediate_size, h)?,
                    down: lin("mlp.down_proj", h, c.intermediate_size)?,
                }
            };
            layers.push(CpuLayer {
                kind: c.attn[l],
                router_scale: c.pack_router_scale(l),
                in_norm: plain_f32(&format!("{p}input_layernorm.weight"))?,
                post_norm: plain_f32(&format!("{p}post_attention_layernorm.weight"))?,
                q: lin("self_attn.q_proj", nh * hd, h)?,
                k: lin("self_attn.k_proj", kv * hd, h)?,
                v: Mat::new(kv * vd, h, vcomp)?,
                o: lin("self_attn.o_proj", h, nh * vd)?,
                sink, indexer, mlp,
            });
        }
        let mut cfg = c;
        cfg.num_layers = n_layers; // the slice IS the model for the oracle
        Ok(CpuModel { cfg, embed, norm, lm_head, layers, value_scale_folded: true })
    }
}

/// `--probe-naive-cpu --model-dir <pack> [--layers N] [--prompt TEXT]`: the real pack through the
/// first N layers on the CPU (N = 48 is the whole model: tens of minutes, no GPU). Prints the
/// residual RMS after each layer, the greedy next token per position and the top-5 at the end — the
/// GPU-free coherence check of the dequantized weights and the reference semantics on real data.
pub fn probe_cpu(dir: &str, n_layers: usize, prompt: &str) -> Result<()> {
    let pack = std::sync::Arc::new(super::pack::NaivePack::open(dir)?);
    let tok = crate::tokenizer::QwenTokenizer::from_file(&format!("{}/tokenizer.json", pack.dir)).context("tokenizer")?;
    let ids = tok.encode(prompt, false)?;
    ensure!(!ids.is_empty(), "empty prompt");
    println!("NAIVE CPU PROBE {} layers {n_layers} prompt {:?} -> {} tokens {:?}", pack.dir, prompt, ids.len(), ids);
    let t0 = std::time::Instant::now();
    let m = CpuModel::load_pack_slice(pack.clone(), n_layers)?;
    println!("  slice loaded (lazy experts) in {:.1} s", t0.elapsed().as_secs_f64());
    let t1 = std::time::Instant::now();
    let (logits, taps) = m.forward(&ids)?;
    println!("  forward {:.1} s", t1.elapsed().as_secs_f64());
    let (t, h, v) = (ids.len(), m.cfg.hidden_size, m.cfg.vocab_size);
    for (l, x) in taps.layer_out.iter().enumerate() {
        let rms = (x.iter().map(|a| (*a as f64) * (*a as f64)).sum::<f64>() / x.len() as f64).sqrt();
        let nan = x.iter().filter(|a| !a.is_finite()).count();
        let picks = taps.router_ids[l].as_ref().map(|r| format!(" router picks (t0): {:?}", &r[..m.cfg.num_experts_per_tok.min(r.len())])).unwrap_or_default();
        let sel = taps.dsa_selected[l].as_ref().map(|s| format!(" dsa keys/row: {:?}", (0..t).map(|q| (0..t).filter(|&k| s[q * t + k]).count()).collect::<Vec<_>>())).unwrap_or_default();
        println!("  layer {l:2} ({:?}): residual rms {rms:.4} nonfinite {nan}{picks}{sel}", m.cfg.attn[l]);
    }
    let _ = h;
    let mut line = String::new();
    for ti in 0..t {
        let row = &logits[ti * v..(ti + 1) * v];
        let am = row.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
        line += &format!("{:?} ", tok.decode(&[am as u32], false).unwrap_or_default());
    }
    println!("  greedy next token per position: {line}");
    let last = &logits[(t - 1) * v..t * v];
    let mut idx: Vec<usize> = (0..v).collect();
    idx.sort_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mx = last[idx[0]];
    let z: f32 = last.iter().map(|x| (x - mx).exp()).sum();
    println!("  top-5 after the prompt:");
    for &i in &idx[..5] { println!("    {:7} p={:.3} {:?}", i, (last[i] - mx).exp() / z, tok.decode(&[i as u32], false).unwrap_or_default()); }
    println!("NAIVE_CPU_DONE");
    Ok(())
}
