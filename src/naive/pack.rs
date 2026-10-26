//! Naive-N0.5-Flash EXL3 pack: census, inventory check and the quad catalogue (S1).
//!
//! The pack (`PLAN/NAIVE_N05_PORT.md` §1.1): exllamav3 1.5.x, mul1 codebook, 3.5 bpw trellises
//! (56 u16 words per 16x16 tile, half-integer K — `naive::frac`), a 6-bit `lm_head`, bf16
//! embeddings, 145,525 tensors in 24 shards. Tensor names are the HF reference's
//! (`model.layers.N.self_attn.q_proj`, `model.layers.N.mlp.experts.E.gate_proj`, ...).
//!
//! Only headers are read here (a census must not page 137 GB through memory); tensor bytes are
//! read by the consumers through `TensorMeta::read_bytes`. The inventory check is the loader's
//! wrong-model guard: every role's count and every quad's geometry must equal what the config
//! implies, or the pack is refused with the first offender named.
//!
//! This file copies the shape of `crate::exl3::Exl3Pack::open` for a different naming scheme; the
//! shared primitives it uses (`shard_header`, `TensorMeta`, the mul1 decode) are read-only reuse.

use anyhow::{bail, ensure, Context, Result};
use std::collections::BTreeMap;

use crate::exl3::{shard_header, TensorMeta};
use super::config::{AttnKind, NaiveConfig};
use super::frac::BitsK;

/// What a quantized matrix is, by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    AttnQ, AttnK, AttnV, AttnO,
    IdxWq, IdxWk, IdxWeights,
    DenseGate, DenseUp, DenseDown,
    ExpertGate, ExpertUp, ExpertDown,
    LmHead,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Role::AttnQ => "q_proj", Role::AttnK => "k_proj", Role::AttnV => "v_proj", Role::AttnO => "o_proj",
            Role::IdxWq => "indexer.wq", Role::IdxWk => "indexer.wk", Role::IdxWeights => "indexer.weights_proj",
            Role::DenseGate => "mlp.gate_proj", Role::DenseUp => "mlp.up_proj", Role::DenseDown => "mlp.down_proj",
            Role::ExpertGate => "experts.gate_proj", Role::ExpertUp => "experts.up_proj", Role::ExpertDown => "experts.down_proj",
            Role::LmHead => "lm_head",
        }
    }
}

/// One EXL3 quadruple `{trellis, suh, svh, mul1}` with its geometry.
#[derive(Clone, Debug)]
pub struct Quad {
    /// Module key without the suffix, e.g. `model.layers.3.mlp.experts.17.down_proj`.
    pub name: String,
    pub role: Role,
    pub layer: Option<usize>,
    pub expert: Option<usize>,
    /// Input features (rows of W^T as stored: `trellis.shape[0] * 16`).
    pub k: usize,
    /// Output features (`trellis.shape[1] * 16`), as STORED — padded to 128 for narrow outputs
    /// (`weights_proj` 16 -> 128) and per-head padded for `v_proj` (kv_heads * head_dim).
    pub n: usize,
    pub bits: BitsK,
    pub trellis: TensorMeta,
    pub suh: TensorMeta,
    pub svh: TensorMeta,
    pub mul1: TensorMeta,
}

/// Plain (non-trellis) tensor by name: norms, sinks, router, embeddings.
pub type Plain = BTreeMap<String, TensorMeta>;

#[derive(Clone, Debug, Default)]
pub struct QuantGlobals {
    pub quant_method: String,
    pub version: String,
    pub bits: f64,
    pub head_bits: f64,
    pub codebook: String,
}

pub struct NaivePack {
    pub dir: String,
    pub cfg: NaiveConfig,
    pub quant: QuantGlobals,
    /// a shipped shard dir (gb10_shard.json): (world, rank) — the inventory holds this rank's experts and
    /// its pre-sliced row quads; None = the whole pack
    pub shard: Option<(usize, usize)>,
    /// expert TP (gb10_shard.json deal `tp`, `naive::shipplan::DEAL_TP`): the shard holds EVERY expert, sliced —
    /// gate/up to n = mi / world columns, down to k = mi / world rows (the loader takes them as they are)
    pub etp: bool,
    pub quads: BTreeMap<String, Quad>,
    pub plain: Plain,
    pub shards: Vec<String>,
    pub total_bytes: usize,
    pub total_tensors: usize,
}

fn classify(name: &str) -> Option<(Role, Option<usize>, Option<usize>)> {
    if name == "lm_head" { return Some((Role::LmHead, None, None)); }
    let rest = name.strip_prefix("model.layers.")?;
    let (li, rest) = rest.split_once('.')?;
    let layer: usize = li.parse().ok()?;
    let role = match rest {
        "self_attn.q_proj" => Role::AttnQ,
        "self_attn.k_proj" => Role::AttnK,
        "self_attn.v_proj" => Role::AttnV,
        "self_attn.o_proj" => Role::AttnO,
        "self_attn.indexer.wq" => Role::IdxWq,
        "self_attn.indexer.wk" => Role::IdxWk,
        "self_attn.indexer.weights_proj" => Role::IdxWeights,
        "mlp.gate_proj" => Role::DenseGate,
        "mlp.up_proj" => Role::DenseUp,
        "mlp.down_proj" => Role::DenseDown,
        _ => {
            let e = rest.strip_prefix("mlp.experts.")?;
            let (ei, proj) = e.split_once('.')?;
            let expert: usize = ei.parse().ok()?;
            let role = match proj {
                "gate_proj" => Role::ExpertGate,
                "up_proj" => Role::ExpertUp,
                "down_proj" => Role::ExpertDown,
                _ => return None,
            };
            return Some((role, Some(layer), Some(expert)));
        }
    };
    Some((role, Some(layer), None))
}

impl NaivePack {
    /// Open + parse (headers only). Refuses: wrong family, wrong quant_method, non-mul1 codebook,
    /// incomplete quadruples, a trellis width that is not a quantizer rate, an inventory that
    /// disagrees with the config.
    pub fn open(dir: &str) -> Result<NaivePack> {
        let dir = dir.trim_end_matches('/').to_string();
        let cfg = NaiveConfig::from_dir(&dir)?;

        let qraw = std::fs::read_to_string(format!("{dir}/quantization_config.json"))
            .with_context(|| format!("naive pack {dir}: quantization_config.json missing"))?;
        let q: serde_json::Value = serde_json::from_str(&qraw).context("quantization_config.json unparsable")?;
        let quant = QuantGlobals {
            quant_method: q["quant_method"].as_str().unwrap_or("").to_string(),
            version: q["version"].as_str().unwrap_or("").to_string(),
            bits: q["bits"].as_f64().unwrap_or(0.0),
            head_bits: q["head_bits"].as_f64().unwrap_or(0.0),
            codebook: q["codebook"].as_str().unwrap_or("").to_string(),
        };
        ensure!(quant.quant_method == "exl3", "naive pack {dir}: quant_method {:?} (wanted exl3)", quant.quant_method);
        ensure!(quant.codebook == "mul1", "naive pack {dir}: codebook {:?} — only mul1 is implemented (half-integer rates exist for mul1 only)", quant.codebook);

        let iraw = std::fs::read_to_string(format!("{dir}/model.safetensors.index.json"))
            .with_context(|| format!("naive pack {dir}: model.safetensors.index.json missing"))?;
        let idx: serde_json::Value = serde_json::from_str(&iraw).context("index unparsable")?;
        let weight_map = idx["weight_map"].as_object().context("index has no weight_map")?;
        let mut shards: Vec<String> = weight_map.values().filter_map(|v| v.as_str().map(str::to_string)).collect();
        shards.sort();
        shards.dedup();

        // name -> TensorMeta from every shard header
        let mut metas: BTreeMap<String, TensorMeta> = BTreeMap::new();
        let mut total_bytes = 0usize;
        for s in &shards {
            for (name, e) in shard_header(&format!("{dir}/{s}"))? {
                total_bytes += e.n_bytes;
                metas.insert(name.clone(), TensorMeta { name, shard: s.clone(), dtype: e.dtype, shape: e.shape, offset: e.offset, n_bytes: e.n_bytes });
            }
        }
        ensure!(metas.len() == weight_map.len(), "naive pack {dir}: {} tensors in shard headers, {} in the index", metas.len(), weight_map.len());
        for name in weight_map.keys() {
            ensure!(metas.contains_key(name), "naive pack {dir}: index names {name} but no shard header has it");
        }
        let total_tensors = metas.len();

        // quads
        let mut quads = BTreeMap::new();
        let mut plain = Plain::new();
        let mut quad_members: std::collections::BTreeSet<String> = Default::default();
        for (name, m) in &metas {
            let Some(module) = name.strip_suffix(".trellis") else { continue };
            ensure!(m.shape.len() == 3, "{name}: trellis rank {} (wanted [k/16, n/16, words])", m.shape.len());
            let pick = |suffix: &str| -> Result<TensorMeta> {
                metas.get(&format!("{module}.{suffix}")).cloned().with_context(|| format!("naive pack {dir}: module {module} is missing .{suffix}"))
            };
            let (suh, svh, mul1) = (pick("suh")?, pick("svh")?, pick("mul1")?);
            let bits = BitsK::from_words(m.shape[2]).with_context(|| format!("{name}"))?;
            let (k, n) = (m.shape[0] * 16, m.shape[1] * 16);
            ensure!(suh.shape == vec![k] && svh.shape == vec![n], "{module}: suh {:?} / svh {:?} do not match k {k} n {n}", suh.shape, svh.shape);
            ensure!(m.dtype == "I16" && suh.dtype == "F16" && svh.dtype == "F16" && mul1.dtype == "I32",
                    "{module}: dtypes trellis {} suh {} svh {} mul1 {}", m.dtype, suh.dtype, svh.dtype, mul1.dtype);
            let (role, layer, expert) = classify(module).with_context(|| format!("naive pack {dir}: unrecognized quantized module {module}"))?;
            for s in ["trellis", "suh", "svh", "mul1"] { quad_members.insert(format!("{module}.{s}")); }
            quads.insert(module.to_string(), Quad { name: module.to_string(), role, layer, expert, k, n, bits, trellis: m.clone(), suh, svh, mul1 });
        }
        for (name, m) in &metas {
            if !quad_members.contains(name) { plain.insert(name.clone(), m.clone()); }
        }

        let (shard, etp) = match crate::shard_plan::read_shard_info(std::path::Path::new(&dir))? {
            Some(info) => {
                ensure!(info.format == "naive", "naive pack {dir}: gb10_shard.json is a {} shard", info.format);
                (Some((info.world, info.rank)), info.deal == super::shipplan::DEAL_TP)
            }
            None => (None, false),
        };
        let pack = NaivePack { dir, cfg, quant, shard, etp, quads, plain, shards, total_bytes, total_tensors };
        pack.verify_inventory()?;
        Ok(pack)
    }

    /// Every count and every geometry the config implies; the first offender is named.
    pub fn verify_inventory(&self) -> Result<()> {
        let c = &self.cfg;
        let h = c.hidden_size;
        // a shipped shard: this rank's experts (e % world == rank) and the row quads pre-sliced to k / world
        let (world, rank) = self.shard.unwrap_or((1, 0));
        let mut counts: BTreeMap<Role, usize> = BTreeMap::new();
        for q in self.quads.values() {
            *counts.entry(q.role).or_default() += 1;
            let l = q.layer.unwrap_or(0);
            let (want_k, want_n): (usize, usize) = match q.role {
                Role::AttnQ => (h, c.num_heads * c.head_dim),
                Role::AttnK => (h, c.kv_heads(l) * c.head_dim),
                // v_proj: each kv head's v_head_dim rows padded to head_dim (exllamav3 naive_vproj)
                Role::AttnV => (h, c.kv_heads(l) * c.head_dim),
                Role::AttnO => (c.num_heads * c.v_head_dim / world, h),
                Role::IdxWq => (h, c.index_n_heads * c.index_head_dim),
                Role::IdxWk => (h, c.index_head_dim.max(128)),
                Role::IdxWeights => (h, c.index_n_heads.max(128)), // 16 -> padded to 128
                Role::DenseGate | Role::DenseUp => (h, c.intermediate_size),
                Role::DenseDown => (c.intermediate_size / world, h),
                Role::ExpertGate | Role::ExpertUp => (h, c.moe_intermediate_size / if self.etp { world } else { 1 }),
                Role::ExpertDown => (c.moe_intermediate_size / if self.etp { world } else { 1 }, h),
                Role::LmHead => (h, c.vocab_size),
            };
            ensure!((q.k, q.n) == (want_k, want_n), "{}: geometry k {} n {} (wanted k {want_k} n {want_n})", q.name, q.k, q.n);
            match q.role {
                Role::LmHead => ensure!(q.bits == BitsK::Int(self.quant.head_bits as usize), "lm_head bits {:?} vs head_bits {}", q.bits, self.quant.head_bits),
                // mixed-rate packs (attention/indexer/dense at one rate, experts at another) are legal: every tensor
                // carries its own rate and the kernels dispatch per tensor (48/56/64/80/96-word tiles = 3/3.5/4/5/6
                // bpw); `quantization_config.bits` is the converter's default rate, not a promise about every tensor
                _ => ensure!(matches!(q.bits.words(), 48 | 56 | 64 | 80 | 96), "{}: {:?} ({} bpw, {} words/tile) has no kernel (3, 3.5, 4, 5 or 6 bpw)", q.name, q.bits, q.bits.bpw(), q.bits.words()),
            }
            if let Some(l) = q.layer {
                ensure!(l < c.num_layers, "{}: layer {l} out of range", q.name);
                let is_idx = matches!(q.role, Role::IdxWq | Role::IdxWk | Role::IdxWeights);
                if is_idx { ensure!(c.attn[l] == AttnKind::Dsa, "{}: indexer on a sliding-window layer", q.name); }
                let is_dense = matches!(q.role, Role::DenseGate | Role::DenseUp | Role::DenseDown);
                if is_dense { ensure!(!c.moe[l], "{}: dense MLP on a MoE layer", q.name); }
                let is_exp = matches!(q.role, Role::ExpertGate | Role::ExpertUp | Role::ExpertDown);
                if is_exp {
                    ensure!(c.moe[l], "{}: experts on a dense layer", q.name);
                    ensure!(q.expert.unwrap() < c.num_experts, "{}: expert out of range", q.name);
                    ensure!(self.etp || q.expert.unwrap() % world == rank, "{}: expert {} is not rank {rank}'s (world {world})", q.name, q.expert.unwrap());
                }
            }
        }
        let ew = if self.etp { 1 } else { world };   // expert TP: every expert on every rank
        let want: [(Role, usize); 14] = [
            (Role::AttnQ, c.num_layers), (Role::AttnK, c.num_layers), (Role::AttnV, c.num_layers), (Role::AttnO, c.num_layers),
            (Role::IdxWq, c.n_dsa()), (Role::IdxWk, c.n_dsa()), (Role::IdxWeights, c.n_dsa()),
            (Role::DenseGate, c.n_dense()), (Role::DenseUp, c.n_dense()), (Role::DenseDown, c.n_dense()),
            (Role::ExpertGate, c.n_moe() * c.num_experts / ew), (Role::ExpertUp, c.n_moe() * c.num_experts / ew), (Role::ExpertDown, c.n_moe() * c.num_experts / ew),
            (Role::LmHead, if c.tie_word_embeddings { 0 } else { 1 }),
        ];
        for (role, n) in want {
            let got = counts.get(&role).copied().unwrap_or(0);
            ensure!(got == n, "{} quads: {got} (wanted {n})", role.name());
        }

        // plain tensors: exact set
        let mut want_plain: BTreeMap<String, (&str, Vec<usize>)> = BTreeMap::new();
        want_plain.insert("model.embed_tokens.weight".into(), ("BF16", vec![c.vocab_size, h]));
        want_plain.insert("model.norm.weight".into(), ("BF16", vec![h]));
        for l in 0..c.num_layers {
            want_plain.insert(format!("model.layers.{l}.input_layernorm.weight"), ("BF16", vec![h]));
            want_plain.insert(format!("model.layers.{l}.post_attention_layernorm.weight"), ("BF16", vec![h]));
            if c.has_sink(l) { want_plain.insert(format!("model.layers.{l}.self_attn.attention_sink_bias"), ("F16", vec![c.num_heads])); }
            if c.attn[l] == AttnKind::Dsa {
                want_plain.insert(format!("model.layers.{l}.self_attn.indexer.k_norm.weight"), ("F16", vec![c.index_head_dim]));
                want_plain.insert(format!("model.layers.{l}.self_attn.indexer.k_norm.bias"), ("F16", vec![c.index_head_dim]));
            }
            if c.moe[l] {
                want_plain.insert(format!("model.layers.{l}.mlp.gate.weight"), ("*", vec![c.num_experts, h]));
                want_plain.insert(format!("model.layers.{l}.mlp.gate.e_score_correction_bias"), ("*", vec![c.num_experts]));
            }
        }
        for (name, (dt, shape)) in &want_plain {
            let m = self.plain.get(name).with_context(|| format!("plain tensor {name} missing"))?;
            ensure!(&m.shape == shape, "{name}: shape {:?} (wanted {shape:?})", m.shape);
            if *dt != "*" { ensure!(m.dtype == *dt, "{name}: dtype {} (wanted {dt})", m.dtype); }
            else { ensure!(matches!(m.dtype.as_str(), "F16" | "BF16" | "F32"), "{name}: dtype {}", m.dtype); }
        }
        for name in self.plain.keys() {
            ensure!(want_plain.contains_key(name), "unexpected plain tensor {name}");
        }
        Ok(())
    }

    pub fn quad(&self, name: &str) -> Result<&Quad> {
        self.quads.get(name).with_context(|| format!("naive pack {}: missing quad {name}", self.dir))
    }
    pub fn plain_tensor(&self, name: &str) -> Result<&TensorMeta> {
        self.plain.get(name).with_context(|| format!("naive pack {}: missing plain tensor {name}", self.dir))
    }

    /// Bytes of the pack by class — the per-node weight budget comes from this.
    pub fn bytes_by_role(&self) -> BTreeMap<Role, usize> {
        let mut m: BTreeMap<Role, usize> = BTreeMap::new();
        for q in self.quads.values() {
            *m.entry(q.role).or_default() += q.trellis.n_bytes + q.suh.n_bytes + q.svh.n_bytes;
        }
        m
    }
}

/// `--probe-naive-pack --model-dir <pack>`: census + inventory check, headers only.
pub fn probe(dir: &str) -> Result<()> {
    let t0 = std::time::Instant::now();
    let p = NaivePack::open(dir)?;
    let c = &p.cfg;
    println!("NAIVE PACK {}", p.dir);
    println!("  exl3 {} codebook {} bits {} head_bits {}", p.quant.version, p.quant.codebook, p.quant.bits, p.quant.head_bits);
    println!("  layers {} (swa {} / dsa {}; dense {} / moe {}), hidden {}, vocab {}, heads {} x {}/{} (kv swa {} dsa {}), window {}, rotary {}, indexer {}x{} top-{}",
             c.num_layers, c.n_swa(), c.n_dsa(), c.n_dense(), c.n_moe(), c.hidden_size, c.vocab_size, c.num_heads, c.head_dim, c.v_head_dim,
             c.swa_kv_heads, c.dsa_kv_heads, c.sliding_window, c.rotary_dim, c.index_n_heads, c.index_head_dim, c.index_top_k);
    println!("  experts {} top-{} inter {}, dense inter {}, router sigmoid+noaux (scale {})", c.num_experts, c.num_experts_per_tok, c.moe_intermediate_size, c.intermediate_size, c.routed_scaling_factor);
    println!("  tensors {} in {} shards, {:.2} GB; quads {} plain {}", p.total_tensors, p.shards.len(), p.total_bytes as f64 / 1e9, p.quads.len(), p.plain.len());
    let by = p.bytes_by_role();
    let mut tot = 0usize;
    for (role, b) in &by { tot += b; println!("    {:22} {:9.3} GB", role.name(), *b as f64 / 1e9); }
    let emb = p.plain.get("model.embed_tokens.weight").map(|m| m.n_bytes).unwrap_or(0);
    println!("    {:22} {:9.3} GB", "embed_tokens (bf16)", emb as f64 / 1e9);
    println!("  quantized {:.2} GB + embeddings {:.2} GB; per rank at TP2 {:.1} GB, TP4 {:.1} GB (experts and attention split, embeddings replicated)",
             tot as f64 / 1e9, emb as f64 / 1e9, (tot as f64 / 2.0 + emb as f64) / 1e9, (tot as f64 / 4.0 + emb as f64) / 1e9);
    let half = p.quads.values().filter(|q| q.bits.is_half()).count();
    println!("  half-integer-K quads {} / {} (3.5 bpw, 56-word tiles); lm_head {:?}", half, p.quads.len(), p.quad("lm_head")?.bits);
    let mut by_rate: BTreeMap<String, usize> = BTreeMap::new();
    for q in p.quads.values() { *by_rate.entry(format!("{}", q.bits.bpw())).or_default() += 1; }
    println!("  quads per rate: {}", by_rate.iter().map(|(b, n)| format!("{b} bpw x{n}")).collect::<Vec<_>>().join(", "));
    println!("NAIVE_PACK_OK inventory verified in {:.1} s", t0.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_names() {
        assert_eq!(classify("lm_head"), Some((Role::LmHead, None, None)));
        assert_eq!(classify("model.layers.3.self_attn.indexer.wq"), Some((Role::IdxWq, Some(3), None)));
        assert_eq!(classify("model.layers.47.mlp.experts.255.down_proj"), Some((Role::ExpertDown, Some(47), Some(255))));
        assert_eq!(classify("model.layers.0.mlp.gate_proj"), Some((Role::DenseGate, Some(0), None)));
        assert_eq!(classify("model.layers.0.mlp.gate"), None);
        assert_eq!(classify("model.embed_tokens"), None);
    }

    #[test]
    fn the_shipped_pack_passes_the_inventory_check() {
        let dir = "/mnt/models/NaiveAI/Naive-N0.5-Flash-exl3-3.5bpw-vllm";
        if !std::path::Path::new(dir).join("model.safetensors.index.json").is_file() { eprintln!("skip: pack not present"); return; }
        let p = NaivePack::open(dir).unwrap();
        assert_eq!(p.total_tensors, 145_525);
        assert_eq!(p.quads.len(), 48 * 4 + 9 * 3 + 3 + 47 * 256 * 3 + 1);
        let q = p.quad("model.layers.1.mlp.experts.0.gate_proj").unwrap();
        assert_eq!((q.k, q.n), (4096, 2048));
        assert!(q.bits.is_half());
        assert_eq!(p.quad("lm_head").unwrap().bits, BitsK::Int(6));
        // v_proj is per-head padded: 8 kv heads x 192 on an SWA layer, 4 x 192 on a DSA layer
        assert_eq!(p.quad("model.layers.1.self_attn.v_proj").unwrap().n, 8 * 192);
        assert_eq!(p.quad("model.layers.0.self_attn.v_proj").unwrap().n, 4 * 192);
        assert_eq!(p.quad("model.layers.0.self_attn.indexer.weights_proj").unwrap().n, 128);
    }
}
