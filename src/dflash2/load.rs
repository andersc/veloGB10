//! The DFlash2 artifact loader — reads the 81-tensor BF16 safetensors (the REAL sha-pinned
//! checkpoint or a shape-identical synthetic), asserts the exact inventory / shapes / dtypes
//! (BF16-only) and an optional sha256 pin, and converts everything to host f32 for the oracle.
//! NO GPU upload in S2F (S3F owns device layout — do not guess it now).

use std::path::Path;

use half::bf16;
use safetensors::SafeTensors;

use crate::dflash2::oracle::{ConvWeights, Dflash2Weights, LayerWeights};
use crate::dflash2::{inventory, n_params, n_tensors};

/// A loaded (and validated) artifact + its host-f32 weights.
pub struct LoadedArtifact {
    pub weights: Dflash2Weights,
    pub n_tensors: usize,
    pub n_params: u64,
    pub sha256: String,
    pub file_size: u64,
    pub header_size: u64,
}

/// FLASHNEXT-DF2 Phase 4: the sha of the artifact the LAST `load()` hashed — the value the
/// serve ENGAGED line names (AGENTS §1b-bis.2: the receipt says which artifact served).
static LAST_LOAD_SHA: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// The sha actually verified by the last `load` (None before any load / on load failure).
pub fn last_load_sha() -> Option<&'static str> {
    LAST_LOAD_SHA.get().map(|s| s.as_str())
}

/// Read `dir/model.safetensors`, validate inventory/shapes/dtypes, optional sha256 pin, and
/// upcast every tensor to f32. Returns an error (not a soft skip) on any mismatch.
pub fn load(dir: &str, sha256_pin: Option<&str>) -> Result<LoadedArtifact, anyhow::Error> {
    // DFLASH2-GEN: the artifact's config.json is the geometry source of truth; install it
    // BEFORE any inventory/buffer work (idempotent; refuses to change after pin_geom()).
    super::geom::install_geom_from_dir(dir)?;
    let path = Path::new(dir).join("model.safetensors");
    let buf = std::fs::read(&path).map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let file_size = buf.len() as u64;

    // Optional sha256 pin (full hex; the REAL artifact pin is `crate::dflash2::REAL_SHA256`).
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(&buf);
    let hexd: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    let _ = LAST_LOAD_SHA.set(hexd.clone());
    if let Some(pin) = sha256_pin {
        let known = crate::dflash2::KNOWN_SHA256.iter().any(|k| hexd.eq_ignore_ascii_case(k));
        anyhow::ensure!(
            hexd.eq_ignore_ascii_case(pin) || known,
            "sha256 mismatch: file={hexd} pin={pin} (accepted: REAL/RTNCAL; see dflash2::KNOWN_SHA256)"
        );
    }

    let st = SafeTensors::deserialize(&buf).map_err(|e| anyhow::anyhow!("parse safetensors: {e}"))?;

    let inv = inventory();
    // P7/W8 (owner 2026-10-18: "just make it valid"): the RTNcal artifact is the ModelOpt NVFP4
    // export of the SAME 81-tensor draft — the 35 GEMM projections arrive as
    // {<name>.weight U8 [r, c/2], <name>.weight_scale F8_E4M3 [r, c/16], <name>.weight_scale_2
    // F32 scalar, <name>.input_scale (activation side, unused — bf16 activations here)};
    // every other tensor is BF16 under its inventory name (5*28 + 5*8 + 6 = 186).
    // Dequant the 35 to BF16 and keep the owner's --draft-dir unchanged.
    let modelopt = st.names().iter().any(|n: &&str| n.ends_with("weight_scale_2"));
    let mo_bf16: Option<std::collections::HashMap<String, Vec<u8>>> = if modelopt {
        Some(build_modelopt_bf16(&st, &inv)?)
    } else {
        None
    };
    if mo_bf16.is_none() {
        // Exact name-set equality vs the 81-tensor inventory (sorted → deterministic comparison).
        let mut file_names: Vec<String> = st.names().iter().map(|s| s.to_string()).collect();
        file_names.sort();
        let mut inv_names: Vec<String> = inv.iter().map(|(n, _)| n.clone()).collect();
        inv_names.sort();
        anyhow::ensure!(
            file_names == inv_names,
            "inventory name mismatch (have {} tensors; expected {})",
            file_names.len(),
            inv_names.len()
        );
        anyhow::ensure!(file_names.len() == n_tensors(), "tensor count {} != {}", file_names.len(), n_tensors());
    }

    // Per-tensor dtype (BF16-only) + exact shape, in FIXED inventory order.
    let mut weights = Dflash2Weights {
        layers: Vec::with_capacity(crate::dflash2::n_layers()),
        fc: Vec::new(),
        hidden_norm: Vec::new(),
        norm: Vec::new(),
        hidden_projection: Vec::new(),
        predecessor_codebook: Vec::new(),
        successor_codebook: Vec::new(),
    };

    let bf16_to_f32 = |name: &str, shape: &[usize]| -> Result<Vec<f32>, anyhow::Error> {
        anyhow::ensure!(!name.contains("embed_tokens") && !name.contains("lm_head"),
            "{name}: embed/lm_head must NOT be in the checkpoint (borrowed from the target)");
        let data: &[u8] = if let Some(m) = &mo_bf16 {
            &m.get(name).ok_or_else(|| anyhow::anyhow!("ModelOpt view missing {name}"))?[..]
        } else {
            let view = st.tensor(name).map_err(|e| anyhow::anyhow!("tensor {name}: {e}"))?;
            anyhow::ensure!(view.dtype() == safetensors::Dtype::BF16, "{name}: not BF16 ({:?})", view.dtype());
            anyhow::ensure!(view.shape() == shape, "{name}: shape {:?} != expected {shape:?}", view.shape());
            view.data()
        };
        Ok(data
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect())
    };

    let mut n_params: u64 = 0;
    for (name, shape) in &inv {
        n_params += shape.iter().map(|&d| d as u64).product::<u64>();
        let v = bf16_to_f32(name, shape)?;
        assign(&mut weights, name, v)?;
    }
    anyhow::ensure!(
        n_params == crate::dflash2::n_params(),
        "param count {n_params} != {}",
        crate::dflash2::n_params()
    );

    Ok(LoadedArtifact {
        weights,
        n_tensors: n_tensors(),
        n_params,
        sha256: hexd,
        file_size,
        header_size: 8u64 + u64::from_le_bytes(buf[0..8].try_into().unwrap()),
    })
}

/// Assign one f32 tensor into the weight struct by name (fixed-order, deterministic).
fn assign(w: &mut Dflash2Weights, name: &str, v: Vec<f32>) -> Result<(), anyhow::Error> {
    match name {
        "fc.weight" => w.fc = v,
        "hidden_norm.weight" => w.hidden_norm = v,
        "norm.weight" => w.norm = v,
        "candidate_selector.hidden_projection.weight" => w.hidden_projection = v,
        "candidate_selector.predecessor_codebook" => w.predecessor_codebook = v,
        "candidate_selector.successor_codebook" => w.successor_codebook = v,
        _ => {
            // layer tensors: `layers.{i}.{suffix}`
            let rest = name.strip_prefix("layers.").ok_or_else(|| anyhow::anyhow!("bad name {name}"))?;
            let (i_str, suffix) = rest.split_once('.').ok_or_else(|| anyhow::anyhow!("bad layer name {name}"))?;
            let i: usize = i_str.parse().map_err(|_| anyhow::anyhow!("bad layer index {name}"))?;
            while w.layers.len() <= i {
                w.layers.push(empty_layer());
            }
            let l = &mut w.layers[i];
            match suffix {
                "self_attn.q_proj.weight" => l.q_proj = v,
                "self_attn.k_proj.weight" => l.k_proj = v,
                "self_attn.v_proj.weight" => l.v_proj = v,
                "self_attn.o_proj.weight" => l.o_proj = v,
                "self_attn.q_norm.weight" => l.q_norm = v,
                "self_attn.k_norm.weight" => l.k_norm = v,
                "input_layernorm.weight" => l.input_ln = v,
                "post_attention_layernorm.weight" => l.post_ln = v,
                "mlp.gate_proj.weight" => l.gate_proj = v,
                "mlp.up_proj.weight" => l.up_proj = v,
                "mlp.down_proj.weight" => l.down_proj = v,
                "attention_conv.base_kernel" => l.attention_conv.base_kernel = v,
                "attention_conv.kernel_projection.weight" => l.attention_conv.kernel_projection = v,
                "mlp_conv.base_kernel" => l.mlp_conv.base_kernel = v,
                "mlp_conv.kernel_projection.weight" => l.mlp_conv.kernel_projection = v,
                _ => anyhow::bail!("unknown tensor {name}"),
            }
        }
    }
    Ok(())
}

fn empty_layer() -> LayerWeights {
    LayerWeights {
        q_proj: Vec::new(),
        k_proj: Vec::new(),
        v_proj: Vec::new(),
        o_proj: Vec::new(),
        q_norm: Vec::new(),
        k_norm: Vec::new(),
        input_ln: Vec::new(),
        post_ln: Vec::new(),
        gate_proj: Vec::new(),
        up_proj: Vec::new(),
        down_proj: Vec::new(),
        attention_conv: ConvWeights { base_kernel: Vec::new(), kernel_projection: Vec::new() },
        mlp_conv: ConvWeights { base_kernel: Vec::new(), kernel_projection: Vec::new() },
    }
}

// ---------------------------------------------------------------------------
// PLAN/25 §1a — the NVFP4 weight-only artifact (draft round-time cuts).
//
// A baked dir (`--df2-bake-nvfp4 <src> <dst>`) holds:
//   * `nvfp4.safetensors` — one packed triple per quantized linear, the same
//     compressed-tensors convention the trunk quantizer emits:
//     `{name}.weight_packed` U8 [M*K/2] (E2M1 nibbles), `{name}.weight_scale`
//     U8 [M,K/16] (E4M3 block scales), `{name}.weight_global_scale` F32 [1].
//   * `model.safetensors` — the tensors that stay hi-prec (selector codebooks +
//     hidden_projection, all norms, conv base_kernels) PLUS the BF16 twins the
//     prompt-prime path needs (`fc.weight`, per-layer `k_proj`/`v_proj`:
//     `gemm_mma_fp4_b` is N≤16 and prime runs M up to 8192, so prime keeps the
//     bf16 `gemm_tiled_b` path — 367 MB, the only duplicated bytes).
//   * `config.json` — the original plus `df2_quant: "nvfp4"`,
//     `df2_quant_source_sha256` (the baked-from pin) and `df2_quant_recipe`.
// Quantization is weight-only RTN; selector codebooks/hp stay BF16 per the plan
// (the walk gathers them row-wise — quantizing saves memory, not bandwidth, and
// spends draft quality to do it).
// ---------------------------------------------------------------------------

/// One packed NVFP4 tensor exactly as stored in `nvfp4.safetensors` (pre-repack;
/// `repack_nvfp4_mma` runs at load, mirroring the trunk's own flow).
#[derive(Clone)]
pub struct PackedNvfp4 {
    pub name: String,
    pub qweight: Vec<u8>,
    pub scales: Vec<u8>,
    pub global_scale: f32,
    pub m: usize,
    pub k: usize,
}

/// One FP8-E4M3 tensor exactly as stored in `fp8.safetensors` (pre-repack;
/// `repack_fp8_mma` runs at load — the trunk's own DSV4-attention flow). 8 bits +
/// one f32 scale PER ROW (multiply on dequant; `gemm_mma_fp8_b` indexes `rs[m]`).
#[derive(Clone)]
pub struct PackedFp8 {
    pub name: String,
    pub qweight: Vec<u8>,
    pub row_scale: Vec<f32>,
    pub m: usize,
    pub k: usize,
}

/// The per-tensor payload of a baked artifact — which weight-only format the
/// sidecar carries. Clone: the round-shard branch (PLAN/ROUND_SHARD_W2_WORKDOC.md §3)
/// picks a tensor by name and then SLICES it, so it needs an owned copy to slice
/// (the unsharded branch still borrows).
#[derive(Clone)]
pub enum QuantTensor {
    Nvfp4(PackedNvfp4),
    Fp8(PackedFp8),
}

impl QuantTensor {
    pub fn name(&self) -> &str {
        match self { QuantTensor::Nvfp4(p) => &p.name, QuantTensor::Fp8(p) => &p.name }
    }
}

// ---------------------------------------------------------------------------
// PLAN/ROUND_SHARD_W2_WORKDOC.md §3.1: the round-shard bands, in the PACKED domain.
// Every format's packed layout is row-major in M, so:
//   * M-bands (q/k/v/gate/up, the OUTPUT-row split) are contiguous row slices;
//   * K-bands (o/down, the K split) are byte-column slices that must stay aligned to
//     the format's scale group AND to the MMA repack's K constraint (nvfp4 K%32 with
//     k0%32==0; fp8 K%16 with k0%16==0) — the repack asserts are the backstop, these
//     asserts are the first line (an odd k-block count would silently skip the last
//     block and still look like a model: quant.rs:1027-1030).
// The struct's `m`/`k` are trimmed so `upload_nvfp4`/`upload_fp8` (and their
// `gs`-length / row-scale bookkeeping) need no change.
// ---------------------------------------------------------------------------
impl PackedNvfp4 {
    /// Output-row band [r0, r1): q/k/v/gate/up. Requires r0 % 1 (row granularity).
    pub fn slice_rows(&self, r0: usize, r1: usize) -> PackedNvfp4 {
        assert!(r0 < r1 && r1 <= self.m, "nvfp4 slice_rows {r0}..{r1} of m={}", self.m);
        let kb = self.k / 2;
        let sb = self.k / 16;
        let mut qweight = Vec::with_capacity((r1 - r0) * kb);
        let mut scales = Vec::with_capacity((r1 - r0) * sb);
        for r in r0..r1 {
            qweight.extend_from_slice(&self.qweight[r * kb..(r + 1) * kb]);
            scales.extend_from_slice(&self.scales[r * sb..(r + 1) * sb]);
        }
        PackedNvfp4 { name: self.name.clone(), qweight, scales,
                      global_scale: self.global_scale, m: r1 - r0, k: self.k }
    }

    /// K band [k0, k1): o/down. `k0` and the band length must be multiples of 32 (the
    /// paired-k MMA group); scale columns move by k0/16.
    pub fn slice_k(&self, k0: usize, k1: usize) -> PackedNvfp4 {
        assert!(k0 < k1 && k1 <= self.k, "nvfp4 slice_k {k0}..{k1} of k={}", self.k);
        assert!(k0 % 32 == 0 && (k1 - k0) % 32 == 0,
                "nvfp4 K-band must be 32-aligned: k0={k0} len={} (the MMA repack asserts K%32;                  an unaligned band would silently drop the last k-block)", k1 - k0);
        let kb = self.k / 2;
        let sb = self.k / 16;
        let mut qweight = Vec::with_capacity(self.m * ((k1 - k0) / 2));
        let mut scales = Vec::with_capacity(self.m * ((k1 - k0) / 16));
        for r in 0..self.m {
            qweight.extend_from_slice(&self.qweight[r * kb + k0 / 2..r * kb + k1 / 2]);
            scales.extend_from_slice(&self.scales[r * sb + k0 / 16..r * sb + k1 / 16]);
        }
        PackedNvfp4 { name: self.name.clone(), qweight, scales,
                      global_scale: self.global_scale, m: self.m, k: k1 - k0 }
    }
}

impl PackedFp8 {
    /// Output-row band [r0, r1): q/k/v/gate/up (`row_scale` moves with M).
    pub fn slice_rows(&self, r0: usize, r1: usize) -> PackedFp8 {
        assert!(r0 < r1 && r1 <= self.m, "fp8 slice_rows {r0}..{r1} of m={}", self.m);
        let kb = self.k;
        let mut qweight = Vec::with_capacity((r1 - r0) * kb);
        for r in r0..r1 { qweight.extend_from_slice(&self.qweight[r * kb..(r + 1) * kb]); }
        PackedFp8 { name: self.name.clone(), qweight,
                    row_scale: self.row_scale[r0..r1].to_vec(), m: r1 - r0, k: self.k }
    }

    /// K band [k0, k1): o/down. `k0` and the band length must be multiples of 16 (the MMA
    /// repack asserts K%16); row scales are K-independent and stay whole.
    pub fn slice_k(&self, k0: usize, k1: usize) -> PackedFp8 {
        assert!(k0 < k1 && k1 <= self.k, "fp8 slice_k {k0}..{k1} of k={}", self.k);
        assert!(k0 % 16 == 0 && (k1 - k0) % 16 == 0,
                "fp8 K-band must be 16-aligned: k0={k0} len={}", k1 - k0);
        let kb = self.k;
        let mut qweight = Vec::with_capacity(self.m * (k1 - k0));
        for r in 0..self.m {
            qweight.extend_from_slice(&self.qweight[r * kb + k0..r * kb + k1]);
        }
        PackedFp8 { name: self.name.clone(), qweight, row_scale: self.row_scale.clone(),
                    m: self.m, k: k1 - k0 }
    }
}

/// The quantized side of a baked artifact (paired with [`load`]'s hi-prec weights).
pub struct QuantArtifact {
    pub tensors: Vec<QuantTensor>,
    pub source_sha256: String,
    pub recipe: String,
}

/// The linear weights §1a quantizes (everything the per-step block pass streams).
/// Selector/codebook/norm/base_kernel tensors are NOT here (stay BF16 in
/// `model.safetensors`). FLASHNEXT-DF2-NVFP4 (2026-10-23): GEOMETRY-DRIVEN — the table was
/// hard-wired to the 27B shapes (5120 / 17408 / fc 25600 / conv 1280), so no other artifact
/// could be baked; the live geometry is the one the bake installs from the source config.
pub fn quantized_inventory() -> Vec<(String, usize, usize)> {
    quantized_inventory_geom(&crate::dflash2::geom())
}

/// [`quantized_inventory`] for an explicit geometry (pure; the bake and the tests use it).
pub fn quantized_inventory_geom(g: &crate::dflash2::Geom) -> Vec<(String, usize, usize)> {
    let q = g.num_heads * g.head_dim;
    let kv = g.num_kv_heads * g.head_dim;
    let conv = 2 * g.conv_kernel * g.conv_groups();
    let mut v = Vec::new();
    v.push(("fc.weight".to_string(), g.hidden, g.tap_concat_dim()));
    for i in 0..g.n_layers {
        let p = move |s: &str| format!("layers.{i}.{s}");
        v.push((p("self_attn.q_proj.weight"), q, g.hidden));
        v.push((p("self_attn.k_proj.weight"), kv, g.hidden));
        v.push((p("self_attn.v_proj.weight"), kv, g.hidden));
        v.push((p("self_attn.o_proj.weight"), g.hidden, q));
        v.push((p("mlp.gate_proj.weight"), g.inter, g.hidden));
        v.push((p("mlp.up_proj.weight"), g.inter, g.hidden));
        v.push((p("mlp.down_proj.weight"), g.hidden, g.inter));
        v.push((p("attention_conv.kernel_projection.weight"), conv, g.hidden));
        v.push((p("mlp_conv.kernel_projection.weight"), conv, g.hidden));
    }
    v
}

/// The BF16 twins prime_window needs (gemm_mma_fp4_b is N≤16; prime is not).
pub fn bf16_twin_inventory() -> Vec<(String, usize, usize)> {
    bf16_twin_inventory_geom(&crate::dflash2::geom())
}

/// [`bf16_twin_inventory`] for an explicit geometry.
pub fn bf16_twin_inventory_geom(g: &crate::dflash2::Geom) -> Vec<(String, usize, usize)> {
    let kv = g.num_kv_heads * g.head_dim;
    let mut v = vec![("fc.weight".to_string(), g.hidden, g.tap_concat_dim())];
    for i in 0..g.n_layers {
        v.push((format!("layers.{i}.self_attn.k_proj.weight"), kv, g.hidden));
        v.push((format!("layers.{i}.self_attn.v_proj.weight"), kv, g.hidden));
    }
    v
}

#[cfg(test)]
mod quant_inventory_tests {
    use super::*;
    /// The 27B default geometry reproduces the old hard-wired table exactly; the Flash-Next
    /// geometry (hidden 2560, inter 7680, 5 taps, conv groups 160) gives its own shapes.
    #[test]
    fn quantized_inventory_follows_the_geometry() {
        let g27 = crate::dflash2::Geom::default();
        let v = quantized_inventory_geom(&g27);
        assert_eq!(v[0], ("fc.weight".to_string(), 5120, 25600));
        assert!(v.contains(&("layers.0.self_attn.q_proj.weight".to_string(), 4096, 5120)));
        assert!(v.contains(&("layers.4.mlp.down_proj.weight".to_string(), 5120, 17408)));
        assert!(v.contains(&("layers.2.mlp_conv.kernel_projection.weight".to_string(), 1280, 5120)));
        assert_eq!(v.len(), 1 + 5 * 9);
        assert_eq!(bf16_twin_inventory_geom(&g27).len(), 1 + 5 * 2);
        let fn_cfg = r#"{"hidden_size": 2560, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
            "intermediate_size": 7680, "num_hidden_layers": 5, "vocab_size": 248320, "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144, "sliding_window": 2048, "is_causal": false,
            "rope_parameters": {"rope_theta": 10000, "rope_type": "default"}, "num_target_layers": 48,
            "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": 248077,
                              "selector_rank": 256, "selector_top_k": 16, "target_layer_ids": [5, 15, 25, 35, 45]}}"#;
        let gfn = crate::dflash2::geom_from_config_json(fn_cfg).expect("flash-next geometry");
        let v = quantized_inventory_geom(&gfn);
        assert_eq!(v[0], ("fc.weight".to_string(), 2560, 12800));
        assert!(v.contains(&("layers.0.self_attn.q_proj.weight".to_string(), 4096, 2560)));
        assert!(v.contains(&("layers.0.self_attn.k_proj.weight".to_string(), 1024, 2560)));
        assert!(v.contains(&("layers.3.mlp.gate_proj.weight".to_string(), 7680, 2560)));
        assert!(v.contains(&("layers.4.attention_conv.kernel_projection.weight".to_string(), 640, 2560)));
        let twins = bf16_twin_inventory_geom(&gfn);
        assert_eq!(twins[0], ("fc.weight".to_string(), 2560, 12800));
        assert!(twins.contains(&("layers.4.self_attn.v_proj.weight".to_string(), 1024, 2560)));
    }
}

/// Cheap baked-dir probe: config.json carries `df2_quant == "nvfp4" | "fp8"`.
pub fn is_baked(dir: &str) -> bool {
    let Ok(cfg_raw) = std::fs::read_to_string(Path::new(dir).join("config.json")) else { return false };
    let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&cfg_raw) else { return false };
    matches!(cfg.get("df2_quant").and_then(|v| v.as_str()), Some("nvfp4") | Some("fp8"))
}

/// The baked format this dir carries (None = not baked).
pub fn baked_fmt(dir: &str) -> Option<&'static str> {
    let cfg_raw = std::fs::read_to_string(Path::new(dir).join("config.json")).ok()?;
    let cfg = serde_json::from_str::<serde_json::Value>(&cfg_raw).ok()?;
    match cfg.get("df2_quant").and_then(|v| v.as_str())? {
        "nvfp4" => Some("nvfp4"),
        "fp8" => Some("fp8"),
        _ => None,
    }
}

/// Read a baked dir (NVFP4 or FP8 sidecar — `config.json: df2_quant` picks): the
/// hi-prec weights via [`load`] (its inventory check is RELAXED to the kept set —
/// see `load_kept`) plus the packed sidecar.
pub fn load_quantized(dir: &str) -> Result<(LoadedArtifact, QuantArtifact), anyhow::Error> {
    // DFLASH2-GEN: geometry from the baked artifact config (same source of truth as the
    // BF16 loader; idempotent if the funnel already installed it).
    super::geom::install_geom_from_dir(dir)?;
    // The hi-prec side: same reader, same asserts, kept-subset inventory.
    let weights = load_kept(dir)?;
    let fmt = baked_fmt(dir).ok_or_else(|| anyhow::anyhow!("config.json: df2_quant is neither nvfp4 nor fp8 — not a baked artifact"))?;
    // The packed side.
    let path = Path::new(dir).join(if fmt == "fp8" { "fp8.safetensors" } else { "nvfp4.safetensors" });
    let buf = std::fs::read(&path).map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let st = SafeTensors::deserialize(&buf).map_err(|e| anyhow::anyhow!("parse safetensors: {e}"))?;
    let cfg: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(dir).join("config.json"))
            .map_err(|e| anyhow::anyhow!("read config.json: {e}"))?)?;
    let source_sha = cfg.get("df2_quant_source_sha256").and_then(|v| v.as_str())
        .unwrap_or("").to_string();
    let recipe = cfg.get("df2_quant_recipe").and_then(|v| v.as_str())
        .unwrap_or(if fmt == "fp8" { "fp8-rtw" } else { "nvfp4-rtw" }).to_string();

    let inv = quantized_inventory();
    let mut tensors = Vec::with_capacity(inv.len());
    for (name, m, k) in &inv {
        let get = |suffix: &str| -> Result<safetensors::tensor::TensorView, anyhow::Error> {
            let full = format!("{name}.{suffix}");
            let view = st.tensor(&full)
                .map_err(|e| anyhow::anyhow!("tensor {full}: {e} — is this a baked artifact?"))?;
            Ok(view)
        };
        if fmt == "fp8" {
            let qw = get("weight_packed")?;
            let rs = get("weight_row_scale")?;
            anyhow::ensure!(qw.dtype() == safetensors::Dtype::U8 && rs.dtype() == safetensors::Dtype::F32,
                "{name}: fp8 packed dtypes must be U8/F32");
            anyhow::ensure!(qw.shape() == [*m, *k], "{name}: qweight {:?} != [{m},{k}]", qw.shape());
            anyhow::ensure!(rs.shape() == [*m], "{name}: row_scale {:?} != [{m}]", rs.shape());
            tensors.push(QuantTensor::Fp8(PackedFp8 {
                name: name.clone(),
                qweight: qw.data().to_vec(),
                row_scale: crate::podread::pod_vec::<f32>(rs.data()),
                m: *m, k: *k,
            }));
        } else {
            let qw = get("weight_packed")?;
            let sc = get("weight_scale")?;
            let gs = get("weight_global_scale")?;
            // weight_scale stores E4M3 bytes — the --quantize emit convention types it F8_E4M3
            // (bytewise identical to U8; accept both so hand-built sidecars also load).
            anyhow::ensure!(qw.dtype() == safetensors::Dtype::U8
                            && (sc.dtype() == safetensors::Dtype::U8 || sc.dtype() == safetensors::Dtype::F8_E4M3)
                            && gs.dtype() == safetensors::Dtype::F32,
                "{name}: packed dtypes must be U8/(U8|F8_E4M3)/F32");
            anyhow::ensure!(qw.shape() == [*m, k / 2], "{name}: qweight {:?} != [{m},{}]", qw.shape(), k / 2);
            anyhow::ensure!(sc.shape() == [*m, k / 16], "{name}: scales {:?} != [{m},{}]", sc.shape(), k / 16);
            anyhow::ensure!(gs.data().len() == 4, "{name}: global_scale must be one f32");
            tensors.push(QuantTensor::Nvfp4(PackedNvfp4 {
                name: name.clone(),
                qweight: qw.data().to_vec(),
                scales: sc.data().to_vec(),
                global_scale: f32::from_le_bytes(gs.data().try_into().unwrap()),
                m: *m, k: *k,
            }));
        }
    }
    anyhow::ensure!(tensors.len() == inv.len(), "packed tensor count {} != {}", tensors.len(), inv.len());
    Ok((weights, QuantArtifact { tensors, source_sha256: source_sha, recipe }))
}

/// The BF16 reader for the KEPT subset (hi-prec tensors + the prime twins).
/// Same per-tensor asserts as [`load`], but the inventory is the kept set, not
/// all 81 names.
fn load_kept(dir: &str) -> Result<LoadedArtifact, anyhow::Error> {
    let path = Path::new(dir).join("model.safetensors");
    let buf = std::fs::read(&path).map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let file_size = buf.len() as u64;
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(&buf);
    let hexd: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();

    let st = SafeTensors::deserialize(&buf).map_err(|e| anyhow::anyhow!("parse safetensors: {e}"))?;

    // Kept = hi-prec (norms/base_kernels/codebooks/hp — everything in the 81-tensor
    // inventory EXCEPT the quantized linears) + the BF16 prime twins.
    let quant: std::collections::HashSet<String> =
        quantized_inventory().into_iter().map(|(n, _, _)| n).collect();
    let twins = bf16_twin_inventory();
    let mut kept: Vec<(String, Vec<usize>)> = Vec::new();
    for (n, s) in inventory() {
        if !quant.contains(&n) { kept.push((n.to_string(), s.to_vec())); }
    }
    for (n, m, k) in &twins {
        kept.push((n.clone(), vec![*m, *k]));
    }
    // `fc` appears in both loops (twin) — dedupe keeps one entry.
    kept.sort();
    kept.dedup();

    let mut file_names: Vec<String> = st.names().iter().map(|s| s.to_string()).collect();
    file_names.sort();
    let mut kept_names: Vec<String> = kept.iter().map(|(n, _)| n.clone()).collect();
    kept_names.sort();
    anyhow::ensure!(file_names == kept_names,
        "baked model.safetensors inventory mismatch (have {}; expected the kept {})",
        file_names.len(), kept_names.len());

    let mut weights = Dflash2Weights {
        layers: Vec::with_capacity(crate::dflash2::n_layers()),
        fc: Vec::new(),
        hidden_norm: Vec::new(),
        norm: Vec::new(),
        hidden_projection: Vec::new(),
        predecessor_codebook: Vec::new(),
        successor_codebook: Vec::new(),
    };
    let bf16_to_f32 = |name: &str, shape: &[usize]| -> Result<Vec<f32>, anyhow::Error> {
        let view = st.tensor(name).map_err(|e| anyhow::anyhow!("tensor {name}: {e}"))?;
        anyhow::ensure!(view.dtype() == safetensors::Dtype::BF16, "{name}: not BF16 ({:?})", view.dtype());
        anyhow::ensure!(view.shape() == shape, "{name}: shape {:?} != expected {shape:?}", view.shape());
        let data = view.data();
        Ok(data.chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect())
    };
    for (name, shape) in &kept {
        let v = bf16_to_f32(name, shape)?;
        assign(&mut weights, name, v)?;
    }
    Ok(LoadedArtifact {
        weights,
        n_tensors: kept.len(),
        n_params: kept.iter().map(|(_, s)| s.iter().map(|&d| d as u64).product::<u64>()).sum(),
        sha256: hexd,
        file_size,
        header_size: 0,
    })
}

/// P7/W8: materialize the 81-tensor BF16 view of the ModelOpt NVFP4 export (186 tensors).
/// Projection dequant: w[r][c] = e2m1(nibble) * e4m3(weight_scale[r][c/16]) * weight_scale_2.
/// Nibble order follows our own `pack_e2m1_pair` (low nibble = even index); a flipped
/// convention shows up immediately as a cratered acceptance, not silent corruption.
fn build_modelopt_bf16(
    st: &SafeTensors,
    inv: &[(String, Vec<usize>)],
) -> Result<std::collections::HashMap<String, Vec<u8>>, anyhow::Error> {
    use crate::quant::{e2m1_to_f32, e4m3_to_f32};
    let mut out = std::collections::HashMap::new();
    for (name, shape) in inv {
        // ModelOpt names the packed triple AFTER the inventory name itself:
        // {name} = U8 packed weight, {name}_scale = F8_E4M3 groups, {name}_scale_2 = F32 scalar.
        // A direct BF16 hit is the passthrough (norms, convs, codebooks, fc).
        if let Ok(view) = st.tensor(name) {
            if view.dtype() == safetensors::Dtype::BF16 {
                anyhow::ensure!(view.shape() == shape, "{name}: shape {:?} != expected {shape:?}", view.shape());
                out.insert(name.clone(), view.data().to_vec());
                continue;
            }
        }
        let wname = name.clone();
        let sname = format!("{name}_scale");
        let tname = format!("{name}_scale_2");
        let wv = st.tensor(&wname).map_err(|e| anyhow::anyhow!("{wname}: {e} (expected a ModelOpt projection)"))?;
        anyhow::ensure!(wv.dtype() == safetensors::Dtype::U8, "{wname}: not U8");
        anyhow::ensure!(shape.len() == 2, "{name}: non-matrix projection {shape:?}");
        let (rows, cols) = (shape[0], shape[1]);
        anyhow::ensure!(wv.shape() == vec![rows, cols / 2], "{wname}: shape {:?} != [{rows}, {}]", wv.shape(), cols / 2);
        let sv = st.tensor(&sname).map_err(|e| anyhow::anyhow!("{sname}: {e}"))?;
        anyhow::ensure!(sv.dtype() == safetensors::Dtype::F8_E4M3, "{sname}: not F8_E4M3");
        anyhow::ensure!(sv.shape() == vec![rows, cols / 16], "{sname}: shape {:?} != [{rows}, {}]", sv.shape(), cols / 16);
        let tv = st.tensor(&tname).map_err(|e| anyhow::anyhow!("{tname}: {e}"))?;
        anyhow::ensure!(tv.dtype() == safetensors::Dtype::F32 && tv.shape().is_empty(), "{tname}: not an F32 scalar");
        let s2 = f32::from_le_bytes(tv.data()[0..4].try_into().unwrap());
        let wd = wv.data();
        let sd = sv.data();
        let mut buf: Vec<u8> = Vec::with_capacity(rows * cols * 2);
        for r in 0..rows {
            let wrow = &wd[r * (cols / 2)..(r + 1) * (cols / 2)];
            let srow = &sd[r * (cols / 16)..(r + 1) * (cols / 16)];
            for c in 0..cols {
                let byte = wrow[c / 2];
                let nib = if c % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F };
                let v = e2m1_to_f32(nib) * e4m3_to_f32(srow[c / 16]) * s2;
                buf.extend_from_slice(&bf16::from_f32(v).to_bits().to_le_bytes());
            }
        }
        out.insert(name.clone(), buf);
    }
    Ok(out)
}
