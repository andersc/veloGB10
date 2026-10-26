//! DFLASH2-GEN (2026-10-23) — the artifact GEOMETRY, installed from the draft's config.json.
//!
//! The S4F round was born hard-wired to the Qwen3.8-27B drafter artifact; every other trunk
//! was refused by the boot guard. A different artifact (Ornith-1.5-122B-DFlash2: hidden 3072,
//! 6 draft layers, 8 taps at [1,7,14,20,26,32,39,45], inter 9216, mask 248077) now installs
//! its geometry here, ONCE, before any round/buffer is constructed — the exact pattern
//! DF2_BLOCK pioneered ("one accessor means there is no site that can disagree"). The consts
//! in `mod.rs` stay as the 27B DEFAULT (and the source of [`Geom::default`]); the LIVE values
//! are the `geom()`-backed accessors, so an un-installed process (synth/oracle/tests) is
//! bit-identical to today's 27B behaviour.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use super::{
    BLOCK, CONV_GROUP, CONV_KERNEL, HEAD_DIM, HIDDEN, INTER, IS_CAUSAL, MASK_TOKEN_ID,
    MAX_BLOCK, MAX_POSITIONS, N_LAYERS, NUM_HEADS, NUM_KV_HEADS, RMS_EPS, ROPE_THETA,
    SELECTOR_RANK, SELECTOR_TOP_K, SLIDING_WINDOW, TAP_LAYERS, VOCAB,
};

/// The varying geometry of a DFlash2 drafter artifact — everything two artifacts may
/// disagree on (parsed from the draft dir's `config.json`, see [`geom_from_config_json`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Geom {
    /// Draft-backbone and tap-projection width (`hidden_size`).
    pub hidden: usize,
    /// Query heads (`num_attention_heads`).
    pub num_heads: usize,
    /// KV heads (`num_key_value_heads`).
    pub num_kv_heads: usize,
    /// Per-head dimension (`head_dim`; defaults to 128 when the config omits it).
    pub head_dim: usize,
    /// SwiGLU intermediate size (`intermediate_size`).
    pub inter: usize,
    /// Vocab size (embed/lm_head are borrowed from the target; `vocab_size`).
    pub vocab: usize,
    /// Number of draft-backbone layers (`num_hidden_layers`).
    pub n_layers: usize,
    /// The block's no-information filler token (`dflash_config.mask_token_id`).
    pub mask_token_id: u32,
    /// RoPE base (`rope_parameters.rope_theta`).
    pub rope_theta: f32,
    /// RMSNorm epsilon (`rms_norm_eps`).
    pub rms_eps: f32,
    /// Maximum RoPE positions (`max_position_embeddings`) — the round's ctx ceiling.
    pub max_positions: usize,
    /// Sliding window on the context KV band (`sliding_window`).
    pub sliding_window: usize,
    /// Top-level causality flag (`is_causal`).
    pub is_causal: bool,
    /// Dynamic-conv kernel size (`dflash_config.conv_kernel_size`).
    pub conv_kernel: usize,
    /// Dynamic-conv channel group size (`dflash_config.conv_group_size`).
    pub conv_group: usize,
    /// Candidate-selector rank (`dflash_config.selector_rank`).
    pub selector_rank: usize,
    /// Candidate-selector top-k (`dflash_config.selector_top_k`).
    pub selector_top_k: usize,
    /// `dflash_config.target_layer_ids` — the TARGET trunk layers tapped at capture.
    pub tap_layers: Vec<usize>,
    /// `num_target_layers` — the trunk depth the artifact was trained against.
    pub num_target_layers: usize,
    /// `dflash_config.block_size` — the artifact's TRAINED draft block. Informational only:
    /// the live block is `--df2-block` (see [`super::set_block`]); a mismatch is logged at
    /// install.
    pub block_size: usize,
}

impl Default for Geom {
    fn default() -> Self {
        Self {
            hidden: HIDDEN,
            num_heads: NUM_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            head_dim: HEAD_DIM,
            inter: INTER,
            vocab: VOCAB,
            n_layers: N_LAYERS,
            mask_token_id: MASK_TOKEN_ID,
            rope_theta: ROPE_THETA,
            rms_eps: RMS_EPS,
            max_positions: MAX_POSITIONS,
            sliding_window: SLIDING_WINDOW,
            is_causal: IS_CAUSAL,
            conv_kernel: CONV_KERNEL,
            conv_group: CONV_GROUP,
            selector_rank: SELECTOR_RANK,
            selector_top_k: SELECTOR_TOP_K,
            tap_layers: TAP_LAYERS.to_vec(),
            num_target_layers: 64,
            block_size: BLOCK,
        }
    }
}

impl Geom {
    /// Number of trunk taps concatenated per position.
    pub fn n_taps(&self) -> usize {
        self.tap_layers.len()
    }
    /// Concatenated tap dim = n_taps × hidden.
    pub fn tap_concat_dim(&self) -> usize {
        self.n_taps() * self.hidden
    }
    /// Dynamic-conv groups (= hidden / conv_group).
    pub fn conv_groups(&self) -> usize {
        self.hidden / self.conv_group
    }
    /// Exact tensor count of the checkpoint: 6 globals + 15 per draft layer (the
    /// reconciliation in `mod.rs::N_TENSORS` generalized).
    pub fn n_tensors(&self) -> usize {
        6 + 15 * self.n_layers
    }
    /// The honest trunk-compatibility predicate (core of `GpuModel::df2_round_compatible`,
    /// free-standing so unit tests can exercise it without a GPU model): the trunk must match
    /// the draft's hidden width, cover the borrowed head's row count, and have exactly the
    /// depth the artifact declares (every tap layer must exist, with room for the +1 offset).
    pub fn trunk_compatible(&self, trunk_hidden: usize, trunk_vocab: usize, trunk_layers: usize) -> bool {
        trunk_hidden == self.hidden
            && trunk_vocab >= self.vocab
            && trunk_layers == self.num_target_layers
            && trunk_layers > *self.tap_layers.last().unwrap_or(&0)
    }

    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.tap_layers.is_empty(), "draft config.json: target_layer_ids empty");
        anyhow::ensure!(self.tap_layers.windows(2).all(|w| w[0] < w[1]),
            "draft config.json: target_layer_ids must be strictly increasing (got {:?})", self.tap_layers);
        anyhow::ensure!(*self.tap_layers.last().unwrap() < self.num_target_layers,
            "draft config.json: tap layer {} not below num_target_layers {}",
            self.tap_layers.last().unwrap(), self.num_target_layers);
        anyhow::ensure!(self.hidden > 0 && self.hidden % self.conv_group == 0,
            "draft config.json: hidden_size {0} must be a positive multiple of conv_group_size {1}",
            self.hidden, self.conv_group);
        anyhow::ensure!(self.n_layers >= 1, "draft config.json: num_hidden_layers must be >= 1");
        anyhow::ensure!(self.inter >= 1, "draft config.json: intermediate_size must be >= 1");
        anyhow::ensure!(self.vocab > 0, "draft config.json: vocab_size must be > 0");
        anyhow::ensure!((self.mask_token_id as usize) < self.vocab,
            "draft config.json: mask_token_id {} out of vocab {}", self.mask_token_id, self.vocab);
        anyhow::ensure!(self.num_kv_heads >= 1 && self.num_heads >= self.num_kv_heads,
            "draft config.json: need 1 <= num_key_value_heads <= num_attention_heads");
        anyhow::ensure!(self.head_dim >= 1, "draft config.json: head_dim must be >= 1");
        anyhow::ensure!(self.block_size >= 1 && self.block_size <= MAX_BLOCK,
            "draft config.json: block_size {} outside 1..={MAX_BLOCK}", self.block_size);
        anyhow::ensure!(self.selector_rank >= 1 && self.selector_top_k >= 1,
            "draft config.json: selector_rank/selector_top_k must be >= 1");
        anyhow::ensure!(self.sliding_window >= 1, "draft config.json: sliding_window must be >= 1");
        anyhow::ensure!(self.max_positions >= 1, "draft config.json: max_position_embeddings must be >= 1");
        Ok(())
    }
}

/// Parse + validate a draft `config.json` body into a [`Geom`] (pure — no global install).
pub fn geom_from_config_json(text: &str) -> anyhow::Result<Geom> {
    let v: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| anyhow::anyhow!("draft config.json parse: {e}"))?;
    let req = |k: &str| -> anyhow::Result<u64> {
        v.get(k).and_then(|x| x.as_u64())
            .ok_or_else(|| anyhow::anyhow!("draft config.json: missing/invalid `{k}`"))
    };
    let df = v.get("dflash_config").cloned()
        .ok_or_else(|| anyhow::anyhow!("draft config.json: missing `dflash_config`"))?;
    let dreq = |k: &str| -> anyhow::Result<u64> {
        df.get(k).and_then(|x| x.as_u64())
            .ok_or_else(|| anyhow::anyhow!("draft config.json: missing/invalid `dflash_config.{k}`"))
    };
    let tap_layers: Vec<usize> = df.get("target_layer_ids")
        .and_then(|x| x.as_array())
        .ok_or_else(|| anyhow::anyhow!("draft config.json: missing/invalid `dflash_config.target_layer_ids`"))?
        .iter().map(|x| x.as_u64().map(|u| u as usize))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| anyhow::anyhow!("draft config.json: target_layer_ids not all integers"))?;
    let g = Geom {
        hidden: req("hidden_size")? as usize,
        num_heads: req("num_attention_heads")? as usize,
        num_kv_heads: req("num_key_value_heads")? as usize,
        head_dim: v.get("head_dim").and_then(|x| x.as_u64()).map(|u| u as usize).unwrap_or(HEAD_DIM),
        inter: req("intermediate_size")? as usize,
        vocab: req("vocab_size")? as usize,
        n_layers: req("num_hidden_layers")? as usize,
        mask_token_id: dreq("mask_token_id")? as u32,
        rope_theta: v.pointer("/rope_parameters/rope_theta").and_then(|x| x.as_f64())
            .unwrap_or(ROPE_THETA as f64) as f32,
        rms_eps: v.get("rms_norm_eps").and_then(|x| x.as_f64()).unwrap_or(RMS_EPS as f64) as f32,
        max_positions: req("max_position_embeddings")? as usize,
        sliding_window: v.get("sliding_window").and_then(|x| x.as_u64()).map(|u| u as usize)
            .unwrap_or(SLIDING_WINDOW),
        is_causal: v.get("is_causal").and_then(|x| x.as_bool()).unwrap_or(IS_CAUSAL),
        conv_kernel: dreq("conv_kernel_size")? as usize,
        conv_group: dreq("conv_group_size")? as usize,
        selector_rank: dreq("selector_rank")? as usize,
        selector_top_k: dreq("selector_top_k")? as usize,
        tap_layers,
        num_target_layers: req("num_target_layers")? as usize,
        block_size: dreq("block_size")? as usize,
    };
    g.validate()?;
    Ok(g)
}

/// The installed geometry, or the 27B default before any install.
static GEOM: RwLock<Option<Geom>> = RwLock::new(None);
/// Set once a round/buffer has been CONSTRUCTED against the installed geometry; after this a
/// DIFFERENT geometry may not be installed (mirrors `set_block`'s cannot-change-after-use).
static GEOM_PINNED: AtomicUsize = AtomicUsize::new(0);

/// The live DFlash2 artifact geometry (27B default until [`install_geom`] runs).
#[inline]
pub fn geom() -> Arc<Geom> {
    static DFLT: OnceLock<Arc<Geom>> = OnceLock::new();
    let r = GEOM.read().unwrap();
    match r.as_ref() {
        Some(g) => Arc::new(g.clone()),
        None => DFLT.get_or_init(|| Arc::new(Geom::default())).clone(),
    }
}

/// Install the artifact geometry. Must run BEFORE any round/buffer construction (the loader
/// does it from the draft's config.json); after [`pin_geom`] only the same value is accepted.
pub fn install_geom(g: Geom) -> anyhow::Result<()> {
    let mut w = GEOM.write().unwrap();
    if GEOM_PINNED.load(Ordering::Acquire) != 0 {
        let cur = w.as_ref().cloned().unwrap_or_default();
        anyhow::ensure!(cur == g,
            "df2 geometry already pinned (a round was constructed against {cur:?}); \
             cannot install a different geometry");
    }
    *w = Some(g);
    Ok(())
}

/// Pin the installed geometry: called by the round/buffer constructors; afterwards
/// [`install_geom`] refuses any different value.
pub fn pin_geom() {
    GEOM_PINNED.store(1, Ordering::Release);
}

/// Install the geometry from a draft dir's `config.json` (the loader's entry point).
/// Logs the one-per-boot geometry line and warns on a block mismatch.
pub fn install_geom_from_dir(dir: &str) -> anyhow::Result<Geom> {
    let path = std::path::Path::new(dir).join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("draft dir {}: cannot read config.json: {e}", dir))?;
    let g = geom_from_config_json(&text)?;
    if g.block_size != super::block() {
        crate::reprintln!("[df2] NOTE: --df2-block {} while the artifact's trained block_size is {} — \
                   the round drafts {} positions; use --df2-block {} for the trained block",
                  super::block(), g.block_size, super::block() - 1, g.block_size);
    }
    crate::reprintln!("[df2] geometry installed from {}: hidden {}, heads {}/{}x{}, inter {}, layers {}, \
               taps {} {:?} (concat {}), mask {}, vocab {}, trunk-target {} layers, block_size {}",
              path.display(), g.hidden, g.num_heads, g.num_kv_heads, g.head_dim, g.inter,
              g.n_layers, g.n_taps(), g.tap_layers, g.tap_concat_dim(), g.mask_token_id,
              g.vocab, g.num_target_layers, g.block_size);
    install_geom(g.clone())?;
    Ok(g)
}

// ---- Live accessors (mod.rs's consts are only the 27B defaults + Geom::default source) ----

#[inline] pub fn hidden() -> usize { geom().hidden }
#[inline] pub fn num_heads() -> usize { geom().num_heads }
#[inline] pub fn num_kv_heads() -> usize { geom().num_kv_heads }
#[inline] pub fn head_dim() -> usize { geom().head_dim }
#[inline] pub fn inter() -> usize { geom().inter }
#[inline] pub fn vocab() -> usize { geom().vocab }
#[inline] pub fn n_layers() -> usize { geom().n_layers }
#[inline] pub fn mask_token_id() -> u32 { geom().mask_token_id }
#[inline] pub fn rope_theta() -> f32 { geom().rope_theta }
#[inline] pub fn rms_eps() -> f32 { geom().rms_eps }
#[inline] pub fn max_positions() -> usize { geom().max_positions }
#[inline] pub fn sliding_window() -> usize { geom().sliding_window }
#[inline] pub fn is_causal() -> bool { geom().is_causal }
#[inline] pub fn conv_kernel() -> usize { geom().conv_kernel }
#[inline] pub fn conv_group() -> usize { geom().conv_group }
#[inline] pub fn conv_groups() -> usize { geom().conv_groups() }
#[inline] pub fn selector_rank() -> usize { geom().selector_rank }
#[inline] pub fn selector_top_k() -> usize { geom().selector_top_k }
#[inline] pub fn n_taps() -> usize { geom().n_taps() }
#[inline] pub fn tap_concat_dim() -> usize { geom().tap_concat_dim() }
#[inline] pub fn n_tensors() -> usize { geom().n_tensors() }
#[inline] pub fn num_target_layers() -> usize { geom().num_target_layers }
/// The tap layer ids (a fresh Vec per call — call sites are per-round, never per-element).
#[inline] pub fn tap_layers() -> Vec<usize> { geom().tap_layers.clone() }
/// The safe absolute-context bound for the round = the artifact's trained positions.
pub fn max_ctx_safe() -> usize { geom().max_positions }

/// Exact parameter count of the checkpoint for the LIVE geometry (was the 27B `N_PARAMS`).
pub fn n_params() -> u64 {
    reconcile_params_geom(&geom())
}

/// The tensor inventory for an arbitrary geometry (mod.rs's `inventory()` generalized:
/// 6 globals then `n_layers` draft layers, exact names per the safetensors header).
pub fn inventory_geom(g: &Geom) -> Vec<(String, Vec<usize>)> {
    let mut v = Vec::with_capacity(g.n_tensors());
    let tap_concat = g.tap_concat_dim();
    let q = g.num_heads * g.head_dim;
    let kv = g.num_kv_heads * g.head_dim;
    let conv = 2 * g.conv_kernel * g.conv_groups();
    v.push(("fc.weight".to_string(), vec![g.hidden, tap_concat]));
    v.push(("hidden_norm.weight".to_string(), vec![g.hidden]));
    v.push(("norm.weight".to_string(), vec![g.hidden]));
    v.push(("candidate_selector.hidden_projection.weight".to_string(), vec![g.selector_rank, g.hidden]));
    v.push(("candidate_selector.predecessor_codebook".to_string(), vec![g.vocab, g.selector_rank]));
    v.push(("candidate_selector.successor_codebook".to_string(), vec![g.vocab, g.selector_rank]));
    for i in 0..g.n_layers {
        let lp = format!("layers.{i}");
        v.push((format!("{lp}.self_attn.q_proj.weight"), vec![q, g.hidden]));
        v.push((format!("{lp}.self_attn.k_proj.weight"), vec![kv, g.hidden]));
        v.push((format!("{lp}.self_attn.v_proj.weight"), vec![kv, g.hidden]));
        v.push((format!("{lp}.self_attn.o_proj.weight"), vec![g.hidden, q]));
        v.push((format!("{lp}.self_attn.q_norm.weight"), vec![g.head_dim]));
        v.push((format!("{lp}.self_attn.k_norm.weight"), vec![g.head_dim]));
        v.push((format!("{lp}.input_layernorm.weight"), vec![g.hidden]));
        v.push((format!("{lp}.post_attention_layernorm.weight"), vec![g.hidden]));
        v.push((format!("{lp}.mlp.gate_proj.weight"), vec![g.inter, g.hidden]));
        v.push((format!("{lp}.mlp.up_proj.weight"), vec![g.inter, g.hidden]));
        v.push((format!("{lp}.mlp.down_proj.weight"), vec![g.hidden, g.inter]));
        v.push((format!("{lp}.attention_conv.base_kernel"), vec![2, g.conv_kernel, g.hidden]));
        v.push((format!("{lp}.attention_conv.kernel_projection.weight"), vec![conv, g.hidden]));
        v.push((format!("{lp}.mlp_conv.base_kernel"), vec![2, g.conv_kernel, g.hidden]));
        v.push((format!("{lp}.mlp_conv.kernel_projection.weight"), vec![conv, g.hidden]));
    }
    debug_assert_eq!(v.len(), g.n_tensors());
    v
}

/// Reconcile the inventory to the exact parameter count for the LIVE geometry.
pub fn reconcile_params_geom(g: &Geom) -> u64 {
    inventory_geom(g).iter().map(|(_, s)| s.iter().map(|&d| d as u64).product::<u64>()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The REAL Ornith-1.5-122B-DFlash2 config.json (verbatim subset of the artifact's file;
    /// every field the parser reads, values checked against the artifact 2026-10-23).
    const ORNITH_CFG: &str = r#"{
        "hidden_size": 3072, "num_attention_heads": 32, "num_key_value_heads": 8,
        "head_dim": 128, "intermediate_size": 9216, "num_hidden_layers": 6,
        "vocab_size": 248320, "rms_norm_eps": 1e-06, "max_position_embeddings": 262144,
        "num_target_layers": 48, "is_causal": false, "sliding_window": 2048,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "dflash_config": {"block_size": 16, "conv_group_size": 16, "conv_kernel_size": 2,
            "mask_token_id": 248077, "selector_rank": 256, "selector_top_k": 16,
            "target_layer_ids": [1, 7, 14, 20, 26, 32, 39, 45]}
    }"#;

    #[test]
    fn ornith_geometry_parses() {
        let g = geom_from_config_json(ORNITH_CFG).expect("ornith config parses");
        assert_eq!(g.hidden, 3072);
        assert_eq!(g.inter, 9216);
        assert_eq!(g.n_layers, 6);
        assert_eq!(g.mask_token_id, 248077);
        assert_eq!(g.tap_layers, vec![1, 7, 14, 20, 26, 32, 39, 45]);
        assert_eq!(g.n_taps(), 8);
        assert_eq!(g.tap_concat_dim(), 24576);
        assert_eq!(g.conv_groups(), 192);
        assert_eq!(g.n_tensors(), 96);
        assert_eq!(g.num_target_layers, 48);
        assert_eq!(g.block_size, 16);
        // param count = 96-tensor inventory sum. The literal is the artifact's OWN header
        // ground truth (model.safetensors parsed 2026-10-23: 96 tensors, 930,278,912 params;
        // fc [3072,24576], q_proj [4096,3072] = 32 heads x 128, k/v [1024,3072], attn conv
        // [768,3072] = hidden/4, mlp [9216,3072], selector codebooks 2x[248320,256]).
        assert_eq!(reconcile_params_geom(&g), 930_278_912u64);
    }

    #[test]
    fn ornith_trunk_compat() {
        let g = geom_from_config_json(ORNITH_CFG).unwrap();
        assert!(g.trunk_compatible(3072, 248320, 48), "the Ornith trunk matches its draft");
        assert!(!g.trunk_compatible(5120, 248320, 64), "the 27B trunk must be refused (hidden)");
        assert!(!g.trunk_compatible(3072, 248320, 64), "wrong depth must be refused");
        assert!(!g.trunk_compatible(3072, 248320, 47), "one layer short must be refused");
        assert!(!g.trunk_compatible(3072, 100000, 48), "smaller vocab must be refused");
        // The 27B default geometry still matches the 27B trunk (regression guard).
        assert!(Geom::default().trunk_compatible(5120, 248320, 64));
    }

    /// The REAL Qwen3.8-Flash-Next-DFlash2 config.json (verbatim subset; values checked
    /// against the artifact header 2026-10-23 FLASHNEXT-DF2 Phase 0: 81 tensors,
    /// 603,065,600 params BF16, fc [2560,12800], codebooks [248320,256]).
    const FLASHNEXT_CFG: &str = r#"{
        "hidden_size": 2560, "num_attention_heads": 32, "num_key_value_heads": 8,
        "head_dim": 128, "intermediate_size": 7680, "num_hidden_layers": 5,
        "vocab_size": 248320, "rms_norm_eps": 1e-06, "max_position_embeddings": 262144,
        "num_target_layers": 48, "is_causal": false, "sliding_window": 2048,
        "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
        "dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
            "mask_token_id": 248077, "selector_rank": 256, "selector_top_k": 16,
            "target_layer_ids": [5, 15, 25, 35, 45]}
    }"#;

    #[test]
    fn flashnext_geometry_parses_and_compat() {
        let g = geom_from_config_json(FLASHNEXT_CFG).expect("flash-next config parses");
        assert_eq!(g.hidden, 2560);
        assert_eq!(g.inter, 7680);
        assert_eq!(g.n_layers, 5);
        assert_eq!(g.mask_token_id, 248077);
        assert_eq!(g.rope_theta, 10000.0, "Flash-Next trains at 1e4, NOT the 27B's 1e7");
        assert_eq!(g.tap_layers, vec![5, 15, 25, 35, 45]);
        assert_eq!(g.n_taps(), 5);
        assert_eq!(g.tap_concat_dim(), 12800);
        assert_eq!(g.conv_groups(), 160);
        assert_eq!(g.n_tensors(), 81);
        assert_eq!(g.num_target_layers, 48);
        assert_eq!(g.block_size, 8);
        // Param count = the artifact's OWN header ground truth (Phase 0 dump).
        assert_eq!(reconcile_params_geom(&g), 603_065_600u64);
        // Trunk compat: the Flash-Next pack (hidden 2560, vocab 248320, 48 layers) matches;
        // the 27B trunk is refused; the default (27B) geometry still matches the 27B trunk.
        assert!(g.trunk_compatible(2560, 248320, 48), "the Flash-Next pack matches its draft");
        assert!(!g.trunk_compatible(5120, 248320, 64), "the 27B trunk must be refused (hidden)");
        assert!(!g.trunk_compatible(2560, 248320, 45), "tap at 45 needs >45 trunk layers");
        assert!(Geom::default().trunk_compatible(5120, 248320, 64));
        // Inventory: the fc concatenates 5 taps x 2560; the selector codebooks at rank 256.
        let inv = inventory_geom(&g);
        assert_eq!(inv.len(), 81);
        let fc = inv.iter().find(|(n, _)| n == "fc.weight").unwrap();
        assert_eq!(fc.1, vec![2560, 12800]);
        let l4q = inv.iter().find(|(n, _)| n == "layers.4.self_attn.q_proj.weight").unwrap();
        assert_eq!(l4q.1, vec![4096, 2560]);
        let cb = inv.iter().find(|(n, _)| n == "candidate_selector.successor_codebook").unwrap();
        assert_eq!(cb.1, vec![248320, 256]);
    }

    #[test]
    fn bad_configs_refused() {
        assert!(geom_from_config_json("{}").is_err(), "missing dflash_config");
        assert!(geom_from_config_json(
            r#"{"hidden_size":3072,"num_attention_heads":32,"num_key_value_heads":8,
                "intermediate_size":9216,"num_hidden_layers":6,"vocab_size":248320,
                "max_position_embeddings":262144,"num_target_layers":48,
                "dflash_config":{"block_size":16,"conv_group_size":16,"conv_kernel_size":2,
                "mask_token_id":248077,"selector_rank":256,"selector_top_k":16,
                "target_layer_ids":[7,1,14]}}"#
        ).is_err(), "non-increasing taps refused");
        assert!(geom_from_config_json(
            r#"{"hidden_size":3072,"num_attention_heads":32,"num_key_value_heads":8,
                "intermediate_size":9216,"num_hidden_layers":6,"vocab_size":248320,
                "max_position_embeddings":262144,"num_target_layers":48,
                "dflash_config":{"block_size":16,"conv_group_size":16,"conv_kernel_size":2,
                "mask_token_id":248077,"selector_rank":256,"selector_top_k":16,
                "target_layer_ids":[48,7]}}"#
        ).is_err(), "tap at/above trunk depth refused");
    }

    #[test]
    fn inventory_names_shape_count() {
        let g = geom_from_config_json(ORNITH_CFG).unwrap();
        let inv = inventory_geom(&g);
        assert_eq!(inv.len(), 96);
        let fc = inv.iter().find(|(n, _)| n == "fc.weight").unwrap();
        assert_eq!(fc.1, vec![3072, 24576]);
        let l5q = inv.iter().find(|(n, _)| n == "layers.5.self_attn.q_proj.weight").unwrap();
        assert_eq!(l5q.1, vec![4096, 3072]);
        let conv = inv.iter().find(|(n, _)| n == "layers.3.mlp_conv.kernel_projection.weight").unwrap();
        assert_eq!(conv.1, vec![768, 3072]);
        // and the 27B default inventory is unchanged (regression guard vs today's 81)
        let d = inventory_geom(&Geom::default());
        assert_eq!(d.len(), 81);
        let fc27 = d.iter().find(|(n, _)| n == "fc.weight").unwrap();
        assert_eq!(fc27.1, vec![5120, 25600]);
    }
}
