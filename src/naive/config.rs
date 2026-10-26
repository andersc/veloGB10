//! Naive-N0.5-Flash architecture config (S1). Parsed from the pack's `config.json` (the vLLM-
//! rewritten one or `config.json.native` — the architecture fields are identical). Ground truth for
//! every field's meaning: `/mnt/models/NaiveAI/Naive-N0.5-Flash-main/{configuration,modeling}_naive_n05_flash.py`
//! and `PLAN/NAIVE_N05_PORT.md` §1. Every assumption the engine makes is an `ensure!` here, so a
//! different checkpoint of the family refuses at load with the field named (the wrong-model hazard).

use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

/// Which attention a layer runs (`hybrid_layer_pattern`: 1 = sliding window, 0 = DSA).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnKind {
    /// Sliding-window attention: window 128, learned per-head sink, swa_* geometry, swa_rope_theta.
    Swa,
    /// DeepSeek sparse attention: lightning indexer -> top-k keys, no sink, rope_theta.
    Dsa,
}

#[derive(Clone, Debug)]
pub struct NaiveConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub rms_eps: f32,
    pub max_position_embeddings: usize,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub tie_word_embeddings: bool,

    // attention — one entry per layer
    pub attn: Vec<AttnKind>,
    pub num_heads: usize,       // 64 q heads, both kinds
    pub head_dim: usize,        // 192 (QK), both kinds
    pub v_head_dim: usize,      // 128, both kinds
    pub dsa_kv_heads: usize,    // 4
    pub swa_kv_heads: usize,    // 8
    pub sliding_window: usize,  // 128
    pub rotary_dim: usize,      // int(192 * 0.334) = 64, NEOX rotate-half on the first 64 dims
    pub rope_theta: f32,        // DSA layers (1e7)
    pub swa_rope_theta: f32,    // SWA layers (1e4)
    pub swa_sink: bool,         // add_swa_attention_sink_bias
    pub dsa_sink: bool,         // add_full_attention_sink_bias (false in the shipped config)
    pub value_scale: f32,       // attention_value_scale 0.707 (folded into o_proj by the quantizer — see pack)

    // DSA indexer
    pub index_n_heads: usize,   // 16
    pub index_head_dim: usize,  // 128
    pub index_top_k: usize,     // 2048
    pub indexer_fp8: bool,      // indexer_activation_dtype == fp8_e4m3

    // MoE
    pub moe: Vec<bool>,              // moe_layer_freq per layer (layer 0 dense)
    pub intermediate_size: usize,    // dense MLP 16384
    pub num_experts: usize,          // 256
    pub num_experts_per_tok: usize,  // 8
    pub moe_intermediate_size: usize, // 2048
    pub routed_scaling_factor: f32,  // 1.0
    pub norm_topk_prob: bool,        // true
    /// Per-layer divisor the EXL3 converter applied to the routed experts' up_proj (exllamav3
    /// architecture/naive_n05.py: `interm_div = 128.0 if idx == 47 else 1.0`, because layer 47's
    /// expert activations overflow f16). The pack stores up_proj / 128 at that layer and the
    /// forward compensates by scaling the ROUTING WEIGHTS by 128 (block_sparse_mlp.py folds it into
    /// routed_scaling_factor). Without it the last layer's MoE comes out 128x too small (the
    /// 2026-10-19 bisect: NLL 2.020 vs 1.619 in exllamav3 on the same pack).
    pub moe_interm_div: Vec<f32>,
}

impl NaiveConfig {
    pub fn from_dir(dir: &str) -> Result<Self> {
        let path = format!("{}/config.json", dir.trim_end_matches('/'));
        let raw = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
        Self::from_json(&raw).with_context(|| format!("naive config {path}"))
    }

    pub fn from_json(raw: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(raw).context("config.json unparsable")?;
        let mt = v["model_type"].as_str().unwrap_or("");
        ensure!(mt == "naive_n05_flash", "model_type {mt:?} is not naive_n05_flash");
        let arch = v["architectures"][0].as_str().unwrap_or("");
        ensure!(arch == "NaiveN05FlashForCausalLM", "architectures[0] {arch:?} is not NaiveN05FlashForCausalLM");

        let u = |k: &str| -> Result<usize> { v[k].as_u64().map(|x| x as usize).with_context(|| format!("config: {k} missing")) };
        let f = |k: &str| -> Result<f32> { v[k].as_f64().map(|x| x as f32).with_context(|| format!("config: {k} missing")) };
        let s = |k: &str| -> String { v[k].as_str().unwrap_or("").to_string() };
        let b = |k: &str, d: bool| -> bool { v[k].as_bool().unwrap_or(d) };

        let num_layers = u("num_hidden_layers")?;
        let hidden_size = u("hidden_size")?;
        let head_dim = u("head_dim")?;
        let v_head_dim = v["v_head_dim"].as_u64().map(|x| x as usize).unwrap_or(head_dim);
        let num_heads = u("num_attention_heads")?;
        let dsa_kv_heads = u("num_key_value_heads")?;
        let swa_kv_heads = v["swa_num_key_value_heads"].as_u64().map(|x| x as usize).unwrap_or(dsa_kv_heads);
        // The two attention kinds must share q heads and head dims (one kernel family).
        ensure!(v["swa_num_attention_heads"].as_u64().map_or(true, |x| x as usize == num_heads), "swa_num_attention_heads != num_attention_heads");
        ensure!(v["swa_head_dim"].as_u64().map_or(true, |x| x as usize == head_dim), "swa_head_dim != head_dim");
        ensure!(v["swa_v_head_dim"].as_u64().map_or(true, |x| x as usize == v_head_dim), "swa_v_head_dim != v_head_dim");
        ensure!(num_heads % dsa_kv_heads == 0 && num_heads % swa_kv_heads == 0, "q heads not a multiple of kv heads");

        ensure!(s("attention_projection_layout") == "split", "attention_projection_layout must be split (q/k/v as separate tensors)");
        ensure!(!b("attention_bias", false), "attention_bias=true is not supported (the pack has no biases)");

        let partial = f("partial_rotary_factor")?;
        let rotary_dim = (head_dim as f32 * partial) as usize;
        ensure!(rotary_dim % 2 == 0 && rotary_dim > 0 && rotary_dim <= head_dim, "rotary dim {rotary_dim} invalid");
        let rope_type = v["rope_scaling"]["rope_type"].as_str().or(v["rope_scaling"]["type"].as_str()).unwrap_or("default");
        ensure!(rope_type == "default", "rope_scaling type {rope_type:?}: only default (no scaling) is supported");

        let pattern = v["hybrid_layer_pattern"].as_array().context("hybrid_layer_pattern missing")?;
        ensure!(pattern.len() == num_layers, "hybrid_layer_pattern has {} entries for {num_layers} layers", pattern.len());
        let attn: Vec<AttnKind> = pattern.iter().map(|x| if x.as_u64() == Some(1) { AttnKind::Swa } else { AttnKind::Dsa }).collect();

        let sliding_window = u("sliding_window")?;
        ensure!(sliding_window > 0, "sliding_window must be positive");

        // DSA indexer
        ensure!(b("enable_dsa", true), "enable_dsa=false is not a Naive-N0.5-Flash checkpoint");
        let index_n_heads = u("index_n_heads")?;
        let index_head_dim = u("index_head_dim")?;
        let index_top_k = u("index_top_k")?;
        ensure!(v["index_n_kv_heads"].as_u64().unwrap_or(1) == 1, "index_n_kv_heads must be 1");
        ensure!(index_top_k > 0, "index_top_k must be positive");
        ensure!(rotary_dim <= index_head_dim, "indexer head dim {index_head_dim} must cover the rotary dim {rotary_dim}");
        let act = s("indexer_activation_dtype");
        let indexer_fp8 = match act.as_str() {
            "fp8_e4m3" => true,
            "bf16" | "" => false,
            other => bail!("indexer_activation_dtype {other:?} unsupported"),
        };

        // MoE
        ensure!(s("scoring_func") == "sigmoid" && s("topk_method") == "noaux_tc", "router must be sigmoid + noaux_tc");
        ensure!(v["n_group"].as_u64().unwrap_or(1) == 1 && v["topk_group"].as_u64().unwrap_or(1) == 1, "grouped routing (n_group/topk_group != 1) unsupported");
        ensure!(v["n_shared_experts"].is_null() || v["n_shared_experts"].as_u64() == Some(0), "shared experts are not part of this family");
        let freq = v["moe_layer_freq"].as_array().context("moe_layer_freq missing")?;
        ensure!(freq.len() == num_layers, "moe_layer_freq has {} entries for {num_layers} layers", freq.len());
        let moe: Vec<bool> = freq.iter().map(|x| x.as_u64() == Some(1)).collect();
        let num_experts = u("n_routed_experts")?;
        let num_experts_per_tok = u("num_experts_per_tok")?;
        let moe_intermediate_size = u("moe_intermediate_size")?;
        let intermediate_size = u("intermediate_size")?;
        ensure!(num_experts_per_tok <= 16 && num_experts_per_tok > 0, "top-k {num_experts_per_tok} out of the kernels' range");
        let hidden_act = s("hidden_act");
        ensure!(hidden_act == "silu", "hidden_act {hidden_act:?}: only silu");

        let eos = v["eos_token_id"].as_u64().or_else(|| v["eos_token_id"].as_array().and_then(|a| a.first()).and_then(|x| x.as_u64()))
            .context("eos_token_id missing")? as u32;
        let pad = v["pad_token_id"].as_u64().unwrap_or(eos as u64) as u32;

        Ok(NaiveConfig {
            hidden_size, num_layers,
            vocab_size: u("vocab_size")?,
            rms_eps: v["layernorm_epsilon"].as_f64().or(v["rms_norm_eps"].as_f64()).unwrap_or(1e-5) as f32,
            max_position_embeddings: u("max_position_embeddings")?,
            eos_token_id: eos, pad_token_id: pad,
            tie_word_embeddings: b("tie_word_embeddings", false),
            attn, num_heads, head_dim, v_head_dim, dsa_kv_heads, swa_kv_heads, sliding_window, rotary_dim,
            rope_theta: f("rope_theta")?,
            swa_rope_theta: v["swa_rope_theta"].as_f64().map(|x| x as f32).unwrap_or(f("rope_theta")?),
            swa_sink: b("add_swa_attention_sink_bias", false),
            dsa_sink: b("add_full_attention_sink_bias", false),
            value_scale: v["attention_value_scale"].as_f64().map(|x| x as f32).unwrap_or(1.0),
            index_n_heads, index_head_dim, index_top_k, indexer_fp8,
            moe, intermediate_size, num_experts, num_experts_per_tok, moe_intermediate_size,
            routed_scaling_factor: v["routed_scaling_factor"].as_f64().map(|x| x as f32).unwrap_or(1.0),
            norm_topk_prob: b("norm_topk_prob", true),
            moe_interm_div: (0..num_layers).map(|l| if l == 47 { 128.0 } else { 1.0 }).collect(),
        })
    }

    /// The routing-weight scale of a PACK-loaded MoE layer: the config's routed_scaling_factor times the
    /// converter's up_proj divisor for that layer (see `moe_interm_div`). A bf16 oracle (undivided
    /// weights) uses `routed_scaling_factor` alone.
    pub fn pack_router_scale(&self, layer: usize) -> f32 { self.routed_scaling_factor * self.moe_interm_div[layer] }

    pub fn kv_heads(&self, layer: usize) -> usize {
        match self.attn[layer] { AttnKind::Swa => self.swa_kv_heads, AttnKind::Dsa => self.dsa_kv_heads }
    }
    pub fn has_sink(&self, layer: usize) -> bool {
        match self.attn[layer] { AttnKind::Swa => self.swa_sink, AttnKind::Dsa => self.dsa_sink }
    }
    pub fn rope_theta_of(&self, layer: usize) -> f32 {
        match self.attn[layer] { AttnKind::Swa => self.swa_rope_theta, AttnKind::Dsa => self.rope_theta }
    }
    pub fn n_swa(&self) -> usize { self.attn.iter().filter(|a| **a == AttnKind::Swa).count() }
    pub fn n_dsa(&self) -> usize { self.attn.iter().filter(|a| **a == AttnKind::Dsa).count() }
    pub fn n_moe(&self) -> usize { self.moe.iter().filter(|m| **m).count() }
    pub fn n_dense(&self) -> usize { self.num_layers - self.n_moe() }
    /// Is this a model-dir this family serves? (cheap: config.json model_type only)
    pub fn is_naive_dir(dir: &str) -> bool {
        std::fs::read_to_string(format!("{}/config.json", dir.trim_end_matches('/'))).ok()
            .and_then(|s| serde_json::from_str::<Value>(&s).ok())
            .map_or(false, |v| v["model_type"].as_str() == Some("naive_n05_flash"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PACK: &str = "/mnt/models/NaiveAI/Naive-N0.5-Flash-exl3-3.5bpw-vllm";

    #[test]
    fn the_shipped_pack_config_parses_to_the_documented_geometry() {
        if !std::path::Path::new(PACK).join("config.json").is_file() { eprintln!("skip: pack not present"); return; }
        let c = NaiveConfig::from_dir(PACK).unwrap();
        assert_eq!((c.num_layers, c.hidden_size, c.vocab_size), (48, 4096, 152_576));
        assert_eq!((c.num_heads, c.head_dim, c.v_head_dim), (64, 192, 128));
        assert_eq!((c.dsa_kv_heads, c.swa_kv_heads, c.sliding_window, c.rotary_dim), (4, 8, 128, 64));
        assert_eq!((c.n_swa(), c.n_dsa(), c.n_moe(), c.n_dense()), (39, 9, 47, 1));
        assert_eq!(c.attn[0], AttnKind::Dsa);
        assert_eq!(c.attn[1], AttnKind::Swa);
        assert_eq!(c.attn[47], AttnKind::Dsa);
        for l in [0usize, 5, 11, 17, 23, 29, 35, 41, 47] { assert_eq!(c.attn[l], AttnKind::Dsa, "layer {l}"); }
        assert!(!c.moe[0] && c.moe[1]);
        assert_eq!((c.index_n_heads, c.index_head_dim, c.index_top_k), (16, 128, 2048));
        assert!(c.indexer_fp8 && c.swa_sink && !c.dsa_sink);
        assert_eq!((c.num_experts, c.num_experts_per_tok, c.moe_intermediate_size, c.intermediate_size), (256, 8, 2048, 16384));
        assert!((c.value_scale - 0.707).abs() < 1e-6);
        assert_eq!(c.moe_interm_div.len(), 48);
        assert!(c.moe_interm_div[..47].iter().all(|d| *d == 1.0) && c.moe_interm_div[47] == 128.0);
        assert_eq!(c.pack_router_scale(47), 128.0 * c.routed_scaling_factor);
        assert_eq!((c.rope_theta, c.swa_rope_theta), (1e7, 1e4));
        assert_eq!(c.max_position_embeddings, 1 << 20);
        assert_eq!((c.eos_token_id, c.pad_token_id), (151_645, 151_643));
        assert!(NaiveConfig::is_naive_dir(PACK));
    }

    #[test]
    fn a_foreign_model_type_is_refused() {
        let e = NaiveConfig::from_json(r#"{"model_type":"qwen4_exp","architectures":["X"]}"#).unwrap_err();
        assert!(e.to_string().contains("naive_n05_flash"));
        assert!(!NaiveConfig::is_naive_dir("/nonexistent"));
    }
}
