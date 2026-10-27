//! DSpark drafter, GPU (S8): one speculation round of `Naive-N0.5-Flash-FP8-Draft` on the device,
//! op for op after the CPU reference [`super::drafter_cpu::CpuDrafter::round`] (PLAN §1.2), kernels in
//! `kernels/gpu_naive_draft.cu` (module `gpu_naive_draft`).
//!
//! The drafter is not quantized: its bf16 weights live on the device as plain f16 (bf16 -> f16 at
//! load; values outside the f16 range are refused, subnormals counted). Norm weights, the mask
//! embedding and the confidence head stay f32 (a few KB; no reason to round them).
//!
//! Split of one round (each piece is a pub method so serving can interleave its own work):
//!   * [`GpuDrafter::append_context`] — committed target positions -> the drafter's context K/V:
//!     `hidden_norm(fc(concat of the 8 tap residuals))` projected by every layer's k/v (k_norm +
//!     rope at the position) into a per-layer f16 ring of `S = window.next_power_of_two()` slots.
//!     Done ONCE per committed position (vLLM caches it the same way); the CPU reference recomputes
//!     it every round, which is the same arithmetic. Only the last `S` rows of a call can matter.
//!   * [`GpuDrafter::backbone`] — the block (anchor embedding + N-1 mask embeddings at positions
//!     t..t+N-1) through the 5 layers, attending NON-causally to the context window + the whole
//!     block; returns the final-norm hidden (f16 for the lm_head, f32 for the confidence head).
//!   * the target's lm_head (caller-supplied: the EXL3 chain in serving, a dense f16 matrix in the
//!     gate) -> base logits f16 [N][V];
//!   * [`GpuDrafter::markov`] — sequential Markov correction + greedy pick + confidence, entirely on
//!     the device (each row's `prev` is read from the token the previous row wrote: no host trip).
//!
//! Rounding points: see the header of `kernels/gpu_naive_draft.cu`.
//!
//! Gate: `--probe-naive-drafter` (GPU round vs the CPU round on the HF-oracle fixture and on a
//! long synthetic context that wraps the ring and binds the window), then 20 timed rounds of the real
//! drafter.

use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use super::drafter_cpu::{CpuDrafter, DrafterConfig};
use super::model::{dptr, dtoh, Ws};
use super::shard::Shard;
use super::tp::NaiveTp;

pub const MODULE: &str = "gpu_naive_draft";
pub const FNS: &[&str] = &[
    "nd_kernel_build_id",
    "nd_gemm_m1", "nd_gemm_m2", "nd_gemm_m4", "nd_gemm_m8", "nd_combine", "nd_combine_silu",
    "nd_rmsnorm", "nd_tap_f16", "nd_ctx_kv", "nd_block_init", "nd_block_qkv", "nd_attn_part", "nd_attn_combine",
    "nd_markov_scan", "nd_markov_pick", "nd_markov_qnorm", "nd_scatter_cols", "nd_markov_fold", "nd_markov_unpack",
];

/// warps per CTA of `nd_gemm_m*` (ND_WPB in the .cu)
const WPB: usize = 4;
/// rows per context-append launch (8 m16 tiles share one weight load)
const APPEND_ROWS: usize = 128;
/// most block rows a round takes (one m16 tile)
pub const MAX_BLOCK: usize = 16;
/// CTAs of the Markov scan (each strides over the vocabulary; partials reduced by the pick kernel)
const MARKOV_BLOCKS: usize = 384;
/// keys per attention CTA (ND_CH in the .cu)
const ATTN_CHUNK: usize = 32;

/// Load the embedded module (idempotent) and refuse a binary/PTX build-id mismatch.
pub fn load_module(dev: &Arc<CudaDevice>) -> Result<()> {
    if dev.get_func(MODULE, "nd_kernel_build_id").is_none() {
        dev.load_ptx(Ptx::from_src(crate::ptx_embed::GPU_NAIVE_DRAFT), MODULE, FNS).context("load gpu_naive_draft module")?;
    }
    let f = dev.get_func(MODULE, "nd_kernel_build_id").context("nd_kernel_build_id missing")?;
    let mut out = dev.alloc_zeros::<u64>(1)?;
    unsafe { f.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 }, (&mut out,)) }?;
    let got = dev.dtoh_sync_copy(&out)?[0];
    let want = u64::from_str_radix(env!("KERNEL_BUILD_ID"), 16).unwrap_or(0);
    ensure!(got == want, "gpu_naive_draft kernel build id {got:016x} != binary {want:016x} (stale PTX)");
    Ok(())
}

pub struct GpuDrafterLayer {
    pub in_norm: CudaSlice<f32>,
    pub post_norm: CudaSlice<f32>,
    /// f16 [(nh + 2 kv) * hd][H]: q rows, then k rows, then v rows (the context path uses the k|v
    /// tail as one [2 kv hd][H] matrix)
    pub qkv: CudaSlice<u16>,
    pub o: CudaSlice<u16>,       // f16 [H][nh*hd]
    pub gate_up: CudaSlice<u16>, // f16 [2I][H]: gate rows, then up rows
    pub down: CudaSlice<u16>,    // f16 [H][I]
    pub q_norm: CudaSlice<f32>,
    pub k_norm: CudaSlice<f32>,
    /// context K / V ring, f16 [S][kv][hd], slot = pos & (S-1)
    pub kc: CudaSlice<u16>,
    pub vc: CudaSlice<u16>,
}

/// What the bf16 -> f16 conversion did to the weights (the drafter's only lossy load step).
#[derive(Clone, Debug, Default)]
pub struct LoadStats {
    pub values: usize,
    /// nonzero values that land in f16's subnormal range or flush to zero (lose relative precision)
    pub subnormal: usize,
    pub max_abs: f32,
}

pub struct GpuDrafter {
    pub dev: Arc<CudaDevice>,
    pub cfg: DrafterConfig,
    pub fc: CudaSlice<u16>, // f16 [H][taps*H]
    pub hidden_norm: CudaSlice<f32>,
    pub mask_embedding: CudaSlice<f32>,
    pub layers: Vec<GpuDrafterLayer>,
    pub norm: CudaSlice<f32>,
    pub markov_w1: CudaSlice<u16>, // f16 [V][r]
    pub markov_w2: CudaSlice<u16>, // f16 [V][r]
    pub conf_w: Option<CudaSlice<f32>>, // [H + r]
    pub conf_b: f32,
    /// ring slots of the context K/V (power of two >= the window)
    pub ring: usize,
    pub stats: LoadStats,
    /// Work item B: the rank's shard of the drafter (world 1 = the whole drafter on this device).
    /// Heads: q [q0, q0 + nh_l), kv [kv0, kv0 + kv_l); MLP intermediate [inter_l] of inter; fc output rows
    /// [h_l] of H (gathered); Markov w2 / the lm_head over the vocabulary slice [v0, v0 + vl).
    pub shard: Shard,
    pub nh_l: usize,
    pub kv_l: usize,
    pub inter_l: usize,
    pub h_l: usize,
    pub v0: usize,
    pub vl: usize,
    /// the exchange buffer (fp32 sum all-reduce = all-gather of zero-padded slots): fc rows, the per-layer
    /// partial sums, the Markov argmax slots
    pub xch: CudaSlice<f32>,
}

/// Device addresses of a round's outputs (all `ws` memory, valid until the caller resets `ws`).
#[derive(Clone, Copy, Debug)]
pub struct DraftBufs {
    pub n: usize,
    /// i32 [n + 1]: tok[0] = anchor, tok[i + 1] = the draft at block row i
    pub tok: u64,
    pub conf: u64,     // f32 [n]
    pub hidden16: u64, // f16 [n][H] (the lm_head input)
    pub hidden32: u64, // f32 [n][H]
}

#[derive(Clone, Debug)]
pub struct DraftRound {
    pub tokens: Vec<u32>,     // [N] drafts (block row i predicts position t + i + 1): greedy argmax, or one draw from q
    pub confidence: Vec<f32>, // [N]
    pub hidden: Vec<f32>,     // [N][H] final-norm hidden (f32)
    /// Sampled drafting only (work item A): the normaliser of q_i, i.e. q_i(v) = exp(qs[i][v] - qmax[i]) / qz[i].
    /// Empty on the greedy path.
    pub qmax: Vec<f32>,
    pub qz: Vec<f32>,
}

/// A sampled draft request (work item A): `qs` = the caller's persistent f32 buffer of `n * vocab`
/// scaled logits (row i at `qs + i*vocab*4`), `qstat` = `2 * n` floats (max, Z per row), `rng` = the
/// round's seed already folded with the step.
#[derive(Clone, Copy, Debug)]
pub struct SmpDraft {
    pub t: f32,
    pub rng: u64,
    pub smb: u64,
    pub qstat: u64,
}

fn grid1(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (((n + 255) / 256).clamp(1, 1024) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}
fn rows_cfg(rows: usize, threads: u32) -> LaunchConfig {
    LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }
}

/// Tensor bytes of one safetensors entry -> f16 bits (counting what the conversion loses).
fn to_f16(st: &safetensors::SafeTensors, name: &str, stats: &mut LoadStats) -> Result<Vec<u16>> {
    let v = to_f32(st, name)?;
    let mut out = Vec::with_capacity(v.len());
    for x in v {
        let a = x.abs();
        ensure!(a.is_finite() && a <= 65504.0, "drafter tensor {name}: value {x} outside the f16 range");
        if a != 0.0 && a < 6.103_515_6e-5 { stats.subnormal += 1; }
        stats.max_abs = stats.max_abs.max(a);
        out.push(half::f16::from_f32(x).to_bits());
    }
    stats.values += out.len();
    Ok(out)
}
fn to_f32(st: &safetensors::SafeTensors, name: &str) -> Result<Vec<f32>> {
    let t = st.tensor(name).with_context(|| format!("drafter tensor {name} missing"))?;
    Ok(match t.dtype() {
        safetensors::Dtype::F32 => t.data().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        safetensors::Dtype::BF16 => t.data().chunks_exact(2).map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
        safetensors::Dtype::F16 => t.data().chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
        other => bail!("{name}: dtype {other:?}"),
    })
}

impl GpuDrafter {
    /// Load `dir/file` (the shipped naming, as `CpuDrafter::load_plain` reads it; `target.*` tensors
    /// are ignored) with `dir/config.json`.
    pub fn load(dev: &Arc<CudaDevice>, dir: &str, file: &str) -> Result<GpuDrafter> {
        let cfg = DrafterConfig::from_json(&std::fs::read_to_string(format!("{}/config.json", dir.trim_end_matches('/')))?)?;
        Self::load_sharded(dev, dir, file, Shard::single(), 0, cfg.vocab)
    }

    /// Work item B: rank `shard.rank` of `shard.world` loads its heads (q and kv heads split evenly, GQA groups
    /// whole), its MLP intermediate slice, its fc output rows and the Markov `w2` rows of the vocabulary slice
    /// [v0, v0 + vl) (= the target model's lm_head slice on this rank). Norms, `w1`, the confidence head and the
    /// mask embedding are replicated. World 1 is the unsharded drafter.
    pub fn load_sharded(dev: &Arc<CudaDevice>, dir: &str, file: &str, shard: Shard, v0: usize, vl: usize) -> Result<GpuDrafter> {
        load_module(dev)?;
        let dir = dir.trim_end_matches('/');
        let cfg = DrafterConfig::from_json(&std::fs::read_to_string(format!("{dir}/config.json"))?)?;
        let (h, nh, kv, hd, inter) = (cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.inter);
        let r = cfg.markov_rank;
        let w = shard.world;
        ensure!(w >= 1 && nh % w == 0 && kv % w == 0 && (nh / w) % (kv / w) == 0, "drafter heads nh {nh} kv {kv} do not split over {w} ranks");
        ensure!(inter % (128 * w) == 0 && h % (128 * w) == 0, "drafter inter {inter} / H {h} do not split over {w} ranks in 128-blocks");
        ensure!(v0 + vl <= cfg.vocab && vl % 8 == 0 && (w == 1 || vl < cfg.vocab), "drafter vocab slice [{v0}, {v0}+{vl}) of {}", cfg.vocab);
        let (nh_l, kv_l, inter_l, h_l) = (nh / w, kv / w, inter / w, h / w);
        let (q0, kv0) = (shard.rank * nh_l, shard.rank * kv_l);
        // row-major slices of a [rows][cols] f16 matrix
        let rows = |v: &[u16], cols: usize, r0: usize, n: usize| -> Vec<u16> { v[r0 * cols..(r0 + n) * cols].to_vec() };
        let cols = |v: &[u16], nrows: usize, ncols: usize, c0: usize, n: usize| -> Vec<u16> {
            let mut o = Vec::with_capacity(nrows * n);
            for rr in 0..nrows { o.extend_from_slice(&v[rr * ncols + c0..rr * ncols + c0 + n]); }
            o
        };
        // kernel geometry limits (the real drafter: H 4096, hd 128, r 256, window 1024)
        ensure!(hd % 32 == 0 && hd <= 128 && nh % kv == 0 && nh < 256 && kv < 256, "head geometry nh {nh} kv {kv} hd {hd}");
        ensure!(h % 128 == 0 && inter % 128 == 0 && (nh * hd) % 128 == 0, "H / inter / nh*hd must be multiples of 128");
        ensure!(r % 8 == 0 && r <= 256 && 32 % (r / 8) == 0, "markov rank {r}: the scan splits it over r/8 lanes of a warp");
        ensure!(cfg.vocab % 8 == 0, "vocab % 8");
        ensure!((1..=1 << 16).contains(&cfg.sliding_window), "sliding window {} outside 1..=65536", cfg.sliding_window);
        let ring = cfg.sliding_window.next_power_of_two();
        let bytes = std::fs::read(format!("{dir}/{file}")).with_context(|| format!("read {dir}/{file}"))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)?;
        let mut stats = LoadStats::default();
        let up16 = |v: Vec<u16>| -> Result<CudaSlice<u16>> { Ok(dev.htod_sync_copy(&v)?) };
        let up32 = |v: Vec<f32>| -> Result<CudaSlice<f32>> { Ok(dev.htod_sync_copy(&v)?) };
        let shape = |name: &str, want: &[usize]| -> Result<()> {
            let t = st.tensor(name).with_context(|| format!("drafter tensor {name} missing"))?;
            ensure!(t.shape() == want, "{name}: shape {:?}, want {want:?}", t.shape());
            Ok(())
        };
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("layers.{l}.");
            shape(&format!("{p}self_attn.q_proj.weight"), &[nh * hd, h])?;
            shape(&format!("{p}self_attn.k_proj.weight"), &[kv * hd, h])?;
            shape(&format!("{p}self_attn.v_proj.weight"), &[kv * hd, h])?;
            shape(&format!("{p}self_attn.o_proj.weight"), &[h, nh * hd])?;
            shape(&format!("{p}mlp.gate_proj.weight"), &[inter, h])?;
            shape(&format!("{p}mlp.down_proj.weight"), &[h, inter])?;
            let qf = to_f16(&st, &format!("{p}self_attn.q_proj.weight"), &mut stats)?;
            let kf = to_f16(&st, &format!("{p}self_attn.k_proj.weight"), &mut stats)?;
            let vf = to_f16(&st, &format!("{p}self_attn.v_proj.weight"), &mut stats)?;
            let mut qkv = rows(&qf, h, q0 * hd, nh_l * hd);
            qkv.extend(rows(&kf, h, kv0 * hd, kv_l * hd));
            qkv.extend(rows(&vf, h, kv0 * hd, kv_l * hd));
            let gf = to_f16(&st, &format!("{p}mlp.gate_proj.weight"), &mut stats)?;
            let uf = to_f16(&st, &format!("{p}mlp.up_proj.weight"), &mut stats)?;
            let mut gate_up = rows(&gf, h, shard.rank * inter_l, inter_l);
            gate_up.extend(rows(&uf, h, shard.rank * inter_l, inter_l));
            let of = to_f16(&st, &format!("{p}self_attn.o_proj.weight"), &mut stats)?;
            let df = to_f16(&st, &format!("{p}mlp.down_proj.weight"), &mut stats)?;
            layers.push(GpuDrafterLayer {
                in_norm: up32(to_f32(&st, &format!("{p}input_layernorm.weight"))?)?,
                post_norm: up32(to_f32(&st, &format!("{p}post_attention_layernorm.weight"))?)?,
                qkv: up16(qkv)?,
                o: up16(cols(&of, h, nh * hd, q0 * hd, nh_l * hd))?,
                gate_up: up16(gate_up)?,
                down: up16(cols(&df, h, inter, shard.rank * inter_l, inter_l))?,
                q_norm: up32(to_f32(&st, &format!("{p}self_attn.q_norm.weight"))?)?,
                k_norm: up32(to_f32(&st, &format!("{p}self_attn.k_norm.weight"))?)?,
                kc: dev.alloc_zeros::<u16>(ring * kv_l * hd)?,
                vc: dev.alloc_zeros::<u16>(ring * kv_l * hd)?,
            });
        }
        shape("fc.weight", &[h, cfg.taps.len() * h])?;
        shape("markov_head.markov_w1.weight", &[cfg.vocab, r])?;
        shape("markov_head.markov_w2.weight", &[cfg.vocab, r])?;
        let (conf_w, conf_b) = if cfg.confidence {
            let w = to_f32(&st, "confidence_head.proj.weight")?;
            ensure!(w.len() == h + r, "confidence proj width {} != H + r", w.len());
            (Some(up32(w)?), to_f32(&st, "confidence_head.proj.bias")?[0])
        } else { (None, 0.0) };
        let fcf = to_f16(&st, "fc.weight", &mut stats)?;
        let w2f = to_f16(&st, "markov_head.markov_w2.weight", &mut stats)?;
        let xch_floats = (APPEND_ROWS * h).max(MAX_BLOCK * h).max(2 * w);
        let d = GpuDrafter {
            dev: dev.clone(),
            shard, nh_l, kv_l, inter_l, h_l, v0, vl,
            xch: dev.alloc_zeros::<f32>(xch_floats)?,
            fc: up16(rows(&fcf, cfg.taps.len() * h, shard.rank * h_l, h_l))?,
            hidden_norm: up32(to_f32(&st, "hidden_norm.weight")?)?,
            mask_embedding: up32(to_f32(&st, "mask_embedding")?)?,
            layers,
            norm: up32(to_f32(&st, "norm.weight")?)?,
            markov_w1: up16(to_f16(&st, "markov_head.markov_w1.weight", &mut stats)?)?,
            markov_w2: up16(rows(&w2f, r, v0, vl))?,
            conf_w, conf_b, ring, stats, cfg,
        };
        dev.synchronize()?;
        Ok(d)
    }

    fn l<P>(&self, s: &CudaStream, name: &str, cfg: LaunchConfig, params: P) -> Result<()>
    where cudarc::driver::CudaFunction: LaunchAsync<P> {
        let f = self.dev.get_func(MODULE, name).with_context(|| format!("{name} missing from {MODULE}"))?;
        unsafe { f.launch_on_stream(s, cfg, params) }.with_context(|| format!("launch {name}"))
    }

    /// part f32 [ks][m][n] = x f16 [m][k] . w f16 [n][k]^T; returns (part, ks). Split-K so a launch
    /// has ~4096 warps (the GB10 needs many loads in flight to approach its ~225 GB/s ceiling).
    fn gemm(&self, s: &CudaStream, ws: &mut Ws, w: u64, x: u64, m: usize, k: usize, n: usize) -> Result<(u64, usize)> {
        ensure!((1..=APPEND_ROWS).contains(&m) && n % 8 == 0 && k % 128 == 0, "gemm m {m} k {k} n {n}");
        let tiles = n / 8;
        let mut ks = 1usize;
        while ks * 2 * tiles <= 4096 && k % (ks * 2 * 128) == 0 { ks *= 2; }
        let mt = (m + 15) / 16;
        let name = match mt { 1 => "nd_gemm_m1", 2 => "nd_gemm_m2", 3 | 4 => "nd_gemm_m4", _ => "nd_gemm_m8" };
        let part = ws.alloc::<f32>(ks * m * n)?;
        let cfg = LaunchConfig { grid_dim: (((tiles + WPB - 1) / WPB) as u32, ks as u32, 1), block_dim: ((WPB * 32) as u32, 1, 1), shared_mem_bytes: 0 };
        self.l(s, name, cfg, (w, x, part, m as i32, k as i32, n as i32, (k / ks) as i32))?;
        Ok((part, ks))
    }

    /// GEMM + combine: out32 (= or += when `add`) and/or out16 (0 = absent).
    #[allow(clippy::too_many_arguments)]
    fn linear(&self, s: &CudaStream, ws: &mut Ws, w: u64, x: u64, m: usize, k: usize, n: usize, out32: u64, out16: u64, add: bool) -> Result<()> {
        let mark = ws.mark();
        let (part, ks) = self.gemm(s, ws, w, x, m, k, n)?;
        self.l(s, "nd_combine", grid1(m * n), (part, ks as i32, (m * n) as i64, out32, out16, add as i32))?;
        ws.release(mark);
        Ok(())
    }

    /// Committed target positions `pos0 .. pos0 + rows` -> every layer's context K/V ring. `taps[j]`
    /// = (device address of f32 rows, row pitch in floats) of the residual after target layer
    /// `cfg.taps[j]` (`hidden_states[l + 1]`, i.e. `GpuModel::step_kernels`' tap `l`), in config order
    /// (the fc input is their concatenation in that order). Only the last `ring` rows can ever be
    /// attended, so earlier rows of a long call (a prefill) are skipped.
    pub fn append_context(&self, s: &CudaStream, ws: &mut Ws, taps: &[(u64, usize)], rows: usize, pos0: usize, tp: Option<&NaiveTp>) -> Result<()> {
        let c = &self.cfg;
        let (h, kv, hd) = (c.hidden, self.kv_l, c.head_dim);
        ensure!((self.shard.world == 1) == tp.is_none() || self.shard.world == 1, "sharded drafter without a transport");
        let nt = c.taps.len();
        ensure!(taps.len() == nt, "append_context: {} taps, the drafter takes {nt}", taps.len());
        let skip = rows.saturating_sub(self.ring);
        let mut r0 = skip;
        while r0 < rows {
            let m = (rows - r0).min(APPEND_ROWS);
            let mark = ws.mark();
            // the fc input: the taps side by side, f16 [m][nt*H] (rounding point)
            let xin = ws.alloc::<u16>(m * nt * h)?;
            for (j, &(p, pitch)) in taps.iter().enumerate() {
                self.l(s, "nd_tap_f16", rows_cfg(m, 256), (xin, p + (r0 * pitch * 4) as u64, pitch as i64, (nt * h) as i32, (j * h) as i32, h as i32))?;
            }
            let ctx32 = if let Some(tp) = tp.filter(|_| self.shard.world > 1) {
                // this rank's h_l output rows of fc -> its column block of the exchange buffer, gathered
                let part = ws.alloc::<f32>(m * self.h_l)?;
                self.linear(s, ws, dptr(&self.fc), xin, m, nt * h, self.h_l, part, 0, false)?;
                self.l(s, "nd_scatter_cols", rows_cfg(m, 256), (&self.xch, part, m as i32, h as i32, self.h_l as i32, (self.shard.rank * self.h_l) as i32))?;
                tp.all_reduce_at(s, &self.xch, 0, m * h)?;
                dptr(&self.xch)
            } else {
                let ctx32 = ws.alloc::<f32>(m * h)?;
                self.linear(s, ws, dptr(&self.fc), xin, m, nt * h, h, ctx32, 0, false)?;
                ctx32
            };
            let ctx16 = ws.alloc::<u16>(m * h)?;
            self.l(s, "nd_rmsnorm", rows_cfg(m, 256), (ctx16, 0u64, ctx32, &self.hidden_norm, h as i32, c.eps))?;
            let raw = ws.alloc::<f32>(m * 2 * kv * hd)?;
            for l in &self.layers {
                // the k|v tail of the fused qkv matrix (this rank's heads)
                let wkv = dptr(&l.qkv) + (self.nh_l * hd * h * 2) as u64;
                self.linear(s, ws, wkv, ctx16, m, h, 2 * kv * hd, raw, 0, false)?;
                self.l(s, "nd_ctx_kv", rows_cfg(m * kv, 64),
                       (&l.kc, &l.vc, raw, &l.k_norm, (pos0 + r0) as i32, kv as i32, hd as i32, self.ring as i32, c.eps, c.rope_theta))?;
            }
            ws.release(mark);
            r0 += m;
        }
        Ok(())
    }

    /// The block through the backbone: anchor (target embedding row, `embed` = bf16 bits [V][H]) at
    /// position `t`, then `n - 1` mask slots; context = positions [max(0, t+i-W+1), t) of the ring
    /// (they must have been appended). Allocates the round's outputs in `ws` (released by the caller).
    pub fn backbone(&self, s: &CudaStream, ws: &mut Ws, t: usize, anchor: u32, n: usize, embed: u64, tp: Option<&NaiveTp>) -> Result<DraftBufs> {
        let c = &self.cfg;
        let (h, nh, kv, hd, inter) = (c.hidden, self.nh_l, self.kv_l, c.head_dim, self.inter_l);
        let tp = tp.filter(|_| self.shard.world > 1);
        ensure!(self.shard.world == 1 || tp.is_some(), "sharded drafter without a transport");
        ensure!((1..=MAX_BLOCK).contains(&n), "block of {n} rows (1..={MAX_BLOCK})");
        ensure!((anchor as usize) < c.vocab, "anchor {anchor}");
        ensure!(t + n < i32::MAX as usize, "position");
        let tok = ws.alloc::<i32>(n + 1)?;
        let conf = ws.alloc::<f32>(n)?;
        let hidden16 = ws.alloc::<u16>(n * h)?;
        let hidden32 = ws.alloc::<f32>(n * h)?;
        let x = ws.alloc::<f32>(n * h)?;
        let hb = ws.alloc::<u16>(n * h)?;
        self.l(s, "nd_block_init", rows_cfg(n, 256), (x, embed, &self.mask_embedding, tok, anchor as i32, h as i32))?;
        let per = nh + 2 * kv;
        let geo = (nh | kv << 8 | hd << 16) as i32;
        // nd_attn_part: one CTA per (kv group, chunk of ATTN_CHUNK keys of the block's key list)
        let nctx0 = t - (t + 1).saturating_sub(c.sliding_window);
        let nsplit = (nctx0 + n + ATTN_CHUNK - 1) / ATTN_CHUNK;
        let attn_smem = (2 * ATTN_CHUNK * (hd + 8) * 2 + n * (nh / kv) * ATTN_CHUNK * 4) as u32;
        ensure!(attn_smem <= 48 * 1024, "attention smem {attn_smem} B");
        for l in &self.layers {
            let mark = ws.mark();
            self.l(s, "nd_rmsnorm", rows_cfg(n, 256), (hb, 0u64, x, &l.in_norm, h as i32, c.eps))?;
            let raw = ws.alloc::<f32>(n * per * hd)?;
            self.linear(s, ws, dptr(&l.qkv), hb, n, h, per * hd, raw, 0, false)?;
            let q16 = ws.alloc::<u16>(n * nh * hd)?;
            let kb16 = ws.alloc::<u16>(n * kv * hd)?;
            let vb16 = ws.alloc::<u16>(n * kv * hd)?;
            self.l(s, "nd_block_qkv", rows_cfg(n * per, 64),
                   (q16, kb16, vb16, raw, &l.q_norm, &l.k_norm, t as i32, nh as i32, kv as i32, hd as i32, c.eps, c.rope_theta))?;
            let att = ws.alloc::<u16>(n * nh * hd)?;
            let apart = ws.alloc::<f32>(n * nh * nsplit * (hd + 2))?;
            self.l(s, "nd_attn_part", LaunchConfig { grid_dim: (kv as u32, nsplit as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: attn_smem },
                   (apart, q16, &l.kc, &l.vc, kb16, vb16, t as i32, n as i32, geo, self.ring as i32, c.sliding_window as i32, (hd as f32).powf(-0.5)))?;
            self.l(s, "nd_attn_combine", rows_cfg(n * nh, hd as u32), (att, apart, (n * nh) as i32, nsplit as i32, hd as i32))?;
            if let Some(tp) = tp {
                // partial over this rank's heads -> summed over ranks -> added to the (replicated) residual
                self.linear(s, ws, dptr(&l.o), att, n, nh * hd, h, dptr(&self.xch), 0, false)?;
                tp.all_reduce_at(s, &self.xch, 0, n * h)?;
                self.l(s, "nd_combine", grid1(n * h), (&self.xch, 1i32, (n * h) as i64, x, 0u64, 1i32))?;
            } else {
                self.linear(s, ws, dptr(&l.o), att, n, nh * hd, h, x, 0, true)?;
            }
            self.l(s, "nd_rmsnorm", rows_cfg(n, 256), (hb, 0u64, x, &l.post_norm, h as i32, c.eps))?;
            let act = ws.alloc::<u16>(n * inter)?;
            {
                let m2 = ws.mark();
                let (part, ks) = self.gemm(s, ws, dptr(&l.gate_up), hb, n, h, 2 * inter)?;
                self.l(s, "nd_combine_silu", grid1(n * inter), (part, ks as i32, n as i32, inter as i32, act))?;
                ws.release(m2);
            }
            if let Some(tp) = tp {
                self.linear(s, ws, dptr(&l.down), act, n, inter, h, dptr(&self.xch), 0, false)?;
                tp.all_reduce_at(s, &self.xch, 0, n * h)?;
                self.l(s, "nd_combine", grid1(n * h), (&self.xch, 1i32, (n * h) as i64, x, 0u64, 1i32))?;
            } else {
                self.linear(s, ws, dptr(&l.down), act, n, inter, h, x, 0, true)?;
            }
            ws.release(mark);
        }
        self.l(s, "nd_rmsnorm", rows_cfg(n, 256), (hidden16, hidden32, x, &self.norm, h as i32, c.eps))?;
        Ok(DraftBufs { n, tok, conf, hidden16, hidden32 })
    }

    /// Sequential Markov correction on the base logits (f16 [n][V], full vocabulary), the pick (greedy
    /// argmax, or one draw from q when `smp` is given) and the confidence, row by row on the device:
    /// writes `b.tok[1..=n]` and `b.conf` (plus the caller's `qs`/`qstat` in sampled mode).
    pub fn markov(&self, s: &CudaStream, ws: &mut Ws, b: &DraftBufs, logits16: u64, smp: Option<SmpDraft>, tp: Option<&NaiveTp>) -> Result<()> {
        let c = &self.cfg;
        let mark = ws.mark();
        let pv = ws.alloc::<f32>(MARKOV_BLOCKS)?;
        let pi = ws.alloc::<i32>(MARKOV_BLOCKS)?;
        let cw = self.conf_w.as_ref().map(dptr).unwrap_or(0);
        let tp = tp.filter(|_| self.shard.world > 1);
        ensure!(self.shard.world == 1 || tp.is_some(), "sharded drafter without a transport");
        ensure!(smp.is_none() || tp.is_none(), "sampled drafts (--spec-draft-temp > 0) need the head-only drafter (--spec-drafter head)");
        let (smb, qstat, temp, rng) = match smp { Some(x) => (x.smb, x.qstat, x.t, x.rng), None => (0u64, 0u64, 0f32, 0u64) };
        let vv: i64 = ((self.v0 as i64) << 32) | self.vl as i64;
        let w = self.shard.world;
        for i in 0..b.n {
            // sampled mode: the scan writes q row i (scaled by 1/T) plus the per-block softmax
            // partials into `smb`, draws d_i ~ q by Gumbel-max, and qnorm folds the partials into
            // qstat[i] = (max, Z). Greedy (temp = 0, smb null) runs the identical greedy body.
            self.l(s, "nd_markov_scan", rows_cfg(MARKOV_BLOCKS, 256),
                   (pv, pi, smb, logits16, &self.markov_w1, &self.markov_w2, b.tok,
                    i as i32, vv, c.markov_rank as i32, temp, rng ^ ((i as u64 + 1) << 40)))?;
            let nb = if let Some(tp) = tp {
                // every rank's best (score, global id) in its slot; the gathered slots are the pick's partials
                self.l(s, "nd_markov_fold", rows_cfg(1, 256), (pv, pi, MARKOV_BLOCKS as i32, &self.xch, self.shard.rank as i32, w as i32))?;
                tp.all_reduce_at(s, &self.xch, 0, 2 * w)?;
                self.l(s, "nd_markov_unpack", rows_cfg(1, 32), (&self.xch, w as i32, pv, pi))?;
                w
            } else { MARKOV_BLOCKS };
            self.l(s, "nd_markov_pick", rows_cfg(1, 256),
                   (b.tok, b.conf, pv, pi, nb as i32, b.hidden32, &self.markov_w1, cw, self.conf_b,
                    i as i32, c.hidden as i32, c.markov_rank as i32))?;
            if smp.is_some() {
                self.l(s, "nd_markov_qnorm", rows_cfg(1, 32), (smb, MARKOV_BLOCKS as i32, i as i32, c.vocab as i32, qstat))?;
            }
        }
        ws.release(mark);
        Ok(())
    }

    /// One round, mirroring `CpuDrafter::round` (context already appended up to `t`): backbone, the
    /// caller's lm_head (`lm(ws, hidden16, n)` -> base logits f16 [n][V] in `ws`), Markov +
    /// confidence; then one stream sync and the results to the host. `n` = block rows (the config's
    /// block size in serving).
    pub fn round(&self, s: &CudaStream, ws: &mut Ws, t: usize, anchor: u32, n: usize, embed: u64,
                 lm: &mut dyn FnMut(&mut Ws, u64, usize) -> Result<u64>, smp: Option<SmpDraft>, tp: Option<&NaiveTp>) -> Result<DraftRound> {
        let b = self.backbone(s, ws, t, anchor, n, embed, tp)?;
        let logits = lm(ws, b.hidden16, n)?;
        self.markov(s, ws, &b, logits, smp, tp)?;
        unsafe { cudarc::driver::result::stream::synchronize(s.stream) }.context("stream synchronize")?;
        let tok = dtoh::<i32>(b.tok, n + 1)?;
        let (qmax, qz) = match smp {
            Some(x) => {
                let v = dtoh::<f32>(x.qstat, 2 * n)?;
                (v.iter().step_by(2).copied().collect(), v.iter().skip(1).step_by(2).copied().collect())
            }
            None => (Vec::new(), Vec::new()),
        };
        Ok(DraftRound {
            tokens: tok[1..].iter().map(|&x| x as u32).collect(),
            confidence: dtoh::<f32>(b.conf, n)?,
            hidden: dtoh::<f32>(b.hidden32, n * self.cfg.hidden)?,
            qmax,
            qz,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------------------------

fn f16r(x: f32) -> f32 { half::f16::from_f32(x).to_f32() }
fn bf16r(x: f32) -> f32 { half::bf16::from_f32(x).to_f32() }

/// The CPU twin of what the GPU holds: every matrix rounded through f16 (norms, mask embedding and
/// confidence head stay f32 on both sides), so the gate compares the kernels, not the load rounding.
fn round_like_gpu(d: &mut CpuDrafter) {
    let r = |v: &mut Vec<f32>| v.iter_mut().for_each(|x| *x = f16r(*x));
    r(&mut d.fc.w);
    r(&mut d.markov_w1);
    r(&mut d.markov_w2.w);
    for l in &mut d.layers {
        for m in [&mut l.q, &mut l.k, &mut l.v, &mut l.o, &mut l.gate, &mut l.up, &mut l.down] { r(&mut m.w); }
    }
}

/// Deterministic pseudo-random f32 in [-a, a) (splitmix64): the synthetic long context.
fn synth(n: usize, seed: u64, a: f32) -> Vec<f32> {
    let mut z = seed;
    (0..n).map(|_| {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        ((x >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * a
    }).collect()
}

/// One fixture comparison: GPU round vs the f16-twin CPU round on `th` ([T][taps*H]) appended in
/// `chunks` pieces. Returns (pass, report line).
#[allow(clippy::too_many_arguments)]
fn gate_case(label: &str, g: &GpuDrafter, cpu: &CpuDrafter, s: &CudaStream, ws: &mut Ws, th: &[f32], t: usize, anchor: u32,
             embed_bf16: &CudaSlice<u16>, embed_cpu: &[f32], lm16: &CudaSlice<u16>, lm_cpu: &super::cpu::Mat, chunks: &[usize]) -> Result<(bool, String)> {
    let c = &g.cfg;
    let (h, nt) = (c.hidden, c.taps.len());
    let th_d = g.dev.htod_sync_copy(th)?;
    ws.reset();
    let mut p0 = 0;
    for &rows in chunks {
        let base = dptr(&th_d) + (p0 * nt * h * 4) as u64;
        let taps: Vec<(u64, usize)> = (0..nt).map(|j| (base + (j * h * 4) as u64, nt * h)).collect();
        g.append_context(s, ws, &taps, rows, p0, None)?;
        p0 += rows;
    }
    ensure!(p0 == t, "chunks cover {p0} rows, T = {t}");
    let n = c.block;
    let lm_ptr = dptr(lm16);
    let mut lm = |ws: &mut Ws, x16: u64, m: usize| -> Result<u64> {
        let out = ws.alloc::<u16>(m * c.vocab)?;
        g.linear(s, ws, lm_ptr, x16, m, c.hidden, c.vocab, 0, out, false)?;
        Ok(out)
    };
    let gr = g.round(s, ws, t, anchor, n, dptr(embed_bf16), &mut lm, None, None)?;
    let cr = cpu.round(th, t, anchor, embed_cpu, lm_cpu)?;
    let e_h = super::cpu::rel_l2(&gr.hidden, &cr.hidden);
    let dc = gr.confidence.iter().zip(&cr.confidence).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let ok = gr.tokens == cr.tokens && dc < 1e-2;
    // the CPU's top-2 margin per row says how close a token flip would have been
    let v = c.vocab;
    let margins: Vec<String> = (0..n).map(|i| {
        let row = &cr.corrected[i * v..(i + 1) * v];
        let mut a = row.to_vec();
        a.sort_by(|x, y| y.partial_cmp(x).unwrap());
        format!("{:.3}", a[0] - a[1])
    }).collect();
    Ok((ok, format!("  {label}: T {t} anchor {anchor}: tokens gpu {:?} cpu {:?} | conf max|d| {dc:.2e} | hidden rel-L2 {e_h:.2e} | cpu top-2 margins [{}] -> {}",
                    gr.tokens, cr.tokens, margins.join(" "), if ok { "ok" } else { "MISMATCH" })))
}

/// `--probe-naive-drafter --model-dir <pack> [--draft-dir <dir>]`: the GPU round vs the CPU round on
/// the HF-oracle fixture (+ a long synthetic context: ring wrap, binding window, chunked append),
/// then 20 timed rounds of the real drafter with the pack's embedding and EXL3 lm_head.
pub fn probe_drafter(model_dir: &str, draft_dir: &str) -> Result<()> {
    use crate::dsv4_load::{read_npz_key, NpyData};
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    let s = dev.fork_default_stream().context("fork stream")?;

    // ---- 1. the gate on the fixture ----
    let fx = "tests/fixtures/naive/drafter";
    let npz = std::path::Path::new(fx).join("oracle.npz");
    ensure!(npz.is_file() && std::path::Path::new(fx).join("drafter.safetensors").is_file(),
            "fixture missing under {fx} (scripts/naive/drafter_oracle.py)");
    let g = GpuDrafter::load(&dev, fx, "drafter.safetensors")?;
    let cpu_f32 = CpuDrafter::load_plain(fx, "drafter.safetensors")?;
    let mut cpu = cpu_f32.clone();
    round_like_gpu(&mut cpu);
    let c = g.cfg.clone();
    let bytes = std::fs::read(format!("{fx}/drafter.safetensors"))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    // target embedding: bf16 on the device (as the pack ships it), the CPU twin sees the same values
    let embed: Vec<f32> = to_f32(&st, "target.embed_tokens.weight")?.into_iter().map(bf16r).collect();
    let embed_d = dev.htod_sync_copy(&embed.iter().map(|x| half::bf16::from_f32(*x).to_bits()).collect::<Vec<u16>>())?;
    let lm: Vec<f32> = to_f32(&st, "target.lm_head.weight")?.into_iter().map(f16r).collect();
    let lm_d = dev.htod_sync_copy(&lm.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect::<Vec<u16>>())?;
    let lm_cpu = super::cpu::Mat::new(c.vocab, c.hidden, lm)?;
    let g_npz = |k: &str| -> Result<Vec<f32>> { Ok(match read_npz_key(&npz, k)?.1 { NpyData::F32(v) => v, NpyData::I64(v) => v.iter().map(|x| *x as f32).collect() }) };
    let th = g_npz("target_hidden")?;
    let t = th.len() / (c.taps.len() * c.hidden);
    let anchor = g_npz("anchor")?[0] as u32;
    let oracle_tokens: Vec<u32> = g_npz("tokens")?.iter().map(|x| *x as u32).collect();
    let cr32 = cpu_f32.round(&th, t, anchor, &to_f32(&st, "target.embed_tokens.weight")?,
                             &super::cpu::Mat::new(c.vocab, c.hidden, to_f32(&st, "target.lm_head.weight")?)?)?;
    let mut ws = Ws::new(&dev, 64 << 20)?;
    println!("NAIVE drafter gate (fixture {fx}: H {} layers {} heads {}/{} hd {} V {} r {} block {} window {}, ring {})",
             c.hidden, c.layers, c.heads, c.kv_heads, c.head_dim, c.vocab, c.markov_rank, c.block, c.sliding_window, g.ring);
    println!("  HF oracle tokens {oracle_tokens:?}; f32 CPU reference {:?} (the f16-weight twin below is the gate's reference)", cr32.tokens);
    let mut all_ok = true;
    let (ok, line) = gate_case("oracle", &g, &cpu, &s, &mut ws, &th, t, anchor, &embed_d, &embed, &lm_d, &lm_cpu, &[t])?;
    println!("{line}"); all_ok &= ok;
    // long synthetic context: T > window (the mask binds for every block row), positions wrap the
    // ring, appended in uneven chunks (the 128-row append tiling + a partial tail)
    let tl = 1100usize;
    let thl = synth(tl * c.taps.len() * c.hidden, 0x5EED, 2.0);
    let (ok, line) = gate_case("long", &g, &cpu, &s, &mut ws, &thl, tl, 77, &embed_d, &embed, &lm_d, &lm_cpu, &[1000, 37, 63])?;
    println!("{line}"); all_ok &= ok;
    // one big append (a prefill): rows before the last `ring` are skipped
    let tb = 1300usize;
    let thb = synth(tb * c.taps.len() * c.hidden, 0xB16, 2.0);
    let (ok, line) = gate_case("prefill", &g, &cpu, &s, &mut ws, &thb, tb, 300, &embed_d, &embed, &lm_d, &lm_cpu, &[tb])?;
    println!("{line}"); all_ok &= ok;
    if !all_ok {
        println!("NAIVE_DRAFTER_FAIL");
        bail!("GPU drafter round differs from the CPU reference");
    }
    println!("NAIVE_DRAFTER_OK");
    drop(g);

    // ---- 2. the real drafter: 20 timed rounds ----
    let pack = super::pack::NaivePack::open(model_dir)?;
    let ring_t = (2 * pack.cfg.sliding_window).next_power_of_two();
    let t0 = std::time::Instant::now();
    let model = super::model::GpuModel::load_slice_on(dev.clone(), &pack, 0, 4096, ring_t, 1, super::shard::Shard::single())?;
    let g = GpuDrafter::load(&dev, draft_dir, "model.safetensors")?;
    let c = g.cfg.clone();
    ensure!(c.vocab >= model.vl && c.hidden == model.cfg.hidden_size, "drafter / target geometry mismatch");
    println!("real drafter {draft_dir}: {} layers H {} V {} r {} block {} window {}; loaded with the pack's embed + lm_head in {:.1} s; \
              bf16->f16: {} values, {} subnormal ({:.2e}), max |w| {:.3}",
             c.layers, c.hidden, c.vocab, c.markov_rank, c.block, c.sliding_window, t0.elapsed().as_secs_f64(),
             g.stats.values, g.stats.subnormal, g.stats.subnormal as f64 / g.stats.values.max(1) as f64, g.stats.max_abs);
    let s = &model.stream;
    let mut ws = Ws::new(&dev, 512 << 20)?;
    // a full window of synthetic committed context (the timing does not depend on the values)
    let t = c.sliding_window - 1;
    let thr = synth(t * c.taps.len() * c.hidden, 0xD5, 1.0);
    let thr_d = dev.htod_sync_copy(&thr)?;
    let (h, nt) = (c.hidden, c.taps.len());
    let taps: Vec<(u64, usize)> = (0..nt).map(|j| (dptr(&thr_d) + (j * h * 4) as u64, nt * h)).collect();
    ws.reset();
    g.append_context(s, &mut ws, &taps, t, 0, None)?;
    model.sync()?;
    let ta = std::time::Instant::now();
    ws.reset();
    g.append_context(s, &mut ws, &taps, t, 0, None)?;
    model.sync()?;
    let append_ms = ta.elapsed().as_secs_f64() * 1e3;
    // one committed row per step (what a decode step appends at acceptance 1)
    let ta = std::time::Instant::now();
    for _ in 0..10 { ws.reset(); g.append_context(s, &mut ws, &taps[..], 1, t - 1, None)?; }
    model.sync()?;
    let append1_ms = ta.elapsed().as_secs_f64() * 1e3 / 10.0;
    let n = c.block;
    ensure!(model.vl == c.vocab, "the timing needs the whole-vocabulary head (world 1): vl {} V {}", model.vl, c.vocab);
    let mut lm = |ws: &mut Ws, x16: u64, m: usize| -> Result<u64> { model.chain_ws(&model.lm_head, x16, m, ws) };
    let anchor = 1000u32;
    for _ in 0..3 { ws.reset(); g.round(s, &mut ws, t, anchor, n, dptr(&model.embed), &mut lm, None, None)?; }
    let iters = 20;
    let tr = std::time::Instant::now();
    let mut last = None;
    for _ in 0..iters { ws.reset(); last = Some(g.round(s, &mut ws, t, anchor, n, dptr(&model.embed), &mut lm, None, None)?); }
    let round_ms = tr.elapsed().as_secs_f64() * 1e3 / iters as f64;
    // phase split (each phase synced; sums to slightly more than a round)
    let sync = || model.sync();
    let (mut tb_ms, mut tl_ms, mut tm_ms) = (0f64, 0f64, 0f64);
    for _ in 0..iters {
        ws.reset();
        let a = std::time::Instant::now();
        let b = g.backbone(s, &mut ws, t, anchor, n, dptr(&model.embed), None)?; sync()?;
        let bb = std::time::Instant::now();
        let lg = lm(&mut ws, b.hidden16, n)?; sync()?;
        let cc = std::time::Instant::now();
        g.markov(s, &mut ws, &b, lg, None, None)?; sync()?;
        let dd = std::time::Instant::now();
        tb_ms += (bb - a).as_secs_f64(); tl_ms += (cc - bb).as_secs_f64(); tm_ms += (dd - cc).as_secs_f64();
    }
    let k = 1e3 / iters as f64;
    let r = last.unwrap();
    println!("  drafts {:?} conf {:?} (synthetic context: values meaningless)", r.tokens,
             r.confidence.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    println!("  context append: {append_ms:.2} ms for {t} rows (prefill), {append1_ms:.3} ms per 1-row step");
    println!("  phases: backbone {:.3} ms, lm_head (EXL3 chain) {:.3} ms, markov+confidence {:.3} ms", tb_ms * k, tl_ms * k, tm_ms * k);
    println!("NAIVE_DRAFTER_ROUND_MS {round_ms:.3} (block {n}, T {t}, {iters} rounds, incl. one sync + readback)");
    Ok(())
}
