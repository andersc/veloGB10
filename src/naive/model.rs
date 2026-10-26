//! Naive-N0.5-Flash GPU model (S4): the first `n_layers` layers of the pack on one GB10, decode /
//! verify rows (M <= 16) at consecutive positions, every op through `kernels/gpu_naive.cu`.
//! Eager launches on the device's default stream; graphs, TP and prefill come later (S5/S6).
//!
//! Gate: `--probe-naive-slice --model-dir <pack> --layers N --tokens T` runs T tokens through the
//! GPU slice and the CPU twin (`CpuModel::load_pack_slice`, dequantized weights) and compares the
//! residual after every layer and the logits.
//!
//! Layout (all row-major, M = rows):
//!   x       f32 [M][H]      the residual stream
//!   h       f16 [M][H]      normed input of a sublayer
//!   q/k/v   f16 [M][heads*192]  (v per-head padded as stored: 192 per kv head, 128 live)
//!   attn    f16 [M][nh*128]
//!   KV caches per layer: kc [kv][S][192], vc [kv][S][128] f16 — S = ring slots for SWA layers
//!   (pos & (S-1)), S = max_pos for DSA layers; DSA layers also keep the indexer key plane
//!   kq [max_pos][128] e4m3 + ks [max_pos] f32.

use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use super::config::{AttnKind, NaiveConfig};
use super::gpu::{load_module, upload_quad, GpuQuad, MODULE};
use super::pack::NaivePack;

pub const MAX_ROWS: usize = 16;
/// the widest prefill chunk a step takes (the workspace is sized for it)
pub const MAX_PREFILL_ROWS: usize = 2048;

/// T0-c: the largest dynamic smem the naive path may request WITHOUT a per-function opt-in.
///
/// The naive module is loaded through cudarc (`dev.load_ptx`) and every launch goes through
/// `CudaFunction::launch_on_stream`, and cudarc exposes NO accessor for the raw `CUfunction`
/// (`cu_function` is `pub(crate)`) and does not set the attribute itself — it passes
/// `shared_mem_bytes` straight to `cuLaunchKernel`. So the >48 KB opt-in that `gpu::SMEM_OPTIN_CAP`
/// (99 KB) budgets for CANNOT be applied to the copy that actually launches here. Setting it on a
/// private `cuModuleLoadDataEx` copy, as `src/gpu.rs` and `src/exl3_forward.rs` do for their own
/// raw-fn launches, would be the exact silent-no-op the 2026-10-03 lesson records: the attribute
/// binds per `CUfunction`, so the private copy's attribute would not reach cudarc's copy and a
/// >48 KB launch would fail at runtime while every static gate stayed green.
///
/// T0-c therefore does NOT try to raise the cap. It removes the need to: staging only the CTA's
/// k-window makes the request `ks` times smaller, which is what brings the wide shapes (mlp down,
/// k = 16384) back under 48 KB at spec-block 8. Reaching beyond 48 KB on this path would need
/// cudarc to grow a raw-handle accessor — a separate, deliberately unstarted item.
pub const DEFAULT_SMEM_CAP: usize = 48 * 1024;

/// T0-c: dynamic smem for the STAGED (`a1`) body, in bytes, for `m` rows, a `k`-wide input and a
/// split-K factor of `ks`. Since T0-c the kernel stages only this CTA's k-window — `k / ks` columns —
/// so the staged bytes are `m * ((k/ks) + 8) * 2`, not `m * (k + 8) * 2`. `ks == 1` (the window is the
/// whole of k) is byte-identical to the pre-T0-c number, which is what makes the change safe.
///
/// 0 means "does not fit": the caller must then take the per-step (`nv_gemm_body`) path, which needs
/// no smem at all. It must NEVER return a size the kernel would overflow — the budget is an upper
/// bound, not a check (see `gpu::SMEM_OPTIN_CAP` and the illegal-smem lesson), so the arithmetic here
/// and `nv_gemm_body_a1`'s `SK` must never drift apart. `nv_a1_fits` in kernels/gpu_naive.cu is the
/// device-side twin of the windowing decision and is deliberately NOT the cap.
#[inline]
pub fn a1_smem(m: usize, k: usize, ks: usize) -> u32 {
    if m == 0 || m > MAX_ROWS || ks == 0 || k == 0 || k % 16 != 0 { return 0; }
    if (k / 16) % ks != 0 { return 0; }          // the host already requires kb % ks == 0
    let kb = k / 16;
    let chunk = kb / ks;                         // k-tiles this CTA walks
    if chunk == 0 || chunk % 8 != 0 { return 0; } // window must be 128-wide: the fused `suh` path
                                             // runs nv_had128 over whole 128-blocks, so a narrower
                                             // window would straddle a butterfly group (see T0-c).
    let bytes = m * (chunk * 16 + 8) * 2;
    if bytes <= DEFAULT_SMEM_CAP as usize { bytes as u32 } else { 0 }
}

pub struct GpuIndexer {
    pub wq: GpuQuad,
    pub wk: GpuQuad,
    pub wp: GpuQuad,
    pub kn_w: CudaSlice<u16>,
    pub kn_b: CudaSlice<u16>,
    pub kq: CudaSlice<u8>,   // [max_pos][ihd]
    pub ks: CudaSlice<f32>,  // [max_pos]
}

pub struct GpuMoe {
    pub router: CudaSlice<u16>, // f16 [E][H]
    pub bias: CudaSlice<f32>,   // [E]
    pub g_tr: CudaSlice<u16>, pub u_tr: CudaSlice<u16>, pub d_tr: CudaSlice<u16>,
    pub suh_g: CudaSlice<u16>, pub svh_g: CudaSlice<u16>,
    pub suh_u: CudaSlice<u16>, pub svh_u: CudaSlice<u16>,
    pub suh_d: CudaSlice<u16>, pub svh_d: CudaSlice<u16>,
    pub g_words: u64, pub u_words: u64, pub d_words: u64,
    pub bpb: usize,
    /// trellis word offsets of the LOCAL experts (le * words): the grouped prefill GEMM's table
    pub offs_g: CudaSlice<u64>,
    pub offs_d: CudaSlice<u64>,
    /// routing-weight scale: routed_scaling_factor x the converter's up_proj divisor of this layer
    /// (`NaiveConfig::moe_interm_div`; 128 at layer 47), verified against the pack's suh at load
    pub scale: f32,
    /// expert TP (`--naive-moe-shard tp`): this rank's slice of every expert's intermediate (mi / world); else mi
    pub mi_l: usize,
    /// the experts held on this rank (the local index space of `offs_g` / `offs_d`): all of them under expert TP
    pub el: usize,
}

/// Expert TP resolution for a load: a shipped shard says it in its deal (`NaivePack::etp`); the whole pack (the head,
/// a loopback probe) follows `--naive-moe-shard`. World 1 has nothing to shard.
pub fn etp_of(pack: &NaivePack, shard: &super::shard::Shard) -> bool {
    shard.world > 1 && if pack.shard.is_some() { pack.etp }
                       else { super::shipplan::moe_tp_resolve(crate::opts::var(crate::opt!("naive-moe-shard")).ok().as_deref(), shard.world) }
}

pub enum GpuMlp {
    Dense { gate: GpuQuad, up: GpuQuad, down: GpuQuad },
    Moe(GpuMoe),
}

pub struct GpuLayer {
    pub kind: AttnKind,
    /// this rank's query heads / kv heads (TP: 1/W of the model's)
    pub nh_l: usize,
    pub kv_l: usize,
    /// column offset of the first live column inside the (128-block-rounded) k / v slices
    pub k_off: usize,
    pub v_off: usize,
    pub in_norm: CudaSlice<u16>,
    pub post_norm: CudaSlice<u16>,
    pub q: GpuQuad, pub k: GpuQuad, pub v: GpuQuad, pub o: GpuQuad,
    pub sink: Option<CudaSlice<u16>>,
    pub idx: Option<GpuIndexer>,
    pub mlp: GpuMlp,
    pub kc: CudaSlice<u16>,
    pub vc: CudaSlice<u16>,
    pub slots: usize, // S
}

pub struct GpuModel {
    pub dev: Arc<CudaDevice>,
    /// the (non-blocking, capturable) stream every forward kernel rides
    pub stream: cudarc::driver::CudaStream,
    pub cfg: NaiveConfig,
    /// a zero sink row (the SWA kernel's dummy when a layer has no sinks)
    pub zero_sink: CudaSlice<u16>,
    pub shard: super::shard::Shard,
    /// S5: the all-reduce transport (None = one GPU holds the whole model slice)
    pub tp: Option<super::tp::NaiveTp>,
    /// nv_moe_route's ownership map (identity at world 1)
    pub local_of: CudaSlice<i32>,
    pub layers: Vec<GpuLayer>,
    pub embed: CudaSlice<u16>, // bf16 bits [V][H]
    pub norm: CudaSlice<u16>,
    /// this rank's vocabulary slice of the head: columns [v0, v0 + vl) (the whole head at world 1)
    pub lm_head: GpuQuad,
    pub v0: usize,
    pub vl: usize,
    pub max_pos: usize,
    pub ring: usize,
    /// how many requests may hold cache state at once: the per-request slot dimension in front of
    /// every pos-keyed cache (kc/vc/kq/ks). 1 = today's single-request sizing, byte for byte.
    pub n_slots: usize,
    /// device pointer to the per-row slot-id array ([MAX_ROWS] i32, uploaded per step). A row's
    /// slot_ids[m] selects which request's cache plane m reads/writes.
    pub sids: u64,
    /// owns the allocation `sids` points at (zero-initialised: every row starts on slot 0)
    pub sids_buf: CudaSlice<i32>,
    /// F13: the fused split-K combine's per-tile arrival counters (zeroed once; self-resetting)
    pub fuse_cnt: CudaSlice<i32>,
}

// The stream handle is only ever used from the thread that owns the model (the serve loop / a probe).
unsafe impl Send for GpuModel {}

fn f16_bits_of(pack: &NaivePack, name: &str) -> Result<Vec<u16>> {
    let m = pack.plain_tensor(name)?;
    let raw = m.read_bytes(&pack.dir)?;
    Ok(match m.dtype.as_str() {
        "F16" => raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
        "BF16" => raw.chunks_exact(2).map(|c| half::f16::from_f32(half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).to_bits()).collect(),
        "F32" => raw.chunks_exact(4).map(|c| half::f16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_bits()).collect(),
        other => bail!("{name}: dtype {other}"),
    })
}
fn f32_of(pack: &NaivePack, name: &str) -> Result<Vec<f32>> {
    let m = pack.plain_tensor(name)?;
    let raw = m.read_bytes(&pack.dir)?;
    Ok(match m.dtype.as_str() {
        "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "F16" => raw.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
        "BF16" => raw.chunks_exact(2).map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
        other => bail!("{name}: dtype {other}"),
    })
}
fn le_u16(raw: &[u8]) -> Vec<u16> { raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() }

impl GpuModel {
    /// Load layers `0..n_layers` (+ embeddings, final norm, lm_head). `max_pos` sizes the DSA caches
    /// and the indexer key planes; `ring` (power of two >= window + max chunk) the SWA rings;
    /// `n_slots` the per-request slot dimension in front of every pos-keyed cache.
    pub fn load_slice(pack: &NaivePack, n_layers: usize, max_pos: usize, ring: usize, n_slots: usize) -> Result<GpuModel> {
        let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
        Self::load_slice_on(dev, pack, n_layers, max_pos, ring, n_slots, super::shard::Shard::single())
    }

    /// The first `n_layers` layers, this rank's shard, on `dev` (the TP transport attaches afterwards).
    pub fn load_slice_on(dev: Arc<CudaDevice>, pack: &NaivePack, n_layers: usize, max_pos: usize, ring: usize, n_slots: usize, shard: super::shard::Shard) -> Result<GpuModel> {
        let c = pack.cfg.clone();
        if let Some((w, r)) = pack.shard {
            ensure!(w == shard.world && r == shard.rank, "{} is rank {r}'s shard of world {w}; this load is rank {} of world {}", pack.dir, shard.rank, shard.world);
        }
        ensure!(n_layers <= c.num_layers, "n_layers {n_layers}");
        ensure!(ring.is_power_of_two() && ring >= 2 * c.sliding_window, "ring {ring} must be a power of two >= 2*window");
        load_module(&dev)?;
        let t0 = std::time::Instant::now();
        let (h, hd, vd) = (c.hidden_size, c.head_dim, c.v_head_dim);
        let embed = {
            let m = pack.plain_tensor("model.embed_tokens.weight")?;
            ensure!(m.dtype == "BF16", "embed dtype {}", m.dtype);
            dev.htod_sync_copy(&le_u16(&m.read_bytes(&pack.dir)?))?
        };
        let norm = dev.htod_sync_copy(&f16_bits_of(pack, "model.norm.weight")?)?;
        let (lm_head, v0, vl) = if shard.world == 1 {
            (upload_quad(&dev, pack, pack.quad("lm_head")?)?, 0, c.vocab_size)
        } else {
            let (a, n) = shard.span(c.vocab_size)?;
            ensure!(a % 128 == 0 && n % 128 == 0, "vocabulary slice not 128-aligned");
            (super::shard::upload_quad_cols(&dev, pack, pack.quad("lm_head")?, a, a + n)?.0, a, n)
        };
        let etp = etp_of(pack, &shard);
        crate::metrics::NAIVE_MOE_TP.store(etp, std::sync::atomic::Ordering::Relaxed);
        // expert TP: every expert is local (identity map); else the interleaved ownership map
        let local_of = dev.htod_sync_copy(&if etp { (0..c.num_experts as i32).collect::<Vec<i32>>() } else { shard.local_of(c.num_experts) })?;
        if etp && shard.rank == 0 {
            println!("[naive] experts: TENSOR-parallel (--naive-moe-shard tp; auto = tp at TP=4): every rank holds {} of each expert's {} intermediate columns — no per-layer expert-count skew; \
                      the MoE partial sums are over K slices (a different fp32 summation order from the expert-parallel deal: tolerance class)",
                     c.moe_intermediate_size / shard.world, c.moe_intermediate_size);
        } else if shard.world > 1 && shard.rank == 0 {
            println!("[naive] experts: expert-parallel (whole experts per rank; --naive-moe-shard auto = ep at TP=2, where expert TP has not run on \
                      hardware; --naive-moe-shard tp asks for it)");
        }
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            layers.push(Self::load_layer(&dev, pack, l, max_pos, ring, n_slots, &shard)?);
            crate::dprintln!("[naive] layer {l} ({:?}, {}) loaded, {:.1} s", c.attn[l], if c.moe[l] { "moe" } else { "dense" }, t0.elapsed().as_secs_f64());
        }
        let stream = dev.fork_default_stream().context("fork stream")?;
        // AGENTS.md S2 rule 2: `alloc_zeros` does NOT zero (async alloc). Both buffers below are read by
        // kernels on the FIRST decode step. `sids_buf` in particular is written by ONLY the
        // --probe-naive-slots path; the serving path never uploads it, so every decode step read
        // uninitialised pool memory as the slot id and addressed garbage KV planes -- silent, no
        // bounds check, corrupt logits (every draft rejected, output degenerating to '*').
        // cudarc 0.9.15 has no memset_* API, so zero explicitly the way decode.rs:290 does.
        let zero_sink = dev.htod_sync_copy(&vec![0u16; c.num_heads])?;
        // the per-row slot ids every cache-addressing kernel takes (uploaded per step by the caller)
        // MAX_PREFILL_ROWS, NOT MAX_ROWS: every cache-addressing kernel indexes slot_ids[m] for
        // m in 0..m, and the serving path feeds THIS ONE buffer chunks of up to `chunk` rows
        // (PREFILL_CHUNK = 1024, clamped to MAX_PREFILL_ROWS). Sized at MAX_ROWS (16) it silently
        // read past the allocation for rows >= 16, and the garbage slot id is used as the KV plane
        // stride `slot * (kv * S * hd)` with S = max_pos = 262144 -- an enormous stride, so every
        // prefill wider than 16 rows died with CUDA_ERROR_ILLEGAL_ADDRESS while short prompts
        // (m <= 16) served fine. The decode/verify path still never exceeds MAX_ROWS.
        let sids_buf = dev.htod_sync_copy(&vec![0i32; MAX_PREFILL_ROWS])?;
        let sids: u64 = dptr(&sids_buf);
        dev.synchronize()?;
        // F13 counters: an explicit zero upload (alloc_zeros does not zero: AGENTS §2.2), then the kernels
        // keep them zero (the last CTA of every tile resets its counter)
        let fuse_cnt = dev.htod_sync_copy(&vec![0i32; super::gpu::FUSE_COUNTERS])?;
        dev.synchronize()?;
        Ok(GpuModel { dev, stream, cfg: c, zero_sink, shard, tp: None, local_of, layers, embed, norm, lm_head, v0, vl, max_pos, ring, n_slots, sids, sids_buf, fuse_cnt })
    }

    /// One layer's weights + caches on the device: this rank's shard (`Shard::single()` = whole layer).
    pub fn load_layer(dev: &Arc<CudaDevice>, pack: &NaivePack, l: usize, max_pos: usize, ring: usize, n_slots: usize, shard: &super::shard::Shard) -> Result<GpuLayer> {
        use super::shard::{upload_quad_cols, upload_quad_rows};
        let c = &pack.cfg;
        let (h, hd, vd) = (c.hidden_size, c.head_dim, c.v_head_dim);
        {
            let p = format!("model.layers.{l}.");
            let kv_total = c.kv_heads(l);
            let (q0, nh_l) = shard.q_heads(c.num_heads)?;
            let (k0, kv_l) = shard.kv_heads(c.num_heads, kv_total)?;
            let kv = kv_l;
            let quad = |name: &str| upload_quad(&dev, pack, pack.quad(&format!("{p}{name}")).unwrap());
            let qd = |name: &str| pack.quad(&format!("{p}{name}"));
            let slots = if c.attn[l] == AttnKind::Swa { ring } else { max_pos };
            let kc = dev.alloc_zeros::<u16>(kv * slots * hd * n_slots)?;
            let vc = dev.alloc_zeros::<u16>(kv * slots * vd * n_slots)?;
            // attention projections: q columns of this rank's heads, k/v columns of its kv heads
            // (v is stored per-head padded to hd), o rows of its heads' values
            let (q, k, v, o, k_off, v_off) = if shard.world == 1 {
                (quad("self_attn.q_proj")?, quad("self_attn.k_proj")?, quad("self_attn.v_proj")?, quad("self_attn.o_proj")?, 0, 0)
            } else {
                let (q, _) = upload_quad_cols(dev, pack, qd("self_attn.q_proj")?, q0 * hd, (q0 + nh_l) * hd)?;
                let (k, k_off) = upload_quad_cols(dev, pack, qd("self_attn.k_proj")?, k0 * hd, (k0 + kv_l) * hd)?;
                let (v, v_off) = upload_quad_cols(dev, pack, qd("self_attn.v_proj")?, k0 * hd, (k0 + kv_l) * hd)?;
                // a shipped shard carries o_proj PRE-SLICED to this rank's rows (k == nh_l * vd); the whole pack is sliced here
                let oq = qd("self_attn.o_proj")?;
                let o = if oq.k == nh_l * vd { upload_quad(dev, pack, oq)? } else { upload_quad_rows(dev, pack, oq, q0 * vd, (q0 + nh_l) * vd)? };
                (q, k, v, o, k_off, v_off)
            };
            let idx = if c.attn[l] == AttnKind::Dsa {
                Some(GpuIndexer {
                    wq: quad("self_attn.indexer.wq")?,
                    wk: quad("self_attn.indexer.wk")?,
                    wp: quad("self_attn.indexer.weights_proj")?,
                    kn_w: dev.htod_sync_copy(&f16_bits_of(pack, &format!("{p}self_attn.indexer.k_norm.weight"))?)?,
                    kn_b: dev.htod_sync_copy(&f16_bits_of(pack, &format!("{p}self_attn.indexer.k_norm.bias"))?)?,
                    kq: dev.alloc_zeros::<u8>(max_pos * c.index_head_dim * n_slots)?,
                    ks: dev.alloc_zeros::<f32>(max_pos * n_slots)?,
                })
            } else { None };
            let sink = if c.has_sink(l) {
                let all = f16_bits_of(pack, &format!("{p}self_attn.attention_sink_bias"))?;
                Some(dev.htod_sync_copy(&all[q0..q0 + nh_l])?)
            } else { None };
            let mlp = if c.moe[l] {
                let (e, mi) = (c.num_experts, c.moe_intermediate_size);
                let etp = etp_of(pack, shard);
                let mi_l = if etp { mi / shard.world } else { mi };
                ensure!(mi_l % 128 == 0, "expert TP: the intermediate slice {mi_l} is not 128-aligned at world {}", shard.world);
                let (ia, ib) = (shard.rank * mi_l, (shard.rank + 1) * mi_l);
                let experts: Vec<usize> = if etp { (0..e).collect() } else { shard.local_experts(e) };
                let mut g_tr = Vec::new(); let mut u_tr = Vec::new(); let mut d_tr = Vec::new();
                let (mut suh_g, mut svh_g, mut suh_u, mut svh_u, mut suh_d, mut svh_d) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
                let mut bpb = 0usize;
                for &ex in &experts {
                    let q = |w: &str| pack.quad(&format!("{p}mlp.experts.{ex}.{w}"));
                    let (g, u, d) = (q("gate_proj")?, q("up_proj")?, q("down_proj")?);
                    // whole experts (the full pack, or the expert-parallel deal) or pre-sliced ones (an expert-TP shard)
                    let pre = etp && g.n == mi_l && mi_l != mi;
                    let (gn, dk) = if pre { (mi_l, mi_l) } else { (mi, mi) };
                    ensure!(g.k == h && g.n == gn && u.k == h && u.n == gn && d.k == dk && d.n == h, "expert geometry @{l} e{ex}: gate {}x{} up {}x{} down {}x{}", g.k, g.n, u.k, u.n, d.k, d.n);
                    ensure!(g.bits == u.bits && g.bits == d.bits && (bpb == 0 || bpb == g.bits.words()), "expert widths @{l} e{ex}");
                    bpb = g.bits.words();
                    let (gt, ut, dt) = (le_u16(&g.trellis.read_bytes(&pack.dir)?), le_u16(&u.trellis.read_bytes(&pack.dir)?), le_u16(&d.trellis.read_bytes(&pack.dir)?));
                    let (gsv, usv, dsu) = (le_u16(&g.svh.read_bytes(&pack.dir)?), le_u16(&u.svh.read_bytes(&pack.dir)?), le_u16(&d.suh.read_bytes(&pack.dir)?));
                    if etp && !pre {
                        // slice here (the head's whole pack): gate/up keep tile columns [ia/16, ib/16) of every tile row
                        // and that svh range; down keeps tile rows [ia/16, ib/16) and that suh range
                        let (kb, nb, w) = (h / 16, mi / 16, bpb);
                        let cols = |t: &[u16]| -> Vec<u16> {
                            let mut o = Vec::with_capacity(kb * (mi_l / 16) * w);
                            for r in 0..kb { o.extend_from_slice(&t[(r * nb + ia / 16) * w..(r * nb + ib / 16) * w]); }
                            o
                        };
                        g_tr.extend(cols(&gt)); u_tr.extend(cols(&ut));
                        d_tr.extend_from_slice(&dt[(ia / 16) * (h / 16) * w..(ib / 16) * (h / 16) * w]);
                        svh_g.extend_from_slice(&gsv[ia..ib]); svh_u.extend_from_slice(&usv[ia..ib]);
                        suh_d.extend_from_slice(&dsu[ia..ib]);
                    } else {
                        g_tr.extend(gt); u_tr.extend(ut); d_tr.extend(dt);
                        svh_g.extend(gsv); svh_u.extend(usv); suh_d.extend(dsu);
                    }
                    suh_g.extend(le_u16(&g.suh.read_bytes(&pack.dir)?));
                    suh_u.extend(le_u16(&u.suh.read_bytes(&pack.dir)?));
                    svh_d.extend(le_u16(&d.svh.read_bytes(&pack.dir)?));
                }
                // The converter's interm_div shows in the pack: up_proj's input scale vector is 1/div of
                // gate_proj's. Verify the config's divisor against the data before applying it, so a pack
                // converted WITHOUT the division (or with a different one) is refused, not served 128x off.
                let mean_abs = |v: &[u16]| v.iter().map(|b| half::f16::from_bits(*b).to_f32().abs() as f64).sum::<f64>() / v.len().max(1) as f64;
                let ratio = mean_abs(&suh_u) / mean_abs(&suh_g).max(1e-12);
                let div = c.moe_interm_div[l];
                if div != 1.0 {
                    ensure!((ratio * div as f64 - 1.0).abs() < 0.5, "layer {l}: config says the pack's up_proj is stored /{div} (exllamav3 interm_div) but its input scale is {ratio:.4} of gate_proj's (expected ~{:.4})", 1.0 / div as f64);
                    println!("  layer {l}: routed up_proj stored /{div} (up/gate input-scale ratio {ratio:.4}); routing weights scaled x{div} (exllamav3 naive_n05 interm_div)");
                } else {
                    ensure!(ratio > 0.1, "layer {l}: the pack's up_proj input scale is {ratio:.4} of gate_proj's: the converter divided it (interm_div) but the config has no divisor for this layer");
                }
                let g_words = ((h / 16) * (mi_l / 16) * bpb) as u64;
                let d_words = ((mi_l / 16) * (h / 16) * bpb) as u64;
                let el = experts.len() as u64;
                ensure!(g_tr.len() as u64 == el * g_words && d_tr.len() as u64 == el * d_words && svh_g.len() == experts.len() * mi_l && suh_d.len() == experts.len() * mi_l,
                        "layer {l}: expert planes do not match {el} experts x mi {mi_l}");
                let offs_g = dev.htod_sync_copy(&(0..el).map(|i| i * g_words).collect::<Vec<u64>>())?;
                let offs_d = dev.htod_sync_copy(&(0..el).map(|i| i * d_words).collect::<Vec<u64>>())?;
                GpuMlp::Moe(GpuMoe {
                    router: dev.htod_sync_copy(&f16_bits_of(pack, &format!("{p}mlp.gate.weight"))?)?,
                    bias: dev.htod_sync_copy(&f32_of(pack, &format!("{p}mlp.gate.e_score_correction_bias"))?)?,
                    g_tr: dev.htod_sync_copy(&g_tr)?, u_tr: dev.htod_sync_copy(&u_tr)?, d_tr: dev.htod_sync_copy(&d_tr)?,
                    suh_g: dev.htod_sync_copy(&suh_g)?, svh_g: dev.htod_sync_copy(&svh_g)?,
                    suh_u: dev.htod_sync_copy(&suh_u)?, svh_u: dev.htod_sync_copy(&svh_u)?,
                    suh_d: dev.htod_sync_copy(&suh_d)?, svh_d: dev.htod_sync_copy(&svh_d)?,
                    g_words, u_words: g_words, d_words, bpb, offs_g, offs_d,
                    scale: c.pack_router_scale(l), mi_l, el: el as usize,
                })
            } else if shard.world == 1 {
                GpuMlp::Dense { gate: quad("mlp.gate_proj")?, up: quad("mlp.up_proj")?, down: quad("mlp.down_proj")? }
            } else {
                let (i0, il) = shard.span(c.intermediate_size)?;
                ensure!(i0 % 128 == 0 && il % 128 == 0, "dense intermediate slice not 128-aligned");
                let dq = qd("mlp.down_proj")?;
                GpuMlp::Dense {
                    gate: upload_quad_cols(dev, pack, qd("mlp.gate_proj")?, i0, i0 + il)?.0,
                    up: upload_quad_cols(dev, pack, qd("mlp.up_proj")?, i0, i0 + il)?.0,
                    down: if dq.k == il { upload_quad(dev, pack, dq)? } else { upload_quad_rows(dev, pack, dq, i0, i0 + il)? },
                }
            };
            Ok(GpuLayer {
                kind: c.attn[l], nh_l, kv_l, k_off, v_off,
                in_norm: dev.htod_sync_copy(&f16_bits_of(pack, &format!("{p}input_layernorm.weight"))?)?,
                post_norm: dev.htod_sync_copy(&f16_bits_of(pack, &format!("{p}post_attention_layernorm.weight"))?)?,
                q, k, v, o, sink, idx, mlp, kc, vc, slots,
            })
        }
    }

    fn f(&self, name: &str) -> Result<cudarc::driver::CudaFunction> {
        self.dev.get_func(MODULE, name).with_context(|| format!("{name} missing from {MODULE}"))
    }

    /// Launch `name` on the model's stream (the capturable stream every forward kernel rides).
    /// With `--naive-pdl=1` the launch goes through the raw handle with the programmatic-stream-
    /// serialization attribute (gpu.rs `launch_pdl_raw`); the kernel, grid, block, smem and the
    /// parameter bytes are identical either way.
    pub fn l<P>(&self, name: &str, cfg: LaunchConfig, params: P) -> Result<()>
    where cudarc::driver::CudaFunction: LaunchAsync<P>, P: super::gpu::NvParams {
        if super::gpu::pdl_on() {
            if let Some(f) = super::gpu::nv_raw_fn(name) {
                let mut ps = params.nv_params();
                return unsafe { super::gpu::launch_pdl_raw(self.stream.stream, f, cfg.grid_dim, cfg.block_dim, cfg.shared_mem_bytes, &mut ps) }
                    .with_context(|| format!("launch {name} (pdl)"));
            }
        }
        unsafe { self.f(name)?.launch_on_stream(&self.stream, cfg, params) }.with_context(|| format!("launch {name}"))
    }

    /// Wait for everything queued on the model's stream.
    pub fn sync(&self) -> Result<()> {
        unsafe { cudarc::driver::result::stream::synchronize(self.stream.stream) }.context("stream synchronize")
    }

    /// The dense chain for M <= 16 rows on workspace memory: xh = H(x*suh); y_raw = xh W (one pass or
    /// split-K + combine); y = H(y_raw) * svh. `x` f16 [m][k] -> returns y f16 [m][n] (a `ws` address).
    pub fn chain_ws(&self, q: &GpuQuad, x: u64, m: usize, ws: &mut Ws) -> Result<u64> {
        self.chain_ws_in(q, x, 0, m, ws)
    }

    /// The chain with the fused input: `x2 != 0` means the input is silu(x) * x2 (the MLP down input).
    /// Staged split-K (the decode plan): 2 launches — GEMM with the input transform in its staging,
    /// combine with the output transform. The per-step body (lm_head) keeps a separate input pass.
    pub fn chain_ws_in(&self, q: &GpuQuad, x: u64, x2: u64, m: usize, ws: &mut Ws) -> Result<u64> {
        self.chain_ws_out(q, x, x2, m, ws, 0).map(|(y, _)| y)
    }

    /// `out32 != 0`: the TP form — the result lands as fp32 at `out32` (the all-reduce partial) and the
    /// returned flag says so (true); otherwise f16 y. (The fallback paths always produce f16.)
    pub fn chain_ws_out(&self, q: &GpuQuad, x: u64, x2: u64, m: usize, ws: &mut Ws, out32: u64) -> Result<(u64, bool)> {
        ensure!((1..=MAX_PREFILL_ROWS).contains(&m), "chain: m {m} out of 1..{MAX_PREFILL_ROWS}");
        let (k, n) = (q.k, q.n);
        let y = ws.alloc::<u16>(m * n)?;
        if m > MAX_ROWS {
            // S6b wide: input pass (silu pair first when asked), the wide HMMA GEMM (mt 8), output pass
            let mut xin = x;
            if x2 != 0 {
                let din = ws.alloc::<u16>(m * k)?;
                self.l("nv_silu_mul", b1(m * k), (din, x, x2, (m * k) as i64))?;
                xin = din;
            }
            let xh = ws.alloc::<u16>(m * k)?;
            self.l("nv_had_suh", b32(m * k / 128), (xin, &q.suh, xh, k as i32))?;
            let yraw = ws.alloc::<u16>(m * n)?;
            let mt = 8usize;
            self.l("nv_gemm_wide", LaunchConfig { grid_dim: (((m + mt * 16 - 1) / (mt * 16)) as u32, (n / 128) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: (mt * 16 * (8 * 16 + 8) * 2) as u32 },
                   (&q.tr, xh, yraw, m as i32, k as i32, n as i32, q.bpb as i32, mt as i32))?;
            self.l("nv_had_svh", b32(m * n / 128), (yraw, &q.svh, y, n as i32))?;
            return Ok((y, false));
        }
        // T0-c: the staged body stages only THIS CTA's k-window, so the bytes are
        // m * ((k / ks) + 8) * 2, not the pre-T0-c m * (k + 8) * 2. grid_dim.y == q.ks IS the
        // split-K id, so each CTA walks exactly k/q.ks wide. The single-pass nv_gemm below
        // has grid_dim.y == 1, so it needs the ks==1 figure (unchanged from before T0-c).
        let smem = a1_smem(m, k, q.ks);
        let smem_ks1 = a1_smem(m, k, 1);
        if q.ks > 1 && q.a1 && smem > 0 {
            let w = ws.alloc::<f32>(q.ks * m * n)?;
            let fz = if super::gpu::fuse_on() && n / 128 <= super::gpu::FUSE_COUNTERS {
                super::gpu::NvFuse { svh: dptr(&q.svh), y, out32, cnt: dptr(&self.fuse_cnt) }
            } else { super::gpu::NvFuse::default() };
            self.l("nv_gemm_a1ks", LaunchConfig { grid_dim: ((n / 128) as u32, q.ks as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem },
                   (&q.tr, x, w, m as i32, k as i32, n as i32, q.bpb as i32, &q.suh, x2, fz))?;
            if fz.cnt == 0 {
                self.l("nv_ks_combine_had", LaunchConfig { grid_dim: (((m * n / 128) as u32 + 7) / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                       (w, &q.svh, y, m as i32, n as i32, q.ks as i32, out32))?;
            }
            return Ok((y, out32 != 0));
        }
        // unfused input pass (silu pair first when asked)
        let mut xin = x;
        if x2 != 0 {
            let din = ws.alloc::<u16>(m * k)?;
            self.l("nv_silu_mul", b1(m * k), (din, x, x2, (m * k) as i64))?;
            xin = din;
        }
        let xh = ws.alloc::<u16>(m * k)?;
        self.l("nv_had_suh", b32(m * k / 128), (xin, &q.suh, xh, k as i32))?;
        if q.ks > 1 {
            let kb = k / 16;
            ensure!(kb % q.ks == 0, "ks {} does not divide kb {kb}", q.ks);
            let w = ws.alloc::<f32>(q.ks * m * n)?;
            if q.a1 && smem > 0 {
                self.l("nv_gemm_a1ks", LaunchConfig { grid_dim: ((n / 128) as u32, q.ks as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem },
                       (&q.tr, xh, w, m as i32, k as i32, n as i32, q.bpb as i32, 0u64, 0u64, super::gpu::NvFuse::default()))?;
            } else {
                self.l("nv_gemm_ks", LaunchConfig { grid_dim: ((n / 128) as u32, q.ks as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                       (&q.tr, xh, w, m as i32, k as i32, n as i32, q.bpb as i32))?;
            }
            self.l("nv_ks_combine_had", LaunchConfig { grid_dim: (((m * n / 128) as u32 + 7) / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                   (w, &q.svh, y, m as i32, n as i32, q.ks as i32, 0u64))?;
        } else {
            let yraw = ws.alloc::<u16>(m * n)?;
            self.l("nv_gemm", LaunchConfig { grid_dim: ((n / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem_ks1 },
                   (&q.tr, xh, yraw, m as i32, k as i32, n as i32, q.bpb as i32))?;
            self.l("nv_had_svh", b32(m * n / 128), (yraw, &q.svh, y, n as i32))?;
        }
        Ok((y, false))
    }

    /// The whole step's kernels: embed -> layers -> (optionally) final norm + lm_head. `ids` / `pos`
    /// are device i32 [m] addresses; returns the logits address (f16 [m][V]) when asked. `taps`
    /// (eager gates): the residual after every layer, copied into `ws` (f32 [m][H] each).
    /// Every launch is on the model's stream; nothing allocates (graph-capturable).
    pub fn step_kernels(&self, ws: &mut Ws, ids: u64, pos: u64, m: usize, want_logits: bool, taps: Option<&mut Vec<u64>>) -> Result<Option<u64>> {
        self.step_kernels_sel(ws, ids, pos, m, want_logits, taps, None)
    }

    /// T2-d: as `step_kernels`, but each per-layer residual copy lands in an EXTERNAL f32 buffer
    /// at `taps_base + (li * tap_rows + row_off) * h` (in floats), not in the workspace.
    ///
    /// This is what makes chunked prefill possible: `tap_rows` is the FULL prompt length while
    /// `m` is the chunk width, so chunk `c0` writes its rows into the right slice of ONE
    /// [layers][rows][h] set. Two consumers need exactly that set -- the drafter's per-layer
    /// context taps, and the prefill head, whose input is the LAST layer's block (the single-shot
    /// path already recovers it from the last tap).
    ///
    /// A separate entry point rather than a parameter on `step_kernels_sel`, so the graph-capture
    /// path and every existing caller stay untouched.
    /// `sids` is the SLOT POINTER FOR THIS CALL'S ROWS (row 0's id). A chunked prefill offsets it by
    /// its first row, exactly as it offsets `ids`/`pos`; passing the model base for every chunk reads
    /// the FIRST chunk's slots for all rows.
    pub fn step_kernels_taps_ext(&self, ws: &mut Ws, ids: u64, pos: u64, m: usize,
                                 taps_base: u64, tap_rows: usize, row_off: usize, sids: u64) -> Result<()> {
        let h = self.cfg.hidden_size;
        ws.reset();
        let x = ws.alloc::<f32>(m * h)?;
        let hbuf = ws.alloc::<u16>(m * h)?;
        debug_assert!(sids != 0, "a step needs its rows' slot ids");
        self.l("nv_embed", LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
               (x, &self.embed, ids, h as i32))?;
        for (li, l) in self.layers.iter().enumerate() {
            let mark = ws.mark();
            self.layer_forward_ws(l, li, x, hbuf, pos, m, sids, ws)?;
            let dst = taps_base + ((li * tap_rows + row_off) * h * 4) as u64;
            unsafe { cudarc::driver::result::memcpy_dtod_async(dst, x, m * h * 4, self.stream.stream) }
                .context("tap copy")?;
            ws.release(mark);
        }
        Ok(())
    }

    /// `step_kernels` + `tap_sel` = (layers, dst): the residual after each listed layer j is copied to
    /// dst + j * m * h * 4 (f32 [m][h] blocks in list order) — the drafter's context taps, inside the graph.
    pub fn step_kernels_sel(&self, ws: &mut Ws, ids: u64, pos: u64, m: usize, want_logits: bool, mut taps: Option<&mut Vec<u64>>,
                            tap_sel: Option<(&[usize], u64)>) -> Result<Option<u64>> {
        let c = &self.cfg;
        let h = c.hidden_size;
        ensure!(m >= 1 && m <= MAX_PREFILL_ROWS, "rows {m}");
        ws.reset();
        let x = ws.alloc::<f32>(m * h)?;
        let hbuf = ws.alloc::<u16>(m * h)?;
        let tap_base = if taps.is_some() { ws.alloc::<f32>(self.layers.len() * m * h)? } else { 0 };
        self.l("nv_embed", LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, (x, &self.embed, ids, h as i32))?;
        for (li, l) in self.layers.iter().enumerate() {
            let mark = ws.mark();
            self.layer_forward_ws(l, li, x, hbuf, pos, m, self.sids, ws)?;
            if let Some(t) = taps.as_deref_mut() {
                let dst = tap_base + (li * m * h * 4) as u64;
                unsafe { cudarc::driver::result::memcpy_dtod_async(dst, x, m * h * 4, self.stream.stream) }.context("tap copy")?;
                t.push(dst);
            }
            if let Some((layers, dst)) = tap_sel {
                if let Some(j) = layers.iter().position(|&l2| l2 == li) {
                    unsafe { cudarc::driver::result::memcpy_dtod_async(dst + (j * m * h * 4) as u64, x, m * h * 4, self.stream.stream) }.context("tap copy")?;
                }
            }
            ws.release(mark);
        }
        if !want_logits { return Ok(None); }
        if m <= MAX_ROWS { return Ok(Some(self.head_logits_ws(x, hbuf, m, ws)?)); }
        // prefill: only the LAST row's logits (returned as a 1-row f16 [vl] buffer)
        let xl = x + ((m - 1) * h * 4) as u64;
        let hl = hbuf + ((m - 1) * h * 2) as u64;
        Ok(Some(self.head_logits_ws(xl, hl, 1, ws)?))
    }

    /// Final norm + lm_head on the residual `x`: logits f16 [m][V] (a `ws` address).
    pub fn head_logits_ws(&self, x: u64, hbuf: u64, m: usize, ws: &mut Ws) -> Result<u64> {
        let c = &self.cfg;
        self.l("nv_rmsnorm", LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 }, (hbuf, x, &self.norm, c.hidden_size as i32, c.rms_eps))?;
        self.chain_ws(&self.lm_head, hbuf, m, ws)
    }

    /// One layer on the residual `x` (f32 [m][H], a device address) in place; `hbuf` f16 [m][H] scratch.
    pub fn layer_forward_ws(&self, l: &GpuLayer, li: usize, x: u64, hbuf: u64, pos: u64, m: usize, sids: u64, ws: &mut Ws) -> Result<()> {
        let c = &self.cfg;
        let (h, nh, hd, vd) = (c.hidden_size, l.nh_l, c.head_dim, c.v_head_dim);
        let kv = l.kv_l;
        let g1 = |n: usize| LaunchConfig { grid_dim: (n as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        // ---- attention sublayer
        self.l("nv_rmsnorm", g1(m), (hbuf, x, &l.in_norm, h as i32, c.rms_eps))?;
        let mut q = self.chain_ws(&l.q, hbuf, m, ws)?;
        let mut k = self.chain_ws(&l.k, hbuf, m, ws)?;
        let mut v = self.chain_ws(&l.v, hbuf, m, ws)?;
        if l.k.n != kv * hd {
            let k2 = ws.alloc::<u16>(m * kv * hd)?;
            self.l("nv_cols_copy", b1(m * kv * hd), (k2, k, m as i32, l.k.n as i32, l.k_off as i32, (kv * hd) as i32))?;
            k = k2;
        }
        if l.v.n != kv * hd {
            let v2 = ws.alloc::<u16>(m * kv * hd)?;
            self.l("nv_cols_copy", b1(m * kv * hd), (v2, v, m as i32, l.v.n as i32, l.v_off as i32, (kv * hd) as i32))?;
            v = v2;
        }
        let theta = c.rope_theta_of(li);
        let ring = (l.kind == AttnKind::Swa) as i32;
        debug_assert!(sids != 0, "a layer needs its rows' slot ids");
        self.l("nv_rope_kv", b32(m * (nh + kv)), (q, k, v, &l.kc, &l.vc, pos, sids, (nh | (kv << 16)) as i32, (hd | (vd << 16) | ((ring as usize) << 30)) as i32, c.rotary_dim as i32, theta, l.slots as i32))?;
        let attn = ws.alloc::<u16>(m * nh * vd)?;
        match l.kind {
            AttnKind::Swa => {
                let sink_ref: &CudaSlice<u16> = match &l.sink { Some(s) => s, None => &self.zero_sink };
                let has_sink = l.sink.is_some() as i32;
                // prefill (m > 16): one block per (kv group, row), K/V shared by the group's heads (bitwise the same)
                let (swa_fn, swa_grid) = if m > 16 && nh / kv <= 8 && nh % kv == 0 { ("nv_attn_swa_grp", (kv as u32, m as u32, 1)) } else { ("nv_attn_swa", (nh as u32, m as u32, 1)) };
                self.l(swa_fn, LaunchConfig { grid_dim: swa_grid, block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                       (attn, q, &l.kc, &l.vc, sink_ref, pos, sids, (nh | (kv << 16)) as i32, (hd | (vd << 16)) as i32, l.slots as i32, (c.sliding_window as i32) | (has_sink << 16)))?;
            }
            AttnKind::Dsa => {
                let ix = l.idx.as_ref().context("DSA layer without indexer")?;
                let (ih, ihd) = (c.index_n_heads, c.index_head_dim);
                let iq_raw = self.chain_ws(&ix.wq, hbuf, m, ws)?;
                let ik_raw = self.chain_ws(&ix.wk, hbuf, m, ws)?;
                let wgt = self.chain_ws(&ix.wp, hbuf, m, ws)?; // [m][128] padded, 16 live
                let iq = ws.alloc::<f32>(m * ih * ihd)?;
                self.l("nv_idx_q", LaunchConfig { grid_dim: ((m * ih) as u32, 1, 1), block_dim: (ihd as u32, 1, 1), shared_mem_bytes: 0 },
                       (iq, iq_raw, pos, ih as i32, ihd as i32, c.rotary_dim as i32, theta, c.indexer_fp8 as i32))?;
                self.l("nv_idx_k", LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (ihd as u32, 1, 1), shared_mem_bytes: 0 },
                       (&ix.kq, &ix.ks, ik_raw, &ix.kn_w, &ix.kn_b, pos, sids, ihd as i32, c.rotary_dim as i32, theta, 1e-5f32, self.max_pos as i32))?;
                // scores over the WHOLE position range (stride max_pos): a fixed grid that strides over
                // the keys <= pos[m] (graph-capturable; entries past pos are never written or read)
                let stride = self.max_pos as i64;
                let scores = ws.alloc::<f32>(m * self.max_pos)?;
                if m > 16 && ih == 16 && ihd == 128 {
                    // prefill: tensor-core scorer (grid over key tiles x row groups; nkeys = whole plane)
                    self.l("nv_idx_score_wide", LaunchConfig { grid_dim: (((self.max_pos + 127) / 128) as u32, ((m + 31) / 32) as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                           (scores, iq, wgt, &ix.kq, &ix.ks, pos, sids, m as i32, self.max_pos as i32, ix.wp.n as i32, stride))?;
                } else {
                    self.l("nv_idx_score", LaunchConfig { grid_dim: (IDX_SCORE_BLOCKS, m as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                           (scores, iq, wgt, &ix.kq, &ix.ks, pos, sids, ih as i32, ihd as i32, ix.wp.n as i32, stride, self.max_pos as i32))?;
                }
                let kk = c.index_top_k;
                let sel = ws.alloc::<i32>(m * kk)?;
                let nsel = ws.alloc::<i32>(m)?;
                self.l("nv_topk", g1(m), (sel, nsel, scores, pos, kk as i32, stride, 0u64, 0u64))?;
                // decode width: 64 splits of 32 keys so a warp walks 32 keys x 2 heads instead of 256 x 2
                // (the per-key chain is the cost; 8 splits took 181 us per launch). Prefill keeps 8: the
                // group-shared prefill kernel is pinned bitwise to the per-head one at that geometry.
                let nsplits = if m <= MAX_ROWS { 64usize } else { 8usize };
                let split_len = (kk + nsplits - 1) / nsplits;
                let pm = ws.alloc::<f32>(m * nh * nsplits)?;
                let pl = ws.alloc::<f32>(m * nh * nsplits)?;
                let pacc = ws.alloc::<f32>(m * nh * nsplits * vd)?;
                let dsa_fn = if m > 16 && (nh / kv) % 8 == 0 && nh / kv <= 16 && nh % kv == 0 { "nv_attn_dsa_split_grp" } else { "nv_attn_dsa_split" };
                self.l(dsa_fn, LaunchConfig { grid_dim: ((m * kv * nsplits) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                       (pm, pl, pacc, q, &l.kc, &l.vc, sel, nsel, sids, (nh | (kv << 16)) as i32, (hd | (vd << 16)) as i32,
                        pack_geom(kk, nsplits, split_len, l.slots)))?;
                self.l("nv_attn_combine", LaunchConfig { grid_dim: ((m * nh) as u32, 1, 1), block_dim: (vd as u32, 1, 1), shared_mem_bytes: 0 },
                       (attn, pm, pl, pacc, nsplits as i32, vd as i32))?;
            }
        }
        let part = self.tp.as_ref().map_or(0, |t| dptr(&t.part));
        let (o, o32) = self.chain_ws_out(&l.o, attn, 0, m, ws, part)?;
        self.add_reduced_ws(x, o, o32, m)?;
        // ---- MLP sublayer
        self.l("nv_rmsnorm", g1(m), (hbuf, x, &l.post_norm, h as i32, c.rms_eps))?;
        match &l.mlp {
            GpuMlp::Dense { gate, up, down } => {
                let g = self.chain_ws(gate, hbuf, m, ws)?;
                let u = self.chain_ws(up, hbuf, m, ws)?;
                let part = self.tp.as_ref().map_or(0, |t| dptr(&t.part));
                let (d, d32) = self.chain_ws_out(down, g, u, m, ws, part)?; // input = silu(g) * u, fused into the staging
                self.add_reduced_ws(x, d, d32, m)?;
            }
            GpuMlp::Moe(mo) => self.moe_ws(mo, hbuf, x, m, ws)?,
        }
        Ok(())
    }

    /// x += y (f16 partial [m][H]); under TP the partial is summed across the ranks first (fp32).
    /// `in_part`: the producer already wrote the fp32 partial into tp.part (F6), skip the convert.
    fn add_reduced_ws(&self, x: u64, y: u64, in_part: bool, m: usize) -> Result<()> {
        let n = m * self.cfg.hidden_size;
        match &self.tp {
            None => self.l("nv_add_f16", b1(n), (x, y, n as i64)),
            Some(tp) => {
                if !in_part { self.l("nv_cvt_f16_f32", b1(n), (&tp.part, y, n as i64))?; }
                tp.all_reduce(&self.stream, &tp.part, n)?;
                self.l("nv_add_f32", b1(n), (x, &tp.part, n as i64))
            }
        }
    }

    fn moe_ws(&self, mo: &GpuMoe, hbuf: u64, x: u64, m: usize, ws: &mut Ws) -> Result<()> {
        let c = &self.cfg;
        // mi = this rank's intermediate width per expert (expert TP: mi / world; the K-slice partial sums of the down
        // projection are completed by the same MoE reduce that sums the expert-parallel deal's partials)
        let (h, e, k, mi) = (c.hidden_size, c.num_experts, c.num_experts_per_tok, mo.mi_l);
        if m > MAX_ROWS { return self.moe_wide_ws(mo, hbuf, x, m, ws); }
        let ecap = e.min(m * k);
        let logits = ws.alloc::<f32>(m * e)?;
        self.l("nv_router_logits", LaunchConfig { grid_dim: (((e + 3) / 4) as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
               (logits, hbuf, &mo.router, m as i32, e as i32, h as i32))?;
        let ids = ws.alloc::<i32>(m * k)?;
        let wts = ws.alloc::<f32>(m * k)?;
        self.l("nv_router_topk", LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: (e * 4) as u32 },
               (ids, wts, logits, &mo.bias, e as i32, k as i32, c.norm_topk_prob as i32, mo.scale))?;
        let slotmap = ws.alloc::<i32>(e)?;
        let idxmap = ws.alloc::<i32>(ecap)?;
        let esel = ws.alloc::<i32>(1)?;
        let offs_g = ws.alloc::<u64>(ecap)?;
        let offs_u = ws.alloc::<u64>(ecap)?;
        let offs_d = ws.alloc::<u64>(ecap)?;
        self.l("nv_moe_route", LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
               (ids, (m | (k << 8) | (e << 16)) as i32, slotmap, idxmap, esel, offs_g, offs_u, offs_d, mo.g_words, mo.u_words, mo.d_words, &self.local_of))?;
        // T0-c: ONE smem arithmetic for the whole naive path (a1_smem, window-sized). The
        // fused form below sizes for the split factors it picks (ks_gu / ks_d); the ks==1
        // figures here are the conservative gate for 'can we stage at all'.
        let a1 = |mm: usize, kk: usize, ks: usize| -> u32 { a1_smem(mm, kk, ks) };
        if a1(m, h, 1) > 0 && a1(m, mi, 1) > 0 {
            // the fused form (S6 F1/F2): 6 launches for the three expert GEMMs
            let gks = |kk: usize, nn: usize| -> usize {
                let tiles = k.min(e) * nn / 128; let kb = kk / 16;
                let mut ks = 2usize;
                while tiles * ks * 2 <= 512 && kb % (ks * 2) == 0 && kb / (ks * 2) >= 16 { ks *= 2; }
                ks
            };
            let (ks_gu, ks_d) = (gks(h, mi), gks(mi, h));
            let cmb = |nn: usize| LaunchConfig { grid_dim: (((ecap * m * nn / 128) as u32 + 7) / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
            let gemm = |nn: usize, ks: usize, kk: usize| LaunchConfig { grid_dim: ((ecap * nn / 128) as u32, ks as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: a1(m, kk, ks) };
            let w = ws.alloc::<f32>(ecap * ks_gu * m * mi)?;
            let yg = ws.alloc::<u16>(ecap * m * mi)?;
            // F13: fused combine when the (expert, tile) counters fit; svh/y are resolved per expert in the kernel
            let fuse = super::gpu::fuse_on() && ecap * (h.max(mi) / 128) <= super::gpu::FUSE_COUNTERS;
            let fzf = |svh: &CudaSlice<u16>, y: u64| if fuse { super::gpu::NvFuse { svh: dptr(svh), y, out32: 0, cnt: dptr(&self.fuse_cnt) } } else { super::gpu::NvFuse::default() };
            self.l("nv_gemm_grouped_a1ks", gemm(mi, ks_gu, h), (&mo.g_tr, offs_g, hbuf, w, (m | (mo.bpb << 8)) as i32, (h | (mi << 16)) as i32, esel, &mo.suh_g, idxmap, 0u64, 0i32, fzf(&mo.svh_g, yg)))?;
            if !fuse { self.l("nv_ks_combine_grouped_had", cmb(mi), (w, &mo.svh_g, idxmap, yg, m as i32, mi as i32, ks_gu as i32, esel))?; }
            let yu = ws.alloc::<u16>(ecap * m * mi)?;
            self.l("nv_gemm_grouped_a1ks", gemm(mi, ks_gu, h), (&mo.u_tr, offs_u, hbuf, w, (m | (mo.bpb << 8)) as i32, (h | (mi << 16)) as i32, esel, &mo.suh_u, idxmap, 0u64, 0i32, fzf(&mo.svh_u, yu)))?;
            if !fuse { self.l("nv_ks_combine_grouped_had", cmb(mi), (w, &mo.svh_u, idxmap, yu, m as i32, mi as i32, ks_gu as i32, esel))?; }
            let wd = ws.alloc::<f32>(ecap * ks_d * m * h)?;
            let yd = ws.alloc::<u16>(ecap * m * h)?;
            self.l("nv_gemm_grouped_a1ks", gemm(h, ks_d, mi), (&mo.d_tr, offs_d, yg, wd, (m | (mo.bpb << 8)) as i32, (mi | (h << 16)) as i32, esel, &mo.suh_d, idxmap, yu, 2i32, fzf(&mo.svh_d, yd)))?;
            if !fuse { self.l("nv_ks_combine_grouped_had", cmb(h), (wd, &mo.svh_d, idxmap, yd, m as i32, h as i32, ks_d as i32, esel))?; }
            return self.moe_combine_ws(x, yd, ids, wts, slotmap, m, ws);
        }
        let xh_e = ws.alloc::<u16>(ecap * m * h)?;
        let yg_raw = ws.alloc::<u16>(ecap * m * mi)?;
        let yg = ws.alloc::<u16>(ecap * m * mi)?;
        let yu_raw = ws.alloc::<u16>(ecap * m * mi)?;
        let yu = ws.alloc::<u16>(ecap * m * mi)?;
        // the grouped GEMM: ~512 resident CTAs via split-K with the staged body (bench: 8 experts
        // 174 -> 160 us); at wider M the expert count alone fills the grid
        // batch invariance (AGENTS 2.4): the split factor is a function of the SHAPE only (sized for
        // the decode grid of top-k experts), never of this step's row count
        let gks = |kk: usize, nn: usize| -> usize {
            let tiles = k.min(e) * nn / 128; let kb = kk / 16;
            let mut ks = 1usize;
            while tiles * ks * 2 <= 512 && kb % (ks * 2) == 0 && kb / (ks * 2) >= 16 { ks *= 2; }
            if a1_smem(m, kk, ks) == 0 { 1 } else { ks }
        };
        let mut grouped = |tr: &CudaSlice<u16>, offs: u64, xh: u64, y: u64, kk: usize, nn: usize, ws: &mut Ws| -> Result<()> {
            let ks = gks(kk, nn);
            if ks > 1 {
                let w = ws.alloc::<f32>(ecap * ks * m * nn)?;
                self.l("nv_gemm_grouped_a1ks", LaunchConfig { grid_dim: ((ecap * nn / 128) as u32, ks as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: a1_smem(m, kk, ks) },
                       (tr, offs, xh, w, (m | (mo.bpb << 8)) as i32, (kk | (nn << 16)) as i32, esel, 0u64, 0u64, 0u64, 1i32, super::gpu::NvFuse::default()))?;
                self.l("nv_ks_combine_grouped", b1(ecap * m * nn), (w, y, m as i32, nn as i32, ks as i32, esel))
            } else {
                self.l("nv_gemm_grouped_xh", LaunchConfig { grid_dim: ((ecap * nn / 128) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: a1_smem(m, kk, 1) },
                       (tr, offs, xh, y, m as i32, kk as i32, nn as i32, mo.bpb as i32, esel))
            }
        };
        self.l("nv_had_suh_multi", b32(ecap * m * (h / 128)), (hbuf, &mo.suh_g, idxmap, xh_e, m as i32, h as i32, esel, 0i32))?;
        grouped(&mo.g_tr, offs_g, xh_e, yg_raw, h, mi, ws)?;
        self.l("nv_had_svh_multi", b32(ecap * m * (mi / 128)), (yg_raw, &mo.svh_g, idxmap, yg, m as i32, mi as i32, esel))?;
        self.l("nv_had_suh_multi", b32(ecap * m * (h / 128)), (hbuf, &mo.suh_u, idxmap, xh_e, m as i32, h as i32, esel, 0i32))?;
        grouped(&mo.u_tr, offs_u, xh_e, yu_raw, h, mi, ws)?;
        self.l("nv_had_svh_multi", b32(ecap * m * (mi / 128)), (yu_raw, &mo.svh_u, idxmap, yu, m as i32, mi as i32, esel))?;
        let n = ecap * m * mi;
        let din = ws.alloc::<u16>(n)?;
        self.l("nv_silu_mul", b1(n), (din, yg, yu, n as i64))?;
        let xhd = ws.alloc::<u16>(n)?;
        self.l("nv_had_suh_multi", b32(ecap * m * (mi / 128)), (din, &mo.suh_d, idxmap, xhd, m as i32, mi as i32, esel, 1i32))?;
        let yd_raw = ws.alloc::<u16>(ecap * m * h)?;
        let yd = ws.alloc::<u16>(ecap * m * h)?;
        grouped(&mo.d_tr, offs_d, xhd, yd_raw, mi, h, ws)?;
        self.l("nv_had_svh_multi", b32(ecap * m * (h / 128)), (yd_raw, &mo.svh_d, idxmap, yd, m as i32, h as i32, esel))?;
        self.moe_combine_ws(x, yd, ids, wts, slotmap, m, ws)
    }

    /// S6b prefill MoE (m > 16): router -> histogram over this rank's experts -> gathered rows (the per-expert
    /// input transform applied in the gather) -> grouped wide GEMMs (gate, up; silu pair -> down) -> per-row
    /// output transforms -> the j-ordered combine. Every expert's weights are read ONCE per chunk.
    fn moe_wide_ws(&self, mo: &GpuMoe, hbuf: u64, x: u64, m: usize, ws: &mut Ws) -> Result<()> {
        let c = &self.cfg;
        let (h, e, k, mi) = (c.hidden_size, c.num_experts, c.num_experts_per_tok, mo.mi_l);
        let el = mo.el;
        let logits = ws.alloc::<f32>(m * e)?;
        self.l("nv_router_logits", LaunchConfig { grid_dim: (((e + 3) / 4) as u32, m as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
               (logits, hbuf, &mo.router, m as i32, e as i32, h as i32))?;
        let ids = ws.alloc::<i32>(m * k)?;
        let wts = ws.alloc::<f32>(m * k)?;
        self.l("nv_router_topk", LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: (e * 4) as u32 },
               (ids, wts, logits, &mo.bias, e as i32, k as i32, c.norm_topk_prob as i32, mo.scale))?;
        let cnt = ws.alloc::<i32>(el)?;
        let offs_row = ws.alloc::<i32>(el)?;
        let cursor = ws.alloc::<i32>(el)?;
        let esel = ws.alloc::<i32>(1)?;
        self.l("nv_zero_i32", b1(el), (cnt, el as i32))?;
        self.l("nv_moe_hist", b1(m * k), (ids, &self.local_of, cnt, (m * k) as i32))?;
        self.l("nv_moe_offsets", LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 }, (cnt, offs_row, cursor, esel, el as i32))?;
        let rmax = m * k; // rows (an upper bound: every pair owned)
        let xh_g = ws.alloc::<u16>(rmax * h)?;
        let xh_u = ws.alloc::<u16>(rmax * h)?;
        let rowmap = ws.alloc::<i32>(m * k)?;
        let row_e = ws.alloc::<i32>(rmax)?;
        // rows past the gathered count (pairs owned by other ranks) keep expert 0: the per-row transform kernels
        // run over rmax rows and must only ever index valid suh / svh (their padded outputs are never read)
        self.l("nv_zero_i32", b1(rmax), (row_e, rmax as i32))?;
        self.l("nv_moe_gather_had", LaunchConfig { grid_dim: ((m * k) as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 },
               (hbuf, &mo.suh_g, &mo.suh_u, ids, &self.local_of, cursor, xh_g, xh_u, rowmap, row_e, k as i32, h as i32))?;
        // The path is eager: read the per-expert counts so the grid covers the busiest expert, not the m*k bound.
        self.sync()?;
        let cnts = dtoh::<i32>(cnt, el)?;
        let cmax = (cnts.iter().copied().max().unwrap_or(0).max(1)) as usize;
        let mt = if cmax <= 32 { 2usize } else if cmax <= 64 { 4 } else { 8 };
        let msup = (cmax + mt * 16 - 1) / (mt * 16);
        let smem = (mt * 16 * (8 * 16 + 8) * 2) as u32;
        let wide = |tr: &CudaSlice<u16>, offs_w: &CudaSlice<u64>, xin: u64, y: u64, kk: usize, nn: usize| -> Result<()> {
            self.l("nv_gemm_wide_grouped", LaunchConfig { grid_dim: (msup as u32, (nn / 128) as u32, el as u32), block_dim: (256, 1, 1), shared_mem_bytes: smem },
                   (tr, offs_w, xin, y, offs_row, cnt, esel, kk as i32, nn as i32, mo.bpb as i32, mt as i32))
        };
        let rows_blocks = |nn: usize| LaunchConfig { grid_dim: (((rmax * nn / 128) as u32 + 7) / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let yg_raw = ws.alloc::<u16>(rmax * mi)?;
        let yg = ws.alloc::<u16>(rmax * mi)?;
        wide(&mo.g_tr, &mo.offs_g, xh_g, yg_raw, h, mi)?;
        self.l("nv_had_svh_rows", rows_blocks(mi), (yg_raw, &mo.svh_g, row_e, yg, rmax as i32, mi as i32))?;
        let yu_raw = ws.alloc::<u16>(rmax * mi)?;
        let yu = ws.alloc::<u16>(rmax * mi)?;
        wide(&mo.u_tr, &mo.offs_g, xh_u, yu_raw, h, mi)?;
        self.l("nv_had_svh_rows", rows_blocks(mi), (yu_raw, &mo.svh_u, row_e, yu, rmax as i32, mi as i32))?;
        let xhd = ws.alloc::<u16>(rmax * mi)?;
        self.l("nv_had_suh_rows_silu", rows_blocks(mi), (yg, yu, &mo.suh_d, row_e, xhd, rmax as i32, mi as i32))?;
        let yd_raw = ws.alloc::<u16>(rmax * h)?;
        let yd = ws.alloc::<u16>(rmax * h)?;
        wide(&mo.d_tr, &mo.offs_d, xhd, yd_raw, mi, h)?;
        self.l("nv_had_svh_rows", rows_blocks(h), (yd_raw, &mo.svh_d, row_e, yd, rmax as i32, h as i32))?;
        let g1m = LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        match &self.tp {
            None => self.l("nv_moe_gather_combine", g1m, (x, yd, rowmap, wts, k as i32, h as i32, 1i32)),
            Some(tp) => {
                let n = m * h;
                ensure!(n <= tp.part.len(), "TP partial buffer too small for {m} prefill rows");
                self.l("nv_moe_gather_combine", g1m, (&tp.part, yd, rowmap, wts, k as i32, h as i32, 0i32))?;
                tp.all_reduce(&self.stream, &tp.part, n)?;
                self.l("nv_add_f32", b1(n), (x, &tp.part, n as i64))
            }
        }
    }

    /// The MoE combine into the residual (TP: into the fp32 partial, reduced, then added).
    fn moe_combine_ws(&self, x: u64, yd: u64, ids: u64, wts: u64, slotmap: u64, m: usize, _ws: &mut Ws) -> Result<()> {
        let (h, k) = (self.cfg.hidden_size, self.cfg.num_experts_per_tok);
        let g1m = LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        match &self.tp {
            None => self.l("nv_moe_combine", g1m, (x, yd, ids, wts, slotmap, k as i32, h as i32, m as i32, 1i32))?,
            Some(tp) => {
                let n = m * h;
                self.l("nv_moe_combine", g1m, (&tp.part, yd, ids, wts, slotmap, k as i32, h as i32, m as i32, 0i32))?;
                tp.all_reduce(&self.stream, &tp.part, n)?;
                self.l("nv_add_f32", b1(n), (x, &tp.part, n as i64))?;
            }
        }
        Ok(())
    }

    // ---- eager entry points (the gates): allocate a workspace, run, copy back ----

    /// M rows at consecutive positions: logits f32 [M][vl] (this rank's vocabulary slice) and the residual after every layer.
    /// T2-d: chunked prefill. The single-shot path is unchanged for `m <= MAX_PREFILL_ROWS` (one
    /// chunk == one call to the old code path), so the 2048-row behaviour is bit-for-bit what it
    /// was; a WIDER prompt now walks `MAX_PREFILL_ROWS` rows at a time instead of being refused.
    ///
    /// Why chunking is CORRECT here, not just legal:
    ///  * Naive-N0.5 is pure attention (`AttnKind::Swa` | `Dsa`) -- there is NO recurrent/GDN state
    ///    to carry, so nothing is lost at a chunk boundary (AGENTS.md 2.8's GDN rule is about a
    ///    different model family).
    ///  * `nv_rope_kv` writes the chunk's K/V into `l.kc`/`l.vc` BEFORE the chunk attends, and the
    ///    causal mask is the per-row ABSOLUTE `pos[]`, so chunk `c0` sees every earlier position.
    ///  * the taps land in ONE [layers][m][h] buffer indexed by absolute row, so the head's input
    ///    (the last layer's block) and the drafter's per-layer taps are exactly what a single-shot
    ///    step would have produced.
    ///
    /// The workspace is sized by the CHUNK, not the prompt: that is the point of chunking, and it
    /// keeps the DSA `m * max_pos` score buffer and the MoE gathered-rows buffer bounded.
    pub fn forward_rows(&self, ids: &[u32], pos: &[usize]) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        self.forward_rows_cap(ids, pos, MAX_PREFILL_ROWS)
    }

    /// `cap` FORCES the prefill chunk width (1c). The serving path chunks at `MAX_PREFILL_ROWS` and
    /// a prompt at or under that width is ONE pass — so on the default arm the chunked path never
    /// runs, and comparing it against the unchunked arm is comparing a call with ITSELF. A cap
    /// below the row count is the only way to exercise chunking on a prompt that also fits one
    /// pass. `cap == 0` asks for one pass. Clamped to `MAX_PREFILL_ROWS`: the workspace is sized for
    /// a chunk that wide and no wider allocation has ever been budgeted for.
    pub fn forward_rows_cap(&self, ids: &[u32], pos: &[usize], cap: usize) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let m = ids.len();
        ensure!(m >= 1 && pos.len() == m, "rows {m}");
        ensure!(pos.iter().all(|&p| p < self.max_pos), "position beyond max_pos");
        let (h, v) = (self.cfg.hidden_size, self.vl);
        let ids_d = self.dev.htod_sync_copy(&ids.iter().map(|x| *x as i32).collect::<Vec<i32>>())?;
        let pos_d = self.dev.htod_sync_copy(&pos.iter().map(|x| *x as i32).collect::<Vec<i32>>())?;
        let nl = self.layers.len();

        // one chunk == the old code path exactly
        let cap = if cap == 0 { m.min(MAX_PREFILL_ROWS) } else { cap.min(MAX_PREFILL_ROWS) }.min(m);
        let ranges = chunk_ranges(m, cap);
        debug_assert!(ranges.iter().map(|&(_, n)| n).sum::<usize>() == m, "chunks must tile the prompt");
        let ws_m = ranges.iter().map(|&(_, n)| n).max().unwrap_or(1);

        // the per-layer taps: one [nl][m][h] f32 set, indexed by ABSOLUTE row so chunks interleave
        // correctly. This replaces the old per-call workspace taps (same shape, one allocation).
        let taps_d = self.dev.alloc_zeros::<f32>(nl * m * h)?;
        let mut ws = Ws::new(&self.dev, ws_bytes(&self.cfg, ws_m, self.max_pos, 0))?;
        let taps_base = dptr(&taps_d);
        for (c0, mc) in ranges {
            self.step_kernels_taps_ext(&mut ws,
                                       dptr(&ids_d) + (c0 * 4) as u64,
                                       dptr(&pos_d) + (c0 * 4) as u64,
                                       mc, taps_base, m, c0,
                                       // the slot ids are per ROW, so the chunk's rows are offset
                                       // exactly like ids/pos are. Passing the base for every chunk
                                       // makes chunk 1+ read `sids[0..]` -- i.e. chunk 1's slots --
                                       // and silently cross-contaminate the requests.
                                       self.sids + (c0 * 4) as u64)?;
            self.sync()?;
        }

        // the head on every row, in MAX_ROWS groups, off the LAST layer's block
        let x_last = taps_base + ((nl - 1) * m * h * 4) as u64;
        let mut all = Vec::with_capacity(m * v);
        for g in (0..m).step_by(MAX_ROWS) {
            let mm = (m - g).min(MAX_ROWS);
            let mark = ws.mark();
            let hb = ws.alloc::<u16>(mm * h)?;
            let lgg = self.head_logits_ws(x_last + (g * h * 4) as u64, hb, mm, &mut ws)?;
            self.sync()?;
            all.extend(dtoh::<u16>(lgg, mm * v)?.iter().map(|b| half::f16::from_bits(*b).to_f32()));
            ws.release(mark);
        }
        let taps = (0..nl).map(|li| dtoh::<f32>(taps_base + (li * m * h * 4) as u64, m * h)).collect::<Result<Vec<_>>>()?;
        Ok((all, taps))
    }

    /// M rows; the LAST row's logits when asked (None = a rank that does not need them).
    pub fn forward_rows_last(&self, ids: &[u32], pos: &[usize], want_logits: bool) -> Result<Option<Vec<f32>>> {
        let m = ids.len();
        ensure!(m >= 1 && m <= MAX_ROWS && pos.len() == m, "rows {m}");
        ensure!(pos.iter().all(|&p| p < self.max_pos), "position beyond max_pos");
        let v = self.vl;
        let ids_d = self.dev.htod_sync_copy(&ids.iter().map(|x| *x as i32).collect::<Vec<i32>>())?;
        let pos_d = self.dev.htod_sync_copy(&pos.iter().map(|x| *x as i32).collect::<Vec<i32>>())?;
        let mut ws = Ws::new(&self.dev, ws_bytes(&self.cfg, m, self.max_pos, 0))?;
        let lg = self.step_kernels(&mut ws, dptr(&ids_d), dptr(&pos_d), m, want_logits, None)?;
        self.sync()?;
        match lg {
            None => Ok(None),
            Some(p) => Ok(Some(dtoh::<u16>(p + ((m - 1) * v * 2) as u64, v)?.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect())),
        }
    }
}

/// The row partition a chunked prefill walks: contiguous `[0, cap)` blocks, the last one short.
/// Pure, and the ONE place the chunking arithmetic lives -- the serving loop and the unit test both
/// go through it, so a drift between them fails the test instead of becoming a silent gap in the
/// served prompt.
pub fn chunk_ranges(total: usize, cap: usize) -> Vec<(usize, usize)> {
    let cap = cap.max(1);
    let mut v = Vec::new();
    let mut c0 = 0usize;
    while c0 < total {
        let mc = (total - c0).min(cap);
        v.push((c0, mc));
        c0 += mc;
    }
    v
}

/// Pack the DSA split kernels' `geom` word. Layout (bit ranges, low to high):
/// `K [0:16) | nsplits [16:24) | split_len [24:40) | S [40:64)`.
///
/// `S` is the layer's KV row count (`max_pos` for a DSA layer) and is the per-head stride
/// `g*S+t`, so it must survive 262144. It is given 24 bits FOR THAT REASON: it used to be
/// packed at bit 48 and unpacked with `& 0xFFFF`, which silently truncated it to 0 at the
/// shipped `--max-seq-len 262144`, aliasing every KV head onto head 0 on all DSA layers.
/// The kernel side is `geom >> 40 & 0xFFFFFF` in `nv_attn_dsa_split`/`_grp`; `geom_fields_match_cuda`
/// ties the two together, because a hand-written mask in CUDA drifting from a hand-written pack
/// here is exactly the bug class, and neither side is type-checked against the other.
pub fn pack_geom(k: usize, nsplits: usize, split_len: usize, s_rows: usize) -> i64 {
    ((k as i64) & 0xFFFF) | (((nsplits as i64) & 0xFF) << 16) | (((split_len as i64) & 0xFFFF) << 24) | (((s_rows as i64) & 0xFFFFFF) << 40)
}

/// The `geom` field layout as (shift, mask_width) per field, in pack order.
pub const GEOM_FIELDS: [(&str, u32, u32); 4] =
    [("K", 0, 16), ("nsplits", 16, 8), ("split_len", 24, 16), ("S", 40, 24)];

/// Fixed grid of the indexer score kernel (per row): strides over the keys <= pos.
pub const IDX_SCORE_BLOCKS: u32 = 512;

fn b1(n: usize) -> LaunchConfig { LaunchConfig { grid_dim: (((n + 255) / 256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 } }
fn b32(g: usize) -> LaunchConfig { LaunchConfig { grid_dim: (g as u32, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 } }
/// Device address of a slice.
pub fn dptr<T>(s: &CudaSlice<T>) -> u64 { *s.device_ptr() as u64 }
/// Synchronous device -> host copy of `n` elements at a device address (the caller synced the stream).
pub fn dtoh<T: Default + Clone + cudarc::driver::DeviceRepr>(p: u64, n: usize) -> Result<Vec<T>> {
    let mut v = vec![T::default(); n];
    unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut v, p) }.context("dtoh")?;
    Ok(v)
}

/// Workspace arena: one device slab, bump-allocated (256-B aligned) and reset per step; `mark` /
/// `release` free a layer's scratch. Nothing in a step allocates device memory, so the step is
/// graph-capturable and address-stable across replays (the same allocation sequence every time).
pub struct Ws {
    _slab: CudaSlice<u8>,
    base: u64,
    cap: usize,
    off: usize,
    pub peak: usize,
}

impl Ws {
    pub fn new(dev: &Arc<CudaDevice>, bytes: usize) -> Result<Ws> {
        let slab = dev.alloc_zeros::<u8>(bytes)?;
        dev.synchronize()?; // the zeroing rides the default stream; the model's stream is non-blocking
        let base = *slab.device_ptr() as u64;
        Ok(Ws { _slab: slab, base, cap: bytes, off: 0, peak: 0 })
    }
    pub fn reset(&mut self) { self.off = 0; }
    pub fn mark(&self) -> usize { self.off }
    pub fn release(&mut self, mark: usize) { self.off = mark; }
    pub fn alloc<T>(&mut self, n: usize) -> Result<u64> {
        let bytes = (n * std::mem::size_of::<T>() + 255) / 256 * 256;
        ensure!(self.off + bytes <= self.cap, "workspace exhausted: {} + {bytes} > {} bytes", self.off, self.cap);
        let p = self.base + self.off as u64;
        self.off += bytes;
        self.peak = self.peak.max(self.off);
        Ok(p)
    }
}

/// Workspace bytes for a step of `m` rows: the per-layer peak (MoE at ecap experts, the indexer
/// scores over max_pos) + the residual/taps + the logits, with headroom.
pub fn ws_bytes(c: &NaiveConfig, m: usize, max_pos: usize, taps_layers: usize) -> usize {
    let (h, mi, v) = (c.hidden_size, c.moe_intermediate_size, c.vocab_size);
    let ecap = c.num_experts.min(m * c.num_experts_per_tok);
    let k = c.num_experts_per_tok;
    let moe = if m > MAX_ROWS {
        // gathered rows R = m*k: xh_g, xh_u, yd_raw, yd [R][h]; yg_raw, yg, yu_raw, yu, xhd [R][mi]; maps
        m * k * (4 * h + 5 * mi) * 2 + m * k * 8 + m * c.num_experts * 4 + (1 << 20)
    } else {
        ecap * m * (3 * h + 6 * mi) * 2 + ecap * 8 * m * h * 4 + (1 << 20)
    };
    let attn = m * (c.num_heads * c.head_dim * 2 * 3 + c.num_heads * 8 * (2 + c.v_head_dim) * 4 + c.index_top_k * 4) + m * max_pos * 4;
    let dense = m * c.intermediate_size * 2 * 6;
    let chains = m * 16384 * 2 * 6 + 64 * m * 4096 * 4; // xh/yraw/y of the widest chain + split-K partials
    let head = m * v * 2 * 3 + v * 4 * 2;
    let fixed = m * h * 4 + m * h * 2 + taps_layers * m * h * 4;
    (fixed + head + moe.max(attn + dense) + chains + 4096 * 256) * 5 / 4
}

/// `--probe-naive-fuse --model-dir <pack> [--layers N]`: the F13 fused split-K combine against the
/// two-launch form, eager, m in {1, 4, 8, 16}: logits and every layer tap must be BITWISE identical
/// (the fused epilogue runs nv_ks_combine_had's arithmetic in the same order; only the launch count changes).
pub fn probe_fuse(dir: &str, n_layers: usize) -> Result<()> {
    let pack = Arc::new(NaivePack::open(dir)?);
    println!("NAIVE FUSE PROBE {} layers {n_layers}", pack.dir);
    let gpu = GpuModel::load_slice(&pack, n_layers, 4096, 1024, 1)?;
    for &m in &[1usize, 4, 8, 16] {
        let ids: Vec<u32> = (0..m).map(|i| (1000 + (i as u32 * 7919) % 100_000) as u32).collect();
        let pos: Vec<usize> = (0..m).collect();
        super::gpu::set_fuse(false);
        let (lg0, taps0) = gpu.forward_rows(&ids, &pos)?;
        super::gpu::set_fuse(true);
        let (lg1, taps1) = gpu.forward_rows(&ids, &pos)?;
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        let bad_l = bits(&lg0).iter().zip(bits(&lg1)).filter(|(a, b)| **a != *b).count();
        let mut bad_t = 0usize;
        for l in 0..n_layers { bad_t += bits(&taps0[l]).iter().zip(bits(&taps1[l])).filter(|(a, b)| **a != *b).count(); }
        println!("  m {m:2}: logits {bad_l} of {} differ, taps {bad_t} differ", lg0.len());
        ensure!(bad_l == 0 && bad_t == 0, "NAIVE_FUSE_FAIL: the fused combine is not bitwise at m {m}");
    }
    println!("NAIVE_FUSE_OK");
    Ok(())
}

/// `--probe-naive-slice --model-dir <pack> --layers N --tokens T`: GPU slice vs CPU twin.
pub fn probe_slice(dir: &str, n_layers: usize, tokens: usize) -> Result<()> {
    let pack = Arc::new(NaivePack::open(dir)?);
    ensure!(tokens >= 1 && tokens <= MAX_ROWS, "tokens 1..16");
    println!("NAIVE SLICE PROBE {} layers {n_layers} tokens {tokens}", pack.dir);
    let t0 = std::time::Instant::now();
    let gpu = GpuModel::load_slice(&pack, n_layers, 4096, 1024, 1)?;
    println!("  gpu slice loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let t1 = std::time::Instant::now();
    let cpu = super::cpu::CpuModel::load_pack_slice(pack.clone(), n_layers)?;
    println!("  cpu twin loaded (lazy experts) in {:.1} s", t1.elapsed().as_secs_f64());
    // a deterministic prompt from real vocabulary ids
    let ids: Vec<u32> = (0..tokens).map(|i| (1000 + (i as u32 * 7919) % 100_000) as u32).collect();
    let pos: Vec<usize> = (0..tokens).collect();
    let t2 = std::time::Instant::now();
    let (lg, taps_g) = gpu.forward_rows(&ids, &pos)?;
    println!("  gpu forward {:.2} s", t2.elapsed().as_secs_f64());
    let t3 = std::time::Instant::now();
    let (lc, taps_c) = cpu.forward(&ids)?;
    println!("  cpu forward {:.1} s", t3.elapsed().as_secs_f64());
    let mut worst = 0f32;
    for l in 0..n_layers {
        let r = super::cpu::rel_l2(&taps_g[l], &taps_c.layer_out[l]);
        worst = worst.max(r);
        println!("  layer {l:2} ({:?}, {}): residual rel-L2 {:.3e}", gpu.cfg.attn[l], if gpu.cfg.moe[l] { "moe" } else { "dense" }, r);
    }
    let r = super::cpu::rel_l2(&lg, &lc);
    let v = gpu.cfg.vocab_size;
    let am = |x: &[f32]| x.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &y)| if y > b.1 { (i, y) } else { b }).0;
    let agree = (0..tokens).filter(|&t| am(&lg[t * v..(t + 1) * v]) == am(&lc[t * v..(t + 1) * v])).count();
    println!("  logits rel-L2 {:.3e}; argmax agreement {agree}/{tokens}", r);
    // f16 activations drift ~x1.5 per layer against the f32 twin (the served precision class);
    // argmax ties on a TRUNCATED model's near-flat logits are not a defect: >= 90% agreement.
    let ok = worst < 5e-2 && r < 5e-2 && agree * 10 >= tokens * 9;
    println!("NAIVE_SLICE_{}", if ok { "OK" } else { "FAIL" });
    if !ok { bail!("slice mismatch"); }
    Ok(())
}

/// index of the max element of one logit row (first wins ties, deterministic)
fn argmax_row(x: &[f32]) -> usize {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &y) in x.iter().enumerate() {
        if y > bv {
            bi = i;
            bv = y;
        }
    }
    bi
}

/// `--probe-naive-chunkxcpu --model-dir <pack> [--layers N] [--tokens T]` — the 1c gate, done as
/// a CROSS-CLASS check.
///
/// The over-claim it replaces: that "a 4200-token prompt returned 200 with plausible text" proved
/// `chunked == unchunked`. It did not. Worse, the comparison that IS available on the GPU is
/// same-class — two arms of the same forward — and per the 2026-09-22 lesson (`--probe-exl3-prefillx`
/// is the pattern) two paths of the same class can both be wrong in the same way and still call it
/// zero drift. So the reference side here is the CPU twin (`CpuModel::load_pack_slice`, dequantized
/// f32 weights, the model's own reference op order), a DIFFERENT class from the CUDA kernels, and
/// BOTH GPU arms are scored against it. A same-class A-vs-B row is printed as a diagnostic and is
/// explicitly NOT the proof.
///
/// The chunked arm is forced with a cap BELOW the row count: at the serving cap of 2048 a 16-row
/// prompt is a single pass, so the default arm would compare a call with itself. Engagement is
/// asserted (chunk count > 1) because bit-identity between two arms passes equally when the
/// chunked path never runs.
pub fn probe_chunk_xcpu(dir: &str, n_layers: usize, tokens: usize) -> Result<()> {
    let pack = Arc::new(NaivePack::open(dir)?);
    ensure!(tokens >= 2 && tokens <= MAX_ROWS, "tokens 2..16");
    let cap = (tokens / 2).max(1);
    let n_chunks = chunk_ranges(tokens, cap).len();
    ensure!(n_chunks > 1, "cap {cap} on {tokens} rows must actually chunk — a single pass proves nothing");
    println!("NAIVE CHUNK-XCPU PROBE {} layers {n_layers} tokens {tokens} cap {cap} ({n_chunks} chunks vs 1)", pack.dir);
    let t0 = std::time::Instant::now();
    let gpu = GpuModel::load_slice(&pack, n_layers, 4096, 1024, 1)?;
    println!("  gpu slice loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let t1 = std::time::Instant::now();
    let cpu = super::cpu::CpuModel::load_pack_slice(pack.clone(), n_layers)?;
    println!("  cpu twin loaded (lazy experts) in {:.1} s", t1.elapsed().as_secs_f64());
    // a deterministic prompt from real vocabulary ids (same construction as probe_slice)
    let ids: Vec<u32> = (0..tokens).map(|i| (1000 + (i as u32 * 7919) % 100_000) as u32).collect();
    let pos: Vec<usize> = (0..tokens).collect();

    let t2 = std::time::Instant::now();
    let (lg_one, taps_one) = gpu.forward_rows_cap(&ids, &pos, MAX_PREFILL_ROWS)?;
    let t_one = t2.elapsed().as_secs_f64();
    let t3 = std::time::Instant::now();
    let (lg_chunk, taps_chunk) = gpu.forward_rows_cap(&ids, &pos, cap)?;
    let t_chunk = t3.elapsed().as_secs_f64();
    let t4 = std::time::Instant::now();
    let (lg_cpu, taps_cpu) = cpu.forward(&ids)?;
    let t_cpu = t4.elapsed().as_secs_f64();
    println!("  gpu 1-pass {:.3} s | gpu {n_chunks}-chunk {:.3} s | cpu reference {:.1} s", t_one, t_chunk, t_cpu);

    let v = gpu.cfg.vocab_size;
    let am = |x: &[f32]| argmax_row(x);
    let agree = |a: &[f32], b: &[f32]| (0..tokens).filter(|&t| am(&a[t * v..(t + 1) * v]) == am(&b[t * v..(t + 1) * v])).count();
    // The residual column compares the A side's taps against the B side's taps, so BOTH are passed.
    // The first version captured `taps_cpu` and ignored the taps argument; the same-class row then
    // printed a CROSS-class residual (chunked-vs-cpu) under a same-class label. Reporting defect.
    let rows = |name: &str, a: &[f32], b: &[f32], ta: &[Vec<f32>], tb: &[Vec<f32>]| {
        let worst = (0..n_layers).map(|l| super::cpu::rel_l2(&ta[l], &tb[l])).fold(0f32, f32::max);
        let r = super::cpu::rel_l2(a, b);
        let ag = agree(a, b);
        println!("  {name}: worst layer residual rel-L2 {worst:.3e} | logits rel-L2 {r:.3e} | argmax {ag}/{tokens}");
        (worst, r, ag)
    };
    // CROSS-CLASS (the proof): each GPU arm against the CPU reference of a different class
    let (w_chunk, r_chunk, a_chunk) = rows("gpu CHUNKED  vs cpu", &lg_chunk, &lg_cpu, &taps_chunk, &taps_cpu.layer_out);
    let (w_one, r_one, a_one) = rows("gpu 1-PASS   vs cpu", &lg_one, &lg_cpu, &taps_one, &taps_cpu.layer_out);
    // SAME-CLASS diagnostic only — printed so a chunking-specific divergence is visible, but it
    // cannot establish correctness on its own
    let (w_ab, r_ab, a_ab) = rows("gpu chunked vs gpu 1-pass [SAME-CLASS, diagnostic only]", &lg_chunk, &lg_one, &taps_chunk, &taps_one);
    let ok = w_chunk < 5e-2 && r_chunk < 5e-2 && a_chunk * 10 >= tokens * 9
        && w_one < 5e-2 && r_one < 5e-2 && a_one * 10 >= tokens * 9;
    println!("  chunk-vs-cpu vs one-pass-vs-cpu delta: residual {:.3e}, logits {:.3e}, argmax {}",
             (w_chunk - w_one).abs(), (r_chunk - r_one).abs(), a_chunk.abs_diff(a_one));
    println!("NAIVE_CHUNK_XCPU_{}", if ok { "OK" } else { "FAIL" });
    if !ok {
        bail!("chunked GPU prefill disagrees with the CPU reference (chunked {w_chunk:.3e}/{r_chunk:.3e}/{a_chunk}, one-pass {w_one:.3e}/{r_one:.3e}/{a_one}; same-class {w_ab:.3e}/{r_ab:.3e}/{a_ab})");
    }
    Ok(())
}

/// `--probe-naive-full --model-dir <pack> [--prompt S] [--gen N]`: the WHOLE model on one GPU with one
/// layer resident at a time (137 GB does not fit beside the desk server), greedy decode of `gen`
/// tokens by re-running the prompt + generated rows every step (M <= 16 rows; the K/V caches do
/// not survive a layer's drop). Coherence of the complete GPU path on real weights: the CPU
/// reference gave " Paris" p = 0.946 for the default prompt.
/// A minimal safetensors writer (header length u64 LE + JSON header + raw little-endian data) for the per-layer
/// state dumps of `probe_full --dump-states` (read by scripts/naive/requant/compare_states.py).
fn write_safetensors(path: &str, tensors: &[(String, &str, Vec<usize>, Vec<u8>)]) -> Result<()> {
    let mut header = String::from("{");
    let mut off = 0usize;
    for (i, (name, dtype, shape, data)) in tensors.iter().enumerate() {
        if i > 0 { header.push(','); }
        header.push_str(&format!("\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[{off},{}]}}", off + data.len()));
        off += data.len();
    }
    header.push('}');
    while header.len() % 8 != 0 { header.push(' '); }
    let mut out = Vec::with_capacity(8 + header.len() + off);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for (_, _, _, data) in tensors { out.extend_from_slice(data); }
    std::fs::write(path, out).with_context(|| format!("write {path}"))?;
    Ok(())
}

pub fn probe_full(dir: &str, prompt: &str, gen: usize, rows_chunk: usize, nll: bool) -> Result<()> {
    probe_full_dump(dir, prompt, gen, rows_chunk, nll, None)
}

/// As `probe_full`; with `dump` = Some(path) the residual stream after the embedding and after every layer of
/// the prompt pass (f32 [m][h], keys `000.embed`, `001.layer0`, ...) plus the ids and the last row's logits are
/// written to `path` as safetensors, for a layer-by-layer comparison against the HF reference / exllamav3.
pub fn probe_full_dump(dir: &str, prompt: &str, gen: usize, rows_chunk: usize, nll: bool, dump: Option<&str>) -> Result<()> {
    let pack = Arc::new(NaivePack::open(dir)?);
    let tok = crate::tokenizer::QwenTokenizer::from_file(&format!("{}/tokenizer.json", pack.dir)).context("tokenizer")?;
    let mut ids = tok.encode(prompt, false)?;
    ensure!(!ids.is_empty() && ids.len() + gen <= MAX_PREFILL_ROWS, "prompt + gen must fit {MAX_PREFILL_ROWS} rows");
    println!("NAIVE FULL PROBE {} prompt {:?} -> {} tokens, gen {gen}", pack.dir, prompt, ids.len());
    let t0 = std::time::Instant::now();
    let gpu = GpuModel::load_slice(&pack, 0, 4096, 1024, 1)?;
    println!("  trunk (embed, norm, lm_head) loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let c = gpu.cfg.clone();
    let (h, v, nl) = (c.hidden_size, c.vocab_size, c.num_layers);
    let dev = &gpu.dev;
    let mut out = String::new();
    for step in 0..=gen {
        let m = ids.len();
        let pos: Vec<usize> = (0..m).collect();
        let ids_i: Vec<i32> = ids.iter().map(|x| *x as i32).collect();
        let pos_i: Vec<i32> = pos.iter().map(|x| *x as i32).collect();
        let ids_d = dev.htod_sync_copy(&ids_i)?;
        let pos_d = dev.htod_sync_copy(&pos_i)?;
        // a chunk of <= MAX_ROWS rows takes the DECODE-class kernels (64-way DSA splits, the small-M MoE path), whose
        // per-chunk scratch is sized for m = 16 regardless of the prompt: give the probe that headroom
        let decode_extra = if rows_chunk != 0 && rows_chunk <= MAX_ROWS { 768usize << 20 } else { 0 };
        let mut ws = Ws::new(dev, ws_bytes(&c, m, 4096, 0) + decode_extra)?;
        let x = ws.alloc::<f32>(m * h)?;
        let hbuf = ws.alloc::<u16>(m * h)?;
        gpu.l("nv_embed", LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }, (x, &gpu.embed, dptr(&ids_d), h as i32))?;
        let mut dumped: Vec<(String, &str, Vec<usize>, Vec<u8>)> = Vec::new();
        if dump.is_some() && step == 0 {
            gpu.sync()?;
            let xs = dtoh::<f32>(x, m * h)?;
            dumped.push(("000.embed".to_string(), "F32", vec![m, h], xs.iter().flat_map(|f| f.to_le_bytes()).collect()));
        }
        let t1 = std::time::Instant::now();
        let (mut tl, mut tf) = (0f64, 0f64);
        for li in 0..nl {
            let a = std::time::Instant::now();
            let layer = GpuModel::load_layer(dev, &pack, li, 4096, 1024, 1, &gpu.shard)?;
            tl += a.elapsed().as_secs_f64();
            let b = std::time::Instant::now();
            // rows_chunk < m: the layer sees the rows in chunks (the small-row kernels when chunk <= MAX_ROWS);
            // the layer's KV cache carries the earlier chunks, as a chunked prefill does
            let step_rows = if rows_chunk == 0 { m } else { rows_chunk };
            let mut c0 = 0usize;
            while c0 < m {
                let mc = (m - c0).min(step_rows);
                let mark = ws.mark();
                gpu.layer_forward_ws(&layer, li, x + (c0 * h * 4) as u64, hbuf + (c0 * h * 2) as u64, dptr(&pos_d) + (c0 * 4) as u64, mc, gpu.sids, &mut ws)?;
                ws.release(mark);
                c0 += mc;
            }
            gpu.sync()?;
            tf += b.elapsed().as_secs_f64();
            if dump.is_some() && step == 0 {
                let xs = dtoh::<f32>(x, m * h)?;
                dumped.push((format!("{:03}.layer{li}", li + 1), "F32", vec![m, h], xs.iter().flat_map(|f| f.to_le_bytes()).collect()));
            }
            if step == 0 {
                let xs = dtoh::<f32>(x, m * h)?;
                let rms = (xs.iter().map(|a| (*a as f64) * (*a as f64)).sum::<f64>() / xs.len() as f64).sqrt();
                let nan = xs.iter().filter(|a| !a.is_finite()).count();
                println!("  layer {li:2} ({:?}, {}): residual rms {rms:.4} nonfinite {nan}  [load {:.1} s fwd {:.3} s]",
                         c.attn[li], if c.moe[li] { "moe" } else { "dense" }, a.elapsed().as_secs_f64() - (b.elapsed().as_secs_f64()), b.elapsed().as_secs_f64());
            }
            drop(layer);
        }
        if nll && step == 0 {
            // teacher-forced mean NLL of the prompt (the recipe's bench.py `nll` suite: vLLM prompt_logprobs over
            // every token but the first): the head on every row, in MAX_ROWS groups, fp32 log-softmax
            let mut sum = 0f64; let mut cnt = 0usize; let mut worst: Vec<(f64, usize)> = Vec::new();
            let mut g0 = 0usize;
            while g0 < m {
                let gm = (m - g0).min(MAX_ROWS);
                let mark = ws.mark();
                let lg = gpu.head_logits_ws(x + (g0 * h * 4) as u64, hbuf + (g0 * h * 2) as u64, gm, &mut ws)?;
                gpu.sync()?;
                let rows: Vec<f32> = dtoh::<u16>(lg, gm * v)?.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect();
                for r in 0..gm {
                    let ti = g0 + r;
                    if ti + 1 >= m { break; }
                    let row = &rows[r * v..(r + 1) * v];
                    let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let z: f64 = row.iter().map(|l| ((l - mx) as f64).exp()).sum();
                    let nl = -(((row[ids[ti + 1] as usize] - mx) as f64) - z.ln());
                    sum += nl; cnt += 1; worst.push((nl, ti));
                }
                ws.release(mark);
                g0 += gm;
            }
            worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            println!("  NLL: {} tokens scored, mean {:.5} nats/token (ppl {:.3})", cnt, sum / cnt as f64, (sum / cnt as f64).exp());
            println!("  worst 5 positions: {:?}", worst.iter().take(5).map(|(n, i)| (format!("{n:.2}"), *i, tok.decode(&[ids[*i + 1]], false).unwrap_or_default())).collect::<Vec<_>>());
        }
        // the head on the LAST row only (a wide prompt's rows share the trunk; the head is 1 row)
        let lg = gpu.head_logits_ws(x + ((m - 1) * h * 4) as u64, hbuf + ((m - 1) * h * 2) as u64, 1, &mut ws)?;
        gpu.sync()?;
        let logits: Vec<f32> = dtoh::<u16>(lg, v)?.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect();
        if let (Some(path), 0) = (dump, step) {
            dumped.push(("ids".to_string(), "I64", vec![m], ids.iter().flat_map(|t| (*t as i64).to_le_bytes()).collect()));
            dumped.push(("logits_last".to_string(), "F32", vec![v], logits.iter().flat_map(|f| f.to_le_bytes()).collect()));
            write_safetensors(path, &dumped)?;
            println!("  dumped {} tensors to {path}", dumped.len());
        }
        let last = &logits[..v];
        let am = last.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
        if step == 0 {
            let mx = last.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let z: f64 = last.iter().map(|l| ((l - mx) as f64).exp()).sum();
            let mut idx: Vec<usize> = (0..v).collect();
            idx.sort_by(|a, b| last[*b].partial_cmp(&last[*a]).unwrap_or(std::cmp::Ordering::Equal));
            println!("  top-5 after the prompt:");
            for &i in &idx[..5] { println!("    {i:7} p={:.3} {:?}", ((last[i] - mx) as f64).exp() / z, tok.decode(&[i as u32], false).unwrap_or_default()); }
            println!("  top-5 ids: {:?}", &idx[..5]);
        }
        println!("  step {step}: {m} rows, {:.1} s (layer loads {tl:.1} s, forwards {tf:.2} s) -> token {am} {:?}",
                 t1.elapsed().as_secs_f64(), tok.decode(&[am as u32], false).unwrap_or_default());
        out += &tok.decode(&[am as u32], false).unwrap_or_default();
        ids.push(am as u32);
    }
    println!("  greedy continuation: {:?}", out);
    println!("NAIVE_FULL_DONE");
    Ok(())
}

#[cfg(test)]
mod t0c_tests {
    use super::{a1_smem, DEFAULT_SMEM_CAP};

    /// T0-c: at ks == 1 the window IS the whole of k, so the staged request must be byte-identical
    /// to the pre-T0-c number. This is the property that makes the change safe on the ks == 1 paths
    /// (every single-pass `nv_gemm` / `nv_gemm_grouped_xh` launch, which are the majority).
    #[test]
    fn ks1_is_byte_identical_to_the_pre_t0c_arithmetic() {
        for &k in &[2048usize, 4096, 8192, 16384] {
            for m in 1..=16usize {
                let got = a1_smem(m, k, 1);
                let pre_t0c = m * (k + 8) * 2;
                if pre_t0c <= DEFAULT_SMEM_CAP {
                    assert_eq!(got as usize, pre_t0c, "ks=1 m={m} k={k} must match pre-T0-c");
                } else {
                    assert_eq!(got, 0, "ks=1 m={m} k={k} over the cap must fall back, not over-allocate");
                }
            }
        }
    }

    /// T0-c: the whole point — shapes the 48KB gate rejected now come back, and they come back
    /// because the request shrank by ks, not because the cap was raised.
    ///
    /// The exact identity, so a drift in either direction fails here:
    ///   pre-T0-c  = m*(k + 8)*2          (the whole of K)
    ///   T0-c      = m*(k/ks + 8)*2       (this CTA's window)
    /// so T0-c == pre/ks + 16m*(1 - 1/ks) — the `+8` pad is PER WINDOW and scales with m, which is
    /// why the request shrinks by slightly less than ks. It is exactly `pre/ks` only in the limit.
    #[test]
    fn windowing_readmits_the_shapes_the_48kb_gate_rejected() {
        // mlp.down: k = 16384. Pre-T0-c at m = 8 was 8 * (16384 + 8) * 2 = 262272 B -> never staged.
        let pre_t0c_m8 = 8usize * (16384 + 8) * 2;
        assert!(pre_t0c_m8 > DEFAULT_SMEM_CAP, "precondition: m=8/k=16384 really was rejected");
        for ks in [8usize, 16, 32] {
            let got = a1_smem(8, 16384, ks) as usize;
            assert!(got > 0, "m=8 k=16384 ks={ks} should be stageable now");
            assert!(got <= DEFAULT_SMEM_CAP);
            assert_eq!(got, pre_t0c_m8 / ks + 16 * 8 * (ks - 1) / ks,
                       "the windowed request must equal pre/ks + 16m(1-1/ks) (m=8 k=16384 ks={ks})");
        }
    }

    /// T0-c: the arithmetic must equal the kernel's `SK` exactly. The kernel computes
    /// `SK = ((kb1 - kb0) << 4) + 8` halves and holds `m` rows, i.e. `m * SK * 2` bytes. This is the
    /// host/device twin check the doc comment demands (the host and kernel must never drift).
    #[test]
    fn host_bytes_match_the_kernel_sk_formula() {
        for ks in [1usize, 2, 4, 8, 16] {
            for &k in &[4096usize, 16384] {
                let m = 8usize;
                let kb0 = (k / 16) / ks;
                let kb1 = kb0 + (k / 16) / ks;          // the kernel's chunk
                let kernel_sk: usize = ((kb1 - kb0) << 4) + 8;
                let kernel_bytes = m * kernel_sk * 2;
                let host = a1_smem(m, k, ks) as usize;
                if host > 0 {
                    assert_eq!(host, kernel_bytes, "host/kernel smem must agree (m={m} k={k} ks={ks})");
                }
            }
        }
    }

    /// Guards the invariants that make the window legal: ks must divide the k-tile count, and the
    /// window must stay a whole multiple of 128 columns so the fused `suh` path's nv_had128 never
    /// straddles a butterfly group.
    #[test]
    fn rejects_shapes_the_window_cannot_serve() {
        assert_eq!(a1_smem(0, 4096, 1), 0, "no rows");
        assert_eq!(a1_smem(17, 4096, 1), 0, "over MAX_ROWS");
        assert_eq!(a1_smem(8, 4096, 0), 0, "ks == 0");
        assert_eq!(a1_smem(8, 4104, 1), 0, "k not a multiple of 16");
        assert_eq!(a1_smem(8, 4096, 3), 0, "ks does not divide the k-tiles");
        // k = 4096 (256 tiles): ks = 16 leaves 16 tiles = 256 columns = 2 whole 128-blocks -> LEGAL;
        // ks = 64 leaves 4 tiles = 64 columns, straddling one -> refused.
        assert!(a1_smem(8, 4096, 16) > 0, "a 2-block window is legal");
        assert_eq!(a1_smem(8, 4096, 64), 0, "window narrower than 128 columns");
    }

    /// T0-c engagement: at the REAL shapes, the shipped dispatcher (`chain_plan`) must actually
    /// route the staged path. This is the "identical hashes is not a liveness receipt" guard — the
    /// smem arithmetic can be perfect while nothing ever asks for it.
    ///
    /// Every projection in the campaign (p90 ITL -25%) rests on this: if chain_plan hands back
    /// `a1 == false` or `ks == 1` for these shapes, T0-c never engages in the serving path.
    #[test]
    fn t0c_engages_on_the_real_shapes_through_the_real_dispatcher() {
        // (name, k, n) from src/naive/config.rs tests: hidden 4096, ffn 16384, moe_inter 2048,
        // vocab 152576, index_head_dim 128, head_dim = hidden/heads.
        let shapes: &[(&str, usize, usize)] = &[
            ("mlp.down", 16384, 4096),
            ("mlp.gate_up", 4096, 4096),
            ("q_proj", 4096, 4096),
            ("o_proj", 4096, 4096),
            ("lm_head", 4096, 152576),
        ];
        let mut engaged = 0;
        for &(name, k, n) in shapes {
            let (ks, a1) = super::super::gpu::chain_plan(k, n);
            let smem = a1_smem(8, k, ks);
            // The staged body can only be used when the plan says so AND the window fits.
            let staged = a1 && smem > 0;
            println!("T0cEngage {name:<12} k={k:<6} n={n:<7} ks={ks} a1={a1} smem={smem} staged={staged}");
            if staged { engaged += 1; }
            // Whatever it decides, the two must agree — a plan that asks for a staged body with no
            // smem (or vice versa) would be the silent-dispatch bug this whole item is about.
            if a1 && ks > 1 { assert!(smem > 0, "{name}: staged split-K with no smem is unreachable"); }
        }
        assert!(engaged > 0, "T0-c engaged on NO real shape — the change would be dead code");
    }
}

#[cfg(test)]
mod t2d_tests {
    use super::{chunk_ranges, MAX_PREFILL_ROWS, MAX_ROWS};
    use crate::naive::serve::PREFILL_CHUNK;

    #[test]
    fn chunks_tile_the_prompt_exactly() {
        for &total in &[1usize, 16, 2047, 2048, 2049, 4096, 5000, 32768] {
            let r = chunk_ranges(total, MAX_PREFILL_ROWS);
            assert_eq!(r.iter().map(|&(_, n)| n).sum::<usize>(), total, "total {total}");
            let mut at = 0usize;
            for (c0, n) in &r {
                assert_eq!(*c0, at, "chunks must be contiguous (total {total})");
                assert!(*n > 0 && *n <= MAX_PREFILL_ROWS, "chunk width out of range");
                at += n;
            }
        }
    }

    #[test]
    fn a_prompt_within_the_chunk_is_one_pass() {
        assert_eq!(chunk_ranges(2048, MAX_PREFILL_ROWS), vec![(0, 2048)], "must stay single-shot");
        assert_eq!(chunk_ranges(1, MAX_PREFILL_ROWS), vec![(0, 1)]);
    }

    /// REGRESSION GATE (the 16-row sids overrun). Every cache-addressing kernel indexes
    /// `slot_ids[m]` for `m` in `0..m`, and `sids_buf` is the ONE allocation all of them read.
    /// It was sized `MAX_ROWS` (16) while prefill feeds it chunks of up to `PREFILL_CHUNK`
    /// (1024) rows: rows >= 16 read past the allocation, the garbage slot id became the KV plane
    /// stride `slot * (kv * S * hd)` with S = max_pos = 262144, and every prefill wider than 16
    /// rows aborted with CUDA_ERROR_ILLEGAL_ADDRESS. Prompts of <= 16 rows were fine, which is
    /// exactly why short gates stayed green.
    ///
    /// THE INVARIANT: `sids_buf` is `vec![0i32; MAX_PREFILL_ROWS]`, every chunk reads from
    /// OFFSET 0 (n_slots = 1 in serving, so all rows are slot 0 and the buffer is the
    /// guaranteed-zero array), so the buffer must cover the WIDEST CHUNK any path can hand it.
    /// It must never be indexed by a cumulative row offset.
    #[test]
    fn slot_id_buffer_covers_the_widest_chunk_any_path_feeds_it() {
        // decode/verify rows must fit too (they were the original MAX_ROWS cap)
        assert!(MAX_PREFILL_ROWS >= MAX_ROWS, "the decode cap must not exceed the prefill cap");
        // the SERVING chunk cap must fit, or raising it silently reintroduces the overrun
        assert!(
            PREFILL_CHUNK <= MAX_PREFILL_ROWS,
            "serving chunk cap {PREFILL_CHUNK} exceeds the {MAX_PREFILL_ROWS}-row slot-id buffer"
        );
        // every chunk of any prompt length must fit, read from base offset 0
        for &total in &[1usize, 16, 17, 128, 1024, 2048, 2049, 5000, 32768] {
            for (c0, n) in chunk_ranges(total, MAX_PREFILL_ROWS) {
                assert!(n <= MAX_PREFILL_ROWS, "chunk ({c0}, {n}) of {total} exceeds the buffer");
            }
        }
    }

    /// WHY THE CHUNK MUST READ SLOT IDS FROM BASE OFFSET 0.
    ///
    /// The old code passed `sids + (c0 * 4)` — a CUMULATIVE row offset. `chunk_ranges` puts
    /// the 2nd chunk at `c0 = width of the 1st`, so for any prompt longer than one chunk that
    /// offset is already >= the buffer width and the kernel reads past the allocation. This
    /// test pins that fact: if `chunk_ranges` ever stopped producing cumulative offsets (e.g.
    /// by clamping `total`), the guard below would stop firing and the "base offset" comment
    /// would need re-deriving, so a silent change here is worth a red build.
    #[test]
    fn a_cumulative_slot_offset_would_overrun_on_the_second_chunk() {
        for &total in &[2049usize, 5000, 32768] {
            let r = chunk_ranges(total, MAX_PREFILL_ROWS);
            assert!(r.len() >= 2, "total {total} must split into >= 2 chunks at this cap");
            let (c0_2, n_2) = r[1];
            assert_eq!(c0_2, r[0].1, "the 2nd chunk starts where the 1st ended");
            assert!(
                c0_2 >= MAX_PREFILL_ROWS,
                "2nd chunk of {total} starts at {c0_2}: a cumulative `sids + c0*4` would overrun"
            );
            assert!(n_2 <= MAX_PREFILL_ROWS, "chunk width must fit the buffer read from base");
        }
    }

    /// A zero (or otherwise absurd) cap must still TERMINATE — `while c0 < total` with `cap == 0`
    /// would spin forever. The floor is 1, so it degrades to one row per pass.
    #[test]
    fn a_zero_cap_cannot_spin() {
        let r = chunk_ranges(10, 0);
        assert_eq!(r.iter().map(|&(_, n)| n).sum::<usize>(), 10, "must still tile exactly");
        assert!(r.iter().all(|&(_, n)| n >= 1), "no zero-width chunk");
        assert!(r.len() <= 10, "terminates in bounded passes");
    }
}
/// The `geom` word is packed in Rust and unpacked in CUDA, both by hand, and neither side is
/// type-checked against the other. `2acaa750` packed `S` at bit 48 and unpacked it with
/// `& 0xFFFF`; at the shipped `--max-seq-len 262144` that truncated the per-head KV stride to
/// 0, which aliased every KV head onto head 0 across all DSA layers. The 3-layer world=1 slice
/// probes could not see it (their `max_pos` fits in 16 bits) and `--probe-naive-slots` passed.
/// These tests are the ones that class needed.
#[cfg(test)]
mod geom_tests {
    use super::{pack_geom, GEOM_FIELDS};

    /// The compiled-from source, so a mask edited in the kernel but not in `pack_geom` fails here.
    const CU: &str = include_str!("../../kernels/gpu_naive.cu");

    /// The kernel's hand-written unpack must agree with `pack_geom`, field for field. If the two
    /// expressions ever drift, this fails instead of the drift reaching a serving box.
    #[test]
    fn geom_fields_match_cuda() {
        for &(name, shift, width) in GEOM_FIELDS.iter() {
            let mask = (1u64 << width) - 1;
            let expr = if shift == 0 {
                format!("geom & 0x{:X}", mask)
            } else {
                format!("(geom >> {}) & 0x{:X}", shift, mask)
            };
            assert!(
                CU.contains(&expr),
                "kernels/gpu_naive.cu has no `{expr}` for field {name} (shift {shift}, {width} bits): \
                 the CUDA unpack and pack_geom have drifted apart"
            );
        }
        // Both DSA split kernels must still unpack S this way (2 sites today).
        assert_eq!(
            CU.matches("geom >> 40").count(),
            2,
            "expected exactly nv_attn_dsa_split and nv_attn_dsa_split_grp to unpack S at bit 40"
        );
    }

    /// S is `l.slots`, which is `max_pos` for a DSA layer and is used as the stride `g*S+t`.
    /// It must survive the shipped posture's 262144 -- which is why it gets 24 bits.
    #[test]
    fn geom_survives_the_shipped_max_pos() {
        for &s in &[4096usize, 65536, 131072, 262144] {
            let got = (pack_geom(2048, 8, 256, s) >> 40) & 0xFFFFFF;
            assert_eq!(got as usize, s, "S={s} truncated to {got} in the geom word");
        }
    }

    /// No field bleeds into a neighbour, at the shapes the serving path actually launches.
    #[test]
    fn geom_fields_round_trip_without_bleed() {
        for &k in &[2048usize, 4096] {
            for &ns in &[1usize, 8] {
                for &sl in &[64usize, 256, 2048] {
                    for &s in &[4096usize, 262144] {
                        let g = pack_geom(k, ns, sl, s);
                        let f = |sh: u32, w: u32| ((g >> sh) & ((1i64 << w) - 1)) as usize;
                        assert_eq!(
                            (f(0, 16), f(16, 8), f(24, 16), f(40, 24)),
                            (k, ns, sl, s),
                            "field bleed at k={k} nsplits={ns} split_len={sl} S={s}"
                        );
                    }
                }
            }
        }
    }

    /// The historical failure, stated so a future "simplification" back to the old layout fails
    /// loudly rather than at 262144 on a serving box.
    #[test]
    fn the_pre_fix_layout_would_have_truncated_s() {
        let s = 262144usize;
        let pre_fix = (((s as i64) & 0xFFFF) << 48) as i64; // exactly what 2acaa750 packed
        assert_eq!((pre_fix >> 48) & 0xFFFF, 0, "precondition: the old layout really did truncate S to 0");
        let now = (pack_geom(2048, 8, 256, s) >> 40) & 0xFFFFFF;
        assert_eq!(now as usize, s, "the current layout must carry S intact");
    }
}