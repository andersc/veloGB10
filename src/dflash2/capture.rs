//! S4F — the DFlash2 trunk tap capture (K-DF2-2, the ONLY trunk edit).
//!
//! The trunk (the 64-layer Qwen3.5 target, `GpuModel`) copies `hidden_states[l+1]` — the
//! post-FFN-add residual of target layer `l` — for `l ∈ TAP_LAYERS = [5,19,33,47,61]`
//! (hidden-state indices 6,20,34,48,62; DECISION A) into a drafter-owned staging buffer at
//! DRAFT TIME ONLY. Flag-gated DEFAULT-OFF (`--df2-capture` rides TpConfig; the single-process
//! probe sets it directly); the flag-off path is a dead `Option::is_none()` — zero launches,
//! zero host work (the R1 timing-free proof).
//!
//! # Buffer layout (the fc mirror's row-major [25600] tap convention)
//!
//! `staging` is `[TAP_CONCAT_DIM=25600, BLOCK=8]` bf16, col-major (the engine convention):
//! ONE COLUMN PER COMMITTED POSITION, 5 layer-rows of 5120 stacked. fc then consumes it as
//! x[m*K+k] (gemm_dsp's col-major activation convention) with K=25600 = the oracle's concat
//! order (layer-major within a row) — no repack between capture and fc.
//!
//! # Capture semantics (bit-identity, DECISION A)
//!
//! The trunk's residual at the end of layer `l` IS `hidden_states[l+1]` (transformers returns
//! [embed_out, after_layer_0, …, after_layer_63]; after_layer_l = the residual AFTER layer l's
//! FFN add). The capture is a pure D2D bf16 copy: what the staging holds is BIT-IDENTICAL to the
//! source residual. The probe proves it against an independent recomputation path (two trunk
//! forwards over the same tokens; captured vs recaptured AND captured vs an independent
//! surface read of the same residual), and the no-capture path is bit-identical trunk behavior
//! (the R1 gates: binv + LOSSLESS ×5 + timing noise).
//!
//! # Threading model
//!
//! The trunk writes ONE COLUMN per call (decode n=1) or the last ≤8 columns (verify n≤8):
//! `forward_batch_dev` calls [`capture_cols`] with its live residual pointer. The drafter
//! consumes WHOLE columns at round time (fc reads staging cols 0..nprev). A decode's col-0
//! write and a verify's cols-0..n-1 write of the same committed span overwrite with identical
//! values (same source row, same positions) — the R1 probe asserts this too (decode-capture vs
//! verify-capture of the same span are bit-identical).

use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DeviceSlice, LaunchAsync};
use std::sync::Arc;

use crate::dflash2::{block, HIDDEN, TAP_CONCAT_DIM, TAP_LAYERS};

/// The DFlash2 tap sink: drafter-owned staging, written by the trunk at draft time only.
///
/// `staging` `[25600, 8]` col-major bf16: element (k, m) at `m*25600 + k`. One sink per
/// process; the trunk holds `Option<Arc<Df2TapSink>>` and the drafter holds the twin `Arc`.
pub struct Df2TapSink {
    pub staging: CudaSlice<half::bf16>,
}

impl Df2TapSink {
    /// Allocate + explicitly zero the staging buffer (a fresh `alloc_zeros` does NOT zero —
    /// AGENTS §2.2; a partial round's never-written columns must read as 0, and the first
    /// draft round may run with C < 8 committed positions).
    pub fn new(dev: &Arc<CudaDevice>) -> Self {
        Self::new_cols(dev, block())
    }
    /// PLAN/25 Phase 1: a WIDE staging for the tree-verify capture (`cols` up to MAX_VERIFY —
    /// a tree verifies more columns than the chain's 8; the accepted path is gathered out of
    /// this buffer into the round's 8-column staging by `sync_staging_from_wide`).
    pub fn new_cols(dev: &Arc<CudaDevice>, cols: usize) -> Self {
        let n = TAP_CONCAT_DIM * cols;
        let zero = dev.htod_sync_copy(&vec![half::bf16::default(); n]).expect("df2 tap staging alloc");
        Df2TapSink { staging: zero }
    }
}

/// S5F — the DFlash2 PROMPT-PRIME sink: a wider tap buffer sized for one prefill window
/// (`[25600, cols]`), written by the trunk's PREFILL path at the tap layers (the prefill
/// cannot use the 8-column staging — a window is up to `PREFILL_CHUNK` columns). The round's
/// `prime_window` consumes it window by window. One per process; the trunk holds
/// `Option<Arc<Df2PrimeSink>>` and the round's caller reads it. `alloc_zeros` does NOT zero —
/// every column the prime reads must have been written by a prefill window first.
pub struct Df2PrimeSink {
    pub taps: CudaSlice<half::bf16>,
}

impl Df2PrimeSink {
    pub fn new(dev: &Arc<CudaDevice>, cols: usize) -> Self {
        let n = TAP_CONCAT_DIM * cols;
        let zero = dev.htod_sync_copy(&vec![half::bf16::default(); n]).expect("df2 prime sink alloc");
        Df2PrimeSink { taps: zero }
    }
}

/// The raw tap-layer copy, parameterized by the destination (the 8-column sink or the wide
/// prime sink): one tap-layer's rows of the trunk's post-FFN residual `[h, n]` bf16 col-major
/// (element (c, m) at `m*h + c`) → `dst_ptr` rows `[tap_li*h, (tap_li+1)*h)` for columns
/// `[0, n)`, with `dst_pitch` = the destination's column stride in ELEMENTS. Stream-ordered
/// 2D D2D on the caller's (compute) stream — AGENTS §2.3; no sync (the copy is async; the
/// drafter's round-time read is stream-ordered after it).
pub fn capture_cols_into(dev: &Arc<CudaDevice>, stream: cudarc::driver::sys::CUstream,
                         dst_ptr: u64, dst_pitch: usize,
                         src_ptr: u64, h: usize, n: usize, tap_li: usize) {
    use cudarc::driver::sys;
    debug_assert_eq!(h, HIDDEN, "df2 tap capture: trunk hidden {h} != DFlash2 hidden {HIDDEN}");
    debug_assert!(n <= dst_pitch, "df2 tap capture cols {n} > dst pitch {dst_pitch}");
    debug_assert!(tap_li < TAP_LAYERS.len());
    let cp = sys::CUDA_MEMCPY2D {
        srcXInBytes: 0, srcY: 0,
        srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(), srcDevice: src_ptr,
        srcArray: std::ptr::null_mut(), srcPitch: h * 2,
        dstXInBytes: (tap_li * h) * 2, dstY: 0,
        dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(), dstDevice: dst_ptr,
        dstArray: std::ptr::null_mut(), dstPitch: dst_pitch * 2,
        WidthInBytes: h * 2, Height: n,
    };
    unsafe {
        let r = sys::cuMemcpy2DAsync_v2(&cp, stream);
        assert!(r == sys::CUresult::CUDA_SUCCESS, "df2 tap capture D2D failed: {r:?}");
    }
}

/// PLAN/25 Phase 1: one full staging ROW (all `TAP_LAYERS·HIDDEN` features of one token) —
/// the tree gather's per-token copy. One driver call per token instead of one per (token,
/// layer): 5× fewer `cuMemcpy2D` issues for the same bytes; both pitches are the row stride.
pub fn copy_row_into(dev: &Arc<CudaDevice>, stream: cudarc::driver::sys::CUstream,
                     dst_ptr: u64, src_ptr: u64, row_elems: usize) {
    use cudarc::driver::sys;
    let cp = sys::CUDA_MEMCPY2D {
        srcXInBytes: 0, srcY: 0,
        srcMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcHost: std::ptr::null(), srcDevice: src_ptr,
        srcArray: std::ptr::null_mut(), srcPitch: row_elems * 2,
        dstXInBytes: 0, dstY: 0,
        dstMemoryType: sys::CUmemorytype::CU_MEMORYTYPE_DEVICE,
        dstHost: std::ptr::null_mut(), dstDevice: dst_ptr,
        dstArray: std::ptr::null_mut(), dstPitch: row_elems * 2,
        WidthInBytes: row_elems * 2, Height: 1,
    };
    unsafe {
        let r = sys::cuMemcpy2DAsync_v2(&cp, stream);
        assert!(r == sys::CUresult::CUDA_SUCCESS, "df2 row copy D2D failed: {r:?}");
    }
}

/// One tap-layer copy into the 8-column staging sink (the S4F layout — see the struct docs).
pub fn capture_cols(dev: &Arc<CudaDevice>, stream: cudarc::driver::sys::CUstream,
                    sink: &Df2TapSink, res_ptr: u64, h: usize, n: usize, tap_li: usize) {
    capture_cols_into(dev, stream, *sink.staging.device_ptr() as u64, TAP_CONCAT_DIM,
                      res_ptr, h, n, tap_li)
}

// EXL27C-DF2 (fwd): the PITCH-AWARE, DTYPE-CONVERTING tap capture for the EXL3 serve path.
//
// DESIGN CHOICE (documented per the exl27c_fwd brief): `capture_cols_into` above is a
// cuMemcpy2DAsync bf16→bf16 D2D — it CANNOT convert element types, and the EXL3 forward
// stages its residual as f32, token-major `[rows][pitch]` (`xq_hc_norm`: stream s of
// token b at b·pitch + s·h; pitch = hcn·h), so the EXL3 capture is a small KERNEL
// instead of a memcpy: `xq_df2_tap_cvt` (kernels/exl3_bench.cu) reads the live-residual
// slice `[r·pitch + 0, r·pitch + h)` of each token row — STREAM 0; for the 27B packs
// (Family::Qwen35, hc_count = 1) that is the WHOLE row; the hcn>1 case reads the same
// stream-0 slice the NVFP4 trunk's tap reads (gpu.rs passes h — not resid_width() — as
// the source stride) — and writes the dense bf16 tap slice `dst[r·dst_pitch + tap_li·h,
// +h)`. One launch per tap layer, stream-ordered on the caller's (compute) stream
// (AGENTS §2.3), no host reads, fixed device addresses → CUDA-graph-safe.
//
// The kernel lives in the exl3_bench module (the EXL3 kernel book); this function fetches
// it by name, so it requires that module to be loaded (any FwdModel path does).
pub fn capture_cols_f32_into(dev: &Arc<CudaDevice>, stream: &cudarc::driver::CudaStream,
                             dst_ptr: u64, dst_pitch: usize, src_ptr: u64, src_pitch: usize,
                             h: usize, n: usize, tap_li: usize) -> anyhow::Result<()> {
    use cudarc::driver::LaunchConfig;
    anyhow::ensure!(h > 0 && n > 0, "df2 f32 tap: empty capture (h={h}, n={n})");
    anyhow::ensure!(src_pitch >= h, "df2 f32 tap: src pitch {src_pitch} < h {h}");
    anyhow::ensure!(h == HIDDEN, "df2 f32 tap: h {h} != HIDDEN {HIDDEN} (non-27B pack?)");
    let elems = (h as i64) * (n as i64);
    let f = dev.get_func("exl3_bench", "xq_df2_tap_cvt")
        .ok_or_else(|| anyhow::anyhow!("df2 f32 tap: xq_df2_tap_cvt not loaded (is the exl3_bench module loaded?)"))?;
    let blocks = (((elems + 255) / 256).min(65535) as u32, 1u32, 1u32);
    unsafe {
        f.clone().launch_on_stream(
            stream,
            LaunchConfig { grid_dim: blocks, block_dim: (256, 1, 1), shared_mem_bytes: 0 },
            (dst_ptr as u64, src_ptr as u64, elems, h as i32, src_pitch as i64,
             tap_li as i32, dst_pitch as i64),
        )?;
    }
    Ok(())
}

/// The TAP-sink wrapper (the 8-column staging; dst pitch = TAP_CONCAT_DIM): the EXL3
/// mirror of `capture_cols` for the decode/verify hooks. `n` must fit the sink's column
/// count (`n <= block()` for the standard sink — the caller gates, same as the trunk).
pub fn capture_cols_f32(dev: &Arc<CudaDevice>, stream: &cudarc::driver::CudaStream,
                        sink: &Df2TapSink, res_ptr: u64, src_pitch: usize,
                        h: usize, n: usize, tap_li: usize) -> anyhow::Result<()> {
    anyhow::ensure!(n <= sink.staging.len() / TAP_CONCAT_DIM,
                    "df2 f32 tap: n {n} > sink columns {}", sink.staging.len() / TAP_CONCAT_DIM);
    capture_cols_f32_into(dev, stream, *sink.staging.device_ptr() as u64, TAP_CONCAT_DIM,
                          res_ptr, src_pitch, h, n, tap_li)
}

/// The PRIME-sink wrapper (one prefill window): writes rows [0, n) of the prime sink —
/// trunk-identical (gpu.rs's S5F prefill capture) — for `prime_window(n, pos_start)` to
/// consume per chunk. The prefill is eager, so this is outside any captured graph.
pub fn capture_prime_f32(dev: &Arc<CudaDevice>, stream: &cudarc::driver::CudaStream,
                         prime: &Df2PrimeSink, res_ptr: u64, src_pitch: usize,
                         h: usize, n: usize, tap_li: usize) -> anyhow::Result<()> {
    anyhow::ensure!(n <= prime.taps.len() / TAP_CONCAT_DIM,
                    "df2 prime tap: n {n} > prime columns {}", prime.taps.len() / TAP_CONCAT_DIM);
    capture_cols_f32_into(dev, stream, *prime.taps.device_ptr() as u64, TAP_CONCAT_DIM,
                          res_ptr, src_pitch, h, n, tap_li)
}

