//! Shared support of the vision capture-race tests (S-B9-REL-VISIONFIX / -VISIONFIX2).
//!
//! * ONE process-wide device and ONE lock. Every test of a file takes the lock before it touches
//!   the device and holds it to its end: all tests of a binary share one CUDA primary context, and
//!   one test's legacy-stream set-up or tear-down landing inside another test's capture window is
//!   exactly the hazard these files measure (the default parallel harness aborted or segfaulted
//!   before this lock). The serialization is in code; no thread-count flag or variable is needed.
//! * A machine without a CUDA device, or without `src/ptx/gpu_vision.ptx` (a build has not run), is
//!   a LOUD skip: a banner written straight to the process's stderr (`io::stderr()` bypasses the
//!   harness's output capture, so it shows without `--nocapture`), never a panic and never a
//!   silent pass.
//! * A capture left open by a failing assertion is closed again (`OpenCapture`), so one red test
//!   does not turn the next test red too.
#![allow(dead_code)]

use cudarc::driver::{result, sys, CudaDevice, CudaFunction, CudaSlice, DevicePtr, LaunchConfig};
use cudarc::nvrtc::Ptx;
use gb10_inference::vision_tower::{TowerDims, VisualBlock, VisualTower};
use std::io::Write;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

pub const N: usize = 1024;
pub type Mode = sys::CUstreamCaptureMode;
pub const TL: Mode = Mode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;

pub fn dp<T>(s: &CudaSlice<T>) -> u64 { *s.device_ptr() as u64 }
pub fn cfg() -> LaunchConfig { LaunchConfig { grid_dim: ((N as u32 + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 } }

static SERIAL: Mutex<()> = Mutex::new(());
static DEVICE: OnceLock<Result<(Arc<CudaDevice>, CudaFunction), String>> = OnceLock::new();

/// The device, the tiny kernel to capture (`vision_add_inplace(out, src, n)`: out[i] += src[i]), and
/// the lock that serializes the tests of this binary (held until this value is dropped).
pub struct Gpu {
    _lock: MutexGuard<'static, ()>,
    pub dev: Arc<CudaDevice>,
    pub f: CudaFunction,
}

/// Write straight to the real stderr: the harness captures `eprintln!` of a passing test.
pub fn say(line: &str) {
    let mut e = std::io::stderr();
    let _ = writeln!(e, "{line}");
    let _ = e.flush();
}

/// `Some(Gpu)` with the lock held, or `None` after a loud skip banner naming `test` and the reason.
pub fn gpu(test: &str) -> Option<Gpu> {
    // a failed test poisons the lock; the next test must still run (OpenCapture closed its capture)
    let lock = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let dev = DEVICE.get_or_init(|| {
        let dev = CudaDevice::new(0).map_err(|e| format!("no usable CUDA device: {e:?}"))?;
        let ptx = std::fs::read_to_string("src/ptx/gpu_vision.ptx")
            .map_err(|e| format!("src/ptx/gpu_vision.ptx is missing ({e}); run `cargo build --release` first"))?;
        dev.load_ptx(Ptx::from_src(ptx), "gpu_vision_race", &["vision_add_inplace"]).map_err(|e| format!("load_ptx: {e:?}"))?;
        let f = dev.get_func("gpu_vision_race", "vision_add_inplace").ok_or_else(|| "vision_add_inplace not found".to_string())?;
        Ok((dev, f))
    });
    match dev {
        Ok((dev, f)) => Some(Gpu { _lock: lock, dev: dev.clone(), f: f.clone() }),
        Err(why) => {
            say(&format!("=== SKIPPED {test}: {why} -- this test did NOT run and proves nothing on this machine ==="));
            None
        }
    }
}

/// Ends and discards a capture still open on a stream when dropped (a failing assertion between
/// `cuStreamBeginCapture` and `cuStreamEndCapture`). `disarm` after a normal end.
pub struct OpenCapture { stream: sys::CUstream, armed: bool }
impl OpenCapture {
    pub fn new(stream: sys::CUstream) -> Self { OpenCapture { stream, armed: true } }
    pub fn disarm(&mut self) { self.armed = false; }
}
impl Drop for OpenCapture {
    fn drop(&mut self) {
        if !self.armed { return; }
        unsafe {
            let mut st = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
            if sys::cuStreamIsCapturing(self.stream, &mut st) == sys::CUresult::CUDA_SUCCESS
                && st != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
                let mut g: sys::CUgraph = std::ptr::null_mut();
                let _ = sys::cuStreamEndCapture(self.stream, &mut g);
                if !g.is_null() { let _ = sys::cuGraphDestroy(g); }
            }
            let _ = result::stream::synchronize(self.stream);
        }
    }
}

/// Instantiate `graph`, launch it once on `sa`, destroy it, and check every element of `x`.
pub fn replay_is(dev: &Arc<CudaDevice>, sa: &cudarc::driver::CudaStream, graph: sys::CUgraph, x: &CudaSlice<f32>, want: f32) -> bool {
    unsafe {
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let mut ok = sys::cuGraphInstantiate_v2(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0) == sys::CUresult::CUDA_SUCCESS;
        if ok {
            ok = sys::cuGraphLaunch(exec, sa.stream) == sys::CUresult::CUDA_SUCCESS
                && sys::cuStreamSynchronize(sa.stream) == sys::CUresult::CUDA_SUCCESS
                && dev.dtoh_sync_copy(x).map_or(false, |got| got.iter().all(|&v| v == want));
            sys::cuGraphExecDestroy(exec);
        }
        sys::cuGraphDestroy(graph);
        ok
    }
}

// =====================================================================================
// A small synthetic vision model (random weights, hidden 64, 2 blocks, 4 heads of 16)
// =====================================================================================

pub fn synthetic_tower() -> VisualTower {
    let dims = TowerDims { hidden: 64, depth: 2, inter: 128, heads: 4, num_pos: 64, patch: 2, temporal: 2, merge: 2, in_ch: 3, out_hidden: 32 };
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut w = move |n: usize, scale: f32, bias: f32| -> Vec<f32> {
        (0..n).map(|_| {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            bias + scale * (((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5)
        }).collect()
    };
    let (h, i, mi) = (dims.hidden, dims.inter, dims.merge_inter());
    let blocks = (0..dims.depth).map(|_| VisualBlock {
        norm1_w: w(h, 0.2, 1.0), norm1_b: w(h, 0.1, 0.0), norm2_w: w(h, 0.2, 1.0), norm2_b: w(h, 0.1, 0.0),
        qkv_w: w(3 * h * h, 0.25, 0.0), qkv_b: w(3 * h, 0.1, 0.0),
        proj_w: w(h * h, 0.25, 0.0), proj_b: w(h, 0.1, 0.0),
        fc1_w: w(i * h, 0.25, 0.0), fc1_b: w(i, 0.1, 0.0), fc2_w: w(h * i, 0.25, 0.0), fc2_b: w(h, 0.1, 0.0),
    }).collect();
    VisualTower {
        dims,
        preproc: gb10_inference::vision_preproc::VisionPreprocConfig {
            patch_size: dims.patch, merge_size: dims.merge, temporal_patch_size: dims.temporal, in_channels: dims.in_ch,
            min_pixels: 16, max_pixels: 1 << 20, max_edge: 0,
        },
        image_pad: 0,
        patch_embed_w: w(h * dims.wpv(), 0.5, 0.0), patch_embed_b: w(h, 0.1, 0.0),
        pos_embed_w: w(dims.num_pos * h, 0.5, 0.0),
        blocks,
        merger_norm_w: w(h, 0.2, 1.0), merger_norm_b: w(h, 0.1, 0.0),
        merger_fc1_w: w(mi * mi, 0.1, 0.0), merger_fc1_b: w(mi, 0.1, 0.0),
        merger_fc2_w: w(dims.out_hidden * mi, 0.1, 0.0), merger_fc2_b: w(dims.out_hidden, 0.1, 0.0),
        source: "synthetic (tests/vision_race_common)".into(),
    }
}

/// Deterministic pixel values for a `gh` x `gw` patch grid of `tower`.
pub fn pixels(tower: &VisualTower, gh: usize, gw: usize) -> Vec<f32> {
    (0..gh * gw * tower.dims.wpv()).map(|i| ((i * 37 + gh * 11 + gw) % 255) as f32 / 127.5 - 1.0).collect()
}

pub const GRIDS: [(usize, usize); 3] = [(4, 4), (6, 4), (8, 8)];
