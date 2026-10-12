//! S-B9-REL-VISIONFIX F1 / S-B9-REL-VISIONFIX2 V1+V2: what a SECOND thread may do on a CUDA device
//! while another thread is capturing a graph on a blocking stream, and the GPU vision tower under
//! exactly that condition.
//!
//! The served failure (PLAN/REL_TP2VERIFY_REPORT.md F1): the GPU vision tower, on a server worker
//! thread, was told `CUDA_ERROR_STREAM_CAPTURE_IMPLICIT` while the scheduler thread was capturing a
//! decode graph, and the request fell back to the CPU tower (173..261 s).
//!
//! Part A is the mechanism at unit scale (no model, a few kB of device memory): thread A opens a
//! capture on its own blocking stream and records launches; inside the window thread B performs ONE
//! operation; A ends the capture, replays the graph and checks its numbers. Every statement the
//! fix relies on is an assertion here. It includes the cuBLASLt calls of the `pfx1-ltplans`
//! thread (`lt_prewarm_start`, src/exl3_forward.rs).
//!
//! Part B runs the REAL `GpuVisualTower` (a small synthetic model: random weights, 2 blocks) inside
//! another thread's capture window and against a free-running capture loop.
//!
//! Part C holds the instruments (ignored by default). Three of them END IN A SEGFAULT INSIDE THE
//! CUDA DRIVER by design: that is their finding. Run one at a time, by name, with `--ignored`.
//!
//! SERIALIZATION. Every test takes one file-wide lock (`vision_race_common::gpu`): all tests share
//! one CUDA primary context, and one test's legacy-stream set-up or tear-down inside another test's
//! capture window crashes the process. The default harness (parallel tests) is therefore safe:
//! `cargo test --release --test vision_capture_race_test`. No thread-count flag or variable.
//! A machine without a CUDA device or without `src/ptx/gpu_vision.ptx` prints a `=== SKIPPED ...`
//! banner per test on stderr and runs nothing; that is not a pass.
//!
//! WHAT "RED ON MASTER" MEANS. The reproducer is `the_gpu_tower_forward_inside_another_threads_
//! capture_window` (Part B). Against master's `src/vision_gpu.rs` (the tower that uploads through
//! cudarc's legacy-stream helpers and the shared `Pool`) it FAILS, deterministically, in about half
//! a second, with the served error: `the GPU tower failed inside another thread's capture window:
//! DriverError(CUDA_ERROR_STREAM_CAPTURE_IMPLICIT, "operation would make the legacy stream depend
//! on a capturing blocking stream")`. This file uses only `GpuVisualTower::new` and `forward`, so
//! it builds against master unchanged. To reproduce, in a clean worktree of this branch:
//!     git show master:src/vision_gpu.rs > src/vision_gpu.rs
//!     cargo test --release --test vision_capture_race_test \
//!         the_gpu_tower_forward_inside_another_threads_capture_window
//!     git checkout src/vision_gpu.rs
//! Run only that test against master: its neighbours exercise master's tower in a free-running
//! race, where the tower's own `unwrap`s abort the process (that IS the other half of F1).
//! Parts A and C do not touch the tower and are green on both.

mod vision_race_common;

use cudarc::cublas::{sys::cublasOperation_t as OP, CudaBlas, Gemm, GemmConfig};
use cudarc::cublaslt::{result as lt, sys as ltsys};
use cudarc::driver::{result, sys, CudaDevice, CudaFunction, CudaSlice, DevicePtr, DevicePtrMut, LaunchAsync};
use gb10_inference::gpu::fork_blocking_stream;
use gb10_inference::vision_gpu::GpuVisualTower;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use vision_race_common::*;

/// Sets a flag when dropped, also while unwinding: a thread that waits on `stop` is released when the
/// other thread panics (review M1: a failing assertion used to leave the other side spinning forever with the
/// file-wide lock held, so the binary hung instead of reporting the regression it guards).
struct SetOnDrop<'a>(&'a AtomicBool);
impl Drop for SetOnDrop<'_> { fn drop(&mut self) { self.0.store(true, Relaxed); } }

/// Liveness cap, not an assertion on speed: if thread B has not returned this long into a window,
/// A closes the window and the outcome is reported as "B blocked" (a refusal that blocks instead of
/// returning would be a deadlock hazard, and that is a finding).
const LIVENESS_CAP: Duration = Duration::from_secs(30);

fn code<T>(r: Result<T, result::DriverError>) -> Result<(), String> { r.map(|_| ()).map_err(|e| format!("{:?}", e.0)) }
fn lt_code<T>(r: Result<T, lt::CublasError>) -> Result<(), String> { r.map(|_| ()).map_err(|e| format!("{:?}", e.0)) }
fn show(r: &Result<(), String>) -> String { match r { Ok(()) => "ok".into(), Err(e) => e.clone() } }

fn mode_name(m: Mode) -> &'static str {
    match m {
        Mode::CU_STREAM_CAPTURE_MODE_GLOBAL => "GLOBAL",
        Mode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL => "THREAD_LOCAL",
        Mode::CU_STREAM_CAPTURE_MODE_RELAXED => "RELAXED",
    }
}

// =====================================================================================
// cuBLASLt, as the `pfx1-ltplans` thread uses it (src/exl3_forward.rs `lt_plan_build`)
// =====================================================================================

/// One plan of the served prefill, built and torn down: descriptor, attributes, preference, layouts,
/// heuristic. Host-side calls only, exactly the engine's sequence. Err = the first failing step.
fn lt_plan(handle: ltsys::cublasLtHandle_t, k: usize, n: usize, m: usize, ta: bool) -> Result<(), String> {
    let f16 = ltsys::cudaDataType_t::CUDA_R_16F;
    let desc = lt::create_matmul_desc(ltsys::cublasComputeType_t::CUBLAS_COMPUTE_32F, ltsys::cudaDataType_t::CUDA_R_32F)
        .map_err(|e| format!("create_matmul_desc {:?}", e.0))?;
    let (ta_v, tb_v): (u32, u32) = (ta as u32, 0);
    let pref = lt::create_matmul_pref().map_err(|e| format!("create_matmul_pref {:?}", e.0))?;
    let ws: usize = 32 << 20;
    let layouts = [if ta { lt::create_matrix_layout(f16, k as u64, n as u64, k as i64) } else { lt::create_matrix_layout(f16, n as u64, k as u64, n as i64) },
                   lt::create_matrix_layout(f16, k as u64, m as u64, k as i64),
                   lt::create_matrix_layout(f16, n as u64, m as u64, n as i64)];
    let r: Result<(), String> = (|| unsafe {
        lt::set_matmul_desc_attribute(desc, ltsys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, &ta_v as *const _ as *const _, 4)
            .map_err(|e| format!("set TRANSA {:?}", e.0))?;
        lt::set_matmul_desc_attribute(desc, ltsys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB, &tb_v as *const _ as *const _, 4)
            .map_err(|e| format!("set TRANSB {:?}", e.0))?;
        lt::set_matmul_pref_attribute(pref, ltsys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                      &ws as *const _ as *const _, 8).map_err(|e| format!("set MAX_WORKSPACE {:?}", e.0))?;
        let (a, b, c) = match &layouts { [Ok(a), Ok(b), Ok(c)] => (*a, *b, *c), _ => return Err("create_matrix_layout failed".into()) };
        lt::get_matmul_algo_heuristic(handle, desc, a, b, c, c, pref).map(|_| ()).map_err(|e| format!("heuristic {:?}", e.0))
    })();
    unsafe {
        let _ = lt::destroy_matmul_pref(pref);
        for l in layouts.iter().flatten() { let _ = lt::destroy_matrix_layout(*l); }
        let _ = lt::destroy_matmul_desc(desc);
    }
    r
}

/// The shapes `lt_plan` cycles over (the 3.05 pack's prefill families, a few chunk row counts).
const LT_SHAPES: [(usize, usize, usize, bool); 4] = [(2048, 2048, 256, true), (2048, 6144, 512, false), (4096, 2048, 1024, true), (2048, 512, 2048, false)];

// =====================================================================================
// Part A — the mechanism
// =====================================================================================

/// What thread B does inside A's capture window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    // ---- cudarc's "sync" helpers: every one of them goes through the legacy (NULL) stream
    HtodSyncCopy,     // dev.htod_sync_copy: cuMemAllocAsync(NULL) + cuMemcpyHtoDAsync(NULL) + cuStreamSynchronize(NULL)
    AllocNull,        // dev.alloc: cuMemAllocAsync(NULL)
    HtodSyncInto,     // cuMemcpyHtoDAsync(NULL) + cuStreamSynchronize(NULL), into an existing buffer
    DtohSync,         // cuMemcpyDtoHAsync(NULL) + cuStreamSynchronize(NULL)
    MemsetNull,       // dev.memset_zeros: cuMemsetD8Async(NULL)
    DevSync,          // dev.synchronize: cuStreamSynchronize(NULL)
    DropSlice,        // CudaSlice::drop: cuMemFreeAsync(NULL)
    MemcpyHtodSync,   // cuMemcpyHtoD_v2 (the truly synchronous copy: also legacy-stream ordered)
    // ---- driver calls with no stream at all
    MallocSync,       // cuMemAlloc_v2
    FreeSync,         // cuMemFree_v2
    // ---- the same work kept on B's OWN blocking stream
    LaunchOwn,        // a kernel launch
    GemmOwn,          // cuBLAS sgemm, the handle pinned to B's stream
    HtodAsyncOwn,     // cuMemcpyHtoDAsync(own)
    DtohAsyncOwn,     // cuMemcpyDtoHAsync(own)
    MemsetOwn,        // cuMemsetD8Async(own)
    AllocOwn,         // cuMemAllocAsync(own)
    FreeOwn,          // cuMemFreeAsync(own)
    SyncOwn,          // cuStreamSynchronize(own)
    // ---- what the `pfx1-ltplans` thread does (host-side cuBLASLt work, no stream argument anywhere)
    LtCreateHandle,   // cublasLtCreate
    LtDestroyHandle,  // cublasLtDestroy
    LtPlanBuild,      // one plan: descriptor + attributes + preference + layouts + heuristic (+ their destruction)
    BindThread,       // cuCtxSetCurrent(primary context) on a thread that had none (the thread's first CUDA call)
    // ---- constructors and destructors (the tower's `new` / `Drop`, every `CudaBlas` and module load)
    BlasCreate,       // CudaBlas::new (cublasCreate_v2)
    BlasDestroy,      // drop of a CudaBlas (cublasDestroy_v2)
    ModuleLoad,       // dev.load_ptx (cuModuleLoadData)
    FirstLaunch,      // the FIRST launch of a kernel of a module loaded before the window (lazy loading
                      // defers the function's code to this launch: the tower never warms up at boot)
}

/// Everything the tower did before the fix, one way or another.
const LEGACY_OPS: [Op; 8] = [Op::HtodSyncCopy, Op::AllocNull, Op::HtodSyncInto, Op::DtohSync, Op::MemsetNull, Op::DevSync,
                             Op::DropSlice, Op::MemcpyHtodSync];
const NO_STREAM_OPS: [Op; 2] = [Op::MallocSync, Op::FreeSync];
/// Everything the tower does after it.
const OWN_OPS: [Op; 8] = [Op::LaunchOwn, Op::GemmOwn, Op::HtodAsyncOwn, Op::DtohAsyncOwn, Op::MemsetOwn, Op::AllocOwn, Op::FreeOwn, Op::SyncOwn];
/// The cuBLASLt pre-population thread.
const LT_OPS: [Op; 4] = [Op::LtCreateHandle, Op::LtDestroyHandle, Op::LtPlanBuild, Op::BindThread];
/// Constructors / destructors of library handles and modules.
const CTOR_OPS: [Op; 4] = [Op::BlasCreate, Op::BlasDestroy, Op::ModuleLoad, Op::FirstLaunch];

#[derive(Debug, Clone)]
struct Outcome {
    b: Result<(), String>,           // what thread B's call returned
    b_ms: f64,                       // how long it took (informational only: no test asserts on it)
    b_blocked: bool,                 // B had not returned LIVENESS_CAP into the window (A then closed it)
    late_launch: Option<String>,     // the first error of A's launches recorded AFTER B's call
    end_capture: Result<(), String>, // cuStreamEndCapture on A
    replay_ok: bool,                 // the captured graph, replayed, computed x = 1 + 2 per launch
}

/// One capture window on thread A (`before` launches, then B's call, then `after` launches) with
/// one operation of thread B inside it. B is told to go only once A sits idle inside the window,
/// so the two never run a driver call at the same instant: this is the ORDERED experiment.
fn probe(dev: &Arc<CudaDevice>, f: &CudaFunction, mode: Mode, op: Op, before: usize, after: usize) -> Outcome {
    let (b_ready_tx, b_ready_rx) = mpsc::channel::<()>();
    let (in_capture_tx, in_capture_rx) = mpsc::channel::<()>();
    let (b_done_tx, b_done_rx) = mpsc::channel::<(Result<(), String>, f64)>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    // `move`: A's channel ends belong to this closure and drop when it panics or returns, which releases B's
    // blocking `recv().unwrap()` (B then panics and the scope can join) instead of deadlocking the join
    std::thread::scope(move |sc| {
        let devb = dev.clone();
        let fb = f.clone();
        sc.spawn(move || {
            devb.bind_to_thread().unwrap();
            let sb = fork_blocking_stream(&devb);
            let blas = CudaBlas::new(devb.clone()).unwrap();
            unsafe { blas.set_stream(Some(&sb)).unwrap(); }
            let host = vec![5f32; N];
            let mut host_out = vec![0f32; N];
            let mut pre = devb.htod_sync_copy(&vec![3f32; N]).unwrap();
            let pre2 = devb.htod_sync_copy(&vec![4f32; N]).unwrap();
            let mut doomed = Some(devb.htod_sync_copy(&vec![9f32; N]).unwrap());
            let (ga, gb) = (devb.htod_sync_copy(&vec![1f32; 64 * 64]).unwrap(), devb.htod_sync_copy(&vec![1f32; 64 * 64]).unwrap());
            let mut gc = devb.htod_sync_copy(&vec![0f32; 64 * 64]).unwrap();
            let own = unsafe { result::malloc_async(sb.stream, N * 4).unwrap() };
            let sync_ptr = unsafe { result::malloc_sync(N * 4).unwrap() };
            let mut keep: Vec<CudaSlice<f32>> = Vec::new();
            let (mut own_extra, mut sync_extra) = (None, None);
            let (mut freed_own, mut freed_sync) = (false, false);
            // cuBLASLt: a handle that exists before the window (destroyed inside it by LtDestroyHandle)
            let lt_pre = lt::create_handle().expect("cublasLtCreate before the window");
            let (mut lt_made, mut lt_destroyed) = (None, false);
            // a cuBLAS handle that exists before the window (dropped inside it by BlasDestroy)
            let mut blas_pre = Some(CudaBlas::new(devb.clone()).expect("cublasCreate before the window"));
            let mut blas_made: Option<CudaBlas> = None;
            // a module loaded and its function fetched before the window, launched for the first time inside it
            let lazy_fn = std::fs::read_to_string("src/ptx/gpu_vision.ptx").ok().and_then(|src| {
                devb.load_ptx(cudarc::nvrtc::Ptx::from_src(src), "gpu_vision_race_lazy", &["vision_add_inplace"]).ok()?;
                devb.get_func("gpu_vision_race_lazy", "vision_add_inplace")
            });
            devb.synchronize().unwrap();
            unsafe { result::stream::synchronize(sb.stream).unwrap(); }
            if op == Op::BindThread {
                // a thread with no context yet: the op below is its first CUDA call
                unsafe { assert_eq!(sys::cuCtxSetCurrent(std::ptr::null_mut()), sys::CUresult::CUDA_SUCCESS); }
            }
            b_ready_tx.send(()).unwrap();
            in_capture_rx.recv().unwrap();
            let t0 = Instant::now();
            let r: Result<(), String> = unsafe {
                match op {
                    Op::HtodSyncCopy => devb.htod_sync_copy(&host).map(|s| keep.push(s)).map_err(|e| format!("{:?}", e.0)),
                    Op::AllocNull => devb.alloc::<f32>(N).map(|s| keep.push(s)).map_err(|e| format!("{:?}", e.0)),
                    Op::HtodSyncInto => code(devb.htod_sync_copy_into(&host, &mut pre)),
                    Op::DtohSync => code(devb.dtoh_sync_copy_into(&pre, &mut host_out)),
                    Op::MemsetNull => code(devb.memset_zeros(&mut pre)),
                    Op::DevSync => code(devb.synchronize()),
                    // CudaSlice::drop unwraps its cuMemFreeAsync: make the same call by hand to see the code
                    Op::DropSlice => code(result::free_async(doomed.take().unwrap().leak(), std::ptr::null_mut())),
                    Op::MemcpyHtodSync => code(result::memcpy_htod_sync(*pre.device_ptr_mut(), &host)),
                    Op::MallocSync => result::malloc_sync(N * 4).map(|p| sync_extra = Some(p)).map_err(|e| format!("{:?}", e.0)),
                    Op::FreeSync => { freed_sync = true; code(result::free_sync(sync_ptr)) }
                    Op::LaunchOwn => code(fb.clone().launch_on_stream(&sb, cfg(), (dp(&pre), dp(&pre2), N as i32))),
                    Op::GemmOwn => {
                        let c = GemmConfig::<f32> { transa: OP::CUBLAS_OP_N, transb: OP::CUBLAS_OP_N, m: 64, n: 64, k: 64,
                                                    alpha: 1.0, lda: 64, ldb: 64, beta: 0.0, ldc: 64 };
                        blas.gemm(c, &ga, &gb, &mut gc).map_err(|e| format!("{e:?}"))
                    }
                    Op::HtodAsyncOwn => code(result::memcpy_htod_async(own, &host, sb.stream)),
                    Op::DtohAsyncOwn => code(result::memcpy_dtoh_async(&mut host_out, own, sb.stream)),
                    Op::MemsetOwn => code(result::memset_d8_async(own, 0, N * 4, sb.stream)),
                    Op::AllocOwn => result::malloc_async(sb.stream, N * 4).map(|p| own_extra = Some(p)).map_err(|e| format!("{:?}", e.0)),
                    Op::FreeOwn => { freed_own = true; code(result::free_async(own, sb.stream)) }
                    Op::SyncOwn => code(result::stream::synchronize(sb.stream)),
                    Op::LtCreateHandle => lt::create_handle().map(|h| lt_made = Some(h)).map_err(|e| format!("{:?}", e.0)),
                    Op::LtDestroyHandle => { lt_destroyed = true; lt_code(lt::destroy_handle(lt_pre)) }
                    Op::LtPlanBuild => lt_plan(lt_pre, 2048, 2048, 256, true),
                    Op::BindThread => devb.bind_to_thread().map_err(|e| format!("{:?}", e.0)),
                    Op::BlasCreate => CudaBlas::new(devb.clone()).map(|b| blas_made = Some(b)).map_err(|e| format!("{e:?}")),
                    Op::BlasDestroy => { drop(blas_pre.take()); Ok(()) }
                    Op::FirstLaunch => match &lazy_fn {
                        Some(lf) => code(lf.clone().launch_on_stream(&sb, cfg(), (dp(&pre), dp(&pre2), N as i32))),
                        None => Err("could not load the lazy module before the window".to_string()),
                    },
                    Op::ModuleLoad => std::fs::read_to_string("src/ptx/gpu_vision.ptx").map_err(|e| format!("{e}")).and_then(|src| {
                        devb.load_ptx(cudarc::nvrtc::Ptx::from_src(src), "gpu_vision_race_window", &["vision_add_inplace"]).map_err(|e| format!("{:?}", e.0))
                    }),
                }
            };
            b_done_tx.send((r, t0.elapsed().as_secs_f64() * 1e3)).unwrap();
            // everything below frees memory through the legacy stream: only after A has closed its capture
            release_rx.recv().unwrap();
            if op == Op::BindThread { let _ = devb.bind_to_thread(); }
            unsafe {
                let _ = result::stream::synchronize(sb.stream);
                if !freed_own { let _ = result::free_async(own, sb.stream); }
                if let Some(p) = own_extra { let _ = result::free_async(p, sb.stream); }
                if !freed_sync { let _ = result::free_sync(sync_ptr); }
                if let Some(p) = sync_extra { let _ = result::free_sync(p); }
                if let Some(h) = lt_made { let _ = lt::destroy_handle(h); }
                if !lt_destroyed { let _ = lt::destroy_handle(lt_pre); }
                drop((blas_pre.take(), blas_made.take()));
                let _ = result::stream::synchronize(sb.stream);
            }
            drop((keep, doomed, pre, pre2, ga, gb, gc));
        });

        dev.bind_to_thread().unwrap();
        let sa = fork_blocking_stream(dev);
        let x = dev.htod_sync_copy(&vec![1f32; N]).unwrap();
        let y = dev.htod_sync_copy(&vec![2f32; N]).unwrap();
        dev.synchronize().unwrap();
        b_ready_rx.recv().unwrap();
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        assert_eq!(unsafe { sys::cuStreamBeginCapture_v2(sa.stream, mode) }, sys::CUresult::CUDA_SUCCESS, "begin capture");
        let mut open = OpenCapture::new(sa.stream);
        for _ in 0..before {
            unsafe { f.clone().launch_on_stream(&sa, cfg(), (dp(&x), dp(&y), N as i32)).expect("captured launch"); }
        }
        in_capture_tx.send(()).unwrap();
        let (b, b_ms, b_blocked) = match b_done_rx.recv_timeout(LIVENESS_CAP) {
            Ok((r, ms)) => (Some(r), ms, false),
            Err(_) => (None, f64::NAN, true),
        };
        let mut late_launch: Option<String> = None;
        for _ in 0..after {
            if let Err(e) = unsafe { f.clone().launch_on_stream(&sa, cfg(), (dp(&x), dp(&y), N as i32)) } {
                late_launch.get_or_insert(format!("{:?}", e.0));
            }
        }
        let end = unsafe { sys::cuStreamEndCapture(sa.stream, &mut graph) };
        open.disarm();
        let (b, b_ms) = match b {
            Some(r) => (r, b_ms),
            None => b_done_rx.recv().unwrap(), // B returns once the capture is closed
        };
        let end_capture = if end == sys::CUresult::CUDA_SUCCESS { Ok(()) } else { Err(format!("{end:?}")) };
        let replay_ok = end_capture.is_ok() && !graph.is_null() && replay_is(dev, &sa, graph, &x, 1.0 + 2.0 * (before + after) as f32);
        release_tx.send(()).unwrap();
        drop((x, y));
        Outcome { b, b_ms, b_blocked, late_launch, end_capture, replay_ok }
    })
}

/// FACT 1 (the served failure, and the half of it nobody saw). In the engine's capture mode, every
/// legacy-stream call from another thread is refused with CAPTURE_IMPLICIT (it does not block),
/// AND the capturing thread's capture is invalidated: its later launches and its EndCapture return
/// CAPTURE_INVALIDATED and there is no graph. Wherever in the window it lands.
#[test]
fn a_legacy_stream_call_from_another_thread_is_refused_and_invalidates_the_capture() {
    let Some(g) = gpu("a_legacy_stream_call_from_another_thread_is_refused_and_invalidates_the_capture") else { return };
    for (before, after) in [(0usize, 0usize), (3, 0), (0, 3), (3, 3)] {
        for op in LEGACY_OPS {
            let o = probe(&g.dev, &g.f, TL, op, before, after);
            let ctx = format!("{op:?} with {before} launches before and {after} after: {o:?}");
            assert!(!o.b_blocked, "the refusal returns while the window is open (it does not wait for the capture): {ctx}");
            assert_eq!(o.b.as_ref().err().map(String::as_str), Some("CUDA_ERROR_STREAM_CAPTURE_IMPLICIT"), "B must be refused: {ctx}");
            assert_eq!(o.end_capture.as_ref().err().map(String::as_str), Some("CUDA_ERROR_STREAM_CAPTURE_INVALIDATED"), "A's capture: {ctx}");
            assert!(!o.replay_ok, "{ctx}");
            if after > 0 {
                assert_eq!(o.late_launch.as_deref(), Some("CUDA_ERROR_STREAM_CAPTURE_INVALIDATED"), "A's launches after the call: {ctx}");
            }
        }
    }
}

/// FACT 2 (what the fix stands on). The same work kept on B's OWN blocking stream is accepted and
/// leaves a thread-local capture alone: launches, cuBLAS, stream-ordered copies and memset,
/// stream-ordered alloc and free, and a drain of that stream. So are the plain cuMemAlloc / cuMemFree.
#[test]
fn work_on_the_second_threads_own_stream_leaves_a_thread_local_capture_alone() {
    let Some(g) = gpu("work_on_the_second_threads_own_stream_leaves_a_thread_local_capture_alone") else { return };
    for (before, after) in [(0usize, 0usize), (3, 0), (3, 3)] {
        for op in OWN_OPS.iter().chain(NO_STREAM_OPS.iter()) {
            let o = probe(&g.dev, &g.f, TL, *op, before, after);
            let ctx = format!("{op:?} with {before} launches before and {after} after: {o:?}");
            assert!(o.b.is_ok() && !o.b_blocked, "B's call: {ctx}");
            assert!(o.late_launch.is_none() && o.end_capture.is_ok() && o.replay_ok, "A's capture: {ctx}");
        }
    }
}

/// FACT 3. The capture mode is not the lever: under GLOBAL and RELAXED the legacy-stream calls are
/// refused and invalidate the capture exactly as under THREAD_LOCAL. (The engine's EXL3 and NVFP4
/// captures already use THREAD_LOCAL.) GLOBAL is stricter still and is why the own-stream fix is
/// stated for thread-local captures only: there it also refuses own-stream alloc / free / drain.
#[test]
fn the_capture_mode_does_not_remove_the_legacy_stream_hazard() {
    let Some(g) = gpu("the_capture_mode_does_not_remove_the_legacy_stream_hazard") else { return };
    for mode in [Mode::CU_STREAM_CAPTURE_MODE_GLOBAL, Mode::CU_STREAM_CAPTURE_MODE_RELAXED] {
        for op in LEGACY_OPS {
            let o = probe(&g.dev, &g.f, mode, op, 3, 0);
            assert!(o.b.is_err() && o.end_capture.is_err(), "{} {op:?}: {o:?}", mode_name(mode));
        }
    }
    for op in [Op::AllocOwn, Op::FreeOwn, Op::SyncOwn] {
        let o = probe(&g.dev, &g.f, Mode::CU_STREAM_CAPTURE_MODE_GLOBAL, op, 3, 0);
        assert!(o.b.is_err() && o.end_capture.is_err(), "GLOBAL polices another thread's own-stream {op:?} too: {o:?}");
    }
    for op in OWN_OPS {
        let o = probe(&g.dev, &g.f, Mode::CU_STREAM_CAPTURE_MODE_RELAXED, op, 3, 0);
        assert!(o.b.is_ok() && o.end_capture.is_ok() && o.replay_ok, "RELAXED {op:?}: {o:?}");
    }
}

/// What thread A does per round of the free-running race.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ASide {
    /// no capture: three launches, an own-stream download, a drain (the control)
    NoCapture,
    /// begin, three launches, end, destroy the graph
    CaptureOnly,
    /// the same, then replay the graph and check its numbers (own-stream copies only)
    CaptureReplay,
}

/// What thread B repeats in the free-running race.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BLoop {
    LegacyHtod, LegacyDtoh, LegacySync, OwnStream,
    /// the `pfx1-ltplans` thread as FIXED: one handle for the whole run, four plan builds per body,
    /// and the handle destroyed only after the capturing thread has finished
    LtPrewarm,
    /// the thread as it WAS: cublasLtCreate, four plan builds, cublasLtDestroy, per body (the
    /// destroy invalidates a capture on another thread: an instrument, expected to lose captures)
    LtPrewarmDestroys,
}

struct RaceResult {
    calls: usize,
    refused: BTreeMap<String, usize>,
    invalidated: usize,
    wrong: usize,
}

/// Free-running race: A runs at least `rounds` rounds back to back (and goes on until B has made
/// `min_calls` calls, up to 20x `rounds`: the count of what happened, not a rate) while B repeats
/// one operation as fast as it can. Nothing orders the two threads, so B's call can land at any
/// point of A's begin / record / end sequence.
fn race(g: &Gpu, b: BLoop, rounds: usize, a: ASide, min_calls: usize) -> RaceResult {
    race_l(g, b, rounds, a, min_calls, 3)
}

/// `race` with `launches` kernel launches recorded per capture window (the engine's windows hold
/// thousands; a short window makes a call that lands inside it rare).
fn race_l(g: &Gpu, b: BLoop, rounds: usize, a: ASide, min_calls: usize, launches: usize) -> RaceResult {
    let (dev, f) = (&g.dev, &g.f);
    let stop = AtomicBool::new(false);
    let b_calls = AtomicUsize::new(0);
    let (ready_tx, ready_rx) = mpsc::channel::<()>();
    std::thread::scope(|sc| {
        let devb = dev.clone();
        let fb = f.clone();
        let (stop, b_calls) = (&stop, &b_calls);
        let _release_b = SetOnDrop(stop); // A panicking must not leave B spinning on `stop` forever
        let hb = sc.spawn(move || {
            devb.bind_to_thread().unwrap();
            let sb = fork_blocking_stream(&devb);
            let host = vec![5f32; N];
            let mut host_out = vec![0f32; N];
            let mut pre = devb.htod_sync_copy(&vec![3f32; N]).unwrap();
            let pre2 = devb.htod_sync_copy(&vec![4f32; N]).unwrap();
            let own = unsafe { result::malloc_async(sb.stream, N * 4).unwrap() };
            devb.synchronize().unwrap();
            let lt_h = if b == BLoop::LtPrewarm { Some(lt::create_handle().expect("cublasLtCreate")) } else { None };
            ready_tx.send(()).unwrap();
            let mut errs = BTreeMap::<String, usize>::new();
            while !stop.load(Relaxed) {
                let r: Result<(), String> = unsafe {
                    match b {
                        BLoop::LegacyHtod => code(devb.htod_sync_copy_into(&host, &mut pre)),
                        BLoop::LegacyDtoh => code(devb.dtoh_sync_copy_into(&pre, &mut host_out)),
                        BLoop::LegacySync => code(devb.synchronize()),
                        // the fixed tower's forward in miniature: upload, compute, download, drain
                        BLoop::OwnStream => code(result::memcpy_htod_async(own, &host, sb.stream))
                            .and_then(|_| code(fb.clone().launch_on_stream(&sb, cfg(), (own as u64, dp(&pre2), N as i32))))
                            .and_then(|_| code(result::memcpy_dtoh_async(&mut host_out, own, sb.stream)))
                            .and_then(|_| code(result::stream::synchronize(sb.stream))),
                        BLoop::LtPrewarm => LT_SHAPES.iter().try_for_each(|&(k, n, m, ta)| lt_plan(lt_h.unwrap(), k, n, m, ta)),
                        BLoop::LtPrewarmDestroys => lt::create_handle().map_err(|e| format!("cublasLtCreate {:?}", e.0)).and_then(|h| {
                            let r = LT_SHAPES.iter().try_for_each(|&(k, n, m, ta)| lt_plan(h, k, n, m, ta));
                            let d = lt_code(lt::destroy_handle(h));
                            r.and(d)
                        }),
                    }
                };
                b_calls.fetch_add(1, Relaxed);
                if let Err(e) = r { *errs.entry(e).or_insert(0) += 1; }
                else if b == BLoop::OwnStream { assert!(host_out.iter().all(|&v| v == 9.0), "B's own result"); }
            }
            unsafe {
                let _ = result::stream::synchronize(sb.stream);
                let _ = result::free_async(own, sb.stream);
                let _ = result::stream::synchronize(sb.stream);
                if let Some(h) = lt_h { let _ = lt::destroy_handle(h); } // after A's last capture
            }
            drop((pre, pre2));
            errs
        });
        dev.bind_to_thread().unwrap();
        let sa = fork_blocking_stream(dev);
        let x = dev.htod_sync_copy(&vec![1f32; N]).unwrap();
        let y = dev.htod_sync_copy(&vec![2f32; N]).unwrap();
        let ones = vec![1f32; N];
        let mut host_x = vec![0f32; N];
        dev.synchronize().unwrap();
        ready_rx.recv().unwrap();
        let (mut invalidated, mut wrong) = (0usize, 0usize);
        let want = 1.0 + 2.0 * launches as f32; // x starts at 1 and each launch adds y = 2
        let mut round = 0usize;
        while round < rounds || (b_calls.load(Relaxed) < min_calls && round < rounds * 20) {
            if round % 500 == 0 { eprintln!("  [race {b:?} {a:?}] round {round}"); }
            round += 1;
            unsafe {
                if a == ASide::NoCapture {
                    let ok = result::memcpy_htod_async(*x.device_ptr(), &ones, sa.stream).is_ok()
                        && (0..launches).all(|_| f.clone().launch_on_stream(&sa, cfg(), (dp(&x), dp(&y), N as i32)).is_ok())
                        && result::memcpy_dtoh_async(&mut host_x, *x.device_ptr(), sa.stream).is_ok()
                        && sys::cuStreamSynchronize(sa.stream) == sys::CUresult::CUDA_SUCCESS;
                    if !ok || !host_x.iter().all(|&v| v == want) { wrong += 1; }
                    continue;
                }
                let mut graph: sys::CUgraph = std::ptr::null_mut();
                if sys::cuStreamBeginCapture_v2(sa.stream, TL) != sys::CUresult::CUDA_SUCCESS { invalidated += 1; continue; }
                let mut body_ok = true;
                for _ in 0..launches { body_ok &= f.clone().launch_on_stream(&sa, cfg(), (dp(&x), dp(&y), N as i32)).is_ok(); }
                let end = sys::cuStreamEndCapture(sa.stream, &mut graph);
                if end != sys::CUresult::CUDA_SUCCESS || !body_ok || graph.is_null() {
                    invalidated += 1;
                    if !graph.is_null() { sys::cuGraphDestroy(graph); }
                    continue;
                }
                if a == ASide::CaptureOnly { sys::cuGraphDestroy(graph); continue; }
                let mut exec: sys::CUgraphExec = std::ptr::null_mut();
                let ok = result::memcpy_htod_async(*x.device_ptr(), &ones, sa.stream).is_ok()
                    && sys::cuGraphInstantiate_v2(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0) == sys::CUresult::CUDA_SUCCESS
                    && sys::cuGraphLaunch(exec, sa.stream) == sys::CUresult::CUDA_SUCCESS
                    && result::memcpy_dtoh_async(&mut host_x, *x.device_ptr(), sa.stream).is_ok()
                    && sys::cuStreamSynchronize(sa.stream) == sys::CUresult::CUDA_SUCCESS;
                if !ok || !host_x.iter().all(|&v| v == want) { wrong += 1; }
                if !exec.is_null() { sys::cuGraphExecDestroy(exec); }
                sys::cuGraphDestroy(graph);
            }
        }
        stop.store(true, Relaxed);
        let refused = hb.join().unwrap();
        unsafe { let _ = sys::cuStreamSynchronize(sa.stream); }
        drop((x, y));
        RaceResult { calls: b_calls.load(Relaxed), refused, invalidated, wrong }
    })
}

/// FACT 4 (control). Without a capture in flight the legacy-stream calls of another thread are
/// harmless: thousands of them against a busy stream, none refused, nothing wrong. The hazard is
/// the capture window, not two threads sharing a device.
#[test]
fn legacy_stream_calls_are_harmless_when_nobody_captures() {
    let Some(g) = gpu("legacy_stream_calls_are_harmless_when_nobody_captures") else { return };
    for b in [BLoop::LegacyHtod, BLoop::LegacyDtoh, BLoop::LegacySync] {
        let r = race(&g, b, 3000, ASide::NoCapture, 1000);
        println!("RACE {b:?} against no capture: B {} calls, refused {:?} | A wrong {}", r.calls, r.refused, r.wrong);
        assert!(r.calls >= 1000, "{b:?}: B made only {} calls, the control did not run", r.calls);
        assert!(r.refused.is_empty() && r.invalidated == 0 && r.wrong == 0, "{b:?}: {:?} {} {}", r.refused, r.invalidated, r.wrong);
    }
}

/// FACT 5. The own-stream sequence the fixed tower uses (upload, launch, download, drain), free
/// running against back-to-back captures that are each replayed and checked: nothing refused, no
/// capture invalidated, every number right on both threads.
#[test]
fn own_stream_copies_survive_a_free_running_capture_race() {
    let Some(g) = gpu("own_stream_copies_survive_a_free_running_capture_race") else { return };
    let r = race(&g, BLoop::OwnStream, 3000, ASide::CaptureReplay, 500);
    println!("RACE OwnStream against capture + replay rounds: B {} calls, refused {:?} | A invalidated {}, wrong {}", r.calls, r.refused, r.invalidated, r.wrong);
    assert!(r.calls >= 500, "B made only {} calls", r.calls);
    assert!(r.refused.is_empty() && r.invalidated == 0 && r.wrong == 0, "{:?} {} {}", r.refused, r.invalidated, r.wrong);
}

// =====================================================================================
// Part A2 — the cuBLASLt pre-population thread (`pfx1-ltplans`) and library handles
// =====================================================================================

/// FACT 6 (S-B9-REL-VISIONFIX2 V2a). The engine starts one more thread after the boot captures: it
/// creates its own cuBLASLt handle, builds prefill plans (descriptors + heuristics) in the
/// background and used to destroy the handle when done (`lt_prewarm_start`). The scheduler captures
/// lazily at any time after boot, so every call of that thread must be harmless to a capture on
/// another thread. Measured here ordered (one call per window, wherever in it) and free-running
/// (the thread's body against back-to-back captures that are replayed and checked).
/// Creating a handle, building plans (descriptor, attributes, preference, layouts, heuristic and
/// their destruction) and binding the context are harmless.
#[test]
fn the_cublaslt_plan_thread_leaves_another_threads_capture_alone() {
    let Some(g) = gpu("the_cublaslt_plan_thread_leaves_another_threads_capture_alone") else { return };
    for (before, after) in [(0usize, 0usize), (3, 0), (3, 3)] {
        for op in [Op::LtCreateHandle, Op::LtPlanBuild, Op::BindThread] {
            let o = probe(&g.dev, &g.f, TL, op, before, after);
            let ctx = format!("{op:?} with {before} launches before and {after} after: {o:?}");
            assert!(o.b.is_ok() && !o.b_blocked, "the cuBLASLt thread's call: {ctx}");
            assert!(o.late_launch.is_none() && o.end_capture.is_ok() && o.replay_ok, "the capture: {ctx}");
        }
    }
    // short windows (3 launches) and engine-sized ones (1,500 launches: milliseconds each)
    for (launches, rounds) in [(3usize, 3000usize), (1500, 300)] {
        let r = race_l(&g, BLoop::LtPrewarm, rounds, ASide::CaptureReplay, 200, launches);
        println!("RACE LtPrewarm against capture + replay of {launches} launches: B {} bodies, refused {:?} | A invalidated {}, wrong {}",
                 r.calls, r.refused, r.invalidated, r.wrong);
        assert!(r.calls >= 200, "the cuBLASLt thread ran only {} bodies", r.calls);
        assert!(r.refused.is_empty() && r.invalidated == 0 && r.wrong == 0, "{launches} launches: {:?} {} {}", r.refused, r.invalidated, r.wrong);
    }
}

/// FACT 7 (S-B9-REL-VISIONFIX2 V2a, the finding behind the engine fix). Destroying a cuBLASLt or a
/// cuBLAS handle is NOT refused (it returns ok) but it INVALIDATES a capture open on another thread,
/// in every capture mode. `pfx1-ltplans` called `cublasLtDestroy` once, when its plans were done,
/// at an unpredictable moment after boot: a lazy capture of the scheduler that was open then was
/// lost (counted and re-captured at the shape's next use). The thread no longer destroys its
/// handle; no library handle may be destroyed on a serving-time thread. (`CudaBlas`'s drop is
/// cublasDestroy: the tower and the models never drop theirs while serving.)
#[test]
fn destroying_a_library_handle_invalidates_another_threads_capture() {
    let Some(g) = gpu("destroying_a_library_handle_invalidates_another_threads_capture") else { return };
    for mode in [TL, Mode::CU_STREAM_CAPTURE_MODE_RELAXED] {
        for op in [Op::LtDestroyHandle, Op::BlasDestroy] {
            let o = probe(&g.dev, &g.f, mode, op, 3, 0);
            let ctx = format!("{} {op:?}: {o:?}", mode_name(mode));
            assert!(o.b.is_ok(), "the destroy itself is not refused: {ctx}");
            assert_eq!(o.end_capture.as_ref().err().map(String::as_str), Some("CUDA_ERROR_STREAM_CAPTURE_INVALIDATED"), "A's capture: {ctx}");
            assert!(!o.replay_ok, "{ctx}");
        }
    }
}

/// FACT 8. Creating a cuBLAS handle, loading a module (the tower's and the models' constructors)
/// and the first launch of a kernel of an already loaded module (the tower is not warmed at boot, so
/// its first image request makes the first launches of its kernels and the first sgemm of its
/// handle, from a worker thread, possibly while the scheduler captures) do not disturb a
/// thread-local capture; the constructors are boot-time work all the same.
#[test]
fn creating_a_blas_handle_or_loading_a_module_leaves_a_thread_local_capture_alone() {
    let Some(g) = gpu("creating_a_blas_handle_or_loading_a_module_leaves_a_thread_local_capture_alone") else { return };
    for op in [Op::BlasCreate, Op::ModuleLoad, Op::FirstLaunch] {
        let o = probe(&g.dev, &g.f, TL, op, 3, 3);
        let ctx = format!("{op:?}: {o:?}");
        assert!(o.b.is_ok() && !o.b_blocked, "B's call: {ctx}");
        assert!(o.late_launch.is_none() && o.end_capture.is_ok() && o.replay_ok, "A's capture: {ctx}");
    }
}

// =====================================================================================
// Part B — the real GPU vision tower
// =====================================================================================

/// The synthetic model runs: the GPU tower agrees with the CPU tower on it, so the tests below
/// exercise the real forward and not a degenerate one. And it is deterministic: the same image
/// twice is the same bits (the second forward runs on reused, non-zeroed scratch).
#[test]
fn the_gpu_tower_matches_the_cpu_tower_on_the_synthetic_model() {
    let Some(g) = gpu("the_gpu_tower_matches_the_cpu_tower_on_the_synthetic_model") else { return };
    let tower = synthetic_tower();
    let mut gvt = GpuVisualTower::new(g.dev.clone(), &tower).expect("gpu tower");
    for (gh, gw) in GRIDS {
        let pv = pixels(&tower, gh, gw);
        let cpu = tower.forward_cpu(&pv, gh, gw);
        let (gpu_out, _) = gvt.forward(&pv, gh, gw, false).expect("gpu forward");
        assert_eq!(gpu_out.len(), cpu.len());
        let (num, den) = cpu.iter().zip(&gpu_out).fold((0f64, 0f64), |(n, d), (c, g)| (n + ((*c - *g) as f64).powi(2), d + (*c as f64).powi(2)));
        let rel = (num / den).sqrt();
        println!("synthetic tower, grid {gh}x{gw}: rel-L2 GPU vs CPU {rel:.2e}");
        assert!(den > 0.0 && rel < 1e-4, "grid {gh}x{gw}: rel-L2 GPU vs CPU {rel:e}");
        let (again, _) = gvt.forward(&pv, gh, gw, false).expect("gpu forward");
        assert!(gpu_out.iter().zip(&again).all(|(a, b)| a.to_bits() == b.to_bits()), "grid {gh}x{gw}: second forward differs");
    }
}

/// THE F1 REPRODUCER. The real tower's `forward`, on a second thread, entirely inside a capture
/// window that the first thread holds open (the scheduler capturing a decode graph while an image
/// request is served). Twice: on a warmed grid, and on a grid the tower has not seen (its scratch
/// pool misses, so the forward allocates inside the window).
///
/// Before the fix this fails at once, with the served error: `forward` returns
/// `CUDA_ERROR_STREAM_CAPTURE_IMPLICIT` and the first thread's capture ends
/// `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`. After it: the forward succeeds with exactly the bits it
/// produces outside a capture, and the capture ends cleanly and replays the right numbers.
#[test]
fn the_gpu_tower_forward_inside_another_threads_capture_window() {
    let Some(g) = gpu("the_gpu_tower_forward_inside_another_threads_capture_window") else { return };
    let (dev, f) = (&g.dev, &g.f);
    let tower = synthetic_tower();
    let mut gvt = GpuVisualTower::new(dev.clone(), &tower).expect("gpu tower");
    let pv_warm = pixels(&tower, 4, 4);
    let (ref_warm, _) = gvt.forward(&pv_warm, 4, 4, false).expect("reference forward outside any capture");
    // the cold grid's reference comes from a SECOND tower, so that the first one really is cold for it
    let pv_cold = pixels(&tower, 8, 8);
    let ref_cold = { let mut other = GpuVisualTower::new(dev.clone(), &tower).unwrap(); other.forward(&pv_cold, 8, 8, false).unwrap().0 };

    for (label, pv, gh, gw, want) in [("warmed 4x4", &pv_warm, 4usize, 4usize, &ref_warm), ("cold 8x8 (allocates in the window)", &pv_cold, 8, 8, &ref_cold)] {
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let (done_tx, done_rx) = mpsc::channel::<Result<Vec<f32>, String>>();
        let gvt_ref = &mut gvt;
        let (end, late, replayed) = std::thread::scope(move |sc| {
            sc.spawn(move || {
                go_rx.recv().unwrap();
                done_tx.send(gvt_ref.forward(pv, gh, gw, false).map(|r| r.0).map_err(|e| format!("{e:#}"))).unwrap();
            });
            dev.bind_to_thread().unwrap();
            let sa = fork_blocking_stream(dev);
            let x = dev.htod_sync_copy(&vec![1f32; N]).unwrap();
            let y = dev.htod_sync_copy(&vec![2f32; N]).unwrap();
            dev.synchronize().unwrap();
            let mut graph: sys::CUgraph = std::ptr::null_mut();
            assert_eq!(unsafe { sys::cuStreamBeginCapture_v2(sa.stream, TL) }, sys::CUresult::CUDA_SUCCESS);
            let mut open = OpenCapture::new(sa.stream);
            for _ in 0..3 { unsafe { f.clone().launch_on_stream(&sa, cfg(), (dp(&x), dp(&y), N as i32)).unwrap(); } }
            go_tx.send(()).unwrap();
            let got = done_rx.recv().unwrap(); // the whole forward ran while this capture was open
            let mut late: Option<String> = None;
            for _ in 0..3 {
                if let Err(e) = unsafe { f.clone().launch_on_stream(&sa, cfg(), (dp(&x), dp(&y), N as i32)) } { late.get_or_insert(format!("{:?}", e.0)); }
            }
            let end = unsafe { sys::cuStreamEndCapture(sa.stream, &mut graph) };
            open.disarm();
            let replayed = end == sys::CUresult::CUDA_SUCCESS && !graph.is_null() && replay_is(dev, &sa, graph, &x, 13.0);
            let got = match got {
                Ok(v) => v,
                Err(e) => panic!("{label}: the GPU tower failed inside another thread's capture window: {e} \
                                  (the capture then ended {end:?}; this is the served F1 failure)"),
            };
            assert!(got.len() == want.len() && got.iter().zip(want.iter()).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{label}: embeddings computed inside the capture window differ from those computed outside one");
            drop((x, y));
            (end, late, replayed)
        });
        assert_eq!(late, None, "{label}: the capturing thread's later launches");
        assert_eq!(end, sys::CUresult::CUDA_SUCCESS, "{label}: the capturing thread's capture must not be invalidated by the tower");
        assert!(replayed, "{label}: the captured graph replays the right numbers");
    }
}

/// The real tower free-running against back-to-back captures: at least 400 forwards cycling over
/// three image sizes on one thread while the other captures, replays and checks graphs without
/// pause (each side goes on until the other has done its minimum, so they overlap throughout). Every
/// forward succeeds with the reference bits of its image, no capture is invalidated, no replay is
/// wrong. (Pool flatness is `tests/vision_tower_pool_test.rs`: it reads a method master lacks.)
#[test]
fn the_gpu_tower_survives_a_free_running_capture_race() {
    let Some(g) = gpu("the_gpu_tower_survives_a_free_running_capture_race") else { return };
    let (dev, f) = (&g.dev, &g.f);
    let tower = synthetic_tower();
    let mut gvt = GpuVisualTower::new(dev.clone(), &tower).expect("gpu tower");
    let inputs: Vec<(usize, usize, Vec<f32>)> = GRIDS.iter().map(|&(gh, gw)| (gh, gw, pixels(&tower, gh, gw))).collect();
    let refs: Vec<Vec<f32>> = inputs.iter().map(|(gh, gw, pv)| gvt.forward(pv, *gh, *gw, false).expect("reference").0).collect();
    const MIN_FORWARDS: usize = 400;
    const MIN_CAPTURES: usize = 100;
    let stop = AtomicBool::new(false);
    let captures_a = AtomicUsize::new(0);
    let (captures, invalidated, wrong, forwards) = std::thread::scope(|sc| {
        let (stop, captures_a, inputs, refs, gvt_ref) = (&stop, &captures_a, &inputs, &refs, &mut gvt);
        let _release_b = SetOnDrop(stop); // A panicking releases B's loop below
        let hb = sc.spawn(move || {
            let _release_a = SetOnDrop(stop); // B panicking (a failed forward, wrong bits) releases A's capture loop
            let mut it = 0usize;
            while !stop.load(Relaxed) && (it < MIN_FORWARDS || captures_a.load(Relaxed) < MIN_CAPTURES) {
                let (gh, gw, pv) = &inputs[it % inputs.len()];
                let got = gvt_ref.forward(pv, *gh, *gw, false).unwrap_or_else(|e| panic!("forward {it} ({gh}x{gw}) failed during the capture race: {e:#}")).0;
                assert!(got.iter().zip(&refs[it % inputs.len()]).all(|(a, b)| a.to_bits() == b.to_bits()), "forward {it} ({gh}x{gw}): wrong bits");
                it += 1;
            }
            stop.store(true, Relaxed);
            it
        });
        dev.bind_to_thread().unwrap();
        let sa = fork_blocking_stream(dev);
        let x = dev.htod_sync_copy(&vec![1f32; N]).unwrap();
        let y = dev.htod_sync_copy(&vec![2f32; N]).unwrap();
        let ones = vec![1f32; N];
        let mut host_x = vec![0f32; N];
        dev.synchronize().unwrap();
        let (mut captures, mut invalidated, mut wrong) = (0usize, 0usize, 0usize);
        while !stop.load(Relaxed) {
            unsafe {
                let mut graph: sys::CUgraph = std::ptr::null_mut();
                captures += 1;
                captures_a.store(captures, Relaxed);
                if sys::cuStreamBeginCapture_v2(sa.stream, TL) != sys::CUresult::CUDA_SUCCESS { invalidated += 1; continue; }
                let mut ok = true;
                for _ in 0..3 { ok &= f.clone().launch_on_stream(&sa, cfg(), (dp(&x), dp(&y), N as i32)).is_ok(); }
                if sys::cuStreamEndCapture(sa.stream, &mut graph) != sys::CUresult::CUDA_SUCCESS || !ok || graph.is_null() {
                    invalidated += 1;
                    if !graph.is_null() { sys::cuGraphDestroy(graph); }
                    continue;
                }
                let mut exec: sys::CUgraphExec = std::ptr::null_mut();
                let good = result::memcpy_htod_async(*x.device_ptr(), &ones, sa.stream).is_ok()
                    && sys::cuGraphInstantiate_v2(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0) == sys::CUresult::CUDA_SUCCESS
                    && sys::cuGraphLaunch(exec, sa.stream) == sys::CUresult::CUDA_SUCCESS
                    && result::memcpy_dtoh_async(&mut host_x, *x.device_ptr(), sa.stream).is_ok()
                    && sys::cuStreamSynchronize(sa.stream) == sys::CUresult::CUDA_SUCCESS;
                if !good || !host_x.iter().all(|&v| v == 7.0) { wrong += 1; }
                if !exec.is_null() { sys::cuGraphExecDestroy(exec); }
                sys::cuGraphDestroy(graph);
            }
        }
        let forwards = hb.join().expect("the tower thread");
        unsafe { let _ = sys::cuStreamSynchronize(sa.stream); }
        drop((x, y));
        (captures, invalidated, wrong, forwards)
    });
    println!("TOWER RACE: {forwards} forwards over {GRIDS:?} against {captures} capture + replay rounds: {invalidated} captures invalidated, {wrong} replays wrong");
    assert!(forwards >= MIN_FORWARDS && captures >= MIN_CAPTURES, "the two threads really overlapped ({forwards} forwards, {captures} capture rounds)");
    assert_eq!((invalidated, wrong), (0, 0), "the tower must not disturb the other thread's captures");
}

// =====================================================================================
// Part C — instruments (ignored). Run by name with --ignored --nocapture.
// =====================================================================================

/// The whole table: every operation in every capture mode.
#[test]
#[ignore]
fn instrument_table_of_every_operation_in_every_mode() {
    let Some(g) = gpu("instrument_table_of_every_operation_in_every_mode") else { return };
    for mode in [TL, Mode::CU_STREAM_CAPTURE_MODE_GLOBAL, Mode::CU_STREAM_CAPTURE_MODE_RELAXED] {
        println!("== capture mode {} (thread A captures 3 launches on its own blocking stream)", mode_name(mode));
        for op in LEGACY_OPS.iter().chain(NO_STREAM_OPS.iter()).chain(OWN_OPS.iter()).chain(LT_OPS.iter()).chain(CTOR_OPS.iter()) {
            let o = probe(&g.dev, &g.f, mode, *op, 3, 0);
            println!("   {:<15} B: {:<38} {:>8.3} ms | A end-capture: {:<38} replay {}",
                     format!("{op:?}"), show(&o.b), o.b_ms, show(&o.end_capture), if o.replay_ok { "ok" } else { "none" });
        }
    }
}

/// One legacy call per capture window, landing while the capturing thread is busy recording 2,000
/// launches (the engine's situation). Measured: 300 of 300 refused, 300 of 300 captures
/// invalidated, no crash, in each of six runs. Counters go to stderr after every window.
#[test]
#[ignore]
fn instrument_one_legacy_call_per_busy_capture_window() {
    let Some(g) = gpu("instrument_one_legacy_call_per_busy_capture_window") else { return };
    let (dev, f) = (&g.dev, &g.f);
    let (go_tx, go_rx) = mpsc::channel::<bool>();
    let (done_tx, done_rx) = mpsc::channel::<bool>();
    std::thread::scope(|sc| {
        let devb = dev.clone();
        sc.spawn(move || {
            devb.bind_to_thread().unwrap();
            devb.synchronize().unwrap();
            while let Ok(true) = go_rx.recv() {
                std::thread::sleep(Duration::from_micros(300)); // land inside the burst, not at its first launch
                done_tx.send(devb.synchronize().is_err()).unwrap();
            }
        });
        dev.bind_to_thread().unwrap();
        let sa = fork_blocking_stream(dev);
        let x = dev.htod_sync_copy(&vec![1f32; N]).unwrap();
        let y = dev.htod_sync_copy(&vec![2f32; N]).unwrap();
        dev.synchronize().unwrap();
        let (mut refused, mut invalid) = (0usize, 0usize);
        for w in 1..=300usize {
            unsafe {
                let mut graph: sys::CUgraph = std::ptr::null_mut();
                assert_eq!(sys::cuStreamBeginCapture_v2(sa.stream, TL), sys::CUresult::CUDA_SUCCESS);
                go_tx.send(true).unwrap();
                for _ in 0..2000 { let _ = f.clone().launch_on_stream(&sa, cfg(), (dp(&x), dp(&y), N as i32)); }
                refused += done_rx.recv().unwrap() as usize;
                invalid += (sys::cuStreamEndCapture(sa.stream, &mut graph) != sys::CUresult::CUDA_SUCCESS) as usize;
                if !graph.is_null() { sys::cuGraphDestroy(graph); }
            }
            eprintln!("BUSY window {w}: B refused {refused} | A captures invalidated {invalid}");
        }
        go_tx.send(false).unwrap();
        drop((x, y));
        println!("BUSY_TOTAL 300 windows survived: B refused {refused} | A invalidated {invalid}");
    });
}

/// The closing edge: A signals B and ends its capture at once, so B's single legacy call races
/// `cuStreamEndCapture`. Counts the four outcomes (refused or not, invalidated or not). "Refused,
/// capture intact" is what the served event's log shows. Counters go to stderr every 25 windows:
/// this race is also where the driver has been seen to segfault.
#[test]
#[ignore]
fn instrument_closing_edge_outcomes() {
    let Some(g) = gpu("instrument_closing_edge_outcomes") else { return };
    let (dev, f) = (&g.dev, &g.f);
    let (go_tx, go_rx) = mpsc::channel::<bool>();
    let (done_tx, done_rx) = mpsc::channel::<bool>();
    std::thread::scope(|sc| {
        let devb = dev.clone();
        sc.spawn(move || {
            devb.bind_to_thread().unwrap();
            devb.synchronize().unwrap();
            while let Ok(true) = go_rx.recv() { done_tx.send(devb.synchronize().is_err()).unwrap(); }
        });
        dev.bind_to_thread().unwrap();
        let sa = fork_blocking_stream(dev);
        let x = dev.htod_sync_copy(&vec![1f32; N]).unwrap();
        let y = dev.htod_sync_copy(&vec![2f32; N]).unwrap();
        dev.synchronize().unwrap();
        let mut n = [[0usize; 2]; 2]; // [refused][invalidated]
        for w in 1..=4000usize {
            unsafe {
                let mut graph: sys::CUgraph = std::ptr::null_mut();
                assert_eq!(sys::cuStreamBeginCapture_v2(sa.stream, TL), sys::CUresult::CUDA_SUCCESS);
                let mut ok = true;
                for _ in 0..(w % 7) { ok &= f.clone().launch_on_stream(&sa, cfg(), (dp(&x), dp(&y), N as i32)).is_ok(); }
                go_tx.send(true).unwrap();
                for _ in 0..(w % 5) { std::hint::spin_loop(); }
                let end = sys::cuStreamEndCapture(sa.stream, &mut graph);
                let refused = done_rx.recv().unwrap();
                n[refused as usize][(end != sys::CUresult::CUDA_SUCCESS || !ok) as usize] += 1;
                if !graph.is_null() { sys::cuGraphDestroy(graph); }
            }
            if w % 25 == 0 {
                eprintln!("EDGE window {w}: not refused + intact {} | REFUSED + intact {} | not refused + invalidated {} | refused + invalidated {}",
                          n[0][0], n[1][0], n[0][1], n[1][1]);
            }
        }
        go_tx.send(false).unwrap();
        drop((x, y));
        println!("EDGE_TOTAL 4000 windows survived: not refused + intact {} | REFUSED + intact {} | not refused + invalidated {} | refused + invalidated {}",
                 n[0][0], n[1][0], n[0][1], n[1][1]);
    });
}

fn race_instrument(name: &str, b: BLoop, a: ASide) {
    let Some(g) = gpu(name) else { return };
    let t = Instant::now();
    // engine-sized windows (1,500 launches) for the destroy instrument; keep A capturing until B has made some bodies
    let (min_calls, launches, rounds) = if b == BLoop::LtPrewarmDestroys { (6, 1500, 400) } else { (0, 3, 3000) };
    let r = race_l(&g, b, rounds, a, min_calls, launches);
    println!("RACE {b:?} {a:?} in {:.2} s | B: {} calls, refused {:?} | A: {} captures invalidated, {} results wrong",
             t.elapsed().as_secs_f64(), r.calls, r.refused, r.invalidated, r.wrong);
}
/// The cuBLASLt thread as it WAS (cublasLtDestroy per body), free-running against back-to-back
/// engine-sized captures: EXPECTED to lose captures (measured on GB10: 328..336 of 400 captures
/// invalidated in two runs; one of three runs died with SIGSEGV inside the driver).
#[test] #[ignore] fn instrument_race_lt_prewarm_with_destroy_loses_captures() { race_instrument("instrument_race_lt_prewarm_with_destroy_loses_captures", BLoop::LtPrewarmDestroys, ASide::CaptureReplay); }
/// EXPECTED TO DIE: a legacy-stream copy free-running against back-to-back captures segfaults
/// inside libcuda (15 of 15 runs on GB10, usually within 500 rounds; gdb: the second thread, in
/// `cuStreamSynchronize(NULL)`). The last "[race ...] round" line on stderr is how far it got.
#[test] #[ignore] fn instrument_race_legacy_htod_dies_in_the_driver() { race_instrument("instrument_race_legacy_htod_dies_in_the_driver", BLoop::LegacyHtod, ASide::CaptureOnly); }
/// EXPECTED TO DIE: as above, with the download.
#[test] #[ignore] fn instrument_race_legacy_dtoh_dies_in_the_driver() { race_instrument("instrument_race_legacy_dtoh_dies_in_the_driver", BLoop::LegacyDtoh, ASide::CaptureOnly); }
/// EXPECTED TO DIE: as above, with the bare legacy-stream drain.
#[test] #[ignore] fn instrument_race_legacy_sync_dies_in_the_driver() { race_instrument("instrument_race_legacy_sync_dies_in_the_driver", BLoop::LegacySync, ASide::CaptureOnly); }
