//! FLASHNEXT-DF2-B (2026-10-23): the DFlash2 tap capture must be the MEAN over the trunk's
//! hyper-connection streams (the artifact's SGLang patch: `hidden.unflatten(-1,(hc,h)).mean(-2)`),
//! and bit-identical to the old stream-0 copy when hc == 1 (the 27B / Ornith trunks).
//!
//! Both capture kernels are exercised on the device against a host reference that uses the
//! kernels' exact arithmetic (ascending stream order, f32 accumulate, one RNE bf16 rounding):
//!   * `xq_df2_tap_cvt`   (kernels/exl3_bench.cu)  — the EXL3 serve path, f32 residual rows
//!   * `df2_tap_hcmean_b` (kernels/gpu_kernels.cu) — the NVFP4 trunk path, bf16 residual rows
//! plus the host-side `band_smem` bound (ISSUES work[112]: the window bound at every ntot).
//!
//! Run: cargo test --release --test df2_tap_hcmean_test -- --nocapture

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig, DevicePtr};
use cudarc::nvrtc::Ptx;
use half::bf16;

fn f32_to_bf16_rne(x: f32) -> bf16 { bf16::from_f32(x) }

/// Deterministic pseudo-random f32 in [-4, 4) (no rand dependency).
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (((*seed >> 33) as u32) as f32 / (1u64 << 31) as f32) * 8.0 - 4.0
}

/// Host reference: token r, feature c -> bf16(mean over streams), exactly the kernels' order.
fn host_tap(rows: &[f32], h: usize, hc: usize, r: usize, c: usize) -> bf16 {
    let row = &rows[r * h * hc..];
    let mut acc = row[c];
    for s in 1..hc { acc += row[s * h + c]; }
    f32_to_bf16_rne(acc / hc as f32)
}

#[test]
fn band_smem_is_the_window_bound_at_every_ntot() {
    use gb10_inference::dflash2::{band_smem, block, HEAD_DIM};
    let window = 2048usize;
    let full = (HEAD_DIM + window + block() + 32) * 4;
    for ntot in [9usize, 300, 348, 900, 2056, 2057, 100_000] {
        assert_eq!(band_smem(window, ntot), full, "ntot {ntot}: smem must not shrink below the window bound");
    }
    assert_eq!(full, 8_864, "window 2048 / block 8: the documented constant");
}

#[test]
fn tap_kernels_hc_mean_and_hc1_identity() {
    let dev = CudaDevice::new(0).expect("cuda device 0");
    dev.load_ptx(Ptx::from_src(gb10_inference::ptx_embed::EXL3_BENCH), "exl3_bench_t", &["xq_df2_tap_cvt"]).expect("exl3_bench ptx");
    dev.load_ptx(Ptx::from_src(gb10_inference::ptx_embed::GPU_KERNELS), "gpu_kernels_t", &["df2_tap_hcmean_b"]).expect("gpu_kernels ptx");
    let f_exl3 = dev.get_func("exl3_bench_t", "xq_df2_tap_cvt").unwrap();
    let f_nvfp4 = dev.get_func("gpu_kernels_t", "df2_tap_hcmean_b").unwrap();

    let h = 64usize;          // small hidden; the kernels are h-generic
    let n = 7usize;           // tokens (a block-8 round's committed columns)
    let ntaps = 5usize;
    let dst_pitch = ntaps * h; // the staging's column stride (TAP_CONCAT_DIM analog)
    let tap_li = 3usize;
    let mut seed = 0xDF2B_2026_u64;
    for hc in [1usize, 4] {
        let rw = h * hc;
        let rows_f32: Vec<f32> = (0..n * rw).map(|_| lcg(&mut seed)).collect();
        let rows_bf16: Vec<bf16> = rows_f32.iter().map(|&x| f32_to_bf16_rne(x)).collect();
        // what the bf16 kernel actually sees (its inputs are already bf16)
        let rows_bf16_as_f32: Vec<f32> = rows_bf16.iter().map(|x| x.to_f32()).collect();

        // ---- EXL3 kernel: f32 rows, pitch = rw ----
        let src = dev.htod_sync_copy(&rows_f32).unwrap();
        let dst = dev.htod_sync_copy(&vec![bf16::from_f32(-7.0); n * dst_pitch]).unwrap();
        let elems = (h * n) as i64;
        unsafe {
            f_exl3.clone().launch(LaunchConfig { grid_dim: (((elems + 255) / 256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                (*dst.device_ptr() as u64, *src.device_ptr() as u64, elems, h as i32, rw as i64, tap_li as i32, dst_pitch as i64, hc as i32)).unwrap();
        }
        dev.synchronize().unwrap();
        let out: Vec<bf16> = dev.dtoh_sync_copy(&dst).unwrap();
        let mut mism = 0usize;
        for r in 0..n { for c in 0..h {
            let got = out[r * dst_pitch + tap_li * h + c];
            let want = host_tap(&rows_f32, h, hc, r, c);
            if got.to_bits() != want.to_bits() { mism += 1; }
            if hc == 1 {
                // the old stream-0 conversion, bit for bit
                assert_eq!(got.to_bits(), f32_to_bf16_rne(rows_f32[r * rw + c]).to_bits(), "exl3 hc=1 identity r{r} c{c}");
            }
        }}
        // untouched staging (other taps' rows) must stay untouched
        for r in 0..n { for k in 0..ntaps { if k == tap_li { continue; } for c in 0..h {
            assert_eq!(out[r * dst_pitch + k * h + c], bf16::from_f32(-7.0), "exl3 kernel wrote outside its tap rows");
        }}}
        assert_eq!(mism, 0, "xq_df2_tap_cvt hc={hc}: {mism} mismatches vs the host mean");

        // ---- NVFP4-path kernel: bf16 rows, rw-wide ----
        let srcb = dev.htod_sync_copy(&rows_bf16).unwrap();
        let dstb = dev.htod_sync_copy(&vec![bf16::from_f32(-7.0); n * dst_pitch]).unwrap();
        unsafe {
            f_nvfp4.clone().launch(LaunchConfig { grid_dim: (((elems + 255) / 256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 },
                (*dstb.device_ptr() as u64, *srcb.device_ptr() as u64, elems, h as i32, hc as i32, tap_li as i32, dst_pitch as i64)).unwrap();
        }
        dev.synchronize().unwrap();
        let outb: Vec<bf16> = dev.dtoh_sync_copy(&dstb).unwrap();
        let mut mismb = 0usize;
        for r in 0..n { for c in 0..h {
            let got = outb[r * dst_pitch + tap_li * h + c];
            let want = host_tap(&rows_bf16_as_f32, h, hc, r, c);
            if got.to_bits() != want.to_bits() { mismb += 1; }
            if hc == 1 {
                assert_eq!(got.to_bits(), rows_bf16[r * rw + c].to_bits(), "nvfp4 hc=1 must be the plain copy r{r} c{c}");
            }
        }}
        for r in 0..n { for k in 0..ntaps { if k == tap_li { continue; } for c in 0..h {
            assert_eq!(outb[r * dst_pitch + k * h + c], bf16::from_f32(-7.0), "nvfp4 kernel wrote outside its tap rows");
        }}}
        assert_eq!(mismb, 0, "df2_tap_hcmean_b hc={hc}: {mismb} mismatches vs the host mean");
        println!("hc={hc}: both tap kernels match the host mean bit-for-bit on {n}x{h} ({} elems)", n * h);
    }
}
