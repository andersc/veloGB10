//! S-B9-REL-VISIONFIX2 V1: the GPU vision tower's scratch pool is flat once warmed.
//!
//! Split from `vision_capture_race_test.rs` so that file builds against master's tower (which has no
//! `pool_stats`): the race reproducer must stay buildable on master, this check cannot.
//!
//! Needs a CUDA device and `src/ptx/gpu_vision.ptx`; without them every test prints a
//! `=== SKIPPED ...` banner on stderr and runs nothing (see `vision_race_common`).

mod vision_race_common;

use gb10_inference::vision_gpu::GpuVisualTower;
use vision_race_common::*;

/// 90 forwards cycling over three image sizes (two passes warm every bucket): after the warm-up the
/// tower's pool neither allocates nor loses a buffer, and the result of every forward is the bits
/// of the first one of its size (a reused, non-zeroed buffer changes nothing).
#[test]
fn the_tower_pool_is_flat_once_warmed() {
    let Some(g) = gpu("the_tower_pool_is_flat_once_warmed") else { return };
    let tower = synthetic_tower();
    let mut gvt = GpuVisualTower::new(g.dev.clone(), &tower).expect("gpu tower");
    let inputs: Vec<(usize, usize, Vec<f32>)> = GRIDS.iter().map(|&(gh, gw)| (gh, gw, pixels(&tower, gh, gw))).collect();
    let refs: Vec<Vec<f32>> = inputs.iter().map(|(gh, gw, pv)| gvt.forward(pv, *gh, *gw, false).expect("reference").0).collect();
    for (gh, gw, pv) in &inputs { gvt.forward(pv, *gh, *gw, false).expect("second warm-up pass"); }
    let (bytes0, allocs0) = gvt.pool_stats();
    assert!(bytes0 > 0 && allocs0 > 0, "the warm-up must have populated the pool ({bytes0} bytes, {allocs0} allocations)");
    for it in 0..90usize {
        let (gh, gw, pv) = &inputs[it % inputs.len()];
        let got = gvt.forward(pv, *gh, *gw, false).unwrap_or_else(|e| panic!("forward {it} ({gh}x{gw}): {e:#}")).0;
        assert!(got.iter().zip(&refs[it % inputs.len()]).all(|(a, b)| a.to_bits() == b.to_bits()), "forward {it} ({gh}x{gw}): wrong bits");
    }
    let (bytes1, allocs1) = gvt.pool_stats();
    println!("TOWER POOL: {bytes0} -> {bytes1} bytes, {allocs0} -> {allocs1} allocations over 90 forwards");
    assert_eq!((bytes1, allocs1), (bytes0, allocs0), "the tower's scratch pool is flat once warmed (no allocation, nothing lost)");
}
