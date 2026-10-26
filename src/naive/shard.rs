//! S5: tensor-parallel sharding of the Naive pack (PLAN/NAIVE_N05_PORT.md §5.1).
//!
//! Rank r of W holds: query heads [r*64/W, (r+1)*64/W), the matching kv heads (GQA groups stay
//! whole: 64/8 = 8 and 64/4 = 16 query heads per kv head, both multiples of 64/W for W in {1, 2, 4}),
//! o_proj's rows of those heads, the sinks of those heads, every expert e with e % W == r
//! (interleaved deal), 1/W of the dense layer-0 intermediate, and REPLICATED copies of the norms,
//! router, indexer, embeddings and lm_head. Two fp32 all-reduces per layer (after o_proj, after the
//! MLP) make the residual identical on every rank.
//!
//! Slicing an EXL3 quad: the trellis is `[k/16][n/16][words]`, `suh` is per input row (k), `svh`
//! per output column (n), and the Hadamards act on 128-blocks of k and of n. A k-slice (o_proj,
//! down_proj) at a 128-multiple is exact: the rank keeps its tile rows and `suh` entries, the
//! output Hadamard sees a partial sum that the all-reduce completes (H is linear). An n-slice at
//! a 128-multiple is exact too (its 128-blocks are whole). A kv-head range that is NOT 128-aligned
//! (DSA at W = 4: 192 columns per rank) is loaded as the covering 128-blocks and compacted after
//! the chain (`col_off`, `nv_cols_copy`).
use anyhow::{ensure, Context, Result};
use cudarc::driver::CudaDevice;
use std::sync::Arc;

use super::gpu::{chain_plan, GpuQuad};
use super::pack::{NaivePack, Quad};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shard {
    pub rank: usize,
    pub world: usize,
}

impl Shard {
    pub fn single() -> Self { Shard { rank: 0, world: 1 } }

    /// This rank's contiguous share [first, first + n) of `total` equal units.
    pub fn span(&self, total: usize) -> Result<(usize, usize)> {
        ensure!(total % self.world == 0, "{total} units do not split over {} ranks", self.world);
        let n = total / self.world;
        Ok((self.rank * n, n))
    }
    /// The query heads of this rank.
    pub fn q_heads(&self, nh: usize) -> Result<(usize, usize)> { self.span(nh) }
    /// The kv heads of this rank (the GQA groups of its query heads).
    pub fn kv_heads(&self, nh: usize, kv: usize) -> Result<(usize, usize)> {
        let (q0, qn) = self.q_heads(nh)?;
        let group = nh / kv;
        ensure!(qn % group == 0, "GQA group {group} does not divide the rank's {qn} query heads");
        Ok((q0 / group, qn / group))
    }
    pub fn owns_expert(&self, e: usize) -> bool { e % self.world == self.rank }
    /// Global ids of this rank's experts, in local order.
    pub fn local_experts(&self, e_total: usize) -> Vec<usize> { (0..e_total).filter(|e| self.owns_expert(*e)).collect() }
    /// local_of[e] = local index or -1 (the `nv_moe_route` ownership map).
    pub fn local_of(&self, e_total: usize) -> Vec<i32> {
        let mut v = vec![-1i32; e_total];
        for (i, e) in self.local_experts(e_total).into_iter().enumerate() { v[e] = i as i32; }
        v
    }
}

fn le_u16(raw: &[u8]) -> Vec<u16> { raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() }

/// A column (n) slice [a, b) of a quad, rounded OUT to 128-blocks. Returns the quad of the covering
/// blocks and the offset of column `a` inside it.
pub fn upload_quad_cols(dev: &Arc<CudaDevice>, pack: &NaivePack, q: &Quad, a: usize, b: usize) -> Result<(GpuQuad, usize)> {
    ensure!(a < b && b <= q.n && a % 16 == 0 && b % 16 == 0, "{}: column slice [{a}, {b}) of n {}", q.name, q.n);
    let (a128, b128) = (a / 128 * 128, (b + 127) / 128 * 128);
    let b128 = b128.min(q.n);
    ensure!(b128 % 128 == 0 || b128 == q.n, "{}: n {} is not a 128-multiple", q.name, q.n);
    let words = q.bits.words();
    let (kb, nb) = (q.k / 16, q.n / 16);
    let tr = le_u16(&q.trellis.read_bytes(&pack.dir)?);
    ensure!(tr.len() == kb * nb * words, "{}: trellis length", q.name);
    let (ta, tb) = (a128 / 16, b128 / 16);
    let mut out = Vec::with_capacity(kb * (tb - ta) * words);
    for r in 0..kb {
        let row = &tr[r * nb * words..(r + 1) * nb * words];
        out.extend_from_slice(&row[ta * words..tb * words]);
    }
    let suh = le_u16(&q.suh.read_bytes(&pack.dir)?);
    let svh = le_u16(&q.svh.read_bytes(&pack.dir)?);
    ensure!(suh.len() == q.k && svh.len() == q.n, "{}: suh/svh length", q.name);
    let n = b128 - a128;
    Ok((GpuQuad {
        name: format!("{}[:, {a128}..{b128}]", q.name),
        tr: dev.htod_sync_copy(&out)?,
        suh: dev.htod_sync_copy(&suh)?,
        svh: dev.htod_sync_copy(&svh[a128..b128])?,
        k: q.k, n, bpb: words, ks: chain_plan(q.k, n).0, a1: chain_plan(q.k, n).1,
    }, a - a128))
}

/// A row (k) slice [a, b) of a quad; a and b must be 128-multiples (exact by linearity of the
/// output Hadamard once the ranks' partial sums are added).
pub fn upload_quad_rows(dev: &Arc<CudaDevice>, pack: &NaivePack, q: &Quad, a: usize, b: usize) -> Result<GpuQuad> {
    ensure!(a < b && b <= q.k && a % 128 == 0 && b % 128 == 0, "{}: row slice [{a}, {b}) of k {} must be 128-aligned", q.name, q.k);
    let words = q.bits.words();
    let (kb, nb) = (q.k / 16, q.n / 16);
    let tr = le_u16(&q.trellis.read_bytes(&pack.dir)?);
    ensure!(tr.len() == kb * nb * words, "{}: trellis length", q.name);
    let (ra, rb) = (a / 16, b / 16);
    let suh = le_u16(&q.suh.read_bytes(&pack.dir)?);
    let svh = le_u16(&q.svh.read_bytes(&pack.dir)?);
    let k = b - a;
    Ok(GpuQuad {
        name: format!("{}[{a}..{b}, :]", q.name),
        tr: dev.htod_sync_copy(&tr[ra * nb * words..rb * nb * words])?,
        suh: dev.htod_sync_copy(&suh[a..b])?,
        svh: dev.htod_sync_copy(&svh)?,
        k, n: q.n, bpb: words, ks: chain_plan(k, q.n).0, a1: chain_plan(k, q.n).1,
    })
}

/// Host-side gather of one expert's trellis / suh / svh (for the rank's concatenated expert set).
pub fn read_quad_host(pack: &NaivePack, q: &Quad) -> Result<(Vec<u16>, Vec<u16>, Vec<u16>)> {
    Ok((le_u16(&q.trellis.read_bytes(&pack.dir).with_context(|| q.name.clone())?),
        le_u16(&q.suh.read_bytes(&pack.dir)?), le_u16(&q.svh.read_bytes(&pack.dir)?)))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn head_and_expert_deals_cover_the_model_exactly() {
        for w in [1usize, 2, 4] {
            let mut q_seen = vec![0u8; 64]; let mut kv8 = vec![0u8; 8]; let mut kv4 = vec![0u8; 4]; let mut ex = vec![0u8; 256];
            for r in 0..w {
                let s = Shard { rank: r, world: w };
                let (q0, qn) = s.q_heads(64).unwrap();
                for h in q0..q0 + qn { q_seen[h] += 1; }
                let (k0, kn) = s.kv_heads(64, 8).unwrap();
                for h in k0..k0 + kn { kv8[h] += 1; }
                // the rank's query heads must map onto its own kv heads (GQA group 8)
                for h in q0..q0 + qn { assert!((h / 8) >= k0 && (h / 8) < k0 + kn, "w{w} r{r} q{h}"); }
                let (k0, kn) = s.kv_heads(64, 4).unwrap();
                for h in k0..k0 + kn { kv4[h] += 1; }
                for h in q0..q0 + qn { assert!((h / 16) >= k0 && (h / 16) < k0 + kn); }
                for e in s.local_experts(256) { ex[e] += 1; }
                let lo = s.local_of(256);
                assert_eq!(lo.iter().filter(|x| **x >= 0).count(), 256 / w);
            }
            assert!(q_seen.iter().all(|c| *c == 1) && kv8.iter().all(|c| *c == 1) && kv4.iter().all(|c| *c == 1) && ex.iter().all(|c| *c == 1), "world {w}");
        }
    }
}

/// `--probe-naive-shard --model-dir <pack>`: the slicing math on ONE GPU, no transport. For real quads of
/// layer 0: a column slice's chain must equal the matching columns of the whole chain (same tiles, same
/// suh, sliced svh — up to the split-K summation order), and the two row slices' chains must SUM to the
/// whole chain (the output Hadamard is linear; f16 rounding of each half's output is the only noise).
pub fn probe_shard(dir: &str) -> Result<()> {
    use super::gpu::{chain, load_module, upload_quad};
    let pack = NaivePack::open(dir)?;
    let dev = CudaDevice::new(0).context("CudaDevice::new(0)")?;
    load_module(&dev)?;
    let m = 4usize;
    let mut ok = true;
    let x_of = |k: usize| -> Vec<u16> { (0..m * k).map(|i| half::f16::from_f32((((i * 2654435761usize) % 1000) as f32 / 500.0 - 1.0) * 0.05).to_bits()).collect() };
    let f32s = |v: &[u16]| -> Vec<f32> { v.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect() };
    // column slices: q_proj (heads), k_proj (kv heads), gate_proj (intermediate)
    for (name, a, b) in [("model.layers.0.self_attn.q_proj", 6144usize, 12288usize), ("model.layers.0.self_attn.k_proj", 384, 768),
                         ("model.layers.0.self_attn.v_proj", 0, 384), ("model.layers.0.mlp.gate_proj", 8192, 16384),
                         ("model.layers.1.self_attn.k_proj", 768, 1536), ("model.layers.1.self_attn.q_proj", 0, 6144)] {
        let q = pack.quad(name)?;
        let whole = upload_quad(&dev, &pack, q)?;
        let (sl, off) = upload_quad_cols(&dev, &pack, q, a, b)?;
        let x = dev.htod_sync_copy(&x_of(q.k))?;
        let yw = f32s(&dev.dtoh_sync_copy(&chain(&dev, &whole, &x, m)?)?);
        let ys = f32s(&dev.dtoh_sync_copy(&chain(&dev, &sl, &x, m)?)?);
        let n = b - a;
        let mut want = Vec::with_capacity(m * n); let mut got = Vec::with_capacity(m * n);
        for r in 0..m { want.extend_from_slice(&yw[r * q.n + a..r * q.n + b]); got.extend_from_slice(&ys[r * sl.n + off..r * sl.n + off + n]); }
        let rel = super::cpu::rel_l2(&got, &want);
        println!("  cols {name} [{a}, {b}) (slice n {} off {off}, ks {} vs {}): rel-L2 {rel:.3e}", sl.n, sl.ks, whole.ks);
        if rel > 1e-3 { ok = false; }
    }
    // row slices: o_proj (attention output), down_proj (intermediate)
    for (name, mid) in [("model.layers.0.self_attn.o_proj", 4096usize), ("model.layers.0.mlp.down_proj", 8192usize)] {
        let q = pack.quad(name)?;
        let whole = upload_quad(&dev, &pack, q)?;
        let xa = x_of(q.k);
        let x = dev.htod_sync_copy(&xa)?;
        let yw = f32s(&dev.dtoh_sync_copy(&chain(&dev, &whole, &x, m)?)?);
        let mut sum = vec![0f32; m * q.n];
        for (a, b) in [(0, mid), (mid, q.k)] {
            let sl = upload_quad_rows(&dev, &pack, q, a, b)?;
            let mut xs = Vec::with_capacity(m * (b - a));
            for r in 0..m { xs.extend_from_slice(&xa[r * q.k + a..r * q.k + b]); }
            let xd = dev.htod_sync_copy(&xs)?;
            let y = f32s(&dev.dtoh_sync_copy(&chain(&dev, &sl, &xd, m)?)?);
            for i in 0..m * q.n { sum[i] += y[i]; }
        }
        let rel = super::cpu::rel_l2(&sum, &yw);
        println!("  rows {name} halves summed vs whole: rel-L2 {rel:.3e}");
        if rel > 1e-2 { ok = false; }
    }
    if !ok { anyhow::bail!("NAIVE_SHARD_FAIL"); }
    println!("NAIVE_SHARD_OK");
    Ok(())
}
