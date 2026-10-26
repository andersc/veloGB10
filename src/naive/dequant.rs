//! Host dequantization of the Naive pack's EXL3 quads (S2b): the ORIGINAL-basis weight the HF
//! reference would multiply by, from the stored trellis + `suh` / `svh`.
//!
//! exllamav3 `LinearEXL3.get_weight_tensor` (modules/quant/exl3.py): `w = reconstruct(trellis)`
//! (the decoded `[in, out]` fp16 matrix in the ROTATED basis, `exl3::decode_trellis_host` /
//! `naive::frac::decode_trellis_host`), then `w = H_k w` (left Hadamard: every column transformed
//! along `in` in 128-blocks), `w *= suh[:, None]`, `w = w H_n` (right Hadamard along `out`),
//! `w *= svh[None, :]`. The 128-block Hadamard is the ORTHONORMAL Sylvester transform
//! (`had_r_128`: butterfly then `/ sqrt(128)`; the engine's `exl3_had128` normalizes the same
//! way). Forward equivalence: `y = ((x * suh) H_k) W_q H_n * svh = x . W_orig`.
//!
//! Structural falsifiers of the chain on REAL tensors (no GPU, no oracle needed): the quantizer saw
//! zeros where the pack pads — `v_proj` rows 128..191 of every kv head and `weights_proj` output
//! columns 16..127 — so a correct chain dequantizes those regions to ~0 and a wrong transform
//! smears energy into them. `--probe-naive-dequant` reports both ratios.

use anyhow::{ensure, Context, Result};

use super::frac::decode_trellis_host;
use super::pack::{NaivePack, Quad, Role};

const HNORM: f32 = 0.088_388_347_648_318_45; // 1/sqrt(128)

/// In-place orthonormal Walsh-Hadamard transform of one 128-element block (`exl3_had128` order).
pub fn had128(v: &mut [f32]) {
    debug_assert_eq!(v.len(), 128);
    let mut len = 1usize;
    while len < 128 {
        for i in 0..128 {
            if i & len == 0 {
                let (a, b) = (v[i], v[i + len]);
                v[i] = a + b;
                v[i + len] = a - b;
            }
        }
        len <<= 1;
    }
    for x in v.iter_mut() { *x *= HNORM; }
}

/// Decode a quad's trellis to the rotated-basis fp16 matrix `[k, n]` as f32 (no suh/svh/Hadamard).
pub fn decode_rotated(pack: &NaivePack, q: &Quad) -> Result<Vec<f32>> {
    let raw = q.trellis.read_bytes(&pack.dir)?;
    let tr: Vec<i16> = raw.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
    ensure!(tr.len() == (q.k / 16) * (q.n / 16) * q.bits.words(), "{}: trellis words {} vs geometry", q.name, tr.len());
    let bits = decode_trellis_host(&tr, q.k / 16, q.n / 16, q.bits);
    Ok(bits.into_iter().map(|b| half::f16::from_bits(b).to_f32()).collect())
}

fn read_f16(pack: &NaivePack, m: &crate::exl3::TensorMeta) -> Result<Vec<f32>> {
    ensure!(m.dtype == "F16", "{}: dtype {} (wanted F16)", m.name, m.dtype);
    Ok(m.read_bytes(&pack.dir)?.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect())
}

/// The original-basis weight `W_orig [k, n]` (row = input feature, column = output feature, i.e.
/// `y = x @ W_orig`; nn.Linear's `weight` is its transpose).
pub fn dequant_quad(pack: &NaivePack, q: &Quad) -> Result<Vec<f32>> { dequant_quad_opt(pack, q, true) }

/// `dequant_quad` with the output scale optional: `apply_svh = false` stops after the right
/// Hadamard. The padded-region falsifier runs on THAT matrix — the quantizer stores svh = 0 for
/// all-zero output columns, so with svh applied even a wrong Hadamard side would read as zero.
pub fn dequant_quad_opt(pack: &NaivePack, q: &Quad, apply_svh: bool) -> Result<Vec<f32>> {
    let (k, n) = (q.k, q.n);
    ensure!(k % 128 == 0 && n % 128 == 0, "{}: dims must be 128-aligned (k {k} n {n})", q.name);
    let mut w = decode_rotated(pack, q)?;
    let suh = read_f16(pack, &q.suh).with_context(|| format!("{} suh", q.name))?;
    let svh = read_f16(pack, &q.svh).with_context(|| format!("{} svh", q.name))?;
    ensure!(suh.len() == k && svh.len() == n, "{}: suh/svh lengths", q.name);
    // left Hadamard along k (per column), then * suh[k]
    let mut col = vec![0f32; 128];
    for c in 0..n {
        for kb in (0..k).step_by(128) {
            for i in 0..128 { col[i] = w[(kb + i) * n + c]; }
            had128(&mut col);
            for i in 0..128 { w[(kb + i) * n + c] = col[i] * suh[kb + i]; }
        }
    }
    // right Hadamard along n (per row), then * svh[n]
    for r in 0..k {
        let row = &mut w[r * n..(r + 1) * n];
        for nb in (0..n).step_by(128) {
            had128(&mut row[nb..nb + 128]);
        }
        if apply_svh { for c in 0..n { row[c] *= svh[c]; } }
    }
    Ok(w)
}

/// `W_orig` transposed into nn.Linear layout `[out, in]` (what `cpu::Mat` holds).
pub fn dequant_linear(pack: &NaivePack, q: &Quad) -> Result<Vec<f32>> {
    let w = dequant_quad(pack, q)?;
    let (k, n) = (q.k, q.n);
    let mut t = vec![0f32; n * k];
    for r in 0..k { for c in 0..n { t[c * k + r] = w[r * n + c]; } }
    Ok(t)
}

/// RMS of a slice.
fn rms(v: &[f32]) -> f32 {
    if v.is_empty() { return 0.0; }
    (v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64).sqrt() as f32
}

/// `--probe-naive-dequant --model-dir <pack> [--layer L]`: dequantize layer L's attention,
/// indexer (DSA) and layer-0 dense MLP quads; report RMS of the live and padded regions.
pub fn probe(dir: &str, layer: usize) -> Result<()> {
    let p = NaivePack::open(dir)?;
    let c = &p.cfg;
    ensure!(layer < c.num_layers, "layer {layer} out of range");
    println!("NAIVE DEQUANT {} layer {layer} ({:?})", p.dir, c.attn[layer]);
    let t0 = std::time::Instant::now();
    let mut worst_pad_ratio = 0f32;
    for q in p.quads.values().filter(|q| q.layer == Some(layer) && q.expert.is_none()) {
        let w_full = dequant_quad(&p, q)?; // [k, n] with svh
        let w = dequant_quad_opt(&p, q, false)?; // before svh: the falsifier's subject
        let (k, n) = (q.k, q.n);
        let live_rms = rms(&w_full);
        let svh = read_f16(&p, &q.svh)?;
        let mut line = format!("  {:55} k {:5} n {:6} {:?}: rms {:.4e}", q.name, k, n, q.bits, live_rms);
        match q.role {
            Role::AttnV => {
                // per kv head: output cols [g*192, g*192+128) live, [g*192+128, g*192+192) padded
                let (hd, vd) = (c.head_dim, c.v_head_dim);
                let kv = n / hd;
                let (mut live, mut pad) = (Vec::new(), Vec::new());
                for r in 0..k { for g in 0..kv { for d in 0..hd {
                    let x = w[r * n + g * hd + d];
                    if d < vd { live.push(x) } else { pad.push(x) }
                } } }
                let ratio = rms(&pad) / rms(&live).max(1e-30);
                worst_pad_ratio = worst_pad_ratio.max(ratio);
                let svh_pad: Vec<f32> = (0..kv).flat_map(|g| (vd..hd).map(move |d| g * hd + d)).map(|c| svh[c]).collect();
                line += &format!("  | pre-svh padded V rows: rms {:.3e} vs live {:.3e} (ratio {:.2e}); svh on padded cols rms {:.2e}", rms(&pad), rms(&live), ratio, rms(&svh_pad));
            }
            Role::IdxWeights => {
                let live: Vec<f32> = (0..k).flat_map(|r| (0..c.index_n_heads).map(move |cc| (r, cc))).map(|(r, cc)| w[r * n + cc]).collect();
                let pad: Vec<f32> = (0..k).flat_map(|r| (c.index_n_heads..n).map(move |cc| (r, cc))).map(|(r, cc)| w[r * n + cc]).collect();
                let ratio = rms(&pad) / rms(&live).max(1e-30);
                worst_pad_ratio = worst_pad_ratio.max(ratio);
                line += &format!("  | pre-svh padded cols 16..: rms {:.3e} vs live {:.3e} (ratio {:.2e}); svh on padded cols rms {:.2e}", rms(&pad), rms(&live), ratio, rms(&svh[c.index_n_heads..n]));
            }
            _ => {}
        }
        println!("{line}");
    }
    println!("  worst padded/live RMS ratio {:.2e} (the padding carries only the 3.5-bpw quantization noise of its 128-block, ~0.06; a wrong Hadamard side or scale puts signal there, ~1)", worst_pad_ratio);
    println!("NAIVE_DEQUANT_{} in {:.1} s", if worst_pad_ratio < 0.25 { "OK" } else { "SUSPECT" }, t0.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn had128_is_an_orthonormal_involution() {
        let mut v: Vec<f32> = (0..128).map(|i| ((i * 37) % 11) as f32 - 5.0).collect();
        let orig = v.clone();
        let e0: f32 = orig.iter().map(|x| x * x).sum();
        had128(&mut v);
        let e1: f32 = v.iter().map(|x| x * x).sum();
        assert!((e0 - e1).abs() < 1e-3 * e0, "orthonormal: energy preserved ({e0} vs {e1})");
        had128(&mut v);
        for i in 0..128 { assert!((v[i] - orig[i]).abs() < 1e-4, "involution at {i}"); }
        let mut d = vec![0f32; 128]; d[0] = 1.0;
        had128(&mut d);
        assert!(d.iter().all(|x| (x - HNORM).abs() < 1e-7), "a delta maps to a uniform 1/sqrt(128) row");
    }

    /// Real pack: layer 0's padded V rows and padded indexer columns dequantize to ~0.
    #[test]
    fn real_pack_padding_dequantizes_to_zero() {
        let dir = "/mnt/models/NaiveAI/Naive-N0.5-Flash-exl3-3.5bpw-vllm";
        if !std::path::Path::new(dir).join("model.safetensors.index.json").is_file() { eprintln!("skip: pack not present"); return; }
        let p = NaivePack::open(dir).unwrap();
        let q = p.quad("model.layers.0.self_attn.indexer.weights_proj").unwrap();
        let w = dequant_quad_opt(&p, q, false).unwrap(); // before svh (svh of padded columns is 0 by construction)
        let (k, n) = (q.k, q.n);
        let live = rms(&(0..k).flat_map(|r| (0..16).map(move |c| (r, c))).map(|(r, c)| w[r * n + c]).collect::<Vec<_>>());
        let pad = rms(&(0..k).flat_map(|r| (16..n).map(move |c| (r, c))).map(|(r, c)| w[r * n + c]).collect::<Vec<_>>());
        assert!(live > 1e-4, "live columns carry weight (rms {live})");
        // 3.5 bpw quantization noise spreads evenly over a 128-block; the padded part carries only that
        // noise (measured 0 and 0.058 of the live rms on layer 0). A wrong Hadamard side or scale puts
        // signal there (ratio ~1).
        assert!(pad < 0.25 * live, "padded columns must carry noise only: pad rms {pad} vs live {live}");
        let q = p.quad("model.layers.0.self_attn.v_proj").unwrap();
        let w = dequant_quad_opt(&p, q, false).unwrap();
        let (k, n, hd, vd) = (q.k, q.n, 192usize, 128usize);
        let mut lv = Vec::new(); let mut pv = Vec::new();
        for r in 0..k { for g in 0..n / hd { for d in 0..hd { let x = w[r * n + g * hd + d]; if d < vd { lv.push(x) } else { pv.push(x) } } } }
        assert!(rms(&pv) < 0.25 * rms(&lv), "padded V rows must carry noise only: {} vs {}", rms(&pv), rms(&lv));
    }
}
