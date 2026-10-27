//! Half-integer-K EXL3 trellis decode (exllamav3 1.5.x `frac.cu`), CPU reference.
//!
//! A 256-weight tile is a ring of `16*K` bits (`K` half-integer -> a whole number of 32-bit words:
//! 3.5 bpw = 56 u16 = 28 u32 per tile). Weight at ring position `i` has `D(i) = KA + bit(i mod 16)
//! of MASK` bits (the quantizer uses MASK = 0xAAAA: 3,4,3,4,... for 3.5 bpw); its 16-bit codebook
//! window is ring bits `[S(i) - 16, S(i))` with `S(i)` the inclusive prefix sum of `D`. Words are
//! the tile's int16 pairs read as little-endian u32 (the C++ casts `(const uint32_t*) packed`),
//! bits MSB-first within a word.
//!
//! Relation to the engine's integer decoder (`crate::exl3::decode_block`, bit-exact vs
//! `ext.reconstruct` on the 3.05 pack): ring position `i` IS the temporal index `t`
//! (`frac_perm^-1 == exl3::tmap` on all 256 elements), and with MASK = 0 the window arithmetic
//! here reproduces `decode_block` bit for bit (`scripts/naive/frac_ref.py check`; the unit tests
//! below). The codebook (mul1), `suh` / `svh` and the Hadamard transform are unchanged.
//!
//! Status: CPU oracle. The GPU kernels (S3) carry the same per-position window table.

use crate::exl3::{mul1_codebook, tmap_table};

/// The quantizer's extra-bit pattern for every half-integer rate (`exl3_lib/quantize.py frac_k`).
pub const FRAC_MASK: u32 = 0xAAAA;

/// Bitrate of one EXL3 tensor: integer `1..=8`, or half-integer `ka + 0.5` (mul1 only, `ka <= 3`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitsK {
    Int(usize),
    Half { ka: usize, mask: u32 },
}

impl BitsK {
    /// From a trellis tile width in u16 words (`shape[2]`): `16*K` words.
    pub fn from_words(words: usize) -> anyhow::Result<BitsK> {
        anyhow::ensure!(words > 0 && words % 8 == 0, "trellis width {words} is not 8*n words");
        if words % 16 == 0 {
            let bits = words / 16;
            anyhow::ensure!((1..=8).contains(&bits), "trellis width {words}: bits {bits} not in 1..8");
            Ok(BitsK::Int(bits))
        } else {
            let ka = words / 16; // words = 16*ka + 8
            anyhow::ensure!((1..=3).contains(&ka), "trellis width {words}: half-K needs ka 1..3 (got {ka})");
            Ok(BitsK::Half { ka, mask: FRAC_MASK })
        }
    }
    /// u16 words per 256-weight tile.
    pub fn words(self) -> usize {
        match self {
            BitsK::Int(b) => 16 * b,
            BitsK::Half { ka, mask } => 16 * ka + mask.count_ones() as usize,
        }
    }
    /// Bits per weight as a float (3.5 for the Naive pack).
    pub fn bpw(self) -> f64 {
        self.words() as f64 / 16.0
    }
    pub fn is_half(self) -> bool { matches!(self, BitsK::Half { .. }) }
}

/// Bits of the weight at ring position `i`.
#[inline]
fn frac_d(i: usize, ka: usize, mask: u32) -> usize {
    ka + ((mask >> (i & 15)) & 1) as usize
}

/// Window END (exclusive) of ring position `i`: the inclusive prefix sum of `D` over the ring.
/// `bpb` = bits per 16 weights (`16*ka + popcount(mask)`).
#[inline]
fn frac_s(i: usize, ka: usize, mask: u32, bpb: usize) -> usize {
    let mut s = (i >> 4) * bpb;
    for j in 0..=(i & 15) {
        s += frac_d(j, ka, mask);
    }
    s
}

/// The 256 window ends `S(i)` for a (ka, mask) pair — the table the GPU kernels carry.
pub fn window_ends(ka: usize, mask: u32) -> [u32; 256] {
    let bpb = 16 * ka + mask.count_ones() as usize;
    let mut t = [0u32; 256];
    for (i, slot) in t.iter_mut().enumerate() {
        *slot = frac_s(i, ka, mask, bpb) as u32;
    }
    t
}

/// 16-bit window starting at ring bit `start` (mod the ring) of `words` (little-endian u32 words,
/// bits MSB-first within a word) — `frac.cu frac_window`.
#[inline]
fn frac_window(words: &[u32], start: usize) -> u32 {
    let nw = words.len();
    let r = nw * 32;
    let start = start % r;
    let wi = start >> 5;
    let o = start & 31;
    let v = ((words[wi] as u64) << 32) | words[(wi + 1) % nw] as u64;
    ((v >> (48 - o)) & 0xFFFF) as u32
}

/// Decode one 16x16 block at a half-integer rate. `ring` = the block's `16*ka + popcount(mask)`
/// u16 words as little-endian u32s; `out[t]` = fp16 bit pattern at temporal index `t`.
pub fn decode_block_frac(ring: &[u32], ka: usize, mask: u32, cb: &[u16], out: &mut [u16; 256]) {
    let bpb = 16 * ka + mask.count_ones() as usize;
    debug_assert_eq!(ring.len() * 2, bpb, "ring words vs bits per 16 weights");
    for t in 0..256 {
        let end = frac_s(t, ka, mask, bpb);
        // `end >= 16` always (D >= 1 and t = 0 has S(0) = D(0) >= 1 — the window wraps, so go via the ring length)
        let start = end + ring.len() * 32 - 16;
        out[t] = cb[frac_window(ring, start) as usize];
    }
}

/// Codebook INDICES only (the fixture format of `scripts/naive/frac_ref.py`).
pub fn decode_block_frac_indices(ring: &[u32], ka: usize, mask: u32) -> [u16; 256] {
    let bpb = 16 * ka + mask.count_ones() as usize;
    let mut out = [0u16; 256];
    for t in 0..256 {
        let end = frac_s(t, ka, mask, bpb);
        let start = end + ring.len() * 32 - 16;
        out[t] = frac_window(ring, start) as u16;
    }
    out
}

/// Reinterpret `[.., 16*K]` int16 as `[.., 8*K]` little-endian u32 per block (K half-integer ok).
pub fn trellis_to_ring(trellis: &[i16], words: usize) -> Vec<u32> {
    let nblocks = trellis.len() / words;
    let w32 = words / 2;
    let mut ring = Vec::with_capacity(nblocks * w32);
    for b in 0..nblocks {
        let base = b * words;
        for j in 0..w32 {
            let lo = trellis[base + 2 * j] as u16 as u32;
            let hi = trellis[base + 2 * j + 1] as u16 as u32;
            ring.push(lo | (hi << 16));
        }
    }
    ring
}

/// Decode a whole trellis tensor `[kb, nb, words]` int16 into `W [kb*16, nb*16]` fp16 bit patterns
/// (row-major) — `ext.reconstruct` before `suh` / `svh`. Integer rates go through the engine's
/// decoder; half rates through [`decode_block_frac`].
pub fn decode_trellis_host(trellis: &[i16], kb: usize, nb: usize, k: BitsK) -> Vec<u16> {
    let cb = mul1_codebook();
    let words = k.words();
    let w32 = words / 2;
    let ring = trellis_to_ring(trellis, words);
    let mut out = vec![0u16; kb * 16 * nb * 16];
    let mut block = [0u16; 256];
    let tmap = tmap_table();
    for kk in 0..kb {
        for n in 0..nb {
            let b = kk * nb + n;
            let r = &ring[b * w32..(b + 1) * w32];
            match k {
                // PORT (NAIVEPORT): master's decode_block took `bits` at the naive branch's base; EXL27C changed the
                // parameter to k2 = 2*bits (odd k2 = the half-K path). This file is otherwise byte-identical to
                // naive-n05; this is the one deliberate adaptation, so the CPU oracle keeps the integer decoder's
                // `bits` semantics.
                BitsK::Int(bits) => crate::exl3::decode_block(r, bits * 2, &cb, &mut block),
                BitsK::Half { ka, mask } => decode_block_frac(r, ka, mask, &cb, &mut block),
            }
            for row in 0..16 {
                let orow = (kk * 16 + row) * (nb * 16);
                for c in 0..16 {
                    out[orow + n * 16 + c] = block[tmap[row][c]];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: &mut u64) -> u32 {
        // xorshift64* — deterministic, no crate
        *seed ^= *seed >> 12;
        *seed ^= *seed << 25;
        *seed ^= *seed >> 27;
        (seed.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    #[test]
    fn bits_from_words() {
        assert_eq!(BitsK::from_words(56).unwrap(), BitsK::Half { ka: 3, mask: FRAC_MASK });
        assert_eq!(BitsK::from_words(96).unwrap(), BitsK::Int(6));
        assert_eq!(BitsK::from_words(48).unwrap(), BitsK::Int(3));
        assert_eq!(BitsK::from_words(56).unwrap().words(), 56);
        assert!((BitsK::from_words(56).unwrap().bpw() - 3.5).abs() < 1e-12);
        assert!(BitsK::from_words(72).is_err(), "ka 4 + half is not a quantizer rate");
        assert!(BitsK::from_words(20).is_err());
    }

    /// With MASK = 0 the half-K window arithmetic IS the integer decoder (the layout proof).
    #[test]
    fn mask_zero_reproduces_the_integer_decoder() {
        let cb = mul1_codebook();
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for bits in 2..=6usize {
            for _ in 0..40 {
                let ring: Vec<u32> = (0..8 * bits).map(|_| rng(&mut seed)).collect();
                let mut a = [0u16; 256];
                let mut b = [0u16; 256];
                // PORT (NAIVEPORT): k2 = 2*bits (EXL27C convention; see decode_trellis_host above).
                crate::exl3::decode_block(&ring, bits * 2, &cb, &mut a);
                decode_block_frac(&ring, bits, 0, &cb, &mut b);
                assert_eq!(a, b, "bits {bits}");
            }
        }
    }

    /// The window ends: period-16 structure, 56 bits per 16 weights at 3.5 bpw, max end = ring length.
    #[test]
    fn window_ends_structure() {
        let t = window_ends(3, FRAC_MASK);
        assert_eq!(t[15], 56);
        assert_eq!(t[255], 896);
        for i in 0..256 {
            let d = if i == 0 { t[0] } else { t[i] - t[i - 1] };
            assert_eq!(d, if i % 2 == 1 { 4 } else { 3 }, "position {i}: MASK 0xAAAA gives the extra bit to odd positions");
        }
    }

    /// Fixtures from the pure-Python port of frac.cu (`scripts/naive/frac_ref.py fixtures`).
    #[test]
    fn matches_the_python_frac_reference() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/naive/frac_tiles.json");
        let raw = std::fs::read_to_string(path).expect("fixture tests/fixtures/naive/frac_tiles.json");
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let cases = v["cases"].as_array().unwrap();
        assert!(cases.len() >= 10);
        let mut half_cases = 0;
        for c in cases {
            let ka = c["ka"].as_u64().unwrap() as usize;
            let mask = c["mask"].as_u64().unwrap() as u32;
            let words: Vec<i16> = c["words_u16"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u16 as i16).collect();
            let want: Vec<u16> = c["indices"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u16).collect();
            let ring = trellis_to_ring(&words, words.len());
            let got = decode_block_frac_indices(&ring, ka, mask);
            assert_eq!(&got[..], &want[..], "ka {ka} mask {mask:#x}");
            if mask != 0 { half_cases += 1; }
        }
        assert!(half_cases >= 8, "the fixture must exercise the half-K path");
    }

    /// Whole-tensor decode places block (kb, nb) temporal index t at tmap(r, c) — same scatter as
    /// `exl3::decode_trellis_host`, checked by running both on an integer-rate tensor.
    #[test]
    fn whole_tensor_decode_matches_the_engine_at_integer_rates() {
        let mut seed = 42u64;
        let (kb, nb, bits) = (2usize, 3usize, 3usize);
        let trellis: Vec<i16> = (0..kb * nb * 16 * bits).map(|_| rng(&mut seed) as u16 as i16).collect();
        // PORT (NAIVEPORT): master's decode_trellis_host takes k2 = 2*bits (EXL27C rename).
        let a = crate::exl3::decode_trellis_host(&trellis, kb, nb, bits * 2);
        let b = decode_trellis_host(&trellis, kb, nb, BitsK::Int(bits));
        assert_eq!(a, b);
        // and the half path at least has the right shape and is finite
        let trellis: Vec<i16> = (0..kb * nb * 56).map(|_| rng(&mut seed) as u16 as i16).collect();
        let h = decode_trellis_host(&trellis, kb, nb, BitsK::from_words(56).unwrap());
        assert_eq!(h.len(), kb * 16 * nb * 16);
        assert!(h.iter().all(|x| half::f16::from_bits(*x).is_finite()));
    }
}

/// `--probe-naive-oracle --model-dir <pack> --oracle-dir <ref>`: our decode vs pinned
/// `ext.reconstruct` dumps (`scripts/naive/gen_frac_ref.py`), BITWISE, whole matrices
/// (the S0 GPU gate; exl3::probe_oracle's shape for the half-K pack).
pub fn probe_oracle(dir: &str, ref_dir: &str) -> anyhow::Result<()> {
    use anyhow::{bail, Context};
    #[derive(serde::Deserialize)]
    struct RefEntry { module: String, k: usize, n: usize, words: usize, file: String }
    let ref_dir = ref_dir.trim_end_matches('/');
    let entries: Vec<RefEntry> = serde_json::from_str(&std::fs::read_to_string(format!("{ref_dir}/manifest.json"))
        .with_context(|| format!("{ref_dir}/manifest.json missing — run scripts/naive/gen_frac_ref.py on a GPU box"))?)?;
    let pack = super::pack::NaivePack::open(dir)?;
    let (mut matrices, mut blocks, mut bad_values, mut bad_matrices) = (0usize, 0usize, 0usize, 0usize);
    let t0 = std::time::Instant::now();
    for e in &entries {
        let q = pack.quad(&e.module)?;
        anyhow::ensure!(q.k == e.k && q.n == e.n && q.bits.words() == e.words, "{}: pack {}x{} words {} vs ref {}x{} words {}",
                        e.module, q.k, q.n, q.bits.words(), e.k, e.n, e.words);
        let raw = q.trellis.read_bytes(&pack.dir)?;
        let tr: Vec<i16> = raw.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        let ours = decode_trellis_host(&tr, q.k / 16, q.n / 16, q.bits);
        let want = std::fs::read(format!("{ref_dir}/{}", e.file)).with_context(|| e.file.clone())?;
        if want.len() != ours.len() * 2 { bail!("{}: ref {} B vs ours {} B", e.module, want.len(), ours.len() * 2); }
        let bad = ours.iter().zip(want.chunks_exact(2)).filter(|(a, b)| a.to_le_bytes() != [b[0], b[1]]).count();
        matrices += 1; blocks += (q.k / 16) * (q.n / 16); bad_values += bad; if bad > 0 { bad_matrices += 1; }
        println!("  {:55} {:5}x{:6} {:?}: {} mismatched values{}", e.module, q.k, q.n, q.bits, bad, if bad == 0 { "" } else { "  <-- FAIL" });
    }
    println!("NAIVE_ORACLE_{} matrices {} blocks {} bad_values {} bad_matrices {} in {:.1} s",
             if bad_values == 0 { "OK" } else { "FAIL" }, matrices, blocks, bad_values, bad_matrices, t0.elapsed().as_secs_f64());
    if bad_values > 0 { bail!("decode is not bitwise against ext.reconstruct"); }
    Ok(())
}
