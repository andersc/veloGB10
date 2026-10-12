//! Alignment-safe reads of file-derived bytes into wider plain-old-data types.
//!
//! WHY (S-B9-REL-PPL, 2026-10-09): a safetensors tensor's data offset is `8 + header_len +
//! data_offsets[0]`. The format does not require the header to be padded to a multiple of 8, and a
//! tensor's offset inside the buffer depends on the byte sizes of the tensors before it, so a
//! `&[u8]` taken from a file (an mmap, or a `Vec<u8>` read from it) is NOT guaranteed to be aligned
//! for `u16` / `u32` / `f32`. `bytemuck::cast_slice::<u8, T>` checks that at run time and PANICS
//! ("cast_slice>TargetAlignmentGreaterAndInputNotAligned"). The v0.7.3 EXL3 pack's shard 1 has a
//! 5,732,415-byte header (header % 8 == 7): all 43,301 of its tensors sit at an odd file offset.
//!
//! These helpers borrow when the bytes happen to be aligned (zero copy, identical to `cast_slice`)
//! and copy into an aligned `Vec` when they are not. A byte length that is not a multiple of
//! `size_of::<T>()` is still a hard error (same as `cast_slice`: that is a corrupt tensor, not an
//! alignment accident).

use std::borrow::Cow;

use bytemuck::Pod;

/// `bytes` as `[T]`: borrowed if aligned for `T`, else an aligned copy.
///
/// Panics if `bytes.len()` is not a multiple of `size_of::<T>()`.
pub fn pod_cow<T: Pod>(bytes: &[u8]) -> Cow<'_, [T]> {
    let sz = std::mem::size_of::<T>();
    assert!(sz > 0 && bytes.len() % sz == 0,
            "pod_cow: {} bytes is not a whole number of {}-byte elements", bytes.len(), sz);
    match bytemuck::try_cast_slice::<u8, T>(bytes) {
        Ok(s) => Cow::Borrowed(s),
        // TargetAlignmentGreaterAndInputNotAligned is the only error left after the length check.
        Err(_) => Cow::Owned(pod_vec_copy::<T>(bytes)),
    }
}

/// `bytes` as an owned, aligned `Vec<T>` (always a copy).
///
/// Panics if `bytes.len()` is not a multiple of `size_of::<T>()`.
pub fn pod_vec<T: Pod>(bytes: &[u8]) -> Vec<T> {
    let sz = std::mem::size_of::<T>();
    assert!(sz > 0 && bytes.len() % sz == 0,
            "pod_vec: {} bytes is not a whole number of {}-byte elements", bytes.len(), sz);
    pod_vec_copy::<T>(bytes)
}

fn pod_vec_copy<T: Pod>(bytes: &[u8]) -> Vec<T> {
    let n = bytes.len() / std::mem::size_of::<T>();
    let mut out: Vec<T> = vec![T::zeroed(); n];
    bytemuck::cast_slice_mut::<T, u8>(&mut out).copy_from_slice(bytes);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `payload` copied into a buffer at an offset that is NOT 4-aligned: the shape of a
    /// safetensors tensor in a shard whose header length is not a multiple of 8. One allocation,
    /// so the misalignment is a property of the pointer, asserted by the callers.
    fn at_odd_offset(payload: &[u8]) -> (Vec<u8>, usize) {
        let mut buf = vec![0xEEu8; payload.len() + 8];
        let base = buf.as_ptr() as usize;
        let s = (1..4).find(|s| (base + s) % 4 != 0).expect("some offset in 1..4 is misaligned");
        buf[s..s + payload.len()].copy_from_slice(payload);
        (buf, s)
    }

    #[test]
    fn misaligned_input_makes_bare_cast_slice_fail_but_pod_cow_reads_it() {
        let want: Vec<f32> = vec![1.0, -2.5, 3.25e-7, f32::MAX, 0.0, 65504.0];
        let payload: &[u8] = bytemuck::cast_slice(&want);
        let (buf, s) = at_odd_offset(payload);
        let sl = &buf[s..s + payload.len()];
        assert_ne!((sl.as_ptr() as usize) % std::mem::align_of::<f32>(), 0, "precondition");
        // the hazard: a bare cast_slice panics here (try_ form so the test can assert it)
        assert!(bytemuck::try_cast_slice::<u8, f32>(sl).is_err());
        // the fix
        let got = pod_cow::<f32>(sl);
        assert!(matches!(got, Cow::Owned(_)), "a misaligned slice must be copied");
        assert_eq!(&got[..], &want[..]);
        assert_eq!(pod_vec::<f32>(sl), want);
    }

    #[test]
    fn aligned_input_is_borrowed_not_copied() {
        let want: Vec<u16> = vec![1, 2, 0xFFFF, 0x8000];
        let payload: &[u8] = bytemuck::cast_slice(&want);
        // a Vec<u16>'s bytes are 2-aligned by construction
        let got = pod_cow::<u16>(payload);
        assert!(matches!(got, Cow::Borrowed(_)));
        assert_eq!(&got[..], &want[..]);
    }

    #[test]
    fn bf16_and_u16_widths_cover_the_odd_offset_case() {
        // the pack's real failure: BF16 / F16 / I16 tensors at an odd file offset (offset % 8 == 7)
        let vals: Vec<half::bf16> = [0.5f32, -1.0, 3.0, 1e-3].iter().map(|&x| half::bf16::from_f32(x)).collect();
        let payload: &[u8] = bytemuck::cast_slice(&vals);
        let mut buf = vec![0u8; payload.len() + 8];
        let base = buf.as_ptr() as usize;
        let s = if base % 2 == 0 { 1 } else { 0 }; // an odd (2-misaligned) start address
        buf[s..s + payload.len()].copy_from_slice(payload);
        let odd = &buf[s..s + payload.len()];
        assert_eq!((odd.as_ptr() as usize) % 2, 1, "precondition: odd address");
        assert!(bytemuck::try_cast_slice::<u8, half::bf16>(odd).is_err());
        assert_eq!(&pod_cow::<half::bf16>(odd)[..], &vals[..]);
        let ints: Vec<u16> = vec![0, 1, 0xFFFF, 0x1234];
        let ip: &[u8] = bytemuck::cast_slice(&ints);
        buf[s..s + ip.len()].copy_from_slice(ip);
        assert_eq!(&pod_cow::<u16>(&buf[s..s + ip.len()])[..], &ints[..]);
    }

    #[test]
    #[should_panic(expected = "not a whole number")]
    fn a_ragged_length_is_still_an_error() {
        let b = [0u8; 7];
        let _ = pod_cow::<u16>(&b);
    }

    #[test]
    fn empty_input_is_an_empty_slice() {
        assert!(pod_cow::<f32>(&[]).is_empty());
        assert!(pod_vec::<u16>(&[]).is_empty());
    }
}
