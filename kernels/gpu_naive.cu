// =============================================================================
// kernels/gpu_naive.cu — Naive-N0.5-Flash family kernels (PLAN/NAIVE_N05_PORT.md, S3).
//
// Owner directive 2026-10-16: Flash-Next code is off-limits. Everything here is a COPY of the
// engine's EXL3 kernels (kernels/exl3_bench.cu, S-A3-c..n) re-parameterized for this pack:
//   * half-integer trellis rates (3.5 bpw = 56 u16 words per 16x16 tile) next to the integer ones
//     (the 6-bit lm_head = 96 words). The template parameter is BPB, the bits per 16 weights (=
//     u16 words per tile): 48 / 56 / 64 / 80 / 96. The ONLY place a half rate differs from an
//     integer one is the window END of temporal index t (nv_win_end): integer K -> (t+1)*K,
//     half K (quantizer mask 0xAAAA: 3,4,3,4,... bits) -> S(t) = (t>>4)*BPB + ka*j + (j>>1),
//     j = (t&15)+1. Verified on the CPU (src/naive/frac.rs) against the exllamav3 frac.cu
//     semantics; with MASK = 0 the arithmetic is the engine's integer decoder bit for bit.
//   * the codebook (mul1), the dp4a decode, the tmap fragment placement, the orthonormal 128-block
//     Hadamard, the m16 x N128 HMMA tile, the K-ascending fp32 accumulation and the split-K
//     fixed-order combine are the engine's verbatim (batch invariance: AGENTS §2.4).
//
// Target: sm_121 baseline (fp16 mma.m16n8k16, dp4a, shuffles). Loaded by src/naive/gpu.rs only.
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>

#define NV_MUL1 0x83DCD12Du

#ifndef KERNEL_BUILD_ID
#define KERNEL_BUILD_ID 0ULL
#endif
extern "C" __global__ void nv_kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }

// PDL entry (every entry kernel: wait for the programmatic predecessor, release dependents; both
// are no-ops on a plain launch).
#define NV_PDL_ENTRY() do { asm volatile("griddepcontrol.wait;" ::: "memory"); \
                            asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); } while (0)

__device__ __forceinline__ float nv_h2f(__half h) { return __half2float(h); }
__device__ __forceinline__ __half nv_f2h(float f) { return __float2half_rn(f); }

// ---------------------------------------------------------------------------
// Trellis decode (exl3_dq_from, verbatim): the 16-bit window (words i1m LOW, i0m HIGH, funnel
// shift sf) -> mul1 codebook value as fp16 bits, single-rounded fp16 FMA, no I2F.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint16_t nv_dq_from(const uint32_t* __restrict__ ring, int i0m, int i1m, int sf) {
    const unsigned lo = ring[i1m];
    const unsigned hi = ring[i0m];
    const unsigned idx = __funnelshift_r(lo, hi, (unsigned)sf) & 0xFFFFu;
    const unsigned sum = __dp4a(idx * NV_MUL1, 0x01010101u, 0x00006400u);
    const __half h = __ushort_as_half((unsigned short)sum);
    const __half w16 = __hfma(h, __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931));
    return (uint16_t)__half_as_ushort(w16);
}

// Window END (unwrapped bit index, >= 16) of temporal index t at BPB bits per 16 weights.
template <int BPB>
__device__ __forceinline__ int nv_win_end(int t) {
    if constexpr (BPB % 16 == 0) {
        return (t + 257) * (BPB / 16);                       // the integer decoder's (t+257)*bits
    } else {
        constexpr int KA = BPB / 16;                          // 3 for 3.5 bpw
        const int j = (t & 15) + 1;
        // mask 0xAAAA: odd ring positions carry the extra bit -> j>>1 of them among the first j
        return (t >> 4) * BPB + KA * j + (j >> 1) + 16 * BPB; // + one ring length keeps START >= 0
    }
}

// (i0m, i1m, sf) of temporal index t: the two ring words holding the window and the funnel shift.
template <int BPB>
__device__ __forceinline__ void nv_win(int t, int& i0m, int& i1m, int& sf) {
    constexpr int WW = BPB / 2;                               // u32 words per tile
    const int b1 = nv_win_end<BPB>(t);
    const int i0 = (b1 - 16) >> 5;
    const int i1 = (b1 - 1) >> 5;
    i0m = i0 % WW;
    i1m = i1 % WW;
    sf = ((i1 + 1) << 5) - b1;
}

// tmap(r, c) -> temporal index (src/exl3.rs tmap, the unit-tested bijection)
__device__ __forceinline__ int nv_tmap(int r, int c) {
    const int l = 8 * ((c >> 1) & 3) + ((r & 7) >> 1);
    return 8 * l + (r & 1) + 2 * ((r >> 3) & 1) + 4 * (c >> 3) + 32 * (c & 1);
}

// ---------------------------------------------------------------------------
// Kernel 1: decode gate. One warp per 16x16 block, lane decodes t in {lane + 32*step}; rc_of_t is
// the inverse tmap LUT. Output: W row-major [kb*16, nb*16] fp16 bits (= ext.reconstruct, rotated
// basis, before suh / svh). Gate: bitwise vs naive::frac::decode_trellis_host.
// ---------------------------------------------------------------------------
template <int BPB>
__device__ __forceinline__ void nv_decode_block(const uint16_t* __restrict__ trellis,
                                                const uint8_t* __restrict__ rc_of_t,
                                                uint16_t* __restrict__ wout, int nb) {
    const int b = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const uint32_t* __restrict__ ring = (const uint32_t*)(trellis + (size_t)b * BPB);
    const int kb = b / nb, nbk = b % nb;
    uint16_t* __restrict__ orow = wout + (size_t)kb * 16 * (nb * 16) + nbk * 16;
    #pragma unroll
    for (int step = 0; step < 8; ++step) {
        const int t = lane + 32 * step;
        int i0m, i1m, sf;
        nv_win<BPB>(t, i0m, i1m, sf);
        const uint16_t v = nv_dq_from(ring, i0m, i1m, sf);
        const int rc = rc_of_t[t];
        orow[(rc >> 4) * (nb * 16) + (rc & 15)] = v;
    }
}
extern "C" __global__ void nv_decode_dump(const uint16_t* __restrict__ trellis,
                                          const uint8_t* __restrict__ rc_of_t,
                                          uint16_t* __restrict__ wout, int nb, int bpb) {
    NV_PDL_ENTRY();
    switch (bpb) {
        case 48: nv_decode_block<48>(trellis, rc_of_t, wout, nb); break;
        case 56: nv_decode_block<56>(trellis, rc_of_t, wout, nb); break;
        case 64: nv_decode_block<64>(trellis, rc_of_t, wout, nb); break;
        case 80: nv_decode_block<80>(trellis, rc_of_t, wout, nb); break;
        case 96: nv_decode_block<96>(trellis, rc_of_t, wout, nb); break;
        default: break;
    }
}

// ---------------------------------------------------------------------------
// Kernels 2/3: the Hadamard passes (exl3_had128 / exl3_had_suh_dev / exl3_had_svh_dev verbatim).
//   xh = H128(x * suh)   per 128-block along K (suh BEFORE the butterfly: the quantizer's order)
//   y  = H128(y_raw) * svh per 128-block along N
// Orthonormal (1/sqrt(128)) — the convention the pack's suh/svh were calibrated under.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void nv_had128(float v[4], int lane) {
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        const int len = 1 << s;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            if ((r & len) == 0) {
                const float a = v[r], b = v[r + len];
                v[r] = a + b;
                v[r + len] = a - b;
            }
        }
    }
    #pragma unroll
    for (int s = 2; s < 7; ++s) {
        const unsigned mask = (unsigned)(1u << (s - 2));
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float t = __shfl_xor_sync(0xFFFFFFFFu, v[r], mask);
            v[r] = (lane & mask) ? (t - v[r]) : (v[r] + t);
        }
    }
    const float hnorm = 0.08838834764831845f;
    #pragma unroll
    for (int r = 0; r < 4; ++r) v[r] *= hnorm;
}

// grid m * (k/128), block 32
extern "C" __global__ void nv_had_suh(const __half* __restrict__ x, const __half* __restrict__ suh,
                                      __half* __restrict__ xh, int k) {
    NV_PDL_ENTRY();
    const int blk = blockIdx.x;
    const int kb = blk % (k >> 7);
    const int m = blk / (k >> 7);
    const int lane = threadIdx.x & 31;
    const size_t base = (size_t)m * k + (size_t)kb * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) v[r] = nv_h2f(x[base + r]) * nv_h2f(suh[kb * 128 + lane * 4 + r]);
    nv_had128(v, lane);
    #pragma unroll
    for (int r = 0; r < 4; ++r) xh[base + r] = nv_f2h(v[r]);
}

// grid m * (n/128), block 32
extern "C" __global__ void nv_had_svh(const __half* __restrict__ yraw, const __half* __restrict__ svh,
                                      __half* __restrict__ y, int n) {
    NV_PDL_ENTRY();
    const int blk = blockIdx.x;
    const int nb = blk % (n >> 7);
    const int m = blk / (n >> 7);
    const int lane = threadIdx.x & 31;
    const size_t base = (size_t)m * n + (size_t)nb * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) v[r] = nv_h2f(yraw[base + r]);
    nv_had128(v, lane);
    #pragma unroll
    for (int r = 0; r < 4; ++r) y[base + r] = nv_f2h(v[r] * nv_h2f(svh[nb * 128 + lane * 4 + r]));
}

// ---------------------------------------------------------------------------
// The HMMA GEMM: y_raw[m, n] = sum_k xh[m, k] * W[n, k], W decoded from the trellis per k16 step
// straight into the two m16n8k16 B fragments. CTA = 256 threads = 8 warps, tile m16 x N128; warp w
// owns columns [n0 + 16w, n0 + 16w + 16). fp32 accumulators, K ascending (fixed order).
// B-fragment slot (lane l, n-tile h, slot s): r = 2*(l%4) + (s&1) + 8*(s>>1), c = 8h + (l>>2),
// t = tmap(r, c) — the hoisted per-lane window constants below (exl3_gemm_body).
// ---------------------------------------------------------------------------
template <int BPB>
__device__ __forceinline__ void nv_slot_windows(int lane, int* s_i0m, int* s_i1m, int* s_sf) {
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            const int r = 2 * (lane & 3) + (s & 1) + 8 * (s >> 1);
            const int c = 8 * h + (lane >> 2);
            nv_win<BPB>(nv_tmap(r, c), s_i0m[h * 4 + s], s_i1m[h * 4 + s], s_sf[h * 4 + s]);
        }
    }
}

// (a) the per-step staged body (exl3_gemm_body / exl3_gemm_body_ks): A tile [16 x 16] re-staged
// every k16 step; works for every M <= 16 and every K. kb range [kb0, kb1); PARTIAL stores fp32
// partials to ws[ks_id][M][N], else fp16 RN to yraw.
template <int BPB, bool PARTIAL>
__device__ __forceinline__ void nv_gemm_body(const uint16_t* __restrict__ trellis,
                                             const __half* __restrict__ xh,
                                             __half* __restrict__ yraw, float* __restrict__ ws,
                                             int M, int K, int N, int cta_tile,
                                             int kb0, int kb1, int ks_id) {
    __shared__ __half sa[16][16];
    constexpr int WW = BPB / 2;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;
    const int nb = N >> 4;

    int s_i0m[8], s_i1m[8], s_sf[8];
    nv_slot_windows<BPB>(lane, s_i0m, s_i1m, s_sf);

    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;

    const uint32_t* __restrict__ ring = (const uint32_t*)(trellis + ((size_t)kb0 * nb + nb_col) * BPB);
    const size_t ring_kstride = (size_t)nb * WW;

    for (int kb = kb0; kb < kb1; ++kb) {
        {
            const int row = tid >> 4, col = tid & 15;
            const bool live = (row < M) && (kb * 16 + col < K);
            sa[row][col] = live ? xh[(size_t)row * K + kb * 16 + col] : __ushort_as_half(0);
        }
        __syncthreads();
        uint32_t bfrag[2][2];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint16_t r0 = nv_dq_from(ring, s_i0m[h*4+0], s_i1m[h*4+0], s_sf[h*4+0]);
            const uint16_t r1 = nv_dq_from(ring, s_i0m[h*4+1], s_i1m[h*4+1], s_sf[h*4+1]);
            const uint16_t r2 = nv_dq_from(ring, s_i0m[h*4+2], s_i1m[h*4+2], s_sf[h*4+2]);
            const uint16_t r3 = nv_dq_from(ring, s_i0m[h*4+3], s_i1m[h*4+3], s_sf[h*4+3]);
            bfrag[h][0] = (uint32_t)r0 | ((uint32_t)r1 << 16);
            bfrag[h][1] = (uint32_t)r2 | ((uint32_t)r3 << 16);
        }
        ring += ring_kstride;
        const int gid = lane >> 2, tig = lane & 3;
        const uint32_t a0 = *(const uint32_t*)&sa[gid][2 * tig];
        const uint32_t a1 = *(const uint32_t*)&sa[gid + 8][2 * tig];
        const uint32_t a2 = *(const uint32_t*)&sa[gid][2 * tig + 8];
        const uint32_t a3 = *(const uint32_t*)&sa[gid + 8][2 * tig + 8];
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bfrag[h][0]), "r"(bfrag[h][1]));
        }
        __syncthreads();
    }

    const int gid = lane >> 2, tig = lane & 3;
    const int row0 = gid, row1 = gid + 8;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int col = cta_n + 16 * warp + 8 * h + 2 * tig;
        if constexpr (PARTIAL) {
            float* base = ws + (size_t)ks_id * M * N;
            if (row0 < M) { base[(size_t)row0 * N + col] = acc[h][0]; base[(size_t)row0 * N + col + 1] = acc[h][1]; }
            if (row1 < M) { base[(size_t)row1 * N + col] = acc[h][2]; base[(size_t)row1 * N + col + 1] = acc[h][3]; }
        } else {
            if (row0 < M) *(__half2*)&yraw[(size_t)row0 * N + col] = __floats2half2_rn(acc[h][0], acc[h][1]);
            if (row1 < M) *(__half2*)&yraw[(size_t)row1 * N + col] = __floats2half2_rn(acc[h][2], acc[h][3]);
        }
    }
}

// ---- F13: the split-K combine folded into the split-K GEMM. `cnt[blockIdx.x]` counts the KS CTAs
// of one n-tile; the LAST one to arrive sums the partials (K ascending, the rounding points of
// nv_ks_combine_had: f16 before the butterfly, f16 after) and writes y / out32, then resets its
// counter (self-cleaning: the host zeroes the array once). Partials are read with __ldcg (L2) after a
// __threadfence on both sides -- the threadFenceReduction pattern. cnt == nullptr: no fusion.
struct NvFuse { const __half* svh; __half* y; float* out32; int* cnt; };
__device__ __forceinline__ void nv_combine_had_warp_cg(const float* __restrict__ ws, const __half* __restrict__ svh,
                                                       __half* __restrict__ y, int M, int N, int KS, long long rb, int lane,
                                                       float* __restrict__ out32) {
    const int nblk = N >> 7;
    const int m = (int)(rb / nblk), blk = (int)(rb - (long long)m * nblk);
    const size_t base = (size_t)m * N + (size_t)blk * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        float acc = 0.f;
        for (int s2 = 0; s2 < KS; ++s2) acc += __ldcg(ws + (size_t)s2 * M * N + base + q);
        v[q] = nv_h2f(nv_f2h(acc));
    }
    nv_had128(v, lane);
    const __half* sr = svh + (size_t)blk * 128 + lane * 4;
    if (out32 != nullptr) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) out32[base + q] = nv_h2f(nv_f2h(v[q] * nv_h2f(sr[q])));
    } else {
        #pragma unroll
        for (int q = 0; q < 4; ++q) y[base + q] = nv_f2h(v[q] * nv_h2f(sr[q]));
    }
}

// (b) the staged-A body (exl3_gemm_body_a1 + the F7 shuffle decode): the M live rows land in
// dynamic smem once (row stride K+8 halves), the K loop runs with no barriers; one coalesced ring
// load per lane per step, decode words by shuffle. Same values as (a) bit for bit. Dynamic smem =
// M*(K+8)*2 bytes (nv_a1_fits).
__host__ __device__ __forceinline__ bool nv_a1_fits(int M, int K) {
    return M >= 1 && M <= 16 && (K & 7) == 0 && (long long)M * (K + 8) * 2 <= 48 * 1024;
}
// Fused input transform (S6 F1): when `suh` is non-null the staged A is xh = H128(x * suh) computed
// in smem (x f16 [M][K]; or, with `x2` non-null, x = silu(x) * x2 first: the MoE down input), the
// SAME ops and rounding points as nv_had_suh / nv_silu_mul (f32 products, the butterfly, one f16
// rounding), so y is bitwise the unfused chain's. `suh == nullptr`: `xh` is already transformed.
template <int BPB, bool PARTIAL>
__device__ __forceinline__ void nv_gemm_body_a1(const uint16_t* __restrict__ trellis,
                                                const __half* __restrict__ xh,
                                                __half* __restrict__ yraw, float* __restrict__ ws,
                                                int M, int K, int N, int cta_tile, int kb0, int kb1, int ks_id,
                                                const __half* __restrict__ suh = nullptr,
                                                const __half* __restrict__ x2 = nullptr,
                                                NvFuse fz = NvFuse{nullptr, nullptr, nullptr, nullptr}) {
    extern __shared__ __half a1_sa[];
    constexpr int WW = BPB / 2;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;
    const int KB = K >> 4;
    const int nb = N >> 4;
    // T0-c: stage ONLY this CTA's k-window [kb0, kb1). A split-K CTA walks only that window, so
    // staging all of K read (and held) M*(K+8)*2 halves of A it never touches: ks times the bytes,
    // and the reason a staged split-K launch was gated to M*(K+8)*2 <= 48KB. Chunking it is
    // bitwise-identical (same words, same order, same rounding points) and ks times less smem AND
    // ks times less global traffic on the staging read. Host mirror: naive::a1_smem(m, k, ks);
    // for ks == 1 the window is [0, K) so SK is unchanged from the pre-T0-c K + 8.
    const int SK = ((kb1 - kb0) << 4) + 8;   // window row stride, halves (was K + 8)
    if (suh == nullptr) {
        const int v0 = kb0 << 1;             // window origin in 8-half (uint4) units (kb0 * 16 / 8)
        const int vn = (kb1 - kb0) << 1;     // window width in those units
        for (int i = tid; i < M * vn; i += blockDim.x) {
            const int r = i / vn, c = i - r * vn;
            *(uint4*)&a1_sa[(size_t)r * SK + c * 8] = *(const uint4*)&xh[(size_t)r * K + (v0 + c) * 8];
        }
    } else {
        // one warp per (row, 128-block): lane owns 4 consecutive elements (nv_had_suh's layout)
        // T0-c: the window is 128-aligned (the host gate requires (K/ks) % 128 == 0), so every
        // butterfly group this CTA needs is wholly inside its own window: restricting the loop to
        // [g0, g0+nblk) leaves each nv_had128 on the same 128 elements as before.
        const int nblk = (kb1 - kb0) >> 3;
        const int g0 = kb0 >> 3;
        for (int b = warp; b < M * nblk; b += blockDim.x >> 5) {
            const int r = b / nblk, blk = b - r * nblk;
            const size_t src = (size_t)r * K + (size_t)(blk + g0) * 128 + lane * 4;
            const __half* sr = suh + (size_t)(blk + g0) * 128 + lane * 4;
            float v[4];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                float xv = nv_h2f(xh[src + q]);
                if (x2 != nullptr) { xv = xv / (1.f + __expf(-xv)) * nv_h2f(x2[src + q]); xv = nv_h2f(nv_f2h(xv)); }
                v[q] = xv * nv_h2f(sr[q]);
            }
            nv_had128(v, lane);
            #pragma unroll
            for (int q = 0; q < 4; ++q) a1_sa[(size_t)r * SK + (size_t)blk * 128 + lane * 4 + q] = nv_f2h(v[q]);
        }
    }
    __syncthreads();

    int s_i0m[8], s_i1m[8], s_sf[8];
    nv_slot_windows<BPB>(lane, s_i0m, s_i1m, s_sf);
    float acc[2][4];
    #pragma unroll
    for (int h = 0; h < 2; ++h)
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[h][r] = 0.0f;

    const uint32_t* __restrict__ ring = (const uint32_t*)(trellis + ((size_t)kb0 * nb + nb_col) * BPB);
    const size_t ring_kstride = (size_t)nb * WW;
    const int gid = lane >> 2, tig = lane & 3;
    const bool hi_rows = M > 8;
    const bool l0 = gid < M, l1 = (gid + 8) < M;
    const __half* ar0 = a1_sa + (size_t)gid * SK + 2 * tig;
    const __half* ar1 = a1_sa + (size_t)(gid + 8) * SK + 2 * tig;
    (void)KB;

    constexpr int PF_D = 8;
    const bool pf_lane = lane * 32 < WW * 4;                 // 32-B chunks of the ring window
    const char* pf_ptr = (const char*)(ring + (size_t)PF_D * ring_kstride) + lane * 32;
    const size_t pf_step = ring_kstride * 4;
    if (pf_lane) {
        const char* p0 = (const char*)ring + lane * 32;
        #pragma unroll
        for (int d = 0; d < PF_D; ++d)
            if (kb0 + d < kb1) asm volatile("prefetch.global.L2 [%0];" :: "l"(p0 + (size_t)d * pf_step));
    }
    // S6 F7: software pipeline — the ring words of the NEXT group of NV_PIPE k-steps are loaded (one
    // coalesced u32 per lane per step) while the current group decodes and multiplies; the decode and
    // the mma sequence per step are unchanged (same words, same order => same bits).
    constexpr int NV_PIPE = 4;
    unsigned wc0[NV_PIPE], wc1[NV_PIPE], wn0[NV_PIPE], wn1[NV_PIPE];
    #pragma unroll
    for (int i = 0; i < NV_PIPE; ++i) {
        const bool live = kb0 + i < kb1;
        const uint32_t* rp = ring + (size_t)i * ring_kstride;
        wc0[i] = (live && lane < WW) ? __ldg(rp + lane) : 0u;
        wc1[i] = (live && WW > 32 && lane + 32 < WW) ? __ldg(rp + lane + 32) : 0u;
    }
    for (int kbg = kb0; kbg < kb1; kbg += NV_PIPE) {
        #pragma unroll
        for (int i = 0; i < NV_PIPE; ++i) {
            const int kbn = kbg + NV_PIPE + i;
            const bool live = kbn < kb1;
            const uint32_t* rp = ring + (size_t)(NV_PIPE + i) * ring_kstride;
            wn0[i] = (live && lane < WW) ? __ldg(rp + lane) : 0u;
            wn1[i] = (live && WW > 32 && lane + 32 < WW) ? __ldg(rp + lane + 32) : 0u;
        }
        if (pf_lane) {
            #pragma unroll
            for (int i = 0; i < NV_PIPE; ++i) {
                if (kbg + i + PF_D < kb1) asm volatile("prefetch.global.L2 [%0];" :: "l"(pf_ptr));
                pf_ptr += pf_step;
            }
        }
        #pragma unroll
        for (int i = 0; i < NV_PIPE; ++i) {
            const int kb = kbg + i;
            if (kb >= kb1) break;
            const unsigned w0 = wc0[i], w1 = wc1[i];
            uint32_t bfrag[2][2];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint16_t rv[4];
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int j = h * 4 + q;
                    unsigned lo = __shfl_sync(0xFFFFFFFFu, w0, s_i1m[j] & 31);
                    unsigned hi = __shfl_sync(0xFFFFFFFFu, w0, s_i0m[j] & 31);
                    if (WW > 32) {
                        const unsigned lo2 = __shfl_sync(0xFFFFFFFFu, w1, s_i1m[j] & 31);
                        const unsigned hi2 = __shfl_sync(0xFFFFFFFFu, w1, s_i0m[j] & 31);
                        lo = s_i1m[j] >= 32 ? lo2 : lo;
                        hi = s_i0m[j] >= 32 ? hi2 : hi;
                    }
                    const unsigned idx = __funnelshift_r(lo, hi, (unsigned)s_sf[j]) & 0xFFFFu;
                    const unsigned sum = __dp4a(idx * NV_MUL1, 0x01010101u, 0x00006400u);
                    rv[q] = (uint16_t)__half_as_ushort(__hfma(__ushort_as_half((unsigned short)sum),
                                __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931)));
                }
                bfrag[h][0] = (uint32_t)rv[0] | ((uint32_t)rv[1] << 16);
                bfrag[h][1] = (uint32_t)rv[2] | ((uint32_t)rv[3] << 16);
            }
            // T0-c: A is window-local now, so the tile's column is relative to the window origin.
            const int col = (kb - kb0) << 4;
            const uint32_t a0 = l0 ? *(const uint32_t*)(ar0 + col) : 0u;
            const uint32_t a2 = l0 ? *(const uint32_t*)(ar0 + col + 8) : 0u;
            uint32_t a1 = 0u, a3 = 0u;
            if (hi_rows) {
                a1 = l1 ? *(const uint32_t*)(ar1 + col) : 0u;
                a3 = l1 ? *(const uint32_t*)(ar1 + col + 8) : 0u;
            }
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[h][0]), "+f"(acc[h][1]), "+f"(acc[h][2]), "+f"(acc[h][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bfrag[h][0]), "r"(bfrag[h][1]));
            }
        }
        ring += (size_t)NV_PIPE * ring_kstride;
        #pragma unroll
        for (int i = 0; i < NV_PIPE; ++i) { wc0[i] = wn0[i]; wc1[i] = wn1[i]; }
    }
    const int row0 = gid, row1 = gid + 8;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int col = cta_n + 16 * warp + 8 * h + 2 * tig;
        if constexpr (PARTIAL) {
            float* base = ws + (size_t)ks_id * M * N;
            if (row0 < M) { base[(size_t)row0 * N + col] = acc[h][0]; base[(size_t)row0 * N + col + 1] = acc[h][1]; }
            if (row1 < M) { base[(size_t)row1 * N + col] = acc[h][2]; base[(size_t)row1 * N + col + 1] = acc[h][3]; }
        } else {
            if (row0 < M) *(__half2*)&yraw[(size_t)row0 * N + col] = __floats2half2_rn(acc[h][0], acc[h][1]);
            if (row1 < M) *(__half2*)&yraw[(size_t)row1 * N + col] = __floats2half2_rn(acc[h][2], acc[h][3]);
        }
    }
    if constexpr (PARTIAL) {
        if (fz.cnt != nullptr) {
            __shared__ int s_last;
            __threadfence();
            __syncthreads();
            if (tid == 0) s_last = (atomicAdd(fz.cnt + blockIdx.x, 1) == (int)gridDim.y - 1);
            __syncthreads();
            if (s_last) {
                __threadfence();
                const int nblk = N >> 7;
                for (int r = warp; r < M; r += 8)
                    nv_combine_had_warp_cg(ws, fz.svh, fz.y, M, N, (int)gridDim.y, (long long)r * nblk + cta_tile, lane, fz.out32);
                if (tid == 0) fz.cnt[blockIdx.x] = 0;
            }
        }
    }
}

#define NV_BPB_SWITCH(CALL)                 \
    switch (bpb) {                          \
        case 48: { constexpr int BPB = 48; CALL; break; } \
        case 56: { constexpr int BPB = 56; CALL; break; } \
        case 64: { constexpr int BPB = 64; CALL; break; } \
        case 80: { constexpr int BPB = 80; CALL; break; } \
        case 96: { constexpr int BPB = 96; CALL; break; } \
        default: break;                     \
    }

// Dense quad, M <= 16: grid (N/128), block 256, dyn smem = M*(K+8)*2 when nv_a1_fits else 0.
extern "C" __global__ void nv_gemm(const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
                                   __half* __restrict__ yraw, int M, int K, int N, int bpb) {
    NV_PDL_ENTRY();
    const int KB = K >> 4;
    if (nv_a1_fits(M, K)) {
        NV_BPB_SWITCH((nv_gemm_body_a1<BPB, false>(trellis, xh, yraw, nullptr, M, K, N, (int)blockIdx.x, 0, KB, 0)));
    } else {
        NV_BPB_SWITCH((nv_gemm_body<BPB, false>(trellis, xh, yraw, nullptr, M, K, N, (int)blockIdx.x, 0, KB, 0)));
    }
}

// Split-K: grid (N/128, ks), KB % ks == 0 (host asserts); fp32 partials ws[ks][M][N].
extern "C" __global__ void nv_gemm_ks(const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
                                      float* __restrict__ ws, int M, int K, int N, int bpb) {
    NV_PDL_ENTRY();
    const int ks = gridDim.y;
    const int KB = K >> 4;
    const int chunk = KB / ks;
    const int kb0 = blockIdx.y * chunk;
    const int kb1 = kb0 + chunk;
    NV_BPB_SWITCH((nv_gemm_body<BPB, true>(trellis, xh, nullptr, ws, M, K, N, (int)blockIdx.x, kb0, kb1, (int)blockIdx.y)));
}

// Fixed-order ascending combine of the split-K partials (exl3_ks_combine verbatim).
extern "C" __global__ void nv_ks_combine(const float* __restrict__ ws, __half* __restrict__ yraw, int M, int N, int KS) {
    NV_PDL_ENTRY();
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    const int m = idx / N;
    const int n = idx - (size_t)m * N;
    float acc = 0.0f;
    for (int s = 0; s < KS; ++s) acc += ws[((size_t)s * M + m) * N + n];
    yraw[(size_t)m * N + n] = __float2half_rn(acc);
}

// Grouped expert GEMM with per-expert xh ([E][M][K]) (xq_gemm_grouped_xh): grid padded to the
// top-k cap, blocks of absent experts exit on the device-routed count *esel. offs[e] = u16 word
// offset of expert e's trellis in `base`. Dyn smem = M*(K+8)*2 when nv_a1_fits else 0.
extern "C" __global__ void nv_gemm_grouped_xh(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                                              const __half* __restrict__ xh, __half* __restrict__ y,
                                              int M, int K, int N, int bpb, const int* __restrict__ esel) {
    NV_PDL_ENTRY();
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    const uint16_t* tr = base + (size_t)offs[expert];
    const __half* xe = xh + (long long)expert * M * K;
    __half* ye = y + (long long)expert * M * N;
    const int KB = K >> 4;
    if (nv_a1_fits(M, K)) {
        NV_BPB_SWITCH((nv_gemm_body_a1<BPB, false>(tr, xe, ye, nullptr, M, K, N, tile, 0, KB, 0)));
    } else {
        NV_BPB_SWITCH((nv_gemm_body<BPB, false>(tr, xe, ye, nullptr, M, K, N, tile, 0, KB, 0)));
    }
}

// S6: split-K with the STAGED body (A in dyn smem once, L2 prefetch): grid (N/128, KS) -> fp32
// partials ws[ks][M][N] (K-ascending chunks), then nv_ks_combine. Dyn smem = M*(K+8)*2 (nv_a1_fits).
// suh: the fused input transform (nullptr = xh already transformed); x2: non-null = x is silu(xh) * x2.
extern "C" __global__ void nv_gemm_a1ks(const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
                                        float* __restrict__ ws, int M, int K, int N, int bpb,
                                        const __half* __restrict__ suh, const __half* __restrict__ x2, NvFuse fz) {
    NV_PDL_ENTRY();
    const int ks = gridDim.y, ks_id = blockIdx.y;
    const int KB = K >> 4;
    const int chunk = KB / ks;
    const int kb0 = ks_id * chunk, kb1 = kb0 + chunk;
    NV_BPB_SWITCH((nv_gemm_body_a1<BPB, true>(trellis, xh, nullptr, ws, M, K, N, (int)blockIdx.x, kb0, kb1, ks_id, suh, x2, fz)));
}
// grouped experts, staged body, split-K: grid (E_cap * N/128, KS) -> ws[e][ks][M][N] -> nv_ks_combine_grouped.
// in_mode: 0 = xh is the SHARED [M][K] input (gate / up: x = hbuf), 1 = per-expert [E][M][K] input,
// 2 = per-expert PAIR (x = silu(xh[e]) * x2[e]: the down input from the gate / up outputs).
// suh_all [E_local][K] + idxmap (local expert index per slot) = the fused per-expert input transform.
// (12-argument launch limit: mb = M | bpb << 8, kn = K | N << 16)
extern "C" __global__ void nv_gemm_grouped_a1ks(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                                                const __half* __restrict__ xh, float* __restrict__ ws,
                                                int mb, int kn, const int* __restrict__ esel,
                                                const __half* __restrict__ suh_all, const int* __restrict__ idxmap,
                                                const __half* __restrict__ x2, int in_mode, NvFuse fz) {
    NV_PDL_ENTRY();
    const int M = mb & 0xFF, bpb = mb >> 8, K = kn & 0xFFFF, N = (int)((unsigned)kn >> 16);
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    const int ks = gridDim.y, ks_id = blockIdx.y;
    const int KB = K >> 4;
    const int chunk = KB / ks;
    const int kb0 = ks_id * chunk, kb1 = kb0 + chunk;
    const uint16_t* tr = base + (size_t)offs[expert];
    const __half* xe = (in_mode == 0) ? xh : xh + (long long)expert * M * K;
    const __half* x2e = (in_mode == 2) ? x2 + (long long)expert * M * K : nullptr;
    const __half* suh = suh_all ? suh_all + (long long)idxmap[expert] * K : nullptr;   // null = xh already transformed
    float* we = ws + (long long)expert * ks * M * N;
    if (fz.cnt != nullptr) {   // F13: this expert's svh row and y slab (the grouped combine's addressing)
        fz.svh = fz.svh + (long long)idxmap[expert] * N;
        fz.y = fz.y + (long long)expert * M * N;
    }
    NV_BPB_SWITCH((nv_gemm_body_a1<BPB, true>(tr, xe, nullptr, we, M, K, N, tile, kb0, kb1, ks_id, suh, x2e, fz)));
}

// ---- F2: split-K combine fused with the output transform: y[m][:] = H128(sum_ks ws) * svh, the sum
// in fixed K-ascending order, rounded to f16 BEFORE the butterfly (nv_ks_combine's rounding point),
// then one f16 rounding after (nv_had_svh's). One warp per (row, 128-block); block 256 = 8 warps.
__device__ __forceinline__ void nv_combine_had_warp(const float* __restrict__ ws, const __half* __restrict__ svh,
                                                    __half* __restrict__ y, int M, int N, int KS, long long rb, int lane,
                                                    float* __restrict__ out32 = nullptr) {
    const int nblk = N >> 7;
    const int m = (int)(rb / nblk), blk = (int)(rb - (long long)m * nblk);
    const size_t base = (size_t)m * N + (size_t)blk * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        float acc = 0.f;
        for (int s2 = 0; s2 < KS; ++s2) acc += ws[(size_t)s2 * M * N + base + q];
        v[q] = nv_h2f(nv_f2h(acc));
    }
    nv_had128(v, lane);
    const __half* sr = svh + (size_t)blk * 128 + lane * 4;
    if (out32 != nullptr) {
        // the TP partial: the f16-rounded value as fp32 (what nv_cvt_f16_f32 of y would give)
        #pragma unroll
        for (int q = 0; q < 4; ++q) out32[base + q] = nv_h2f(nv_f2h(v[q] * nv_h2f(sr[q])));
    } else {
        #pragma unroll
        for (int q = 0; q < 4; ++q) y[base + q] = nv_f2h(v[q] * nv_h2f(sr[q]));
    }
}
// out32 non-null (TP): the result lands as fp32 in out32 (the all-reduce partial) instead of f16 y.
extern "C" __global__ void nv_ks_combine_had(const float* __restrict__ ws, const __half* __restrict__ svh,
                                             __half* __restrict__ y, int M, int N, int KS, float* __restrict__ out32) {
    NV_PDL_ENTRY();
    const long long rb = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (rb >= (long long)M * (N >> 7)) return;
    nv_combine_had_warp(ws, svh, y, M, N, KS, rb, threadIdx.x & 31, out32);
}
// grouped: ws[e][ks][M][N], svh_all [E_local][N] via idxmap, y [E_cap][M][N]. grid over E_cap*M*N/128 warps.
extern "C" __global__ void nv_ks_combine_grouped_had(const float* __restrict__ ws, const __half* __restrict__ svh_all,
                                                     const int* __restrict__ idxmap, __half* __restrict__ y,
                                                     int M, int N, int KS, const int* __restrict__ esel) {
    NV_PDL_ENTRY();
    const long long rb = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const long long per = (long long)M * (N >> 7);
    const int e = (int)(rb / per);
    if (e >= *esel) return;
    nv_combine_had_warp(ws + (size_t)e * KS * M * N, svh_all + (size_t)idxmap[e] * N, y + (size_t)e * M * N, M, N, KS, rb - (long long)e * per, threadIdx.x & 31);
}

// =============================================================================
// S6b: the WIDE GEMM (prefill, M > 16) — exl3_gemm_body_wide re-parameterized for the half-K decoder:
// each warp decodes its 16x16 trellis tile ONCE per k-step (the a1 body's shuffle decode) and multiplies
// MT m-tiles of 16 rows from the staged A [MT*16][KS*16 + 8] (KS = 8 k-steps per stage). Rows >= M and
// columns >= K are zero. y_raw f16 [M][N], RN. grid (ceil(M / (MT*16)), N/128), block 256.
// K-ascending fp32 accumulation per output element: per-row results are bitwise independent of M and of
// the m-tile a row lands in (batch invariance). Same decode words / order as the small-M bodies.
// =============================================================================
#define NV_WIDE_KS 8
template <int BPB, int MT>
__device__ __forceinline__ void nv_gemm_body_wide(const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
                                                  __half* __restrict__ yraw, int M, int K, int N, int cta_tile, int m_sup,
                                                  __half* __restrict__ sa_raw) {
    typedef __half SaRow[NV_WIDE_KS * 16 + 8];
    SaRow* sa = (SaRow*)sa_raw;
    constexpr int WW = BPB / 2;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_n = cta_tile * 128;
    const int nb_col = cta_n / 16 + warp;
    const int KB = K >> 4;
    const int nb = N >> 4;
    const int row_base = m_sup * (MT * 16);
    int s_i0m[8], s_i1m[8], s_sf[8];
    nv_slot_windows<BPB>(lane, s_i0m, s_i1m, s_sf);
    float acc[MT][2][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[mt][h][r] = 0.0f;
    const uint32_t* __restrict__ ring = (const uint32_t*)(trellis + ((size_t)nb_col) * BPB);
    const size_t ring_kstride = (size_t)nb * WW;
    const int gid = lane >> 2, tig = lane & 3;
    for (int kb0 = 0; kb0 < KB; kb0 += NV_WIDE_KS) {
        {
            constexpr int C8 = NV_WIDE_KS * 2;
            for (int i = tid; i < MT * 16 * C8; i += blockDim.x) {
                const int row = i / C8, c8 = i - row * C8;
                const int gr = row_base + row;
                const int col = kb0 * 16 + c8 * 8;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                if (gr < M && col < K) v = *(const uint4*)&xh[(size_t)gr * K + col];
                *(uint4*)&sa[row][c8 * 8] = v;
            }
        }
        __syncthreads();
        for (int s = 0; s < NV_WIDE_KS; ++s) {
            const int kb = kb0 + s;
            if (kb >= KB) break;
            const uint32_t* rp = ring + (size_t)kb * ring_kstride;
            const unsigned w0 = (lane < WW) ? __ldg(rp + lane) : 0u;
            const unsigned w1 = (WW > 32 && lane + 32 < WW) ? __ldg(rp + lane + 32) : 0u;
            uint32_t bfrag[2][2];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint16_t rv[4];
                #pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int j = h * 4 + q;
                    unsigned lo = __shfl_sync(0xFFFFFFFFu, w0, s_i1m[j] & 31);
                    unsigned hi = __shfl_sync(0xFFFFFFFFu, w0, s_i0m[j] & 31);
                    if (WW > 32) {
                        const unsigned lo2 = __shfl_sync(0xFFFFFFFFu, w1, s_i1m[j] & 31);
                        const unsigned hi2 = __shfl_sync(0xFFFFFFFFu, w1, s_i0m[j] & 31);
                        lo = s_i1m[j] >= 32 ? lo2 : lo;
                        hi = s_i0m[j] >= 32 ? hi2 : hi;
                    }
                    const unsigned idx = __funnelshift_r(lo, hi, (unsigned)s_sf[j]) & 0xFFFFu;
                    const unsigned sum = __dp4a(idx * NV_MUL1, 0x01010101u, 0x00006400u);
                    rv[q] = (uint16_t)__half_as_ushort(__hfma(__ushort_as_half((unsigned short)sum),
                                __ushort_as_half((unsigned short)0x1EEE), __ushort_as_half((unsigned short)0xC931)));
                }
                bfrag[h][0] = (uint32_t)rv[0] | ((uint32_t)rv[1] << 16);
                bfrag[h][1] = (uint32_t)rv[2] | ((uint32_t)rv[3] << 16);
            }
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const SaRow* sm = sa + mt * 16;
                const int cs = s * 16;
                const uint32_t a0 = *(const uint32_t*)&sm[gid][cs + 2 * tig];
                const uint32_t a1 = *(const uint32_t*)&sm[gid + 8][cs + 2 * tig];
                const uint32_t a2 = *(const uint32_t*)&sm[gid][cs + 2 * tig + 8];
                const uint32_t a3 = *(const uint32_t*)&sm[gid + 8][cs + 2 * tig + 8];
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[mt][h][0]), "+f"(acc[mt][h][1]), "+f"(acc[mt][h][2]), "+f"(acc[mt][h][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(bfrag[h][0]), "r"(bfrag[h][1]));
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int row0 = row_base + mt * 16 + gid;
        const int row1 = row0 + 8;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = cta_n + 16 * warp + 8 * h + 2 * tig;
            if (row0 < M) *(__half2*)&yraw[(size_t)row0 * N + col] = __floats2half2_rn(acc[mt][h][0], acc[mt][h][1]);
            if (row1 < M) *(__half2*)&yraw[(size_t)row1 * N + col] = __floats2half2_rn(acc[mt][h][2], acc[mt][h][3]);
        }
    }
}
#define NV_WIDE_SMEM(MT) ((MT) * 16 * (NV_WIDE_KS * 16 + 8) * 2)
// mt in {1, 2, 4, 8}; dyn smem = NV_WIDE_SMEM(mt). grid (ceil(M / (mt*16)), N/128).
extern "C" __global__ void nv_gemm_wide(const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
                                        __half* __restrict__ yraw, int M, int K, int N, int bpb, int mt) {
    NV_PDL_ENTRY();
    extern __shared__ __half wide_sa[];
    const int cta_tile = (int)blockIdx.y, m_sup = (int)blockIdx.x;
#define NV_WIDE_CASE(B) case B: switch (mt) { \
        case 1: nv_gemm_body_wide<B, 1>(trellis, xh, yraw, M, K, N, cta_tile, m_sup, wide_sa); break; \
        case 2: nv_gemm_body_wide<B, 2>(trellis, xh, yraw, M, K, N, cta_tile, m_sup, wide_sa); break; \
        case 4: nv_gemm_body_wide<B, 4>(trellis, xh, yraw, M, K, N, cta_tile, m_sup, wide_sa); break; \
        case 8: nv_gemm_body_wide<B, 8>(trellis, xh, yraw, M, K, N, cta_tile, m_sup, wide_sa); break; \
        default: break; } break;
    switch (bpb) { NV_WIDE_CASE(48) NV_WIDE_CASE(56) NV_WIDE_CASE(64) NV_WIDE_CASE(80) NV_WIDE_CASE(96) default: break; }
#undef NV_WIDE_CASE
}
// grouped (prefill MoE): experts' rows are gathered contiguously: rows [offs_row[z], offs_row[z] + cnt[z]) of xh / y
// belong to compact expert slot z (trellis at base + offs[z]). grid (msup_max, N/128, esel_cap): a block exits
// when its slot is empty or its m-tile is past the slot's rows.
extern "C" __global__ void nv_gemm_wide_grouped(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                                                const __half* __restrict__ xh, __half* __restrict__ y,
                                                const int* __restrict__ offs_row, const int* __restrict__ cnt,
                                                const int* __restrict__ esel, int K, int N, int bpb, int mt) {
    NV_PDL_ENTRY();
    extern __shared__ __half wide_sa[];
    const int z = blockIdx.z;
    if (z >= *esel) return;
    const int ce = cnt[z];
    if (ce == 0) return;
    const int m_sup = (int)blockIdx.x;
    if (m_sup * mt * 16 >= ce) return;
    const int row0 = offs_row[z];
    const uint16_t* tr = base + (size_t)offs[z];
    const __half* xe = xh + (size_t)row0 * K;
    __half* ye = y + (size_t)row0 * N;
    const int cta_tile = (int)blockIdx.y;
#define NV_WG_CASE(B) case B: switch (mt) { \
        case 1: nv_gemm_body_wide<B, 1>(tr, xe, ye, ce, K, N, cta_tile, m_sup, wide_sa); break; \
        case 2: nv_gemm_body_wide<B, 2>(tr, xe, ye, ce, K, N, cta_tile, m_sup, wide_sa); break; \
        case 4: nv_gemm_body_wide<B, 4>(tr, xe, ye, ce, K, N, cta_tile, m_sup, wide_sa); break; \
        case 8: nv_gemm_body_wide<B, 8>(tr, xe, ye, ce, K, N, cta_tile, m_sup, wide_sa); break; \
        default: break; } break;
    switch (bpb) { NV_WG_CASE(48) NV_WG_CASE(56) NV_WG_CASE(64) NV_WG_CASE(80) NV_WG_CASE(96) default: break; }
#undef NV_WG_CASE
}

// =============================================================================
// S6b: prefill MoE on gathered rows. R = M * k (routed (token, slot) pairs); rows of one LOCAL expert are
// contiguous: [offs_row[le], offs_row[le] + cnt[le]). rowmap[m][j] = the gathered row of pair (m, j), or -1
// when the expert belongs to another rank. The combine sums the k slots in j order per token: deterministic.
// =============================================================================
// cnt[le] (zeroed by the host memset... no: zeroed here by block 0 would race; the caller passes a zeroed buffer
// from the workspace via nv_zero_i32) += 1 per routed pair of an owned expert. grid ceil(M*k/256).
extern "C" __global__ void nv_zero_i32(int* __restrict__ p, int n) {
    NV_PDL_ENTRY();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0;
}
extern "C" __global__ void nv_moe_hist(const int* __restrict__ ids, const int* __restrict__ local_of, int* __restrict__ cnt, int MK) {
    NV_PDL_ENTRY();
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= MK) return;
    const int le = local_of[ids[i]];
    if (le >= 0) atomicAdd(&cnt[le], 1);
}
// exclusive prefix over the local experts -> offs_row; cursor[le] = offs_row[le] (the gather's running index);
// esel[0] = E_local. One block.
extern "C" __global__ void __launch_bounds__(1024)
nv_moe_offsets(const int* __restrict__ cnt, int* __restrict__ offs_row, int* __restrict__ cursor, int* __restrict__ esel, int EL) {
    NV_PDL_ENTRY();
    __shared__ int sc[1024];
    const int tid = threadIdx.x;
    int v = (tid < EL) ? cnt[tid] : 0;
    sc[tid] = v; __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) { const int x = (tid >= o) ? sc[tid - o] : 0; __syncthreads(); sc[tid] += x; __syncthreads(); }
    if (tid < EL) { const int off = sc[tid] - v; offs_row[tid] = off; cursor[tid] = off; }
    if (tid == 0) esel[0] = EL;
}
// gather: pair (m, j) of an owned expert takes row r = cursor[le]++ (atomic order: every row's results are
// batch-invariant and the combine reads rows through rowmap in j order, so the order is immaterial);
// xh_g[r] = H128(x[m] * suh_g[le]) and xh_u[r] = H128(x[m] * suh_u[le]) (gate and up share the row);
// rowmap[m][j] = r; row_e[r] = le. One block (128 threads) per pair; nv_had_suh's lane layout per 128-block.
extern "C" __global__ void __launch_bounds__(128)
nv_moe_gather_had(const __half* __restrict__ x, const __half* __restrict__ suh_g, const __half* __restrict__ suh_u,
                  const int* __restrict__ ids, const int* __restrict__ local_of, int* __restrict__ cursor,
                  __half* __restrict__ xh_g, __half* __restrict__ xh_u, int* __restrict__ rowmap, int* __restrict__ row_e,
                  int k, int K) {
    NV_PDL_ENTRY();
    __shared__ int s_r;
    const int pair = blockIdx.x;
    const int m = pair / k;
    const int le = local_of[ids[pair]];
    if (threadIdx.x == 0) s_r = (le >= 0) ? atomicAdd(&cursor[le], 1) : -1;
    __syncthreads();
    const int r = s_r;
    if (threadIdx.x == 0) { rowmap[pair] = r; if (r >= 0) row_e[r] = le; }
    if (r < 0) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nblk = K >> 7;
    const __half* xr = x + (size_t)m * K;
    const __half* sg = suh_g + (size_t)le * K;
    const __half* su = suh_u + (size_t)le * K;
    __half* og = xh_g + (size_t)r * K;
    __half* ou = xh_u + (size_t)r * K;
    for (int blk = warp; blk < nblk; blk += 4) {
        float xv[4], v[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) xv[q] = nv_h2f(xr[blk * 128 + lane * 4 + q]);
        #pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = xv[q] * nv_h2f(sg[blk * 128 + lane * 4 + q]);
        nv_had128(v, lane);
        #pragma unroll
        for (int q = 0; q < 4; ++q) og[blk * 128 + lane * 4 + q] = nv_f2h(v[q]);
        #pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = xv[q] * nv_h2f(su[blk * 128 + lane * 4 + q]);
        nv_had128(v, lane);
        #pragma unroll
        for (int q = 0; q < 4; ++q) ou[blk * 128 + lane * 4 + q] = nv_f2h(v[q]);
    }
}
// per-row output transform with the row's expert: y[r] = H128(yraw[r]) * svh[row_e[r]]. grid R * (N/128) warps (block 256).
extern "C" __global__ void nv_had_svh_rows(const __half* __restrict__ yraw, const __half* __restrict__ svh_all,
                                           const int* __restrict__ row_e, __half* __restrict__ y, int R, int N) {
    NV_PDL_ENTRY();
    const long long wid = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int nblk = N >> 7;
    if (wid >= (long long)R * nblk) return;
    const int r = (int)(wid / nblk), blk = (int)(wid - (long long)r * nblk);
    const int lane = threadIdx.x & 31;
    const size_t base = (size_t)r * N + (size_t)blk * 128 + lane * 4;
    const __half* sr = svh_all + (size_t)row_e[r] * N + (size_t)blk * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) v[q] = nv_h2f(yraw[base + q]);
    nv_had128(v, lane);
    #pragma unroll
    for (int q = 0; q < 4; ++q) y[base + q] = nv_f2h(v[q] * nv_h2f(sr[q]));
}
// per-row input transform with the row's expert (the down input): xh[r] = H128(silu(g[r]) * u[r] * suh[row_e[r]]),
// the silu product rounded to f16 first (nv_silu_mul's rounding point). grid R * (K/128) warps.
extern "C" __global__ void nv_had_suh_rows_silu(const __half* __restrict__ g, const __half* __restrict__ u,
                                                const __half* __restrict__ suh_all, const int* __restrict__ row_e,
                                                __half* __restrict__ xh, int R, int K) {
    NV_PDL_ENTRY();
    const long long wid = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int nblk = K >> 7;
    if (wid >= (long long)R * nblk) return;
    const int r = (int)(wid / nblk), blk = (int)(wid - (long long)r * nblk);
    const int lane = threadIdx.x & 31;
    const size_t base = (size_t)r * K + (size_t)blk * 128 + lane * 4;
    const __half* sr = suh_all + (size_t)row_e[r] * K + (size_t)blk * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float gv = nv_h2f(g[base + q]);
        const float d = nv_h2f(nv_f2h(gv / (1.f + __expf(-gv)) * nv_h2f(u[base + q])));
        v[q] = d * nv_h2f(sr[q]);
    }
    nv_had128(v, lane);
    #pragma unroll
    for (int q = 0; q < 4; ++q) xh[base + q] = nv_f2h(v[q]);
}
// combine in j order per token: x[m][:] (+)= sum_j wts[m][j] * y_d[rowmap[m][j]] (skipping -1). grid M, block 1024.
extern "C" __global__ void nv_moe_gather_combine(float* __restrict__ x, const __half* __restrict__ y_d, const int* __restrict__ rowmap,
                                                 const float* __restrict__ wts, int k, int H, int accumulate) {
    NV_PDL_ENTRY();
    const int m = blockIdx.x;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < k; j++) {
            const int r = rowmap[m * k + j];
            if (r >= 0) acc += wts[m * k + j] * nv_h2f(y_d[(size_t)r * H + i]);
        }
        if (accumulate) x[(size_t)m * H + i] += acc; else x[(size_t)m * H + i] = acc;
    }
}

// S6 bench control: the per-step body regardless of nv_a1_fits (same values, no A staging / prefetch).
extern "C" __global__ void nv_gemm_nostage(const uint16_t* __restrict__ trellis, const __half* __restrict__ xh,
                                           __half* __restrict__ yraw, int M, int K, int N, int bpb) {
    NV_PDL_ENTRY();
    const int KB = K >> 4;
    NV_BPB_SWITCH((nv_gemm_body<BPB, false>(trellis, xh, yraw, nullptr, M, K, N, (int)blockIdx.x, 0, KB, 0)));
}

// S6: grouped expert GEMM with split-K: grid (E_cap * N/128, KS); fp32 partials ws[e][ks][M][N]
// (the K-ascending fixed-order chunks of nv_gemm_body PARTIAL), then nv_ks_combine_grouped.
extern "C" __global__ void nv_gemm_grouped_ks(const uint16_t* __restrict__ base, const uint64_t* __restrict__ offs,
                                              const __half* __restrict__ xh, float* __restrict__ ws,
                                              int M, int K, int N, int bpb, const int* __restrict__ esel) {
    NV_PDL_ENTRY();
    const int tiles = N / 128;
    const int expert = blockIdx.x / tiles;
    const int tile = blockIdx.x - expert * tiles;
    if (expert >= *esel) return;
    const int ks = gridDim.y, ks_id = blockIdx.y;
    const int KB = K >> 4;
    const int chunk = KB / ks;
    const int kb0 = ks_id * chunk, kb1 = kb0 + chunk;
    const uint16_t* tr = base + (size_t)offs[expert];
    const __half* xe = xh + (long long)expert * M * K;
    float* we = ws + (long long)expert * ks * M * N;
    NV_BPB_SWITCH((nv_gemm_body<BPB, true>(tr, xe, nullptr, we, M, K, N, tile, kb0, kb1, ks_id)));
}
// y[e][m][n] = sum_ks ws[e][ks][m][n] (fixed order), f16 RN. grid ceil(E_cap*M*N/256).
extern "C" __global__ void nv_ks_combine_grouped(const float* __restrict__ ws, __half* __restrict__ y, int M, int N, int KS,
                                                 const int* __restrict__ esel) {
    NV_PDL_ENTRY();
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long per = (long long)M * N;
    const int e = (int)(idx / per);
    if (e >= *esel) return;
    const long long mn = idx - (long long)e * per;
    const float* w = ws + (long long)e * KS * per + mn;
    float acc = 0.f;
    for (int s2 = 0; s2 < KS; s2++) acc += w[(long long)s2 * per];
    y[idx] = __float2half_rn(acc);
}

// =============================================================================
// S3b: the layer glue — norms, embedding, rope, KV writes, windowed attention with a sink, the
// DSA indexer (fp8-rounded q/k, scores, exact top-k), gathered sparse attention, router + MoE
// glue. Semantics: the HF reference modeling file, as encoded by src/naive/cpu.rs (the oracle).
// =============================================================================

// ---- embedding gather: out f32 [M][H] = table bf16 [V][H] rows ids[M]. grid (M), block 256.
extern "C" __global__ void nv_embed(float* __restrict__ out, const unsigned short* __restrict__ table,
                                    const int* __restrict__ ids, int H) {
    NV_PDL_ENTRY();
    const int m = blockIdx.x;
    const long long row = (long long)ids[m] * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const unsigned u = (unsigned)table[row + i] << 16;        // bf16 -> f32 is a shift
        out[(long long)m * H + i] = __uint_as_float(u);
    }
}

// ---- LlamaRMSNorm: h f16 [M][H] = x * rsqrt(mean(x^2) + eps) * w  (w f16 [H], the pack's bf16
// norms are converted to f16 at load). grid (M), block 1024, H % 1024 == 0 not required.
extern "C" __global__ void nv_rmsnorm(__half* __restrict__ h, const float* __restrict__ x,
                                      const __half* __restrict__ w, int H, float eps) {
    NV_PDL_ENTRY();
    __shared__ float red[1024];
    const int m = blockIdx.x, tid = threadIdx.x;
    const float* xr = x + (long long)m * H;
    float ss = 0.f;
    for (int i = tid; i < H; i += blockDim.x) { const float v = xr[i]; ss += v * v; }
    red[tid] = ss; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    const float inv = rsqrtf(red[0] / (float)H + eps);
    for (int i = tid; i < H; i += blockDim.x) h[(long long)m * H + i] = nv_f2h(xr[i] * inv * nv_h2f(w[i]));
}

// ---- residual add: x f32 [n] += y f16 [n]
extern "C" __global__ void nv_add_f16(float* __restrict__ x, const __half* __restrict__ y, long long n) {
    NV_PDL_ENTRY();
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += nv_h2f(y[i]);
}

// ---- GPT-NeoX partial rotary on the first `rd` dims of each head (rotate_half), in place on f16
// rows. One warp per (row, head); lane owns dims lane and lane+rd/2 (rd = 64 -> lanes 0..31).
// cos/sin from double-precision sincos of pos * theta^(-2i/rd) (the f32 reference formula loses
// precision past ~10^5 positions; both sides of every gate here use this f64 form).
__device__ __forceinline__ void nv_rope_head(__half* v, int pos, float theta, int rd, int lane) {
    const int half = rd >> 1;
    if (lane >= half) return;
    const double inv = pow((double)theta, -(2.0 * lane) / (double)rd);
    double sn, cs;
    sincos((double)pos * inv, &sn, &cs);
    const float c = (float)cs, s = (float)sn;
    const float x1 = nv_h2f(v[lane]), x2 = nv_h2f(v[lane + half]);
    v[lane] = nv_f2h(x1 * c - x2 * s);
    v[lane + half] = nv_f2h(x2 * c + x1 * s);
}
// grid (M * heads), block 32; rows of `heads` heads of `hd` dims at pitch heads*hd.
extern "C" __global__ void nv_rope_rows(__half* __restrict__ x, const int* __restrict__ pos, int heads, int hd,
                                        int rd, float theta) {
    NV_PDL_ENTRY();
    const int m = blockIdx.x / heads, h = blockIdx.x % heads;
    nv_rope_head(x + ((long long)m * heads + h) * hd, pos[m], theta, rd, threadIdx.x & 31);
}

// ---- KV cache write. k [M][kv*192] f16 (roped), v [M][kv*192] f16 (per-head padded: the first vd of
// each 192 slice is live). Cache layout: kc [kv][S][hd], vc [kv][S][vd], slot = ring ? pos & (S-1) : pos.
// grid (M * kv), block 192.
extern "C" __global__ void nv_kv_write(__half* __restrict__ kc, __half* __restrict__ vc,
                                       const __half* __restrict__ k, const __half* __restrict__ v,
                                       const int* __restrict__ pos, const int* __restrict__ slot_ids,
                                       int kv, int hd, int vd, int S, int ring) {
    NV_PDL_ENTRY();
    const int m = blockIdx.x / kv, g = blockIdx.x % kv, d = threadIdx.x;
    const int p = pos[m];
    const int slot = ring ? (p & (S - 1)) : p;
    // slot_ids[m] = the request row m writes its cache in: kc [n_slots][kv][S][hd] (vc [n_slots][kv][S][vd])
    const long long skc = (long long)slot_ids[m] * ((long long)kv * S * hd);
    const long long svc = (long long)slot_ids[m] * ((long long)kv * S * vd);
    kc[skc + ((long long)g * S + slot) * hd + d] = k[((long long)m * kv + g) * hd + d];
    if (d < vd) vc[svc + ((long long)g * S + slot) * vd + d] = v[((long long)m * kv + g) * hd + d];
}

// ---- S6 F4: rope(q) + rope(k) + K/V cache write in one launch. grid (M * (nh + kv)), block 32: warps
// < nh rope a q head; the rest rope a k head and copy its k (roped) and v rows into the cache slot
// (the SAME per-element ops as nv_rope_rows + nv_kv_write). nhkv = nh | kv << 16, hdvd = hd | vd << 16.
extern "C" __global__ void nv_rope_kv(__half* __restrict__ q, __half* __restrict__ k, const __half* __restrict__ v,
                                      __half* __restrict__ kc, __half* __restrict__ vc, const int* __restrict__ pos,
                                      const int* __restrict__ slot_ids,
                                      int nhkv, int hdvd, int rd, float theta, int S) {
    NV_PDL_ENTRY();
    const int nh = nhkv & 0xFFFF, kv = nhkv >> 16, hd = hdvd & 0xFFFF, vd = (hdvd >> 16) & 0x3FFF;
    const int ring = (int)((unsigned)hdvd >> 30);   // packed by the host (cudarc's 12-arg launch cap)
    const int per = nh + kv;
    const int m = blockIdx.x / per, h = blockIdx.x % per;
    const int lane = threadIdx.x & 31;
    const int p = pos[m];
    if (h < nh) { nv_rope_head(q + ((long long)m * nh + h) * hd, p, theta, rd, lane); return; }
    const int g = h - nh;
    __half* kr = k + ((long long)m * kv + g) * hd;
    nv_rope_head(kr, p, theta, rd, lane);
    __syncwarp();
    const int slot = ring ? (p & (S - 1)) : p;
    // slot_ids[m] = the request row m writes its cache in: kc [n_slots][kv][S][hd] (vc [n_slots][kv][S][vd])
    __half* kd = kc + (long long)slot_ids[m] * ((long long)kv * S * hd) + ((long long)g * S + slot) * hd;
    __half* vdst = vc + (long long)slot_ids[m] * ((long long)kv * S * vd) + ((long long)g * S + slot) * vd;
    const __half* vr = v + ((long long)m * kv + g) * hd;
    for (int d = lane; d < hd; d += 32) kd[d] = kr[d];
    for (int d = lane; d < vd; d += 32) vdst[d] = vr[d];
}

// ---- sliding-window attention with a learned sink, decode / verify rows (M <= 16).
// q f16 [M][nh*hd] (roped), caches as nv_kv_write (ring of S slots), sink f16 [nh] (has_sink = 0: unused).
// Keys t in [max(0, p-W+1), p]; logits = q.k * hd^-0.5 ; softmax over keys U {sink}; out f16
// [M][nh*vd] = sum p_t V_t (the reference scales V by attention_value_scale BEFORE attention; the
// pack folds that into o_proj, so nothing here). grid (nh, M), block 256 (W <= 256).
extern "C" __global__ void nv_attn_swa(__half* __restrict__ out, const __half* __restrict__ q,
                                       const __half* __restrict__ kc, const __half* __restrict__ vc,
                                       const __half* __restrict__ sink, const int* __restrict__ pos,
                                       const int* __restrict__ slot_ids,
                                       int nhkv, int hdvd, int S, int w_sink) {
    NV_PDL_ENTRY();
    const int nh = nhkv & 0xFFFF, kv = nhkv >> 16, hd = hdvd & 0xFFFF, vd = hdvd >> 16;
    const int W = w_sink & 0xFFFF, has_sink = w_sink >> 16;
    __shared__ float qs[256];        // hd <= 256
    __shared__ float lg[256];        // one logit per key slot in the window
    __shared__ float red[256];
    const int h = blockIdx.x, m = blockIdx.y, tid = threadIdx.x;
    const int g = h / (nh / kv);
    // slot_ids[m] = the request whose cache row m attends over: kc [n_slots][kv][S][hd] (vc [n_slots][kv][S][vd])
    const long long skc = (long long)slot_ids[m] * ((long long)kv * S * hd);
    const long long svc = (long long)slot_ids[m] * ((long long)kv * S * vd);
    const int p = pos[m];
    const int t0 = p - W + 1 > 0 ? p - W + 1 : 0;
    const int nk = p - t0 + 1;                                   // <= W <= 256
    for (int d = tid; d < hd; d += blockDim.x) qs[d] = nv_h2f(q[((long long)m * nh + h) * hd + d]);
    __syncthreads();
    const float scale = rsqrtf((float)hd);
    float lv = -INFINITY;
    if (tid < nk) {
        const int t = t0 + tid;
        const __half* kr = kc + skc + ((long long)g * S + (t & (S - 1))) * hd;
        float dot = 0.f;
        // 8 halves per load (hd % 8 == 0; rows are 16-B aligned), the SAME sequential fp32 accumulation
        // over d as the scalar loop it replaces: bitwise identical, 8x fewer load instructions.
        for (int d0 = 0; d0 < hd; d0 += 8) {
            const uint4 kv4 = *(const uint4*)(kr + d0);
            const __half* kh = (const __half*)&kv4;
            #pragma unroll
            for (int j = 0; j < 8; j++) dot += qs[d0 + j] * nv_h2f(kh[j]);
        }
        lv = dot * scale;
    }
    lg[tid] = lv;
    // max over keys and the sink
    float mx = lv;
    if (has_sink && tid == 0) mx = fmaxf(mx, nv_h2f(sink[h]));
    red[tid] = mx; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
    mx = red[0];
    __syncthreads();
    float e = (tid < nk) ? __expf(lv - mx) : 0.f;
    lg[tid] = e;
    red[tid] = e; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    float den = red[0];
    if (has_sink) den += __expf(nv_h2f(sink[h]) - mx);
    __syncthreads();
    // weighted V: thread d < vd sums over keys ascending
    if (tid < vd) {
        float acc = 0.f;
        for (int i = 0; i < nk; i++) {
            const int t = t0 + i;
            acc += lg[i] * nv_h2f(vc[svc + ((long long)g * S + (t & (S - 1))) * vd + tid]);
        }
        out[((long long)m * nh + h) * vd + tid] = nv_f2h(acc / den);
    }
}

// ---- PREFILL variant of nv_attn_swa (m > 16): ONE BLOCK PER (kv group, row); the gqa (<= 8) query heads of the
// group share the K / V rows (each key's K row is read once for all heads; the V column once per key). Per head
// the arithmetic, summation order (sequential dot over d, the 256-wide tree sums, ascending-key weighted V),
// sink and window semantics are exactly nv_attn_swa's: the output is bitwise identical. grid (kv, M), block 256.
extern "C" __global__ void nv_attn_swa_grp(__half* __restrict__ out, const __half* __restrict__ q,
                                           const __half* __restrict__ kc, const __half* __restrict__ vc,
                                           const __half* __restrict__ sink, const int* __restrict__ pos,
                                           const int* __restrict__ slot_ids,
                                           int nhkv, int hdvd, int S, int w_sink) {
    NV_PDL_ENTRY();
    const int nh = nhkv & 0xFFFF, kv = nhkv >> 16, hd = hdvd & 0xFFFF, vd = hdvd >> 16;
    const int W = w_sink & 0xFFFF, has_sink = w_sink >> 16;
    __shared__ float qs[8][256];
    __shared__ float lg[8][256];
    __shared__ float red[8][256];
    const int g = blockIdx.x, m = blockIdx.y, tid = threadIdx.x;
    const int gqa = nh / kv;                                      // <= 8 (host guarantees)
    // slot_ids[m] = the request whose cache row m attends over: kc [n_slots][kv][S][hd] (vc [n_slots][kv][S][vd])
    const long long skc = (long long)slot_ids[m] * ((long long)kv * S * hd);
    const long long svc = (long long)slot_ids[m] * ((long long)kv * S * vd);
    const int p = pos[m];
    const int t0 = p - W + 1 > 0 ? p - W + 1 : 0;
    const int nk = p - t0 + 1;
    for (int hh = 0; hh < gqa; hh++)
        for (int d = tid; d < hd; d += blockDim.x) qs[hh][d] = nv_h2f(q[((long long)m * nh + g * gqa + hh) * hd + d]);
    __syncthreads();
    const float scale = rsqrtf((float)hd);
    float lv[8];
    #pragma unroll
    for (int hh = 0; hh < 8; hh++) lv[hh] = -INFINITY;
    if (tid < nk) {
        const int t = t0 + tid;
        const __half* kr = kc + skc + ((long long)g * S + (t & (S - 1))) * hd;
        float dot[8];
        #pragma unroll
        for (int hh = 0; hh < 8; hh++) dot[hh] = 0.f;
        for (int d = 0; d < hd; d++) {
            const float kd = nv_h2f(kr[d]);
            #pragma unroll
            for (int hh = 0; hh < 8; hh++) if (hh < gqa) dot[hh] += qs[hh][d] * kd;
        }
        #pragma unroll
        for (int hh = 0; hh < 8; hh++) lv[hh] = dot[hh] * scale;
    }
    #pragma unroll
    for (int hh = 0; hh < 8; hh++) if (hh < gqa) {
        lg[hh][tid] = lv[hh];
        float mx = lv[hh];
        if (has_sink && tid == 0) mx = fmaxf(mx, nv_h2f(sink[g * gqa + hh]));
        red[hh][tid] = mx;
    }
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int hh = 0; hh < 8; hh++) if (hh < gqa) red[hh][tid] = fmaxf(red[hh][tid], red[hh][tid + s]);
        }
        __syncthreads();
    }
    float mxh[8];
    #pragma unroll
    for (int hh = 0; hh < 8; hh++) mxh[hh] = (hh < gqa) ? red[hh][0] : 0.f;
    __syncthreads();
    #pragma unroll
    for (int hh = 0; hh < 8; hh++) if (hh < gqa) {
        const float e = (tid < nk) ? __expf(lv[hh] - mxh[hh]) : 0.f;
        lg[hh][tid] = e;
        red[hh][tid] = e;
    }
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int hh = 0; hh < 8; hh++) if (hh < gqa) red[hh][tid] += red[hh][tid + s];
        }
        __syncthreads();
    }
    float den[8];
    #pragma unroll
    for (int hh = 0; hh < 8; hh++) {
        den[hh] = (hh < gqa) ? red[hh][0] : 1.f;
        if (hh < gqa && has_sink) den[hh] += __expf(nv_h2f(sink[g * gqa + hh]) - mxh[hh]);
    }
    __syncthreads();
    if (tid < vd) {
        float acc[8];
        #pragma unroll
        for (int hh = 0; hh < 8; hh++) acc[hh] = 0.f;
        for (int i = 0; i < nk; i++) {
            const int t = t0 + i;
            const float v = nv_h2f(vc[svc + ((long long)g * S + (t & (S - 1))) * vd + tid]);
            #pragma unroll
            for (int hh = 0; hh < 8; hh++) if (hh < gqa) acc[hh] += lg[hh][i] * v;
        }
        #pragma unroll
        for (int hh = 0; hh < 8; hh++) if (hh < gqa) out[((long long)m * nh + g * gqa + hh) * vd + tid] = nv_f2h(acc[hh] / den[hh]);
    }
}

// ---- DSA indexer. The reference: q = rope(wq(x)) [16][128], k = rope(LayerNorm(wk(x))) [128], both
// rounded to fp8 e4m3 per row (scale = amax.clamp_min(1e-4)/448, RNE, saturate); scores[t] =
// sum_h relu(q_h . k_t) * (weights_proj(x)[h] * 16^-0.5); causal; top-k by stable descending sort.
// Keys are cached as e4m3 bytes + one f32 scale per position: kq [S][ihd] u8, ks [S] f32.

__device__ __forceinline__ float nv_e4m3_round(float v, float scale) {
    // (v / scale).clamp(-448, 448) -> e4m3 (RN, saturate) -> f32 * scale
    const float x = fminf(fmaxf(v / scale, -448.f), 448.f);
    const __nv_fp8_storage_t b = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
    return __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)) * scale;
}
__device__ __forceinline__ float nv_e4m3_val(unsigned char b) {
    return __half2float(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3));
}

// q side: iq f32 [M][ih][ihd] from the wq chain output f16 [M][ih*ihd]: rope (first rd dims), then
// the per-(row, head) fp8 round-trip. grid (M * ih), block = ihd (128).
extern "C" __global__ void nv_idx_q(float* __restrict__ iq, const __half* __restrict__ raw, const int* __restrict__ pos,
                                    int ih, int ihd, int rd, float theta, int fp8) {
    NV_PDL_ENTRY();
    __shared__ float v[128];
    __shared__ float red[128];
    const int m = blockIdx.x / ih, h = blockIdx.x % ih, d = threadIdx.x;
    const long long base = ((long long)m * ih + h) * ihd;
    float x = nv_h2f(raw[base + d]);
    // rope in f32 on the first rd dims (rotate_half)
    const int half = rd >> 1;
    v[d] = x; __syncthreads();
    if (d < rd) {
        const int i = d < half ? d : d - half;
        const double inv = pow((double)theta, -(2.0 * i) / (double)rd);
        double sn, cs; sincos((double)pos[m] * inv, &sn, &cs);
        const float c = (float)cs, s = (float)sn;
        x = d < half ? (v[d] * c - v[d + half] * s) : (v[d] * c + v[d - half] * s);
    }
    if (fp8) {
        red[d] = fabsf(x); __syncthreads();
        for (int s = ihd >> 1; s > 0; s >>= 1) { if (d < s) red[d] = fmaxf(red[d], red[d + s]); __syncthreads(); }
        const float scale = fmaxf(red[0], 1e-4f) / 448.f;
        x = nv_e4m3_round(x, scale);
    }
    iq[base + d] = x;
}

// k side: raw f16 [M][ihd] from the wk chain; LayerNorm (biased var, eps 1e-5, weight + bias f16),
// rope, fp8 round -> kq[pos][ihd] bytes + ks[pos]. grid (M), block = ihd (128).
extern "C" __global__ void nv_idx_k(unsigned char* __restrict__ kq, float* __restrict__ ks,
                                    const __half* __restrict__ raw, const __half* __restrict__ nw,
                                    const __half* __restrict__ nb, const int* __restrict__ pos,
                                    const int* __restrict__ slot_ids,
                                    int ihd, int rd, float theta, float eps, int max_pos) {
    NV_PDL_ENTRY();
    __shared__ float v[128];
    __shared__ float red[128];
    const int m = blockIdx.x, d = threadIdx.x;
    float x = nv_h2f(raw[(long long)m * ihd + d]);
    red[d] = x; __syncthreads();
    for (int s = ihd >> 1; s > 0; s >>= 1) { if (d < s) red[d] += red[d + s]; __syncthreads(); }
    const float mean = red[0] / (float)ihd;
    __syncthreads();
    red[d] = (x - mean) * (x - mean); __syncthreads();
    for (int s = ihd >> 1; s > 0; s >>= 1) { if (d < s) red[d] += red[d + s]; __syncthreads(); }
    const float inv = rsqrtf(red[0] / (float)ihd + eps);
    x = (x - mean) * inv * nv_h2f(nw[d]) + nv_h2f(nb[d]);
    __syncthreads();
    v[d] = x; __syncthreads();
    const int half = rd >> 1;
    if (d < rd) {
        const int i = d < half ? d : d - half;
        const double invf = pow((double)theta, -(2.0 * i) / (double)rd);
        double sn, cs; sincos((double)pos[m] * invf, &sn, &cs);
        const float c = (float)cs, s = (float)sn;
        x = d < half ? (v[d] * c - v[d + half] * s) : (v[d] * c + v[d - half] * s);
    }
    __syncthreads();
    red[d] = fabsf(x); __syncthreads();
    for (int s = ihd >> 1; s > 0; s >>= 1) { if (d < s) red[d] = fmaxf(red[d], red[d + s]); __syncthreads(); }
    const float scale = fmaxf(red[0], 1e-4f) / 448.f;
    const float xq = fminf(fmaxf(x / scale, -448.f), 448.f);
    // slot_ids[m] = the request whose index cache row m writes: kq [n_slots][max_pos][ihd], ks [n_slots][max_pos]
    kq[(long long)slot_ids[m] * max_pos * ihd + (long long)pos[m] * ihd + d] = (unsigned char)__nv_cvt_float_to_fp8(xq, __NV_SATFINITE, __NV_E4M3);
    if (d == 0) ks[(long long)slot_ids[m] * max_pos + pos[m]] = scale;
}

// scores[m][t] for t <= pos[m]: sum_h relu(iq[m][h] . k_t) * wgt[m][h]; wgt = weights_proj row (f16,
// padded to 128; the first ih live) * ih^-0.5. One warp per (row, key): lane owns 4 dims (ihd 128).
// grid (ceil((maxpos+1)/8), M), block 256 (8 warps = 8 keys). Entries past pos[m] are not written.
extern "C" __global__ void nv_idx_score(float* __restrict__ scores, const float* __restrict__ iq,
                                        const __half* __restrict__ wgt, const unsigned char* __restrict__ kq,
                                        const float* __restrict__ ks, const int* __restrict__ pos,
                                        const int* __restrict__ slot_ids,
                                        int ih, int ihd, int wpitch, long long stride, int max_pos) {
    NV_PDL_ENTRY();
    // S6: a FIXED grid (graph-capturable) that strides over the keys t <= pos[m]
    const int m = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int dpl = ihd >> 5;                                    // 4
    const float wscale = rsqrtf((float)ih);
    const int n = pos[m] + 1;
    // slot_ids[m] = the request whose index cache row m reads: kq [n_slots][max_pos][ihd], ks [n_slots][max_pos]
    const long long skq = (long long)slot_ids[m] * max_pos * ihd;
    const long long sks = (long long)slot_ids[m] * max_pos;
    for (int t = blockIdx.x * 8 + warp; t < n; t += gridDim.x * 8) {
        float kvv[4];
        const unsigned char* kr = kq + skq + (long long)t * ihd + lane * dpl;
        const float sc = ks[sks + t];
        #pragma unroll
        for (int i = 0; i < 4; i++) kvv[i] = (i < dpl) ? nv_e4m3_val(kr[i]) * sc : 0.f;
        float s = 0.f;
        for (int h = 0; h < ih; h++) {
            const float* qr = iq + ((long long)m * ih + h) * ihd + lane * dpl;
            float dot = 0.f;
            #pragma unroll
            for (int i = 0; i < 4; i++) if (i < dpl) dot += qr[i] * kvv[i];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
            s += fmaxf(dot, 0.f) * (nv_h2f(wgt[(long long)m * wpitch + h]) * wscale);
        }
        if (lane == 0) scores[(long long)m * stride + t] = s;
    }
}

// PREFILL variant of nv_idx_score (M > 16 rows, ih == 16, ihd == 128): tensor cores. The 16 index heads of one
// row are exactly one m16 tile: C[h][key] = sum_d iq[m][h][d] * K[key][d] (mma m16n8k16 f16, fp32 accumulate).
// K is staged as raw e4m3 values (exact in f16) per tile of NVW_KT keys; the per-key scale is applied to the
// fp32 accumulator (ks > 0, so relu commutes). Then relu * (w[m][h] * ih^-0.5), summed over h in fp32.
// grid (ceil(nkeys / NVW_KT), ceil(M / NVW_RB)), block 256 (8 warps, each walks rows r = warp, warp+8, ...).
// Entries past pos[m] are not written.
#define NVW_KT 128
#define NVW_RB 32
#define NVW_LD 136
extern "C" __global__ void nv_idx_score_wide(float* __restrict__ scores, const float* __restrict__ iq,
                                             const __half* __restrict__ wgt, const unsigned char* __restrict__ kq,
                                             const float* __restrict__ ks, const int* __restrict__ pos,
                                             const int* __restrict__ slot_ids,
                                             int M, int nkeys, int wpitch, long long stride) {
    // 2026-10-20: the ONE naive entry kernel without the PDL entry. Under --naive-pdl=1 (default since the same
    // night) it could start before nv_idx_q / nv_idx_k finished writing iq / kq and score stale keys: the 7,475-token
    // prompt's sampled stream diverged at ~token 15 while every short-prompt stream stayed identical.
    NV_PDL_ENTRY();
    __shared__ __align__(16) __half Ks[NVW_KT * NVW_LD];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const int t0 = blockIdx.x * NVW_KT;
    const int rbase = blockIdx.y * NVW_RB;
    int maxp = -1;
    for (int r = 0; r < NVW_RB && rbase + r < M; r++) maxp = max(maxp, pos[rbase + r]);
    if (t0 > maxp) return;
    const float wscale = 0.25f;                              // 16^-0.5
    // The key tile is staged by ALL threads of the block, so it comes from ONE request's plane: the block's rows are
    // one prefill chunk's (this scorer runs for M > 16 only, i.e. a prefill chunk, i.e. one request). 2026-10-21: each
    // thread used its OWN warp's row slot here, and 0 for a row with no key in this tile (a row before the tile, or past
    // the chunk), so a request in a KV slot other than 0 (--naive-lanes > 1) scored tiles mixed with slot 0's keys;
    // past 2,048 keys (the top-k) its DSA selection, and its text, drifted from the same request in slot 0.
    const long long skq_t = (long long)slot_ids[rbase] * nkeys * 128;
    for (int r = warp; r < NVW_RB; r += 8) {
        const int m = rbase + r;
        const bool live = (m < M) && (t0 <= pos[m]);
        // slot_ids[m] = the request whose index cache row m reads: kq [n_slots][nkeys][128], ks [n_slots][nkeys]
        const long long sks = live ? (long long)slot_ids[m] * nkeys : 0;
        __syncthreads();
        for (int i = tid; i < NVW_KT * 8; i += 256) {      // 8 chunks of 16 bytes per key
            const int key = i >> 3, c = i & 7, t = t0 + key;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (t < nkeys) v = *(const uint4*)(kq + skq_t + (long long)t * 128 + c * 16);
            const unsigned char* b = (const unsigned char*)&v;
            __half2 o[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) o[j] = __floats2half2_rn(nv_e4m3_val(b[2 * j]), nv_e4m3_val(b[2 * j + 1]));
            *(uint4*)&Ks[key * NVW_LD + c * 16] = *(uint4*)&o[0];
            *(uint4*)&Ks[key * NVW_LD + c * 16 + 8] = *(uint4*)&o[4];
        }
        __syncthreads();
        if (!live) continue;
        const int pm = pos[m];
        uint32_t a[8][4];
        const float* q0 = iq + ((long long)m * 16 + gid) * 128;
        const float* q1 = q0 + 8 * 128;
        #pragma unroll
        for (int kc = 0; kc < 8; kc++) {
            const int d = kc * 16 + 2 * tig;
            const float2 x0 = *(const float2*)(q0 + d), x1 = *(const float2*)(q1 + d);
            const float2 x2 = *(const float2*)(q0 + d + 8), x3 = *(const float2*)(q1 + d + 8);
            __half2 h0 = __floats2half2_rn(x0.x, x0.y), h1 = __floats2half2_rn(x1.x, x1.y);
            __half2 h2 = __floats2half2_rn(x2.x, x2.y), h3 = __floats2half2_rn(x3.x, x3.y);
            a[kc][0] = *(uint32_t*)&h0; a[kc][1] = *(uint32_t*)&h1; a[kc][2] = *(uint32_t*)&h2; a[kc][3] = *(uint32_t*)&h3;
        }
        const float w0 = nv_h2f(wgt[(long long)m * wpitch + gid]) * wscale;
        const float w1 = nv_h2f(wgt[(long long)m * wpitch + gid + 8]) * wscale;
        for (int nb = 0; nb < NVW_KT / 8; nb++) {
            const int tb = t0 + nb * 8;
            if (tb > pm) break;
            float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
            const __half* kr = &Ks[(nb * 8 + gid) * NVW_LD + 2 * tig];
            #pragma unroll
            for (int kc = 0; kc < 8; kc++) {
                const uint32_t b0 = *(const uint32_t*)(kr + kc * 16), b1 = *(const uint32_t*)(kr + kc * 16 + 8);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                             : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                             : "r"(a[kc][0]), "r"(a[kc][1]), "r"(a[kc][2]), "r"(a[kc][3]), "r"(b0), "r"(b1));
            }
            const int ta = tb + 2 * tig;
            const float s0 = (ta < nkeys) ? ks[sks + ta] : 0.f, s1 = (ta + 1 < nkeys) ? ks[sks + ta + 1] : 0.f;
            float v0 = fmaxf(c0 * s0, 0.f) * w0 + fmaxf(c2 * s0, 0.f) * w1;
            float v1 = fmaxf(c1 * s1, 0.f) * w0 + fmaxf(c3 * s1, 0.f) * w1;
            #pragma unroll
            for (int o = 4; o < 32; o <<= 1) { v0 += __shfl_xor_sync(0xffffffffu, v0, o); v1 += __shfl_xor_sync(0xffffffffu, v1, o); }
            if (gid == 0) {
                if (ta <= pm) scores[(long long)m * stride + ta] = v0;
                if (ta + 1 <= pm) scores[(long long)m * stride + ta + 1] = v1;
            }
        }
    }
}

// monotonic unsigned key of a float (negative scores are legal here: weights_proj can be negative)
__device__ __forceinline__ unsigned nv_fkey(float f) {
    const unsigned b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// block-wide exclusive scan of two flags (blockDim 1024 = 32 warps); returns prefixes and totals
__device__ __forceinline__ void nv_exscan2(int a, int c, int& pa, int& pc, int& ta, int& tc, int* wa, int* wc) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int ia = a, ic = c;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int xa = __shfl_up_sync(0xffffffffu, ia, off), xc = __shfl_up_sync(0xffffffffu, ic, off);
        if (lane >= off) { ia += xa; ic += xc; }
    }
    if (lane == 31) { wa[warp] = ia; wc[warp] = ic; }
    __syncthreads();
    int ba = 0, bc = 0, sa = 0, sc = 0;
    const int nw = blockDim.x >> 5;
    for (int w = 0; w < nw; w++) { if (w < warp) { ba += wa[w]; bc += wc[w]; } sa += wa[w]; sc += wc[w]; }
    pa = ba + ia - a; pc = bc + ic - c; ta = sa; tc = sc;
    __syncthreads();
}

// top-K over scores[m][0..pos[m]] (n = pos+1 keys): the K largest under (value desc, index asc) —
// exactly torch's stable descending argsort. Radix select (4 x 8-bit passes, MSB first) over the
// monotonic keys finds the K-th key T and how many == T to keep (kp); the compaction keeps every
// key > T and the first kp keys == T in index order, emitted ASCENDING into sel[m][0..nsel), nsel[m]
// = min(K, n). One 1024-thread block per row (the engine's qsa_topk_b, ratio 1, signed keys).
extern "C" __global__ void __launch_bounds__(1024)
nv_topk(int* __restrict__ sel, int* __restrict__ nsel, const float* __restrict__ scores,
        const int* __restrict__ pos, int K, long long stride, const int* __restrict__ acc,
        const int* __restrict__ force) {
    NV_PDL_ENTRY();
    // sampler fallback gate (the indexer passes nullptr): skipped once a Gumbel try was accepted,
    // UNLESS `force[0]` (the exact speculative path of work item A builds p from the gathered
    // candidate list on EVERY row, on EVERY rank: a skipped gather leaves sel/nsel stale or never
    // written, and the accept kernel then reads an empty list -- the 2026-10-19 `anchor 4286578688`
    // served fatal: keep == 0 -> pick == -1 -> rtok = sp_i[-1] = the -INFINITY pad of sp_v)
    if (acc != nullptr && acc[0] && !(force != nullptr && force[0])) return;
    __shared__ int hist[256];
    __shared__ int s_bin, s_kp;
    __shared__ int wa[32], wc[32];
    const int m = blockIdx.x, tid = threadIdx.x;
    const int n = pos[m] + 1;
    const float* sc = scores + (long long)m * stride;
    int* srow = sel + (long long)m * K;
    unsigned T = 0u; int kp = 0x7fffffff;
    if (n > K) {
        unsigned prefix = 0u, mask = 0u;
        kp = K;
        for (int pass = 3; pass >= 0; pass--) {
            const int shift = pass * 8;
            for (int i = tid; i < 256; i += blockDim.x) hist[i] = 0;
            __syncthreads();
            for (int j = tid; j < n; j += blockDim.x) {
                const unsigned key = nv_fkey(sc[j]);
                if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1);
            }
            __syncthreads();
            if (tid == 0) {
                int cum = 0, bin = 0, rem = kp;
                for (int i = 255; i >= 0; i--) {
                    const int c = hist[i];
                    if (cum + c >= kp) { bin = i; rem = kp - cum; break; }
                    cum += c;
                }
                s_bin = bin; s_kp = rem;
            }
            __syncthreads();
            prefix |= ((unsigned)s_bin) << shift;
            mask |= 0xFFu << shift;
            kp = s_kp;
            __syncthreads();
        }
        T = prefix;
    }
    int run_sel = 0, run_eq = 0;
    for (int base = 0; base < n; base += blockDim.x) {
        const int j = base + tid;
        const bool valid = j < n;
        const unsigned key = valid ? nv_fkey(sc[j]) : 0u;
        const bool gt = valid && (n <= K || key > T);
        const bool eq = valid && n > K && key == T;
        int p_eq, p_d, t_eq, t_d;
        nv_exscan2(eq ? 1 : 0, 0, p_eq, p_d, t_eq, t_d, wa, wc);
        const bool selected = gt || (eq && (run_eq + p_eq) < kp);
        int p_s, t_s;
        nv_exscan2(selected ? 1 : 0, 0, p_s, p_d, t_s, t_d, wa, wc);
        if (selected) srow[run_sel + p_s] = j;
        run_sel += t_s; run_eq += t_eq;
    }
    if (tid == 0) nsel[m] = run_sel;
}

// ---- gathered sparse attention (DSA layers): q f16 [M][nh*hd] roped, full caches kc [kv][S][hd],
// vc [kv][S][vd], key list sel[m][0..nsel[m]) (ascending). Split-K online softmax, no sink: grid
// (M * kv * nsplits), block 256 = 8 warps; warp w owns heads kvh*gqa + w + 8j (j < gqa/8); lane
// owns hd/32 = 6 K dims and vd/32 = 4 V dims. Partials pm/pl [M*nh*nsplits], pacc [..][vd]; empty
// split: pm = -inf, pl = 0. Then nv_attn_combine merges the splits (ascending) -> out f16 [M][nh*vd].
extern "C" __global__ void nv_attn_dsa_split(float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
                                             const __half* __restrict__ q, const __half* __restrict__ kc,
                                             const __half* __restrict__ vc, const int* __restrict__ sel,
                                             const int* __restrict__ nsel, const int* __restrict__ slot_ids,
                                             int nhkv, int hdvd, long long geom) {
    NV_PDL_ENTRY();
    const int nh = nhkv & 0xFFFF, kv = nhkv >> 16, hd = hdvd & 0xFFFF, vd = hdvd >> 16;
    // geom layout, packed host-side at the nv_attn_dsa_split launch in model.rs:
    //     K [0:16) | nsplits [16:24) | split_len [24:40) | S [40:64)
    // S gets 24 bits ON PURPOSE: it is the layer's KV row count (= max_pos for a DSA layer) and is
    // the per-head stride g*S+t, and 262144 does not fit in 16. It used to be packed at bit 48 and
    // masked with 0xFFFF, which truncated it to 0 at the serving context and aliased every KV head
    // onto head 0's keys -- silently, and only once max_pos > 65535, so every slice probe stayed green.
    const int K = (int)(geom & 0xFFFF), nsplits = (int)((geom >> 16) & 0xFF), split_len = (int)((geom >> 24) & 0xFFFF), S = (int)((geom >> 40) & 0xFFFFFF);
    const int sp = blockIdx.x % nsplits;
    const int g = (blockIdx.x / nsplits) % kv;
    const int m = blockIdx.x / (nsplits * kv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gqa = nh / kv;
    const int dq = hd >> 5, dv = vd >> 5;                       // 6, 4
    // slot_ids[m] = the request whose cache row m gathers from: kc [n_slots][kv][S][hd] (vc [n_slots][kv][S][vd])
    const long long skc = (long long)slot_ids[m] * ((long long)kv * S * hd);
    const long long svc = (long long)slot_ids[m] * ((long long)kv * S * vd);
    const int ns = nsel[m];
    const int begin = sp * split_len;
    const int end = min((sp + 1) * split_len, ns);
    const float scale = rsqrtf((float)hd);
    const int* srow = sel + (long long)m * K;
    for (int w = warp; w < gqa; w += 8) {
        const int h = g * gqa + w;
        const __half* qr = q + ((long long)m * nh + h) * hd + lane * dq;
        float qv[6];
        #pragma unroll
        for (int i = 0; i < 6; i++) qv[i] = (i < dq) ? nv_h2f(qr[i]) : 0.f;
        float mx = -INFINITY, l = 0.f, acc[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) acc[i] = 0.f;
        // Software pipeline (bitwise): key r+1's K (dq halves) and V (dv halves) rows are loaded while
        // key r is reduced. The loads are packed u32 pairs of halves at the lane's 4-B-aligned offset
        // (dq = 6, dv = 4 halves per lane); the values and the per-key arithmetic order are those of
        // the scalar loop this replaces. Without this each key cost a full dependent load latency
        // (~1 us/key, 181 us per launch at the owner's contexts).
        uint32_t kw[3] = {0u, 0u, 0u}, vw[2] = {0u, 0u};
        if (begin < end) {
            const int t0 = srow[begin];
            const uint32_t* kp = (const uint32_t*)(kc + skc + ((long long)g * S + t0) * hd + lane * dq);
            const uint32_t* vp = (const uint32_t*)(vc + svc + ((long long)g * S + t0) * vd + lane * dv);
            #pragma unroll
            for (int i = 0; i < 3; i++) if (2 * i < dq) kw[i] = kp[i];
            #pragma unroll
            for (int i = 0; i < 2; i++) if (2 * i < dv) vw[i] = vp[i];
        }
        for (int r = begin; r < end; r++) {
            uint32_t kn[3] = {0u, 0u, 0u}, vn[2] = {0u, 0u};
            if (r + 1 < end) {
                const int tn = srow[r + 1];
                const uint32_t* kp = (const uint32_t*)(kc + skc + ((long long)g * S + tn) * hd + lane * dq);
                const uint32_t* vp = (const uint32_t*)(vc + svc + ((long long)g * S + tn) * vd + lane * dv);
                #pragma unroll
                for (int i = 0; i < 3; i++) if (2 * i < dq) kn[i] = kp[i];
                #pragma unroll
                for (int i = 0; i < 2; i++) if (2 * i < dv) vn[i] = vp[i];
            }
            float dot = 0.f;
            #pragma unroll
            for (int i = 0; i < 6; i++) if (i < dq) dot += qv[i] * nv_h2f(__ushort_as_half((unsigned short)((i & 1) ? (kw[i >> 1] >> 16) : (kw[i >> 1] & 0xFFFFu))));
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
            const float s = dot * scale;
            const float mn = fmaxf(mx, s);
            const float f = __expf(mx - mn), wt = __expf(s - mn);
            l = l * f + wt;
            #pragma unroll
            for (int i = 0; i < 4; i++) if (i < dv) acc[i] = acc[i] * f + wt * nv_h2f(__ushort_as_half((unsigned short)((i & 1) ? (vw[i >> 1] >> 16) : (vw[i >> 1] & 0xFFFFu))));
            mx = mn;
            #pragma unroll
            for (int i = 0; i < 3; i++) kw[i] = kn[i];
            #pragma unroll
            for (int i = 0; i < 2; i++) vw[i] = vn[i];
        }
        const long long hg = (long long)m * nh + h;
        if (lane == 0) { pm[hg * nsplits + sp] = mx; pl[hg * nsplits + sp] = l; }
        #pragma unroll
        for (int i = 0; i < 4; i++) if (i < dv) pacc[(hg * nsplits + sp) * vd + lane * dv + i] = acc[i];
    }
}

// PREFILL variant of nv_attn_dsa_split (m > 16): same grid / partial layout, but each warp walks the key list
// ONCE for its (up to 2) heads w and w + 8, sharing the K / V loads. Per head the arithmetic and order are
// nv_attn_dsa_split's: bitwise identical partials. Requires gqa % 8 == 0 and gqa <= 16.
extern "C" __global__ void nv_attn_dsa_split_grp(float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc,
                                                 const __half* __restrict__ q, const __half* __restrict__ kc,
                                                 const __half* __restrict__ vc, const int* __restrict__ sel,
                                                 const int* __restrict__ nsel, const int* __restrict__ slot_ids,
                                                 int nhkv, int hdvd, long long geom) {
    NV_PDL_ENTRY();
    const int nh = nhkv & 0xFFFF, kv = nhkv >> 16, hd = hdvd & 0xFFFF, vd = hdvd >> 16;
    // geom layout, packed host-side at the nv_attn_dsa_split launch in model.rs:
    //     K [0:16) | nsplits [16:24) | split_len [24:40) | S [40:64)
    // S gets 24 bits ON PURPOSE: it is the layer's KV row count (= max_pos for a DSA layer) and is
    // the per-head stride g*S+t, and 262144 does not fit in 16. It used to be packed at bit 48 and
    // masked with 0xFFFF, which truncated it to 0 at the serving context and aliased every KV head
    // onto head 0's keys -- silently, and only once max_pos > 65535, so every slice probe stayed green.
    const int K = (int)(geom & 0xFFFF), nsplits = (int)((geom >> 16) & 0xFF), split_len = (int)((geom >> 24) & 0xFFFF), S = (int)((geom >> 40) & 0xFFFFFF);
    const int sp = blockIdx.x % nsplits;
    const int g = (blockIdx.x / nsplits) % kv;
    const int m = blockIdx.x / (nsplits * kv);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gqa = nh / kv;
    const int nj = gqa >> 3;                                    // 1 or 2 heads per warp
    const int dq = hd >> 5, dv = vd >> 5;
    // slot_ids[m] = the request whose cache row m gathers from: kc [n_slots][kv][S][hd] (vc [n_slots][kv][S][vd])
    const long long skc = (long long)slot_ids[m] * ((long long)kv * S * hd);
    const long long svc = (long long)slot_ids[m] * ((long long)kv * S * vd);
    const int ns = nsel[m];
    const int begin = sp * split_len;
    const int end = min((sp + 1) * split_len, ns);
    const float scale = rsqrtf((float)hd);
    const int* srow = sel + (long long)m * K;
    float qv[2][6], mx[2], l[2], acc[2][4];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        const __half* qr = q + ((long long)m * nh + g * gqa + warp + 8 * j) * hd + lane * dq;
        #pragma unroll
        for (int i = 0; i < 6; i++) qv[j][i] = (j < nj && i < dq) ? nv_h2f(qr[i]) : 0.f;
        mx[j] = -INFINITY; l[j] = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; i++) acc[j][i] = 0.f;
    }
    for (int r = begin; r < end; r++) {
        const int t = srow[r];
        const __half* kr = kc + skc + ((long long)g * S + t) * hd + lane * dq;
        const __half* vr = vc + svc + ((long long)g * S + t) * vd + lane * dv;
        float kk[6], vv[4];
        #pragma unroll
        for (int i = 0; i < 6; i++) kk[i] = (i < dq) ? nv_h2f(kr[i]) : 0.f;
        #pragma unroll
        for (int i = 0; i < 4; i++) vv[i] = (i < dv) ? nv_h2f(vr[i]) : 0.f;
        #pragma unroll
        for (int j = 0; j < 2; j++) if (j < nj) {
            float dot = 0.f;
            #pragma unroll
            for (int i = 0; i < 6; i++) if (i < dq) dot += qv[j][i] * kk[i];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
            const float s = dot * scale;
            const float mn = fmaxf(mx[j], s);
            const float f = __expf(mx[j] - mn), wt = __expf(s - mn);
            l[j] = l[j] * f + wt;
            #pragma unroll
            for (int i = 0; i < 4; i++) if (i < dv) acc[j][i] = acc[j][i] * f + wt * vv[i];
            mx[j] = mn;
        }
    }
    #pragma unroll
    for (int j = 0; j < 2; j++) if (j < nj) {
        const long long hg = (long long)m * nh + g * gqa + warp + 8 * j;
        if (lane == 0) { pm[hg * nsplits + sp] = mx[j]; pl[hg * nsplits + sp] = l[j]; }
        #pragma unroll
        for (int i = 0; i < 4; i++) if (i < dv) pacc[(hg * nsplits + sp) * vd + lane * dv + i] = acc[j][i];
    }
}

// grid (M * nh), block vd
extern "C" __global__ void nv_attn_combine(__half* __restrict__ out, const float* __restrict__ pm,
                                           const float* __restrict__ pl, const float* __restrict__ pacc,
                                           int nsplits, int vd) {
    NV_PDL_ENTRY();
    const long long hg = blockIdx.x;
    const int d = threadIdx.x;
    float mg = -INFINITY;
    for (int s = 0; s < nsplits; s++) mg = fmaxf(mg, pm[hg * nsplits + s]);
    float l = 0.f, acc = 0.f;
    for (int s = 0; s < nsplits; s++) {
        const float m = pm[hg * nsplits + s];
        if (m > -INFINITY) { const float w = __expf(m - mg); l += pl[hg * nsplits + s] * w; acc += pacc[(hg * nsplits + s) * vd + d] * w; }
    }
    out[hg * vd + d] = nv_f2h(acc / l);
}

// ---- router GEMV in f32: logits [M][E] = h f16 [M][H] . W f16 [E][H]^T (the pack stores the gate as
// F16; the reference computes in f32). grid (ceil(E/4), M), block 128 = 4 warps = 4 experts.
extern "C" __global__ void nv_router_logits(float* __restrict__ logits, const __half* __restrict__ h,
                                            const __half* __restrict__ w, int M, int E, int H) {
    NV_PDL_ENTRY();
    const int m = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int e = blockIdx.x * 4 + warp;
    if (e >= E) return;
    const __half* hr = h + (long long)m * H;
    const __half* wr = w + (long long)e * H;
    float acc = 0.f;
    for (int i = lane; i < H; i += 32) acc += nv_h2f(hr[i]) * nv_h2f(wr[i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) logits[(long long)m * E + e] = acc;
}

// ---- sigmoid + noaux_tc selection (moe_router_topk_sigmoid_f32_b, row-major ids/wts [M][K]):
// select on sigmoid(l) + bias (ties -> lowest id), weights = the un-biased sigmoid at the picks,
// renormalized (norm_topk) and scaled. grid (M), block 256, smem E floats.
extern "C" __global__ void nv_router_topk(int* __restrict__ ids, float* __restrict__ wts, const float* __restrict__ logits,
                                          const float* __restrict__ bias, int E, int K, int norm_topk, float scaling) {
    NV_PDL_ENTRY();
    const int m = blockIdx.x;
    extern __shared__ float s[];
    for (int e = threadIdx.x; e < E; e += blockDim.x) s[e] = 1.f / (1.f + __expf(-logits[(long long)m * E + e])) + bias[e];
    __syncthreads();
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        for (int j = 0; j < K; j++) {
            float lv = -1e30f; int li = 0x7fffffff;
            for (int e = lane; e < E; e += 32) if (s[e] > lv || (s[e] == lv && e < li)) { lv = s[e]; li = e; }
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_down_sync(0xffffffffu, lv, o);
                const int oi = __shfl_down_sync(0xffffffffu, li, o);
                if (ov > lv || (ov == lv && oi < li)) { lv = ov; li = oi; }
            }
            const int best = __shfl_sync(0xffffffffu, li, 0);
            if (lane == 0) ids[(long long)m * K + j] = best;
            __syncwarp();
            if (lane == 0) s[best] = -1e30f;
            __syncwarp();
        }
        if (lane == 0) {
            float ev[16], wsum = 0.f;
            for (int j = 0; j < K; j++) {
                const float l = logits[(long long)m * E + ids[(long long)m * K + j]];
                ev[j] = 1.f / (1.f + __expf(-l)); wsum += ev[j];
            }
            const float winv = norm_topk ? 1.f / (wsum + 1e-20f) : 1.f;
            for (int j = 0; j < K; j++) wts[(long long)m * K + j] = ev[j] * winv * scaling;
        }
    }
}

// ---- device-side routing (xq_moe_route): first-seen slots over ids[0..M*K] row-major; per-slot
// trellis word offsets for the three expert tables; esel[0] = live expert count. grid (1), block 256.
extern "C" __global__ void nv_moe_route(const int* __restrict__ ids, int mke,
                                        int* __restrict__ slotmap, int* __restrict__ idxmap, int* __restrict__ esel,
                                        unsigned long long* __restrict__ offs_g, unsigned long long* __restrict__ offs_u,
                                        unsigned long long* __restrict__ offs_d,
                                        unsigned long long g_words, unsigned long long u_words, unsigned long long d_words,
                                        const int* __restrict__ local_of) {
    NV_PDL_ENTRY();
    // TP (NAIVE §5.1): local_of[e] = this rank's LOCAL index of expert e (its position in the rank's
    // concatenated trellis / suh / svh), or -1 = another rank's expert: it gets no slot here (slotmap
    // stays -1, so nv_moe_combine skips it on this rank; the owning rank adds it, the fp32 all-reduce
    // of the partial combines sums the ranks). idxmap carries LOCAL indices; slotmap is keyed GLOBAL.
    const int M = mke & 0xFF, K = (mke >> 8) & 0xFF, E = mke >> 16;
    __shared__ int cnt;
    const int tid = threadIdx.x;
    for (int e = tid; e < E; e += blockDim.x) slotmap[e] = -1;
    __syncthreads();
    if (tid == 0) {
        int c = 0;
        for (int t = 0; t < M * K; t++) {
            const int e = ids[t];
            if (e >= 0 && e < E && slotmap[e] == -1) {
                const int le = local_of[e];
                if (le < 0) continue;
                slotmap[e] = c; idxmap[c] = le; c++;
            }
        }
        cnt = c; esel[0] = c;
    }
    __syncthreads();
    for (int s2 = tid; s2 < cnt; s2 += blockDim.x) {
        const int le = idxmap[s2];
        offs_g[s2] = (unsigned long long)le * g_words;
        offs_u[s2] = (unsigned long long)le * u_words;
        offs_d[s2] = (unsigned long long)le * d_words;
    }
}

// ---- per-expert input Hadamard (xq_had_suh_multi): xh[e][m][:] = H(x[m][:] * suh[idxmap[e]][:]);
// per_expert_in: x is itself [E][M][K]. grid E*M*(K/128), block 32.
extern "C" __global__ void nv_had_suh_multi(const __half* __restrict__ x, const __half* __restrict__ suh_all,
                                            const int* __restrict__ idxmap, __half* __restrict__ xh, int M, int K,
                                            const int* __restrict__ esel, int per_expert_in) {
    NV_PDL_ENTRY();
    const int nblk = K >> 7;
    const int blk = blockIdx.x % nblk;
    const long long em = blockIdx.x / nblk;
    const int m = (int)(em % M), e = (int)(em / M);
    if (e >= *esel) return;
    const int lane = threadIdx.x & 31;
    const long long base_in = (long long)(per_expert_in ? e : 0) * M * K + (long long)m * K + (long long)blk * 128 + lane * 4;
    const long long base_out = ((long long)e * M + m) * K + (long long)blk * 128 + lane * 4;
    const __half* sr = suh_all + (long long)idxmap[e] * K + (long long)blk * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) v[r] = nv_h2f(x[base_in + r]) * nv_h2f(sr[r]);
    nv_had128(v, lane);
    #pragma unroll
    for (int r = 0; r < 4; r++) xh[base_out + r] = nv_f2h(v[r]);
}

// ---- per-expert output Hadamard (xq_had_svh_multi): y[e][m][:] = H(yraw[e][m][:]) * svh[idxmap[e]][:].
extern "C" __global__ void nv_had_svh_multi(const __half* __restrict__ yraw, const __half* __restrict__ svh_all,
                                            const int* __restrict__ idxmap, __half* __restrict__ y, int M, int N,
                                            const int* __restrict__ esel) {
    NV_PDL_ENTRY();
    const int nblk = N >> 7;
    const int blk = blockIdx.x % nblk;
    const long long em = blockIdx.x / nblk;
    const int m = (int)(em % M), e = (int)(em / M);
    if (e >= *esel) return;
    const int lane = threadIdx.x & 31;
    const long long base = ((long long)e * M + m) * N + (long long)blk * 128 + lane * 4;
    const __half* sr = svh_all + (long long)idxmap[e] * N + (long long)blk * 128 + lane * 4;
    float v[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) v[r] = nv_h2f(yraw[base + r]);
    nv_had128(v, lane);
    #pragma unroll
    for (int r = 0; r < 4; r++) y[base + r] = nv_f2h(v[r] * nv_h2f(sr[r]));
}

// ---- din[i] = silu(g[i]) * u[i] over n f16 (the expert gate/up products; grids padded to the
// top-k cap are sized by the host from E_cap*M*mi; absent slots compute on zeros harmlessly).
extern "C" __global__ void nv_silu_mul(__half* __restrict__ din, const __half* __restrict__ g,
                                       const __half* __restrict__ u, long long n) {
    NV_PDL_ENTRY();
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = nv_h2f(g[i]);
    din[i] = nv_f2h(gv / (1.f + __expf(-gv)) * nv_h2f(u[i]));
}

// ---- MoE combine into the f32 residual: x[m][:] += sum_j wts[m][j] * y_d[slot(ids[m][j])][m][:].
// grid (M), block 1024.
extern "C" __global__ void nv_moe_combine(float* __restrict__ x, const __half* __restrict__ y_d,
                                          const int* __restrict__ ids, const float* __restrict__ wts,
                                          const int* __restrict__ slotmap, int K, int H, int M, int accumulate) {
    NV_PDL_ENTRY();
    // accumulate = 1: x += combine (the residual); 0: x = combine (TP: this rank's fp32 partial)
    const int m = blockIdx.x;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < K; j++) {
            const int slot = slotmap[ids[(long long)m * K + j]];
            if (slot >= 0) acc += wts[(long long)m * K + j] * nv_h2f(y_d[((long long)slot * M + m) * H + i]);
        }
        if (accumulate) x[(long long)m * H + i] += acc; else x[(long long)m * H + i] = acc;
    }
}

// ---- TP glue (NAIVE §5.1): the partial sums travel as fp32 through the doorbell all-reduce.
// dst[i] = (float) src[i]
extern "C" __global__ void nv_cvt_f16_f32(float* __restrict__ dst, const __half* __restrict__ src, long long n) {
    NV_PDL_ENTRY();
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = nv_h2f(src[i]);
}
// x[i] += y[i] (both f32)
extern "C" __global__ void nv_add_f32(float* __restrict__ x, const float* __restrict__ y, long long n) {
    NV_PDL_ENTRY();
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
// Column compaction: dst[m][0..n) = src[m][off .. off + n) for M rows (the 128-block-rounded k/v
// slices of a rank whose kv-head range is not 128-aligned: TP4 on the DSA layers). grid ceil(M*n/256).
extern "C" __global__ void nv_cols_copy(__half* __restrict__ dst, const __half* __restrict__ src, int M, int src_n, int off, int n) {
    NV_PDL_ENTRY();
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)M * n) return;
    const int m = (int)(i / n), c = (int)(i % n);
    dst[i] = src[(long long)m * src_n + off + c];
}

// =============================================================================
// S6: GPU sampling (the head). sp = [T, top_p, top_k, min_p, rep, pres, freq, _] (device f32).
// =============================================================================
__device__ __forceinline__ unsigned long long nv_splitmix(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// =============================================================================
// S6 F8: the fast exact sampler (multi-block, ~20 small kernels, ~0.1 ms) replacing the single-block
// radix top-k over the vocabulary (0.68 ms). Blocks of 1024 vocabulary entries (SMP_B = ceil(V/1024)).
//   prep:   scaled + penalized logits -> out[V]; per-block max -> bmax[b]
//   stats:  stats[0] = max(bmax)                      z: per-block sum exp(out - max) -> bz;  stats[1] = sum(bz)
//   try t (unrolled 3x in the graph, each gated by acc[0] == 0):
//     gumbel: per-block argmax of out[i] + g_i (g_i = -log(-log(u_i)), u_i = splitmix(seed, step, t, i));
//             T <= 0: g = 0 (the plain argmax; ties -> lower index)  -> bval[b], bidx[b]
//     pick:   c = argmax over the blocks (ties -> lower index)       -> cand[0]
//     mass:   per-block sum of p_i over i with out[i] > out[c], and the count of them -> bmass[b], bcnt[b]
//     accept: nucleus test: mass < top_p (strict), count < top_k (if top_k > 0), p_c >= min_p * p_max;
//             T <= 0 accepts unconditionally. Accepted -> tok[0] = c, acc[0] = 1.
//   fallback (acc[0] still 0 after the tries: a probability <= (1 - nucleus mass)^3, or a tight top-k):
//     the exact single-block radix top-k + nv_sample_cands, both exiting at once when acc[0] == 1.
//   note:   counts[tok] += 1; rng[1] += 1 (the step).
// The sample is EXACT: rejection sampling from the full softmax, accepting iff the draw lies in the
// nucleus = sampling from the renormalized nucleus.
// =============================================================================
#define SMP_T 1024
// Distributed form (S6c, vocab-parallel head): this rank holds the logits of the vocabulary slice
// [v0, v0 + Vl). Every cross-rank value travels in the exchange buffer xb as ZERO-PADDED SLOTS: a
// rank writes its own slot and zeros the others, the fp32 sum all-reduce (exact: adding zeros) leaves
// every rank with every slot = an all-gather on the doorbell transport. World 1: xb holds the one
// slot and no reduce runs. Slot layouts (W = world, r = rank):
//   XB_STATS  [4r + 0..3] = (local max, local Z relative to the local max, cancel flag (rank 0), 0)
//   XB_TRY(t) [4r + 0..3] = (gumbel argmax value, its GLOBAL index, its scaled logit s, 0)
//   XB_MASS(t)[4r + 0..3] = (mass above the candidate, count above, 0, 0)
//   XB_CAND   [2 * (r * CANDL + j) + 0..1] = (s, global index) of the rank's exact top-CANDL (the fallback)
#define XB_STATS 0
#define XB_TRY(t) (16 + 16 * (t))
#define XB_MASS(t) (64 + 16 * (t))
#define XB_CAND 128
#define XB_FLOATS (XB_CAND + 2 * 1024)
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_prep(float* __restrict__ out, float* __restrict__ bmax, const __half* __restrict__ logits,
            const int* __restrict__ counts, const unsigned char* __restrict__ seen, const float* __restrict__ sp, int Vl, int v0) {
    NV_PDL_ENTRY();
    __shared__ float red[SMP_T];
    const int tid = threadIdx.x, v = blockIdx.x * SMP_T + tid;
    const float T = sp[0], rep = sp[4], pres = sp[5], freq = sp[6];
    const float inv_t = (T > 0.f) ? 1.f / T : 1.f;
    float l = -INFINITY;
    if (v < Vl) {
        l = nv_h2f(logits[v]);
        const int c = counts[v0 + v];
        if (rep != 1.f && rep > 0.f && (seen[v0 + v] || c > 0)) l = (l > 0.f) ? l / rep : l * rep;
        if (c > 0) l -= pres + freq * (float)c;
        l *= inv_t;
        out[v] = l;
    }
    red[tid] = l; __syncthreads();
    for (int o = SMP_T / 2; o > 0; o >>= 1) { if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]); __syncthreads(); }
    if (tid == 0) bmax[blockIdx.x] = red[0];
}
// local max -> lmax[0]
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_stats(const float* __restrict__ bmax, float* __restrict__ lmax, int nb) {
    NV_PDL_ENTRY();
    __shared__ float red[SMP_T];
    const int tid = threadIdx.x;
    float m = -INFINITY;
    for (int b = tid; b < nb; b += SMP_T) m = fmaxf(m, bmax[b]);
    red[tid] = m; __syncthreads();
    for (int o = SMP_T / 2; o > 0; o >>= 1) { if (tid < o) red[tid] = fmaxf(red[tid], red[tid + o]); __syncthreads(); }
    if (tid == 0) lmax[0] = red[0];
}
// per-block sum exp(out - local max) -> bz
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_z(const float* __restrict__ out, const float* __restrict__ lmax, float* __restrict__ bz, int Vl) {
    NV_PDL_ENTRY();
    __shared__ float red[SMP_T];
    const int tid = threadIdx.x, v = blockIdx.x * SMP_T + tid;
    red[tid] = (v < Vl) ? __expf(out[v] - lmax[0]) : 0.f; __syncthreads();
    for (int o = SMP_T / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    if (tid == 0) bz[blockIdx.x] = red[0];
}
// the STATS slot: (local max, local Z, cancel flag from rank 0), the other slots zeroed; acc = 0
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_z2(const float* __restrict__ bz, const float* __restrict__ lmax, int nb, float* __restrict__ xb, int rank, int world,
          const int* __restrict__ cancel, int* __restrict__ acc) {
    NV_PDL_ENTRY();
    __shared__ float red[SMP_T];
    const int tid = threadIdx.x;
    float z = 0.f;
    for (int b = tid; b < nb; b += SMP_T) z += bz[b];
    red[tid] = z; __syncthreads();
    for (int o = SMP_T / 2; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
    if (tid < 4 * world) xb[XB_STATS + tid] = 0.f;
    __syncthreads();
    if (tid == 0) {
        xb[XB_STATS + 4 * rank + 0] = lmax[0];
        xb[XB_STATS + 4 * rank + 1] = red[0];
        xb[XB_STATS + 4 * rank + 2] = (rank == 0) ? (float)cancel[0] : 0.f;
        acc[0] = 0;
    }
}
// after the STATS exchange: the global max and Z (Z = sum_r Z_r * exp(max_r - max)), the cancel flag
__device__ __forceinline__ void nv_smp_global_stats(const float* __restrict__ xb, int world, float& mx, float& z) {
    mx = -INFINITY;
    for (int r = 0; r < world; r++) mx = fmaxf(mx, xb[XB_STATS + 4 * r]);
    z = 0.f;
    for (int r = 0; r < world; r++) z += xb[XB_STATS + 4 * r + 1] * __expf(xb[XB_STATS + 4 * r] - mx);
}
__device__ __forceinline__ float nv_gumbel(unsigned long long seed, unsigned long long step, int t, int i) {
    const unsigned long long r = nv_splitmix(seed ^ (step * 0x9E3779B97F4A7C15ull) ^ ((unsigned long long)(t + 1) << 56) ^ ((unsigned long long)i * 0xD1B54A32D192ED03ull));
    const float u = ((float)(r >> 40) + 0.5f) * (1.f / 16777216.f);   // (0, 1)
    return -__logf(-__logf(u));
}
// per-block argmax of out[i] + gumbel(global index) (T <= 0: plain argmax). Ties -> lower index.
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_gumbel(const float* __restrict__ out, const float* __restrict__ sp, const unsigned long long* __restrict__ rng,
              int t, float* __restrict__ bval, int* __restrict__ bidx, int Vl, int v0, const int* __restrict__ acc) {
    NV_PDL_ENTRY();
    if (acc[0]) return;
    __shared__ float rv[SMP_T];
    __shared__ int ri[SMP_T];
    const int tid = threadIdx.x, v = blockIdx.x * SMP_T + tid;
    float val = -INFINITY;
    if (v < Vl) { val = out[v]; if (sp[0] > 0.f) val += nv_gumbel(rng[0], rng[1], t, v0 + v); }
    rv[tid] = val; ri[tid] = v0 + v; __syncthreads();
    for (int o = SMP_T / 2; o > 0; o >>= 1) {
        if (tid < o) {
            const float a = rv[tid], b = rv[tid + o];
            if (b > a || (b == a && ri[tid + o] < ri[tid])) { rv[tid] = b; ri[tid] = ri[tid + o]; }
        }
        __syncthreads();
    }
    if (tid == 0) { bval[blockIdx.x] = rv[0]; bidx[blockIdx.x] = ri[0]; }
}
// the rank's candidate -> its TRY slot (value, global index, s), other slots zeroed
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_pick(const float* __restrict__ bval, const int* __restrict__ bidx, int nb, const float* __restrict__ out, int v0,
            float* __restrict__ xb, int t, int rank, int world, const int* __restrict__ acc) {
    NV_PDL_ENTRY();
    if (acc[0]) return;
    __shared__ float rv[SMP_T];
    __shared__ int ri[SMP_T];
    const int tid = threadIdx.x;
    float val = -INFINITY; int idx = 0x7fffffff;
    for (int b = tid; b < nb; b += SMP_T) { const float x = bval[b]; const int i = bidx[b]; if (x > val || (x == val && i < idx)) { val = x; idx = i; } }
    rv[tid] = val; ri[tid] = idx; __syncthreads();
    for (int o = SMP_T / 2; o > 0; o >>= 1) {
        if (tid < o) {
            const float a = rv[tid], b = rv[tid + o];
            if (b > a || (b == a && ri[tid + o] < ri[tid])) { rv[tid] = b; ri[tid] = ri[tid + o]; }
        }
        __syncthreads();
    }
    if (tid < 4 * world) xb[XB_TRY(t) + tid] = 0.f;
    __syncthreads();
    if (tid == 0) {
        xb[XB_TRY(t) + 4 * rank + 0] = rv[0];
        xb[XB_TRY(t) + 4 * rank + 1] = (float)ri[0];
        xb[XB_TRY(t) + 4 * rank + 2] = out[ri[0] - v0];
    }
}
// after the TRY exchange: the global candidate (max value, ties -> lower index) and its s
__device__ __forceinline__ void nv_smp_global_cand(const float* __restrict__ xb, int t, int world, int& ci, float& sc) {
    float best = -INFINITY; ci = 0x7fffffff; sc = 0.f;
    for (int r = 0; r < world; r++) {
        const float v = xb[XB_TRY(t) + 4 * r];
        const int i = (int)xb[XB_TRY(t) + 4 * r + 1];
        if (v > best || (v == best && i < ci)) { best = v; ci = i; sc = xb[XB_TRY(t) + 4 * r + 2]; }
    }
}
// per-block: mass and count of the LOCAL entries strictly above the candidate's s
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_mass(const float* __restrict__ out, const float* __restrict__ xb, int t, int world,
            float* __restrict__ bmass, int* __restrict__ bcnt, int Vl, const int* __restrict__ acc) {
    NV_PDL_ENTRY();
    if (acc[0]) return;
    __shared__ float red[SMP_T];
    __shared__ int cnt[SMP_T];
    const int tid = threadIdx.x, v = blockIdx.x * SMP_T + tid;
    float mx, z; nv_smp_global_stats(xb, world, mx, z);
    int ci; float sc; nv_smp_global_cand(xb, t, world, ci, sc);
    float m = 0.f; int c = 0;
    if (v < Vl && out[v] > sc) { m = __expf(out[v] - mx) / z; c = 1; }
    red[tid] = m; cnt[tid] = c; __syncthreads();
    for (int o = SMP_T / 2; o > 0; o >>= 1) { if (tid < o) { red[tid] += red[tid + o]; cnt[tid] += cnt[tid + o]; } __syncthreads(); }
    if (tid == 0) { bmass[blockIdx.x] = red[0]; bcnt[blockIdx.x] = (float)cnt[0] == 0.f ? 0 : cnt[0]; bcnt[blockIdx.x] = cnt[0]; }
}
// the rank's (mass, count) -> its MASS slot
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_mass2(const float* __restrict__ bmass, const int* __restrict__ bcnt, int nb, float* __restrict__ xb, int t, int rank, int world,
             const int* __restrict__ acc) {
    NV_PDL_ENTRY();
    if (acc[0]) return;
    __shared__ float red[SMP_T];
    __shared__ int cnt[SMP_T];
    const int tid = threadIdx.x;
    float m = 0.f; int c = 0;
    for (int b = tid; b < nb; b += SMP_T) { m += bmass[b]; c += bcnt[b]; }
    red[tid] = m; cnt[tid] = c; __syncthreads();
    for (int o = SMP_T / 2; o > 0; o >>= 1) { if (tid < o) { red[tid] += red[tid + o]; cnt[tid] += cnt[tid + o]; } __syncthreads(); }
    if (tid < 4 * world) xb[XB_MASS(t) + tid] = 0.f;
    __syncthreads();
    if (tid == 0) { xb[XB_MASS(t) + 4 * rank + 0] = red[0]; xb[XB_MASS(t) + 4 * rank + 1] = (float)cnt[0]; }
}
// after the MASS exchange: the nucleus test, identical on every rank
extern "C" __global__ void nv_smp_accept(const float* __restrict__ xb, int t, int world, const float* __restrict__ sp,
                                         int* __restrict__ tok, int* __restrict__ acc) {
    NV_PDL_ENTRY();
    if (threadIdx.x != 0 || acc[0]) return;
    float mx, z; nv_smp_global_stats(xb, world, mx, z);
    int ci; float sc; nv_smp_global_cand(xb, t, world, ci, sc);
    float mass = 0.f; int cnt = 0;
    for (int r = 0; r < world; r++) { mass += xb[XB_MASS(t) + 4 * r]; cnt += (int)xb[XB_MASS(t) + 4 * r + 1]; }
    const float T = sp[0], top_p = sp[1], min_p = sp[3];
    const int top_k = (int)sp[2];
    bool ok = true;
    if (T > 0.f) {
        if (top_p > 0.f && top_p < 1.f && mass >= top_p) ok = false;
        if (top_k > 0 && cnt >= top_k) ok = false;
        if (min_p > 0.f) { const float pc = __expf(sc - mx) / z; if (pc < min_p / z) ok = false; }
    }
    if (ok) { tok[0] = ci; acc[0] = 1; }
}
// the exact fallback, distributed: this rank's top-CANDL (sel over the local slice, from nv_topk) ->
// (s, global index) pairs in its CAND slot (other slots zeroed; nsel < CANDL pads with -inf)
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_cands_out(const int* __restrict__ sel, const int* __restrict__ nsel, const float* __restrict__ out, int v0,
                 float* __restrict__ xb, int candl, int rank, int world) {
    NV_PDL_ENTRY();
    const int tid = threadIdx.x;
    for (int j = tid; j < candl * world; j += SMP_T) { xb[XB_CAND + 2 * j] = 0.f; xb[XB_CAND + 2 * j + 1] = 0.f; }
    __syncthreads();
    const int n = *nsel;
    for (int j = tid; j < candl; j += SMP_T) {
        const int slot = XB_CAND + 2 * (rank * candl + j);
        if (j < n) { const int i = sel[j]; xb[slot] = out[i]; xb[slot + 1] = (float)(v0 + i); }
        else { xb[slot] = -INFINITY; xb[slot + 1] = -1.f; }
    }
}
// after the CAND exchange: sort the gathered candidates (p desc, index asc), the cuts, one draw —
// identical on every rank. Only when no Gumbel try was accepted.
extern "C" __global__ void __launch_bounds__(SMP_T)
nv_smp_sample_cands(int* __restrict__ tok, const float* __restrict__ xb, int world, int candl, const float* __restrict__ sp,
                    const unsigned long long* __restrict__ rng, const int* __restrict__ acc) {
    NV_PDL_ENTRY();
    if (acc[0]) return;
    __shared__ float sp_v[1024];
    __shared__ int sp_i[1024];
    __shared__ float red[1024];
    const int tid = threadIdx.x;
    const int n = candl * world;    // <= 1024
    const float T = sp[0], top_p = sp[1], min_p = sp[3];
    const int top_k = (int)sp[2];
    float mx, z; nv_smp_global_stats(xb, world, mx, z);
    if (tid < n) {
        const float s = xb[XB_CAND + 2 * tid]; const int i = (int)xb[XB_CAND + 2 * tid + 1];
        sp_i[tid] = (i < 0) ? 0x7fffffff : i;
        sp_v[tid] = (i < 0) ? -INFINITY : ((T > 0.f) ? __expf(s - mx) / z : s);
    } else { sp_i[tid] = 0x7fffffff; sp_v[tid] = -INFINITY; }
    __syncthreads();
    for (int k = 2; k <= 1024; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int ixj = tid ^ j;
            if (ixj > tid) {
                const float a = sp_v[tid], b = sp_v[ixj];
                const int ia = sp_i[tid], ib = sp_i[ixj];
                const bool a_first = (a > b) || (a == b && ia < ib);
                const bool up = ((tid & k) == 0);
                if (up != a_first) { sp_v[tid] = b; sp_v[ixj] = a; sp_i[tid] = ib; sp_i[ixj] = ia; }
            }
            __syncthreads();
        }
    }
    int live = 0;
    for (int i = 0; i < 1024; i++) { if (sp_v[i] == -INFINITY) break; live++; }   // (sorted: the -inf pads are last)
    if (T <= 0.f) { if (tid == 0) tok[0] = sp_i[0]; return; }
    int keep = live;
    if (top_k > 0) keep = min(keep, top_k);
    if (min_p > 0.f) {
        const float floor_p = sp_v[0] * min_p;
        red[tid] = (tid < keep && sp_v[tid] >= floor_p) ? 1.f : 0.f; __syncthreads();
        for (int o = 512; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
        keep = max(1, (int)red[0]); __syncthreads();
    }
    red[tid] = (tid < keep) ? sp_v[tid] : 0.f; __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) { const float v = (tid >= o) ? red[tid - o] : 0.f; __syncthreads(); red[tid] += v; __syncthreads(); }
    if (top_p > 0.f && top_p < 1.f) {
        __shared__ int s_cut;
        if (tid == 0) s_cut = keep;
        __syncthreads();
        if (tid < keep && red[tid] >= top_p && (tid == 0 || red[tid - 1] < top_p)) s_cut = tid + 1;
        __syncthreads();
        keep = min(keep, s_cut);
    }
    if (tid == 0) {
        const float total = red[keep - 1];
        const unsigned long long r = nv_splitmix(rng[0] ^ (rng[1] * 0x9E3779B97F4A7C15ull));
        const float u = (float)(r >> 40) * (1.f / 16777216.f) * total;
        int pick = keep - 1;
        for (int i = 0; i < keep; i++) { if (u < red[i]) { pick = i; break; } }
        tok[0] = sp_i[pick];
    }
}
// ---- exact speculative acceptance (work item A of PLAN/NAIVE_DRAFT2_PROMPT_2026-10-19.md), for
// SAMPLED drafts. ONE BLOCK PER DRAFT ROW j (grid k); no exchange, so the head may run it while the
// node ranks skip it (their doorbell epochs are unaffected).
//
//   p_trunc  = the SAME distribution nv_smp_sample_cands draws from: the sorted all-gathered
//              candidates, cut by top_k / min_p / top_p and renormalised by the nucleus mass.
//   q(v)     = exp(qs[j][v] - qstat[2j]) / qstat[2j+1]   (the head's full-vocabulary drafter logits,
//              already scaled by 1/T, for draft row j)
//
// rule (exact speculative sampling, Leviathan 2023 / Chen 2023):
//   u < min(1, p(d)/q(d))  ->  accept d, the block continues
//   otherwise              ->  commit one draw from norm(max(0, p - q)), the block STOPS
//
// out: acc[j] = 1/0, rtok[j] = the residual draw (meaningful only when rejected). `on[0]` == 0 (the
// greedy path, or a node rank) makes the whole kernel a no-op so the greedy token stream stays
// bit-identical and the graph layout is shared.
// grid (1) per draft row, block 1024; launched inside the verify row loop.
extern "C" __global__ void __launch_bounds__(1024)
nv_spec_accept_rows(const int* __restrict__ on, const float* __restrict__ xb, int world, int n,
                    const float* __restrict__ sp, int row, const int* __restrict__ draft, const float* __restrict__ qs,
                    const float* __restrict__ qstat, const unsigned long long* __restrict__ rngp,
                    int* __restrict__ acc, int* __restrict__ rtok) {
    NV_PDL_ENTRY();
    if (!on[0]) return;
    // ONE launch per row, inside the row loop: `xb` holds THAT row's gathered candidate list and
    // XB_STATS that row's (max, z), both overwritten by the next row. The accept draw's seed/step are
    // read from device memory because the graph bakes kernel ARGUMENTS at capture.
    const int j = row;
    const unsigned long long rng = rngp[0];
    const int step = (int)rngp[1];
    // rngp[2] = 1: the drafts are argmax draws (F = 0 / the sharded drafter): q is ONE-HOT at the draft, so the
    // test is u < p(d) and the residual is p without d (no q buffer is written in that mode)
    const int onehot = (int)rngp[2];
    const int d = draft[j];
    __shared__ float sp_v[1024];
    __shared__ int sp_i[1024];
    __shared__ float red[1024];
    __shared__ int s_cut;
    const int tid = threadIdx.x;
    const float T = sp[0], top_p = sp[1], min_p = sp[3];
    const int top_k = (int)sp[2];
    float mx, z; nv_smp_global_stats(xb, world, mx, z);
    if (tid < n) {
        const float s = xb[XB_CAND + 2 * tid]; const int i = (int)xb[XB_CAND + 2 * tid + 1];
        sp_i[tid] = (i < 0) ? 0x7fffffff : i;
        sp_v[tid] = (i < 0) ? -INFINITY : ((T > 0.f) ? __expf(s - mx) / z : s);
    } else { sp_i[tid] = 0x7fffffff; sp_v[tid] = -INFINITY; }
    __syncthreads();
    // p descending, index ascending: the order nv_smp_sample_cands samples in
    for (int k = 2; k <= 1024; k <<= 1) {
        for (int jj = k >> 1; jj > 0; jj >>= 1) {
            const int ixj = tid ^ jj;
            if (ixj > tid) {
                const float a = sp_v[tid], b = sp_v[ixj];
                const int ia = sp_i[tid], ib = sp_i[ixj];
                const bool a_first = (a > b) || (a == b && ia < ib);
                const bool up = ((tid & k) == 0);
                if (up != a_first) { sp_v[tid] = b; sp_v[ixj] = a; sp_i[tid] = ib; sp_i[ixj] = ia; }
            }
            __syncthreads();
        }
    }
    int live = 0;
    for (int i = 0; i < 1024; i++) { if (sp_v[i] == -INFINITY) break; live++; }
    if (T <= 0.f) return;                       // greedy never reaches this kernel (guard kept explicit)
    int keep = live;
    if (top_k > 0) keep = min(keep, top_k);
    if (min_p > 0.f) {
        const float floor_p = sp_v[0] * min_p;
        red[tid] = (tid < keep && sp_v[tid] >= floor_p) ? 1.f : 0.f; __syncthreads();
        for (int o = 512; o > 0; o >>= 1) { if (tid < o) red[tid] += red[tid + o]; __syncthreads(); }
        keep = max(1, (int)red[0]); __syncthreads();
    }
    red[tid] = (tid < keep) ? sp_v[tid] : 0.f; __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) { const float v = (tid >= o) ? red[tid - o] : 0.f; __syncthreads(); red[tid] += v; __syncthreads(); }
    if (top_p > 0.f && top_p < 1.f) {
        if (tid == 0) s_cut = keep;
        __syncthreads();
        if (tid < keep && red[tid] >= top_p && (tid == 0 || red[tid - 1] < top_p)) s_cut = tid + 1;
        __syncthreads();
        keep = min(keep, s_cut);
    }
    const float total = (keep > 0) ? red[keep - 1] : 0.f;
    // No candidate list (the gather did not run) or no mass: nothing to test against. Reject the row
    // and hand the decision back (rtok = -1 -> the host commits the target's own draw for the row and
    // counts the fallback) instead of indexing the list at -1.
    if (keep <= 0 || total <= 0.f) { if (tid == 0) { acc[j] = 0; rtok[j] = -1; } return; }   // -1: no candidate list
    // the accept test, and the residual weights r(i) = max(0, p_trunc(i) - q(id_i))
    float ri = 0.f;
    if (tid < keep && total > 0.f) {
        const float p = sp_v[tid] / total;
        const int id = sp_i[tid];
        const float q = onehot ? ((id == d) ? 1.f : 0.f) : ((id >= 0) ? __expf(qs[id] - qstat[2 * j]) / qstat[2 * j + 1] : 0.f);
        ri = fmaxf(0.f, p - q);
    }
    __syncthreads();
    red[tid] = ri; __syncthreads();
    for (int o = 1; o < 1024; o <<= 1) { const float v = (tid >= o) ? red[tid - o] : 0.f; __syncthreads(); red[tid] += v; __syncthreads(); }
    if (tid == 0) {
        float pd = 0.f;
        for (int i = 0; i < keep; i++) if (sp_i[i] == d) { pd = sp_v[i] / total; break; }
        const float qd = onehot ? 1.f : ((d >= 0) ? __expf(qs[d] - qstat[2 * j]) / qstat[2 * j + 1] : 0.f);
        const unsigned long long r0 = nv_splitmix(rng ^ ((unsigned long long)(step + 1) * 0x9E3779B97F4A7C15ull)
                                                      ^ ((unsigned long long)(j + 1) * 0xA24BAED4963EE407ull));
        const float u = (float)(r0 >> 40) * (1.f / 16777216.f);          // [0, 1)
        const bool ok = (qd <= 0.f) || (u * qd < pd);                     // u < min(1, pd/qd)
        acc[j] = ok ? 1 : 0;
        if (!ok) {
            const float tot = red[keep - 1];                              // = 1 - sum_c min(p, q)
            if (!(tot > 0.f)) { rtok[j] = -2; }                           // -2: empty residual (q >= p on every candidate, or NaN): host fallback
            else {
                const unsigned long long r1 = nv_splitmix(r0 ^ 0x5EC0DEADBEEFCAFEull);
                const float u2 = (float)(r1 >> 40) * (1.f / 16777216.f) * tot;
                int pick = keep - 1;
                for (int i = 0; i < keep; i++) { if (u2 < red[i]) { pick = i; break; } }
                rtok[j] = sp_i[pick];
            }
        }
    }
}
// the fallback gate for the exact path + the per-step bookkeeping
extern "C" __global__ void nv_smp_note(int* __restrict__ counts, const int* __restrict__ tok, unsigned long long* __restrict__ rng,
                                       const float* __restrict__ xb, int* __restrict__ flags, int count) {
    NV_PDL_ENTRY();
    if (threadIdx.x == 0) { if (count) atomicAdd(&counts[tok[0]], 1); rng[1] += 1ull; flags[0] = (int)xb[XB_STATS + 2]; }
}
// ---- work item C (PLAN/NAIVE_DRAFT2_PROMPT_2026-10-19.md §6): the verify block's rows sampled PHASE-MAJOR ----
// rows_pack: dir 0 gathers section [off, off+n) of every row's exchange buffer (row stride `stride` floats)
// into a contiguous staging area [m][n] so ONE all-reduce carries every row's slots; dir 1 scatters it back.
extern "C" __global__ void nv_rows_pack(float* __restrict__ xs, float* __restrict__ xb, int m, int stride, int off, int n, int dir) {
    NV_PDL_ENTRY();
    const int j = blockIdx.x;
    if (j >= m) return;
    float* row = xb + (size_t)j * stride + off;
    float* st = xs + (size_t)j * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { if (dir == 0) st[i] = row[i]; else row[i] = st[i]; }
}
// rng_rows: the per-row path advances the lane's step once per row (nv_smp_note), so row j draws from
// (seed, step + j). The batched rows read exactly those pairs from their own copies (bitwise the same draws).
extern "C" __global__ void nv_rng_rows(const unsigned long long* __restrict__ rng, unsigned long long* __restrict__ rows, int m) {
    NV_PDL_ENTRY();
    const int j = threadIdx.x;
    if (j < m) { rows[2 * j] = rng[0]; rows[2 * j + 1] = rng[1] + (unsigned long long)j; }
}
// the committed tokens of a speculative round -> the frequency / presence table (eager, after the round)
extern "C" __global__ void nv_smp_count(int* __restrict__ counts, const int* __restrict__ toks, int n) {
    NV_PDL_ENTRY();
    const int i = threadIdx.x;
    if (i < n) atomicAdd(&counts[toks[i]], 1);
}
