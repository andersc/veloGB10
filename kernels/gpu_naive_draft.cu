// =============================================================================
// kernels/gpu_naive_draft.cu — the Naive-N0.5-Flash DSpark drafter round on the GPU
// (PLAN/NAIVE_N05_PORT.md §1.2, stage S8). Loaded by src/naive/drafter_gpu.rs only.
//
// The drafter is NOT quantized (bf16 checkpoint): every weight lives on the device as plain f16
// (bf16 -> f16 at load; the loader counts any value outside the f16 range). The semantics are those
// of the CPU reference `naive::drafter_cpu::CpuDrafter::round` (itself gated against the HF modules).
//
// Rounding points (every other value is f32):
//   * the GEMM inputs: normed activations, the tap concat (fc input), silu(g)*u, the final hidden
//     going to the lm_head — f16 (single rounding from f32);
//   * q / k / v after q/k RMSNorm + rope — f16 (the K/V ring holds f16);
//   * the attention output (o_proj input) — f16;
//   * the base logits — f16 (that is what the target's lm_head chain produces);
//   * GEMM accumulation is fp32 (mma.sync f16 x f16 -> f32), split-K partials summed in a FIXED
//     order by the combine kernels (batch / run invariant, no atomics).
//   The residual stream, the norms, softmax, the Markov bias, the corrected logits and the
//   confidence logit stay f32.
//
// Target: sm_121 baseline (fp16 mma.m16n8k16, shuffles). M (rows) is small: <= 16 for a round,
// up to 128 per launch for the context append (MT m16-tiles per warp share one weight load).
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#ifndef KERNEL_BUILD_ID
#define KERNEL_BUILD_ID 0ULL
#endif
extern "C" __global__ void nd_kernel_build_id(unsigned long long* out) { *out = KERNEL_BUILD_ID; }

// PDL entry (same contract as gpu_naive.cu: no-ops on a plain launch).
#define NV_PDL_ENTRY() do { asm volatile("griddepcontrol.wait;" ::: "memory"); \
                            asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); } while (0)

__device__ __forceinline__ float nv_h2f(__half h) { return __half2float(h); }
__device__ __forceinline__ __half nv_f2h(float f) { return __float2half_rn(f); }

// Weights are read exactly once per round: stream them past L1.
__device__ __forceinline__ uint4 nd_ld_stream(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

__device__ __forceinline__ void nd_mma(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ---------------------------------------------------------------------------
// Small-M dense GEMM: part[ks][M][N] (f32) = X f16 [M][K] . W f16 [N][K]^T over K-chunk ks.
// One warp = one n8 column tile x MT m16 row tiles x one K chunk. grid (N/8/ND_WPB, KS), block
// ND_WPB*32. K % 128 == 0 per chunk (kchunk % 128 == 0), N % 8 == 0, X rows >= M read as zero.
//
// Why the K permutation: the m16n8k16 B fragment wants, per thread (g = lane/4, t = lane%4),
// W[n0+g][k0+2t+{0,1}] and W[n0+g][k0+2t+8+{0,1}] — 4-byte scattered loads. A dot product does not
// care about the ORDER of k, so the 32 k of one load step are relabelled: thread t reads the 16
// contiguous bytes W[n0+g][k0+8t .. 8t+7] (one 128-bit load) and the two mma k16 steps take halves
// {0..3} and {4..7} of it. The A fragment is loaded with the SAME relabelling (X[row][k0+8t..]), so
// every product pairs the same physical k on both sides. Fixed k order per thread + fixed shuffle-
// free mma reduction => deterministic.
// ---------------------------------------------------------------------------
#define ND_WPB 4

template <int MT>
__device__ __forceinline__ void nd_gemm_body(const __half* __restrict__ W, const __half* __restrict__ X,
                                             float* __restrict__ part, int M, int K, int N, int kchunk) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * ND_WPB + warp) * 8;
    if (n0 >= N) return;
    const int g = lane >> 2, t = lane & 3;
    const int kb = blockIdx.y * kchunk;
    const __half* wrow = W + (size_t)(n0 + g) * K + kb + t * 8;
    float c[MT][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) { c[mt][0] = c[mt][1] = c[mt][2] = c[mt][3] = 0.f; }
    for (int k = 0; k < kchunk; k += 128) {
        uint4 b[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) b[u] = nd_ld_stream(wrow + k + 32 * u);
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int ra = mt * 16 + g, rb = ra + 8;
            const __half* xa = X + (size_t)ra * K + kb + k + t * 8;
            const __half* xb = X + (size_t)rb * K + kb + k + t * 8;
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                const uint4 z = make_uint4(0, 0, 0, 0);
                const uint4 a0 = ra < M ? __ldg((const uint4*)(xa + 32 * u)) : z;
                const uint4 a1 = rb < M ? __ldg((const uint4*)(xb + 32 * u)) : z;
                nd_mma(c[mt], a0.x, a1.x, a0.y, a1.y, b[u].x, b[u].y);
                nd_mma(c[mt], a0.z, a1.z, a0.w, a1.w, b[u].z, b[u].w);
            }
        }
    }
    float* p = part + (size_t)blockIdx.y * M * N;
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int ra = mt * 16 + g, rb = ra + 8, col = n0 + 2 * t;
        if (ra < M) { p[(size_t)ra * N + col] = c[mt][0]; p[(size_t)ra * N + col + 1] = c[mt][1]; }
        if (rb < M) { p[(size_t)rb * N + col] = c[mt][2]; p[(size_t)rb * N + col + 1] = c[mt][3]; }
    }
}

#define ND_GEMM(MT) \
extern "C" __global__ void __launch_bounds__(ND_WPB * 32) nd_gemm_m##MT(const __half* __restrict__ W, const __half* __restrict__ X, \
        float* __restrict__ part, int M, int K, int N, int kchunk) { \
    NV_PDL_ENTRY(); nd_gemm_body<MT>(W, X, part, M, K, N, kchunk); }
ND_GEMM(1)
ND_GEMM(2)
ND_GEMM(4)
ND_GEMM(8)

// ---- split-K combine: r = sum_{s < KS} part[s][i] (fixed order); add != 0: out32[i] += r, else
// out32[i] = r (out32 may be null); out16 (may be null) = f16(final value). grid-stride, block 256.
extern "C" __global__ void nd_combine(const float* __restrict__ part, int KS, long long MN,
                                      float* __restrict__ out32, __half* __restrict__ out16, int add) {
    NV_PDL_ENTRY();
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < MN; i += (long long)gridDim.x * blockDim.x) {
        float r = 0.f;
        for (int s = 0; s < KS; ++s) r += part[s * MN + i];
        if (add) r += out32[i];
        if (out32) out32[i] = r;
        if (out16) out16[i] = nv_f2h(r);
    }
}

// ---- combine of the fused gate|up GEMM (N = 2I: gate cols [0, I), up cols [I, 2I)) straight into
// the down_proj input: act f16 [M][I] = f16(silu(g) * u), g/u the f32 split-K sums.
extern "C" __global__ void nd_combine_silu(const float* __restrict__ part, int KS, int M, int I, __half* __restrict__ act) {
    NV_PDL_ENTRY();
    const long long MN = (long long)M * 2 * I;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < (long long)M * I; i += (long long)gridDim.x * blockDim.x) {
        const long long m = i / I, n = i % I;
        float gs = 0.f, us = 0.f;
        for (int s = 0; s < KS; ++s) { gs += part[s * MN + m * 2 * I + n]; us += part[s * MN + m * 2 * I + I + n]; }
        act[i] = nv_f2h(gs / (1.f + __expf(-gs)) * us);
    }
}

// ---- block-wide f32 sum (blockDim.x a multiple of 32, <= 1024); `red` >= 32 floats.
__device__ __forceinline__ float nd_block_sum(float v, float* red) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float s = 0.f;
    for (int w = 0; w < nw; ++w) s += red[w];   // every thread, same fixed order
    return s;
}
// ---- RMSNorm (Qwen3 / Llama): y = x * rsqrt(mean(x^2) + eps) * w; x f32 [M][H], w f32 [H].
// out16 f16 [M][H] (the next GEMM's input), out32 (may be null) the same values unrounded (the
// final hidden feeds the confidence head in f32, as the reference does). grid (M), block 256.
extern "C" __global__ void nd_rmsnorm(__half* __restrict__ out16, float* __restrict__ out32, const float* __restrict__ x,
                                      const float* __restrict__ w, int H, float eps) {
    NV_PDL_ENTRY();
    __shared__ float red[32];
    const int m = blockIdx.x;
    const float* xr = x + (size_t)m * H;
    float ss = 0.f;
    for (int i = threadIdx.x; i < H; i += blockDim.x) ss += xr[i] * xr[i];
    const float inv = rsqrtf(nd_block_sum(ss, red) / (float)H + eps);
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const float y = xr[i] * inv * w[i];
        if (out16) out16[(size_t)m * H + i] = nv_f2h(y);
        if (out32) out32[(size_t)m * H + i] = y;
    }
}

// ---- one target tap (f32 rows at pitch src_stride) -> its column block of the fc input
// dst f16 [rows][dst_stride] at column col_off (rounding point: the fc input). grid (rows), block 256.
extern "C" __global__ void nd_tap_f16(__half* __restrict__ dst, const float* __restrict__ src, long long src_stride,
                                      int dst_stride, int col_off, int H) {
    NV_PDL_ENTRY();
    const int r = blockIdx.x;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        dst[(size_t)r * dst_stride + col_off + i] = nv_f2h(src[r * src_stride + i]);
}

// ---- full-dim NeoX rotary on one head held in smem f32 (rotate_half pairs (d, d + hd/2)), angle
// from double-precision sincos of pos * theta^(-2d/hd) (gpu_naive.cu's form: exact at 1M positions).
__device__ __forceinline__ void nd_rope_smem(float* v, int pos, float theta, int hd) {
    const int half = hd >> 1;
    for (int d = threadIdx.x; d < half; d += blockDim.x) {
        const double inv = pow((double)theta, -(2.0 * d) / (double)hd);
        double sn, cs;
        sincos((double)pos * inv, &sn, &cs);
        const float c = (float)cs, s = (float)sn;
        const float x1 = v[d], x2 = v[d + half];
        v[d] = x1 * c - x2 * s;
        v[d + half] = x2 * c + x1 * s;
    }
    __syncthreads();
}

// per-head RMSNorm (weight [hd]) + rope of src f32 [hd] -> dst f16 [hd]; block of >= hd/2 threads
__device__ __forceinline__ void nd_head_norm_rope(__half* __restrict__ dst, const float* __restrict__ src, const float* __restrict__ w,
                                                  int hd, float eps, int pos, float theta, float* sv, float* red) {
    float ss = 0.f;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) { const float v = src[d]; sv[d] = v; ss += v * v; }
    const float inv = rsqrtf(nd_block_sum(ss, red) / (float)hd + eps);   // (the sum's syncs publish sv)
    for (int d = threadIdx.x; d < hd; d += blockDim.x) sv[d] = sv[d] * inv * w[d];
    __syncthreads();
    nd_rope_smem(sv, pos, theta, hd);
    for (int d = threadIdx.x; d < hd; d += blockDim.x) dst[d] = nv_f2h(sv[d]);
}

// ---- context rows -> one layer's K/V ring: raw f32 [rows][2*kv*hd] (k cols then v cols) at
// positions pos0 + r; k: k_norm + rope at its position; slot = pos & (S-1); ring kc/vc f16
// [S][kv][hd]. grid (rows * kv), block 64 (hd <= 128).
extern "C" __global__ void nd_ctx_kv(__half* __restrict__ kc, __half* __restrict__ vc, const float* __restrict__ raw,
                                     const float* __restrict__ kn, int pos0, int kv, int hd, int S, float eps, float theta) {
    NV_PDL_ENTRY();
    __shared__ float sv[128];
    __shared__ float red[32];
    const int r = blockIdx.x / kv, g = blockIdx.x % kv;
    const int p = pos0 + r;
    const int slot = p & (S - 1);
    const float* kr = raw + (size_t)r * 2 * kv * hd + g * hd;
    const float* vr = kr + kv * hd;
    nd_head_norm_rope(kc + ((size_t)slot * kv + g) * hd, kr, kn, hd, eps, p, theta, sv, red);
    for (int d = threadIdx.x; d < hd; d += blockDim.x) vc[((size_t)slot * kv + g) * hd + d] = nv_f2h(vr[d]);
}

// ---- block rows: x f32 [n][H] = anchor embedding (target table, bf16 bits) then the learned mask
// embedding (f32) for rows 1..n-1; tok[0] = anchor (the Markov chain's first `prev`). grid (n), block 256.
extern "C" __global__ void nd_block_init(float* __restrict__ x, const unsigned short* __restrict__ embed,
                                         const float* __restrict__ mask, int* __restrict__ tok, int anchor, int H) {
    NV_PDL_ENTRY();
    const int m = blockIdx.x;
    if (m == 0 && threadIdx.x == 0) tok[0] = anchor;
    for (int i = threadIdx.x; i < H; i += blockDim.x)
        x[(size_t)m * H + i] = m == 0 ? __uint_as_float((unsigned)embed[(size_t)anchor * H + i] << 16) : mask[i];
}

// ---- block q/k/v: raw f32 [n][(nh + 2 kv) * hd] (q | k | v cols) -> q16 [n][nh][hd], kb16/vb16
// [n][kv][hd]; q/k: per-head RMSNorm + rope at position t + i. grid (n * (nh + 2 kv)), block 64.
extern "C" __global__ void nd_block_qkv(__half* __restrict__ q16, __half* __restrict__ kb16, __half* __restrict__ vb16,
                                        const float* __restrict__ raw, const float* __restrict__ qn, const float* __restrict__ kn,
                                        int t, int nh, int kv, int hd, float eps, float theta) {
    NV_PDL_ENTRY();
    __shared__ float sv[128];
    __shared__ float red[32];
    const int per = nh + 2 * kv;
    const int i = blockIdx.x / per, h = blockIdx.x % per;
    const float* src = raw + (size_t)i * per * hd + (size_t)h * hd;
    if (h < nh) {
        nd_head_norm_rope(q16 + ((size_t)i * nh + h) * hd, src, qn, hd, eps, t + i, theta, sv, red);
    } else if (h < nh + kv) {
        nd_head_norm_rope(kb16 + ((size_t)i * kv + (h - nh)) * hd, src, kn, hd, eps, t + i, theta, sv, red);
    } else {
        __half* dst = vb16 + ((size_t)i * kv + (h - nh - kv)) * hd;
        for (int d = threadIdx.x; d < hd; d += blockDim.x) dst[d] = nv_f2h(src[d]);
    }
}

// ---- the block's attention, NON-causal within the block (dflash_config.causal = false):
// query row i (position t + i) sees the context keys p in [max(0, t+i-W+1), t) from the ring and ALL
// n block keys; softmax(q.k * scale) . V; out f16 [n][nh][hd].
//
// Split-K (flash-decoding) over the key list of the WHOLE block: key j < nctx0 is ring position
// lo0 + j (lo0 = row 0's window start, the earliest any row sees), key j >= nctx0 is block row
// j - nctx0. One CTA = one kv group x one chunk of <= ND_CH keys, for ALL n rows x gqa heads of that
// group: every K/V row is read from L2 once per round instead of once per (row, head) (the first
// version, a CTA per (head, row), spent 0.44 ms a layer re-reading the ring 56 times).
// Rows whose window starts later mask the chunk's early context keys. Partials per (query, chunk):
// running max m, sum l, unnormalized acc[hd]; nd_attn_combine merges the chunks in fixed order.
// part layout (Q = n*nh queries, qid = i*nh + h, ns = gridDim.y): acc [Q][ns][hd], m [Q][ns], l [Q][ns].
// geo = nh | kv << 8 | hd << 16. grid (kv, ns), block 256, dynamic smem 2*ND_CH*(hd+8) halves +
// (n*gqa)*ND_CH floats.
#define ND_CH 32
extern "C" __global__ void __launch_bounds__(256)
nd_attn_part(float* __restrict__ part, const __half* __restrict__ q16, const __half* __restrict__ kc, const __half* __restrict__ vc,
             const __half* __restrict__ kb16, const __half* __restrict__ vb16, int t, int n, int geo, int S, int W, float scale) {
    NV_PDL_ENTRY();
    extern __shared__ float sm[];
    const int nh = geo & 0xFF, kv = (geo >> 8) & 0xFF, hd = geo >> 16;
    const int gqa = nh / kv, g = blockIdx.x, ns = gridDim.y, sp = blockIdx.y;
    const int lo0 = max(0, t - W + 1);
    const int nctx0 = t - lo0;
    const int L = nctx0 + n;
    const int chunk = (L + ns - 1) / ns;
    const int j0 = sp * chunk, cn = max(0, min(chunk, L - j0));
    const int pitch = hd + 8;                      // padded smem rows: 16-B reads of different rows spread over banks
    __half* ks = (__half*)sm;
    __half* vs = ks + ND_CH * pitch;
    float* sc = (float*)(vs + ND_CH * pitch);      // [Qg][ND_CH]
    const int Qg = n * gqa;
    for (int e = threadIdx.x; e < cn * (hd / 8); e += blockDim.x) {
        const int kk = e / (hd / 8), c8 = (e % (hd / 8)) * 8;
        const int j = j0 + kk;
        const size_t off = j < nctx0 ? ((size_t)((lo0 + j) & (S - 1)) * kv + g) * hd : ((size_t)(j - nctx0) * kv + g) * hd;
        const __half* kr = j < nctx0 ? kc : kb16;
        const __half* vr = j < nctx0 ? vc : vb16;
        *(uint4*)(ks + kk * pitch + c8) = *(const uint4*)(kr + off + c8);
        *(uint4*)(vs + kk * pitch + c8) = *(const uint4*)(vr + off + c8);
    }
    __syncthreads();
    // scores (q from global/L1: Qg rows of hd halves)
    for (int e = threadIdx.x; e < Qg * ND_CH; e += blockDim.x) {
        const int qq = e / ND_CH, kk = e % ND_CH;
        const int i = qq / gqa, h = g * gqa + qq % gqa;
        float s = -INFINITY;
        const int j = j0 + kk;
        if (kk < cn && (j >= nctx0 || lo0 + j >= max(0, t + i - W + 1))) {
            const __half* qr = q16 + ((size_t)i * nh + h) * hd;
            float a = 0.f;
            for (int d = 0; d < hd; d += 8) {
                const uint4 qa = __ldg((const uint4*)(qr + d));
                const uint4 ka = *(const uint4*)(ks + kk * pitch + d);
                const __half2* q2 = (const __half2*)&qa;
                const __half2* k2 = (const __half2*)&ka;
                #pragma unroll
                for (int u = 0; u < 4; ++u) { const float2 x = __half22float2(q2[u]), y = __half22float2(k2[u]); a += x.x * y.x + x.y * y.y; }
            }
            s = a * scale;
        }
        sc[qq * ND_CH + kk] = s;
    }
    __syncthreads();
    const int Q = n * nh;
    float* pacc = part;
    float* pm = part + (size_t)Q * ns * hd;
    float* pl = pm + (size_t)Q * ns;
    for (int qq = threadIdx.x; qq < Qg; qq += blockDim.x) {
        float m = -INFINITY;
        for (int kk = 0; kk < ND_CH; ++kk) m = fmaxf(m, sc[qq * ND_CH + kk]);
        float l = 0.f;
        for (int kk = 0; kk < ND_CH; ++kk) {
            const float p = m == -INFINITY ? 0.f : __expf(sc[qq * ND_CH + kk] - m);   // masked -> 0
            sc[qq * ND_CH + kk] = p; l += p;
        }
        const int qid = (qq / gqa) * nh + g * gqa + qq % gqa;
        pm[(size_t)qid * ns + sp] = m;
        pl[(size_t)qid * ns + sp] = l;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < Qg * hd; e += blockDim.x) {
        const int qq = e / hd, d = e % hd;
        float a = 0.f;
        for (int kk = 0; kk < cn; ++kk) a += sc[qq * ND_CH + kk] * nv_h2f(vs[kk * pitch + d]);
        const int qid = (qq / gqa) * nh + g * gqa + qq % gqa;
        pacc[((size_t)qid * ns + sp) * hd + d] = a;
    }
}

// ---- merge the key chunks: out f16 [Q][hd] = sum_s acc_s e^(m_s - M) / sum_s l_s e^(m_s - M), chunks
// in fixed order, fully-masked chunks (m = -inf) skipped. grid (Q), block hd.
extern "C" __global__ void nd_attn_combine(__half* __restrict__ out, const float* __restrict__ part, int Q, int ns, int hd) {
    NV_PDL_ENTRY();
    const int qid = blockIdx.x, d = threadIdx.x;
    const float* pm = part + (size_t)Q * ns * hd;
    const float* pl = pm + (size_t)Q * ns;
    float M = -INFINITY;
    for (int s = 0; s < ns; ++s) M = fmaxf(M, pm[(size_t)qid * ns + s]);
    float l = 0.f, a = 0.f;
    for (int s = 0; s < ns; ++s) {
        const float m = pm[(size_t)qid * ns + s];
        if (m == -INFINITY) continue;
        const float w = __expf(m - M);
        l += pl[(size_t)qid * ns + s] * w;
        a += part[((size_t)qid * ns + s) * hd + d] * w;
    }
    out[(size_t)qid * hd + d] = nv_f2h(a / l);
}

// Sampled-draft scratch layout inside the persistent `smb` buffer (must match QROWS/QB in
// drafter_gpu.rs): QROWS full-vocabulary q rows, then the per-block softmax max / sum-exp partials.
#define ND_QMARK 16
#define ND_QB 1024
// ---- Markov stage, row i, scan: corrected[j] = logits16[i][j] + markov_w2[j] . markov_w1[prev]
// (prev = tok[i]); per block the (max, argmax) over its rows -> pv/pi[blockIdx.x]. Ties keep the
// LOWEST id (the reference's strict `>` fold). The rank r is split over gs = r/8 lanes (one 128-bit
// load each), 32/gs rows per warp step. grid (nb), block 256.
//
// SAMPLED DRAFT (T > 0 and qs != null; work item A of PLAN/NAIVE_DRAFT2_PROMPT_2026-10-19.md): the
// same corrected logit scaled by 1/T is written to qs[i*V + j] -- the drafter's FULL-vocabulary q
// row, which the verify side needs for the ratio test and the residual -- accumulated into a
// per-block softmax (max, sum-exp) -> mxs/qsum, and the pick becomes a Gumbel-max draw
// d_j ~ softmax(corrected/T). T <= 0 keeps the greedy body bit-for-bit (no qs write, no sums).
__device__ __forceinline__ void nd_argmax_merge(float& v, int& ix, float v2, int i2) {
    if (v2 > v || (v2 == v && i2 < ix)) { v = v2; ix = i2; }
}
// splitmix64 / Gumbel: the same streams gpu_naive.cu draws, duplicated because each .cu is its own
// module (a distinct constant keeps the two modules' draws independent).
__device__ __forceinline__ unsigned long long nd_splitmix(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
__device__ __forceinline__ float nd_gumbel(unsigned long long seed, int v, int row) {
    const unsigned long long r = nd_splitmix(seed ^ ((unsigned long long)v * 0xD1B54A32D192ED03ull)
                                                  ^ ((unsigned long long)(row + 1) << 48));
    const float u = ((float)(r >> 40) + 0.5f) * (1.f / 16777216.f);   // (0, 1)
    return -__logf(-__logf(u));
}
// online softmax merge: S holds the sum of exp(x - M) over the values merged so far
__device__ __forceinline__ void nd_merge_ms(float& M, float& S, float m2, float s2) {
    if (!(s2 > 0.f)) return;
    if (!(S > 0.f)) { M = m2; S = s2; return; }
    const float nm = fmaxf(M, m2);
    S = S * __expf(M - nm) + s2 * __expf(m2 - nm);
    M = nm;
}
extern "C" __global__ void __launch_bounds__(256)
nd_markov_scan(float* __restrict__ pv, int* __restrict__ pi, float* __restrict__ smb,
               const __half* __restrict__ logits16, const __half* __restrict__ w1,
               const __half* __restrict__ w2, const int* __restrict__ tok,
               int i, long long vv, int r, float T, unsigned long long rng) {
    NV_PDL_ENTRY();
    // vv = (v0 << 32) | V: this rank's vocabulary slice [v0, v0 + V) (the whole vocabulary at world 1).
    // logits16 / w2 / qrow are indexed by the LOCAL j; the argmax carries the GLOBAL id j + v0.
    const int V = (int)(vv & 0xffffffffll), v0 = (int)(vv >> 32);
    __shared__ float e[256];
    __shared__ float rv[8];
    __shared__ int ri[8];
    __shared__ float smx[8], ssu[8];
    const int prev = tok[i];
    for (int k = threadIdx.x; k < r; k += blockDim.x) e[k] = nv_h2f(w1[(size_t)prev * r + k]);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gs = r >> 3, rpw = 32 / gs;          // lanes per row, rows per warp step
    const int sub = lane / gs, gl = lane % gs;
    float ev[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) ev[k] = e[gl * 8 + k];
    const __half* lrow = logits16 + (size_t)i * V;
    const bool smp = (T > 0.f) && (smb != nullptr);
    const float inv = smp ? (1.f / T) : 0.f;
    float* qrow = smp ? smb + (size_t)i * V : nullptr;
    float* mxs = smp ? smb + (size_t)ND_QMARK * V : nullptr;
    float* qsum = smp ? mxs + ND_QB : nullptr;
    float best = -INFINITY; int bix = 0x7fffffff;
    float M = -INFINITY, S = 0.f;
    const int step = gridDim.x * 8 * rpw;
    for (int j0 = (blockIdx.x * 8 + warp) * rpw; j0 < V; j0 += step) {
        const int j = j0 + sub;
        float s = 0.f;
        if (j < V) {
            const uint4 u = nd_ld_stream(w2 + (size_t)j * r + gl * 8);
            const __half2* h2 = (const __half2*)&u;
            #pragma unroll
            for (int k = 0; k < 4; ++k) { const float2 f = __half22float2(h2[k]); s += f.x * ev[2 * k] + f.y * ev[2 * k + 1]; }
        }
        for (int o = gs >> 1; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (j < V && gl == 0) {
            const float base = nv_h2f(lrow[j]) + s;
            if (smp) {
                const float x = base * inv;
                qrow[j] = x;
                if (x > M) { S = S * __expf(M - x) + 1.f; M = x; }
                else S += __expf(x - M);
                nd_argmax_merge(best, bix, x + nd_gumbel(rng, j + v0, i), j + v0);
            } else {
                nd_argmax_merge(best, bix, base, j + v0);
            }
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, best, o);
        const int i2 = __shfl_xor_sync(0xffffffffu, bix, o);
        nd_argmax_merge(best, bix, v2, i2);
        if (smp) nd_merge_ms(M, S, __shfl_xor_sync(0xffffffffu, M, o), __shfl_xor_sync(0xffffffffu, S, o));
    }
    if (lane == 0) { rv[warp] = best; ri[warp] = bix; if (smp) { smx[warp] = M; ssu[warp] = S; } }
    __syncthreads();
    if (threadIdx.x == 0) {
        float b = rv[0]; int bi = ri[0];
        for (int w = 1; w < 8; ++w) nd_argmax_merge(b, bi, rv[w], ri[w]);
        pv[blockIdx.x] = b; pi[blockIdx.x] = bi;
        if (smp) {
            float M2 = smx[0], S2 = ssu[0];
            for (int w = 1; w < 8; ++w) nd_merge_ms(M2, S2, smx[w], ssu[w]);
            mxs[blockIdx.x] = M2; qsum[blockIdx.x] = S2;
        }
    }
}

// ---- Markov stage, row i, pick: tok[i+1] = the argmax over the scan partials; confidence[i] =
// sigmoid(b + w[:H] . hidden32[i] + w[H:] . markov_w1[prev]) with prev = tok[i] (the reference
// scores position i with the embedding it was conditioned on); cw == null: no confidence head.
// grid (1), block 256.
extern "C" __global__ void __launch_bounds__(256)
nd_markov_pick(int* __restrict__ tok, float* __restrict__ conf, const float* __restrict__ pv, const int* __restrict__ pi, int nb,
               const float* __restrict__ hidden32, const __half* __restrict__ w1, const float* __restrict__ cw, float cb,
               int i, int H, int r) {
    NV_PDL_ENTRY();
    __shared__ float rv[8];
    __shared__ int ri[8];
    __shared__ float red[32];
    float best = -INFINITY; int bix = 0x7fffffff;
    for (int k = threadIdx.x; k < nb; k += blockDim.x) nd_argmax_merge(best, bix, pv[k], pi[k]);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, best, o);
        const int i2 = __shfl_xor_sync(0xffffffffu, bix, o);
        nd_argmax_merge(best, bix, v2, i2);
    }
    if (lane == 0) { rv[warp] = best; ri[warp] = bix; }
    __syncthreads();
    const int prev = tok[i];
    if (cw) {
        float s = 0.f;
        for (int k = threadIdx.x; k < H; k += blockDim.x) s += cw[k] * hidden32[(size_t)i * H + k];
        for (int k = threadIdx.x; k < r; k += blockDim.x) s += cw[H + k] * nv_h2f(w1[(size_t)prev * r + k]);
        s = nd_block_sum(s, red) + cb;
        if (threadIdx.x == 0) conf[i] = 1.f / (1.f + __expf(-s));
    }
    if (threadIdx.x == 0) {
        float b = rv[0]; int bi = ri[0];
        for (int w = 1; w < (int)(blockDim.x >> 5); ++w) nd_argmax_merge(b, bi, rv[w], ri[w]);
        tok[i + 1] = bi;
    }
}

// Sampled draft, the normalisation fold for row i: the scan and pick above leave per-block softmax
// partials (max, sum-exp) at smb[ND_QMARK*V ..], which this kernel folds once (cheap: a few hundred
// iterations on one thread) into qstat[i] = (global max, Z), i.e. q_i(v) = exp(smb[i*V + v] - max)/Z.
// A separate kernel rather than extra args on nd_markov_pick: cudarc only launches 12-tuples.
// ---- work item B (sharded drafter) helpers ----
// scatter this rank's column block of a row-major f32 matrix into the exchange buffer: dst [m][H] gets
// src [m][hl] at columns [c0, c0 + hl) and ZEROS elsewhere, so an fp32 sum all-reduce is an all-gather.
// grid (m), block 256.
extern "C" __global__ void nd_scatter_cols(float* __restrict__ dst, const float* __restrict__ src, int m, int H, int hl, int c0) {
    NV_PDL_ENTRY();
    const int r = blockIdx.x;
    for (int i = threadIdx.x; i < H; i += blockDim.x) {
        const int k = i - c0;
        dst[(size_t)r * H + i] = (k >= 0 && k < hl) ? src[(size_t)r * hl + k] : 0.f;
    }
}
// fold this rank's scan partials (pv/pi [nb]) into its slot of the exchange buffer xch [world][2] =
// (best, (float)global id), other slots zero (the all-reduce then gathers every rank's best). grid (1), block 256.
extern "C" __global__ void __launch_bounds__(256)
nd_markov_fold(const float* __restrict__ pv, const int* __restrict__ pi, int nb, float* __restrict__ xch, int rank, int world) {
    NV_PDL_ENTRY();
    __shared__ float rv[8];
    __shared__ int ri[8];
    float best = -INFINITY; int bix = 0x7fffffff;
    for (int k = threadIdx.x; k < nb; k += blockDim.x) nd_argmax_merge(best, bix, pv[k], pi[k]);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float v2 = __shfl_xor_sync(0xffffffffu, best, o);
        const int i2 = __shfl_xor_sync(0xffffffffu, bix, o);
        nd_argmax_merge(best, bix, v2, i2);
    }
    if (lane == 0) { rv[warp] = best; ri[warp] = bix; }
    __syncthreads();
    if (threadIdx.x < 2 * world) xch[threadIdx.x] = 0.f;
    __syncthreads();
    if (threadIdx.x == 0) {
        float b = rv[0]; int bi = ri[0];
        for (int w = 1; w < 8; ++w) nd_argmax_merge(b, bi, rv[w], ri[w]);
        xch[2 * rank] = b; xch[2 * rank + 1] = (float)bi;    // ids < 2^24: exact as a float
    }
}
// the gathered slots back into (pv, pi) [world] for nd_markov_pick (nb = world). grid (1), block 32.
extern "C" __global__ void nd_markov_unpack(const float* __restrict__ xch, int world, float* __restrict__ pv, int* __restrict__ pi) {
    NV_PDL_ENTRY();
    const int k = threadIdx.x;
    if (k < world) { pv[k] = xch[2 * k]; pi[k] = (int)xch[2 * k + 1]; }
}

extern "C" __global__ void __launch_bounds__(32)
nd_markov_qnorm(const float* __restrict__ smb, int nb, int i, int V, float* __restrict__ qstat) {
    NV_PDL_ENTRY();
    if (threadIdx.x != 0 || smb == nullptr) return;
    const float* mxs = smb + (size_t)ND_QMARK * V;
    const float* qsum = mxs + ND_QB;
    float M = -INFINITY;
    for (int k = 0; k < nb; ++k) M = fmaxf(M, mxs[k]);
    float S = 0.f;
    for (int k = 0; k < nb; ++k) if (qsum[k] > 0.f) S += qsum[k] * __expf(mxs[k] - M);
    qstat[2 * i] = M; qstat[2 * i + 1] = S;
}
