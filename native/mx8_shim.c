// EXP-MX8: MXFP8 (e4m3 values, one ue8m0 scale per 32 along K, both operands) dense GEMM via cuBLASLt.
// Flat C ABI for src/exl3_forward.rs (cudarc 0.9's cublasLt bindings predate the block-scale modes).
//   Y[m][n] (f16) = sum_k X[m][k] * W[n][k],  X/W e4m3 K-contiguous, scales in cuBLAS's 128x4 swizzle.
// Batch invariance: the algorithm is chosen ONCE per (k, n) at a canonical m (the --exl3-lt-canon pin, or
// 4095) and reused for every m (cublasLtMatmulAlgoCheck); each output row then depends only on its own
// input row, whatever m is. If the canonical algorithm does not apply at some m, that m falls back to
// its own heuristic and the shim reports it once (returns 2 instead of 0).
#include <cublasLt.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int k, n; cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la; cublasLtMatmulAlgo_t algo; int ok; } Mx8Kn;
typedef struct { int k, n, m; cublasLtMatrixLayout_t lb, lc; cublasLtMatmulAlgo_t algo; int fallback; } Mx8Knm;
#define MX8_KN 128
#define MX8_KNM 4096
static Mx8Kn g_kn[MX8_KN]; static int g_nkn = 0;
static Mx8Knm g_knm[MX8_KNM]; static int g_nknm = 0;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_warned = 0;

static Mx8Kn* kn_get(cublasLtHandle_t h, int k, int n, int mc, size_t wsz) {
    for (int i = 0; i < g_nkn; ++i) if (g_kn[i].k == k && g_kn[i].n == n) return &g_kn[i];
    if (g_nkn >= MX8_KN) return NULL;
    Mx8Kn* p = &g_kn[g_nkn]; memset(p, 0, sizeof(*p)); p->k = k; p->n = n;
    if (cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F)) return NULL;
    cublasOperation_t T = CUBLAS_OP_T, N = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T));
    cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &N, sizeof(N));
    int md = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &md, sizeof(md));
    cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &md, sizeof(md));
    // the heuristic rejects block-scale modes without scale pointers (status 7): placeholders, replaced per call
    { const void* ph = (const void*)(size_t)256;
      cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ph, sizeof(ph));
      cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &ph, sizeof(ph)); }
    if (cublasLtMatrixLayoutCreate(&p->la, CUDA_R_8F_E4M3, k, n, k)) return NULL;
    // canonical-m algorithm
    cublasLtMatrixLayout_t lb, lc;
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, k, mc, k);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F, n, mc, n);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz));
    cublasLtMatmulHeuristicResult_t res; int nr = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(h, p->op, p->la, lb, lc, lc, pref, 1, &res, &nr);
    cublasLtMatmulPreferenceDestroy(pref); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
    if (st || nr == 0) { fprintf(stderr, "[exp-mx8] no MXFP8 algorithm for k=%d n=%d (status %d, ws %zu)\n", k, n, (int)st, wsz); return NULL; }
    p->algo = res.algo; p->ok = 1; g_nkn++;
    {   // report the config; MX8_FORCE_SPLITK1=1 forces split-K 1 / no reduction (batch invariance)
        int id = 0, tile = 0, sk = 0, red = 0, sw = 0, co = 0; size_t w;
        cublasLtMatmulAlgoConfigGetAttribute(&p->algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id), &w);
        cublasLtMatmulAlgoConfigGetAttribute(&p->algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), &w);
        cublasLtMatmulAlgoConfigGetAttribute(&p->algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, sizeof(sk), &w);
        cublasLtMatmulAlgoConfigGetAttribute(&p->algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &red, sizeof(red), &w);
        cublasLtMatmulAlgoConfigGetAttribute(&p->algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &sw, sizeof(sw), &w);
        cublasLtMatmulAlgoConfigGetAttribute(&p->algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &co, sizeof(co), &w);
        // split-K 1 / no reduction by default: the heuristic's "auto" split-K (-2) re-splits K by problem
        // size, which breaks row invariance in m AND collapses to ~57 TFLOPS at k 2560 n 10240 m 8800
        // (2.9 ms forced vs 8.1 ms auto). MX8_AUTO_SPLITK=1 keeps the heuristic's choice.
        const char* f = getenv("MX8_AUTO_SPLITK");
        if (!(f && f[0] == '1')) {
            int one = 1, none = 0;
            cublasLtMatmulAlgoConfigSetAttribute(&p->algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &one, sizeof(one));
            cublasLtMatmulAlgoConfigSetAttribute(&p->algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &none, sizeof(none));
        }
        if (getenv("MX8_VERBOSE")) fprintf(stderr, "[exp-mx8] k=%d n=%d algo id %d tile %d splitK %d red %d swz %d custom %d\n", k, n, id, tile, sk, red, sw, co);
    }
    return p;
}

static Mx8Knm* knm_get(cublasLtHandle_t h, Mx8Kn* p, int m, size_t wsz) {
    for (int i = 0; i < g_nknm; ++i) if (g_knm[i].k == p->k && g_knm[i].n == p->n && g_knm[i].m == m) return &g_knm[i];
    if (g_nknm >= MX8_KNM) return NULL;
    Mx8Knm* q = &g_knm[g_nknm]; memset(q, 0, sizeof(*q)); q->k = p->k; q->n = p->n; q->m = m;
    if (cublasLtMatrixLayoutCreate(&q->lb, CUDA_R_8F_E4M3, p->k, m, p->k)) return NULL;
    if (cublasLtMatrixLayoutCreate(&q->lc, CUDA_R_16F, p->n, m, p->n)) return NULL;
    cublasLtMatmulHeuristicResult_t chk;
    if (cublasLtMatmulAlgoCheck(h, p->op, p->la, q->lb, q->lc, q->lc, &p->algo, &chk) == CUBLAS_STATUS_SUCCESS
        && chk.workspaceSize <= wsz) {
        q->algo = p->algo;
    } else {
        cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz));
        cublasLtMatmulHeuristicResult_t res; int nr = 0;
        cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(h, p->op, p->la, q->lb, q->lc, q->lc, pref, 1, &res, &nr);
        cublasLtMatmulPreferenceDestroy(pref);
        if (st || nr == 0) return NULL;
        q->algo = res.algo; q->fallback = 1;
        if (!g_warned) { g_warned = 1; fprintf(stderr, "[exp-mx8] canonical algorithm not valid at k=%d n=%d m=%d: own heuristic (batch invariance lost there)\n", p->k, p->n, m); }
    }
    g_nknm++;
    return q;
}

// 0 = ok, 2 = ok on a fallback algorithm, negative = not run (caller falls back to f16)
int mx8_gemm(void* handle, int k, int n, int m, int m_canon,
             const void* qw, const void* sw, const void* qx, const void* sx, void* y,
             void* ws, size_t wsz, void* stream) {
    cublasLtHandle_t h = (cublasLtHandle_t)handle;
    pthread_mutex_lock(&g_mu);
    Mx8Kn* p = kn_get(h, k, n, m_canon > 0 ? m_canon : 4095, wsz);
    Mx8Knm* q = p ? knm_get(h, p, m, wsz) : NULL;
    if (!q) { pthread_mutex_unlock(&g_mu); return -1; }
    cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sw, sizeof(sw));
    cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sx, sizeof(sx));
    float one = 1.f, zero = 0.f;
    cublasStatus_t st = cublasLtMatmul(h, p->op, &one, qw, p->la, qx, q->lb, &zero, y, q->lc, y, q->lc,
                                       &q->algo, ws, wsz, (cudaStream_t)stream);
    int fb = q->fallback;
    pthread_mutex_unlock(&g_mu);
    if (st) { fprintf(stderr, "[exp-mx8] cublasLtMatmul status %d (k=%d n=%d m=%d)\n", (int)st, k, n, m); return -2; }
    return fb ? 2 : 0;
}
