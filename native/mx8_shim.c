// EXP-MX8: MXFP8 (e4m3 values, one ue8m0 scale per 32 along K, both operands) dense GEMM via cuBLASLt.
// Flat C ABI for src/exl3_forward.rs (cudarc 0.9's cublasLt bindings predate the block-scale modes).
//   Y[m][n] (f16) = sum_k X[m][k] * W[n][k],  X/W e4m3 K-contiguous, scales in cuBLAS's 128x4 swizzle.
// Batch invariance: the algorithm is chosen ONCE per (k, n) at a canonical m (the --exl3-lt-canon pin, or
// 4095) and reused for every m (cublasLtMatmulAlgoCheck); each output row then depends only on its own
// input row, whatever m is. If the canonical algorithm does not apply at some m, that m is refused
// (negative return); mx8_prepare() finds such m before any traffic.
#include <cublasLt.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Per-m entries are kept in a per-(k, n) array indexed by m (no global table: the old fixed 4096-entry
// table filled up after ~512 distinct pass sizes and then every new m failed, which made the caller
// switch MXFP8 off for good mid-run). The caller splits passes into chunks of at most MX8_MAX_M rows,
// so the array is bounded; mx8_prepare() validates every m up front.
typedef struct { int k, n, m; cublasLtMatrixLayout_t lb, lc; cublasLtMatmulAlgo_t algo; int bad; } Mx8Knm;
typedef struct { int k, n; cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la; cublasLtMatmulAlgo_t algo; int ok;
                 Mx8Knm** bym; int cap; } Mx8Kn;
#define MX8_KN 128
#define MX8_MAX_M 4096
static Mx8Kn g_kn[MX8_KN]; static int g_nkn = 0;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

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
    if (m < 1 || m > MX8_MAX_M) return NULL;
    if (!p->bym) { p->bym = (Mx8Knm**)calloc(MX8_MAX_M + 1, sizeof(Mx8Knm*)); if (!p->bym) return NULL; p->cap = MX8_MAX_M + 1; }
    Mx8Knm* q = p->bym[m];
    if (q) return q->bad ? NULL : q;
    q = (Mx8Knm*)calloc(1, sizeof(Mx8Knm)); if (!q) return NULL;
    q->k = p->k; q->n = p->n; q->m = m; p->bym[m] = q;
    if (cublasLtMatrixLayoutCreate(&q->lb, CUDA_R_8F_E4M3, p->k, m, p->k) ||
        cublasLtMatrixLayoutCreate(&q->lc, CUDA_R_16F, p->n, m, p->n)) {
        q->bad = 1; fprintf(stderr, "[exp-mx8] ERROR layout create failed k=%d n=%d m=%d\n", p->k, p->n, m); return NULL;
    }
    // strict: only the canonical algorithm (row invariance in m). No own-heuristic fallback any more.
    cublasLtMatmulHeuristicResult_t chk;
    if (cublasLtMatmulAlgoCheck(h, p->op, p->la, q->lb, q->lc, q->lc, &p->algo, &chk) == CUBLAS_STATUS_SUCCESS
        && chk.workspaceSize <= wsz) {
        q->algo = p->algo;
        return q;
    }
    q->bad = 1;
    fprintf(stderr, "[exp-mx8] ERROR canonical algorithm not valid at k=%d n=%d m=%d\n", p->k, p->n, m);
    return NULL;
}

// Validate (and cache) every m in [1, mmax] for (k, n) once, before traffic: returns the number of m that
// cannot run the canonical algorithm (0 = MXFP8 is safe for this shape at every chunk size).
int mx8_prepare(void* handle, int k, int n, int mmax, int m_canon, size_t wsz) {
    cublasLtHandle_t h = (cublasLtHandle_t)handle;
    if (mmax > MX8_MAX_M) mmax = MX8_MAX_M;
    pthread_mutex_lock(&g_mu);
    Mx8Kn* p = kn_get(h, k, n, m_canon > 0 ? m_canon : 4095, wsz);
    int bad = 0;
    if (!p) bad = mmax; else for (int m = 1; m <= mmax; ++m) if (!knm_get(h, p, m, wsz)) ++bad;
    pthread_mutex_unlock(&g_mu);
    return bad;
}
int mx8_max_m(void) { return MX8_MAX_M; }

// 0 = ok, negative = not run (caller falls back to f16 for this call only and logs it)
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
    pthread_mutex_unlock(&g_mu);
    if (st) { fprintf(stderr, "[exp-mx8] ERROR cublasLtMatmul status %d (k=%d n=%d m=%d)\n", (int)st, k, n, m); return -2; }
    return 0;
}
