//! `naive` — Naive-N0.5-Flash (309B-A15.5B MoE, hybrid SWA / DeepSeek-sparse attention, EXL3
//! 3.5 bpw) as its own model family: TP2 / TP4 over the engine's RDMA all-reduce, single stream,
//! 1M context, DSpark speculation. Plan, stage ledger and back-port notes: `PLAN/NAIVE_N05_PORT.md`.
//!
//! Discipline (owner, 2026-10-16): Flash-Next code is off-limits — this tree COPIES what it needs
//! (kernels, loader primitives, round drivers) and dispatches to itself from one line in the
//! shared entry points. Shared library modules (`tp_xport`, `net`, `cluster`, `server`, `tokenizer`,
//! `tools`, the `exl3` decode primitives) are used read-only.
//!
//! Modules land stage by stage (the plan's S0..S10):
//!   * [`frac`]   S0 — half-integer-K (3.5 bpw) trellis decode, CPU oracle.
//!   * [`config`] S1 — the architecture as the engine needs it, parsed from the pack's config.json.
//!   * [`pack`]   S1 — the EXL3 pack census and loader (quads with `k_words`, plain tensors, the
//!                full inventory check).
//!   * [`cpu`]    S2 — the f32 CPU reference forward (op for op after the HF modeling file), the
//!                oracle for every kernel; checked against the HF reference on a tiny replica.
//!   * [`dequant`] S2b — host dequantization of real pack quads (suh / svh / Hadamard chain) with
//!                the padded-region falsifier.
//!   * [`gpu`]    S3 — the `gpu_naive` kernel module (kernels/gpu_naive.cu): loader, build-id
//!                handshake, and the GPU-vs-CPU gates on real tensors (`--probe-naive-gemm`).
//!   * [`model`]  S4 — the GPU slice model (first N layers, M <= 16 rows) and its CPU-twin gate
//!                (`--probe-naive-slice`).
//!   * [`drafter_cpu`] S8 — the DSpark drafter round on the CPU (f32), gated against the HF modules on
//!                a tiny instance (scripts/naive/drafter_oracle.py).
//!   * [`drafter_gpu`] S8 — the same round on the GPU (kernels/gpu_naive_draft.cu): context K/V ring,
//!                block backbone, on-device Markov + confidence; gate `--probe-naive-drafter`.

pub mod frac;
pub mod config;
pub mod pack;
pub mod cpu;
pub mod dequant;
pub mod gpu;
pub mod model;
pub mod drafter_cpu;
pub mod drafter_gpu;
pub mod shard;
pub mod tp;
pub mod serve;
pub mod decode;
pub mod shipplan;
