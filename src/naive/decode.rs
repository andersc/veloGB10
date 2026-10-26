//! S6: the graph-captured decode step + GPU sampling (PLAN/NAIVE_N05_PORT.md §5.3 S6a).
//!
//! `DecodeStep` owns a workspace and fixed device buffers (ids, positions, sampler parameters,
//! penalty tables, RNG state, the token) and records ONE step of `m` rows — embed, 48 layers (with
//! the TP doorbell reduces), final norm + lm_head, penalties, top-candidate select, sample — into
//! a CUDA graph on the model's stream. A replay is: sync, write ids/pos, launch, (head) read the
//! token. Everything data-dependent (positions, routing, the sampled token) lives on the device, so
//! the graph is shape-stable; the sampler's knobs ride a device parameter row written per request.
use anyhow::{ensure, Context, Result};
use cudarc::driver::sys;
use cudarc::driver::{CudaSlice, DevicePtr, LaunchConfig};

use super::model::{dptr, ws_bytes, GpuModel, Ws, MAX_ROWS};

/// Candidates of the exact fallback (gathered over the ranks): exact for top_k <= CAND; a top_p nucleus
/// wider than CAND tokens is truncated to them there (the Gumbel draws before it have no such limit).
pub const CAND: usize = 1024;
/// exchange buffer floats (kernels/gpu_naive.cu XB_* layout)
const XB_FLOATS: usize = 128 + 2 * CAND;
const XB_STATS: usize = 0;
fn xb_try(t: usize) -> usize { 16 + 16 * t }
fn xb_mass(t: usize) -> usize { 64 + 16 * t }
const XB_CAND: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub rep_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

impl SampleParams {
    pub const GREEDY: SampleParams = SampleParams { temperature: 0.0, top_p: 1.0, top_k: 0, min_p: 0.0, rep_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0 };
    fn row(&self) -> [f32; 8] { [self.temperature, self.top_p, self.top_k as f32, self.min_p, self.rep_penalty, self.presence_penalty, self.frequency_penalty, 0.0] }
}

/// Row j's addresses in the batched sampler's per-row scratch (work item C).
struct Rv { lg: u64, xb: u64, scaled: u64, bval: u64, bidx: u64, sel: u64, nsel: u64, lmax: u64, acc: u64, rng: u64, tok: u64 }

/// G1 (`--naive-lanes N`, 2026-10-20): the sampler state of N concurrent requests, SHARED by every step of the
/// engine (the m = 1 decode step, the verify blocks and the multi-lane steps), so a request keeps one state whichever
/// step serves it. Same lane layout as a step's own tables (knobs 8 f32, `counts` / `seen` vocab-wide, RNG pair).
/// Every buffer is explicitly zeroed (AGENTS §2.2: `alloc_zeros` does not zero).
pub struct LaneTab {
    pub lanes: usize,
    pub v: usize,
    pub sp: CudaSlice<f32>,
    pub counts: CudaSlice<i32>,
    pub seen: CudaSlice<u8>,
    pub rng: CudaSlice<u64>,
}

impl LaneTab {
    pub fn new(model: &GpuModel, lanes: usize) -> Result<LaneTab> {
        let dev = &model.dev;
        let v = model.cfg.vocab_size;
        let t = LaneTab { lanes, v, sp: dev.htod_sync_copy(&vec![0f32; 8 * lanes])?, counts: dev.htod_sync_copy(&vec![0i32; v * lanes])?,
                          seen: dev.htod_sync_copy(&vec![0u8; v * lanes])?, rng: dev.htod_sync_copy(&vec![0u64; 2 * lanes])? };
        dev.synchronize()?;
        Ok(t)
    }
    fn ptrs(&self) -> (u64, u64, u64, u64, usize) { (dptr(&self.sp), dptr(&self.counts), dptr(&self.seen), dptr(&self.rng), self.lanes) }
    /// Compaction: lane `from`'s whole state into lane `to` (device copies on the model's stream, AGENTS §2.3).
    pub fn copy_lane(&self, model: &GpuModel, from: usize, to: usize) -> Result<()> {
        if from == to { return Ok(()); }
        let s = model.stream.stream;
        let v = self.v as u64;
        let cp = |base: u64, stride: u64| -> Result<()> {
            unsafe { cudarc::driver::result::memcpy_dtod_async(base + stride * to as u64, base + stride * from as u64, stride as usize, s) }
                .map_err(|e| anyhow::anyhow!("lane copy: {e:?}"))
        };
        cp(dptr(&self.sp), 32)?;
        cp(dptr(&self.rng), 16)?;
        cp(dptr(&self.counts), 4 * v)?;
        cp(dptr(&self.seen), v)?;
        Ok(())
    }
}

pub struct DecodeStep {
    pub m: usize,
    pub head: bool,
    /// the FULL vocabulary width (`cfg.vocab_size`), used as the per-lane stride of
    /// `counts`/`seen`.
    ///
    /// This is deliberately NOT `model.vl`. At TP>1 the lm_head is vocabulary-parallel, so `vl`
    /// is this rank's SLICE of the head (76288 of 152576 at TP=2) -- but the penalty tables are
    /// indexed by GLOBAL token id: `nv_smp_prep` reads `counts[v0 + v]` with `v0` the slice
    /// origin (gpu_naive.cu:1916-1917), and `begin_request_lane` marks `seen[l*v + t]` with `t`
    /// straight from the prompt tokenizer. Sizing the tables by `vl` therefore leaves every rank
    /// room for only half the vocabulary, and admitting a request panics at TP>1 with "range end
    /// index 152576 out of range for slice of length 76288". At world 1 `vl == vocab_size`,
    /// which is why every single-GPU probe passed.
    v: usize,
    ws: Ws,
    ids_d: CudaSlice<i32>,
    pos_d: CudaSlice<i32>,
    /// G1: the per-request sampler state, LANE-STRIDED. `m` concurrent requests each own their knobs,
    /// penalty tables and RNG, so a step samples `m` INDEPENDENT requests rather than `m` rows of one
    /// speculative block. Lane `l` lives at `sp[l*8..l*8+8]`, `counts[l*v..(l+1)*v]`, `seen[l*v..(l+1)*v]`,
    /// `rng[l*2..l*2+2]`.
    ///
    /// The kernels read these through a POINTER plus an offset (`sp[0]`, `counts[v0+v]`, `rng[0]` --
    /// gpu_naive.cu:1876/1881/1955), so selecting a lane needs only the base address, not a signature
    /// change. `nv_gumbel` consumes `rng[0]`/`rng[1]` as one stream, but they are two u64 loads from
    /// the passed base, so a lane base still selects that lane's stream (verified, G1-A3: NO .cu edit).
    /// G1: which lane's sampler state the NEXT `step`/`step_block` samples. The serving loop sets it to
    /// the lane it just admitted (`begin_request_lane`); the single-stream paths leave it 0, so their
    /// behaviour — and the m=1 allocation — is byte-for-byte what it was before lanes existed.
    pub cur_lane: usize,
    sp: CudaSlice<f32>,
    counts: CudaSlice<i32>,
    seen: CudaSlice<u8>,
    posv: CudaSlice<i32>,
    stats: CudaSlice<f32>,
    scaled: CudaSlice<f32>,
    sel: CudaSlice<i32>,
    nsel: CudaSlice<i32>,
    rng: CudaSlice<u64>,
    tok: CudaSlice<i32>,
    /// fast-sampler scratch: per-block values / indices (reused as mass / count), the local max, the accept flag
    bval: CudaSlice<f32>,
    bidx: CudaSlice<i32>,
    lmax: CudaSlice<f32>,
    acc: CudaSlice<i32>,
    /// the cross-rank exchange buffer (zero-padded slots, all-reduced), the head's cancel flag in, the flags out
    xb: CudaSlice<f32>,
    cancel: CudaSlice<i32>,
    flags: CudaSlice<i32>,
    /// Work item C (batched sampler exchange): with `batched` set and m > 1 the block's rows are sampled
    /// PHASE-MAJOR off per-row scratch (one copy of the sampler's buffers per row), so each sampler phase
    /// makes ONE exchange carrying every row's slots -- 8 per block instead of 8 per row. `rxb` is
    /// [m][XB_FLOATS], `xs` the contiguous staging area of one phase ([m][2*CAND] at most), `rrng` the
    /// per-row (seed, step + row) pairs written by `nv_rng_rows`. Off (the default at world 1, where there
    /// is nothing to exchange) the per-row loop in `kernels` runs unchanged.
    batched: bool,
    rxb: CudaSlice<f32>, xs: CudaSlice<f32>, rscaled: CudaSlice<f32>, rbval: CudaSlice<f32>, rbidx: CudaSlice<i32>,
    rsel: CudaSlice<i32>, rnsel: CudaSlice<i32>, rlmax: CudaSlice<f32>, racc: CudaSlice<i32>, rrng: CudaSlice<u64>,
    /// Exact speculative acceptance (work item A, SAMPLED requests only): `smp_on[0]` gates the new
    /// kernel (0 = greedy, or a node rank = a no-op launch), `smp_draft` = this round's draft ids,
    /// `smp_rng` = [seed, step] for the accept draws, `smp_acc`/`smp_rtok` = the head's per-row
    /// accept flags and residual draws, and `smp_q`/`smp_qstat`/`smp_stride` address the head's
    /// full-vocabulary drafter logits (q). The q address is baked into the captured graph, so it is
    /// assigned once before the first capture; the buffer lives as long as the engine.
    smp_on: CudaSlice<i32>,
    /// `smp_gather[0]` = 1 while a sampled block runs, on EVERY rank (the head and the nodes alike):
    /// forces the per-row top-CANDL gather that the fast sampler otherwise skips after an accepted
    /// Gumbel try, so the exchanged candidate list the accept kernel reads is this row's on all ranks
    smp_gather: CudaSlice<i32>,
    smp_draft: CudaSlice<i32>,
    smp_rng: CudaSlice<u64>,
    pub smp_acc: CudaSlice<i32>,
    pub smp_rtok: CudaSlice<i32>,
    pub smp_q: u64,
    pub smp_qstat: u64,
    pub smp_stride: usize,
    /// the value currently in `smp_on`: written only when it changes, so a greedy round issues no
    /// host->device copy at all (cudarc copies ride the NULL stream — AGENTS.md §2.1)
    smp_on_cur: bool,
    smp_gather_cur: bool,
    /// S8: one sampled token per row (the verify block), and the drafter's context taps of the block:
    /// f32 [tap_layers.len()][m][h], copied inside the graph (`tap_layers` empty = none)
    pub toks: CudaSlice<i32>,
    pub tap_layers: Vec<usize>,
    pub tap_buf: CudaSlice<f32>,
    exec: Option<sys::CUgraphExec>,
    /// the logits address inside `ws` (address-stable across replays)
    logits: u64,
    /// G1: the shared lane table (sp, counts, seen, rng, lanes) this step samples from instead of its own (None = its own)
    ext: Option<(u64, u64, u64, u64, usize)>,
    /// G1: an m-row step of m DIFFERENT requests: row j samples lane j and counts its token (else the m rows are one
    /// request's verify block)
    multilane: bool,
}

// The graph handle is only ever used from the thread that owns the step (the serve loop).
unsafe impl Send for DecodeStep {}

impl DecodeStep {
    /// G1: the device pointer to lane `l`'s sampler knobs (8 f32). Clamped to the last lane so a
    /// partially-filled step can never read past the allocation.
    pub fn sp_lane(&self, l: usize) -> u64 {
        match self.ext { Some((sp, _, _, _, n)) => sp + 32 * l.min(n - 1) as u64, None => dptr(&self.sp) + 4 * 8 * l.min(self.m - 1) as u64 }
    }
    /// G1: lane `l`'s repeat/penalty tables -- `counts` (i32) and `seen` (u8), each `v` wide.
    pub fn counts_lane(&self, l: usize) -> u64 {
        match self.ext { Some((_, c, _, _, n)) => c + (4 * self.v * l.min(n - 1)) as u64, None => dptr(&self.counts) + (4 * self.v * l.min(self.m - 1)) as u64 }
    }
    pub fn seen_lane(&self, l: usize) -> u64 {
        match self.ext { Some((_, _, se, _, n)) => se + (self.v * l.min(n - 1)) as u64, None => dptr(&self.seen) + (self.v * l.min(self.m - 1)) as u64 }
    }
    /// G1: lane `l`'s RNG pair (2 u64).
    pub fn rng_lane(&self, l: usize) -> u64 {
        match self.ext { Some((_, _, _, r, n)) => r + 16 * l.min(n - 1) as u64, None => dptr(&self.rng) + 8 * 2 * l.min(self.m - 1) as u64 }
    }
    /// G1: sample from the shared lane table (call before `capture`: the graph bakes the lane addresses).
    pub fn use_lane_tab(&mut self, t: &LaneTab) { self.ext = Some(t.ptrs()); }
    /// G1: make this m-row step a multi-request step (row j = lane j). Before `capture`.
    pub fn set_multilane(&mut self, on: bool) { self.multilane = on && self.m > 1; }
}

impl Drop for DecodeStep {
    fn drop(&mut self) { if let Some(e) = self.exec.take() { unsafe { sys::cuGraphExecDestroy(e); } } }
}

impl DecodeStep {
    pub fn new(model: &GpuModel, m: usize, head: bool) -> Result<DecodeStep> {
        Self::new_with_taps(model, m, head, Vec::new())
    }

    /// A step whose graph also copies the residual after `tap_layers` (the drafter's context taps).
    pub fn new_with_taps(model: &GpuModel, m: usize, head: bool, tap_layers: Vec<usize>) -> Result<DecodeStep> {
        ensure!(m >= 1 && m <= MAX_ROWS, "decode rows {m}");
        let dev = &model.dev;
        let v = model.cfg.vocab_size;
        let vl = model.vl;
        let h = model.cfg.hidden_size;
        let ws = Ws::new(dev, ws_bytes(&model.cfg, m, model.max_pos, 0))?;
        let mut d = DecodeStep {
            m, head, v, cur_lane: 0,
            ws,
            ids_d: dev.alloc_zeros::<i32>(m)?, pos_d: dev.alloc_zeros::<i32>(m)?,
            // G1: LANE-STRIDED sampler state -- `m` requests, each with its own knobs, penalty
            // tables and RNG. `counts`/`seen` are FULL-vocab wide per lane (see the `v` field: the
            // kernels index them by GLOBAL token id, so a TP=2 rank needs all 152576 columns even
            // though its logits slice is 76288). The remaining sampler scratch (`posv`, `scaled`,
            // `bval`, `bidx`) is logits-side and stays at the local slice width `vl`.
            sp: dev.alloc_zeros::<f32>(8 * m)?, counts: dev.alloc_zeros::<i32>(v * m)?, seen: dev.alloc_zeros::<u8>(v * m)?,
            posv: dev.htod_sync_copy(&[(vl - 1) as i32])?, stats: dev.alloc_zeros::<f32>(2)?, scaled: dev.alloc_zeros::<f32>(vl)?,
            sel: dev.alloc_zeros::<i32>(CAND)?, nsel: dev.alloc_zeros::<i32>(1)?, rng: dev.alloc_zeros::<u64>(2 * m)?, tok: dev.alloc_zeros::<i32>(1)?,
            bval: dev.alloc_zeros::<f32>((vl + 1023) / 1024)?, bidx: dev.alloc_zeros::<i32>((vl + 1023) / 1024)?,
            lmax: dev.alloc_zeros::<f32>(1)?, acc: dev.alloc_zeros::<i32>(1)?,
            xb: dev.alloc_zeros::<f32>(XB_FLOATS)?, cancel: dev.alloc_zeros::<i32>(1)?, flags: dev.alloc_zeros::<i32>(1)?,
            batched: false,
            rxb: dev.alloc_zeros::<f32>(m * XB_FLOATS)?, xs: dev.alloc_zeros::<f32>(m * 2 * CAND)?, rscaled: dev.alloc_zeros::<f32>(m * vl)?,
            rbval: dev.alloc_zeros::<f32>(m * ((vl + 1023) / 1024))?, rbidx: dev.alloc_zeros::<i32>(m * ((vl + 1023) / 1024))?,
            rsel: dev.alloc_zeros::<i32>(m * CAND)?, rnsel: dev.alloc_zeros::<i32>(m)?, rlmax: dev.alloc_zeros::<f32>(m)?,
            racc: dev.alloc_zeros::<i32>(m)?, rrng: dev.alloc_zeros::<u64>(2 * m)?,
            smp_on: dev.alloc_zeros::<i32>(1)?, smp_gather: dev.alloc_zeros::<i32>(1)?, smp_draft: dev.alloc_zeros::<i32>(MAX_ROWS)?,
            smp_rng: dev.alloc_zeros::<u64>(3)?, smp_acc: dev.alloc_zeros::<i32>(MAX_ROWS)?,
            smp_rtok: dev.alloc_zeros::<i32>(MAX_ROWS)?, smp_q: 0, smp_qstat: 0, smp_stride: 0, smp_on_cur: false, smp_gather_cur: false,
            toks: dev.alloc_zeros::<i32>(m)?, tap_buf: dev.alloc_zeros::<f32>(tap_layers.len().max(1) * m * h)?, tap_layers,
            exec: None, logits: 0, ext: None, multilane: false,
        };
        // §2.2: `alloc_zeros` does NOT zero (async alloc). `smp_on` gates the accept kernel and
        // `smp_draft` feeds it, so both must be truthful before the first round: otherwise the device
        // gate can read garbage and the exact path runs on a GREEDY request — the accept kernel then
        // consumes an unwritten `smp_draft` and writes a bogus committed list into the TP ship (the
        // 2026-10-19 served-TP4 fatal `anchor 4286578688` = f32 -inf bits read as a token id).
        // The gate stays on the DEVICE (the kernel is captured into the graph; a host-side skip would
        // leave it out of a graph captured while speech was off).
        d.smp_on = dev.htod_sync_copy(&[0i32])?;
        d.smp_gather = dev.htod_sync_copy(&[0i32])?;
        d.smp_draft = dev.htod_sync_copy(&vec![0i32; MAX_ROWS])?;
        d.smp_acc = dev.htod_sync_copy(&vec![0i32; MAX_ROWS])?;
        d.smp_rtok = dev.htod_sync_copy(&vec![0i32; MAX_ROWS])?;
        dev.synchronize()?;
        Ok(d)
    }

    /// The step's kernels (eager or under capture): the forward, then the sampler on the LAST row's
    /// logits (every rank: the head is vocabulary-parallel and the sampler exchanges its slot values).
    fn kernels(&mut self, model: &GpuModel) -> Result<()> {
        let vl = model.vl;
        let sel = if self.tap_layers.is_empty() { None } else { Some((self.tap_layers.as_slice(), dptr(&self.tap_buf))) };
        let lg = model.step_kernels_sel(&mut self.ws, dptr(&self.ids_d), dptr(&self.pos_d), self.m, true, None, sel)?.unwrap();
        self.logits = lg;
        if self.m == 1 {
            // plain decode: the one row's token, counted into the penalty table inside the graph
            let lane = self.cur_lane;
            return self.sampler_kernels_lane(model, lg, dptr(&self.tok), true, lane);
        }
        // the verify block: every row sampled in SEQUENCE off the SAME lane's state (the rows are one
        // request's speculative block, so they share its knobs, its RNG — advanced once per row —
        // and its penalty tables), nothing counted — the committed tokens are counted after the
        // acceptance (nv_smp_count). The per-row scratch (sel/scaled/bval/posv/xb) is shared and safe
        // because each row's sampler completes before the next begins.
        let m = self.m;
        if self.multilane {
            // G1: m requests, one row each: row j samples LANE j and counts its own token (in the graph, like m = 1)
            for j in 0..m { self.sampler_kernels_lane(model, lg + (j * vl * 2) as u64, dptr(&self.toks) + (j * 4) as u64, true, j)?; }
            return Ok(());
        }
        let lane = self.cur_lane;
        if self.batched { return self.sampler_block_batched(model, lg, lane); }
        let candl = CAND / model.shard.world;
        for j in 0..m {
            self.sampler_kernels_lane(model, lg + (j * vl * 2) as u64, dptr(&self.toks) + (j * 4) as u64, false, lane)?;
            // Exact speculative acceptance for draft row j (work item A). INSIDE the loop on purpose:
            // `xb` and XB_STATS hold row j's gathered candidates only until the next row runs. The
            // kernel does no exchange and returns at once when `smp_on[0] == 0` (a greedy request, or a
            // node rank — q lives only on the head), so the greedy token stream is untouched.
            if j + 1 < m {
                model.l("nv_spec_accept_rows", LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 },
                        (&self.smp_on, &self.xb, model.shard.world as i32, (candl * model.shard.world) as i32,
                         self.sp_lane(lane), j as i32, &self.smp_draft,
                         self.smp_q + (j as u64 * self.smp_stride as u64 * 4),
                         self.smp_qstat, &self.smp_rng, &self.smp_acc, &self.smp_rtok))?;
            }
        }
        Ok(())
    }

    /// The sampler on one LOCAL logits row (f16 [vl] at `row`): penalties + scaling, (max, Z) per rank ->
    /// exchange, up to 3 Gumbel-max draws (local argmax -> exchange -> nucleus mass -> exchange -> the
    /// test, identical on every rank), then the gated exact fallback (local top-CANDL -> exchange ->
    /// sorted sampling). 8 slot exchanges per step at TP (none at world 1).
    fn sampler_kernels(&self, model: &GpuModel, row: u64, tok: u64, count: bool) -> Result<()> {
        self.sampler_kernels_lane(model, row, tok, count, 0)
    }

    /// G1: as `sampler_kernels`, but reading and advancing LANE `lane`'s sampler state (0 = the
    /// single-stream path, byte-for-byte unchanged). Every kernel reads its state through a POINTER,
    /// so a lane is just a base address -- no kernel signature changed.
    fn sampler_kernels_lane(&self, model: &GpuModel, row: u64, tok: u64, count: bool, lane: usize) -> Result<()> {
        let (vl, v0) = (model.vl as i32, model.v0 as i32);
        let sp = self.sp_lane(lane);
        let counts = self.counts_lane(lane);
        let seen = self.seen_lane(lane);
        let rng = self.rng_lane(lane);
        let w = model.shard.world;
        let r = model.shard.rank as i32;
        let nb = (model.vl + 1023) / 1024;
        let candl = CAND / w;
        let g1 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let gb = LaunchConfig { grid_dim: (nb as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let xchg = |off: usize, n: usize| -> Result<()> {
            if let Some(tp) = &model.tp { tp.all_reduce_at(&model.stream, &self.xb, off, n)?; }
            Ok(())
        };
        model.l("nv_smp_prep", gb, (&self.scaled, &self.bval, row, counts, seen, sp, vl, v0))?;
        model.l("nv_smp_stats", g1, (&self.bval, &self.lmax, nb as i32))?;
        model.l("nv_smp_z", gb, (&self.scaled, &self.lmax, &self.bval, vl))?;
        model.l("nv_smp_z2", g1, (&self.bval, &self.lmax, nb as i32, &self.xb, r, w as i32, &self.cancel, &self.acc))?;
        xchg(XB_STATS, 4 * w)?;
        for t in 0..3usize {
            model.l("nv_smp_gumbel", gb, (&self.scaled, sp, rng, t as i32, &self.bval, &self.bidx, vl, v0, &self.acc))?;
            model.l("nv_smp_pick", g1, (&self.bval, &self.bidx, nb as i32, &self.scaled, v0, &self.xb, t as i32, r, w as i32, &self.acc))?;
            xchg(xb_try(t), 4 * w)?;
            model.l("nv_smp_mass", gb, (&self.scaled, &self.xb, t as i32, w as i32, &self.bval, &self.bidx, vl, &self.acc))?;
            model.l("nv_smp_mass2", g1, (&self.bval, &self.bidx, nb as i32, &self.xb, t as i32, r, w as i32, &self.acc))?;
            xchg(xb_mass(t), 4 * w)?;
            model.l("nv_smp_accept", LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 }, (&self.xb, t as i32, w as i32, sp, tok, &self.acc))?;
        }
        // the exact fallback: the local top-CANDL (skipped when a draw was accepted), ALWAYS exchanged
        // (the doorbell epochs must match on every rank), sampled only when needed
        model.l("nv_topk", g1, (&self.sel, &self.nsel, &self.scaled, &self.posv, candl as i32, vl as i64, &self.acc, &self.smp_gather))?;
        model.l("nv_smp_cands_out", g1, (&self.sel, &self.nsel, &self.scaled, v0, &self.xb, candl as i32, r, w as i32))?;
        xchg(XB_CAND, 2 * candl * w)?;
        model.l("nv_smp_sample_cands", g1, (tok, &self.xb, w as i32, candl as i32, sp, rng, &self.acc))?;
        model.l("nv_smp_note", LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 }, (counts, tok, rng, &self.xb, &self.flags, count as i32))
    }

    /// Work item C: sample the block's rows with the batched exchange (`--spec-sampler`). Set before `capture`
    /// (the graph records whichever path is on) and identically on every rank (the exchange sequence must match).
    pub fn set_batched(&mut self, on: bool) { self.batched = on && self.m > 1; }
    pub fn batched(&self) -> bool { self.batched }

    /// Row j's view of the per-row scratch (device addresses; every sampler kernel takes pointers).
    fn rv(&self, model: &GpuModel, lg: u64, j: usize) -> Rv {
        let vl = model.vl;
        let nb = (vl + 1023) / 1024;
        let j64 = j as u64;
        Rv { lg: lg + (j * vl * 2) as u64, xb: dptr(&self.rxb) + j64 * (XB_FLOATS * 4) as u64,
             scaled: dptr(&self.rscaled) + j64 * (vl * 4) as u64, bval: dptr(&self.rbval) + j64 * (nb * 4) as u64,
             bidx: dptr(&self.rbidx) + j64 * (nb * 4) as u64, sel: dptr(&self.rsel) + j64 * (CAND * 4) as u64,
             nsel: dptr(&self.rnsel) + j64 * 4, lmax: dptr(&self.rlmax) + j64 * 4, acc: dptr(&self.racc) + j64 * 4,
             rng: dptr(&self.rrng) + j64 * 16, tok: dptr(&self.toks) + j64 * 4 }
    }

    /// Work item C: the verify block's rows sampled PHASE-MAJOR, one exchange per sampler phase for ALL rows
    /// (stats, 3 x (try, mass), candidates = 8 per block) instead of 8 per row. The rows are independent inside
    /// a block (the penalty tables are read-only until the round commits; row j's draws come from
    /// (seed, step + j), the pairs the per-row path consumed), so the kernels are the per-row kernels on per-row
    /// scratch, and the draws are bit-identical to the per-row path (gate: `--probe-naive-spec --spec-mode sampled`
    /// with `--spec-sampler rows` vs `batched`, same seed, world 2 and 4). The exact acceptance (item A) runs per row
    /// after the candidate exchange, exactly as before.
    fn sampler_block_batched(&self, model: &GpuModel, lg: u64, lane: usize) -> Result<()> {
        let m = self.m;
        let (vl, v0) = (model.vl as i32, model.v0 as i32);
        let sp = self.sp_lane(lane);
        let counts = self.counts_lane(lane);
        let seen = self.seen_lane(lane);
        let rng = self.rng_lane(lane);
        let w = model.shard.world;
        let r = model.shard.rank as i32;
        let nb = (model.vl + 1023) / 1024;
        let candl = CAND / w;
        let g1 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let gb = LaunchConfig { grid_dim: (nb as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        let g32 = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
        let gm = LaunchConfig { grid_dim: (m as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
        // ONE exchange of section [off, off + n) of every row: pack -> all-reduce [m][n] -> unpack
        let xch = |off: usize, n: usize| -> Result<()> {
            if let Some(tp) = &model.tp {
                model.l("nv_rows_pack", gm, (&self.xs, &self.rxb, m as i32, XB_FLOATS as i32, off as i32, n as i32, 0i32))?;
                tp.all_reduce(&model.stream, &self.xs, m * n)?;
                model.l("nv_rows_pack", gm, (&self.xs, &self.rxb, m as i32, XB_FLOATS as i32, off as i32, n as i32, 1i32))?;
            }
            Ok(())
        };
        model.l("nv_rng_rows", g32, (rng, &self.rrng, m as i32))?;
        for j in 0..m {
            let v = self.rv(model, lg, j);
            model.l("nv_smp_prep", gb, (v.scaled, v.bval, v.lg, counts, seen, sp, vl, v0))?;
            model.l("nv_smp_stats", g1, (v.bval, v.lmax, nb as i32))?;
            model.l("nv_smp_z", gb, (v.scaled, v.lmax, v.bval, vl))?;
            model.l("nv_smp_z2", g1, (v.bval, v.lmax, nb as i32, v.xb, r, w as i32, &self.cancel, v.acc))?;
        }
        xch(XB_STATS, 4 * w)?;
        for t in 0..3usize {
            for j in 0..m {
                let v = self.rv(model, lg, j);
                model.l("nv_smp_gumbel", gb, (v.scaled, sp, v.rng, t as i32, v.bval, v.bidx, vl, v0, v.acc))?;
                model.l("nv_smp_pick", g1, (v.bval, v.bidx, nb as i32, v.scaled, v0, v.xb, t as i32, r, w as i32, v.acc))?;
            }
            xch(xb_try(t), 4 * w)?;
            for j in 0..m {
                let v = self.rv(model, lg, j);
                model.l("nv_smp_mass", gb, (v.scaled, v.xb, t as i32, w as i32, v.bval, v.bidx, vl, v.acc))?;
                model.l("nv_smp_mass2", g1, (v.bval, v.bidx, nb as i32, v.xb, t as i32, r, w as i32, v.acc))?;
            }
            xch(xb_mass(t), 4 * w)?;
            for j in 0..m {
                let v = self.rv(model, lg, j);
                model.l("nv_smp_accept", g32, (v.xb, t as i32, w as i32, sp, v.tok, v.acc))?;
            }
        }
        for j in 0..m {
            let v = self.rv(model, lg, j);
            model.l("nv_topk", g1, (v.sel, v.nsel, v.scaled, &self.posv, candl as i32, vl as i64, v.acc, &self.smp_gather))?;
            model.l("nv_smp_cands_out", g1, (v.sel, v.nsel, v.scaled, v0, v.xb, candl as i32, r, w as i32))?;
        }
        xch(XB_CAND, 2 * candl * w)?;
        for j in 0..m {
            let v = self.rv(model, lg, j);
            model.l("nv_smp_sample_cands", g1, (v.tok, v.xb, w as i32, candl as i32, sp, v.rng, v.acc))?;
            // the lane's step advances once per row, as in the per-row path (the flags carry the cancel bit)
            model.l("nv_smp_note", g32, (counts, v.tok, rng, v.xb, &self.flags, 0i32))?;
            if j + 1 < m {
                model.l("nv_spec_accept_rows", g1,
                        (&self.smp_on, v.xb, w as i32, (candl * w) as i32, sp, j as i32, &self.smp_draft,
                         self.smp_q + (j as u64 * self.smp_stride as u64 * 4), self.smp_qstat, &self.smp_rng, &self.smp_acc, &self.smp_rtok))?;
            }
        }
        Ok(())
    }

    /// Record the step into a graph (once). The inputs hold zeros at capture; the recorded kernels
    /// read them at replay.
    pub fn capture(&mut self, model: &GpuModel) -> Result<()> {
        let raw = model.stream.stream;
        model.sync()?;
        unsafe {
            let r = sys::cuStreamBeginCapture_v2(raw, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
            ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuStreamBeginCapture: {r:?}");
        }
        let body = self.kernels(model);
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let r2 = unsafe { sys::cuStreamEndCapture(raw, &mut graph) };
        body?;
        ensure!(r2 == sys::CUresult::CUDA_SUCCESS, "cuStreamEndCapture: {r2:?}");
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let r3 = unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        unsafe { sys::cuGraphDestroy(graph); }
        ensure!(r3 == sys::CUresult::CUDA_SUCCESS, "cuGraphInstantiate: {r3:?}");
        self.exec = Some(exec);
        crate::dprintln!("[naive-decode] step graph captured: m {} head {} (workspace peak {:.1} MB)", self.m, self.head, self.ws.peak as f64 / 1e6);
        Ok(())
    }

    /// The head's cancel flag for the NEXT step (rides the step's first exchange to every rank).
    pub fn set_cancel(&mut self, model: &GpuModel, cancel: bool) -> Result<()> {
        model.dev.htod_sync_copy_into(&[cancel as i32], &mut self.cancel)?;
        Ok(())
    }

    /// A request begins: the sampler knobs, the RNG seed, the penalty tables (the prompt is "seen").
    /// `lane` is the G1 slot this request owns; it defaults to 0 so the single-stream path is unchanged.
    pub fn begin_request(&mut self, model: &GpuModel, sp: &SampleParams, seed: u64, prompt: &[u32]) -> Result<()> {
        self.begin_request_lane(model, sp, seed, prompt, 0)
    }

    /// G1: as `begin_request`, but into lane `l`'s strided region. Only that lane's slice is written --
    /// the other lanes' sampler state, penalty tables and RNG are left untouched.
    pub fn begin_request_lane(&mut self, model: &GpuModel, sp: &SampleParams, seed: u64, prompt: &[u32], l: usize) -> Result<()> {
        model.sync()?;
        let dev = &model.dev;
        if let Some((spp, cp, sep, rp, n)) = self.ext {
            // G1 shared table: write lane l's regions only (direct copies at the lane offsets)
            let l = l.min(n - 1);
            let v = model.cfg.vocab_size;
            dev.htod_sync_copy_into(&[0i32], &mut self.cancel)?;
            let mut seen = vec![0u8; v];
            for &t in prompt { if (t as usize) < v { seen[t as usize] = 1; } }
            unsafe {
                use cudarc::driver::result::memcpy_htod_sync as up;
                up(spp + 32 * l as u64, &sp.row()).map_err(|e| anyhow::anyhow!("lane knobs: {e:?}"))?;
                up(rp + 16 * l as u64, &[seed, 0u64]).map_err(|e| anyhow::anyhow!("lane rng: {e:?}"))?;
                up(cp + (4 * v * l) as u64, &vec![0i32; v]).map_err(|e| anyhow::anyhow!("lane counts: {e:?}"))?;
                up(sep + (v * l) as u64, &seen).map_err(|e| anyhow::anyhow!("lane seen: {e:?}"))?;
            }
            dev.synchronize()?;
            self.cur_lane = 0;   // the m = 1 step and the verify blocks always sample lane 0 (lanes stay compacted)
            return Ok(());
        }
        let l = l.min(self.m - 1);
        dev.htod_sync_copy_into(&[0i32], &mut self.cancel)?;
        // lane l's 8 f32 of knobs and its RNG pair. Same read-modify-write discipline as the tables below:
        // only lane l's region changes; every other lane's bytes are read back and written back as-is.
        let row = sp.row();
        let mut knobs = dev.dtoh_sync_copy(&self.sp)?;
        knobs[8 * l..8 * l + 8].copy_from_slice(&row);
        dev.htod_sync_copy_into(&knobs, &mut self.sp)?;
        let mut seeds = dev.dtoh_sync_copy(&self.rng)?;
        // BOTH words are set: rng[l*2] is the seed, rng[l*2+1] the stream's step counter, which must
        // start at 0 for this request. Preserving a carried-over counter (the obvious
        // read-modify-write mistake) makes a re-admitted lane continue its PREVIOUS stream instead
        // of restarting from its seed -- caught by probe_lanes check 2 (same seed, two lanes, sums
        // must match).
        seeds[2 * l] = seed;
        seeds[2 * l + 1] = 0;
        dev.htod_sync_copy_into(&seeds, &mut self.rng)?;
        // the penalty tables for LANE l only: `counts[l*v..] = 0` and `seen[l*v..]` marks the prompt. Read
        // the current device contents first and write them back unchanged, so admitting lane l NEVER
        // disturbs a lane already admitted (a request only ever touches its own region).
        let v = model.cfg.vocab_size;
        let mut counts = dev.dtoh_sync_copy(&self.counts)?;
        counts[l * v..(l + 1) * v].iter_mut().for_each(|c| *c = 0);
        dev.htod_sync_copy_into(&counts, &mut self.counts)?;
        // `seen` MUST be cleared, not just marked. Two reasons, both from AGENTS.md §2.2 ("alloc_zeros
        // does NOT zero; fresh buffers a kernel reads must go through Pool::get or explicit memset"):
        //   1. On the FIRST request the buffer is uninitialized pool memory, so `seen` is garbage.
        //   2. On a LATER request it still holds the previous request's marks, and the
        //      read-modify-write below preserves them.
        // `nv_smp_prep` applies the repetition penalty to any token with `seen[v0+v] ||
        // counts[v0+v] > 0` (gpu_naive.cu:1917), so every token the last request sampled -- plus a
        // random subset on the first request -- gets its logit divided by `rep` on every later
        // request. That is the degenerate single-token output this lane path produces.
        //
        // The pre-G1 `begin_request` wrote the whole buffer from a fresh `vec![0u8; v]`, so it was
        // always clean; the lane version turned that into a read-modify-write and lost it.
        let mut seen = dev.dtoh_sync_copy(&self.seen)?;
        seen[l * v..(l + 1) * v].iter_mut().for_each(|s| *s = 0);
        for &t in prompt { if (t as usize) < v { seen[l * v + t as usize] = 1; } }
        dev.htod_sync_copy_into(&seen, &mut self.seen)?;
        dev.synchronize()?;
        // the next step samples THIS request's lane
        self.cur_lane = l;
        Ok(())
    }

    /// One step: rows `ids` at `pos` (m of each) -> (the sampled token of the last row, the head's cancel
    /// flag as every rank saw it) — identical on every rank.
    pub fn step(&mut self, model: &GpuModel, ids: &[u32], pos: &[usize]) -> Result<(u32, bool)> {
        ensure!(ids.len() == self.m && pos.len() == self.m, "step rows");
        ensure!(pos.iter().all(|&p| p < model.max_pos), "position beyond max_pos");
        model.sync()?;
        let dev = &model.dev;
        dev.htod_sync_copy_into(&ids.iter().map(|x| *x as i32).collect::<Vec<i32>>(), &mut self.ids_d)?;
        dev.htod_sync_copy_into(&pos.iter().map(|x| *x as i32).collect::<Vec<i32>>(), &mut self.pos_d)?;
        match self.exec {
            Some(exec) => {
                let r = unsafe { sys::cuGraphLaunch(exec, model.stream.stream) };
                ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuGraphLaunch: {r:?}");
            }
            None => self.kernels(model)?,
        }
        model.sync()?;
        let t = dev.dtoh_sync_copy(&self.tok)?[0];
        let f = dev.dtoh_sync_copy(&self.flags)?[0];
        Ok((t as u32, f != 0))
    }

    /// G1: one multi-request step (`set_multilane`): row j = lane j's next token at `pos[j]` -> every lane's sampled token.
    pub fn step_lanes(&mut self, model: &GpuModel, ids: &[u32], pos: &[usize]) -> Result<Vec<u32>> {
        ensure!(self.multilane && ids.len() == self.m && pos.len() == self.m, "lane step rows");
        ensure!(pos.iter().all(|&p| p < model.max_pos), "position beyond max_pos");
        model.sync()?;
        let dev = &model.dev;
        dev.htod_sync_copy_into(&ids.iter().map(|x| *x as i32).collect::<Vec<i32>>(), &mut self.ids_d)?;
        dev.htod_sync_copy_into(&pos.iter().map(|x| *x as i32).collect::<Vec<i32>>(), &mut self.pos_d)?;
        match self.exec {
            Some(exec) => { let r = unsafe { sys::cuGraphLaunch(exec, model.stream.stream) }; ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuGraphLaunch: {r:?}"); }
            None => self.kernels(model)?,
        }
        model.sync()?;
        Ok(dev.dtoh_sync_copy(&self.toks)?.iter().map(|x| *x as u32).collect())
    }

    /// The verify block (m > 1): rows `ids` at `pos` -> (every row's sampled token, the cancel flag).
    pub fn step_block(&mut self, model: &GpuModel, ids: &[u32], pos: &[usize]) -> Result<(Vec<u32>, bool)> {
        self.step_block_smp(model, ids, pos, None)
    }

    /// The verify block, optionally with the exact speculative acceptance of `drafts` (work item A):
    /// `Some((drafts, seed, step))` turns `smp_on` on for this replay (a no-op on a node rank, whose
    /// graph has no q). Returns `(row draws, cancel)`; the accept flags are read via `smp_accepts`.
    pub fn step_block_smp(&mut self, model: &GpuModel, ids: &[u32], pos: &[usize], smp: Option<(&[u32], u64, u64, bool)>) -> Result<(Vec<u32>, bool)> {
        ensure!(self.m > 1 && ids.len() == self.m && pos.len() == self.m, "block rows");
        ensure!(pos.iter().all(|&p| p < model.max_pos), "position beyond max_pos");
        model.sync()?;
        let dev = &model.dev;
        if let Some((drafts, seed, step, onehot)) = smp {
            ensure!(drafts.len() == self.m - 1, "drafts {} != block rows - 1", drafts.len());
            // cudarc's copy asserts exact length equality, and `smp_draft` is a fixed MAX_ROWS buffer;
            // pad (the kernel reads draft[row] for row < k = m-1 only).
            let mut dv = drafts.iter().map(|x| *x as i32).collect::<Vec<i32>>();
            dv.resize(MAX_ROWS, 0);
            dev.htod_sync_copy_into(&dv, &mut self.smp_draft)?;
            dev.htod_sync_copy_into(&[seed, step, onehot as u64], &mut self.smp_rng)?;
            // HEAD ONLY: q lives only on the head, so the accept kernel must be a no-op on a node
            // rank. Enabling it there made it read qs == NULL (an illegal address at TP).
            let want = self.head;
            if want != self.smp_on_cur { dev.htod_sync_copy_into(&[want as i32], &mut self.smp_on)?; self.smp_on_cur = want; }
            // EVERY rank gathers its candidates for every row of a sampled block (see `smp_gather`)
            if !self.smp_gather_cur { dev.htod_sync_copy_into(&[1i32], &mut self.smp_gather)?; self.smp_gather_cur = true; }
        } else {
            if self.head && self.smp_on_cur { dev.htod_sync_copy_into(&[0i32], &mut self.smp_on)?; self.smp_on_cur = false; }
            if self.smp_gather_cur { dev.htod_sync_copy_into(&[0i32], &mut self.smp_gather)?; self.smp_gather_cur = false; }
        }
        dev.htod_sync_copy_into(&ids.iter().map(|x| *x as i32).collect::<Vec<i32>>(), &mut self.ids_d)?;
        dev.htod_sync_copy_into(&pos.iter().map(|x| *x as i32).collect::<Vec<i32>>(), &mut self.pos_d)?;
        match self.exec {
            Some(exec) => { let r = unsafe { sys::cuGraphLaunch(exec, model.stream.stream) }; ensure!(r == sys::CUresult::CUDA_SUCCESS, "cuGraphLaunch: {r:?}"); }
            None => self.kernels(model)?,
        }
        model.sync()?;
        let t = dev.dtoh_sync_copy(&self.toks)?;
        let f = dev.dtoh_sync_copy(&self.flags)?[0];
        Ok((t.iter().map(|x| *x as u32).collect(), f != 0))
    }

    /// The per-row accept flags / residual draws of the last sampled verify block (valid only when
    /// `step_block_smp` was given drafts; `k` = m - 1). A residual draw of -1 means the kernel had
    /// nothing to draw from (no candidate list, or an empty residual): the caller commits the
    /// target's own draw for that row instead.
    pub fn smp_accepts(&self, model: &GpuModel, k: usize) -> Result<(Vec<i32>, Vec<i32>)> {
        let a = model.dev.dtoh_sync_copy(&self.smp_acc)?;
        let r = model.dev.dtoh_sync_copy(&self.smp_rtok)?;
        Ok((a[..k].to_vec(), r[..k].to_vec()))
    }

    /// Count the committed tokens of a round into the penalty table (eager). Writes LANE 0; a G1 lane
    /// calls `note_commit_lane` so each request counts into its OWN table.
    pub fn note_commit(&mut self, model: &GpuModel, toks: &[u32]) -> Result<()> {
        self.note_commit_lane(model, toks, 0)
    }

    /// G1: as `note_commit`, but into lane `l`'s strided `counts` region.
    pub fn note_commit_lane(&mut self, model: &GpuModel, toks: &[u32], l: usize) -> Result<()> {
        if toks.is_empty() { return Ok(()); }
        let d = model.dev.htod_sync_copy(&toks.iter().map(|x| *x as i32).collect::<Vec<i32>>())?;
        // `counts_lane` is the base of lane l's table; nv_smp_count indexes it like the old single
        // `counts` (one token index per committed token).
        let base = self.counts_lane(l);
        model.l("nv_smp_count", LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 }, (&d, base, toks.len() as i32))?;
        model.sync()?;
        Ok(())
    }

    /// The drafter's tap view of the block: (address, pitch) per tap layer, rows 0.. of the block.
    pub fn tap_view(&self, model: &GpuModel) -> Vec<(u64, usize)> {
        let h = model.cfg.hidden_size;
        (0..self.tap_layers.len()).map(|j| (dptr(&self.tap_buf) + (j * self.m * h * 4) as u64, h)).collect()
    }

    /// Sample from an arbitrary LOCAL logits row (f16 [vl] at a device address the caller keeps alive):
    /// the prefill's last row. Eager (not graphed). -> (token, cancel flag)
    pub fn sample_row(&mut self, model: &GpuModel, row: u64) -> Result<(u32, bool)> {
        let lane = self.cur_lane;
        self.sample_row_lane(model, row, lane)
    }

    /// G1: as `sample_row`, but off lane `l`'s knobs/RNG/penalty tables.
    pub fn sample_row_lane(&mut self, model: &GpuModel, row: u64, l: usize) -> Result<(u32, bool)> {
        model.sync()?;
        self.sampler_kernels_lane(model, row, dptr(&self.tok), true, l)?;
        model.sync()?;
        let t = model.dev.dtoh_sync_copy(&self.tok)?[0] as u32;
        let f = model.dev.dtoh_sync_copy(&self.flags)?[0];
        Ok((t, f != 0))
    }
}

/// `--probe-naive-graph --model-dir <pack> [--layers N]`: on one GPU, the graphed step vs the eager
/// step (bitwise logits over a 4-token greedy continuation), then the replay time.
pub fn probe_graph(dir: &str, n_layers: usize, bench_only: bool) -> Result<()> {
    use super::pack::NaivePack;
    let pack = NaivePack::open(dir)?;
    let dev = cudarc::driver::CudaDevice::new(0).context("CudaDevice::new(0)")?;
    let model = GpuModel::load_slice_on(dev, &pack, n_layers, 4096, 1024, 1, super::shard::Shard::single())?;
    let prompt: Vec<u32> = vec![785, 6722, 315, 9625, 374];
    if bench_only {
        // m = 1 replays only (a clean per-kernel profile under nsys)
        let mut d1 = DecodeStep::new(&model, 1, true)?;
        d1.capture(&model)?;
        d1.begin_request(&model, &SampleParams::GREEDY, 1, &prompt)?;
        let mut t = prompt[0];
        for i in 0..5 { t = d1.step(&model, &[t], &[i])?.0; }
        let t2 = std::time::Instant::now();
        let n = 100;
        for i in 0..n { t = d1.step(&model, &[t], &[5 + i])?.0; }
        println!("  graphed m=1 step: {:.2} ms over {n} replays", t2.elapsed().as_secs_f64() / n as f64 * 1e3);
        println!("NAIVE_GRAPH_BENCH_DONE");
        return Ok(());
    }
    // eager reference: prefill (one chunk), then greedy tokens through forward_rows_last
    let mut eager = Vec::new();
    let mut ids = prompt.clone();
    for _ in 0..4 {
        let pos: Vec<usize> = (0..ids.len()).collect();
        let lg = model.forward_rows_last(&ids, &pos, true)?.unwrap();
        let am = lg.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32;
        eager.push(am); ids.push(am);
    }
    // graphed: a 5-row step for the prompt (m = 5, eager sampler on its last row), then m = 1 steps
    let mut d5 = DecodeStep::new(&model, prompt.len(), true)?;
    let mut d1 = DecodeStep::new(&model, 1, true)?;
    d1.capture(&model)?;
    d5.begin_request(&model, &SampleParams::GREEDY, 1, &prompt)?;
    d1.begin_request(&model, &SampleParams::GREEDY, 1, &prompt)?;
    let pos: Vec<usize> = (0..prompt.len()).collect();
    let t0 = *d5.step_block(&model, &prompt, &pos)?.0.last().unwrap();
    let mut graphed = vec![t0];
    let mut p = prompt.len();
    let mut t = t0;
    let t1 = std::time::Instant::now();
    for _ in 0..3 { t = d1.step(&model, &[t], &[p])?.0; graphed.push(t); p += 1; }
    let dt = t1.elapsed().as_secs_f64() / 3.0;
    println!("  eager greedy   {eager:?}\n  graphed greedy {graphed:?}");
    // replay timing at steady state (positions advance; the K/V writes are harmless here)
    let t2 = std::time::Instant::now();
    let n = 20;
    for i in 0..n { d1.step(&model, &[t], &[p + i])?; }
    let per = t2.elapsed().as_secs_f64() / n as f64;
    println!("  graphed m=1 step: {:.2} ms (first 3: {:.2} ms); eager m=1 step for comparison:", per * 1e3, dt * 1e3);
    let t3 = std::time::Instant::now();
    for i in 0..5 { model.forward_rows_last(&[t], &[p + n + i], true)?; }
    println!("  eager m=1 step: {:.2} ms", t3.elapsed().as_secs_f64() / 5.0 * 1e3);
    ensure!(eager == graphed, "NAIVE_GRAPH_FAIL: tokens differ");
    println!("NAIVE_GRAPH_OK");
    Ok(())
}

/// `--probe-naive-sampler --model-dir <pack>`: the GPU sampler against a CPU reference on a real logits
/// row (2 layers): greedy == CPU argmax; for (T, top_p, top_k, min_p) settings, 3000 draws with distinct
/// seeds: every draw inside the CPU nucleus set and the total-variation distance to the CPU truncated
/// distribution small.
pub fn probe_sampler(dir: &str) -> Result<()> {
    use super::pack::NaivePack;
    let pack = NaivePack::open(dir)?;
    let dev = cudarc::driver::CudaDevice::new(0).context("CudaDevice::new(0)")?;
    let model = GpuModel::load_slice_on(dev.clone(), &pack, 2, 4096, 1024, 1, super::shard::Shard::single())?;
    let v = model.cfg.vocab_size;
    let ids: Vec<u32> = vec![785, 6722, 315, 9625, 374];
    let pos: Vec<usize> = (0..ids.len()).collect();
    let (lg, _) = model.forward_rows(&ids, &pos)?;
    let row: Vec<f32> = lg[(ids.len() - 1) * v..ids.len() * v].to_vec();
    // the row on the device as f16 (what the step hands the sampler)
    let row16: Vec<u16> = row.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect();
    let row_d = dev.htod_sync_copy(&row16)?;
    let row_f: Vec<f32> = row16.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect();
    let mut d = DecodeStep::new(&model, 1, true)?;
    let argmax = row_f.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
    d.begin_request(&model, &SampleParams::GREEDY, 7, &[])?;
    let g = d.sample_row(&model, dptr(&row_d))?.0 as usize;
    println!("  greedy: gpu {g} cpu {argmax} {}", if g == argmax { "OK" } else { "MISMATCH" });
    let mut ok = g == argmax;
    for (label, sp) in [("T=1 top_p=0.95", SampleParams { temperature: 1.0, top_p: 0.95, top_k: 0, min_p: 0.0, rep_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0 }),
                        ("T=0.7 top_p=0.8 top_k=20", SampleParams { temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0.0, rep_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0 }),
                        ("T=1 min_p=0.1", SampleParams { temperature: 1.0, top_p: 1.0, top_k: 0, min_p: 0.1, rep_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0 })] {
        // CPU truncated distribution
        let inv_t = 1.0 / sp.temperature;
        let mx = row_f.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut p: Vec<f64> = row_f.iter().map(|x| (((x - mx) * inv_t) as f64).exp()).collect();
        let z: f64 = p.iter().sum();
        for x in p.iter_mut() { *x /= z; }
        let mut idx: Vec<usize> = (0..v).collect();
        idx.sort_by(|a, b| p[*b].partial_cmp(&p[*a]).unwrap().then(a.cmp(b)));
        let mut keep = v;
        if sp.top_k > 0 { keep = keep.min(sp.top_k); }
        if sp.min_p > 0.0 { let fl = p[idx[0]] * sp.min_p as f64; keep = keep.min(idx.iter().take_while(|i| p[**i] >= fl).count().max(1)); }
        if sp.top_p < 1.0 { let mut acc = 0f64; let mut k = 0; for &i in &idx[..keep] { acc += p[i]; k += 1; if acc >= sp.top_p as f64 { break; } } keep = k.max(1); }
        let nucleus: std::collections::HashSet<usize> = idx[..keep].iter().cloned().collect();
        let zk: f64 = idx[..keep].iter().map(|i| p[*i]).sum();
        let n = 3000usize;
        let mut hist: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut outside = 0usize;
        for seed in 0..n as u64 {
            d.begin_request(&model, &sp, 1000 + seed, &[])?;
            let t = d.sample_row(&model, dptr(&row_d))?.0 as usize;
            if !nucleus.contains(&t) { outside += 1; }
            *hist.entry(t).or_insert(0) += 1;
        }
        // the distribution test at the resolution the draws support: the nucleus split into 10 buckets of
        // equal CPU mass (in sorted order); each bucket's draw share vs 0.1 (noise ~0.006 at 3000 draws)
        // buckets by cumulative CPU mass; a single token heavier than a bucket owns it, so compare each
        // bucket's draw share with ITS expected mass (not 0.1)
        let mut got = vec![0f64; 10];
        let mut want = vec![0f64; 10];
        let mut cum = 0f64;
        for &i in &idx[..keep] {
            let b = ((cum / zk) * 10.0).floor().min(9.0) as usize;
            got[b] += *hist.get(&i).unwrap_or(&0) as f64 / n as f64;
            want[b] += p[i] / zk;
            cum += p[i];
        }
        let tv: f64 = if keep <= 64 {
            0.5 * idx[..keep].iter().map(|i| ((*hist.get(i).unwrap_or(&0) as f64 / n as f64) - p[*i] / zk).abs()).sum::<f64>() + 0.5 * (outside as f64 / n as f64)
        } else {
            0.5 * got.iter().zip(&want).map(|(g, w)| (g - w).abs()).sum::<f64>() + 0.5 * (outside as f64 / n as f64)
        };
        println!("  {label}: nucleus {keep} tokens (mass {:.3}); {n} draws: {outside} outside, TV {tv:.3}{}", zk,
                 if keep > 64 { format!(" got {:?} want {:?}", got.iter().map(|b| (b * 1000.0).round() / 1000.0).collect::<Vec<_>>(), want.iter().map(|b| (b * 1000.0).round() / 1000.0).collect::<Vec<_>>()) } else { String::new() });
        if outside > 0 || tv > 0.05 { ok = false; }
    }
    if !ok { anyhow::bail!("NAIVE_SAMPLER_FAIL"); }
    println!("NAIVE_SAMPLER_OK");
    Ok(())
}
/// `--probe-naive-lanes --model-dir <pack>`: G1 lane isolation — `m` independent requests in one
/// DecodeStep must not share ANY sampler state. Three checks, each of which fails loudly (this is
/// the check that catches the admit-wipes-a-live-lane bug a whole-buffer upload would introduce):
///   1. ADMIT ISOLATION — admitting lane L leaves every other lane's knobs / RNG / penalty tables
///      BYTE-IDENTICAL (read the whole strided buffers back and compare the untouched lanes).
///   2. RNG INDEPENDENCE — the same seed on two lanes draws the SAME token (proves the lane base
///      really selects the stream), while different seeds differ.
///   3. KNOBS INDEPENDENCE — lane L's temperature/top_p/top_k take effect and do not leak into the
///      other lanes: a greedy lane stays greedy while a sampled lane's draw stays inside ITS nucleus.
pub fn probe_lanes(dir: &str) -> Result<()> {
    use super::pack::NaivePack;
    let pack = NaivePack::open(dir)?;
    let dev = cudarc::driver::CudaDevice::new(0).context("CudaDevice::new(0)")?;
    let model = GpuModel::load_slice_on(dev.clone(), &pack, 2, 4096, 1024, 1, super::shard::Shard::single())?;
    let v = model.cfg.vocab_size;
    const M: usize = 4;
    let ids: Vec<u32> = vec![785, 6722, 315, 9625, 374];
    let pos: Vec<usize> = (0..ids.len()).collect();
    let (lg, _) = model.forward_rows(&ids, &pos)?;
    let row: Vec<f32> = lg[(ids.len() - 1) * v..ids.len() * v].to_vec();
    let row16: Vec<u16> = row.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect();
    let row_d = dev.htod_sync_copy(&row16)?;
    let argmax = row16.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, b16)| {
        let x = half::f16::from_bits(*b16).to_f32(); if x > b.1 { (i, x) } else { b }
    }).0;
    let mut d = DecodeStep::new(&model, M, true)?;
    let dev = &model.dev;
    let mut ok = true;

    // ---- 1. admit isolation: every lane distinct, then admit L and diff the others -------------
    let hot = SampleParams { temperature: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0, rep_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0 };
    let cold = SampleParams { temperature: 0.7, top_p: 0.5, top_k: 0, min_p: 0.0, rep_penalty: 1.0, presence_penalty: 0.0, frequency_penalty: 0.0 };
    for l in 0..M {
        let (sp, seed, prompt) = if l % 2 == 0 { (&hot, 1000u64 + l as u64, &ids[..l + 1]) } else { (&cold, 5000u64 + l as u64, &ids[..M - l]) };
        d.begin_request_lane(&model, sp, seed, prompt, l)?;
    }
    let (sp0, cnt0, seen0, rng0) = (dev.dtoh_sync_copy(&d.sp)?, dev.dtoh_sync_copy(&d.counts)?, dev.dtoh_sync_copy(&d.seen)?, dev.dtoh_sync_copy(&d.rng)?);
    // re-admit lane 2 (a HOT lane, so its knobs really differ from a cold lane's)
    d.begin_request_lane(&model, &hot, 1000 + 2, &ids[..3], 2)?;
    let (sp1, cnt1, seen1, rng1) = (dev.dtoh_sync_copy(&d.sp)?, dev.dtoh_sync_copy(&d.counts)?, dev.dtoh_sync_copy(&d.seen)?, dev.dtoh_sync_copy(&d.rng)?);
    let mut bad = 0usize;
    for l in 0..M {
        if l == 2 { continue; }
        if sp0[8 * l..8 * l + 8] != sp1[8 * l..8 * l + 8] { bad += 1; println!("  FAIL knobs lane {l} disturbed"); }
        if rng0[2 * l..2 * l + 2] != rng1[2 * l..2 * l + 2] { bad += 1; println!("  FAIL rng lane {l} disturbed"); }
        if cnt0[l * v..(l + 1) * v] != cnt1[l * v..(l + 1) * v] { bad += 1; println!("  FAIL counts lane {l} disturbed"); }
        if seen0[l * v..(l + 1) * v] != seen1[l * v..(l + 1) * v] { bad += 1; println!("  FAIL seen lane {l} disturbed"); }
    }
    println!("  admit isolation: {} lanes, {bad} cross-lane writes {}", M, if bad == 0 { "OK" } else { "FAIL" });
    ok &= bad == 0;

    // ---- 2. RNG independence: same seed on two lanes -> same token ---------------------------
    // Same seed on lane 0 and lane 2 of a 4-lane step MUST draw the same token (the lane base picks
    // the stream, and both lanes carry that seed); a third stream (fresh seed, lane 0) must not.
    let mut d1 = DecodeStep::new(&model, M, true)?;
    let (mut l0, mut l2, mut other) = (0u32, 0u32, 0u32);
    for i in 0..64u64 {
        d1.begin_request_lane(&model, &hot, 4242, &[], 0)?;
        l0 += d1.sample_row_lane(&model, dptr(&row_d), 0)?.0;
        d1.begin_request_lane(&model, &hot, 4242, &[], 2)?;
        l2 += d1.sample_row_lane(&model, dptr(&row_d), 2)?.0;
        d1.begin_request_lane(&model, &hot, 9000 + i, &[], 0)?;
        other += d1.sample_row_lane(&model, dptr(&row_d), 0)?.0;
    }
    println!("  rng streams: seed 4242 on lane 0 summed {l0}, on lane 2 summed {l2} (must match); a third stream summed {other}");
    ok &= l0 == l2 && l0 != other;

    // ---- 3. knob independence: lane 0 greedy, lane 1 sampled; each obeys ITS own params -------
    let mut d2 = DecodeStep::new(&model, M, true)?;
    d2.begin_request_lane(&model, &SampleParams::GREEDY, 77, &[], 0)?;
    d2.begin_request_lane(&model, &hot, 88, &[], 1)?;
    let mut greedy_ok = true;
    for _ in 0..32 {
        d2.begin_request_lane(&model, &SampleParams::GREEDY, 77, &[], 0)?;
        if d2.sample_row_lane(&model, dptr(&row_d), 0)?.0 as usize != argmax { greedy_ok = false; }
    }
    // lane 1 stays inside its own nucleus (top_k 20 of a CPU-sorted row)
    let mut order: Vec<usize> = (0..v).collect();
    let rf: Vec<f32> = row16.iter().map(|b| half::f16::from_bits(*b).to_f32()).collect();
    order.sort_by(|a, b| rf[*b].partial_cmp(&rf[*a]).unwrap().then(a.cmp(b)));
    let nucleus: std::collections::HashSet<usize> = order[..20].iter().copied().collect();
    let mut outside = 0u32;
    for _ in 0..256 {
        d2.begin_request_lane(&model, &hot, 88, &[], 1)?;
        let t = d2.sample_row_lane(&model, dptr(&row_d), 1)?.0 as usize;
        if !nucleus.contains(&t) { outside += 1; }
    }
    println!("  knob independence: lane 0 greedy {} ({argmax}); lane 1 top-k=20 nucleus misses {outside}/256",
             if greedy_ok { "OK" } else { "FAIL" });
    ok &= greedy_ok && outside == 0;

    if !ok { anyhow::bail!("NAIVE_LANES_FAIL"); }
    println!("NAIVE_LANES_OK");
    Ok(())
}/// `--probe-naive-slots --model-dir <pack>`: the per-request slot dimension (option A). The
/// cross-chain check the naive path has never had — `forward_rows` is the SEQUENTIAL reference (it
/// was the only way to run any sequence before) and the batched 2-row call is the new class, so
/// "they agree" is evidence neither path silently reads another request's cache.
///
/// Three checks:
///   1. TWO SEQUENCES, TWO SLOTS vs TWO SEQUENTIAL FORWARDS — bitwise identical logits. This is the
///      whole point: with both rows on slot 0 (the old addressing) the second row overwrites the
///      first's K/V and this FAILS, so it is a positive assertion, not an empty-vs-empty one.
///   2. SLOT INDEPENDENCE IS NOT ACCIDENTAL — the same sequence run twice in DIFFERENT slots must
///      give identical logits (a slot plane is fully transparent), while a deliberately corrupted
///      slot-0 plane must CHANGE row 1's logits (proving row 1 really reads its own plane).
///   3. n_slots == 1 REPRODUCES TODAY — the 2-row call with both rows on slot 0 equals the same
///      call before the slot dimension existed (row 0's logits), i.e. the patch is inert at 1 slot.
///
/// Layer count is tiny (a slice): this is an addressing check, not a quality check.

pub fn probe_slots(dir: &str) -> Result<()> {
    use super::pack::NaivePack;
    let pack = NaivePack::open(dir)?;
    let dev = cudarc::driver::CudaDevice::new(0).context("CudaDevice::new(0)")?;
    const SLOTS: usize = 2;
    const NLAYERS: usize = 3;
    let v = pack.cfg.vocab_size;
    let mut ok = true;

    // ONE sequence split into two rows at CONSECUTIVE positions: row 0 at pos 0, row 1 at pos 1.
    // Consecutive positions are what make the test decisive. Causal attention gives row 0 (pos 0) a
    // window of only its own key, so it can NEVER see row 1's K/V whatever the slot; row 1 (pos 1)
    // sees positions 0 and 1, so WHETHER row 0's key is visible is decided purely by which slot
    // row 1 reads. That is the whole slot dimension, isolated to one comparison.
    let seq: Vec<u32> = vec![785, 6722, 315, 9625];
    let p_one: Vec<usize> = (0..seq.len()).collect();
    let two: Vec<u32> = seq[..2].to_vec();
    let p_two: Vec<usize> = vec![0, 1];

    // A FRESH MODEL PER RUN. `sids` is model-level state, so a shared model lets a run inherit the
    // previous one's slot assignment -- the bug this probe first shipped with.
    let fresh = |slots: usize| GpuModel::load_slice_on(dev.clone(), &pack, NLAYERS, 4096, 1024, slots, super::shard::Shard::single());
    let set_sids = |mo: &mut GpuModel, row0: i32, row1: i32| -> Result<()> {
        let mut s = vec![0i32; MAX_ROWS];
        s[0] = row0; s[1] = row1;
        mo.dev.htod_sync_copy_into(&s, &mut mo.sids_buf)?;
        Ok(())
    };
    // `alloc_zeros` is an ASYNC alloc that does not zero (AGENTS.md 2.2), so a fresh model's cache
    // is NOT clean: every plane must be explicitly memset before a run that claims a baseline.
    let zero_cache = |mo: &mut GpuModel| -> Result<()> {
        let n = mo.dev.dtoh_sync_copy(&mo.layers[0].kc)?.len();
        let z = vec![0u16; n];
        mo.dev.htod_sync_copy_into(&z, &mut mo.layers[0].kc)?;
        Ok(())
    };
    fn row<'x>(lg: &'x [f32], r: usize, v: usize) -> &'x [f32] { &lg[r * v..(r + 1) * v] }

    // run A: both rows on slot 0 -- they share a cache plane, so row 1 DOES see row 0's key.
    let mut ma = fresh(SLOTS)?;
    set_sids(&mut ma, 0, 0)?;
    zero_cache(&mut ma)?;
    let (lg_a, _) = ma.forward_rows(&two, &p_two)?;
    // run B: row 0 on slot 0, row 1 on slot 1 -- row 1's plane holds only its own key.
    let mut mb = fresh(SLOTS)?;
    set_sids(&mut mb, 0, 1)?;
    zero_cache(&mut mb)?;
    let (lg_b, _) = mb.forward_rows(&two, &p_two)?;
    // run C: the same sequence sequentially, entirely on slot 0 -- the pre-slot reference.
    let mut mc = fresh(SLOTS)?;
    set_sids(&mut mc, 0, 0)?;
    zero_cache(&mut mc)?;
    let (lg_c, _) = mc.forward_rows(&seq, &p_one)?;

    // (1) row 0 is slot-invariant: pos 0 sees only its own key in every run.
    let r0_stable = row(&lg_a, 0, v) == row(&lg_b, 0, v) && row(&lg_a, 0, v) == row(&lg_c, 0, v);
    println!("  row0 (pos 0, cannot see the other row) is slot-invariant: {}", if r0_stable { "OK" } else { "MISMATCH" });
    ok &= r0_stable;

    // (2) row 1 on a SHARED plane equals the sequential reference: same slot, same positions.
    let r1_shared_ok = row(&lg_a, 1, v) == row(&lg_c, 1, v);
    println!("  row1 on slot 0 == sequential reference: {}", if r1_shared_ok { "OK" } else { "MISMATCH" });
    ok &= r1_shared_ok;

    // (3) THE load-bearing check: move row 1 to its own slot and row 1 MUST change, because it no
    // longer sees row 0's key. Unchanged here would mean the slot dimension is decorative -- every
    // row reading slot 0 regardless of sids.
    let r1_isolated = row(&lg_b, 1, v) != row(&lg_a, 1, v);
    println!("  row1 on slot 1 != row1 on slot 0 (it lost row0's key): {}", if r1_isolated { "OK" } else { "IDENTICAL (BAD)" });
    ok &= r1_isolated;

    // (4) row 1 in isolation == row 1 computed on its own plane alone at pos 1: its slot plane
    // contains exactly one key, so batching must not change it.
    let mut md = fresh(SLOTS)?;
    set_sids(&mut md, 0, 0)?;
    zero_cache(&mut md)?;
    let (lg_d, _) = md.forward_rows(&seq[1..2], &[1usize])?;
    let r1_solo = row(&lg_d, 0, v) == row(&lg_b, 1, v);
    println!("  row1 (slot 1, batched) == row1 (slot 1, solo at pos 1): {}", if r1_solo { "OK" } else { "MISMATCH" });
    ok &= r1_solo;

    // (5) n_slots == 1 is pre-patch behaviour: the patch is inert at one slot.
    let mut m1 = fresh(1)?;
    set_sids(&mut m1, 0, 0)?;
    zero_cache(&mut m1)?;
    let (lg_1, _) = m1.forward_rows(&seq, &p_one)?;
    let inert = lg_1[..] == lg_c[..];
    println!("  n_slots=1 reproduces the pre-slot result: {}", if inert { "OK" } else { "MISMATCH" });
    ok &= inert;

    if !ok { anyhow::bail!("NAIVE_SLOTS_FAIL"); }
    println!("NAIVE_SLOTS_OK");
    Ok(())
}
