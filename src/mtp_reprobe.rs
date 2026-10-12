//! S-B9-REL-MTPREPROBE: the per-request MTP profitability policy — WHEN the §6
//! auto-off fires and HOW a lane that went off comes back.
//!
//! The 2026-10 defect class this replaces: the off decision compared a ~16-round
//! cost EMA against `ema_plain` at a 2% margin and then NEVER re-armed (the only
//! writers of `lane.mtp` after an off required it to be `Some` already). Two public
//! user logs showed the trap: a request averaging 3.80 tok/round (MTP ~1.5x faster
//! than plain) hit a local 2.69 tok/round window, beat the single plain sample by
//! 2.3%, and then ran 57-63% of its length at plain speed. The economics rule
//! (AGENTS §6: speculate iff `1+p+p^2+.. > r(d)`) is unchanged — only the estimator,
//! the margin and the ONE-WAY-ness are fixed:
//!
//! * margin 2% -> 10% (staying on at a <=10% loss for a while is cheap; a wrong
//!   off costs the whole multiplier, 30%+ on this workload class);
//! * cost window 1/16 -> 1/64 EMA (spans local troughs; a trough must be SUSTAINED
//!   and deep before the estimate crosses);
//! * a plain baseline that cannot be a single stale sample: the trip asks for a
//!   fresh control round (`confirm`) and compares against `max(ema_plain, fresh)`,
//!   and while a lane is OFF its own plain steps (batch of exactly 1) continuously
//!   refresh a per-request plain EMA at the CURRENT context — plain step time grows
//!   with context, and a context-stale-low baseline biases toward tripping;
//! * re-probe with exponential backoff: after `backoff_base` plain steps the lane
//!   re-arms for a probe window; a failed probe doubles the backoff (capped),
//!   a successful probe resets it. Worst-case probe overhead on a lane that should
//!   stay off is bounded by (probe_window rounds x per-round penalty) / backoff.
//!
//! PURE state machine: no GPU, no clocks, no I/O. Every observation value the
//! scheduler feeds in is the HEAD's number (`tp_round` returns them on every rank),
//! so under TP the nodes mirror the head's decisions bit-for-bit (AGENTS §2.10).

/// The schedule. Defaults are the serving posture; `drill()` is the test-drill
/// posture (tiny gates + tiny backoff so an identity run crosses hundreds of
/// off/re-arm transitions). `--mtp-reprobe=off` keeps NONE of this (the lane goes
/// off exactly as before and never re-arms: the previous behaviour, bit for bit).
#[derive(Clone, Copy, Debug)]
pub struct ReProbeCfg {
    /// trip only if the sustained cost estimate exceeds plain_ref * (1 + margin)
    pub margin: f64,
    /// per-round EMA weight on (ms, toks) — 1/64 spans local troughs
    pub ema_w: f64,
    /// MTP rounds before a first off decision can fire (fresh requests)
    pub min_rounds: usize,
    /// rounds before the cost EMA carries data (graph captures etc. excluded today)
    pub warmup: usize,
    /// probe rounds before a probe verdict may FAIL (lets the post-gap EMAs settle)
    pub probe_warmup: usize,
    /// probe rounds after which the probe counts as SUCCEEDED
    pub probe_window: usize,
    /// plain steps before the first re-probe
    pub backoff_base: usize,
    /// backoff cap (doubled per failed probe)
    pub backoff_cap: usize,
    /// TEST DRILL ONLY: force the off decision to fire as soon as the estimator is
    /// warm (margin ignored) and force every probe to SUCCEED, so an identity run
    /// crosses hundreds of off/re-arm transitions on ANY workload — a workload whose
    /// MTP never crosses the margin would otherwise cross zero transitions.
    pub drill_force: bool,
}

impl Default for ReProbeCfg {
    fn default() -> Self {
        ReProbeCfg {
            margin: 0.10,
            ema_w: 1.0 / 64.0,
            min_rounds: 64,
            warmup: 4,
            probe_warmup: 16,
            probe_window: 32,
            backoff_base: 256,
            backoff_cap: 4096,
            drill_force: false,
        }
    }
}

impl ReProbeCfg {
    /// `--mtp-reprobe-drill`: the same state machine on a shrunken schedule with
    /// the decisions FORCED (off as soon as the estimator is warm, probes always
    /// succeed) — a drill whose transitions depend on the workload crossing the
    /// margin would cross ZERO of them on fast-drafting content, which is exactly
    /// what the identity runs must hammer. Affects only WHEN off/probe transitions
    /// happen — never the greedy token stream (speculation is lossless; the
    /// transitions are draft-only).
    pub fn drill() -> Self {
        ReProbeCfg {
            margin: 0.10,
            ema_w: 1.0 / 8.0,
            min_rounds: 6,
            warmup: 2,
            probe_warmup: 3,
            probe_window: 6,
            backoff_base: 6,
            backoff_cap: 24,
            drill_force: true,
        }
    }
}

/// What the lane should do after one observed MTP round.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RoundVerdict {
    /// keep speculating
    Continue,
    /// sustained cost crossed the margin, but the plain reference may be stale:
    /// run one control round NOW (fresh plain sample at the current context),
    /// then decide on `max(ema_plain, fresh)`.
    ConfirmControl,
    /// turn MTP off for this request (a fresh decision; backoff starts at base)
    Off,
    /// a re-probe FAILED: off again, backoff doubles
    ProbeFailed,
    /// a re-probe SUCCEEDED: stay on (backoff resets to base)
    ProbeSucceeded,
}

/// What the lane should do after one observed plain step while off.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlainVerdict {
    /// stay off, keep counting
    StayOff,
    /// backoff elapsed: re-arm for a probe round window
    ProbeDue,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase {
    /// speculation on (normal §6 guard active)
    Armed,
    /// off; plain steps count toward the next probe
    Off,
    /// on inside a probe window (verdict by probe_warmup/probe_window)
    Probing,
}



/// Per-request policy state. The lane EMAs (`ema_ms`, `ema_tok`, `mtp_rounds`)
/// stay where they are on `Lane`; this owns only the decision machinery.
pub struct ReProbe {
    pub phase: Phase,
    /// MTP rounds observed since the request start / the last re-arm (W1.2): the
    /// min_rounds hold-off counts from the RE-ARM too, so a just-re-armed lane
    /// cannot trip again on fresh-estimator noise for min_rounds rounds.
    pub rounds: usize,
    /// plain steps since the last off
    pub plain_steps: usize,
    /// plain steps to wait before the next probe (base after a fresh off /
    /// success; doubled per failed probe, capped)
    pub backoff: usize,
    /// probe rounds observed so far in the current window
    pub probe_rounds: usize,
    /// a ConfirmControl is pending: the next control round's fresh plain sample
    /// decides (stored by the scheduler via `confirm_sample`)
    pub confirm_pending: bool,
    /// the fresh control-round plain sample gathered for a pending confirm
    pub confirm_plain: f64,
    /// this request's OWN plain ms/tok measured while off (batch-of-1 plain steps,
    /// i.e. at the current context). 0 = no sample yet.
    pub off_plain_ema: f64,
    /// taps_keep row holding the tap that predicted `last_tok`: the row the next
    /// head prime restores from. Set at off time (the round's out.tap_row); row 0
    /// after every primed plain step snapshots it fresh.
    pub prime_tap_row: usize,
    /// observability: [mtp-stats] off_events= reprobes= reprobe_ok=
    pub off_events: usize,
    pub reprobes: usize,
    pub reprobe_ok: usize,
}

/// The scheduler's cost-EMA update for the re-probe policy, as a pure function
/// so the policy tests drive the EXACT production math (W1.1). After an estimator
/// reset (request start, re-arm) the first `window` rounds form a RUNNING MEAN:
/// a one-round SEED decays at only `w` per round, so with w = 1/64 the rounds
/// 16-31 of a probe window still carry 0.79-0.62 of one possibly poisoned round
/// (a graph-capture round, a 1-token round) — enough for a spurious ProbeFailed
/// and a doubled backoff. `captured` rounds never enter the estimate at all.
/// At `window` the running mean hands over to the EMA (same effective memory).
pub fn blend_cost_ema(ema_ms: &mut f64, ema_tok: &mut f64, n: &mut usize, ms: f64, toks: f64,
                      captured: bool, window: usize, w: f64) {
    if captured || ms <= 0.0 || toks <= 0.0 {
        return;
    }
    let window = window.max(1);
    *n += 1;
    if *n == 1 {
        *ema_ms = ms;
        *ema_tok = toks;
    } else if *n <= window {
        *ema_ms += (ms - *ema_ms) / *n as f64;
        *ema_tok += (toks - *ema_tok) / *n as f64;
    } else {
        *ema_ms = (1.0 - w) * *ema_ms + w * ms;
        *ema_tok = (1.0 - w) * *ema_tok + w * toks;
    }
}

impl Default for ReProbe {
    fn default() -> Self {
        ReProbe::new(&ReProbeCfg::default())
    }
}

impl ReProbe {
    pub fn new(cfg: &ReProbeCfg) -> Self {
        ReProbe {
            phase: Phase::Armed,
            rounds: 0,
            plain_steps: 0,
            backoff: cfg.backoff_base,
            probe_rounds: 0,
            confirm_pending: false,
            confirm_plain: 0.0,
            off_plain_ema: 0.0,
            prime_tap_row: 0,
            off_events: 0,
            reprobes: 0,
            reprobe_ok: 0,
        }
    }

    /// The plain reference this request should be compared against: never a single
    /// stale sample. Engine `ema_plain` (control rounds, persists across requests),
    /// raised to the request's own measured plain cost at the current context when
    /// available, and to a pending confirm's fresh sample.
    pub fn plain_reference(&self, ema_plain: f64) -> f64 {
        let mut r = ema_plain;
        if self.off_plain_ema > 0.0 && self.off_plain_ema > r {
            r = self.off_plain_ema;
        }
        if self.confirm_pending && self.confirm_plain > 0.0 && self.confirm_plain > r {
            r = self.confirm_plain;
        }
        r
    }

    /// The trip test (pure): sustained per-token cost vs the plain reference.
    pub fn trips(cost: f64, plain_ref: f64, cfg: &ReProbeCfg) -> bool {
        plain_ref > 0.0 && cost > plain_ref * (1.0 + cfg.margin)
    }

    /// Observe one MTP round of this lane. `cost` = the lane's cost estimate
    /// (ema_ms/ema_tok AFTER this round's EMA update), `ema_plain` = the engine's
    /// plain EMA. The policy counts rounds itself: for a fresh request that is
    /// the lane's cumulative MTP round count, and after a re-arm it restarts at
    /// zero so min_rounds is a hold-off on BOTH (W1.2: report §13's "no off for
    /// 64 rounds after a re-arm" is now real, not inherited from history).
    pub fn observe_round(
        &mut self,
        cost: f64,
        ema_plain: f64,
        cfg: &ReProbeCfg,
    ) -> RoundVerdict {
        match self.phase {
            Phase::Off => RoundVerdict::Continue, // the caller never routes off rounds here
            Phase::Armed => {
                self.rounds += 1;
                if self.rounds < cfg.min_rounds || cost <= 0.0 {
                    self.confirm_pending = false;
                    self.confirm_plain = 0.0;
                    return RoundVerdict::Continue;
                }
                let plain_ref = self.plain_reference(ema_plain);
                if !cfg.drill_force && !Self::trips(cost, plain_ref, cfg) {
                    self.confirm_pending = false;
                    self.confirm_plain = 0.0;
                    return RoundVerdict::Continue;
                }
                if !self.confirm_pending {
                    // crossed the margin — but the reference may be stale: get ONE
                    // fresh plain sample at the current context before deciding.
                    self.confirm_pending = true;
                    self.confirm_plain = 0.0;
                    return RoundVerdict::ConfirmControl;
                }
                // confirm already ran and its sample is folded into plain_reference
                RoundVerdict::Off
            }
            Phase::Probing => {
                self.rounds += 1;
                self.probe_rounds += 1;
                if self.probe_rounds >= cfg.probe_window {
                    // W1.5: unconditional by design — surviving the full window IS
                    // the success criterion; a genuinely unprofitable lane already
                    // failed at probe_warmup on the trips() test below.
                    return RoundVerdict::ProbeSucceeded;
                }
                if !cfg.drill_force && self.probe_rounds >= cfg.probe_warmup && cost > 0.0 {
                    let plain_ref = self.plain_reference(ema_plain);
                    if Self::trips(cost, plain_ref, cfg) {
                        return RoundVerdict::ProbeFailed;
                    }
                }
                RoundVerdict::Continue
            }
        }
    }

    /// The scheduler feeds the fresh control-round plain sample here while a
    /// confirm is pending.
    pub fn confirm_sample(&mut self, plain_ms: f64) {
        if self.confirm_pending {
            self.confirm_plain = plain_ms;
        }
    }

    /// The off decision committed (the scheduler sets `lane.mtp = None` and saves
    /// the round's tap row). `from_probe`: a failed probe doubles the backoff; a
    /// fresh off starts at base.
    pub fn went_off(&mut self, tap_row: usize, from_probe: bool, cfg: &ReProbeCfg) {
        self.phase = Phase::Off;
        self.plain_steps = 0;
        self.probe_rounds = 0;
        self.confirm_pending = false;
        self.confirm_plain = 0.0;
        self.prime_tap_row = tap_row;
        self.off_events += 1;
        self.backoff = if from_probe {
            (self.backoff.saturating_mul(2)).min(cfg.backoff_cap).max(cfg.backoff_base)
        } else {
            cfg.backoff_base
        };
    }

    /// Observe one plain step while off. `plain_ms` = this step's ms when the lane
    /// was ALONE in its plain batch (an exact per-token plain cost at the current
    /// context; 0 = batched with others, not comparable). Counts toward backoff.
    pub fn observe_plain(&mut self, plain_ms: f64, cfg: &ReProbeCfg) -> PlainVerdict {
        if self.phase != Phase::Off {
            return PlainVerdict::StayOff;
        }
        self.plain_steps += 1;
        if plain_ms > 0.0 {
            self.off_plain_ema = if self.off_plain_ema <= 0.0 {
                plain_ms
            } else {
                0.75 * self.off_plain_ema + 0.25 * plain_ms
            };
        }
        if self.plain_steps >= self.backoff {
            PlainVerdict::ProbeDue
        } else {
            PlainVerdict::StayOff
        }
    }

    /// The scheduler re-armed the lane (`mtp = Some(0)` at a control-round-clean
    /// seam). Probe rounds start counting; the cost EMAs were reset by the caller
    /// (together with its post-reset round count `ema_n`). W1.2: `rounds` restarts
    /// too, so the post-re-arm hold-off is min_rounds — same gate as a fresh
    /// request, no new knob.
    pub fn armed_for_probe(&mut self) {
        self.phase = Phase::Probing;
        self.rounds = 0;
        self.probe_rounds = 0;
        self.plain_steps = 0;
        self.reprobes += 1;
    }

    /// A probe succeeded: stay on, reset the backoff ladder. W1.3: the off-period
    /// plain floor is dropped — it was measured at the off's (older) context and
    /// would otherwise keep raising plain_reference and biasing trips toward
    /// staying ON long after the lane is healthy again.
    pub fn probe_ok(&mut self, cfg: &ReProbeCfg) {
        self.phase = Phase::Armed;
        self.probe_rounds = 0;
        self.backoff = cfg.backoff_base;
        self.off_plain_ema = 0.0;
        self.reprobe_ok += 1;
    }

    /// Does this lane need its draft head kept primed on plain steps? Only while
    /// off — an armed/probing lane is primed by the round machinery itself, and a
    /// disabled policy (`--mtp-reprobe=off`) never primes (the previous behaviour).
    pub fn keep_primed(&self) -> bool {
        self.phase == Phase::Off
    }

    /// Is a probe due right now (dispatch-time check)?
    pub fn probe_due(&self, cfg: &ReProbeCfg) -> bool {
        self.phase == Phase::Off && self.plain_steps >= self.backoff.max(cfg.backoff_base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One scheduler EMA step, driving the PRODUCTION math (W1.1: no test mirror
    /// to drift): a running mean over the first `window` post-reset rounds, then
    /// the cfg.ema_w EMA; cost = ema_ms / ema_tok (the ratio of means). `captured`
    /// passes through so capture-poisoning is testable.
    fn step_ema(cfg: &ReProbeCfg, ms: f64, toks: f64, ema_ms: &mut f64, ema_tok: &mut f64, n: &mut usize) -> f64 {
        step_ema_c(cfg, ms, toks, ema_ms, ema_tok, n, false)
    }

    fn step_ema_c(cfg: &ReProbeCfg, ms: f64, toks: f64, ema_ms: &mut f64, ema_tok: &mut f64, n: &mut usize, captured: bool) -> f64 {
        crate::mtp_reprobe::blend_cost_ema(ema_ms, ema_tok, n, ms, toks, captured,
                                           (1.0 / cfg.ema_w) as usize, cfg.ema_w);
        if *ema_tok > 0.0 { *ema_ms / *ema_tok } else { 0.0 }
    }

    #[test]
    fn caseb_replay_does_not_trip() {
        // The public user log (2026-10-05): 250 rounds at 3.80 tok/round,
        // 69.6 ms/round (= 18.3 ms/tok) vs plain 27.22, then a window at
        // 74.8 ms/round / 2.69 tok/round (= 27.8 ms/tok, +2.3% over plain).
        // The old 2%-vs-1/16-EMA rule tripped and the request then ran 63% of
        // its tokens plain; the new estimator must NOT trip, even forever.
        let cfg = ReProbeCfg::default();
        let mut rp = ReProbe::new(&cfg);
        let (mut ema_ms, mut ema_tok, mut n) = (0.0, 0.0, 0usize);
        for _ in 0..250 {
            let cost = step_ema(&cfg, 69.6, 3.80, &mut ema_ms, &mut ema_tok, &mut n);
            assert_eq!(rp.observe_round(cost, 27.22, &cfg), RoundVerdict::Continue);
        }
        for _ in 0..2000 {
            let cost = step_ema(&cfg, 74.8, 2.69, &mut ema_ms, &mut ema_tok, &mut n);
            assert_eq!(rp.observe_round(cost, 27.22, &cfg), RoundVerdict::Continue,
                       "a sustained +2.3% trough must not trip");
        }
    }

    #[test]
    fn second_user_replay_does_not_trip() {
        // Second reporter: acceptance ~49% in a reasoning-heavy stretch, off fired
        // after 161 rounds at 24.58 ms/tok vs plain 23.35 (+5.3%). Must not trip.
        let cfg = ReProbeCfg::default();
        let mut rp = ReProbe::new(&cfg);
        let (mut ema_ms, mut ema_tok, mut n) = (0.0, 0.0, 0usize);
        for _ in 0..2000 {
            let cost = step_ema(&cfg, 49.2, 2.0, &mut ema_ms, &mut ema_tok, &mut n);
            assert_eq!(rp.observe_round(cost, 23.35, &cfg), RoundVerdict::Continue);
        }
    }

    #[test]
    fn noisy_single_plain_sample_cannot_trip() {
        // MTP cost genuinely a bit worse than an old plain sample (cost 30.0 vs
        // ema_plain 22.0, +36%): the policy must demand a fresh confirm sample
        // first; when that fresh sample says plain is now 28+ (context grew), no
        // trip — the old rule would have gone off on the stale 22.
        let cfg = ReProbeCfg::default();
        let mut rp = ReProbe::new(&cfg);
        let (mut ema_ms, mut ema_tok, mut n) = (0.0, 0.0, 0usize);
        let mut saw_confirm = false;
        let mut saw_off = false;
        for _ in 0..400 {
            let cost = step_ema(&cfg, 75.0, 2.5, &mut ema_ms, &mut ema_tok, &mut n);
            match rp.observe_round(cost, 22.0, &cfg) {
                RoundVerdict::ConfirmControl => {
                    saw_confirm = true;
                    rp.confirm_sample(28.0); // the fresh control round says plain = 28
                }
                RoundVerdict::Off => saw_off = true,
                _ => {}
            }
        }
        assert!(saw_confirm);
        assert!(!saw_off, "a stale-low plain baseline must not produce an off");
    }

    #[test]
    fn genuinely_unprofitable_still_turns_off() {
        // acceptance ~0: 74 ms/round for 1 token = 74 ms/tok vs plain 27.
        let cfg = ReProbeCfg::default();
        let mut rp = ReProbe::new(&cfg);
        let (mut ema_ms, mut ema_tok, mut n) = (0.0, 0.0, 0usize);
        let mut saw_off = false;
        for _ in 0..400 {
            let cost = step_ema(&cfg, 74.0, 1.0, &mut ema_ms, &mut ema_tok, &mut n);
            match rp.observe_round(cost, 27.0, &cfg) {
                RoundVerdict::ConfirmControl => rp.confirm_sample(27.0),
                RoundVerdict::Off => { saw_off = true; break; }
                _ => {}
            }
        }
        assert!(saw_off, "a sustained 2.7x cost must still turn off");
    }

    #[test]
    fn trough_then_recovery_rearms_and_stays_on() {
        let cfg = ReProbeCfg::default();
        let mut rp = ReProbe::new(&cfg);
        rp.went_off(2, false, &cfg);
        assert_eq!(rp.phase, Phase::Off);
        assert_eq!(rp.backoff, cfg.backoff_base);
        let mut verdict = PlainVerdict::StayOff;
        for _ in 0..cfg.backoff_base {
            verdict = rp.observe_plain(27.0, &cfg);
        }
        assert_eq!(verdict, PlainVerdict::ProbeDue);
        assert!(rp.off_plain_ema > 26.0 && rp.off_plain_ema < 28.0);
        rp.armed_for_probe();
        assert_eq!(rp.phase, Phase::Probing);
        let mut last = RoundVerdict::Continue;
        for _ in 0..cfg.probe_window {
            last = rp.observe_round(18.0, 27.0, &cfg);
        }
        assert_eq!(last, RoundVerdict::ProbeSucceeded);
        rp.probe_ok(&cfg);
        assert_eq!(rp.phase, Phase::Armed);
        assert_eq!(rp.backoff, cfg.backoff_base);
        assert_eq!(rp.reprobes, 1);
        assert_eq!(rp.reprobe_ok, 1);
    }

    #[test]
    fn pure_bad_stays_off_with_exponential_backoff_and_bounded_overhead() {
        let cfg = ReProbeCfg::default();
        let mut rp = ReProbe::new(&cfg);
        rp.went_off(0, false, &cfg);
        let (mut probe_rounds, mut plain_steps) = (0usize, 0usize);
        for _ in 0..8 {
            while rp.observe_plain(27.0, &cfg) == PlainVerdict::StayOff {}
            plain_steps += rp.plain_steps;
            rp.armed_for_probe(); // the scheduler arms the due probe at dispatch
            let mut v = RoundVerdict::Continue;
            for _ in 0..(cfg.probe_warmup + 1) {
                v = rp.observe_round(74.0, 27.0, &cfg);
            }
            assert_eq!(v, RoundVerdict::ProbeFailed);
            rp.went_off(0, true, &cfg);
            probe_rounds += cfg.probe_warmup + 1;
        }
        assert_eq!(rp.backoff, cfg.backoff_cap);
        let frac = probe_rounds as f64 / (probe_rounds + plain_steps) as f64;
        assert!(frac < 0.02, "probe fraction {frac:.4} must stay bounded (< 2% here)");
    }

    #[test]
    fn drill_schedule_transitions_fast() {
        // The identity drill must FORCE transitions on any workload: armed trips as
        // soon as the estimator is warm (74 >> 27 here, but the margin is ignored),
        // probes SUCCEED at the window end, and the next armed trip follows at once
        // — a ~15-step cycle with 2 transitions, so a 3200-token run crosses ~500.
        let cfg = ReProbeCfg::drill();
        assert!(cfg.drill_force);
        let mut rp = ReProbe::new(&cfg);
        // W1.2: a fresh request is held off for min_rounds (the policy counts its
        // own rounds now), then the forced trip fires (confirm first, then off)
        for _ in 0..(cfg.min_rounds - 1) {
            assert_eq!(rp.observe_round(74.0, 27.0, &cfg), RoundVerdict::Continue);
        }
        assert_eq!(rp.observe_round(74.0, 27.0, &cfg), RoundVerdict::ConfirmControl);
        rp.confirm_sample(27.0);
        assert_eq!(rp.observe_round(74.0, 27.0, &cfg), RoundVerdict::Off);
        rp.went_off(0, false, &cfg);
        let mut v = PlainVerdict::StayOff;
        for _ in 0..6 { v = rp.observe_plain(27.0, &cfg); }
        assert_eq!(v, PlainVerdict::ProbeDue);
        rp.armed_for_probe();
        let mut last = RoundVerdict::Continue;
        for _ in 0..6 { last = rp.observe_round(74.0, 27.0, &cfg); }
        assert_eq!(last, RoundVerdict::ProbeSucceeded);
        rp.probe_ok(&cfg);
        assert_eq!(rp.backoff, 6);
        assert_eq!(rp.reprobe_ok, 1);
        // re-armed (confirm was cleared by went_off): the probe window already
        // counted min_rounds rounds (W1.2), so the forced trip fires again at
        // once, and with the confirm taken the off fires even though 9.9 << 27
        // (drill).
        assert_eq!(rp.observe_round(9.9, 27.0, &cfg), RoundVerdict::ConfirmControl);
        rp.confirm_sample(27.0);
        assert_eq!(rp.observe_round(9.9, 27.0, &cfg), RoundVerdict::Off);
        // and the cycle density: 1000 simulated steps of a cycling lane cross >100 offs
        let mut rp2 = ReProbe::new(&cfg);
        let (mut ms, mut tok, mut rounds) = (0.0, 0.0, 0usize);
        let mut transitions = 0usize;
        let mut pending_confirm = false;
        for _ in 0..1000 {
            match rp2.phase {
                Phase::Off => {
                    if rp2.observe_plain(27.0, &cfg) == PlainVerdict::ProbeDue {
                        rp2.armed_for_probe();
                        transitions += 1;
                    }
                }
                Phase::Armed | Phase::Probing => {
                    let c = step_ema(&cfg, 20.0, 3.0, &mut ms, &mut tok, &mut rounds);
                    match rp2.observe_round(c, 27.0, &cfg) {
                        RoundVerdict::Off => { rp2.went_off(0, false, &cfg); transitions += 1; }
                        RoundVerdict::ProbeSucceeded => { rp2.probe_ok(&cfg); transitions += 1; }
                        RoundVerdict::ProbeFailed => { rp2.went_off(0, true, &cfg); transitions += 1; }
                        RoundVerdict::ConfirmControl => { pending_confirm = true; rp2.confirm_sample(27.0); }
                        RoundVerdict::Continue => {}
                    }
                    let _ = pending_confirm;
                }
            }
        }
        assert!(transitions > 100, "drill cycle too sparse: {transitions} transitions / 1000 steps");
    }

    #[test]
    fn full_lifecycle_through_scheduler_entry_points() {
        // W2: one end-to-end drive of the policy through the exact entry points the
        // scheduler calls (observe_round / confirm_sample / went_off / observe_plain
        // / armed_for_probe / probe_ok), with the cost estimate produced by the
        // PRODUCTION blend (blend_cost_ema via step_ema_c). Covers: healthy ->
        // trough -> confirm -> off -> backoff -> re-arm -> a capture-poisoned probe
        // that SUCCEEDS (W1.1) -> the post-re-arm hold-off (W1.2) -> off -> a probe
        // that FAILS on genuinely bad cost -> doubled backoff; W1.3's off_plain_ema
        // clear is asserted at the success seam.
        let cfg = ReProbeCfg::default();
        let mut rp = ReProbe::new(&cfg);
        fn step(cfg: &ReProbeCfg, ms: f64, toks: f64, captured: bool, ema_plain: f64,
                rp: &mut ReProbe, ema_ms: &mut f64, ema_tok: &mut f64, n: &mut usize) -> RoundVerdict {
            let cost = step_ema_c(cfg, ms, toks, ema_ms, ema_tok, n, captured);
            rp.observe_round(cost, ema_plain, cfg)
        }
        let (mut ema_ms, mut ema_tok, mut n) = (0.0, 0.0, 0usize);
        // healthy: 100 rounds at ~18.3 ms/tok vs plain 27 - never trips
        for _ in 0..100 {
            assert_eq!(step(&cfg, 69.6, 3.80, false, 27.0, &mut rp, &mut ema_ms, &mut ema_tok, &mut n),
                       RoundVerdict::Continue);
        }
        // the trough collapses (acceptance ~0): the crossing demands a fresh confirm
        let mut v = RoundVerdict::Continue;
        for _ in 0..200 {
            v = step(&cfg, 74.0, 1.0, false, 27.0, &mut rp, &mut ema_ms, &mut ema_tok, &mut n);
            if v != RoundVerdict::Continue { break; }
        }
        assert_eq!(v, RoundVerdict::ConfirmControl);
        rp.confirm_sample(27.0); // the fresh control round: plain really is 27
        assert_eq!(step(&cfg, 74.0, 1.0, false, 27.0, &mut rp, &mut ema_ms, &mut ema_tok, &mut n),
                   RoundVerdict::Off);
        rp.went_off(2, false, &cfg);
        assert_eq!(rp.backoff, cfg.backoff_base);
        assert_eq!(rp.phase, Phase::Off);
        // off: plain steps count toward the backoff and feed off_plain_ema
        let mut pv = PlainVerdict::StayOff;
        for _ in 0..cfg.backoff_base { pv = rp.observe_plain(27.0, &cfg); }
        assert_eq!(pv, PlainVerdict::ProbeDue);
        assert!(rp.off_plain_ema > 26.0 && rp.off_plain_ema < 28.0);
        // re-arm: the scheduler resets the estimator at the same seam
        rp.armed_for_probe();
        assert_eq!(rp.phase, Phase::Probing);
        (ema_ms, ema_tok, n) = (0.0, 0.0, 0usize);
        // W1.1: probe round 1 captures a graph (500 ms), round 2 is a 1-token round
        // at 74 ms/tok. Under a one-round EMA seed that poison survives into the
        // probe-fail window (weight 0.79-0.62 at rounds 16-31, plain_ref 27) and
        // fails the probe; with the running mean it is 1-2 of <=64 samples and the
        // probe SUCCEEDS on healthy data.
        assert_eq!(step(&cfg, 500.0, 5.0, true, 27.0, &mut rp, &mut ema_ms, &mut ema_tok, &mut n),
                   RoundVerdict::Continue);
        assert_eq!(step(&cfg, 74.0, 1.0, false, 27.0, &mut rp, &mut ema_ms, &mut ema_tok, &mut n),
                   RoundVerdict::Continue);
        let mut last = RoundVerdict::Continue;
        for _ in 0..(cfg.probe_window - 2) {
            last = step(&cfg, 69.6, 3.80, false, 27.0, &mut rp, &mut ema_ms, &mut ema_tok, &mut n);
            assert_ne!(last, RoundVerdict::ProbeFailed, "capture poison must not fail the probe");
        }
        assert_eq!(last, RoundVerdict::ProbeSucceeded);
        rp.probe_ok(&cfg);
        assert_eq!(rp.phase, Phase::Armed);
        assert_eq!(rp.off_plain_ema, 0.0, "W1.3: the off-period plain floor must clear on success");
        assert_eq!(rp.backoff, cfg.backoff_base);
        // W1.2: rounds restarted at the re-arm; the probe window spent 32 of
        // min_rounds=64, so a poisoned post-success stretch is held off for the
        // remainder and only then crosses (confirm first, then off).
        for _ in 0..(cfg.min_rounds - cfg.probe_window - 1) {
            assert_eq!(step(&cfg, 74.0, 1.0, false, 27.0, &mut rp, &mut ema_ms, &mut ema_tok, &mut n),
                       RoundVerdict::Continue, "W1.2: post-re-arm hold-off");
        }
        assert_eq!(step(&cfg, 74.0, 1.0, false, 27.0, &mut rp, &mut ema_ms, &mut ema_tok, &mut n),
                   RoundVerdict::ConfirmControl);
        rp.confirm_sample(27.0);
        assert_eq!(step(&cfg, 74.0, 1.0, false, 27.0, &mut rp, &mut ema_ms, &mut ema_tok, &mut n),
                   RoundVerdict::Off);
        rp.went_off(0, false, &cfg);
        // second cycle: the probe now FAILS on genuinely bad cost -> backoff doubles
        let mut pv = PlainVerdict::StayOff;
        while pv != PlainVerdict::ProbeDue { pv = rp.observe_plain(27.0, &cfg); }
        rp.armed_for_probe();
        (ema_ms, ema_tok, n) = (0.0, 0.0, 0usize);
        for _ in 0..(cfg.probe_warmup - 1) {
            assert_eq!(step(&cfg, 74.0, 1.0, false, 27.0, &mut rp, &mut ema_ms, &mut ema_tok, &mut n),
                       RoundVerdict::Continue, "a probe fails only past its warmup");
        }
        let mut v = RoundVerdict::Continue;
        for _ in 0..8 {
            v = step(&cfg, 74.0, 1.0, false, 27.0, &mut rp, &mut ema_ms, &mut ema_tok, &mut n);
            if v == RoundVerdict::ProbeFailed { break; }
        }
        assert_eq!(v, RoundVerdict::ProbeFailed, "a sustained 74 vs plain 27 must fail the probe");
        rp.went_off(0, true, &cfg);
        assert_eq!(rp.backoff, (cfg.backoff_base * 2).min(cfg.backoff_cap));
        // three off events: two armed trips + the failed probe; two probe windows opened
        assert_eq!(rp.off_events, 3);
        assert_eq!(rp.reprobes, 2);
        assert_eq!(rp.reprobe_ok, 1);
    }

    #[test]
    fn off_plain_ema_raises_the_reference() {
        // Context grew: the request's own plain steps now cost 31 while the engine
        // ema_plain is a stale 27. The reference must follow (no stale-low trips).
        let cfg = ReProbeCfg::default();
        let mut rp = ReProbe::new(&cfg);
        rp.went_off(0, false, &cfg);
        for _ in 0..32 {
            rp.observe_plain(31.0, &cfg);
        }
        assert!(rp.off_plain_ema > 30.0);
        let ref_ = rp.plain_reference(27.0);
        assert!(ref_ > 30.0);
        assert!(!ReProbe::trips(30.0, ref_, &cfg));
        assert!(ReProbe::trips(74.0, ref_, &cfg));
    }
}
