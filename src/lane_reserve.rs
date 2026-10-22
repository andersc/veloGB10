//! Lane reservation for the EXL3 TP=1 server (`--lane-reserve`): keep an agent's prefix cache
//! when conversations outnumber lanes.
//!
//! Problem and design: andrei.gnezdilov's report (2026-10-14, `contrib/alexei/LANE_RESERVATION.md`,
//! session `V074-LANESRES1` per `PLAN/LANERES1_PROMPT.md`). The prefix cache is per lane; when one
//! conversation too many is alive, today's admission (`admit_req`'s `pick`) hands every free lane to
//! whoever asks first, so one excess conversation evicts every conversation in turn (~45% cached,
//! TTFT p90 minutes). The fix is in ADMISSION, not in the victim choice: (1) a waiting request whose
//! prompt-end snapshot sits in a free lane is admitted first, into that lane; (2) a lane whose
//! request finished normally less than R seconds ago is reserved for that conversation; (3) the
//! queue head may steal a reserved lane after waiting P seconds, at most once per S seconds.
//!
//! Every decision is a pure function over plain views, unit-tested here without a GPU. The module
//! imports only std, so `rustc --edition 2021 --test src/lane_reserve.rs` also runs it standalone.
//!
//! With `reserve == 0` (the default) the scheduler does not call [`plan`] at all — admission stays
//! byte-for-byte today's FIFO.

use std::time::{Duration, Instant};

/// The flags: `--lane-reserve R --lane-reserve-patience P --lane-reserve-swap S`.
#[derive(Clone, Copy, Debug)]
pub struct Cfg {
    /// R: keep a lane for its conversation this long after a NORMAL finish (0 = off = today).
    pub reserve: Duration,
    /// P: a waiting request may take a reserved lane after waiting this long.
    pub patience: Duration,
    /// S: at most one forced take (steal) per this many seconds; 0 = no rate limit.
    pub swap: Duration,
}

impl Cfg {
    pub fn off() -> Cfg {
        Cfg { reserve: Duration::ZERO, patience: Duration::ZERO, swap: Duration::ZERO }
    }
    pub fn on(reserve_s: u64, patience_s: u64, swap_s: u64) -> Cfg {
        Cfg {
            reserve: Duration::from_secs(reserve_s),
            patience: Duration::from_secs(patience_s),
            swap: Duration::from_secs(swap_s),
        }
    }
}

/// What the scheduler knows about one lane/slot right now.
#[derive(Clone, Copy, Debug)]
pub struct SlotView {
    /// A request is running on this lane.
    pub busy: bool,
    /// The lane holds a prompt-end snapshot (`cache[slot].is_some()`).
    pub has_snapshot: bool,
    /// When the lane's last request released it (None = never ran).
    pub released_at: Option<Instant>,
    /// That release was a normal finish (stop / length / tool call / loop-detected), not a
    /// cancel, an error or a failed prefill.
    pub normal_finish: bool,
}

/// One waiting request. `arrived_at` is the request's `received_at` (stamped when the server
/// accepted it, before the queue): the patience clock is the client's wait, not the scheduler's.
/// `own_slot` is the prompt-end hit test of `admit_req`'s pick block, evaluated over FREE lanes
/// only (`t.len() < prompt.len() && prompt[..t.len()] == t`); None = no free lane holds it.
#[derive(Clone, Copy, Debug)]
pub struct WaitView {
    pub arrived_at: Instant,
    pub own_slot: Option<usize>,
}

/// Which waiting request may be admitted, and the lanes `admit_req` may pick from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Rule 1: the request's own prompt-end snapshot sits in this free lane.
    Own,
    /// Rule 2: unreserved free lanes exist; the queue head takes them.
    Unreserved,
    /// Rule 3: a forced take — the queue head waited ≥ P and stole a reserved lane.
    Forced,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Plan {
    Admit { waiting_index: usize, allowed: Vec<usize>, kind: Kind },
    /// None = until a request or a lane changes (no computable deadline).
    Wait { until: Option<Instant> },
}

/// Rate-limiter state for forced takes (rule 3). One per scheduler.
#[derive(Default)]
pub struct State {
    pub next_forced: Option<Instant>,
}

impl State {
    pub fn new() -> State {
        State { next_forced: None }
    }
}

/// Is this slot reserved for its conversation right now? A reservation needs a normal finish
/// within R that left a snapshot on a free lane. Cancels, errors and failed prefills never
/// reserve (their `normal_finish` is false, and a failed prefill clears the snapshot).
pub fn reserved(cfg: &Cfg, sv: &SlotView, now: Instant) -> bool {
    !cfg.reserve.is_zero()
        && !sv.busy
        && sv.normal_finish
        && sv.has_snapshot
        && sv.released_at.map_or(false, |t| now.duration_since(t) < cfg.reserve)
}

/// The admission decision (`Exl3Scheduler::run`, reservation active). Order:
/// (1) the OLDEST waiting request whose own snapshot sits in a free lane → `Admit { [own] }`
///     (it jumps the queue: admitting it is nearly free, anything else destroys its cache);
/// (2) unreserved free lanes exist → admit the queue head with them;
/// (3) the head waited ≥ P and the last forced take was ≥ S ago → admit it with ALL free lanes
///     and arm the limiter;
/// (4) otherwise wait until the earliest of the next reservation expiry, the head's patience
///     expiry and the limiter.
///
/// With `reserve == 0` this is exactly FIFO: the head with all free lanes (rule 2 with nothing
/// reserved), though the scheduler does not call `plan` at all in that mode.
pub fn plan(cfg: &Cfg, st: &mut State, now: Instant, slots: &[SlotView], waiting: &[WaitView]) -> Plan {
    let free: Vec<usize> = (0..slots.len()).filter(|&i| !slots[i].busy).collect();
    let Some(_head) = waiting.first() else {
        return Plan::Wait { until: None };
    };
    // (1) own snapshot first: the oldest waiting request that has one in a FREE lane.
    if !cfg.reserve.is_zero() {
        if let Some((idx, own)) = waiting
            .iter()
            .enumerate()
            .filter(|(_, w)| w.own_slot.map_or(false, |s| free.contains(&s)))
            .min_by_key(|(_, w)| w.arrived_at)
            .map(|(i, w)| (i, w.own_slot.unwrap()))
        {
            return Plan::Admit { waiting_index: idx, allowed: vec![own], kind: Kind::Own };
        }
    }
    // (2) unreserved free lanes: the head takes them (with reserve == 0 this is all free lanes,
    // i.e. today's FIFO).
    let unreserved: Vec<usize> = free.iter().copied().filter(|&i| !reserved(cfg, &slots[i], now)).collect();
    if !unreserved.is_empty() {
        return Plan::Admit { waiting_index: 0, allowed: unreserved, kind: Kind::Unreserved };
    }
    // (3) bounded stealing. From here on every free lane (if any) is reserved.
    let head_waited = now.duration_since(waiting[0].arrived_at);
    if !free.is_empty()
        && head_waited >= cfg.patience
        && st.next_forced.map_or(true, |t| now >= t)
    {
        st.next_forced = if cfg.swap.is_zero() { None } else { Some(now + cfg.swap) };
        return Plan::Admit { waiting_index: 0, allowed: free, kind: Kind::Forced };
    }
    // (4) wait for the earliest deadline that can change the answer. A deadline already past is
    // not a deadline — there is nothing left to wait for, and `until == None` tells the caller
    // that only a new request or a lane change can alter the answer.
    let mut until: Option<Instant> = None;
    let mut merge = |t: Option<Instant>| {
        if let Some(t) = t.filter(|&t| t > now) {
            until = Some(match until {
                Some(u) => u.min(t),
                None => t,
            });
        }
    };
    for (i, sv) in slots.iter().enumerate() {
        if free.contains(&i) && reserved(cfg, sv, now) {
            merge(sv.released_at.map(|t| t + cfg.reserve)); // next reservation expiry
        }
    }
    merge(Some(waiting[0].arrived_at + cfg.patience)); // the head's patience expiry
    merge(st.next_forced); // the limiter
    Plan::Wait { until }
}

/// The owner-change victim pick among the allowed FREE lanes (reservation ON only): an empty
/// lane (nothing cached to destroy), else the deepest WP16 checkpoint among the allowed lanes
/// (ties: the highest slot, as today's `(q, i)` max), else the least recently released lane.
/// Replaces `free.first()` (slot 0), which is what let one conversation evict every other.
/// Returns the allowed index unchanged when the caller's view is empty (the caller turns that
/// into the usual "server busy" finish).
pub fn victim(allowed: &[usize], slots: &[SlotView], wp16_depth: &[usize]) -> Option<usize> {
    let first = *allowed.first()?;
    // an empty lane first: no snapshot of anyone to evict
    if let Some(&i) = allowed.iter().find(|&&i| !slots[i].has_snapshot) {
        return Some(i);
    }
    // else the deepest WP16 resume among the allowed lanes only (0 = no checkpoint there)
    if let Some(i) = allowed
        .iter()
        .copied()
        .filter(|&i| wp16_depth.get(i).copied().unwrap_or(0) > 0)
        .max_by_key(|&i| (wp16_depth[i], i))
    {
        return Some(i);
    }
    // else the least recently released lane (a lane that never ran counts as the oldest)
    Some(allowed.iter().copied().min_by_key(|&i| slots[i].released_at).unwrap_or(first))
}

/// The finish-reason classification shared by every lane-release site: normal = a completed
/// turn (stop token, max length, a stop string, a tool call, loop-detected) — not a cancel,
/// not an error. Only normal releases are eligible for a reservation.
pub fn normal_finish(reason: &str) -> bool {
    reason != "cancelled" && !reason.starts_with("error")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const R: u64 = 20;
    const P: u64 = 150;
    const S: u64 = 60;

    fn cfg() -> Cfg {
        Cfg::on(R, P, S)
    }
    fn t0() -> Instant {
        // Instant sef-domains per process; tests only need A moment to diff against.
        Instant::now()
    }
    /// A slot table of n lanes, all idle, no snapshots, never released.
    fn slots_idle(n: usize) -> Vec<SlotView> {
        (0..n)
            .map(|_| SlotView { busy: false, has_snapshot: false, released_at: None, normal_finish: false })
            .collect()
    }
    fn waiting_at(now: Instant, dts: &[u64], owns: &[Option<usize>]) -> Vec<WaitView> {
        dts.iter()
            .zip(owns)
            .map(|(&dt, &own)| WaitView { arrived_at: now - Duration::from_secs(dt), own_slot: own })
            .collect()
    }

    // 1. reserve == 0 → always FIFO: the head with all free lanes, even with snapshots around.
    #[test]
    fn reserve_zero_is_fifo() {
        let now = t0();
        let cfg = Cfg::off();
        let mut st = State::new();
        let mut slots = slots_idle(3);
        slots[2].has_snapshot = true;
        slots[2].released_at = Some(now);
        slots[2].normal_finish = true;
        let w = waiting_at(now, &[0, 10], &[Some(2), None]);
        for _ in 0..3 {
            match plan(&cfg, &mut st, now, &slots, &w) {
                Plan::Admit { waiting_index: 0, allowed, kind: Kind::Unreserved } => {
                    assert_eq!(allowed, vec![0, 1, 2])
                }
                other => panic!("expected FIFO head with all free lanes, got {other:?}"),
            }
        }
    }

    // 2. Own snapshot jumps the queue: an older request without an own slot waits behind a
    //    younger one that has it; and the own admit is scoped to exactly that lane.
    #[test]
    fn own_snapshot_jumps_the_queue() {
        let now = t0();
        let cfg = cfg();
        let mut st = State::new();
        let mut slots = slots_idle(3);
        slots[1].has_snapshot = true; // the younger request's snapshot lives here
        let w = waiting_at(now, &[120, 1], &[None, Some(1)]);
        match plan(&cfg, &mut st, now, &slots, &w) {
            Plan::Admit { waiting_index: 1, allowed, kind: Kind::Own } => assert_eq!(allowed, vec![1]),
            other => panic!("expected the younger own-snapshot request, got {other:?}"),
        }
        // without a snapshot on any lane the same load falls back to the head (rule 2). The
        // WaitViews are the caller's view — rebuilt like the scheduler does each pass, so the
        // stale own_slot cannot leak (in production own_snapshot_slot recomputes from the cache).
        slots[1].has_snapshot = false;
        let w2 = waiting_at(now, &[120, 1], &[None, None]);
        match plan(&cfg, &mut st, now, &slots, &w2) {
            Plan::Admit { waiting_index: 0, kind: Kind::Unreserved, .. } => {}
            other => panic!("expected the head via rule 2, got {other:?}"),
        }
    }

    // 3. A reserved slot is not handed to a stranger before P; it is after P (rate-limited).
    #[test]
    fn reserved_lane_holds_until_patience() {
        let now = t0();
        // R big enough that the reservation outlives every clock jump in this test.
        let cfg = Cfg::on(300, 150, 60);
        let mut st = State::new();
        let mut slots = slots_idle(1); // the ONLY lane is reserved: the head must wait
        slots[0].has_snapshot = true;
        slots[0].released_at = Some(now - Duration::from_secs(5));
        slots[0].normal_finish = true;
        let w = waiting_at(now, &[10], &[None]); // the head waited 10 s < P=150
        match plan(&cfg, &mut st, now, &slots, &w) {
            Plan::Wait { .. } => {}
            other => panic!("head must wait while the only lane is reserved, got {other:?}"),
        }
        // after P the head may steal — the first time (the lane is still reserved: fresh stamp)
        let now2 = now + Duration::from_secs(160);
        let w2 = waiting_at(now2, &[160], &[None]);
        slots[0].released_at = Some(now2 - Duration::from_secs(10));
        match plan(&cfg, &mut st, now2, &slots, &w2) {
            Plan::Admit { waiting_index: 0, allowed, kind: Kind::Forced } => assert_eq!(allowed, vec![0]),
            other => panic!("expected a forced take after P, got {other:?}"),
        }
        // but the limiter blocks the next one within S
        let now3 = now2 + Duration::from_secs(1);
        slots[0].released_at = Some(now3 - Duration::from_secs(10));
        match plan(&cfg, &mut st, now3, &slots, &w2) {
            Plan::Wait { .. } => {}
            other => panic!("second forced take within S must wait, got {other:?}"),
        }
        // 4. ...and goes after S.
        let now4 = now2 + cfg.swap;
        slots[0].released_at = Some(now4 - Duration::from_secs(10));
        match plan(&cfg, &mut st, now4, &slots, &w2) {
            Plan::Admit { kind: Kind::Forced, .. } => {}
            other => panic!("forced take must go after S, got {other:?}"),
        }
    }

    // 5. A reservation expires after R: the slot becomes unreserved (rule 2 serves the head).
    #[test]
    fn reservation_expires_after_r() {
        let now = t0();
        let cfg = cfg();
        let mut st = State::new();
        let mut slots = slots_idle(1);
        slots[0].has_snapshot = true;
        slots[0].normal_finish = true;
        slots[0].released_at = Some(now - Duration::from_secs(R + 1));
        let w = waiting_at(now, &[1], &[None]);
        match plan(&cfg, &mut st, now, &slots, &w) {
            Plan::Admit { waiting_index: 0, allowed, kind: Kind::Unreserved } => assert_eq!(allowed, vec![0]),
            other => panic!("expired reservation must leave the lane unreserved, got {other:?}"),
        }
    }

    // 6. Cancelled and error releases are never reserved (only a normal finish is).
    #[test]
    fn cancel_and_error_never_reserve() {
        let now = t0();
        let cfg = cfg();
        let mut st = State::new();
        for normal in [false, false] {
            let mut slots = slots_idle(1);
            slots[0].has_snapshot = true;
            slots[0].normal_finish = normal;
            slots[0].released_at = Some(now - Duration::from_secs(1));
            let w = waiting_at(now, &[0], &[None]);
            match plan(&cfg, &mut st, now, &slots, &w) {
                Plan::Admit { kind: Kind::Unreserved, allowed, .. } => assert_eq!(allowed, vec![0]),
                other => panic!("a {normal}-finish release must not reserve, got {other:?}"),
            }
        }
        // a snapshot-less release is not a reservation either (failed prefill cleared it)
        let mut slots = slots_idle(1);
        slots[0].has_snapshot = false;
        slots[0].normal_finish = true;
        slots[0].released_at = Some(now - Duration::from_secs(1));
        let w = waiting_at(now, &[0], &[None]);
        match plan(&cfg, &mut st, now, &slots, &w) {
            Plan::Admit { kind: Kind::Unreserved, .. } => {}
            other => panic!("a snapshot-less release must not reserve, got {other:?}"),
        }
        // and the reason classification agrees with the release sites' reasons
        assert!(normal_finish("stop"));
        assert!(normal_finish("length"));
        assert!(normal_finish("loop_detected"));
        assert!(!normal_finish("cancelled"));
        assert!(!normal_finish("error: prefill failed: boom"));
    }

    // 7. victim: empty slot > deepest WP16 among allowed > least recently released; ties on WP16
    //    depth go to the higher slot (today's (q, i) max); a zero depth never wins the WP16 arm.
    #[test]
    fn victim_order() {
        let now = t0();
        let mut slots = slots_idle(4);
        let no_wp16 = [0usize; 4];
        // empty lanes win over everything else
        slots[2].has_snapshot = true;
        slots[3].has_snapshot = true;
        assert_eq!(victim(&[0, 1, 2, 3], &slots, &no_wp16), Some(0));
        // else the deepest WP16 among allowed (only where a checkpoint exists)
        let mut slots2 = slots_idle(4);
        for s in 1..4 {
            slots2[s].has_snapshot = true;
        }
        let depth = [0usize, 4096, 24570, 8192];
        assert_eq!(victim(&[1, 2, 3], &slots2, &depth), Some(2));
        // WP16 tie on depth → the higher slot, like today's max on (q, i)
        let tie = [0usize, 4096, 4096, 0];
        assert_eq!(victim(&[1, 2], &slots2, &tie), Some(2));
        // all-zero depths never take the WP16 arm: least recently released instead
        slots2[1].released_at = Some(now - Duration::from_secs(50));
        slots2[2].released_at = Some(now - Duration::from_secs(5));
        slots2[3].released_at = Some(now - Duration::from_secs(500));
        assert_eq!(victim(&[1, 2, 3], &slots2, &no_wp16), Some(3));
        // a lane never released counts as the oldest
        slots2[3].released_at = None;
        assert_eq!(victim(&[1, 2, 3], &slots2, &no_wp16), Some(3));
        // never slot 0 by default: the answer comes from the allowed set alone
        assert_eq!(victim(&[1], &slots2, &no_wp16), Some(1));
        // empty allowed → None (the caller finishes the request with "server busy")
        assert_eq!(victim(&[], &slots2, &no_wp16), None);
    }

    // 8. Wait.until is the earliest of the three deadlines: next reservation expiry, the head's
    //    patience expiry, the forced-take limiter.
    #[test]
    fn wait_until_is_the_earliest_deadline() {
        let now = t0();
        let cfg = cfg();
        let mut st = State::new();
        let mut slots = slots_idle(1); // the only lane is reserved
        slots[0].has_snapshot = true;
        slots[0].normal_finish = true;
        slots[0].released_at = Some(now - Duration::from_secs(10)); // expires at +10
        let w = waiting_at(now, &[5], &[None]); // patience at +145
        match plan(&cfg, &mut st, now, &slots, &w) {
            Plan::Wait { until } => {
                let d = until.expect("a reserved lane gives a deadline");
                assert!((d - now) <= Duration::from_secs(10) + Duration::from_millis(1));
                assert!((d - now) >= Duration::from_secs(9));
            }
            other => panic!("expected Wait, got {other:?}"),
        }
        // arm the limiter; make it the earliest deadline
        st.next_forced = Some(now + Duration::from_secs(3));
        match plan(&cfg, &mut st, now, &slots, &w) {
            Plan::Wait { until } => {
                let d = until.expect("the limiter gives a deadline");
                assert!((d - now) <= Duration::from_secs(3) + Duration::from_millis(1));
            }
            other => panic!("expected Wait on the limiter, got {other:?}"),
        }
        // no deadline at all: every lane busy, patience long gone, no limiter
        let mut st2 = State::new();
        let mut busy = slots_idle(1);
        busy[0].busy = true;
        let w2 = waiting_at(now, &[10_000], &[None]);
        match plan(&cfg, &mut st2, now, &busy, &w2) {
            Plan::Wait { until: None } => {}
            other => panic!("expected an open-ended Wait, got {other:?}"),
        }
    }

    // Bonus: rule 2 serves the queue head — not an older own-snapshot request whose lane is
    // BUSY (own requires a FREE lane) — and rule 1's own lane is admissible while it is still
    // reserved for its owner (it IS the owner).
    #[test]
    fn own_lane_busy_falls_back_and_own_reserved_lane_serves() {
        let now = t0();
        let cfg = cfg();
        let mut st = State::new();
        let mut slots = slots_idle(2);
        slots[1].busy = true; // the snapshot lives on a BUSY lane: not "own" for admission
        slots[1].has_snapshot = true;
        let w = waiting_at(now, &[30, 1], &[None, Some(1)]);
        match plan(&cfg, &mut st, now, &slots, &w) {
            Plan::Admit { waiting_index: 0, allowed, kind: Kind::Unreserved } => assert_eq!(allowed, vec![0]),
            other => panic!("a busy own lane must fall back to the head, got {other:?}"),
        }
        // the owner of a reserved lane gets it back immediately (rule 1, even though reserved)
        let mut slots2 = slots_idle(1);
        slots2[0].has_snapshot = true;
        slots2[0].normal_finish = true;
        slots2[0].released_at = Some(now - Duration::from_secs(2));
        let w2 = waiting_at(now, &[0], &[Some(0)]);
        match plan(&cfg, &mut st, now, &slots2, &w2) {
            Plan::Admit { waiting_index: 0, allowed, kind: Kind::Own } => assert_eq!(allowed, vec![0]),
            other => panic!("the owner must get its reserved lane back, got {other:?}"),
        }
    }
}
