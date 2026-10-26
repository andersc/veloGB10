//! SSD spill tier for evicted lane snapshots (`--kv-spill`), LANERES2.
//!
//! Problem (owner directive 2026-10-15, after LANERES1): the reservation policy keeps a
//! conversation's prefix cache while it waits, but when conversations outnumber lanes a HARD
//! eviction still happens — the queue head out-waits `--lane-reserve-patience` and takes a
//! reserved lane, or an unreserved victim is chosen. Today the victim's snapshot is dropped and
//! its next request re-prefills from scratch. Andrei's simulator (extended for this session,
//! `contrib/alexei/slot_policy_sim.py --kv-spill-gb`) prices that residual at 15–36% of trace
//! wall time across the agent-load cells even with reservation ON — far above the ~5% gate.
//!
//! Design: when a snapshot is evicted, its bytes are saved to NVMe by a background writer
//! thread (never in the serving round, never on the compute stream); when the conversation's
//! next request arrives and its snapshot is NOT in any free lane, the store is consulted: if
//! `bytes / nvme_read_BW` beats `tokens / prefill_rate` the snapshot is restored (bit-exact
//! through the engine's per-slot checkpoint path), else it is dropped and the re-prefill
//! counter incremented. The store has a byte cap with LRU shift-out. A miss is never an error:
//! the admission path falls back to today's re-prefill (graceful, no refusals).
//!
//! This module holds the pure policy (cap, LRU, cost model) and the disk store façade, imports
//! only std, and is unit-tested here without a GPU — `rustc --edition 2021 --test src/kv_spill.rs`
//! runs standalone, same as `lane_reserve.rs`. The engine hooks live in `exl3_serve.rs`
//! (capture at the take-over site, restore in the admission block); TP ranks each write their
//! own `rank<N>` files, so TP=1/2/4 need no policy change.

use std::collections::VecDeque;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::Arc;
use std::thread;

/// ~0.5 GiB of KV + states per 38K prompt tokens (Flash-Next 27B EXL3 4.05bpw, TP=1 shard;
/// the sim's calibration). A per-token figure read from the model would be better; this is the
/// documented default and `--kv-spill-gb` sizing uses it.
pub const KV_BYTES_PER_TOKEN: u64 = 14_300;

/// .13-measured restore-read bandwidth (O_DIRECT 2-GiB dd, 2026-10-16): ~5.1 GB/s. The cost
/// model is hardware-class, not per-boot; the replay cells re-measure it if it moves.
pub const NVME_READ_BPS: u64 = 5_100_000_000;
/// The admission prefill rate class the cost model compares the read against (Flash-Next 27B
/// EXL3, TP=1, .13, agent trace): tok/s. A restore wins when bytes/BPS < tokens/TPS.
pub const PREFILL_TPS: f64 = 1600.0;

/// File magic: "GB10KSP1" (GB10 KV-spill, format 1).
const MAGIC: &[u8; 8] = b"GB10KSP1";
/// Header: magic(8) + version u32 + tokens u64 + payload_len u64 + crc u64 = 36 bytes.
const HEADER_LEN: u64 = 8 + 4 + 8 + 8 + 8;

/// FNV-1a 64: payload integrity without pulling in a crate (std-only module).
fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The flags: `--kv-spill <auto|off|PATH> --kv-spill-gb N`.
#[derive(Clone, Debug)]
pub struct Cfg {
    /// false = the tier is off: no capture, no restore, counters stay zero (today's behavior).
    pub on: bool,
    /// Spill directory. `auto` resolves to `/tmp/gb10_kv_spill_<port>`; PATH is used verbatim.
    pub dir: PathBuf,
    /// Store cap in bytes (`--kv-spill-gb`, default 64).
    pub cap_bytes: u64,
    /// NVMe read rate used by the restore cost model (bytes/s; boot-measured class, dd anchor).
    pub nvme_read_bps: u64,
    /// Prefill rate used by the restore cost model (tokens/s).
    pub prefill_tps: f64,
}

impl Cfg {
    pub fn off() -> Self {
        Cfg { on: false, dir: PathBuf::from("/tmp/gb10_kv_spill"), cap_bytes: 0, nvme_read_bps: 5_100_000_000, prefill_tps: 1600.0 }
    }
}

/// The LRU byte-cap policy over the store (pure; the files live in [`Store`]).
#[derive(Debug, Default)]
pub struct Policy {
    cap_bytes: u64,
    bytes: u64,
    /// (conv key, bytes, tokens), front = least recently put.
    lru: VecDeque<(u64, u64, u64)>,
}

impl Policy {
    pub fn new(cap_bytes: u64) -> Self {
        Policy { cap_bytes, bytes: 0, lru: VecDeque::new() }
    }

    pub fn len(&self) -> usize {
        self.lru.len()
    }
    pub fn is_empty(&self) -> bool {
        self.lru.is_empty()
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    /// LANERES3: the configured byte cap — the TP capture gate reads it as the
    /// deterministic budget check (identical on every rank, unlike the live writer queue).
    pub fn cap_bytes(&self) -> u64 {
        self.cap_bytes
    }
    pub fn contains(&self, conv: u64) -> bool {
        self.lru.iter().any(|&(c, _, _)| c == conv)
    }

    /// Insert/refresh a snapshot of `bytes`; returns the conv keys shifted out (LRU victims
    /// whose files the caller should delete). Re-putting an existing conv refreshes it and
    /// replaces its size. A snapshot bigger than the whole cap is refused (returns no victims,
    /// nothing inserted — the engine then just recomputes for it).
    pub fn put(&mut self, conv: u64, bytes: u64, tokens: u64) -> Vec<u64> {
        if bytes > self.cap_bytes {
            return Vec::new();
        }
        let mut victims = Vec::new();
        if let Some(pos) = self.lru.iter().position(|&(c, _, _)| c == conv) {
            let (_, old, _) = self.lru.remove(pos).unwrap_or((conv, 0, 0));
            self.bytes = self.bytes.saturating_sub(old);
        }
        while !self.lru.is_empty() && self.bytes + bytes > self.cap_bytes {
            if let Some((c, b, _)) = self.lru.pop_front() {
                self.bytes = self.bytes.saturating_sub(b);
                victims.push(c);
            }
        }
        self.lru.push_back((conv, bytes, tokens));
        self.bytes += bytes;
        victims
    }

    /// The conv's snapshot was restored (or explicitly dropped): take it out of the policy.
    /// Returns true when it was present (a hit).
    pub fn remove(&mut self, conv: u64) -> bool {
        match self.lru.iter().position(|&(c, _, _)| c == conv) {
            Some(pos) => {
                if let Some((_, b, _)) = self.lru.remove(pos) {
                    self.bytes = self.bytes.saturating_sub(b);
                }
                true
            }
            None => false,
        }
    }

    /// LANERES2: iterate (conv, bytes, tokens), LRU-front order (deterministic).
    pub fn entries(&self) -> impl Iterator<Item = (u64, u64, u64)> + '_ {
        self.lru.iter().copied()
    }

    /// LANERES2: find the stored snapshot that is a PREFIX of `prompt` and passes the restore
    /// cost model. Returns (conv key, tokens); the caller reads the file, then remove()s the
    /// entry. Pure scan: at most cap / bytes-per-snapshot entries (~hundreds), each one hash
    /// of prompt[..tokens] — a few ms of CPU in the admission path, only on own-snapshot
    /// misses. MRU-first scan: the hottest eviction wins a tie.
    pub fn lookup(&self, prompt: &[u32], nvme_read_bps: u64, prefill_tps: f64) -> Option<(u64, usize)> {
        let mut logged = 0;
        for &(conv, bytes, tokens) in self.lru.iter().rev() {
            let n = tokens as usize;
            if n == 0 || n > prompt.len() {
                if logged < 2 {
                    logged += 1;
                    crate::drprintln!("[kv-spill] miss: snapshot n={n} vs prompt.len()={}", prompt.len());
                }
                continue;
            }
            if conv_key(&prompt[..n]) == conv {
                if self.restore_wins(bytes, tokens, nvme_read_bps, prefill_tps) {
                    return Some((conv, n));
                }
                if logged < 2 {
                    logged += 1;
                    crate::drprintln!("[kv-spill] miss: hash ok, cost model dropped (n={n} bytes={bytes})");
                }
            } else if logged < 2 {
                logged += 1;
                crate::drprintln!("[kv-spill] miss: hash mismatch at n={n} prompt.len()={}", prompt.len());
            }
        }
        None
    }


    /// The restore cost model: restore only if the NVMe read beats the re-prefill it replaces.
    /// `tokens` is the snapshot's token count, `prefill_tps` the engine's admission prefill rate.
    pub fn restore_wins(&self, bytes: u64, tokens: u64, nvme_read_bps: u64, prefill_tps: f64) -> bool {
        if bytes == 0 {
            return false; // nothing to read
        }
        if nvme_read_bps == 0 || prefill_tps <= 0.0 {
            return false;
        }
        let t_read = bytes as f64 / nvme_read_bps as f64;
        let t_recompute = tokens as f64 / prefill_tps;
        t_read < t_recompute
    }
}

/// The disk store: a background writer thread owns all writes/deletes; the serving thread only
/// enqueues (async, never in the round) and, on restore, does one synchronous read — the same
/// blocking an admission prefill already costs. Payloads are opaque to this module: the engine
/// serializes lane snapshots through its per-slot checkpoint path, so bit-exactness is theirs.
#[derive(Clone)]
pub struct Store {
    dir: PathBuf,
    tx: Option<Sender<Job>>,
    pub written: Arc<AtomicU64>,
    pub failed: Arc<AtomicU64>,
    pub durable: Arc<AtomicU64>,
    /// Bytes queued but not yet durable. Bounded by max_pending: a capture over budget is
    /// refused here (Err) and the caller counts a recompute-miss — the UNBOUNDED queue held
    /// GiB-scale payload Vecs and killed the b5x10 spill-ON cell on the memwatch.
    pub pending: Arc<AtomicU64>,
    pub max_pending: u64,
}

enum Job {
    Write { path: PathBuf, data: Vec<u8> },
    Delete { path: PathBuf },
}

fn path_for(dir: &Path, conv: u64, rank: usize) -> PathBuf {
    dir.join(format!("conv{conv:016x}_rank{rank}.ksp"))
}

impl Store {
    /// Create (or reuse) the spill directory and start the writer thread.
    pub fn spawn(dir: PathBuf) -> io::Result<Store> {
        Self::spawn_with(dir, 4 << 30)
    }

    /// LANERES2: spawn with an explicit pending-bytes budget — the serve side passes
    /// max(cap/4, 4 GiB), so a capture storm DROPS snapshots (counted on /metrics) instead of
    /// piling their GiB-scale payloads into host RAM behind the NVMe writer.
    pub fn spawn_with(dir: PathBuf, max_pending: u64) -> io::Result<Store> {
        std::fs::create_dir_all(&dir)?;
        let pending = Arc::new(AtomicU64::new(0));
        let (tx, rx) = mpsc::channel::<Job>();
        let written = Arc::new(AtomicU64::new(0));
        let failed = Arc::new(AtomicU64::new(0));
        let durable = Arc::new(AtomicU64::new(0));
        let p = Arc::clone(&pending);
        let (w, f, d) = (Arc::clone(&written), Arc::clone(&failed), Arc::clone(&durable));
        thread::Builder::new()
            .name("kv-spill-writer".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    match job {
                        Job::Write { path, data } => {
                            let tmp = path.with_extension("ksp.part");
                            let ok = std::fs::File::create(&tmp).and_then(|mut f| {
                                f.write_all(&data)?;
                                f.sync_all()
                            });
                            match ok.and_then(|_| std::fs::rename(&tmp, &path)) {
                                Ok(()) => {
                                    w.fetch_add(1, Ordering::Relaxed);
                                    d.fetch_add(data.len() as u64, Ordering::Relaxed);
                                }
                                Err(_) => {
                                    let _ = std::fs::remove_file(&tmp);
                                    f.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            p.fetch_sub(data.len() as u64, Ordering::Relaxed);
                        }
                        Job::Delete { path } => {
                            let _ = std::fs::remove_file(path);
                        }
                    }
                }
            })?;
        Ok(Store { dir, tx: Some(tx), written, failed, durable, pending, max_pending })
    }

    /// Enqueue the write of one snapshot: `tokens` + `payload` (bit-exact engine bytes).
    /// `total_len` = the on-disk size (header + payload) for the policy's byte accounting.
    pub fn put(&self, conv: u64, rank: usize, tokens: u64, payload: &[u8]) -> io::Result<u64> {
        let total = HEADER_LEN + payload.len() as u64;
        let mut data = Vec::with_capacity(total as usize);
        data.extend_from_slice(MAGIC);
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&tokens.to_le_bytes());
        data.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        data.extend_from_slice(&fnv1a64(payload).to_le_bytes());
        data.extend_from_slice(payload);
        if let Some(tx) = &self.tx {
            if self.pending.load(Ordering::Relaxed) + total > self.max_pending {
                return Err(io::Error::new(io::ErrorKind::WouldBlock, "spill queue over budget"));
            }
            tx.send(Job::Write { path: path_for(&self.dir, conv, rank), data })
                .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "spill writer gone"))?;
            self.pending.fetch_add(total, Ordering::Relaxed);
        }
        Ok(total)
    }

    /// LANERES2: enqueue a PREBUILT file body (finalize_header layout) — the single-copy path
    /// for captures; the payload Vec is moved, not re-copied.
    pub fn put_prebuilt(&self, conv: u64, rank: usize, data: Vec<u8>) -> io::Result<u64> {
        let total = data.len() as u64;
        if let Some(tx) = &self.tx {
            if self.pending.load(Ordering::Relaxed) + total > self.max_pending {
                return Err(io::Error::new(io::ErrorKind::WouldBlock, "spill queue over budget"));
            }
            tx.send(Job::Write { path: path_for(&self.dir, conv, rank), data })
                .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "spill writer gone"))?;
            self.pending.fetch_add(total, Ordering::Relaxed);
        }
        Ok(total)
    }

    /// LANERES3 (TP): write one snapshot SYNCHRONOUSLY from the calling (scheduler) thread —
    /// the TP capture hook needs the outcome at the lockstep point where it is agreed
    /// across ranks (`TpSync::spill_agree`), not an async completion that lands at an
    /// arbitrary later step. Same file format and accounting as the writer thread's
    /// Job::Write (tmp + rename); no queue, so no `max_pending` pressure — the TP path has
    /// at most one capture in flight at a time by construction (admission is serial).
    pub fn put_sync(&self, conv: u64, rank: usize, data: &[u8]) -> io::Result<u64> {
        let path = path_for(&self.dir, conv, rank);
        let tmp = path.with_extension("ksp.part");
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(data)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &path)?;
        self.written.fetch_add(1, Ordering::Relaxed);
        self.durable.fetch_add(data.len() as u64, Ordering::Relaxed);
        Ok(data.len() as u64)
    }

    /// Synchronous read for a restore: tokens + payload; Err on any miss/corruption (the caller
    /// falls back to re-prefill and counts the miss).
    pub fn get(&self, conv: u64, rank: usize) -> io::Result<(u64, Vec<u8>)> {
        let path = path_for(&self.dir, conv, rank);
        let mut f = std::fs::File::open(&path)?;
        let mut hdr = [0u8; 36];
        f.read_exact(&mut hdr)?;
        if &hdr[0..8] != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        let tokens = u64::from_le_bytes(hdr[12..20].try_into().unwrap());
        let plen = u64::from_le_bytes(hdr[20..28].try_into().unwrap()) as usize;
        let crc = u64::from_le_bytes(hdr[28..36].try_into().unwrap());
        if plen > (1 << 34) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "len absurd"));
        }
        let mut payload = vec![0u8; plen];
        f.read_exact(&mut payload)?;
        if fnv1a64(&payload) != crc {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "crc mismatch"));
        }
        Ok((tokens, payload))
    }

    pub fn remove(&self, conv: u64, rank: usize) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(Job::Delete { path: path_for(&self.dir, conv, rank) });
        }
    }
}

/// The conversation key: FNV-1a over the snapshot's prompt tokens. The engine owns token
/// identity; this keeps the spill module std-only and the key stable across ranks (every rank
/// hashes the same prompt, so TP ranks agree on the file names).
/// LANERES2: fill the reserved 36-byte header of a capture buffer — the exact layout `put`
/// writes and `get` reads (MAGIC, version 1, tokens, payload_len = buf.len()-36, crc), so the
/// payload is assembled ONCE and shipped with `put_prebuilt` (no GiB-scale re-copy).
pub fn finalize_header(buf: &mut [u8], tokens: u64) {
    let crc = fnv1a64(&buf[36..]);
    let plen = (buf.len() - 36) as u64;
    buf[0..8].copy_from_slice(MAGIC);
    buf[8..12].copy_from_slice(&1u32.to_le_bytes());
    buf[12..20].copy_from_slice(&tokens.to_le_bytes());
    buf[20..28].copy_from_slice(&plen.to_le_bytes());
    buf[28..36].copy_from_slice(&crc.to_le_bytes());
}

pub fn conv_key(tokens: &[u32]) -> u64 {
    let mut buf = Vec::with_capacity(tokens.len() * 4);
    for t in tokens {
        buf.extend_from_slice(&t.to_le_bytes());
    }
    fnv1a64(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Cost model: admits only when the read is strictly cheaper than the re-prefill.
    #[test]
    fn restore_cost_model() {
        let p = Policy::new(1 << 30);
        // 0.5 GiB at 5.1 GB/s = ~104 ms; 38K tokens at 1600 tps = ~23.8 s -> restore wins.
        assert!(p.restore_wins(1 << 29, 38_000, 5_100_000_000, 1600.0));
        // Tiny NVMe: recompute is cheaper -> drop.
        assert!(!p.restore_wins(1 << 29, 38_000, 10_000_000, 1600.0));
        // Boundary: equal is a drop (strictly cheaper required).
        let bytes = (38_000.0 / 1600.0 * 5.1e9) as u64;
        assert!(!p.restore_wins(bytes, 38_000, 5_100_000_000, 1600.0));
        assert!(p.restore_wins(bytes - 1, 38_000, 5_100_000_000, 1600.0));
        // Degenerate rates never admit.
        assert!(!p.restore_wins(1, 1, 0, 1600.0));
        assert!(!p.restore_wins(1, 1, 5_100_000_000, 0.0));
    }

    // 2. LRU cap: shift-out returns the oldest victim; refresh moves + resizes; bytes exact.
    #[test]
    fn lru_cap_and_refresh() {
        let mut p = Policy::new(100);
        assert!(p.put(1, 40, 1).is_empty());
        assert!(p.put(2, 40, 1).is_empty());
        assert_eq!(p.put(3, 40, 1), vec![1]); // oldest shifted out
        assert_eq!(p.len(), 2);
        assert!(!p.contains(1));
        assert!(p.contains(2) && p.contains(3));
        assert_eq!(p.put(2, 30, 1), Vec::<u64>::new()); // refresh + shrink: no victim
        assert_eq!(p.bytes(), 70);
        assert_eq!(p.put(4, 40, 1), vec![3]); // 2 refreshed, so 3 is now the LRU
        assert!(p.contains(2) && p.contains(4));
        assert!(!p.contains(3));
        assert_eq!(p.put(5, 200, 1), Vec::<u64>::new()); // bigger than the cap: refused
        assert!(!p.contains(5));
        assert_eq!(p.bytes(), 70);
        assert!(p.remove(4));
        assert!(!p.remove(4));
        assert_eq!(p.bytes(), 30);
    }


    // 2b. lookup(): the stored snapshot that prefixes the prompt, cost-model gated.
    #[test]
    fn lookup_prefix_and_cost() {
        let mut p = Policy::new(1 << 20);
        let snap: Vec<u32> = vec![7, 8, 9, 10, 11];
        let key = conv_key(&snap);
        assert!(p.put(key, 4096, snap.len() as u64).is_empty());
        // same conversation grown: the prompt starts with the snapshot
        let mut prompt = snap.clone();
        prompt.extend_from_slice(&[12, 13, 14]);
        // 4096 bytes at 1e10 B/s reads in 0.4 us; 5 tokens at 5 tps = 1 s: restore wins
        assert_eq!(p.lookup(&prompt, 10_000_000_000, 5.0), Some((key, 5)));
        // cost model off (read slower than re-prefill): no restore
        assert_eq!(p.lookup(&prompt, 1, 5.0), None);
        // an unrelated prompt does not match
        assert_eq!(p.lookup(&[1, 2, 3, 4, 5], 10_000_000_000, 5.0), None);
    }

    // 3. Store roundtrip: put -> get is byte-identical, tokens intact; wrong conv misses.
    #[test]
    fn store_roundtrip() {
        let dir = std::env::temp_dir().join(format!("gb10_kv_spill_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let st = Store::spawn(dir.clone()).unwrap();
        let payload: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        let total = st.put(0xdead_beef, 2, 38_000, &payload).unwrap();
        assert_eq!(total as usize, payload.len() + 36);
        // The writer is async: poll briefly for durability (test-only busy wait on a counter).
        for _ in 0..2000 {
            if st.written.load(Ordering::Relaxed) == 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(st.written.load(Ordering::Relaxed), 1);
        assert_eq!(st.durable.load(Ordering::Relaxed), payload.len() as u64 + 36);
        let (tokens, back) = st.get(0xdead_beef, 2).unwrap();
        assert_eq!(tokens, 38_000);
        assert_eq!(back, payload);
        assert!(st.get(0xfeed_face, 2).is_err());
        st.remove(0xdead_beef, 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 4. Corruption: a truncated/garbled file is Err (graceful recompute), never a panic.
    #[test]
    fn corrupt_file_is_graceful() {
        let dir = std::env::temp_dir().join(format!("gb10_kv_spill_bad_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = path_for(&dir, 7, 0);
        std::fs::write(&p, b"garbage not a spill file").unwrap();
        let st = Store { dir: dir.clone(), tx: None, written: Arc::new(AtomicU64::new(0)),
                         failed: Arc::new(AtomicU64::new(0)), durable: Arc::new(AtomicU64::new(0)),
                         pending: Arc::new(AtomicU64::new(0)), max_pending: u64::MAX };
        assert!(st.get(7, 0).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 5. Conv key: stable across calls and order-sensitive (prompt identity).
    #[test]
    fn conv_key_stable() {
        let a = vec![1u32, 2, 3, 0xdead_beef];
        let b = a.clone();
        assert_eq!(conv_key(&a), conv_key(&b));
        assert_ne!(conv_key(&a), conv_key(&[2u32, 1, 3, 0xdead_beef]));
        assert_ne!(conv_key(&a), conv_key(&a[..3]));
    }
}
#[cfg(test)]
mod spill_lookup_tests {
    use super::*;
#[test]
fn lookup_matches_prefix_of_full_prompt() {
    // The returning request's prompt EXTENDS the snapshot (append-only conversation).
    // lookup must scan the FULL prompt: a snapshot matches iff its tokens are a prefix.
    let snap: Vec<u32> = (0..64u32).collect();
    let mut pol = Policy::new(10_000);
    let conv = conv_key(&snap);
    pol.put(conv, 1000, snap.len() as u64);
    let mut full = snap.clone();
    full.extend_from_slice(&[64, 65, 66]);
    // The evicted-state regime: the caller's prefix-cache hit is 0, so the old
    // `&prompt[..0]` slicing found nothing; the full prompt must find it.
    assert!(pol.lookup(&full, NVME_READ_BPS, PREFILL_TPS).is_some());
    // An unrelated prompt must not match.
    let other: Vec<u32> = (100..164u32).collect();
    assert!(pol.lookup(&other, NVME_READ_BPS, PREFILL_TPS).is_none());
}
}
