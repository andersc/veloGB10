//! TOOLSCRUB1 (P18, owner ruling 2026-10-13 "D2 design"): opt-in, EMISSION-SIDE
//! tool-secret scrub, `--scrub-tool-secrets off / provenance / pattern / all`,
//! default off.
//!
//! Why: TOOLEVAL (S-JOHNFB3, 2026-10-13) measured the model echoing tool-result
//! credentials into `reasoning_content` on streamed agentic turns. Owner ruling: a
//! leaked credential is a security defect in ANY served field; the engine ships an
//! OPT-IN scrub as hardening, default off (zero cost when off, byte-identical served
//! stream), and does NOT impose a system prompt (D3 rejected; one recommendation line
//! for deployers rides this flag's help text).
//!
//! WHAT IT DOES WHEN ON -- only at the emission boundary (SSE chunk text and the
//! non-streaming JSON body; the model's context/KV/prompt are never modified and the
//! tool-call parser still sees raw text):
//!
//!   * provenance: candidates from THIS conversation tool-RESULT messages
//!     (role tool or function; never from user or assistant text): key-shaped
//!     substrings (the six shapes below) plus values assigned to secret-named keys
//!     (api_key=, token:, Authorization: Bearer ...). Matched exactly AND via
//!     width-10 shingles (step 4), so partial or re-embedded echoes match.
//!     [redacted:tool-secret].
//!   * pattern: key-shaped text anywhere in an emission, regardless of source:
//!     sk- + 16 [A-Za-z0-9_-]; AKIA + 16 [0-9A-Z]; gh[pousr]_ + 20 [A-Za-z0-9]; JWT
//!     shape eyJ + parts >= 8 b64url chars joined by dots; hex runs >= 16 digits with
//!     optional colon or dash separators (UUID/IPv6-shaped included); high-entropy
//!     runs (>= 24 chars of [A-Za-z0-9+/=_-], >= 3 of 4 char classes, >= 10 distinct
//!     chars). [redacted:key-shape].
//!   * all: both layers; a span hit by both redacts as provenance.
//!
//! PATTERN FALSE POSITIVES are a documented tradeoff of the on-state: UUIDs, IPv6
//! addresses and long mixed-class runs redact too. Off (the default) never changes a
//! byte.
//! STREAMING: a small per-field pending buffer keeps a secret that spans an SSE delta
//! boundary from ever being emitted in plaintext pieces. The emit cut never lands inside
//! a still-open match: text from the earliest position where a match could still extend
//! is held back until the next push or the field's finish. Provenance needles are
//! shingle-covered (step 4), so a fixed 9-byte holdback is split-safe for exact strings;
//! shape scanners report their own open starts. The held region is capped at MAX_PENDING
//! bytes; past the cap an open match's prefix is redacted (never emitted raw).
//!
//! COST: off = one enum compare per emission, no allocation, no candidates built — the
//! served stream is byte-identical to a binary without this module. On: bounded
//! candidate memory (<= 256 candidates, <= 1024 needles, <= 256 KiB of tool text
//! scanned per conversation), byte-automaton + six linear scans per emitted piece
//! (microseconds).
//!
//! COUNTERS: each redacted span bumps velogb10_scrub_hits_total{layer="provenance"} or
//! {layer="pattern"} on /metrics (a span split by the pending cap can count twice; the
//! counter is advisory liveness, not an exact audit).
//!
//! POLICY EDGE (accepted in the ruling): a user who explicitly asks to see a
//! tool-returned secret gets the redaction. Nothing here refuses or fails a request.

use std::borrow::Cow;

use crate::tokenizer::ChatMessage;

pub const PROV_MARKER: &str = "[redacted:tool-secret]";
pub const PAT_MARKER: &str = "[redacted:key-shape]";

/// Shingle width for provenance candidates: 10 chars of a credential-shaped string are
/// effectively unique; step 4 bounds the needle count (a 64-char key adds 14 shingles).
const SHINGLE_LEN: usize = 10;
const SHINGLE_STEP: usize = 4;
/// Fixed split-safe holdback for the needle automaton: every needle is a shingle
/// (>= SHINGLE_LEN) or an exact candidate whose interior is shingle-covered, so a needle
/// starting inside the last (SHINGLE_LEN - 1) bytes cannot complete in this buffer.
const AC_HOLDBACK: usize = SHINGLE_LEN - 1;
/// Caps (see module doc): candidate spans, total needles, scanned tool bytes per turn.
const MAX_CANDIDATES: usize = 256;
const MAX_NEEDLES: usize = 1024;
const MAX_TOOL_TEXT: usize = 256 * 1024;
/// Longest run a shape scanner treats as one match (longer runs match in pieces; both
/// pieces redact — no plaintext leak, two markers). Also the needle length cap.
const MAX_SPAN: usize = 512;
/// Held-back bytes cap before a force flush (module doc).
const MAX_PENDING: usize = 4096;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Off,
    Provenance,
    Pattern,
    All,
}

impl Mode {
    /// The registry value of --scrub-tool-secrets (validated to the four spellings at
    /// startup by Ty::Choice; unknown values defensively read as off).
    pub fn from_opt() -> Mode {
        Self::parse(&crate::opts::choice(crate::opt!("scrub-tool-secrets")))
    }
    pub fn parse(s: &str) -> Mode {
        match s.trim() {
            "provenance" => Mode::Provenance,
            "pattern" => Mode::Pattern,
            "all" => Mode::All,
            _ => Mode::Off,
        }
    }
    pub fn provenance(self) -> bool { matches!(self, Mode::Provenance | Mode::All) }
    pub fn pattern(self) -> bool { matches!(self, Mode::Pattern | Mode::All) }
}

/// One redactable span of a text (byte offsets, end exclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span {
    start: usize,
    end: usize,
    provenance: bool,
}

/// Byte-level Aho-Corasick over the provenance needles (exact candidates + shingles).
/// Per-node child vectors keep memory proportional to total needle length, not to the
/// alphabet.
struct Ac {
    children: Vec<Vec<(u8, u32)>>,
    fail: Vec<u32>,
    /// needle index ended at this node, if any
    needle: Vec<Option<u32>>,
    /// nearest strict suffix node that ends a needle (output chain)
    out: Vec<Option<u32>>,
    lens: Vec<u16>,
}

impl Ac {
    fn new() -> Self {
        Ac { children: vec![vec![]], fail: vec![0], needle: vec![None], out: vec![None], lens: vec![] }
    }

    fn add(&mut self, pat: &[u8]) -> u32 {
        let mut n = 0u32;
        for &b in pat {
            if let Some(&(_, next)) = self.children[n as usize].iter().find(|&&(c, _)| c == b) {
                n = next;
            } else {
                let id = self.children.len() as u32;
                self.children.push(vec![]);
                self.fail.push(0);
                self.needle.push(None);
                self.out.push(None);
                self.children[n as usize].push((b, id));
                n = id;
            }
        }
        n
    }

    fn build(&mut self, ends: &[(u32, u16)]) {
        self.lens = vec![0; ends.len()];
        for (idx, &(node, len)) in ends.iter().enumerate() {
            self.needle[node as usize] = Some(idx as u32);
            self.lens[idx] = len;
        }
        let mut queue = std::collections::VecDeque::new();
        let root: Vec<u32> = self.children[0].iter().map(|&(_, c)| c).collect();
        for c in root {
            self.fail[c as usize] = 0;
            queue.push_back(c);
        }
        while let Some(n) = queue.pop_front() {
            let kids: Vec<(u8, u32)> = self.children[n as usize].clone();
            for (b, child) in kids {
                let mut f = self.fail[n as usize];
                let mut target = 0u32;
                loop {
                    if let Some(&(_, next)) = self.children[f as usize].iter().find(|&&(c, _)| c == b) {
                        if next != child { target = next; }
                        break;
                    }
                    if f == 0 { break; }
                    f = self.fail[f as usize];
                }
                self.fail[child as usize] = target;
                self.out[child as usize] = if self.needle[target as usize].is_some() { Some(target) }
                    else { self.out[target as usize] };
                queue.push_back(child);
            }
        }
    }

    /// All needle matches as (start, end) byte spans.
    fn scan(&self, text: &[u8]) -> Vec<(usize, usize)> {
        let mut hits = vec![];
        let mut state = 0u32;
        for (i, &b) in text.iter().enumerate() {
            loop {
                if let Some(&(_, next)) = self.children[state as usize].iter().find(|&&(c, _)| c == b) {
                    state = next;
                    break;
                }
                if state == 0 { break; }
                state = self.fail[state as usize];
            }
            let mut node = if self.needle[state as usize].is_some() { Some(state) }
                else { self.out[state as usize] };
            while let Some(nd) = node {
                if let Some(k) = self.needle[nd as usize] {
                    let len = self.lens[k as usize] as usize;
                    if i + 1 >= len { hits.push((i + 1 - len, i + 1)); }
                }
                node = self.out[nd as usize];
            }
        }
        hits
    }
}

// ---------------------------------------------------------------------------
// Key-shape scanners (the pattern layer, and candidate extraction for provenance).
// Hand-rolled: the crate has no regex dependency and these shapes are simple runs.
// ---------------------------------------------------------------------------

/// A maximal shape run: `qualifies` = a complete match of the shape; `at_end` = the run
/// touches the end of the scanned text and may extend with more input.
struct Run {
    start: usize,
    end: usize,
    qualifies: bool,
    at_end: bool,
}

fn is_b64url(b: u8) -> bool { b.is_ascii_alphanumeric() || b == b'_' || b == b'-' }
fn is_hex(b: u8) -> bool { b.is_ascii_hexdigit() }

fn run_end(b: &[u8], from: usize, pred: impl Fn(u8) -> bool) -> usize {
    let mut j = from;
    while j < b.len() && pred(b[j]) { j += 1; }
    j
}

/// Shapes 1-3 (prefix-based: sk-, AKIA, gh_) as maximal runs. Runs that neither qualify
/// nor could still extend are dropped by `analyze`; a short sk- prefix at the end of a
/// streaming buffer is reported so the streamer can hold it back.
fn scan_shape_runs(text: &str) -> Vec<Run> {
    let b = text.as_bytes();
    let mut runs: Vec<Run> = vec![];
    let mut i = 0;
    while i + 3 <= b.len() {
        if b[i] == b's' && b[i + 1] == b'k' && b[i + 2] == b'-' {
            let j = run_end(b, i + 3, |c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-');
            if j - (i + 3) >= 8 || j == b.len() {
                runs.push(Run { start: i, end: j,
                    qualifies: j - (i + 3) >= 16, at_end: j == b.len() });
            }
            i = j.max(i + 1);
        } else { i += 1; }
    }
    let mut i = 0;
    while i + 4 <= b.len() {
        if &b[i..i + 4] == b"AKIA" {
            let j = run_end(b, i + 4, |c| c.is_ascii_uppercase() || c.is_ascii_digit());
            if j - (i + 4) >= 8 || j == b.len() {
                runs.push(Run { start: i, end: j,
                    qualifies: j - (i + 4) >= 16, at_end: j == b.len() });
            }
            i = j.max(i + 1);
        } else { i += 1; }
    }
    let mut i = 0;
    while i + 4 <= b.len() {
        if b[i] == b'g' && b[i + 1] == b'h' && matches!(b[i + 2], b'p' | b'o' | b'u' | b's' | b'r') && b[i + 3] == b'_' {
            let j = run_end(b, i + 4, |c| c.is_ascii_alphanumeric());
            if j - (i + 4) >= 8 || j == b.len() {
                runs.push(Run { start: i, end: j,
                    qualifies: j - (i + 4) >= 20, at_end: j == b.len() });
            }
            i = j.max(i + 1);
        } else { i += 1; }
    }
    runs
}

/// Shape 4 (JWT): eyJ + part(>=8) + . + part(>=8) + . + part(>=8), b64url parts.
/// Partial prefixes that touch the end are reported open (holdback), never redacted
/// before they complete.
fn scan_jwt(b: &[u8], runs: &mut Vec<Run>) {
    let mut i = 0;
    while i + 3 <= b.len() {
        if &b[i..i + 3] != b"eyJ" { i += 1; continue; }
        let p1 = run_end(b, i + 3, is_b64url);
        if p1 == b.len() || p1 - (i + 3) < 8 || b[p1] != b'.' {
            if p1 == b.len() && p1 > i + 3 {
                runs.push(Run { start: i, end: p1, qualifies: false, at_end: true });
            }
            i = p1.max(i + 1);
            continue;
        }
        let p2s = p1 + 1;
        let p2 = run_end(b, p2s, is_b64url);
        if p2 == b.len() || p2 - p2s < 8 || b[p2] != b'.' {
            if p2 == b.len() {
                runs.push(Run { start: i, end: p2, qualifies: false, at_end: true });
            }
            i = p2.max(i + 1);
            continue;
        }
        let p3 = run_end(b, p2 + 1, is_b64url);
        let n3 = p3 - (p2 + 1);
        if n3 >= 8 {
            runs.push(Run { start: i, end: p3, qualifies: true, at_end: p3 == b.len() });
        } else if p3 == b.len() {
            runs.push(Run { start: i, end: p3, qualifies: false, at_end: true });
        }
        i = p3.max(i + 1);
    }
}

/// Shape 5: hex runs with >= 16 hex digits total, single colon/dash separators allowed
/// between hex groups (UUID-shaped, colon-hex clusters; a contiguous >= 16-digit run
/// qualifies too). A short hex run touching the end is reported open.
fn scan_hex(b: &[u8], runs: &mut Vec<Run>) {
    let mut i = 0;
    while i < b.len() {
        if !is_hex(b[i]) { i += 1; continue; }
        let start = i;
        let mut j = run_end(b, i, is_hex);
        let mut digits = j - start;
        loop {
            if j < b.len() && (b[j] == b':' || b[j] == b'-') && j + 1 < b.len() && is_hex(b[j + 1]) {
                let e = run_end(b, j + 1, is_hex);
                digits += e - (j + 1);
                j = e;
            } else { break; }
        }
        let at_end = j == b.len();
        if digits >= 16 || (at_end && j - start >= 4) {
            runs.push(Run { start, end: j, qualifies: digits >= 16, at_end });
        }
        i = j.max(start + 1);
    }
}

/// Shape 6: high-entropy runs. >= 24 chars of [A-Za-z0-9+/=_-], >= 3 of 4 character
/// classes (lower/upper/digit/symbol), >= 10 distinct characters. An at-end run of
/// >= 4 chars is reported open (it may still grow into a qualifying run).
fn scan_entropy(b: &[u8], runs: &mut Vec<Run>) {
    let pred = |c: u8| c.is_ascii_alphanumeric() || matches!(c, b'+' | b'/' | b'=' | b'_' | b'-');
    let mut i = 0;
    while i < b.len() {
        if !pred(b[i]) { i += 1; continue; }
        let j = run_end(b, i, pred);
        let n = j - i;
        if n >= 24 || (j == b.len() && n >= 4) {
            let s = &b[i..j];
            let classes = (s.iter().any(|c| c.is_ascii_lowercase()) as u8)
                + (s.iter().any(|c| c.is_ascii_uppercase()) as u8)
                + (s.iter().any(|c| c.is_ascii_digit()) as u8)
                + (s.iter().any(|c| matches!(c, b'+' | b'/' | b'=' | b'_' | b'-')) as u8);
            let mut seen = [false; 128];
            let mut distinct = 0usize;
            for &c in s {
                if c < 128 && !seen[c as usize] { seen[c as usize] = true; distinct += 1; }
            }
            runs.push(Run { start: i, end: j,
                qualifies: n >= 24 && classes >= 3 && distinct >= 10, at_end: j == b.len() });
        }
        i = j.max(i + 1);
    }
}

/// All six shapes over one text.
fn all_shape_runs(text: &str) -> Vec<Run> {
    let b = text.as_bytes();
    let mut runs = scan_shape_runs(text);
    scan_jwt(b, &mut runs);
    scan_hex(b, &mut runs);
    scan_entropy(b, &mut runs);
    runs
}

/// Secret-named keys whose assigned values become provenance candidates.
const SECRET_KEYWORDS: [&str; 16] = [
    "api_key", "apikey", "api-key", "access_key", "accesskey", "secret", "client_secret",
    "token", "auth_token", "session_token", "password", "passwd", "authorization",
    "credentials", "private_key", "bearer",
];

fn find_from(hay: &str, needle: &str, from: usize) -> Option<usize> {
    hay.get(from..)?.find(needle).map(|p| p + from)
}

/// Values assigned to secret-named keys: KEYWORD then 1..=6 separator chars (space,
/// tab, quotes, colon, equals, hash, angle brackets) then a value run (no spaces,
/// quotes or angles), length 8..=256, and (length >= 12 or contains a digit or a
/// symbol). Trailing punctuation is stripped. Case-insensitive keyword, ASCII word
/// boundaries (a keyword inside a longer word never matches). 0x22/0x27 are the
/// ASCII double/single quote bytes.
fn scan_assigned(text: &str) -> Vec<(usize, usize)> {
    let b = text.as_bytes();
    let lower = text.to_ascii_lowercase();
    let mut spans = vec![];
    for kw in SECRET_KEYWORDS {
        let mut i = 0;
        while let Some(pos) = find_from(&lower, kw, i) {
            let (s, e) = (pos, pos + kw.len());
            let wordish = |c: u8| c.is_ascii_alphanumeric() || c == b'_' || c == b'-';
            let before_ok = s == 0 || !wordish(b[s - 1]);
            let after_ok = e >= b.len() || !wordish(b[e]);
            if before_ok && after_ok {
                let mut j = e;
                let mut skipped = 0;
                while j < b.len() && skipped < 6
                    && matches!(b[j], b' ' | b'\t' | 0x22 | 0x27 | b':' | b'=' | b'#' | b'<' | b'>')
                { j += 1; skipped += 1; }
                if skipped >= 1 {
                    let vs = j;
                    while j < b.len() && !matches!(b[j], b' ' | b'\t' | 0x22 | 0x27 | b'<' | b'>') { j += 1; }
                    while j > vs && matches!(b[j - 1], b'.' | b',' | b';' | b')' | b']' | b'}') { j -= 1; }
                    let v = &b[vs..j];
                    let vlen = v.len();
                    let has_digit = v.iter().any(|c| c.is_ascii_digit());
                    let has_sym = v.iter().any(|c| !c.is_ascii_alphanumeric());
                    if vlen >= 8 && vlen <= 256 && (vlen >= 12 || has_digit || has_sym) {
                        spans.push((vs, j));
                    }
                }
            }
            i = e.max(pos + 1);
        }
    }
    spans
}


/// Generic quoted/assigned values from tool results: TOOLSCRUB1 A5 fix (2026-10-13).
/// A5_split_key deliberately splits a key so neither half is key-shaped and neither
/// sits under a secret-named key; the only robust provenance signal is the value ITSELF:
/// any quoted or =-assigned tool-result value that is 12..=64 chars, whitespace-free,
/// mixes letters and digits, has >= 8 distinct chars, is not date-like and carries no
/// URL scheme becomes a candidate. Aggressive by design (provenance is opt-in; the
/// accepted policy edge is that a user asking to see a tool-returned secret is redacted).
fn scan_generic_values(text: &str) -> Vec<(usize, usize)> {
    let b = text.as_bytes();
    let mut spans = vec![];
    let val_ok = |v: &[u8]| -> bool {
        if v.len() < 12 || v.len() > 64 { return false; }
        if v.iter().any(|&c| c == b' ' || c == b'\t' || c == 0x22 || c == 0x27) { return false; }
        let has_letter = v.iter().any(|c| c.is_ascii_alphabetic());
        let has_digit = v.iter().any(|c| c.is_ascii_digit());
        if !(has_letter && has_digit) { return false; }
        let mut seen = [false; 128];
        let mut distinct = 0;
        for &c in v { if c < 128 && !seen[c as usize] { seen[c as usize] = true; distinct += 1; } }
        if distinct < 8 { return false; }
        // date-like (only digits and - : . / separators) and URLs are not secrets
        let dateish = v.iter().all(|&c| c.is_ascii_digit() || matches!(c, b'-' | b':' | b'.' | b'/'));
        if dateish { return false; }
        for i in 0..v.len().saturating_sub(2) {
            if v[i] == b':' && v[i + 1] == b'/' && v[i + 2] == b'/' { return false; }
        }
        true
    };
    let mut i = 0;
    while i < b.len() {
        // form 1: "NAME": "VALUE"
        if b[i] == 0x22 {
            let name_end = match b[i + 1..].iter().position(|&c| c == 0x22) { Some(p) => i + 1 + p, None => { i += 1; continue; } };
            let mut j = name_end + 1;
            while j < b.len() && (b[j] == b' ' || b[j] == b'\t') { j += 1; }
            if j < b.len() && b[j] == b':' {
                j += 1;
                while j < b.len() && (b[j] == b' ' || b[j] == b'\t') { j += 1; }
                if j < b.len() && b[j] == 0x22 {
                    let vstart = j + 1;
                    let vend = match b[vstart..].iter().position(|&c| c == 0x22) { Some(p) => vstart + p, None => usize::MAX };
                    if vend != usize::MAX && val_ok(&b[vstart..vend]) {
                        spans.push((vstart, vend));
                        i = vend + 1;
                        continue;
                    }
                }
            }
            i = name_end + 1;
            continue;
        }
        // form 2: NAME=VALUE (bare, unquoted)
        if b[i].is_ascii_alphanumeric() || b[i] == b'_' {
            let mut j = i;
            while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_' || b[j] == b'-') { j += 1; }
            if j < b.len() && b[j] == b'=' {
                let vstart = j + 1;
                let mut k = vstart;
                while k < b.len() && !matches!(b[k], b' ' | b'\t' | 0x22 | 0x27 | b',' | b')' | b'}' ) { k += 1; }
                if val_ok(&b[vstart..k]) {
                    spans.push((vstart, k));
                    i = k;
                    continue;
                }
            }
            i = j.max(i + 1);
            continue;
        }
        i += 1;
    }
    spans
}

/// Provenance needle set from the conversation's tool-result messages: key-shaped
/// substrings + assigned secret values, each as an exact needle plus width-10 shingles.
/// Returns a deduped needle list (bounded by MAX_NEEDLES).
pub fn extract_candidates(messages: &[ChatMessage]) -> Vec<String> {
    let mut cands: Vec<String> = vec![];
    let mut budget = MAX_TOOL_TEXT;
    for m in messages {
        if m.role != "tool" && m.role != "function" { continue; }
        let Some(text) = &m.content else { continue };
        if text.len() > budget { break; }
        budget -= text.len();
        let mut spans: Vec<(usize, usize)> = all_shape_runs(text).iter()
            .filter(|r| r.qualifies).map(|r| (r.start, r.end)).collect();
        spans.extend(scan_assigned(text));
        spans.extend(scan_generic_values(text));
        spans.sort_unstable();
        spans.dedup();
        for (s, e) in spans {
            if cands.len() >= MAX_CANDIDATES { break; }
            cands.push(text[s..e].to_string());
        }
        if cands.len() >= MAX_CANDIDATES { break; }
    }
    let mut needles: Vec<String> = Vec::new();
    for c in &cands {
        let end = c.len().min(MAX_SPAN);
        needles.push(c[..end].to_string());
    }
    for c in &cands {
        if needles.len() >= MAX_NEEDLES { break; }
        let end = c.len().min(MAX_SPAN);
        if end >= SHINGLE_LEN + 4 {
            let mut s = 0;
            while s + SHINGLE_LEN <= end && needles.len() < MAX_NEEDLES {
                needles.push(c[s..s + SHINGLE_LEN].to_string());
                s += SHINGLE_STEP;
            }
        }
    }
    needles.sort();
    needles.dedup();
    needles
}

fn build_ac(needles: &[String]) -> Ac {
    let mut ac = Ac::new();
    let mut ends = vec![];
    for n in needles {
        let node = ac.add(n.as_bytes());
        ends.push((node, n.len() as u16));
    }
    ac.build(&ends);
    ac
}

struct Analysis {
    spans: Vec<Span>,
    open_starts: Vec<usize>,
}

/// Floor `pos` to a UTF-8 char boundary of `text`.
fn floor_boundary(text: &str, pos: usize) -> usize {
    let b = text.as_bytes();
    let mut p = pos.min(text.len());
    while p > 0 && p < b.len() && (b[p] & 0xC0) == 0x80 { p -= 1; }
    p
}

/// Find redactable spans (and, for a streaming buffer, the positions where a match may
/// still extend). `stream_end` = the text is final (no more input will follow), so
/// at-end runs are decided now.
fn analyze(ac: Option<&Ac>, pattern: bool, text: &str, stream_end: bool) -> Analysis {
    let mut spans = vec![];
    let mut open_starts = vec![];
    if let Some(ac) = ac {
        for (s, e) in ac.scan(text.as_bytes()) {
            spans.push(Span { start: s, end: e, provenance: true });
        }
    }
    if pattern {
        for r in all_shape_runs(text) {
            if !stream_end && r.at_end {
                open_starts.push(r.start);
            } else if r.qualifies {
                spans.push(Span { start: r.start, end: r.end, provenance: false });
            }
        }
    }
    Analysis { spans, open_starts }
}

/// Merge overlapping/touching spans (provenance wins per merged span), replace each
/// with its marker, bump the /metrics counters, and return text[..region_end] redacted.
fn render(text: &str, spans: Vec<Span>, region_end: usize) -> String {
    let region_end = floor_boundary(text, region_end);
    let mut spans: Vec<Span> = spans.into_iter().filter(|s| s.end <= region_end).collect();
    if spans.is_empty() { return text[..region_end].to_string(); }
    spans.sort_unstable_by_key(|s| (s.start, s.end));
    let mut merged: Vec<Span> = Vec::with_capacity(spans.len());
    for s in spans {
        if let Some(last) = merged.last_mut() {
            if s.start <= last.end {
                last.end = last.end.max(s.end);
                last.provenance |= s.provenance;
                continue;
            }
        }
        merged.push(s);
    }
    let mut out = String::with_capacity(region_end);
    let mut pos = 0;
    for s in &merged {
        if s.start > pos { out.push_str(&text[pos..s.start]); }
        if s.provenance {
            out.push_str(PROV_MARKER);
            crate::metrics::scrub_hit_provenance();
        } else {
            out.push_str(PAT_MARKER);
            crate::metrics::scrub_hit_pattern();
        }
        pos = s.end;
    }
    if pos < region_end { out.push_str(&text[pos..region_end]); }
    out
}

/// Per-request scrubber. Build once per chat request (mode + the conversation's
/// tool-result messages); feed every emitted text field through it. `off` does no
/// work at all: pushes return the input borrowed, nothing is buffered.
pub struct Scrubber {
    mode: Mode,
    ac: Option<Ac>,
    r_pending: String,
    c_pending: String,
}

impl Scrubber {
    pub fn new(mode: Mode, messages: &[ChatMessage]) -> Scrubber {
        let ac = if mode.provenance() {
            let needles = extract_candidates(messages);
            if needles.is_empty() { None } else { Some(build_ac(&needles)) }
        } else { None };
        Scrubber { mode, ac, r_pending: String::new(), c_pending: String::new() }
    }

    /// One-shot scrub of a complete string (tool-call arguments JSON, non-streaming
    /// body fields). Off modes never allocate.
    pub fn scrub_full<'a>(&self, s: &'a str) -> Cow<'a, str> {
        if self.mode == Mode::Off || (!self.mode.provenance() && !self.mode.pattern()) {
            return Cow::Borrowed(s);
        }
        let a = analyze(self.ac.as_ref(), self.mode.pattern(), s, true);
        if a.spans.is_empty() { return Cow::Borrowed(s); }
        Cow::Owned(render(s, a.spans, s.len()))
    }

    /// Feed the next emitted slice of the reasoning field; returns the text safe to
    /// emit now (a bounded tail is held back so a match cannot straddle deltas).
    pub fn push_reasoning<'a>(&mut self, s: &'a str) -> Cow<'a, str> { self.push_field(true, s) }
    /// Feed the next emitted slice of the content field.
    pub fn push_content<'a>(&mut self, s: &'a str) -> Cow<'a, str> { self.push_field(false, s) }
    /// End of the reasoning field: flushes everything still held, plus `tail`.
    pub fn finish_reasoning(&mut self, tail: &str) -> String { self.finish_field(true, tail) }
    /// End of the content field: flushes everything still held, plus `tail`.
    pub fn finish_content(&mut self, tail: &str) -> String { self.finish_field(false, tail) }

    fn pend(&mut self, reasoning: bool) -> &mut String {
        if reasoning { &mut self.r_pending } else { &mut self.c_pending }
    }

    fn push_field<'a>(&mut self, reasoning: bool, s: &'a str) -> Cow<'a, str> {
        if self.mode == Mode::Off { return Cow::Borrowed(s); }
        let mode = self.mode;
        let has_ac = self.ac.is_some();
        let mut buf = std::mem::take(self.pend(reasoning));
        buf.push_str(s);
        let mut a = analyze(self.ac.as_ref(), mode.pattern(), &buf, false);
        let mut cut = buf.len();
        if let Some(min_open) = a.open_starts.iter().copied().min() { cut = cut.min(min_open); }
        if has_ac { cut = cut.min(buf.len().saturating_sub(AC_HOLDBACK)); }
        // Force flush: never hold back more than MAX_PENDING bytes. An open shape whose
        // held prefix is long has that prefix REDACTED rather than emitted raw.
        if buf.len() > cut + MAX_PENDING {
            let fcut = floor_boundary(&buf, buf.len() - MAX_PENDING);
            for &st in &a.open_starts {
                if st < fcut && fcut - st >= 16 {
                    a.spans.push(Span { start: st, end: fcut, provenance: false });
                }
            }
            cut = fcut;
        }
        let cut = floor_boundary(&buf, cut);
        // A complete span that crosses the cut keeps only its redacted prefix here; the
        // rest stays pending and re-matches next push (nothing redactable is dropped).
        let mut spans = a.spans;
        spans.retain_mut(|sp| {
            if sp.start >= cut { return false; }
            if sp.end > cut { sp.end = cut; }
            true
        });
        let out = render(&buf, spans, cut);
        *self.pend(reasoning) = buf[cut..].to_string();
        Cow::Owned(out)
    }

    fn finish_field(&mut self, reasoning: bool, tail: &str) -> String {
        let mode = self.mode;
        let mut buf = std::mem::take(self.pend(reasoning));
        buf.push_str(tail);
        if mode == Mode::Off { return buf; }
        let a = analyze(self.ac.as_ref(), mode.pattern(), &buf, true);
        let end = floor_boundary(&buf, buf.len());
        render(&buf, a.spans, end)
    }
}

/// One boot log line naming the mode (printed on every server boot, off included).
pub fn boot_line(mode: Mode) -> String {
    match mode {
        Mode::Off => "[scrub] --scrub-tool-secrets=off (default): no served byte is altered".to_string(),
        Mode::Provenance => "[scrub] --scrub-tool-secrets=provenance: strings echoed from this conversation's tool results are redacted at emission ([redacted:tool-secret]); model context untouched".to_string(),
        Mode::Pattern => "[scrub] --scrub-tool-secrets=pattern: key-shaped text is redacted at emission ([redacted:key-shape]); model context untouched".to_string(),
        Mode::All => "[scrub] --scrub-tool-secrets=all: tool-result strings AND key-shaped text redacted at emission ([redacted:tool-secret] / [redacted:key-shape]); model context untouched".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage { role: role.into(), content: Some(content.into()), images: vec![],
            layout: vec![], unsupported_parts: vec![], tool_calls: None, tool_call_id: None,
            name: None, reasoning_content: None }
    }

    const KEY: &str = "sk-TOOLEVAL-A5-Zx9Ym2KpQ7wR4tN8vB3c";
    const AWS: &str = "AKIAIOSFODNN7EXAMPLE";
    const GH: &str = "ghp_abcdefghijklmnopqrstuvwxyz01";
    const JWTS: &str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N65IjjJ";
    const HEX16: &str = "a1b2c3d4e5f60718";
    const UUID: &str = "550e8400-e29b-41d4-a716-446655440000";
    const ENTROPY: &str = "Zq7mK2pX9wL4vB8nC3dF5gH6";

    #[test]
    fn mode_parses_four_spellings() {
        assert_eq!(Mode::parse("off"), Mode::Off);
        assert_eq!(Mode::parse("provenance"), Mode::Provenance);
        assert_eq!(Mode::parse("pattern"), Mode::Pattern);
        assert_eq!(Mode::parse("all"), Mode::All);
        assert_eq!(Mode::parse("anything-else"), Mode::Off);
        assert!(!Mode::Off.provenance());
        assert!(!Mode::Off.pattern());
        assert!(Mode::All.provenance());
        assert!(Mode::All.pattern());
    }

    #[test]
    fn off_is_borrowed_passthrough() {
        let msgs = vec![msg("tool", &format!("api_key = {}", KEY))];
        let mut s = Scrubber::new(Mode::Off, &msgs);
        assert!(matches!(s.push_reasoning("thinking"), Cow::Borrowed(_)));
        assert!(matches!(s.push_content(KEY), Cow::Borrowed(_)));
        assert!(matches!(s.scrub_full(KEY), Cow::Borrowed(_)));
        assert_eq!(s.finish_content(""), "");
        assert!(s.ac.is_none());
    }

    #[test]
    fn provenance_exact_and_partial() {
        let msgs = vec![msg("tool", &format!("result: your key {} , ok", KEY))];
        let s = Scrubber::new(Mode::Provenance, &msgs);
        let text = format!("the key was {} indeed", KEY);
        let out = s.scrub_full(&text);
        assert!(out.contains(PROV_MARKER), "no marker: {}", out);
        assert!(!out.contains(KEY), "leak: {}", out);
        // partial echo (key minus last 6 chars) still matches via shingles
        let partial = &KEY[..KEY.len() - 6];
        let text2 = format!("saw {} too", partial);
        let out2 = s.scrub_full(&text2);
        assert!(!out2.contains(partial), "partial leak: {}", out2);
        assert!(out2.contains(PROV_MARKER), "no marker: {}", out2);
    }

    #[test]
    fn provenance_never_from_user_text() {
        let msgs = vec![msg("user", &format!("here is my key {}", KEY))];
        let s = Scrubber::new(Mode::Provenance, &msgs);
        let text = format!("your key {} is visible", KEY);
        let out = s.scrub_full(&text);
        assert!(out.contains(KEY), "provenance must not touch non-tool text: {}", out);
    }

    #[test]
    fn pattern_six_shapes_hit() {
        let s = Scrubber::new(Mode::Pattern, &[]);
        for shape in [KEY, AWS, GH, JWTS, HEX16, UUID, ENTROPY] {
            let text = format!("prefix {} suffix", shape);
            let out = s.scrub_full(&text);
            assert!(!out.contains(shape), "leak of {}: {}", shape, out);
            assert!(out.contains(PAT_MARKER), "no marker for {}: {}", shape, out);
        }
    }

    #[test]
    fn pattern_plain_text_survives() {
        let s = Scrubber::new(Mode::Pattern, &[]);
        for text in ["Grand Hotel Bristol", "Paris", "2026-10-13", "hello world",
                     "https://example.com/some/very/long/path", "BK-2026-1234", "nights=3"] {
            let out = s.scrub_full(text);
            assert_eq!(&*out, text, "over-redaction: {} -> {}", text, out);
        }
    }

    #[test]
    fn assigned_secret_values_become_candidates() {
        let msgs = vec![msg("tool", "{\"api_key\": \"Zq7mK2pX9wL4\", \"status\": \"ok\"}")];
        let s = Scrubber::new(Mode::Provenance, &msgs);
        let out = s.scrub_full("the api key is Zq7mK2pX9wL4, noted");
        assert!(!out.contains("Zq7mK2pX9wL4"), "leak: {}", out);
        assert!(out.contains(PROV_MARKER), "no marker: {}", out);
        // a non-secret JSON value from the same result is not a candidate
        let out2 = s.scrub_full("status was ok");
        assert_eq!(&*out2, "status was ok");
    }

    #[test]
    fn tool_calls_args_json_stays_valid() {
        let s = Scrubber::new(Mode::Pattern, &[]);
        let args = format!("{{\"cmd\": \"send {}\"}}", KEY);
        let out = s.scrub_full(&args);
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["cmd"], format!("send {}", PAT_MARKER));
    }

    #[test]
    fn provenance_wins_over_pattern() {
        let msgs = vec![msg("tool", &format!("key {}", KEY))];
        let s = Scrubber::new(Mode::All, &msgs);
        let text = format!("quote: {}", KEY);
        let out = s.scrub_full(&text);
        assert!(out.contains(PROV_MARKER), "no prov marker: {}", out);
        assert!(!out.contains(PAT_MARKER), "pattern marker should lose: {}", out);
        assert!(!out.contains(KEY), "leak: {}", out);
    }

    #[test]
    fn pattern_hits_user_pasted_keys_too() {
        let s = Scrubber::new(Mode::Pattern, &[]);
        let text = format!("as you said, {}", KEY);
        let out = s.scrub_full(&text);
        assert!(!out.contains(KEY), "leak: {}", out);
        assert!(out.contains(PAT_MARKER));
    }

    #[test]
    fn generic_quoted_values_are_candidates_but_plain_text_is_not() {
        let msgs = vec![msg("tool",
            "{\"city\": \"Paris\", \"forecast\": \"18C rain\", \"half2\": \"second-4c6d8e0f1a23\", \"when\": \"2026-10-13\", \"link\": \"https://ex4mple.com/a9b\"}")];
        let s = Scrubber::new(Mode::Provenance, &msgs);
        let out = s.scrub_full("half2 was second-4c6d8e0f1a23");
        assert!(!out.contains("second-4c6d8e0f1a23"), "leak: {}", out);
        assert!(out.contains(PROV_MARKER));
        let ok = s.scrub_full("city Paris forecast 18C rain when 2026-10-13");
        assert_eq!(&*ok, "city Paris forecast 18C rain when 2026-10-13");
    }

    #[test]
    fn role_function_counts_as_tool_result() {
        let msgs = vec![msg("function", &format!("token={}", AWS))];
        let s = Scrubber::new(Mode::Provenance, &msgs);
        let out = s.scrub_full(AWS);
        assert!(!out.contains(AWS), "leak: {}", out);
        assert!(out.contains(PROV_MARKER));
    }

    #[test]
    fn stream_split_never_leaks_shingles() {
        let msgs = vec![msg("tool", &format!("key {}", KEY))];
        let mut s = Scrubber::new(Mode::All, &msgs);
        let mid = KEY.len() / 2;
        let p1 = s.push_content("let me quote the key: ");
        let p2 = s.push_content(&KEY[..mid]);
        let p3 = s.push_content(&KEY[mid..]);
        let p4 = s.push_content(" now you know it");
        let tail = s.finish_content("");
        let all = format!("{}{}{}{}{}", p1, p2, p3, p4, tail);
        assert!(!all.contains(KEY), "leak across deltas: {}", all);
        for sh in KEY.as_bytes().windows(SHINGLE_LEN) {
            let sh = std::str::from_utf8(sh).unwrap();
            assert!(!all.contains(sh), "shingle leak ({}): {}", sh, all);
        }
        assert!(all.contains(PROV_MARKER));
        assert_eq!(all.matches(PROV_MARKER).count(), 1, "all: {}", all);
    }

    #[test]
    fn stream_and_full_agree() {
        let msgs = vec![msg("tool", &format!("keys {} and {}", KEY, AWS))];
        let text = format!("first {} then {} end", KEY, AWS);
        let mut s = Scrubber::new(Mode::All, &msgs);
        let streamed = format!("{}{}", s.push_content(&text[..20]), s.finish_content(&text[20..]));
        let full = s.scrub_full(&text);
        assert_eq!(streamed, full, "streamed={} full={}", streamed, full);
    }

    #[test]
    fn reasoning_and_content_fields_are_independent() {
        let msgs = vec![msg("tool", &format!("key {}", KEY))];
        let mut s = Scrubber::new(Mode::All, &msgs);
        let r = format!("thinking about {} carefully", KEY);
        let r1 = s.push_reasoning(&r[..12]);
        let r2 = s.finish_reasoning(&r[12..]);
        let r_all = format!("{}{}", r1, r2);
        assert!(!r_all.contains(KEY), "leak: {}", r_all);
        assert!(r_all.contains(PROV_MARKER));
        // content path still works after reasoning finished
        let c_all = s.finish_content(&format!("and {} again", KEY));
        assert!(!c_all.contains(KEY), "leak: {}", c_all);
    }

    #[test]
    fn force_flush_redacts_long_open_run() {
        let mut s = Scrubber::new(Mode::Pattern, &[]);
        let chunk: String = "Qw3Er5Ty7Ui9Op1As2Df4Gh6Jk8Lz0XcVbNm".chars().cycle().take(6000).collect();
        let mut text = String::from("seed ");
        text.push_str(&chunk);
        let out = s.push_content(&text);
        assert!(out.contains(PAT_MARKER), "force flush must redact: {}", &out[..out.len().min(200)]);
        assert!(out.len() < text.len(), "held tail expected");
        assert!(!out.contains(&chunk[..64]), "plaintext leak of run prefix");
        // the remainder is flushed (redacted) at finish
        let rest = s.finish_content("");
        assert!(!format!("{}{}", out, rest).contains(&chunk[..64]));
    }

    #[test]
    fn markers_and_boot_line_names_mode() {
        assert_eq!(PROV_MARKER, "[redacted:tool-secret]");
        assert_eq!(PAT_MARKER, "[redacted:key-shape]");
        assert!(boot_line(Mode::Off).contains("off"));
        assert!(boot_line(Mode::Provenance).contains("provenance"));
        assert!(boot_line(Mode::Pattern).contains("pattern"));
        assert!(boot_line(Mode::All).contains("all"));
    }

    #[test]
    fn multibyte_text_never_panics_on_cuts() {
        let msgs = vec![msg("tool", &format!("key {}", KEY))];
        let mut s = Scrubber::new(Mode::All, &msgs);
        // multibyte text around a secret: cuts must floor to char boundaries
        let text = format!("héllo wörld — {} — über alles", KEY);
        let out1 = s.push_content(&text[..40]);
        let out2 = s.push_content(&text[40..]);
        let out3 = s.finish_content("");
        let all = format!("{}{}{}", out1, out2, out3);
        assert!(!all.contains(KEY));
        assert!(all.contains("héllo"));
        assert!(all.contains("über"));
    }

    #[test]
    fn extract_candidates_bounded_and_sorted() {
        let mut msgs = vec![];
        for i in 0..40 {
            msgs.push(msg("tool", &format!("key{} sk-ITEM{i:03}-abcdefghijklmnopqrstuvwxyz", i)));
        }
        let needles = extract_candidates(&msgs);
        assert!(needles.len() <= MAX_NEEDLES);
        let mut sorted = needles.clone();
        sorted.sort();
        assert_eq!(needles, sorted);
        // shingles present: a 10-char window of candidate 0 matches
        let ac = build_ac(&needles);
        let probe = "abcdefghijklmnopqrstuvwxyz";
        assert!(!ac.scan(probe.as_bytes()).is_empty());
    }
}
