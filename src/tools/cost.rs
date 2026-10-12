//! R3, the cost-model reading of one `<tool_call>` block (`--tool-quote-pairing cost`).
//!
//! S-B9-TOOLPARSE3 T4. The expert's idea (PLAN/EXPERT_REPLY_TOOLCALL_COSTMODEL_2026-10-10.md, 4.5):
//! instead of deciding the end of every value greedily, score WHOLE segmentations of the block and
//! take the cheapest. A segmentation is one choice per value of which `</parameter>` closes it;
//! what the outer procedure of `parse_xml_calls` does after that choice is deterministic, so the
//! choices fix the calls. The calls are not built here: this module only returns, for every value
//! the cheapest segmentation reads, the offset of the close it picked (`Plan`), and the ordinary
//! outer loop of `parse_xml_calls` then builds the calls from those closes exactly as it does for
//! the other settings. A value the plan does not know (the block was too big, or the plan's model
//! of the block differs from the loop's) is ended by the v0.7.3 rule.
//!
//! Costs, in one table (`Weights`): R1 (the v0.7.3 rule) is the segmentation with zero SKIP cost,
//! every other reading pays for each acceptable close it passes over and earns it back by removing
//! evidence of a mis-cut (a value ending in tag text, a stray close inside a value, a duplicate or
//! undeclared parameter, a call to a tool that is not offered, an extra call). Ties go to fewer
//! calls, then to the earlier close (R1's).
//!
//! Deviations from the expert's pseudocode, each pinned by a test:
//!  * the skip accumulator keeps the maximum and the COUNT separately (the pseudocode's single
//!    variable under-charges when a dearer skip follows cheaper ones: our reading note 1);
//!  * the states are keyed exactly (tool, parameters given so far, current key), not Viterbi-style,
//!    so that the program can be checked against brute force; `Merge::Viterbi` is the expert's
//!    approximation and `tests::viterbi_is_exact_on_the_exhaustive_sets` reports how far it is;
//!  * a value that is never closed ends the path (the outer loop breaks there).
//!
//! # The work bound (S-B9-TOOLPARSE3 part B, independent review B1)
//!
//! Every kind of work the program does is charged to a counter of WORK UNITS, and the program stops
//! when the counter passes its limit: a block that spends `BLOCK_UNITS` is read by the v0.7.3 rule
//! alone (`Decline::Budget`), and the blocks of one response share `RESPONSE_UNITS`. The counter is
//! deterministic (no clock): the same text under the same catalogue is declined or planned the same
//! way on every machine and every load, which is what a SERVED rule needs. A unit is one elementary
//! step (an event found, a candidate close evaluated, a state merged, eight bytes walked). On a GB10
//! core a unit is 4 to 18 ns on the shapes that matter (`bound_tests::work_units_table` prints units
//! and clock side by side), so `RESPONSE_UNITS` is at most about 4 ms of one core; the report has the
//! table. What is charged:
//!  * the scan of the block for tags (bytes / 32 and the events found);
//!  * the profile of every parameter, built one candidate at a time and only as far as a search asks
//!    (bytes / 8 for the line walk, one unit per few tags);
//!  * every candidate close of every state (its follow, its fit, its successor), the keys given, the
//!    lookup of a state, and a tie-break by path (the length of both paths);
//!  * the scan of the rest of the block by the v0.7.3 rule (bytes / 32, once per parameter).
//! `bound_tests::the_cost_model_work_is_bounded_on_the_review_shapes` asserts the limit on the
//! review's shapes with catalogues of 4, 30 and 100 tools, and that the units charged cover an
//! independent count of the raw operations.
//!
//! How the work is kept small, none of it changing which reading is chosen (the equality with brute
//! force is a test, `the_bound_never_changes_the_cheapest_reading`, and the pinned counts of the
//! exhaustive sets are unchanged):
//!  * the tool catalogue is a name lookup built ONCE per response (`Ctx`), not a scan per transition
//!    (the review measured 825 ms for one response that offered 98 tools);
//!  * branch and bound: a greedy walk finds one complete reading first (a well-formed call costs 0, so
//!    its bound is 0 and the search visits one path); a path whose cost passes the bound is not
//!    extended, and the candidates of a value are built only until the cost of the closes passed over
//!    exceeds it. Ties are kept, so tie-breaks are the unbounded search's;
//!  * ties are broken by comparing paths only when cost and calls are equal, never per comparison
//!    (the review: a path rebuild on every comparison);
//!  * the exact search may spend a quarter of a block's units before the Viterbi merge takes over.
//!
//! What the bound costs in reach: a value with more than `Weights::candidates` (64) closes cannot be
//! ended at its real close by this model at all (it is not a candidate), so a document that quotes
//! more than about 30 two-parameter calls is read by the v0.7.3 rule whether the block is declined or
//! not; the bound only stops the model from spending its budget finding that out.

use std::cell::{Cell, OnceCell, RefCell};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;

use serde_json::Value;

use super::param_end_v073;

/// The weights of R3. ONE table; every field says which measurement supports it.
///
/// `Weights::EXPERT` is the expert's table, verbatim. `Weights::SHIPPED` is what
/// `--tool-quote-pairing cost` runs; where it differs from the expert's, `docs`/the report say why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Weights {
    /// Passing over an acceptable close whose follow is a next parameter, no catalogue opinion.
    pub skip_param: u32,
    /// The same when a catalogue is present and says the parameter fits (a dearer deviation).
    pub skip_param_cat: u32,
    /// Passing over an acceptable close whose follow is a next call.
    pub skip_call: u32,
    /// Every further acceptable close passed over inside the SAME value.
    pub skip_more: u32,
    /// Each call in the block beyond the first (a tie-breaker: the anti-phantom bias lives in
    /// `skip_call < skip_param`).
    pub call: u32,
    /// A call that ends without `</function>` (the existing missing-close repair).
    pub no_fclose: u32,
    /// A value that is empty after the format's own newlines are stripped.
    pub empty: u32,
    /// A value that contains any tag text.
    pub tag_content: u32,
    /// A value with an open tag never closed inside it (a prose mention, a quoted opening).
    pub dangling: u32,
    /// A value with a close tag that has no open before it inside the value (the first call's
    /// REAL close, swallowed).
    pub stray: u32,
    /// A value whose last line is a close tag while it also has a line that is not a tag.
    pub tail: u32,
    /// Per required parameter missing from a finished call.
    pub required: u32,
    /// A parameter name repeated within a call (catalogue-free).
    pub duplicate: u32,
    /// A parameter the catalogue does not declare for the tool.
    pub undeclared: u32,
    /// A call to a tool the catalogue does not offer.
    pub unoffered: u32,
    /// A chosen close the v0.7.3 lookahead would reject (R0-style fallback stays reachable).
    pub junk: u32,
    /// Candidate closes considered per value (R1's own choice is added when this is exhausted).
    pub candidates: usize,
}

impl Weights {
    /// The expert's table (4.5 of the follow-up), verbatim.
    pub const EXPERT: Weights = Weights {
        skip_param: 17, skip_param_cat: 20, skip_call: 10, skip_more: 2, call: 3, no_fclose: 2, empty: 2,
        tag_content: 1, dangling: 2, stray: 8, tail: 8, required: 2, duplicate: 30, undeclared: 30,
        unoffered: 40, junk: 100, candidates: 64,
    };
    /// What `--tool-quote-pairing cost` runs: the expert's table with the three weights of facts about a
    /// call (`duplicate`, `undeclared`, `unoffered`) brought BELOW the cost of the cheapest reading that
    /// swallows the call into an earlier value (tag_content + stray + dangling = 11).
    ///
    /// Why (measured, S-B9-TOOLPARSE3; `tests::benign_diffs_at_expert_weights`): at the expert's 30 / 30 /
    /// 40 the program reads a REAL second call to a tool the catalogue does not offer, and a real
    /// parameter the catalogue does not declare, as text of the previous argument, because the swallowed
    /// reading (11) is cheaper than the call's own evidence (43, 30). 104 of 1,640 parses of the logged
    /// benchmark calls changed (with the benchmark's own catalogue, 4 pairs of 820 rows:
    /// `send_email` + a call to a tool the request offered but the catalogue file did not hold). The
    /// swallow is dangerous, not merely wrong: the earlier call is executed with the later call's text
    /// in its argument. With 8 / 8 / 5 the benign corpus is byte-identical to v0.7.3 (0 differences) and the
    /// realistic corpus (iv) reads the same as at the expert's weights.
    pub const SHIPPED: Weights = Weights { duplicate: 8, undeclared: 8, unoffered: 5, ..Weights::EXPERT };
}

/// How states are merged. `Exact` keys a state by (open parameter, call, parameters given, current
/// key); `Viterbi` keeps one path per (open parameter, call): the expert's approximation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Merge { Exact, Viterbi }

/// Why no plan: the block holds no call tag (nothing to read), or it exceeded a budget (it is read by
/// the v0.7.3 rule; the caller counts these).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Decline { NoCall, Budget }

/// Work units one block may spend (the exact search and the Viterbi search together).
pub const BLOCK_UNITS: usize = 60_000;
/// Work units all the blocks of one response may spend together.
pub const RESPONSE_UNITS: usize = 200_000;
/// The clock time one work unit stands for on a GB10 core, in nanoseconds: the figure the texts that
/// state the bound in milliseconds are computed from (`work_units_table` prints the units and the
/// clock side by side; the report has the table).
pub const NOMINAL_NS_PER_UNIT: u64 = 20;

/// The bound of one response in milliseconds, as stated in the boot line, the option help and the report.
pub fn response_ms_nominal() -> u64 { (RESPONSE_UNITS as u64 * NOMINAL_NS_PER_UNIT).div_ceil(1_000_000) }

/// The exact search may spend this fraction (1/N) of a block's units before the program falls back
/// to the expert's Viterbi merge (independent review I4: one well-formed call with 16 distinct
/// parameters exceeded 8,192 exact states).
const EXACT_SHARE: usize = 4;
/// A block of fewer units than this left in the response is not read at all.
const MIN_UNITS: usize = 64;
const MAX_EVENTS: usize = 40_000;
const MAX_STATES: usize = 8_192;
const EXACT_STATES: usize = 2_048;
/// Bytes of block text one unit pays for when the text is scanned for tags or for the v0.7.3 rule.
const SCAN_BYTES: usize = 32;
/// Bytes one unit pays for when the text is walked line by line.
const WALK_BYTES: usize = 8;
/// Keys of a call's `given` list one unit pays for when they are compared or copied.
const GIVEN_PER_UNIT: usize = 8;

// ---------------------------------------------------------------------------------------------
// The request's catalogue and the response's budget
// ---------------------------------------------------------------------------------------------

/// What the program needs of one offered tool (derived from its schema on first use).
struct ToolInfo<'t> {
    /// `Some(keys)`: the catalogue declares exactly these parameters for the tool and says nothing
    /// that widens them; `None`: no opinion (the same silences as `continuation_fits`: no declared
    /// properties, `additionalProperties`, a schema that composes others).
    declared: Option<HashSet<&'t str>>,
    /// `parameters.required`, in the order the schema lists them.
    required: Vec<&'t str>,
}

impl<'t> ToolInfo<'t> {
    fn of(t: &'t Value) -> ToolInfo<'t> {
        let params = t.pointer("/function/parameters");
        let declared = params.and_then(|p| {
            let props = p.get("properties").and_then(Value::as_object).filter(|o| !o.is_empty())?;
            if matches!(p.get("additionalProperties"), Some(Value::Bool(true)) | Some(Value::Object(_))) { return None; }
            if ["anyOf", "oneOf", "allOf", "$ref", "patternProperties"].iter().any(|k| p.get(*k).is_some()) { return None; }
            Some(props.keys().map(String::as_str).collect::<HashSet<&str>>())
        });
        let required = t.pointer("/function/parameters/required").and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
        ToolInfo { declared, required }
    }
}

/// The tools of a request as a lookup by name (the first tool of a name wins, as the linear scan it
/// replaces did). Built once per response, on first use; the facts about one tool are derived the
/// first time the program asks about it.
pub(crate) struct Catalogue<'t> {
    by_name: HashMap<&'t str, (&'t Value, OnceCell<ToolInfo<'t>>)>,
}

impl<'t> Catalogue<'t> {
    fn new(tools: &'t [Value]) -> Catalogue<'t> {
        let mut by_name = HashMap::with_capacity(tools.len());
        for t in tools {
            let Some(name) = t.pointer("/function/name").and_then(Value::as_str) else { continue };
            by_name.entry(name).or_insert((t, OnceCell::new()));
        }
        Catalogue { by_name }
    }

    fn info(&self, name: &str) -> Option<&ToolInfo<'t>> {
        let (t, cell) = self.by_name.get(name)?;
        Some(cell.get_or_init(|| ToolInfo::of(t)))
    }
}

/// One response's cost-model context: the catalogue lookup (built on the first block that needs it)
/// and the work units the response's blocks may still spend.
pub struct Ctx<'t> {
    tools: Option<&'t [Value]>,
    cat: OnceCell<Catalogue<'t>>,
    left: Cell<usize>,
    used: Cell<usize>,
    declined: Cell<usize>,
}

impl<'t> Ctx<'t> {
    /// The context of one response with the default response budget.
    pub fn new(tools: Option<&'t [Value]>) -> Ctx<'t> { Ctx::with_budget(tools, RESPONSE_UNITS) }

    pub fn with_budget(tools: Option<&'t [Value]>, units: usize) -> Ctx<'t> {
        Ctx { tools: tools.filter(|t| !t.is_empty()), cat: OnceCell::new(), left: Cell::new(units), used: Cell::new(0), declined: Cell::new(0) }
    }

    /// Work units this response has spent in the cost model so far.
    pub fn used(&self) -> usize { self.used.get() }

    /// Blocks of this response the cost model declined (read by the v0.7.3 rule).
    pub fn declined(&self) -> usize { self.declined.get() }

    fn catalogue(&self) -> Option<&Catalogue<'t>> {
        let tools = self.tools?;
        Some(self.cat.get_or_init(|| Catalogue::new(tools)))
    }

    fn spend(&self, units: usize) {
        self.used.set(self.used.get().saturating_add(units));
        self.left.set(self.left.get().saturating_sub(units));
    }
}

/// Test instrument: the raw elementary operations the program performs (a byte scan counts a quarter
/// of an operation per byte, a line walk one per byte, a key compared, a path element rebuilt or an
/// event produced one), counted independently of the work units the program charges.
/// `bound_tests::the_cost_model_work_is_bounded_on_the_review_shapes` asserts that the units charged
/// are never less than 1/32 of this count, which is what keeps the work bound honest when a loop is
/// added without a charge.
#[cfg(test)]
thread_local! { pub(crate) static OPS: Cell<usize> = const { Cell::new(0) }; }

#[cfg(test)]
fn ops(n: usize) { OPS.with(|c| c.set(c.get() + n)); }
#[cfg(not(test))]
#[inline(always)]
fn ops(_n: usize) {}

// ---------------------------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind { FnOpen, FnClose, ParamOpen, ParamClose }

#[derive(Clone, Copy, Debug)]
struct Ev {
    kind: Kind,
    /// byte offset of the tag
    pos: usize,
    /// byte offset just past the tag (past the `>` for an open tag)
    end: usize,
    /// trimmed name / key range (opens only)
    name: (usize, usize),
}

const NONE: usize = usize::MAX;

fn is_ws(b: u8) -> bool { matches!(b, b'\n' | b'\r' | b' ' | b'\t') }

/// The tags of a block, in order: `<function=NAME>`, `</function>`, `<parameter=KEY>`,
/// `</parameter>`, plus a line-start `function=NAME>` when the block has no `<function=` at all
/// (the repair for a dropped `<`, with the boundary rule of `find_function_tag_mode`).
fn events(body: &str, wf_only: bool) -> Option<Vec<Ev>> {
    let b = body.as_bytes();
    let mut evs: Vec<Ev> = Vec::new();
    let mut i = 0usize;
    // the next '>' at or after the last query: monotone, so every '>' is found once
    let mut gt_at = 0usize;
    let mut gt_none = false;
    let next_gt = |from: usize, gt_at: &mut usize, gt_none: &mut bool| -> Option<usize> {
        if *gt_none { return None; }
        if *gt_at < from {
            match body[from..].find('>') {
                Some(r) => *gt_at = from + r,
                None => { *gt_none = true; return None; }
            }
        }
        Some(*gt_at)
    };
    while let Some(rel) = b[i..].iter().position(|&c| c == b'<') {
        let p = i + rel;
        let rest = &body[p..];
        let ev = if rest.starts_with("</parameter>") {
            Some(Ev { kind: Kind::ParamClose, pos: p, end: p + 12, name: (0, 0) })
        } else if rest.starts_with("</function>") {
            Some(Ev { kind: Kind::FnClose, pos: p, end: p + 11, name: (0, 0) })
        } else if rest.starts_with("<parameter=") || rest.starts_with("<function=") {
            let (kind, tl) = if rest.starts_with("<parameter=") { (Kind::ParamOpen, 11) } else { (Kind::FnOpen, 10) };
            match next_gt(p + tl, &mut gt_at, &mut gt_none) {
                Some(gt) => {
                    let raw = &body[p + tl..gt];
                    let lead = raw.len() - raw.trim_start().len();
                    let n0 = p + tl + lead;
                    Some(Ev { kind, pos: p, end: gt + 1, name: (n0, n0 + raw.trim().len()) })
                }
                None => None,
            }
        } else { None };
        match ev {
            Some(e) => { i = e.end; evs.push(e); }
            None => i = p + 1,
        }
        if evs.len() > MAX_EVENTS { return None; }
    }
    if !wf_only {
        // bare `function=NAME>` at the start of a line (or right after a close or a call tag, where the
        // outer loop's slice starts). `evs` is sorted and its events do not overlap, so "an event ends
        // exactly here" and "an event covers this byte" are binary searches, and the next '>' is found
        // with a monotone pointer (independent review I3: both were linear scans per candidate).
        let mut from = 0usize;
        let mut extra: Vec<Ev> = Vec::new();
        let (mut gt_at, mut gt_none) = (0usize, false);
        while let Some(r) = body[from..].find("function=") {
            let q = from + r;
            from = q + 9;
            let prev = if q == 0 { None } else { Some(b[q - 1]) };
            let k = evs.partition_point(|e| e.end <= q);
            let boundary = match prev {
                None => true,
                Some(c) if is_ws(c) => true,
                Some(b'>') => k > 0 && evs[k - 1].end == q && matches!(evs[k - 1].kind, Kind::ParamClose | Kind::FnOpen),
                _ => false,
            };
            if !boundary { continue; }
            if k < evs.len() && evs[k].pos <= q && q < evs[k].end { continue; }
            // the next '>' at or after q + 9
            if gt_none { break; }
            if gt_at < q + 9 {
                match body[q + 9..].find('>') {
                    Some(r2) => gt_at = q + 9 + r2,
                    None => { gt_none = true; break; }
                }
            }
            let gt = gt_at;
            let raw = &body[q + 9..gt];
            let lead = raw.len() - raw.trim_start().len();
            let n0 = q + 9 + lead;
            extra.push(Ev { kind: Kind::FnOpen, pos: q, end: gt + 1, name: (n0, n0 + raw.trim().len()) });
            if extra.len() > MAX_EVENTS { return None; }
        }
        if !extra.is_empty() {
            evs.extend(extra);
            evs.sort_by_key(|e| e.pos);
        }
    }
    Some(evs)
}

// ---------------------------------------------------------------------------------------------
// Components of a cost
// ---------------------------------------------------------------------------------------------

pub const N_COMP: usize = 13;
pub const COMP_NAMES: [&str; N_COMP] = [
    "skip", "empty", "tag_content", "stray", "dangling", "tail", "call", "no_fclose", "unoffered",
    "required", "duplicate", "undeclared", "junk",
];
const C_SKIP: usize = 0;
const C_EMPTY: usize = 1;
const C_TAGC: usize = 2;
const C_STRAY: usize = 3;
const C_DANG: usize = 4;
const C_TAIL: usize = 5;
const C_CALL: usize = 6;
const C_NOFC: usize = 7;
const C_UNOFF: usize = 8;
const C_REQ: usize = 9;
const C_DUP: usize = 10;
const C_UNDECL: usize = 11;
const C_JUNK: usize = 12;

type Comps = [u32; N_COMP];

fn total(c: &Comps) -> u32 { c.iter().sum() }

/// What the text of one value adds to a cost, by component (already weighted).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct VDelta { empty: u32, tagc: u32, stray: u32, dang: u32, tail: u32 }

impl VDelta {
    fn add_to(&self, c: &mut Comps) {
        c[C_EMPTY] += self.empty;
        c[C_TAGC] += self.tagc;
        c[C_STRAY] += self.stray;
        c[C_DANG] += self.dang;
        c[C_TAIL] += self.tail;
    }
}

/// Does the last line of `seg` (trailing white space aside) hold exactly one `</parameter>` or
/// `</function>`? Returns the answer and the bytes scanned back.
fn tail_is_close(seg: &str) -> (bool, usize) {
    let t = seg.trim_end();
    for tag in ["</parameter>", "</function>"] {
        if let Some(head) = t.strip_suffix(tag) {
            // the last line is `tag` iff only white space (not a newline) precedes it on its line
            let mut scanned = 0usize;
            for c in head.chars().rev() {
                if c == '\n' { return (true, scanned); }
                if !c.is_whitespace() { return (false, scanned); }
                scanned += c.len_utf8();
            }
            return (true, scanned);
        }
    }
    (false, 0)
}

// ---------------------------------------------------------------------------------------------
// The block
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fit { Yes, No, NoOpinion }

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Follow {
    NextParam(usize),
    NextCall { ev: usize, direct: bool },
    End,
    Invalid,
}

/// The state of a path whose value of parameter `pi` is about to be closed.
#[derive(Clone, Debug)]
struct St<'a> {
    pi: usize,
    /// the `<function=` event of the call this value belongs to
    tool: usize,
    /// keys given earlier in this call (sorted, unique); shared by the states one parent creates
    given: Rc<Vec<&'a str>>,
    cur: &'a str,
    comps: Comps,
    ncalls: u32,
}

#[derive(Clone, Debug)]
enum Out<'a> { Param(St<'a>), Done { comps: Comps, ncalls: u32 } }

#[derive(Clone, Debug)]
struct Trans<'a> {
    /// the close chosen for the open value (`NONE`: the value was never closed)
    close: usize,
    out: Out<'a>,
}

/// The candidate closes of one value and what ending the value at each of them costs, built one
/// candidate at a time in one forward pass over the text (cached per parameter, shared by every
/// state at that parameter): a search that is bounded asks for the first few candidates only.
struct Profile {
    /// (candidate close event, the cost of ending the value there), in order; at most
    /// `Weights::candidates`
    cands: RefCell<Vec<(usize, VDelta)>>,
    /// no candidate after those in `cands` will be produced (the window is full or the closes ran out)
    finished: Cell<bool>,
    /// the close the v0.7.3 rule picks when it lies beyond the window of candidates (computed on
    /// request, once)
    far: OnceCell<Option<(usize, VDelta)>>,
    scan: RefCell<ProfScan>,
}

/// The running state of a profile's forward pass.
struct ProfScan {
    /// the next candidate close event (`NONE`: no more closes)
    ci: usize,
    /// the value is "empty" while its text is newlines only: the first byte that is not
    first_non_nl: usize,
    /// open / close tags of the value so far
    fd: usize,
    pd: usize,
    stray: bool,
    /// events [pi + 1, folded) are counted above
    folded: usize,
    /// tags matched by complete lines; whether every line so far was a tag or blank
    k: usize,
    lines_ok: bool,
    /// start of the first line not yet classified
    line_from: usize,
    /// the last candidate produced
    last_ci: usize,
}

struct Block<'a, 'c, 't> {
    body: &'a str,
    evs: Vec<Ev>,
    cat: Option<&'c Catalogue<'t>>,
    w: &'c Weights,
    /// per event: the first open event (function or parameter) at or after it
    next_open: Vec<usize>,
    /// per event: the first ParamClose at or after it
    next_close: Vec<usize>,
    /// per event: is the text between it and the next event (or the end of the block) not blank
    junk_after: Vec<bool>,
    profiles: RefCell<HashMap<usize, Rc<Profile>>>,
    /// work units spent so far, and the most this block may spend now
    work: Cell<usize>,
    limit: Cell<usize>,
    /// the empty list of keys given (a call that has just begun)
    no_keys: Rc<Vec<&'a str>>,
    /// branch and bound: a path whose cost passes this cannot be the cheapest reading, so it is not
    /// extended. `u32::MAX` = no bound (the brute-force instruments); the program sets it to the
    /// cost of a complete reading it has already found (`greedy_bound`). Ties are kept, so the
    /// reading chosen, tie-breaks included, is the one the unbounded search chooses.
    bound: Cell<u32>,
    /// the greedy walk: the bound of a state is the cheapest way out found so far in it
    greedy_mode: Cell<bool>,
}

impl<'a, 'c, 't> Block<'a, 'c, 't> {
    fn new(body: &'a str, cat: Option<&'c Catalogue<'t>>, wf_only: bool, w: &'c Weights, limit: usize) -> Result<Block<'a, 'c, 't>, Decline> {
        let evs = events(body, wf_only).ok_or(Decline::Budget)?;
        let n = evs.len();
        ops(body.len() / 4 + 3 * n);
        let blk_work = body.len() / SCAN_BYTES + 3 * n;
        if blk_work > limit { return Err(Decline::Budget); }
        let mut next_open = vec![NONE; n + 1];
        let mut next_close = vec![NONE; n + 1];
        for i in (0..n).rev() {
            next_open[i] = if matches!(evs[i].kind, Kind::FnOpen | Kind::ParamOpen) { i } else { next_open[i + 1] };
            next_close[i] = if evs[i].kind == Kind::ParamClose { i } else { next_close[i + 1] };
        }
        let mut junk_after = vec![false; n];
        for i in 0..n {
            let to = if i + 1 < n { evs[i + 1].pos } else { body.len() };
            junk_after[i] = !body.as_bytes()[evs[i].end.min(to)..to].iter().all(|&c| is_ws(c));
        }
        Ok(Block { body, evs, cat, w, next_open, next_close, junk_after, profiles: RefCell::new(HashMap::new()),
                   work: Cell::new(blk_work), limit: Cell::new(limit), no_keys: Rc::new(Vec::new()), bound: Cell::new(u32::MAX), greedy_mode: Cell::new(false) })
    }

    fn has_call(&self) -> bool { self.evs.iter().any(|e| e.kind == Kind::FnOpen) }

    fn name(&self, e: usize) -> &'a str {
        let (a, b) = self.evs[e].name;
        &self.body[a..b]
    }

    /// Charge work; false once the block's limit is passed (and from then on).
    fn charge(&self, units: usize) -> bool {
        let w = self.work.get().saturating_add(units);
        self.work.set(w);
        w <= self.limit.get()
    }

    fn blown(&self) -> bool { self.work.get() > self.limit.get() }

    // ---- the catalogue ----------------------------------------------------------------------

    fn offered(&self, name: &str) -> Option<bool> {
        self.cat.map(|c| c.by_name.contains_key(name))
    }

    /// Does the catalogue declare `key` for `tool`? `None` = no opinion (the same silences as
    /// `continuation_fits`: no catalogue, a tool that is not offered, no declared properties,
    /// `additionalProperties`, a schema that composes others).
    fn declared(&self, tool: &str, key: &str) -> Option<bool> {
        let t = self.cat?.info(tool)?;
        t.declared.as_ref().map(|d| d.contains(key))
    }

    fn required_missing(&self, tool_ev: usize, given: &[&str], cur: Option<&str>) -> u32 {
        let Some(cat) = self.cat else { return 0 };
        let Some(t) = cat.info(self.name(tool_ev)) else { return 0 };
        ops(t.required.len() * (given.len() + 1));
        self.charge(1 + t.required.len() * (given.len() / GIVEN_PER_UNIT + 1) / 4);
        t.required.iter().filter(|k| !given.contains(k) && Some(**k) != cur).count() as u32
    }

    // ---- follow / fits ---------------------------------------------------------------------

    /// What the outer procedure does after the event `prev` (a close, or a call tag with no
    /// parameter right after it): the structure that follows and whether text that is not blank
    /// sits in between.
    fn follow(&self, prev: usize) -> (Follow, bool) {
        let e = prev + 1;
        let junk0 = self.junk_after[prev];
        if e >= self.evs.len() { return (Follow::End, junk0); }
        match self.evs[e].kind {
            Kind::ParamOpen => (Follow::NextParam(e), junk0),
            Kind::FnOpen => (Follow::NextCall { ev: e, direct: true }, junk0),
            Kind::ParamClose => (Follow::Invalid, true),
            Kind::FnClose => {
                let junk = junk0;
                let e2 = e + 1;
                if e2 >= self.evs.len() { return (Follow::End, junk || self.junk_after[e]); }
                match self.evs[e2].kind {
                    Kind::FnOpen => (Follow::NextCall { ev: e2, direct: false }, junk || self.junk_after[e]),
                    _ => (Follow::Invalid, true),
                }
            }
        }
    }

    fn fits(&self, tool: usize, given: &[&str], cur: Option<&str>, f: Follow) -> Fit {
        match f {
            Follow::NextParam(e) => {
                let key = self.name(e);
                if given.contains(&key) || Some(key) == cur { return Fit::No; }
                match self.declared(self.name(tool), key) {
                    Some(true) => Fit::Yes,
                    Some(false) => Fit::No,
                    None => Fit::NoOpinion,
                }
            }
            Follow::NextCall { ev, .. } => match self.offered(self.name(ev)) {
                Some(true) => Fit::Yes,
                Some(false) => Fit::No,
                None => Fit::NoOpinion,
            },
            Follow::End => Fit::Yes,
            Follow::Invalid => Fit::No,
        }
    }

    // ---- the cost of a value ----------------------------------------------------------------

    /// The parts of the cost of the value between parameter event `pi` and close event `ci`, computed
    /// directly (one walk over the text between them). The profile computes the same numbers
    /// incrementally for every candidate of a value at once; this is the definition, used for the
    /// far choice of the v0.7.3 rule and as the oracle of `tests::the_profile_equals_the_definition`.
    fn delta_direct(&self, pi: usize, ci: usize) -> Option<VDelta> {
        let start = self.evs[pi].end;
        let end = self.evs[ci].pos;
        let s = self.body[start..end].trim_matches('\n');
        let mut d = VDelta::default();
        if s.is_empty() { d.empty = self.w.empty; return Some(d); }
        let tags = &self.evs[pi + 1..ci];
        if tags.is_empty() { return Some(d); }
        ops(tags.len() + 1 + s.len());
        if !self.charge(tags.len() + 1 + s.len() / WALK_BYTES) { return None; }
        d.tagc = self.w.tag_content;
        if self.all_bare_tag_lines(s, tags) { return Some(d); }
        let (mut fd, mut pd) = (0usize, 0usize);
        let mut stray = false;
        for e in tags {
            match e.kind {
                Kind::FnOpen => fd += 1,
                Kind::ParamOpen => pd += 1,
                Kind::FnClose => if fd == 0 { stray = true } else { fd -= 1 },
                Kind::ParamClose => if pd == 0 { stray = true } else { pd -= 1 },
            }
        }
        if stray { d.stray = self.w.stray; }
        if fd > 0 || pd > 0 { d.dang = self.w.dangling; }
        let last = s.trim_end().rsplit('\n').next().unwrap_or("").trim();
        if last == "</parameter>" || last == "</function>" { d.tail = self.w.tail; }
        Some(d)
    }

    /// Is every (non-blank) line of `s` exactly one tag of the block? `tags` are the events inside
    /// the value (`s` is the value without its newlines).
    fn all_bare_tag_lines(&self, s: &str, tags: &[Ev]) -> bool {
        let s0 = s.as_ptr() as usize - self.body.as_ptr() as usize;
        let mut k = 0usize;
        let mut off = 0usize;
        for line in s.split('\n') {
            let l0 = off;
            off += line.len() + 1;
            let t = line.trim_matches(|c| c == ' ' || c == '\t' || c == '\r');
            if t.is_empty() { continue; }
            if k >= tags.len() { return false; }
            let lead = line.len() - line.trim_start_matches(|c| c == ' ' || c == '\t' || c == '\r').len();
            let a = s0 + l0 + lead;
            let b = a + t.len();
            if tags[k].pos != a || tags[k].end != b { return false; }
            k += 1;
        }
        k == tags.len()
    }

    /// The profile of the value that opens at event `pi` (created on first use, filled lazily).
    fn profile(&self, pi: usize) -> Option<Rc<Profile>> {
        if let Some(p) = self.profiles.borrow().get(&pi) { return Some(p.clone()); }
        let start = self.evs[pi].end;
        let b = self.body.as_bytes();
        let ci = self.next_close[pi + 1];
        // the value is "empty" while its text is newlines only
        let first_non_nl = if ci == NONE { start } else {
            start + b[start..].iter().position(|&c| c != b'\n').unwrap_or(b.len() - start)
        };
        ops((first_non_nl - start) / 4 + 1);
        if !self.charge((first_non_nl - start) / SCAN_BYTES + 1) { return None; }
        let p = Rc::new(Profile {
            cands: RefCell::new(Vec::new()),
            finished: Cell::new(ci == NONE),
            far: OnceCell::new(),
            scan: RefCell::new(ProfScan { ci, first_non_nl, fd: 0, pd: 0, stray: false, folded: pi + 1, k: 0, lines_ok: true, line_from: start, last_ci: NONE }),
        });
        self.profiles.borrow_mut().insert(pi, p.clone());
        Some(p)
    }

    /// The `j`-th candidate close of the value at `pi` and its cost: `Some(Some(..))`, or
    /// `Some(None)` when the value has fewer candidates, or `None` when the block ran out of work.
    fn cand(&self, pi: usize, prof: &Profile, j: usize) -> Option<Option<(usize, VDelta)>> {
        loop {
            if let Some(c) = prof.cands.borrow().get(j) { return Some(Some(*c)); }
            if prof.finished.get() { return Some(None); }
            self.extend(pi, prof)?;
        }
    }

    /// Produce the next candidate of a profile (one step of the forward pass).
    fn extend(&self, pi: usize, prof: &Profile) -> Option<()> {
        let w = self.w;
        let start = self.evs[pi].end;
        let b = self.body.as_bytes();
        let mut sc = prof.scan.borrow_mut();
        let ci = sc.ci;
        debug_assert!(ci != NONE);
        let end = self.evs[ci].pos;
        let before = sc.folded;
        while sc.folded < ci {
            match self.evs[sc.folded].kind {
                Kind::FnOpen => sc.fd += 1,
                Kind::ParamOpen => sc.pd += 1,
                Kind::FnClose => if sc.fd == 0 { sc.stray = true } else { sc.fd -= 1 },
                Kind::ParamClose => if sc.pd == 0 { sc.stray = true } else { sc.pd -= 1 },
            }
            sc.folded += 1;
        }
        ops(2 + sc.folded - before);
        if !self.charge(2 + (sc.folded - before) / 4) { return None; }
        let is_pad = |c: u8| c == b' ' || c == b'\t' || c == b'\r';
        // (start, end) of the line [from, to) without leading and trailing padding; empty when blank
        let trim = |from: usize, to: usize| -> (usize, usize) {
            let (mut a, mut z) = (from, to);
            while a < z && is_pad(b[a]) { a += 1; }
            while z > a && is_pad(b[z - 1]) { z -= 1; }
            (a, z)
        };
        let mut d = VDelta::default();
        if sc.first_non_nl >= end {
            d.empty = w.empty;
        } else if ci > pi + 1 {
            d.tagc = w.tag_content;
            // complete lines up to `end`, each classified once (a line that is not a tag ends the
            // walk for good: the text only grows, so nothing later can be bare)
            if sc.lines_ok {
                let from = sc.line_from;
                let mut p = from;
                while let Some(rel) = b[p..end].iter().position(|&c| c == b'\n') {
                    let le = p + rel;
                    let (a, z) = trim(p, le);
                    if a < z {
                        match self.evs.get(pi + 1 + sc.k) {
                            Some(e) if pi + 1 + sc.k < ci && e.pos == a && e.end == z => sc.k += 1,
                            _ => { sc.lines_ok = false; break; }
                        }
                    }
                    p = le + 1;
                }
                if sc.lines_ok { sc.line_from = p; }
                ops(end.saturating_sub(from));
                if !self.charge(end.saturating_sub(from) / WALK_BYTES) { return None; }
            }
            let bare = sc.lines_ok && {
                // the last line, up to the close
                let (a, z) = trim(sc.line_from, end);
                if a < z {
                    pi + 1 + sc.k < ci && matches!(self.evs.get(pi + 1 + sc.k), Some(e) if e.pos == a && e.end == z) && sc.k + 1 == ci - pi - 1
                } else {
                    sc.k == ci - pi - 1
                }
            };
            if !bare {
                if sc.stray { d.stray = w.stray; }
                if sc.fd > 0 || sc.pd > 0 { d.dang = w.dangling; }
                let (tail, scanned) = tail_is_close(&self.body[start..end]);
                ops(scanned + 1);
                if !self.charge(scanned / WALK_BYTES + 1) { return None; }
                if tail { d.tail = w.tail; }
            }
        }
        sc.last_ci = ci;
        let n = {
            let mut c = prof.cands.borrow_mut();
            c.push((ci, d));
            c.len()
        };
        let next = if ci + 1 < self.next_close.len() { self.next_close[ci + 1] } else { NONE };
        sc.ci = next;
        if next == NONE || n >= w.candidates { prof.finished.set(true); }
        Some(())
    }

    /// R1's own choice when it lies beyond the window of candidates: the window must be complete
    /// (`finished` with more closes after it).
    fn far(&self, pi: usize, prof: &Profile) -> Option<Option<(usize, VDelta)>> {
        if let Some(f) = prof.far.get() { return Some(*f); }
        let (more, last_ci) = { let sc = prof.scan.borrow(); (sc.ci != NONE, sc.last_ci) };
        let mut r = None;
        if more {
            // the candidate bound is exhausted: R1's own choice is the last candidate
            let start = self.evs[pi].end;
            ops((self.body.len() - start) / 4 + 1);
            if !self.charge((self.body.len() - start) / SCAN_BYTES + 1) { return None; }
            if let Some(off) = param_end_v073(&self.body[start..]) {
                let pos = start + off;
                if let Ok(j) = self.evs.binary_search_by_key(&pos, |e| e.pos) {
                    if self.evs[j].kind == Kind::ParamClose && j > last_ci {
                        r = Some((j, self.delta_direct(pi, j)?));
                    }
                }
            }
        }
        let _ = prof.far.set(r);
        Some(r)
    }

    // ---- entering calls and moving on --------------------------------------------------------

    fn call_costs(&self, ev: usize, direct: bool, first: bool, comps: &mut Comps, ncalls: &mut u32) {
        if !first { comps[C_CALL] += self.w.call; }
        if direct { comps[C_NOFC] += self.w.no_fclose; }
        if self.offered(self.name(ev)) == Some(false) { comps[C_UNOFF] += self.w.unoffered; }
        *ncalls += 1;
    }

    /// Settle what comes after `prev` (a chosen close, or the tag of a call without a parameter
    /// right after it): the outcome is the next open value, or the end of the block. `given` are the
    /// keys given so far and `next_given` the same with `cur` added (computed once per state by the
    /// caller and shared by every state this state creates).
    fn advance(&self, mut comps: Comps, mut ncalls: u32, mut tool: usize, given_in: &Rc<Vec<&'a str>>, next_given_in: &Rc<Vec<&'a str>>,
               cur_in: Option<&'a str>, mut prev: usize, mut first_kind: Option<(Follow, bool)>) -> Option<Out<'a>> {
        let mut given: &Rc<Vec<&'a str>> = given_in;
        let mut next_given: &Rc<Vec<&'a str>> = next_given_in;
        let mut cur = cur_in;
        loop {
            ops(1 + given.len());
            if !self.charge(2 + given.len() / GIVEN_PER_UNIT) { return None; }
            let (mut f, junk) = match first_kind.take() { Some(x) => x, None => self.follow(prev) };
            if junk || f == Follow::Invalid { comps[C_JUNK] += self.w.junk; }
            if f == Follow::Invalid {
                // the outer procedure goes on from the next open tag, whatever came in between
                f = match self.next_open[prev + 1] {
                    NONE => Follow::End,
                    e if self.evs[e].kind == Kind::ParamOpen => Follow::NextParam(e),
                    e => Follow::NextCall { ev: e, direct: true },
                };
            }
            match f {
                Follow::NextParam(e) => {
                    let key = self.name(e);
                    if given.contains(&key) || Some(key) == cur { comps[C_DUP] += self.w.duplicate; }
                    else if self.declared(self.name(tool), key) == Some(false) { comps[C_UNDECL] += self.w.undeclared; }
                    return Some(Out::Param(St { pi: e, tool, given: next_given.clone(), cur: key, comps, ncalls }));
                }
                Follow::End | Follow::Invalid => {
                    comps[C_REQ] += self.w.required * self.required_missing(tool, given, cur);
                    return Some(Out::Done { comps, ncalls });
                }
                Follow::NextCall { ev, direct } => {
                    comps[C_REQ] += self.w.required * self.required_missing(tool, given, cur);
                    // the outer loop skips a call tag with no name and looks for the next call tag
                    let (mut ev, mut direct) = (ev, direct);
                    while self.name(ev).is_empty() {
                        let mut j = ev + 1;
                        while j < self.evs.len() && self.evs[j].kind != Kind::FnOpen { j += 1; }
                        if !self.charge(j - ev) { return None; }
                        if j >= self.evs.len() { return Some(Out::Done { comps, ncalls }); }
                        ev = j;
                        direct = true;
                    }
                    self.call_costs(ev, direct, false, &mut comps, &mut ncalls);
                    tool = ev; given = &self.no_keys; next_given = &self.no_keys; cur = None;
                    if ev + 1 < self.evs.len() && self.evs[ev + 1].kind == Kind::ParamOpen {
                        let p = ev + 1;
                        return Some(Out::Param(St { pi: p, tool, given: self.no_keys.clone(), cur: self.name(p), comps, ncalls }));
                    }
                    prev = self.after_json_body(ev);
                }
            }
        }
    }

    /// The first call of the block.
    fn start(&self) -> Option<Out<'a>> {
        let f = self.evs.iter().position(|e| e.kind == Kind::FnOpen && !self.name_empty(e))?;
        let mut comps = [0u32; N_COMP];
        let mut ncalls = 0u32;
        self.call_costs(f, false, true, &mut comps, &mut ncalls);
        if f + 1 < self.evs.len() && self.evs[f + 1].kind == Kind::ParamOpen {
            let p = f + 1;
            return Some(Out::Param(St { pi: p, tool: f, given: self.no_keys.clone(), cur: self.name(p), comps, ncalls }));
        }
        let none = self.no_keys.clone();
        self.advance(comps, ncalls, f, &none, &none, None, self.after_json_body(f), None)
    }

    fn name_empty(&self, e: &Ev) -> bool { e.name.0 == e.name.1 }

    /// A call that carries no `<parameter=` but a JSON body, followed by ONE stray `</parameter>` (the
    /// model imitates the JSON history form and closes with the tag it also sees): the outer loop strips
    /// that close (`strip_stray_param_close`), so it is no evidence of a mis-cut. Returns the event to
    /// continue from: the stray close, or the call tag itself (independent review I5: without this the
    /// stray close was INVALID, junk 100, and the swallow reading won).
    fn after_json_body(&self, call: usize) -> usize {
        match self.evs.get(call + 1) {
            // (bare `function=` events can overlap what follows them: their name runs to the next '>')
            Some(e) if e.kind == Kind::ParamClose && e.pos >= self.evs[call].end => {
                let t = self.body[self.evs[call].end..e.pos].trim_start();
                if t.starts_with('{') || t.starts_with('[') { call + 1 } else { call }
            }
            _ => call,
        }
    }

    /// One complete reading, found by always taking the cheapest way out of each state: its cost is
    /// an upper bound of the cheapest reading (a well-formed call costs nothing, so its bound is 0
    /// and the search below visits one path). `None`: the block ran out of work.
    fn greedy(&self) -> Option<Reading> {
        self.greedy_mode.set(true);
        let r = self.greedy_walk();
        self.greedy_mode.set(false);
        r
    }

    fn greedy_walk(&self) -> Option<Reading> {
        let mut o = self.start()?;
        let mut closes: Vec<(usize, usize)> = Vec::new();
        loop {
            match o {
                Out::Done { comps, ncalls } => return Some(reading_of(comps, ncalls, closes)),
                Out::Param(st) => {
                    let ts = self.transitions(&st)?;
                    let best = ts.into_iter().min_by_key(|t| match &t.out {
                        Out::Param(ns) => (total(&ns.comps), ns.ncalls),
                        Out::Done { comps, ncalls } => (total(comps), *ncalls),
                    })?;
                    if best.close != NONE { closes.push((self.evs[st.pi].end, self.evs[best.close].pos)); }
                    o = best.out;
                }
            }
        }
    }

    /// Every way out of the open value of `st`: one transition per candidate close.
    fn transitions(&self, st: &St<'a>) -> Option<Vec<Trans<'a>>> {
        ops(1 + st.given.len());
        if !self.charge(4 + st.given.len() / GIVEN_PER_UNIT) { return None; }
        let prof = self.profile(st.pi)?;
        if self.next_close[st.pi + 1] == NONE {
            // never closed: the outer loop breaks here and the call ends with what it has
            let mut comps = st.comps;
            comps[C_REQ] += self.w.required * self.required_missing(st.tool, &st.given, Some(st.cur));
            return Some(vec![Trans { close: NONE, out: Out::Done { comps, ncalls: st.ncalls } }]);
        }
        let mut res: Vec<Trans<'a>> = Vec::with_capacity(8);
        // the keys given once this value is closed: the same for every candidate close
        let next_given: Rc<Vec<&'a str>> = if st.given.contains(&st.cur) { st.given.clone() } else {
            let mut g: Vec<&'a str> = st.given.to_vec();
            g.push(st.cur);
            g.sort_unstable();
            Rc::new(g)
        };
        let (n_skips, max_skip) = (Cell::new(0u32), Cell::new(0u32));
        let static_bound = self.bound.get();
        let greedy = self.greedy_mode.get();
        // (greedy walk only) the cheapest successor found so far in this state
        let best_t = Cell::new(u32::MAX);
        let bound_now = || if greedy { static_bound.min(best_t.get()) } else { static_bound };
        // the cost of the closes passed over so far only grows: once it is past the bound, so is every
        // later candidate
        let mut dead_tail = false;
        let eval = |ci: usize, d: &VDelta, res: &mut Vec<Trans<'a>>, dead_tail: &mut bool| -> bool {
            ops(1 + st.given.len());
            if !self.charge(6 + st.given.len() / GIVEN_PER_UNIT) { return false; }
            let (f, junk) = self.follow(ci);
            let fit = self.fits(st.tool, &st.given, Some(st.cur), f);
            let accept = !junk && f != Follow::Invalid && fit != Fit::No;
            let mut comps = st.comps;
            if n_skips.get() > 0 { comps[C_SKIP] += max_skip.get() + self.w.skip_more * (n_skips.get() - 1); }
            if comps[C_SKIP] > bound_now() { *dead_tail = true; return true; }
            d.add_to(&mut comps);
            if total(&comps) <= bound_now() {
                let Some(out) = self.advance(comps, st.ncalls, st.tool, &st.given, &next_given, Some(st.cur), ci, Some((f, junk))) else { return false };
                let t = match &out { Out::Param(ns) => total(&ns.comps), Out::Done { comps, .. } => total(comps) };
                if t <= bound_now() {
                    best_t.set(best_t.get().min(t));
                    res.push(Trans { close: ci, out });
                }
            }
            if accept {
                let sc = match f {
                    Follow::NextParam(_) => if self.cat.is_some() && fit == Fit::Yes { self.w.skip_param_cat } else { self.w.skip_param },
                    Follow::NextCall { .. } => self.w.skip_call,
                    _ => 0,
                };
                if sc > 0 { n_skips.set(n_skips.get() + 1); max_skip.set(max_skip.get().max(sc)); }
            }
            true
        };
        let mut j = 0usize;
        loop {
            // the cost of the closes passed over only grows: past the bound, no later candidate is
            // built (building one can walk the whole value)
            if n_skips.get() > 0 && max_skip.get() + self.w.skip_more * (n_skips.get() - 1) > bound_now() { dead_tail = true; break; }
            let Some((ci, d)) = self.cand(st.pi, &prof, j)? else { break };
            if !eval(ci, &d, &mut res, &mut dead_tail) { return None; }
            if dead_tail { break; }
            j += 1;
        }
        if !dead_tail {
            if let Some((fj, d)) = self.far(st.pi, &prof)? {
                if !eval(fj, &d, &mut res, &mut dead_tail) { return None; }
            }
        }
        Some(res)
    }
}

// ---------------------------------------------------------------------------------------------
// The program
// ---------------------------------------------------------------------------------------------

/// What the program found: for every value it reads, the byte offset (in the block body) of the
/// close it picked, keyed by the offset where the value starts (just past the `>` of its tag).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub closes: HashMap<usize, usize>,
    pub reading: Reading,
    /// work units the block spent (both searches, if the exact one gave up)
    pub units: usize,
}

/// One complete reading of a block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reading {
    pub comps: [u32; N_COMP],
    pub total: u32,
    pub ncalls: u32,
    /// (value start, close position) per value that was closed
    pub closes: Vec<(usize, usize)>,
}

impl Reading {
    /// `skip=17 stray=8 ...` for the non-zero parts, then `= total`.
    pub fn describe(&self) -> String {
        let mut s = String::new();
        for (i, n) in COMP_NAMES.iter().enumerate() {
            if self.comps[i] > 0 { s.push_str(&format!("{n}={} ", self.comps[i])); }
        }
        format!("{}= {} ({} call{})", s, self.total, self.ncalls, if self.ncalls == 1 { "" } else { "s" })
    }
}

struct Node<'a> {
    st: St<'a>,
    back: Option<(usize, usize)>,
}

fn reading_of(comps: Comps, ncalls: u32, closes: Vec<(usize, usize)>) -> Reading {
    Reading { total: total(&comps), comps, ncalls, closes }
}

/// The cheapest reading of `body`, by the dynamic program. `None`: no call tag in the block, or a
/// budget was exceeded (the caller reads the block by the v0.7.3 rule). With `Merge::Exact` the
/// search keeps at most `EXACT_STATES` states (and a quarter of the block's work units) and then
/// reruns with the Viterbi merge.
pub fn plan(body: &str, tools: Option<&[Value]>, wf_only: bool, w: &Weights, merge: Merge) -> Option<Plan> {
    plan_ex(body, tools, wf_only, w, merge).ok()
}

/// `plan`, saying why there is none.
pub fn plan_ex(body: &str, tools: Option<&[Value]>, wf_only: bool, w: &Weights, merge: Merge) -> Result<Plan, Decline> {
    plan_metered(body, tools, wf_only, w, merge).0
}

/// `plan_ex` that also returns the work units spent (a decline reports what was spent before the
/// program gave up). A fresh context: the response budget is not shared.
pub fn plan_metered(body: &str, tools: Option<&[Value]>, wf_only: bool, w: &Weights, merge: Merge) -> (Result<Plan, Decline>, usize) {
    let ctx = Ctx::new(tools);
    let r = plan_in(&ctx, body, wf_only, w, merge);
    (r, ctx.used())
}

/// The program on one block of a response, spending the response's budget.
pub(crate) fn plan_in<'t>(ctx: &Ctx<'t>, body: &str, wf_only: bool, w: &Weights, merge: Merge) -> Result<Plan, Decline> {
    let limit = BLOCK_UNITS.min(ctx.left.get());
    if limit < MIN_UNITS || body.len() / SCAN_BYTES > limit { return Err(Decline::Budget); }
    let blk = match Block::new(body, ctx.catalogue(), wf_only, w, limit) {
        Ok(b) => b,
        Err(d) => { ctx.spend(body.len() / SCAN_BYTES + 1); return Err(d); }
    };
    if !blk.has_call() { ctx.spend(blk.work.get()); return Err(Decline::NoCall); }
    let greedy = match blk.greedy() {
        Some(g) if !blk.blown() => g,
        _ => { ctx.spend(blk.work.get()); return Err(Decline::Budget); }
    };
    blk.bound.set(greedy.total);
    let r = match merge {
        Merge::Exact => {
            blk.limit.set((blk.work.get() + limit / EXACT_SHARE).min(limit));
            match run(&blk, Merge::Exact, EXACT_STATES, &greedy) {
                Err(Decline::Budget) => {
                    blk.limit.set(limit);
                    run(&blk, Merge::Viterbi, MAX_STATES, &greedy)
                }
                r => r,
            }
        }
        Merge::Viterbi => run(&blk, Merge::Viterbi, MAX_STATES, &greedy),
    };
    let units = blk.work.get();
    ctx.spend(units);
    match r {
        Ok(mut p) => { p.units = units; Ok(p) }
        Err(d) => Err(d),
    }
}

fn run<'a>(blk: &Block<'a, '_, '_>, merge: Merge, max_states: usize, greedy: &Reading) -> Result<Plan, Decline> {
    let mut arena: Vec<Node> = Vec::new();
    let mut buckets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    // state key -> first node of that key; `chain[n]` is the next node with the same key (a hash
    // collision chains; `same` decides)
    let mut index: HashMap<(usize, usize, u64), usize> = HashMap::new();
    let mut chain: Vec<usize> = Vec::new();
    let mut best_done: Option<(Comps, u32, Option<(usize, usize)>)> = None;
    let mut states = 0usize;

    let path = |arena: &Vec<Node>, mut at: Option<(usize, usize)>| -> Vec<usize> {
        let mut v = Vec::new();
        while let Some((n, c)) = at { v.push(c); at = arena[n].back; }
        v.reverse();
        v
    };
    // is (cost, ncalls, path a) strictly better than (cost, ncalls, path b)? The paths are rebuilt
    // only on an exact tie of cost and calls (and charged), never per comparison.
    let better = |arena: &Vec<Node>, ac: u32, an: u32, ab: Option<(usize, usize)>, bc: u32, bn: u32, bb: Option<(usize, usize)>| -> bool {
        if ac != bc { return ac < bc; }
        if an != bn { return an < bn; }
        let (ap, bp) = (path(arena, ab), path(arena, bb));
        ops(ap.len() + bp.len());
        blk.charge(ap.len() + bp.len());
        ap < bp
    };
    let key_of = |st: &St| -> (usize, usize, u64) {
        use std::hash::{Hash, Hasher};
        if merge == Merge::Viterbi { return (st.pi, st.tool, 0); }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        st.given.hash(&mut h);
        st.cur.hash(&mut h);
        (st.pi, st.tool, h.finish())
    };

    match blk.start().ok_or(Decline::Budget)? {
        Out::Done { comps, ncalls } => best_done = Some((comps, ncalls, None)),
        Out::Param(st) => {
            let key = key_of(&st);
            buckets.entry(st.pi).or_default().push(0);
            index.insert(key, 0);
            chain.push(NONE);
            arena.push(Node { st, back: None });
        }
    }
    while let Some((_, nodes)) = buckets.pop_first() {
        for ni in nodes {
            if blk.blown() { return Err(Decline::Budget); }
            let trans = blk.transitions(&arena[ni].st).ok_or(Decline::Budget)?;
            for t in trans {
                if !blk.charge(2) { return Err(Decline::Budget); }
                let back = Some((ni, t.close));
                match t.out {
                    Out::Done { comps, ncalls } => {
                        let take = match &best_done {
                            None => true,
                            Some((bc, bn, bb)) => better(&arena, total(&comps), ncalls, back, total(bc), *bn, *bb),
                        };
                        if take { best_done = Some((comps, ncalls, back)); }
                    }
                    Out::Param(ns) => {
                        let key = key_of(&ns);
                        let same = |a: &St, b: &St| a.tool == b.tool && (merge == Merge::Viterbi || (a.given == b.given && a.cur == b.cur));
                        let mut existing = None;
                        let mut at = index.get(&key).copied().unwrap_or(NONE);
                        while at != NONE {
                            if same(&arena[at].st, &ns) { existing = Some(at); break; }
                            at = chain[at];
                        }
                        if let Some(ex) = existing {
                            let (ac, bc) = (total(&ns.comps), total(&arena[ex].st.comps));
                            if better(&arena, ac, ns.ncalls, back, bc, arena[ex].st.ncalls, arena[ex].back) {
                                arena[ex] = Node { st: ns, back };
                            }
                        } else {
                            states += 1;
                            if states > max_states { return Err(Decline::Budget); }
                            ops(1 + ns.given.len());
                            if !blk.charge(3 + ns.given.len() / GIVEN_PER_UNIT) { return Err(Decline::Budget); }
                            buckets.entry(ns.pi).or_default().push(arena.len());
                            chain.push(index.insert(key, arena.len()).unwrap_or(NONE));
                            arena.push(Node { st: ns, back });
                        }
                    }
                }
            }
        }
    }
    if blk.blown() { return Err(Decline::Budget); }
    let Some((comps, ncalls, back)) = best_done else {
        // every path of the merged search passed the bound (the Viterbi merge can replace the state the
        // greedy reading went through by one that is cheaper so far and dearer in the end): the greedy
        // reading is a complete reading and not dearer than the bound
        let map: HashMap<usize, usize> = greedy.closes.iter().copied().collect();
        return Ok(Plan { closes: map, reading: greedy.clone(), units: blk.work.get() });
    };
    let mut closes = Vec::new();
    let mut at = back;
    while let Some((n, c)) = at {
        if c != NONE { closes.push((blk.evs[arena[n].st.pi].end, blk.evs[c].pos)); }
        at = arena[n].back;
    }
    closes.reverse();
    let map: HashMap<usize, usize> = closes.iter().copied().collect();
    Ok(Plan { closes: map, reading: reading_of(comps, ncalls, closes), units: blk.work.get() })
}

/// Every complete reading of `body` (brute force over the same transitions, no merging), at most
/// `limit` of them. For tests and for the cost breakdowns the report prints: the program above
/// must find the cheapest of these.
pub fn all_readings(body: &str, tools: Option<&[Value]>, wf_only: bool, w: &Weights, limit: usize) -> Option<Vec<Reading>> {
    let ctx = Ctx::new(tools);
    let blk = Block::new(body, ctx.catalogue(), wf_only, w, usize::MAX / 4).ok()?;
    if !blk.has_call() { return None; }
    let mut out: Vec<Reading> = Vec::new();
    fn go<'a>(blk: &Block<'a, '_, '_>, o: Out<'a>, closes: &mut Vec<(usize, usize)>, out: &mut Vec<Reading>, limit: usize) -> bool {
        match o {
            Out::Done { comps, ncalls } => { out.push(reading_of(comps, ncalls, closes.clone())); out.len() < limit }
            Out::Param(st) => {
                let Some(ts) = blk.transitions(&st) else { return false };
                for t in ts {
                    let pushed = t.close != NONE;
                    if pushed { closes.push((blk.evs[st.pi].end, blk.evs[t.close].pos)); }
                    let go_on = go(blk, t.out, closes, out, limit);
                    if pushed { closes.pop(); }
                    if !go_on { return false; }
                }
                true
            }
        }
    }
    let start = blk.start()?;
    let mut closes = Vec::new();
    if !go(&blk, start, &mut closes, &mut out, limit) && out.len() < limit { return None; }
    Some(out)
}

/// The cost of ONE given segmentation: `choose(value start offset, candidate close positions)`
/// returns the close it picks. `None` when a pick is not a candidate.
pub fn score(body: &str, tools: Option<&[Value]>, wf_only: bool, w: &Weights,
             choose: &dyn Fn(usize, &[usize]) -> Option<usize>) -> Option<Reading> {
    let ctx = Ctx::new(tools);
    let blk = Block::new(body, ctx.catalogue(), wf_only, w, usize::MAX / 4).ok()?;
    if !blk.has_call() { return None; }
    let mut closes = Vec::new();
    let mut o = blk.start()?;
    loop {
        match o {
            Out::Done { comps, ncalls } => return Some(reading_of(comps, ncalls, closes)),
            Out::Param(st) => {
                let ts = blk.transitions(&st)?;
                let vs = blk.evs[st.pi].end;
                let cands: Vec<usize> = ts.iter().filter(|t| t.close != NONE).map(|t| blk.evs[t.close].pos).collect();
                if cands.is_empty() { o = ts.into_iter().next()?.out; continue; }
                let pick = choose(vs, &cands)?;
                let t = ts.into_iter().find(|t| t.close != NONE && blk.evs[t.close].pos == pick)?;
                closes.push((vs, pick));
                o = t.out;
            }
        }
    }
}

/// Plan for `parse_xml_calls`: the picks only. A block that exceeded its budget is counted in the
/// context (the caller reports it); `None` means the v0.7.3 rule reads the block.
pub(super) fn closes_for(body: &str, ctx: &Ctx, wf_only: bool, w: &Weights) -> Option<HashMap<usize, usize>> {
    match plan_in(ctx, body, wf_only, w, Merge::Exact) {
        Ok(p) => Some(p.closes),
        Err(Decline::NoCall) => None,
        Err(Decline::Budget) => { ctx.declined.set(ctx.declined.get() + 1); None }
    }
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
pub(in crate::tools) mod tests {
    use super::*;
    use crate::tools::tests as t;
    use crate::tools::{parse_xml_calls_plan, QuotePairing};
    use crate::tokenizer::ToolCall;

    /// The catalogue of the expert's worked examples (section 5 of the request).
    pub(in crate::tools) fn expert_catalogue() -> Vec<Value> {
        let tool = |name: &str, props: &[&str], req: &[&str]| serde_json::json!({"type": "function", "function": {
            "name": name, "parameters": {"type": "object",
            "properties": props.iter().map(|p| (p.to_string(), serde_json::json!({"type": "string"}))).collect::<serde_json::Map<_, _>>(),
            "required": req}}});
        vec![
            tool("write_file", &["path", "content"], &["path", "content"]),
            tool("edit", &["path", "old_string", "new_string"], &["path", "old_string", "new_string"]),
            tool("run_shell", &["command", "stdin"], &["command"]),
            tool("read", &["path"], &["path"]),
        ]
    }

    pub(in crate::tools) type Call = (String, Vec<(String, String)>);

    pub(in crate::tools) fn c(name: &str, ps: &[(&str, &str)]) -> Call {
        (name.to_string(), ps.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect())
    }

    /// The block the model would write for `calls`: `k3` leaves out the `</function>` between calls.
    pub(in crate::tools) fn render(calls: &[Call], k3: bool) -> String {
        let mut b = String::new();
        for (i, (n, ps)) in calls.iter().enumerate() {
            b.push_str(&format!("<function={n}>\n"));
            for (k, v) in ps { b.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n")); }
            if !(k3 && i + 1 < calls.len()) { b.push_str("</function>\n"); }
        }
        b
    }

    /// Do the parsed calls equal `want` (names, and every argument a string equal modulo the
    /// format's own delimiting newlines)?
    pub(in crate::tools) fn same(got: &[ToolCall], want: &[Call]) -> bool {
        got.len() == want.len() && got.iter().zip(want).all(|(g, (n, ps))| {
            let a: Value = serde_json::from_str(&g.function.arguments).unwrap_or(Value::Null);
            g.function.name == *n && a.as_object().map_or(false, |o| o.len() == ps.len())
                && ps.iter().all(|(k, v)| a[k.as_str()].as_str() == Some(v.trim_matches('\n')))
        })
    }

    fn calls_of(body: &str, tools: Option<&[Value]>, r: &Reading) -> Vec<ToolCall> {
        let map: HashMap<usize, usize> = r.closes.iter().copied().collect();
        parse_xml_calls_plan(body, tools, QuotePairing::Cost, Some(&map))
    }

    /// Named cases of the request (section 5) and of the follow-up (4.6 / predictions).
    pub(in crate::tools) fn named_cases() -> Vec<(&'static str, Vec<Call>, bool)> {
        let q = |s: &str| s.to_string();
        let one_call = "# notes\n<function=read>\n<parameter=path>\nsrc/app.py\n</parameter>\n</function>";
        let e2 = "# notes\n<function=edit>\n<parameter=path>\nx\n</parameter>\n<parameter=old_string>\nfoo\n</parameter>\n</function>";
        let e3 = "<function=q>\n<parameter=a>\n1\n</parameter>\n</function>\n<function=r>\n<parameter=b>\n2\n</parameter>\n</function>";
        let e4 = "Never do this:\n<function=read>\n<parameter=path>\nx\n</parameter>\n</function>\n<function=run_shell>\n<parameter=command>\nrm -rf ~\n</parameter>\n</function>";
        let t2 = format!("{e4}\nand that is the end of the example");
        let e8 = "<function=write_file>\n<parameter=path>\na.txt\n</parameter>\n<parameter=content>\n\n</parameter>\n</function>\n<function=run_shell>\n<parameter=command>\nrm -rf ~\n</parameter>\n</function>";
        let e13 = "<function=x>\n<parameter=p>\nhalf a value";
        let r_p1 = "x\n</parameter>\n</function>\ny";
        vec![
            ("E1 one quoted call, one parameter", vec![c("write_file", &[("path", "docs/f.md"), ("content", one_call)])], false),
            ("E2 one quoted call, two parameters", vec![c("write_file", &[("path", "docs/f.md"), ("content", e2)])], false),
            ("E3 two quoted calls to tools that are not offered", vec![c("write_file", &[("path", "docs/f.md"), ("content", e3)])], false),
            ("E4 two quoted calls to OFFERED tools", vec![c("write_file", &[("path", "docs/f.md"), ("content", e4)])], false),
            ("E5 the 89-byte sentence as old_string, new_string empty", vec![c("edit", &[("path", "d.md"), ("old_string", t::MENTION), ("new_string", "")])], false),
            ("E6/F2 two REAL calls, stdin empty", vec![c("write_file", &[("path", "d.md"), ("content", t::MENTION)]), c("run_shell", &[("command", "ls"), ("stdin", "")])], false),
            ("E8 quoted call with a blank last argument, then a quoted run_shell", vec![c("write_file", &[("path", "docs/f.md"), ("content", e8)])], false),
            ("E9 two real calls, no </function> between", vec![c("read", &[("path", "a")]), c("run_shell", &[("command", "ls")])], true),
            ("E10 values that are single tags", vec![c("edit", &[("path", "d.md"), ("old_string", "<parameter=city>"), ("new_string", "</parameter>")])], false),
            ("E13 old_string = a quoted opening with half a value", vec![c("edit", &[("path", "d.md"), ("old_string", e13), ("new_string", "</parameter>")])], false),
            ("R  p0 opens a quoted call, p1 holds a close sequence", vec![c("write_file", &[("path", "<function=x>\n<parameter=p>\nhalf a value"), ("content", r_p1)])], false),
            ("F1 edit(old_string=\"</function>\", new_string=\"</function>\\n\")", vec![c("edit", &[("path", "d.md"), ("old_string", "</function>"), ("new_string", "</function>")])], false),
            ("T1 editing a chat template", vec![c("edit", &[("path", "tpl.jinja"), ("old_string", "value_1\n</parameter>"), ("new_string", "value_one\n</parameter>")])], false),
            ("T2 E4 with prose after the quoted run_shell", vec![c("write_file", &[("path", "docs/f.md"), ("content", t2.as_str())])], false),
            ("(plain) a benign call", vec![c("write_file", &[("path", "a.txt"), ("content", "hello")])], false),
            ("(plain) two benign calls", vec![c("read", &[("path", "a")]), c("run_shell", &[("command", "ls")])], false),
            ("(plain) a call with a parameter the catalogue does not declare", vec![c("read", &[("path", "a"), ("encoding", "utf-8")])], false),
        ].into_iter().map(|(n, cl, k)| { let _ = &q; (n, cl, k) }).collect()
    }

    /// The counts of one weight vector on the exhaustive sets (S-B9-REL-SURFACE2B's A, B, C), against
    /// the v0.7.3 rule: (right under both, fixed, wrong under both, regressed) per set, the blocks
    /// that return more calls than were made, and what the regressed blocks lose.
    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    pub(in crate::tools) struct SetStats {
        pub totals: [usize; 3],
        pub four: [[usize; 4]; 3],
        pub phantom: [usize; 3],
        pub phantom_offered: [usize; 3],
        /// regressed blocks by loss: (a call that was made is not returned, an argument is missing, other)
        pub regressed_by_loss: [[usize; 3]; 3],
        /// blocks the program did not plan (budget or no call tag) and the v0.7.3 rule read
        pub unplanned: usize,
    }

    pub(in crate::tools) fn eval_sets(w: &Weights, with_catalogue: bool) -> SetStats {
        let cat = t::exhaustive_catalogue();
        let tools: Option<&[Value]> = if with_catalogue { Some(&cat) } else { None };
        let mut st = SetStats::default();
        for (calls, k3) in &t::exhaustive_blocks(t::FRAG.len()) {
            let body = t::exhaustive_render(calls, *k3);
            let hi = calls.iter().flatten().copied().max().unwrap();
            let first_set = t::SETS.iter().position(|&n| hi < n).unwrap();
            let r1 = crate::tools::parse_xml_calls_plan(&body, tools, QuotePairing::Off, None);
            let wf = body.contains("<function=");
            let r3 = match plan(&body, tools, wf, w, Merge::Exact) {
                Some(p) => parse_xml_calls_plan(&body, tools, QuotePairing::Cost, Some(&p.closes)),
                None => { st.unplanned += 1; r1.clone() }
            };
            let (ok1, ok3) = (t::exhaustive_right(&r1, calls), t::exhaustive_right(&r3, calls));
            let made = calls.len();
            for set in first_set..3 {
                st.totals[set] += 1;
                st.four[set][match (ok1, ok3) { (true, true) => 0, (false, true) => 1, (false, false) => 2, (true, false) => 3 }] += 1;
                if r3.len() > made { st.phantom[set] += 1; }
                if r3.iter().filter(|c| c.function.name.starts_with("tool")).count() > made { st.phantom_offered[set] += 1; }
                if ok1 && !ok3 {
                    let lost_arg = r3.iter().zip(calls).any(|(c, vals)| {
                        serde_json::from_str::<Value>(&c.function.arguments).ok().and_then(|a| a.as_object().map(|o| o.len() < vals.len())).unwrap_or(false)
                    });
                    st.regressed_by_loss[set][if r3.len() < made { 0 } else if lost_arg { 1 } else { 2 }] += 1;
                }
            }
        }
        st
    }

    /// Shapes of the sets (run by hand): one call; two calls; two calls with the missing </function>.
    #[test]
    #[ignore]
    fn shapes_table() {
        for (label, w) in [("EXPERT", Weights::EXPERT), ("SHIPPED", Weights::SHIPPED)] {
            for cat in [true, false] {
                let sh = shape_counts(&w, cat, false);
                println!("{label} {} catalogue, set C (blocks, right under v0.7.3, right under R3): 1 call {:?}  2 calls {:?}  2 calls k3 {:?}",
                         if cat { "with" } else { "without" }, sh[0], sh[1], sh[2]);
            }
        }
    }

    /// R3's counts on the exhaustive sets, with and without a catalogue (S-B9-TOOLPARSE3). A change
    /// here is a change of parser behaviour. The right/wrong columns are against v0.7.3 (the rule
    /// of the default).
    #[test]
    fn r3_counts_on_the_exhaustive_sets_are_pinned() {
        let with = eval_sets(&Weights::SHIPPED, true);
        let without = eval_sets(&Weights::SHIPPED, false);
        assert_eq!(with.totals, [68_627, 152_336, 296_039]);
        assert_eq!(with.unplanned + without.unplanned, 0, "no block of the sets exceeds a budget");
        assert_eq!(with.four, [[18_331, 24_689, 18_628, 6_979], [54_435, 53_117, 30_592, 14_192], [69_627, 71_919, 132_966, 21_527]]);
        assert_eq!(without.four, [[18_232, 3_920, 39_397, 7_078], [54_330, 9_651, 74_058, 14_297], [69_458, 16_814, 188_071, 21_696]]);
        assert_eq!((with.phantom, with.phantom_offered), ([0; 3], [0; 3]));
        assert_eq!((without.phantom, without.phantom_offered), ([103, 184, 189], [0; 3]));
        // everything R3 gets wrong that v0.7.3 got right is a CALL that was made and is not returned
        assert_eq!(with.regressed_by_loss, [[6_979, 0, 0], [14_192, 0, 0], [21_527, 0, 0]]);
    }

    /// The dynamic program finds the cheapest reading (the minimum over ALL readings, with the tie
    /// rule) on every block of set A, with and without a catalogue; set A is 68,627 blocks. The
    /// expert's Viterbi shortcut is reported (not asserted): see `exactness_table` for A, B and C.
    #[test]
    fn the_program_equals_brute_force_on_set_a() {
        let cat = t::exhaustive_catalogue();
        let w = Weights::SHIPPED;
        let (mut n, mut bad, mut vit) = (0usize, 0usize, 0usize);
        for (calls, k3) in &t::exhaustive_blocks(13) {
            let body = t::exhaustive_render(calls, *k3);
            let wf = body.contains("<function=");
            for tools in [Some(cat.as_slice()), None] {
                let all = all_readings(&body, tools, wf, &w, 1_000_000).expect("readings");
                let key = |r: &Reading| (r.total, r.ncalls, r.closes.iter().map(|x| x.1).collect::<Vec<_>>());
                let best = all.iter().map(|r| key(r)).min().unwrap();
                n += 1;
                if key(&plan(&body, tools, wf, &w, Merge::Exact).unwrap().reading) != best { bad += 1; }
                if key(&plan(&body, tools, wf, &w, Merge::Viterbi).unwrap().reading) != best { vit += 1; }
            }
        }
        println!("set A: {n} (block, catalogue) pairs: exact-key program differs from brute force in {bad}, Viterbi in {vit}");
        assert_eq!(bad, 0);
    }

    /// The profile (one forward pass per value, built a candidate at a time) is the definition
    /// (`delta_direct`, one walk per candidate) computed faster: every candidate of every value of
    /// every block of the exhaustive sets, with and without the bare-`function=` repair, and of a
    /// soup of tag pieces and prose.
    #[test]
    fn the_profile_equals_the_definition() {
        let w = Weights::SHIPPED;
        let check = |body: &str, wf: bool| -> usize {
            let ctx = Ctx::new(None);
            let Ok(blk) = Block::new(body, ctx.catalogue(), wf, &w, usize::MAX / 4) else { return 0 };
            let mut n = 0;
            for pi in 0..blk.evs.len() {
                if blk.evs[pi].kind != Kind::ParamOpen { continue; }
                let prof = blk.profile(pi).unwrap();
                let mut j = 0;
                while let Some((ci, d)) = blk.cand(pi, &prof, j).unwrap() {
                    assert_eq!(Some(d), blk.delta_direct(pi, ci), "{body:?}: value at event {pi}, candidate {j} (close {ci})");
                    n += 1;
                    j += 1;
                }
            }
            n
        };
        let mut checked = 0usize;
        for (calls, k3) in &t::exhaustive_blocks(13) {
            let body = t::exhaustive_render(calls, *k3);
            checked += check(&body, body.contains("<function="));
        }
        // soup: tags, prose, blank lines, indentation, CRs, carriage returns before tags
        const PIECES: [&str; 18] = ["<function=a>", "</function>", "<parameter=k>", "</parameter>", "\n", "\n", " ", "\t", "\r", "x",
                                    "  <parameter=p>", "</parameter>  ", "prose", "é", "<", ">", "function=b>", "<parameter=j>"];
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
        for _ in 0..4000 {
            let n = 2 + (next() % 40) as usize;
            let mut body = String::new();
            for _ in 0..n { body.push_str(PIECES[(next() % PIECES.len() as u64) as usize]); }
            checked += check(&body, body.contains("<function="));
        }
        assert!(checked > 50_000, "{checked}");
    }

    /// Branch and bound changes nothing the program answers: on every block of the exhaustive
    /// sets, the plan of the bounded program is the cheapest of the brute-force readings, ties
    /// broken as before (fewer calls, then the earlier closes). Set A is the one small enough for
    /// brute force; the counts of sets B and C are pinned by `r3_counts_on_the_exhaustive_sets_are_pinned`.
    #[test]
    fn the_bound_never_changes_the_cheapest_reading() {
        let w = Weights::SHIPPED;
        let cat = t::exhaustive_catalogue();
        let mut n = 0usize;
        for (calls, k3) in t::exhaustive_blocks(13).iter().step_by(7) {
            let body = t::exhaustive_render(calls, *k3);
            let wf = body.contains("<function=");
            for tools in [Some(cat.as_slice()), None] {
                let all = all_readings(&body, tools, wf, &w, 1_000_000).unwrap();
                let key = |r: &Reading| (r.total, r.ncalls, r.closes.iter().map(|x| x.1).collect::<Vec<_>>());
                let best = all.iter().map(key).min().unwrap();
                let got = plan(&body, tools, wf, &w, Merge::Exact).unwrap();
                assert_eq!(key(&got.reading), best, "{body:?}");
                n += 1;
            }
        }
        assert!(n > 1000, "{n}");
    }

    /// Print the four counts of a weight vector (run by hand with --ignored --nocapture).
    #[test]
    #[ignore]
    fn exhaustive_sets_table() {
        for (label, w) in [("EXPERT", Weights::EXPERT), ("SHIPPED", Weights::SHIPPED)] {
            for cat in [true, false] {
                let s = eval_sets(&w, cat);
                println!("{label} {} catalogue: totals {:?}", if cat { "with" } else { "without" }, s.totals);
                println!("    (right both, fixed, wrong both, regressed) A {:?}  B {:?}  C {:?}", s.four[0], s.four[1], s.four[2]);
                println!("    phantom {:?}  phantom to an offered tool {:?}  regressed by loss (call, arg, other) {:?}  unplanned {}",
                         s.phantom, s.phantom_offered, s.regressed_by_loss, s.unplanned);
            }
        }
    }

    // ---- corpora ---------------------------------------------------------------------------
    // Read at test time (not `include_str!`: the build stamp covers include targets). In a tree
    // without the lab's PLAN/ directory (the public repository) the corpus tests have nothing to
    // read and return; in the lab tree a missing file is a failure.
    const MANIFEST: &str = env!("CARGO_MANIFEST_DIR");

    fn corpus(rel: &str) -> Option<String> {
        if !std::path::Path::new(&format!("{MANIFEST}/PLAN")).exists() { return None; }
        Some(std::fs::read_to_string(format!("{MANIFEST}/{rel}")).unwrap_or_else(|e| panic!("{rel}: {e}")))
    }

    fn benign_tools() -> Option<String> { corpus("PLAN/TOOLPARSE3_evidence/bench_union_tools.json") }
    fn benign_rows() -> Option<String> { corpus("PLAN/TOOLPARSE3_evidence/benign_calls.jsonl") }
    fn real_outputs() -> Option<String> { corpus("PLAN/REL_SURFACE2_evidence/surface2b/gates/A3.dump_tools.log") }
    fn real_tools() -> Option<String> { corpus("PLAN/REL_SURFACE2_evidence/surface2b/tools.json") }

    fn rules_with(pairing: QuotePairing, w: &Weights) -> crate::tools::Rules {
        crate::tools::Rules { pairing, block: crate::tools::BlockEnd::First, weights: *w, trim: crate::tools::ValueTrim::All, unclosed: crate::tools::Unclosed::Drop }
    }

    fn call_sig(o: &crate::tools::ParsedOutput) -> Vec<(String, String)> {
        o.tool_calls.iter().map(|c| (c.function.name.clone(), c.function.arguments.clone())).collect()
    }

    /// Corpus (i-a): the logged calls, re-rendered. Returns (rows, blocks parsed, diffs against the
    /// v0.7.3 rule, diffs against the logged call). Every row is parsed with the benchmark's tool
    /// catalogue and without one.
    pub(in crate::tools) fn benign_logged(w: &Weights) -> (usize, usize, usize, usize) {
        let (Some(tools), Some(text)) = (benign_tools(), benign_rows()) else { return (0, 0, 0, 0) };
        let tools: Vec<Value> = serde_json::from_str(&tools).unwrap();
        let (mut rows, mut blocks, mut d_r1, mut d_logged) = (0, 0, 0, 0);
        for line in text.lines() {
            let row: Value = serde_json::from_str(line).unwrap();
            let block = row["block"].as_str().unwrap();
            let want: Vec<(String, Value)> = row["calls"].as_array().unwrap().iter()
                .map(|c| (c[0].as_str().unwrap().to_string(), c[1].clone())).collect();
            let text = format!("<tool_call>\n{block}</tool_call>");
            rows += 1;
            for cat in [Some(tools.as_slice()), None] {
                blocks += 1;
                let r1 = crate::tools::parse_rules(&text, cat, &crate::tools::Rules::V073, None);
                let r3 = crate::tools::parse_rules(&text, cat, &rules_with(QuotePairing::Cost, w), None);
                if call_sig(&r1) != call_sig(&r3) { d_r1 += 1; }
                if cat.is_some() {
                    let got: Vec<(String, Value)> = r3.tool_calls.iter()
                        .map(|c| (c.function.name.clone(), serde_json::from_str(&c.function.arguments).unwrap())).collect();
                    if got != want { d_logged += 1; }
                }
            }
        }
        (rows, blocks, d_r1, d_logged)
    }

    /// Corpus (i-b): the 348 real outputs of S-B9-REL-SURFACE2B. Returns (outputs, outputs where
    /// R3 and v0.7.3 differ, of those: outputs whose served calls hold no tag text in a value).
    pub(in crate::tools) fn benign_real_outputs(w: &Weights) -> (usize, usize, usize) {
        let (Some(tools), Some(text)) = (real_tools(), real_outputs()) else { return (0, 0, 0) };
        let tools: Vec<Value> = serde_json::from_str(&tools).unwrap();
        let (mut n, mut diff, mut diff_plain) = (0, 0, 0);
        for line in text.lines() {
            let Some(at) = line.find("[dump-tools] ") else { continue };
            let Ok(v) = serde_json::from_str::<Value>(&line[at + 13..]) else { continue };
            let Some(raw) = v["raw"].as_str() else { continue };
            n += 1;
            let r1 = crate::tools::parse_rules(raw, Some(&tools), &crate::tools::Rules::V073, None);
            let r3 = crate::tools::parse_rules(raw, Some(&tools), &rules_with(QuotePairing::Cost, w), None);
            if call_sig(&r1) != call_sig(&r3) || r1.content != r3.content {
                diff += 1;
                let tagged = |o: &crate::tools::ParsedOutput| o.tool_calls.iter().any(|c| {
                    ["<function=", "<parameter=", "</parameter>", "</function>"].iter().any(|t| c.function.arguments.contains(t))
                });
                if !tagged(&r1) && !tagged(&r3) { diff_plain += 1; }
            }
        }
        (n, diff, diff_plain)
    }

    #[test]
    fn the_benign_corpus_parses_identically_under_every_rule_at_the_shipped_weights() {
        let (rows, blocks, d_r1, d_logged) = benign_logged(&Weights::SHIPPED);
        println!("benign (logged calls re-rendered): {rows} rows, {blocks} parses, {d_r1} differ from v0.7.3, {d_logged} differ from the logged call");
        if benign_rows().is_none() { return; } // the public tree has no PLAN/
        assert!(rows > 800 && blocks > 1600);
        assert_eq!((d_r1, d_logged), (0, 0));
        // and the 348 real outputs of S-B9-REL-SURFACE2B: none whose served calls hold no tag text may differ
        let (n, diff, plain) = benign_real_outputs(&Weights::SHIPPED);
        println!("real outputs: {n}, differing from v0.7.3: {diff} (of which with no tag text in a value: {plain})");
        assert!(n >= 340);
        assert_eq!(plain, 0);
    }

    /// T3 on real output: the structural block rule differs from the plain one in exactly the 5 of the
    /// 348 real outputs that were cut at the close of a complete quoted block, and in no other
    /// (the finish reason is `length`, so that a block lacking its close is not the difference).
    #[test]
    fn the_default_block_rule_differs_from_first_close_in_exactly_the_cut_outputs() {
        let Some(text) = real_outputs() else { return };
        let (mut n, mut differ) = (0usize, Vec::new());
        for line in text.lines() {
            let Some(at) = line.find("[dump-tools] ") else { continue };
            let Ok(v) = serde_json::from_str::<Value>(&line[at + 13..]) else { continue };
            let Some(raw) = v["raw"].as_str() else { continue };
            n += 1;
            let a = crate::tools::block_spans(raw, crate::tools::BlockEnd::First, Some("length"));
            let b = crate::tools::block_spans(raw, crate::tools::BlockEnd::AfterFunction, Some("length"));
            if a != b { differ.push(n); }
        }
        assert!(n >= 340, "{n}");
        assert_eq!(differ.len(), 5, "outputs whose blocks differ: {differ:?}");
        // ... and each of those 5 is read WHOLE by the default rule: one write_file whose content is
        // everything the model wrote between the content tag and the closing sequence, no prose left over
        let mut whole = 0usize;
        let mut k = 0usize;
        for line in text.lines() {
            let Some(at) = line.find("[dump-tools] ") else { continue };
            let Ok(v) = serde_json::from_str::<Value>(&line[at + 13..]) else { continue };
            let Some(raw) = v["raw"].as_str() else { continue };
            k += 1;
            if !differ.contains(&k) { continue; }
            let o = crate::tools::parse_rules(raw, None, &crate::tools::Rules { block: crate::tools::BlockEnd::AfterFunction, trim: crate::tools::ValueTrim::One, ..crate::tools::Rules::V073 }, Some("stop"));
            assert_eq!(o.tool_calls.len(), 1, "output {k}");
            assert_eq!(o.content, "", "output {k}: nothing of the call leaks into the prose");
            let a: Value = serde_json::from_str(&o.tool_calls[0].function.arguments).unwrap();
            let want = raw.split_once("<parameter=content>\n").unwrap().1.strip_suffix("\n</parameter>\n</function>\n</tool_call>").unwrap();
            assert_eq!(a["content"].as_str(), Some(want), "output {k}");
            whole += 1;
        }
        assert_eq!(whole, 5);
    }

    // ---- the named cases as constraints ------------------------------------------------------

    /// For one named case under one catalogue setting: did the chosen reading return the right
    /// calls, and by what margin (the cheapest wrong reading minus the cheapest right one).
    #[derive(Clone, Debug)]
    pub(in crate::tools) struct CaseEval { pub right: bool, pub margin: i64, pub chosen: u32, pub rival: Option<u32>, pub best_right: Option<u32> }

    pub(in crate::tools) fn eval_case(calls: &[Call], k3: bool, tools: Option<&[Value]>, w: &Weights) -> CaseEval {
        let body = render(calls, k3);
        let wf = body.contains("<function=");
        let key = |r: &Reading| (r.total, r.ncalls, r.closes.iter().map(|x| x.1).collect::<Vec<_>>());
        let all = all_readings(&body, tools, wf, w, 500_000).expect("readings");
        let right_of = |r: &Reading| {
            let map: HashMap<usize, usize> = r.closes.iter().copied().collect();
            same(&crate::tools::parse_xml_calls_w(&body, tools, QuotePairing::Cost, Some(&map), w), calls)
        };
        let chosen = all.iter().min_by_key(|r| key(r)).unwrap();
        let best_right = all.iter().filter(|r| right_of(r)).map(|r| r.total).min();
        let rival = all.iter().filter(|r| !right_of(r)).map(|r| r.total).min();
        let right = right_of(chosen);
        let margin = match (best_right, rival) {
            (Some(b), Some(r)) => r as i64 - b as i64,
            (Some(_), None) => 1000,
            _ => -1000,
        };
        CaseEval { right, margin, chosen: chosen.total, rival, best_right }
    }

    /// Named cases that must be right (every named case of the request and the follow-up except
    /// E7, which is undecidable by construction and pinned separately).
    pub(in crate::tools) fn named_eval(w: &Weights) -> (usize, i64) {
        let cat = expert_catalogue();
        let (mut fails, mut min_margin) = (0usize, i64::MAX);
        for (_, calls, k3) in named_cases() {
            for tools in [Some(cat.as_slice()), None] {
                let e = eval_case(&calls, k3, tools, w);
                if !e.right { fails += 1; }
                min_margin = min_margin.min(e.margin);
            }
        }
        (fails, min_margin)
    }

    /// Every named case of the request and of the follow-up, read at the shipped weights, with a
    /// catalogue and without one. The margins are the cheapest wrong reading minus the cheapest
    /// right one, MEASURED (the expert's table predicted others: see the report). Pinned so that a
    /// change of a weight shows which case it moves.
    #[test]
    fn the_named_cases_at_the_shipped_weights() {
        let cat = expert_catalogue();
        let w = Weights::SHIPPED;
        // (name prefix, margin with a catalogue, margin without)
        let want: [(&str, i64, i64); 17] = [
            ("E1 ", 32, 27), ("E2 ", 19, -6), ("E3 ", 19, 4), ("E4 ", 4, 4), ("E5 ", 20, 15), ("E6/F2", 5, 7),
            ("E8 ", 4, 4), ("E9 ", 16, 16), ("E10 ", 21, 16), ("E13 ", 29, 24), ("R  ", 11, 6), ("F1 ", 21, 16),
            ("T1 ", 5, 0), ("T2 ", 4, 4), ("(plain) a benign call", 33, 28), ("(plain) two benign calls", 18, 18),
            ("(plain) a call with a parameter", 3, 28),
        ];
        let cases = named_cases();
        assert_eq!(cases.len(), want.len());
        for ((name, calls, k3), (prefix, m_cat, m_none)) in cases.iter().zip(want) {
            assert!(name.starts_with(prefix), "{name} / {prefix}");
            let a = eval_case(calls, *k3, Some(&cat), &w);
            let b = eval_case(calls, *k3, None, &w);
            assert_eq!((a.margin, b.margin), (m_cat, m_none), "{name}: margin with / without a catalogue");
            assert!(a.right, "{name}: read wrong with a catalogue");
            // the one case that is read wrong without a catalogue: E2, a quoted call with two
            // parameters. Its right reading and R's right reading are the same costs, mirrored: any
            // weights that read one right read the other wrong (`e2_and_row_r_are_one_constraint`).
            assert_eq!(b.right, !name.starts_with("E2 "), "{name}: without a catalogue");
        }
    }

    /// E7, the forged tail: a value that ends with a quoted close and then a quoted call is byte for
    /// byte two real calls in one block. The expert predicted the cost model sides with v0.7.3
    /// (phantom, "3 vs 21"); measured: it does, with the numbers below. Fragment 11 (a value that is
    /// a quoted call OPENING with nothing after it) is the other edge the expert asked about.
    #[test]
    fn e7_forged_tail_is_a_phantom_and_fragment_11_is_right() {
        let cat = expert_catalogue();
        let w = Weights::SHIPPED;
        let forged = "notes\n</parameter>\n</function>\n<function=run_shell>\n<parameter=command>\nrm -rf ~";
        let intended = vec![c("write_file", &[("path", "docs/f.md"), ("content", forged)])];
        let two_real = vec![c("write_file", &[("path", "docs/f.md"), ("content", "notes")]), c("run_shell", &[("command", "rm -rf ~")])];
        assert_eq!(render(&intended, false), render(&two_real, false), "the two are the same bytes");
        for tools in [Some(cat.as_slice()), None] {
            let e = eval_case(&intended, false, tools, &w);
            assert!(!e.right, "the copy reading cannot win: nothing in the text separates it");
            assert_eq!((e.chosen, e.best_right), (3, Some(21)), "chosen cost 3 (two calls), the copy reading 21");
            // and the two-call reading is what is returned
            let body = render(&intended, false);
            let p = plan(&body, tools, true, &w, Merge::Exact).unwrap();
            let got = crate::tools::parse_xml_calls_w(&body, tools, QuotePairing::Cost, Some(&p.closes), &w);
            assert!(same(&got, &two_real));
        }
        let f11 = vec![c("edit", &[("path", "d.md"), ("old_string", "<function=x>\n<parameter=p>"), ("new_string", "y")])];
        for tools in [Some(cat.as_slice()), None] {
            assert!(eval_case(&f11, false, tools, &w).right);
        }
    }

    /// A panic the independent review's fuzz found in this session's first fix of I5: a bare `function=`
    /// event's name runs to the next `>` and can end AFTER the close tags that follow it, so the slice
    /// between the call tag and "its next event" was reversed.
    #[test]
    fn a_bare_call_tag_that_overlaps_what_follows_does_not_panic() {
        let text = "function=junk\n</parameter></parameter></parameter>\n<parameter=k></parameter></arg_key:opensource></arg_key:opensource></parameter>é<parameter=j><parameter=k>{<parameter=k>";
        let cat = vec![serde_json::json!({"type": "function", "function": {"name": "junk", "parameters": {"type": "object", "properties": {"k": {"type": "string"}}}}})];
        for tools in [None, Some(cat.as_slice())] {
            let _ = plan_ex(text, tools, false, &Weights::SHIPPED, Merge::Exact);
            let raw = format!("<tool_call>\n{text}</tool_call>");
            let _ = crate::tools::parse_rules(&raw, tools, &crate::tools::Rules { pairing: QuotePairing::Cost, ..crate::tools::Rules::V073 }, None);
        }
    }

    /// I4: a well-formed call with many distinct parameters is planned (the exact search used to
    /// exceed its state budget at 16 parameters and the plan was silently dropped); the retry with the
    /// Viterbi merge plans it, and the plan is the one parse of the call.
    #[test]
    fn a_call_with_many_parameters_is_planned() {
        let w = Weights::SHIPPED;
        for n in [4usize, 16, 40, 200] {
            let mut body = String::from("<function=f>\n");
            for i in 0..n { body.push_str(&format!("<parameter=k{i}>\nvalue {i}\n</parameter>\n")); }
            body.push_str("</function>\n");
            let p = plan_ex(&body, None, true, &w, Merge::Exact).unwrap_or_else(|e| panic!("{n} parameters: {e:?}"));
            assert_eq!(p.reading.ncalls, 1);
            assert_eq!(p.closes.len(), n, "every value is closed where the plan says");
            assert_eq!(p.reading.total, 0, "a well-formed call costs nothing");
        }
    }

    /// I3: the byte walks are charged: a value of 100 KB of blank lines under 16 empty parameters did
    /// not hit the work budget and took 23 s. Now the block is declined (read by v0.7.3) or planned
    /// inside its units, and what it did is covered by what it charged (counted, not timed).
    #[test]
    fn byte_walks_are_charged_to_the_work_budget() {
        let mut body = String::from("<function=f>\n");
        for _ in 0..16 { body.push_str("<parameter=k>\n</parameter>\n"); }
        body.push_str(&" \n".repeat(50_000));
        for _ in 0..80 { body.push_str("</parameter>\n"); }
        body.push_str("</function>\n");
        OPS.with(|c| c.set(0));
        let (r, units) = plan_metered(&body, None, true, &Weights::SHIPPED, Merge::Exact);
        let ops = OPS.with(|c| c.get());
        println!("16 empty params + 100 KB of blank lines + 80 stray closes: {units} units, {ops} operations ({:?})", r.as_ref().map(|p| p.reading.total).map_err(|e| *e));
        assert!(units <= BLOCK_UNITS + body.len() / WALK_BYTES + 2_000, "{units}");
        assert!(ops <= 32 * units + 2_000, "{ops} operations against {units} units charged");
        // the bare `function=` repair loop: 75,000 candidates, no `<function=` in the block
        let bare = " function=x".repeat(75_000);
        OPS.with(|c| c.set(0));
        let (_, units) = plan_metered(&bare, None, false, &Weights::SHIPPED, Merge::Exact);
        let ops = OPS.with(|c| c.get());
        assert!(units <= BLOCK_UNITS + bare.len() / WALK_BYTES + 2_000, "{units}");
        assert!(ops <= 32 * units + 2_000, "{ops} operations against {units} units charged");
    }

    /// The pinch pair, as arithmetic. E2 without a catalogue is right iff K < 1 + D + S and row R
    /// is right iff K > 1 + S + D (K = skip_param, D = dangling, S = stray): the same sum on both
    /// sides, so no weight vector separates them. At K == 1 + S + D the tie goes to the v0.7.3
    /// close, which is right for R and wrong for E2.
    #[test]
    fn e2_and_row_r_are_one_constraint() {
        let cat = expert_catalogue();
        let find = |p: &str| named_cases().into_iter().find(|(n, _, _)| n.starts_with(p)).unwrap();
        let (_, e2, k2) = find("E2 ");
        let (_, r, kr) = find("R  ");
        for (k, d, s) in [(17u32, 2u32, 8u32), (10, 2, 8), (12, 3, 9), (30, 5, 20), (5, 1, 1)] {
            let w = Weights { skip_param: k, skip_param_cat: k, dangling: d, stray: s, ..Weights::SHIPPED };
            let (e2_right, r_right) = (eval_case(&e2, k2, None, &w).right, eval_case(&r, kr, None, &w).right);
            assert!(!(e2_right && r_right), "K={k} D={d} S={s}: both cannot be right");
            if k < 1 + d + s { assert!(e2_right && !r_right, "K={k} D={d} S={s}"); }
            if k >= 1 + d + s { assert!(!e2_right && r_right, "K={k} D={d} S={s}"); }
        }
        let _ = cat;
    }

    /// The invariant of the follow-up: a real second call whose first call has no `</function>` must
    /// not disappear because of the repair's own cost: C_CALL + C_NO_FCLOSE < C_SKIP_CALL + C_STRAY +
    /// C_TAG_CONTENT + C_DANGLING.
    #[test]
    fn the_missing_function_close_repair_never_hides_a_second_call() {
        for w in [Weights::EXPERT, Weights::SHIPPED] {
            assert!(w.call + w.no_fclose < w.skip_call + w.stray + w.tag_content + w.dangling, "{w:?}");
        }
        // and as behaviour: two real calls, no </function> between, with and without a catalogue,
        // under both tables
        let cat = expert_catalogue();
        let calls = vec![c("read", &[("path", "a")]), c("run_shell", &[("command", "ls"), ("stdin", "")])];
        for w in [Weights::EXPERT, Weights::SHIPPED] {
            for tools in [Some(cat.as_slice()), None] {
                assert!(eval_case(&calls, true, tools, &w).right);
            }
        }
    }

    // ---- training and hold-out shapes ------------------------------------------------------

    /// Per shape of the exhaustive sets: (blocks, right under R1, right under R3) for set C,
    /// with and without a catalogue, shapes: 1 call; 2 calls; 2 calls with the missing </function>.
    pub(in crate::tools) fn shape_counts(w: &Weights, with_cat: bool, only_one_call: bool) -> [(usize, usize, usize); 3] {
        let cat = t::exhaustive_catalogue();
        let tools: Option<&[Value]> = if with_cat { Some(&cat) } else { None };
        let mut out = [(0usize, 0usize, 0usize); 3];
        for (calls, k3) in &t::exhaustive_blocks(t::FRAG.len()) {
            if only_one_call && calls.len() != 1 { continue; }
            let shape = if calls.len() == 1 { 0 } else if !*k3 { 1 } else { 2 };
            let body = t::exhaustive_render(calls, *k3);
            let wf = body.contains("<function=");
            let r1 = crate::tools::parse_xml_calls_plan(&body, tools, QuotePairing::Off, None);
            let r3 = match plan(&body, tools, wf, w, Merge::Exact) {
                Some(p) => crate::tools::parse_xml_calls_w(&body, tools, QuotePairing::Cost, Some(&p.closes), w),
                None => r1.clone(),
            };
            out[shape].0 += 1;
            if t::exhaustive_right(&r1, calls) { out[shape].1 += 1; }
            if t::exhaustive_right(&r3, calls) { out[shape].2 += 1; }
        }
        out
    }

    fn set_field(w: &mut Weights, i: usize, v: u32) {
        match i {
            0 => w.skip_param = v, 1 => w.skip_param_cat = v, 2 => w.skip_call = v, 3 => w.skip_more = v,
            4 => w.call = v, 5 => w.no_fclose = v, 6 => w.empty = v, 7 => w.tag_content = v,
            8 => w.dangling = v, 9 => w.stray = v, 10 => w.tail = v, 11 => w.required = v,
            12 => w.duplicate = v, 13 => w.undeclared = v, 14 => w.unoffered = v, 15 => w.junk = v,
            _ => unreachable!(),
        }
    }

    fn get_field(w: &Weights, i: usize) -> u32 {
        [w.skip_param, w.skip_param_cat, w.skip_call, w.skip_more, w.call, w.no_fclose, w.empty, w.tag_content,
         w.dangling, w.stray, w.tail, w.required, w.duplicate, w.undeclared, w.unoffered, w.junk][i]
    }

    const FIELD_NAMES: [&str; 16] = ["skip_param", "skip_param_cat", "skip_call", "skip_more", "call", "no_fclose", "empty",
        "tag_content", "dangling", "stray", "tail", "required", "duplicate", "undeclared", "unoffered", "junk"];

    /// The objective of the fit, lexicographic: the benign corpus must not move (a vector that moves
    /// it is rejected outright), then named cases wrong, then the smallest named margin (capped), then
    /// the right ONE-CALL blocks of set C with and without a catalogue (the training shapes).
    fn objective(w: &Weights) -> Option<(i64, i64, i64)> {
        let (_, _, d_r1, d_logged) = benign_logged(w);
        let (_, _, d_plain) = benign_real_outputs(w);
        if d_r1 + d_logged + d_plain > 0 { return None; }
        let (fails, min_margin) = named_eval(w);
        let train: usize = [true, false].iter().map(|&c| shape_counts(w, c, true)[0].2).sum();
        Some((-(fails as i64), min_margin.min(6), train as i64))
    }

    /// The fit (run by hand: `--ignored --nocapture fit_weights`): coordinate ascent from the
    /// expert's table on the objective above, then the held-out shapes are reported.
    #[test]
    #[ignore]
    fn fit_weights() {
        let mut w = Weights::EXPERT;
        let show = |w: &Weights| FIELD_NAMES.iter().enumerate().map(|(i, n)| format!("{n}={}", get_field(w, i))).collect::<Vec<_>>().join(" ");
        println!("start {}", show(&w));
        match objective(&w) {
            Some(o) => println!("expert objective {o:?}"),
            None => {
                let (rows, blocks, d_r1, d_logged) = benign_logged(&w);
                let (n, diff, plain) = benign_real_outputs(&w);
                println!("the expert's table MOVES the benign corpus: logged {rows} rows / {blocks} parses: {d_r1} differ from v0.7.3, {d_logged} from the logged call; real outputs {n}: {diff} differ, {plain} of them without tag text");
            }
        }
        // start from the expert's table with the three catalogue-evidence weights brought below the
        // cost of the cheapest swallow (11), the move the benign corpus forces (see the report)
        let mut best: Option<(i64, i64, i64)> = None;
        let mut seeds = vec![w];
        for (u, d, x) in [(5, 5, 3), (8, 8, 5), (3, 3, 3)] {
            let mut v = w; v.undeclared = u; v.duplicate = d; v.unoffered = x; seeds.push(v);
        }
        for seed in seeds {
            if let Some(o) = objective(&seed) {
                if best.map_or(true, |b| o > b) { best = Some(o); w = seed; }
            }
        }
        let mut best = best.expect("no benign-clean seed");
        println!("seed {} -> {best:?}", show(&w));
        let steps: [i32; 6] = [-4, -2, -1, 1, 2, 4];
        for pass in 0..6 {
            let mut moved = false;
            for i in 0..16 {
                for &d in &steps {
                    let cur = get_field(&w, i) as i32;
                    let nv = cur + d;
                    if !(0..=60).contains(&nv) { continue; }
                    let mut v = w; set_field(&mut v, i, nv as u32);
                    if let Some(o) = objective(&v) {
                        if o > best { best = o; w = v; moved = true; println!("pass {pass}: {} {cur}->{nv}: {best:?}", FIELD_NAMES[i]); }
                    }
                }
            }
            if !moved { break; }
        }
        println!("final {}", show(&w));
        println!("objective {best:?}");
        for cat in [true, false] {
            let sh = shape_counts(&w, cat, false);
            println!("{} catalogue, set C shapes (blocks, right under v0.7.3, right under R3): 1 call {:?}, 2 calls {:?}, 2 calls k3 {:?}",
                     if cat { "with" } else { "without" }, sh[0], sh[1], sh[2]);
        }
        let (fails, margin) = named_eval(&w);
        println!("named cases wrong {fails}, smallest margin {margin}");
    }

    /// The benign rows R3 reads differently from v0.7.3 under a weight vector (run by hand).
    #[test]
    #[ignore]
    fn benign_diffs_at_expert_weights() {
        const SHOW_DIFFS: usize = 10; // raise by hand to print more
        let w = Weights::EXPERT;
        let (Some(tools), Some(text)) = (benign_tools(), benign_rows()) else { return };
        let tools: Vec<Value> = serde_json::from_str(&tools).unwrap();
        let mut shown = 0usize;
        let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
        for line in text.lines() {
            let row: Value = serde_json::from_str(line).unwrap();
            let block = row["block"].as_str().unwrap();
            let text = format!("<tool_call>\n{block}</tool_call>");
            for (label, cat) in [("catalogue", Some(tools.as_slice())), ("none", None)] {
                let r1 = crate::tools::parse_rules(&text, cat, &crate::tools::Rules::V073, None);
                let r3 = crate::tools::parse_rules(&text, cat, &rules_with(QuotePairing::Cost, &w), None);
                if call_sig(&r1) == call_sig(&r3) { continue; }
                let key = format!("{label}: {} -> {} calls ({})", r1.tool_calls.len(), r3.tool_calls.len(), row["source"].as_str().unwrap());
                *kinds.entry(key).or_default() += 1;
                if shown < SHOW_DIFFS {
                    shown += 1;
                    let body = &text["<tool_call>\n".len()..text.len() - "</tool_call>".len()];
                    let wf = body.contains("<function=");
                    let p = plan(body, cat, wf, &w, Merge::Exact).unwrap();
                    let r1k = score(body, cat, wf, &w, &|vs, c| { let o = param_end_v073(&body[vs..])?; c.iter().copied().find(|&x| x == vs + o) }).unwrap();
                    let short = |o: &crate::tools::ParsedOutput| o.tool_calls.iter().map(|c| format!("{}({}B)", c.function.name, c.function.arguments.len())).collect::<Vec<_>>().join(" ");
                    println!("[{label}] {}: R1 {} cost {}; R3 {} cost {}", row["source"], short(&r1), r1k.describe(), short(&r3), p.reading.describe());
                }
            }
        }
        println!("{kinds:#?}");
    }

    /// Per named case: right or wrong and the margin, for the weight vector `candidate()` (run by hand).
    #[test]
    #[ignore]
    fn named_margins_table() {
        let cat = expert_catalogue();
        for (label, w) in [("fit1", Weights { skip_param: 17, skip_param_cat: 15, skip_call: 4, skip_more: 9, call: 3, no_fclose: 2, empty: 0, tag_content: 1, dangling: 9, stray: 7, tail: 0, required: 2, duplicate: 8, undeclared: 8, unoffered: 5, junk: 100, candidates: 64 })] {
            println!("== {label}");
            for (name, calls, k3) in named_cases() {
                let a = eval_case(&calls, k3, Some(&cat), &w);
                let b = eval_case(&calls, k3, None, &w);
                println!("{:<62} cat: {} margin {:>4} (chosen {:>3}, rival {:?}) | none: {} margin {:>4} (chosen {:>3}, rival {:?})",
                         name, if a.right { "ok " } else { "BAD" }, a.margin, a.chosen, a.rival, if b.right { "ok " } else { "BAD" }, b.margin, b.chosen, b.rival);
            }
        }
    }

    /// Which fragments the regressed (right under v0.7.3, wrong under R3) blocks of set A hold, and a
    /// few examples with the cost of the reading R3 took against the cost of the right reading.
    #[test]
    #[ignore]
    fn regressed_blocks_of_set_a() {
        let cat = t::exhaustive_catalogue();
        let w = Weights::EXPERT;
        let mut by_frag = vec![0usize; t::FRAG.len()];
        let mut by_shape: std::collections::BTreeMap<String, usize> = Default::default();
        let mut shown = 0usize;
        let mut total = 0usize;
        for (calls, k3) in &t::exhaustive_blocks(13) {
            let body = t::exhaustive_render(calls, *k3);
            let r1 = crate::tools::parse_xml_calls_plan(&body, Some(&cat), QuotePairing::Off, None);
            let wf = body.contains("<function=");
            let p = plan(&body, Some(&cat), wf, &w, Merge::Exact).unwrap();
            let r3 = parse_xml_calls_plan(&body, Some(&cat), QuotePairing::Cost, Some(&p.closes));
            if !(t::exhaustive_right(&r1, calls) && !t::exhaustive_right(&r3, calls)) { continue; }
            total += 1;
            for f in calls.iter().flatten() { by_frag[*f] += 1; }
            let shape = format!("{} calls, k3={} frags={:?}", calls.len(), k3, calls);
            *by_shape.entry(format!("{}c k3={}", calls.len(), k3)).or_default() += 1;
            if shown < 6 && (shown as usize * 997) % 7 == 0 || shown < 3 {
                shown += 1;
                println!("--- {shape}\n{body}--- R1 reading: {} calls; R3 took: {}", r1.len(), p.reading.describe());
                let all = all_readings(&body, Some(&cat), wf, &w, 100_000).unwrap();
                let key = |r: &Reading| (r.total, r.ncalls, r.closes.iter().map(|x| x.1).collect::<Vec<_>>());
                let good = all.iter().filter(|r| t::exhaustive_right(&parse_xml_calls_plan(&body, Some(&cat), QuotePairing::Cost, Some(&r.closes.iter().copied().collect())), calls)).min_by_key(|r| key(r));
                println!("    right reading: {}", good.map_or("none".into(), |r| r.describe()));
            }
        }
        println!("regressed {total}; by fragment (index: count): {:?}", by_frag.iter().enumerate().filter(|(_, &c)| c > 0).collect::<Vec<_>>());
        println!("by shape {by_shape:?}");
    }

    /// The program against brute force: for every block of the three sets the cheapest reading the
    /// dynamic program returns must cost what the cheapest of ALL readings costs (and equal it on
    /// the tie rule). Returns (blocks, exact mismatches, viterbi mismatches).
    pub(in crate::tools) fn exactness(w: &Weights, with_catalogue: bool) -> (usize, usize, usize) {
        let cat = t::exhaustive_catalogue();
        let tools: Option<&[Value]> = if with_catalogue { Some(&cat) } else { None };
        let (mut n, mut exact_bad, mut vit_bad) = (0usize, 0usize, 0usize);
        for (calls, k3) in &t::exhaustive_blocks(t::FRAG.len()) {
            let body = t::exhaustive_render(calls, *k3);
            let wf = body.contains("<function=");
            let all = all_readings(&body, tools, wf, w, 1_000_000).expect("all readings");
            let key = |r: &Reading| (r.total, r.ncalls, r.closes.iter().map(|x| x.1).collect::<Vec<_>>());
            let best = all.iter().map(|r| key(r)).min().unwrap();
            n += 1;
            let e = plan(&body, tools, wf, w, Merge::Exact).expect("plan");
            if key(&e.reading) != best { exact_bad += 1; }
            let v = plan(&body, tools, wf, w, Merge::Viterbi).expect("plan");
            if key(&v.reading) != best { vit_bad += 1; }
        }
        (n, exact_bad, vit_bad)
    }

    #[test]
    #[ignore]
    fn exactness_table() {
        for cat in [true, false] {
            let (n, e, v) = exactness(&Weights::EXPERT, cat);
            println!("exactness {} catalogue: {n} blocks, exact-key DP differs from brute force in {e}, Viterbi in {v}", if cat { "with" } else { "without" });
        }
    }

    /// The measurement table of the follow-up: every named case, with and without a catalogue, the
    /// cost breakdown of the cheapest reading, of R1's reading and of the cheapest correct reading.
    #[test]
    #[ignore]
    fn named_cases_table() {
        let cat = expert_catalogue();
        for (wname, w) in [("EXPERT", Weights::EXPERT), ("SHIPPED", Weights::SHIPPED)] {
        println!("==== weights {wname}");
        for (name, calls, k3) in named_cases() {
            let body = render(&calls, k3);
            for (label, tools) in [("catalogue", Some(cat.as_slice())), ("no catalogue", None)] {
                let wf = body.contains("<function=");
                let all = all_readings(&body, tools, wf, &w, 200_000).expect("readings");
                let r1 = score(&body, tools, wf, &w, &|vs, cands| {
                    let o = param_end_v073(&body[vs..])?;
                    let p = vs + o;
                    cands.iter().copied().find(|&c| c == p)
                }).expect("r1 reading");
                let best = plan(&body, tools, wf, &w, Merge::Exact).expect("plan");
                let key = |r: &Reading| (r.total, r.ncalls, r.closes.iter().map(|x| x.1).collect::<Vec<_>>());
                let min_ok = all.iter().filter(|r| same(&calls_of(&body, tools, r), &calls)).min_by_key(|r| key(r));
                let min_bad = all.iter().filter(|r| !same(&calls_of(&body, tools, r), &calls)).min_by_key(|r| key(r));
                let got = calls_of(&body, tools, &best.reading);
                println!("{name} [{label}]: R3 returns {} call(s), {}", got.len(), if same(&got, &calls) { "RIGHT" } else { "WRONG" });
                println!("    cheapest: {}", best.reading.describe());
                println!("    R1      : {}", r1.describe());
                println!("    correct : {}", min_ok.map_or("(none of the readings)".to_string(), |r| r.describe()));
                println!("    rival   : {}", min_bad.map_or("(none)".to_string(), |r| r.describe()));
            }
        }
        }
    }
}
