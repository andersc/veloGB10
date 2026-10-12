//! FROZEN COPY of the v0.7.3 XML tool-call parser (`src/tools.rs` as it was at master 8674fa30, the
//! base of S-B9-TOOLPARSE3), compiled in the TEST build only.
//!
//! It is the reference the in-tree differential test (`default_identity.rs`) compares the shipped
//! parser's default flags against: the claim "a default server returns what v0.7.3 returned" is a
//! test that runs in the lib suite, not a file outside the build. Do not edit it. The only line that
//! differs from the original is `quote_pairing()`, which returns `Off` instead of reading the option
//! (a frozen reference must not depend on a flag). Its own unit tests are not carried over (they are
//! the shipped module's tests, which still pass).
#![allow(dead_code, unused_imports, unused_variables)]

//! Parse the model's tool-call syntax back into OpenAI `tool_calls`.
//!
//! Qwen3.5's own chat template instructs the model to emit calls in an XML-ish form, NOT the JSON
//! blob most people expect:
//!
//! ```text
//! <tool_call>
//! <function=get_weather>
//! <parameter=city>
//! Paris
//! </parameter>
//! <parameter=units>
//! c
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Every value arrives as TEXT. OpenAI's `arguments` is a JSON string whose values must have the types
//! the tool's JSON Schema declares -- a harness will feed them straight into a real function, so
//! sending `"count": "3"` where the schema says `integer` is a bug that surfaces in the caller, not
//! here. So we coerce each parameter against the declared schema, and fall back to string when the
//! schema is silent or the value doesn't parse.

use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;

use crate::tokenizer::{FunctionCall, ToolCall};

/// Tool-call ids must be unique across the WHOLE conversation, not just within one response.
///
/// They used to be `call_{index-within-this-response}`, so every turn of an agent loop emitted
/// `call_0` again. Harnesses match a tool RESULT back to its call by id — duplicate ids across turns
/// are exactly how a tool appears to run and then quietly has no effect, because the result gets
/// attached to the wrong (earlier) call. A process-wide counter costs nothing and removes the class.
static CALL_SEQ: AtomicU64 = AtomicU64::new(0);

const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";

// hy_v3's tool-call markup (its chat template instructs this form; the `:opensource` suffix is the
// family signature). A call is:
//   <tool_calls:opensource>
//   <tool_call:opensource>get_weather<tool_sep:opensource>
//   <arg_key:opensource>city</arg_key:opensource>
//   <arg_value:opensource>Paris</arg_value:opensource>
//   </tool_call:opensource>
//   </tool_calls:opensource>
const HY_CALL_OPEN: &str = "<tool_call:opensource>";
const HY_CALL_CLOSE: &str = "</tool_call:opensource>";
const HY_SEP: &str = "<tool_sep:opensource>";
const HY_AK: &str = "<arg_key:opensource>";
const HY_AK_END: &str = "</arg_key:opensource>";
const HY_AV: &str = "<arg_value:opensource>";
const HY_AV_END: &str = "</arg_value:opensource>";

/// Everything the model produced, split into the prose it wrote and the calls it made.
pub struct ParsedOutput {
    /// Text outside any `<tool_call>` block. The template explicitly permits reasoning BEFORE a call.
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

/// True if `s` contains the start of a tool call — used by the streaming path to decide whether it
/// must hold text back rather than forward it to the client as content. `<tool_call` is the shared
/// prefix of both families' tags.
pub fn has_tool_call(s: &str) -> bool {
    s.contains("<tool_call")
}

/// Parse a completed generation. `tools` is the request's `tools` array, used only to type-coerce
/// arguments; parsing still works (as strings) when it is absent. Format dispatch is by content:
/// hy_v3's `:opensource` markup or qwen's `<function=…>` markup.
pub fn parse(text: &str, tools: Option<&[Value]>) -> ParsedOutput {
    parse_with(text, tools, quote_pairing())
}

/// `parse` with the value-terminator rule given explicitly (`--tool-quote-pairing`; the tests and
/// the offline replay of `--dump-tools` collections drive both settings through this).
pub fn parse_with(text: &str, tools: Option<&[Value]>, pairing: QuotePairing) -> ParsedOutput {
    if text.contains(HY_CALL_OPEN) {
        return parse_hy3(text, tools);
    }
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    let mut rest = text;

    while let Some(open) = rest.find(CALL_OPEN) {
        content.push_str(&rest[..open]);
        let after = &rest[open + CALL_OPEN.len()..];
        // A truncated call (hit max_tokens mid-emit) has no close tag. Drop it entirely: a half-parsed
        // call is worse than none, because the harness would invoke a real function with missing
        // arguments. `rest` MUST be cleared before breaking -- leaving it would append the partial XML
        // to content (and duplicate the prose before it), which is the exact leak this guards against.
        let Some(close) = after.find(CALL_CLOSE) else { rest = ""; break };
        for tc in parse_block(&after[..close], tools, tool_calls.len(), pairing) {
            tool_calls.push(tc);
        }
        rest = &after[close + CALL_CLOSE.len()..];
    }
    content.push_str(rest);

    ParsedOutput { content: content.trim().to_string(), tool_calls }
}

/// hy_v3 variant: prose is everything before the wrapper; every `<tool_call:opensource>…</…>`
/// block inside it becomes a call. A truncated block is dropped (same half-call rule as qwen).
fn parse_hy3(text: &str, tools: Option<&[Value]>) -> ParsedOutput {
    let mut tool_calls = Vec::new();
    // The wrapper `<tool_calls:opensource>` and the first call block start one char apart —
    // prose ends at whichever comes first, so the wrapper never leaks into the content.
    let first = ["<tool_calls:opensource>", HY_CALL_OPEN].iter()
        .filter_map(|t| text.find(t)).min().unwrap();
    let prose = &text[..first];
    let mut rest = &text[first..];
    while let Some(open) = rest.find(HY_CALL_OPEN) {
        let after = &rest[open + HY_CALL_OPEN.len()..];
        let Some(close) = after.find(HY_CALL_CLOSE) else { break };
        if let Some(tc) = parse_one_hy3(&after[..close], tools) {
            tool_calls.push(tc);
        }
        rest = &after[close + HY_CALL_CLOSE.len()..];
    }
    ParsedOutput { content: prose.trim().to_string(), tool_calls }
}

/// Parse one hy_v3 call body: `NAME<tool_sep:opensource>` then
/// `<arg_key:opensource>K</arg_key:opensource><arg_value:opensource>V</arg_value:opensource>` pairs.
fn parse_one_hy3(body: &str, tools: Option<&[Value]>) -> Option<ToolCall> {
    let sep = body.find(HY_SEP)?;
    let name = body[..sep].trim().to_string();
    if name.is_empty() { return None; }
    let schema = tools.and_then(|ts| param_schema(ts, &name));
    let mut args = serde_json::Map::new();
    let mut rest = &body[sep + HY_SEP.len()..];
    loop {
        let Some(k0) = rest.find(HY_AK) else { break };
        let a = &rest[k0 + HY_AK.len()..];
        let Some(k1) = a.find(HY_AK_END) else { break };
        let key = a[..k1].trim().to_string();
        let v = &a[k1 + HY_AK_END.len()..];
        let Some(v0) = v.find(HY_AV) else { break };
        let v = &v[v0 + HY_AV.len()..];
        let Some(v1) = v.find(HY_AV_END) else { break };
        let raw = v[..v1].trim_matches('\n');
        args.insert(key.clone(), coerce(raw, schema.and_then(|s| s.get(&key))));
        rest = &v[v1 + HY_AV_END.len()..];
    }
    Some(ToolCall {
        id: format!("call_{}", CALL_SEQ.fetch_add(1, Ordering::Relaxed)),
        kind: "function".to_string(),
        function: FunctionCall {
            name,
            arguments: serde_json::to_string(&Value::Object(args)).unwrap_or_else(|_| "{}".into()),
        },
    })
}

/// Locate the function tag inside one `<tool_call>` body.
///
/// The well-formed tag is `<function=NAME>`. In live serving (2026-08-27, 2 of 38 sampled
/// tool requests) the model SOMETIMES DROPS THE `<` and emits `<tool_call>\nfunction=NAME>`.
/// That used to fail `parse_one` entirely, and the two response modes then diverged: streaming
/// (which had already held the block back from the content stream) dropped the text, while
/// non-streaming leaked the raw block into `content`. Inside a `<tool_call>` block a `function=`
/// at the start of a line is unambiguous, so accept the bare form as a repair — the well-formed
/// match always wins when both are present. Returns `(byte offset, tag length)`.
fn find_function_tag(body: &str) -> Option<(usize, usize)> {
    find_function_tag_mode(body, false)
}

/// `wf_only`: the whole `<tool_call>` body contains a well-formed `<function=` somewhere, so a bare `function=NAME>` is never a
/// repair candidate in it (the well-formed form always wins — also AFTER the first call, where a bare-looking line inside a
/// parameter value used to be taken for a second call).
fn find_function_tag_mode(body: &str, wf_only: bool) -> Option<(usize, usize)> {
    const WF: &str = "<function=";
    const BARE: &str = "function=";
    if let Some(i) = body.find(WF) {
        return Some((i, WF.len()));
    }
    if wf_only { return None; }
    let mut from = 0usize;
    while let Some(rel) = body[from..].find(BARE) {
        let i = from + rel;
        let boundary_ok = match body[..i].chars().next_back() {
            None | Some('\n') | Some('\r') | Some(' ') | Some('\t') => true,
            _ => false,
        };
        if boundary_ok {
            return Some((i, BARE.len()));
        }
        from = i + BARE.len();
    }
    None
}

/// Parse one `<tool_call>…</tool_call>` body into ZERO OR MORE calls.
///
/// Two shapes are supported; the body decides:
///  * a JSON object (the Froggeric `tool_call_format=json` emission) — one call, with the
///    tolerant salvage rules in `parse_json_call`;
///  * the XML form — EVERY `<function=NAME>` in the block becomes its own call. Before Phase 2
///    only the FIRST function tag produced a call and every later function's `<parameter=…>`
///    pairs were merged into it, same-key values overwritten (RENDER_AUDIT.md 9.2).
fn parse_block(body: &str, tools: Option<&[Value]>, idx: usize, pairing: QuotePairing) -> Vec<ToolCall> {
    let trimmed = body.trim_start();
    if trimmed.starts_with('{') {
        if let Some(tc) = parse_json_call(trimmed, tools, idx) {
            return vec![tc];
        }
        // Unparseable JSON body: fall through to the XML path rather than dropping the call.
    }
    parse_xml_calls(body, tools, pairing)
}

/// Skip the whitespace that may sit between structural tags. Deliberately the SAME set
/// `find_function_tag_mode` accepts before a bare `function=` repair (newline, CR, space, tab) —
/// not arbitrary Unicode whitespace, which is content.
fn skip_tag_ws(s: &str) -> &str {
    let mut i = 0;
    for b in s.bytes() {
        if matches!(b, b'\n' | b'\r' | b' ' | b'\t') { i += 1 } else { break }
    }
    &s[i..]
}

/// The structural-terminator lookahead (v0.7.3): does the text AFTER a candidate `</parameter>`
/// prove the candidate is the real close of the value?
///
/// The template's close is always followed (tag whitespace aside) by one of:
///  * `<parameter=` — the next parameter of the same call;
///  * `</function>` — the call close, itself followed by the end of the block body or a further
///    `<function=` / bare `function=` tag (another call in the same `<tool_call>` block);
///  * the next call's tag directly — `<function=` or a bare `function=` repair (K3: malformed
///    output can OMIT `</function>` between two calls; the same tag detection
///    `find_function_tag_mode` applies decides what counts as a tag);
///  * nothing — the end of the block body (today's tolerance for a missing `</function>`).
///
/// Anything else (e.g. `\n</parameter>…` — the reporter's file-about-tools payload embeds the
/// sequence `</parameter>\n</function>` MID-value) means the candidate is quoted content, not
/// the close.
fn structural_continuation(after: &str) -> bool {
    let rest = skip_tag_ws(after);
    if rest.is_empty() || rest.starts_with("<parameter=") {
        return true;
    }
    // K3 (REL v0.7.3 review): the next call's tag DIRECTLY after the close (no
    // `</function>` between the calls) also proves the candidate is the real close.
    // Before this the close was treated as quoted content and the next call's
    // `<parameter=…>` was swallowed into THIS call's value — the pre-v0.7.3 first-close
    // parser returned both calls. `skip_tag_ws` already consumed exactly the boundary
    // characters `find_function_tag_mode` accepts before a bare tag, so a tag at
    // position 0 here is a tag there too.
    if rest.starts_with("<function=") || rest.starts_with("function=") {
        return true;
    }
    let Some(tail) = rest.strip_prefix("</function>") else { return false };
    let tail = skip_tag_ws(tail);
    tail.is_empty() || tail.starts_with("<function=") || tail.starts_with("function=")
}

/// End of a `<parameter=…>` value under the v0.7.3 rule: the byte offset (relative to `vstart`,
/// which runs to the end of the block body) of the STRUCTURAL `</parameter>`, or `None` when the
/// value is never closed (truncated output).
///
/// Pre-v0.7.3 this was `vstart.find("</parameter>")` — the FIRST close tag — so a value quoting
/// the tool format itself was silently truncated at the embedded tag (an agent writing a file
/// ABOUT tool calls lost the tail of the file). Each candidate close is validated by
/// `structural_continuation`; rejected candidates keep the scan going.
///
/// Fallback: when NO candidate satisfies the rule (malformed or truncated output) the FIRST
/// candidate is used — exactly the old behaviour, so garbage in still parses as it did.
///
/// Cost on the common path (exactly one `</parameter>` in the value): one `find` plus one O(1)
/// lookahead check; the fallback scan never runs. The candidate scan is SEQUENTIAL (each `find`
/// resumes where the previous candidate ended), so even a 1 MB value full of embedded close tags
/// costs one linear pass, never quadratic restarts.
///
/// S-B9-REL-SURFACE2B R1: this is the DEFAULT (`--tool-quote-pairing off`) and its body is the
/// v0.7.3 function unchanged. Do not edit it: `quote_pairing_off_is_v073` compares it with the
/// verbatim copy in the tests on every block of the exhaustive sets.
fn param_end_v073(vstart: &str) -> Option<usize> {
    const CLOSE: &str = "</parameter>";
    let first = vstart.find(CLOSE)?;
    let mut cand = first;
    loop {
        let after = &vstart[cand + CLOSE.len()..];
        if structural_continuation(after) {
            return Some(cand);
        }
        let Some(rel) = after.find(CLOSE) else { break };
        cand += CLOSE.len() + rel;
    }
    // No candidate satisfies the structural rule: keep today's behaviour (the first close tag).
    Some(first)
}

/// `--tool-quote-pairing`: how the close of a `<parameter=…>` value is found (XML tool calls).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuotePairing {
    /// The v0.7.3 rule alone (`param_end_v073`). The default: byte-identical to v0.7.3.
    Off,
    /// Pair the tags of quoted calls first (`pair_scan`); the v0.7.3 rule decides whatever the
    /// pairing does not.
    On,
}

/// The server-wide setting (the registry row's default when the option is not set).
pub fn quote_pairing() -> QuotePairing {
    QuotePairing::Off // FROZEN COPY: the option is not read (the default of v0.7.3 is what this module is)
}

/// What `pair_scan` found for one value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Scan {
    /// The first candidate the v0.7.3 lookahead accepts, reached without paying any debt: this IS
    /// the v0.7.3 answer (the two scans tested the same candidates in the same order).
    Plain(usize),
    /// A close accepted after `paid` earlier closes were skipped as the closes of quoted
    /// parameters. `tail_clean`: no `<parameter=` and no `<function=` lies between the last
    /// skipped close and this one.
    Paired { cand: usize, tail_clean: bool },
    /// No close was accepted before the end of the block (`scanned` bytes read).
    Undecided { scanned: usize },
}

/// S-B9-REL-SURFACE2 item 10 (nesting), the pairing scan. The v0.7.3 lookahead alone accepts the
/// close of a parameter that was opened INSIDE the value whenever more structure follows it: a
/// value quoting one call with two parameters is cut at the first quoted close (what follows is
/// `<parameter=`), and a value quoting TWO calls comes back as two calls, the outer one cut short
/// and a second, PHANTOM call built from the quoted text, which an agent would execute. So this
/// scan pairs the tags of QUOTED CALLS: once the value has shown a `<function=` tag, every
/// `<parameter=` it passes is owed one `</parameter>`, and a close that pays such a debt is
/// content. Only a close with nothing owed is tested with the lookahead. Quoted calls are balanced
/// text, so the value's own close is the first one left over.
///
/// Why only after a `<function=`: a value may mention a parameter tag ALONE (`old_string` =
/// `<parameter=city>` in an edit of a document about this format). Counting that open would make
/// the value's real close pay for it and send the scan on into the following parameters and calls
/// (two edits in one block, one replacing the open tag and one the close tag, came back as one
/// call). For the same reason an open tag with a BLANK value is not counted: a value that ends in
/// the quoted opening of a call (`<function=x>` + `<parameter=p>`, then the real close) is
/// byte-identical to a quoted parameter with no value.
///
/// The scan reports; `param_end_paired` decides (see there for the guard and the fallbacks).
fn pair_scan(vstart: &str, first: usize) -> Scan {
    const OPEN: &str = "<parameter=";
    const CLOSE: &str = "</parameter>";
    const CALL: &str = "<function=";
    // `owed` = `<parameter=` tags opened inside the value, after a `<function=` tag, and not yet
    // closed. `paid_end` = the end of the last close that paid such a debt.
    let (mut owed, mut scanned, mut cand, mut in_quoted_call) = (0usize, 0usize, first, false);
    let mut paid_end: Option<usize> = None;
    loop {
        let window = &vstart[scanned..cand];
        let from = if in_quoted_call { Some(0) } else { window.find(CALL) };
        // this close directly follows an open tag with a blank value
        let mut blank_pair = false;
        if let Some(f) = from {
            in_quoted_call = true;
            let w = &window[f..];
            let mut opened = w.matches(OPEN).count();
            // An open tag with NOTHING between it and this close is not counted: a value that ends
            // in a quoted `<parameter=k>` looks exactly like that (the close is then the value's own).
            if let Some(last) = w.rfind(OPEN) {
                if let Some(gt) = w[last..].find('>') {
                    if skip_tag_ws(&w[last + gt + 1..]).is_empty() { opened -= 1; blank_pair = true; }
                }
            }
            owed += opened;
        }
        scanned = cand + CLOSE.len();
        let after = &vstart[scanned..];
        if owed > 0 {
            // The close of a quoted parameter: content. It pays one debt, unless it is the close
            // of the blank pair just seen: that open was not counted, so its close pays for no
            // OTHER open either (review 2026-10-10: it used to, and a later value holding
            // `<parameter=p></parameter>` then ended an earlier value that mentions the tags).
            if !blank_pair {
                owed -= 1;
                paid_end = Some(scanned);
            }
        } else if structural_continuation(after) {
            return match paid_end {
                None => Scan::Plain(cand),
                Some(p) => {
                    let tail = &vstart[p..cand];
                    Scan::Paired { cand, tail_clean: !tail.contains(OPEN) && !tail.contains(CALL) }
                }
            };
        }
        let Some(rel) = after.find(CLOSE) else { break };
        cand = scanned + rel;
    }
    Scan::Undecided { scanned: vstart.len() }
}

/// Per-block state of `--tool-quote-pairing on`: the bytes pairing scans may still read without
/// deciding anything (S-B9-REL-SURFACE2B R1 [m1]).
///
/// A pairing scan whose close is not used (not trusted, refused by the catalogue, or none found)
/// has read past the close the v0.7.3 rule then picks, and the next value of the block starts
/// before the place the scan reached. A block of many values that
/// each open more quoted parameters than they close made every value scan to the end of the
/// block: quadratic (479 KB took 3.2 s, on the request path). The budget makes the total linear:
/// scans whose close IS used advance the parse by what they read, and the others share
/// `PAIR_BUDGET` x the block length. When it is spent the remaining values of the block use the v0.7.3 rule alone.
struct PairState {
    budget: usize,
}

/// Undecided pairing scans may read this many times the block length (plus a floor for small
/// blocks) before pairing is switched off for the rest of the block. A block of N values can
/// waste at most N block lengths, so no block of 8 or fewer values is affected.
const PAIR_BUDGET: usize = 8;

impl PairState {
    fn new(body_len: usize) -> Self { PairState { budget: PAIR_BUDGET * body_len + 4096 } }
}

/// The call a value belongs to: what the catalogue tie-break of `param_end_paired` looks at.
struct ValueCx<'a> {
    /// the request's `tools` array (`None` = not known: no tie-break)
    tools: Option<&'a [Value]>,
    /// the tool the call names
    name: &'a str,
    /// the parameter whose value is being read
    key: &'a str,
    /// the parameters of this call read so far
    given: &'a serde_json::Map<String, Value>,
}

/// Does the structure right after a `</parameter>` fit the request's catalogue as REAL structure?
/// A next parameter must be declared for the call's tool and not given yet; a next call (directly,
/// or after `</function>`) must name an offered tool; the end of the block fits. `None` = the
/// catalogue says nothing: no catalogue, text that is not structure, or a parameter of a tool that
/// is not offered, declares no properties, accepts additional ones (`additionalProperties` true
/// or a schema; absent counts as closed), or composes its schema (`anyOf` / `oneOf` / `allOf` /
/// `$ref` / `patternProperties`: only the top-level `properties` are read).
fn continuation_fits(after: &str, cx: &ValueCx) -> Option<bool> {
    let tools = cx.tools?;
    let tool = |name: &str| tools.iter().find(|t| t.pointer("/function/name").and_then(Value::as_str) == Some(name));
    let offered = |r: &str| r.find('>').map(|gt| tool(r[..gt].trim()).is_some());
    fn call_tag(r: &str) -> Option<&str> { r.strip_prefix("<function=").or_else(|| r.strip_prefix("function=")) }
    let rest = skip_tag_ws(after);
    if rest.is_empty() { return Some(true); }
    if let Some(r) = rest.strip_prefix("<parameter=") {
        let k = r[..r.find('>')?].trim();
        let params = tool(cx.name)?.pointer("/function/parameters")?;
        let declared = params.get("properties").and_then(Value::as_object).filter(|p| !p.is_empty())?;
        if matches!(params.get("additionalProperties"), Some(Value::Bool(true)) | Some(Value::Object(_))) { return None; }
        // a schema that composes others may declare parameters this list does not show
        if ["anyOf", "oneOf", "allOf", "$ref", "patternProperties"].iter().any(|k| params.get(*k).is_some()) { return None; }
        return Some(k != cx.key && !cx.given.contains_key(k) && declared.contains_key(k));
    }
    if let Some(r) = call_tag(rest) { return offered(r); }
    let tail = skip_tag_ws(rest.strip_prefix("</function>")?);
    if tail.is_empty() { return Some(true); }
    offered(call_tag(tail)?)
}

/// End of a value under `--tool-quote-pairing on`.
///
/// 1. `pair_scan`. If it accepts a close without having skipped any, that is the v0.7.3 answer.
/// 2. If it skipped closes as the closes of quoted parameters, the result is TRUSTED only when
///    no `<parameter=` and no `<function=` lies between the last skipped close and the accepted
///    one (the guard of the independent review, 2026-10-09). A quoted call is balanced text, so
///    after its last parameter closes nothing opens again before the value's own close. Without
///    the guard a value that merely MENTIONS the format with more open tags than close tags (one
///    prose sentence is enough) ran on into the following real parameters, and a later real
///    parameter with a blank value ended it there: `edit(path, old_string=<that sentence>,
///    new_string="")` lost `new_string`, and two real calls came back as one.
/// 3. A trusted result that differs from the v0.7.3 close is checked against the request's
///    catalogue when there is one (`continuation_fits`): if what follows the v0.7.3 close fits
///    the catalogue and what follows the paired close does NOT (a parameter this tool does not
///    declare or was already given, a call to a tool that is not offered), the paired close lies
///    inside quoted text and the v0.7.3 close is kept. The catalogue only ever breaks this tie:
///    when both fit (a quoted call to an OFFERED tool with its declared parameters) it cannot
///    tell a quotation from real structure, and the pairing stands.
/// 4. Otherwise (not trusted, or no close accepted) the v0.7.3 rule decides, and after it the
///    first close tag.
///
/// The counts of each step against v0.7.3 are in `exhaustive_small_blocks`.
///
/// NOT solved, and not solvable inside this format (pinned by the tests named here):
///  * a value that ENDS with a quoted close followed by a quoted call is byte-for-byte two real
///    calls in one block (`a_value_that_ends_in_a_forged_close_and_call`);
///  * a quoted call whose LAST parameter is blank is not paired (the blank-value exemption of
///    `pair_scan`), so a second quoted call after it is still returned as a call
///    (`a_quoted_call_with_a_blank_last_argument_still_yields_a_phantom`);
///  * a value that opens more quoted parameters than it closes, in a block where a later value
///    holds a `</parameter>` that no open tag of a quoted call accounts for (a lone close tag),
///    runs on to that close: the class this setting reads wrongly and v0.7.3 rightly
///    (`exhaustive_small_blocks`, the "regressed" column). A MALFORMED block (text between a
///    real close and the next tag) can differ too: there both rules run on, to different closes.
fn param_end_paired(vstart: &str, st: &mut PairState, cx: &ValueCx) -> Option<usize> {
    const CLOSE: &str = "</parameter>";
    let first = vstart.find(CLOSE)?;
    if st.budget > 0 {
        let scan = pair_scan(vstart, first);
        #[cfg(test)]
        PAIR_READ.with(|c| c.set(c.get() + match scan { Scan::Plain(c) | Scan::Paired { cand: c, .. } => c, Scan::Undecided { scanned } => scanned }));
        match scan {
            Scan::Plain(cand) => return Some(cand),
            Scan::Paired { cand, tail_clean: true } => {
                let old = param_end_v073(vstart)?;
                let fits = |c: usize| continuation_fits(&vstart[c + CLOSE.len()..], cx);
                let quoted_tail = old != cand && fits(old) == Some(true) && fits(cand) == Some(false);
                if !quoted_tail { return Some(cand); }
                // the scan read to `cand` and the parse only advances to `old`: that is a scan
                // that did not decide, and it is charged like the others (review 2026-10-10:
                // uncharged, a block of such values was quadratic again)
                st.budget = st.budget.saturating_sub(cand);
                return Some(old);
            }
            Scan::Paired { cand, tail_clean: false } => st.budget = st.budget.saturating_sub(cand),
            Scan::Undecided { scanned } => st.budget = st.budget.saturating_sub(scanned),
        }
    }
    param_end_v073(vstart)
}

#[cfg(test)]
thread_local! {
    /// Bytes the pairing scans of this thread have read (the linearity test reads it).
    static PAIR_READ: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The XML form: one call per `<function=NAME>` tag, each owning the `<parameter=…>` pairs up
/// to the next tag. Also accepts a JSON-object body inside the function tag (the Froggeric
/// template's string-arguments history form) when no `<parameter=` pair is present.
fn parse_xml_calls(body: &str, tools: Option<&[Value]>, pairing: QuotePairing) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut pair = PairState::new(body.len());
    // A body with any well-formed tag takes no bare repairs at all (see find_function_tag_mode).
    let wf_only = body.contains("<function=");
    // The scan is SEQUENTIAL: a call owns its `<parameter=…>…</parameter>` spans, and a function tag is only a tag when it sits
    // OUTSIDE every value span consumed so far. A parameter value that merely contains the text `function=x>` or `<function=x>`
    // (a coding agent writing a file about this very format) stays payload and never becomes a second call.
    let mut from = 0usize;
    while let Some((i, tlen)) = find_function_tag_mode(&body[from..], wf_only) {
        let start = from + i;
        let name_from = start + tlen;
        let Some(gt) = body[name_from..].find('>') else { from = name_from; continue };
        let name = body[name_from..name_from + gt].trim().to_string();
        let after_name = name_from + gt + 1;
        if name.is_empty() { from = after_name; continue; }

        let schema = tools.and_then(|ts| param_schema(ts, &name));
        let mut args = serde_json::Map::new();
        // `cursor` is always outside a value span. The next `<parameter=` belongs to THIS call unless the next function tag comes
        // first (then it is that call's); the end of this call's text is the next tag after the last consumed span.
        let mut cursor = after_name;
        loop {
            let next_tag = find_function_tag_mode(&body[cursor..], wf_only).map(|(ti, _)| cursor + ti);
            let next_param = body[cursor..].find("<parameter=").map(|pi| cursor + pi);
            let Some(popen) = next_param else { break };
            if next_tag.map_or(false, |t| t < popen) { break; }
            let a = &body[popen + "<parameter=".len()..];
            let Some(gt2) = a.find('>') else { break };
            let key = a[..gt2].trim().to_string();
            let vstart = &a[gt2 + 1..];
            // v0.7.3: the STRUCTURAL close, not merely the first close tag — a value that quotes
            // the tool format keeps its embedded `</parameter>` as content. `--tool-quote-pairing
            // on` pairs the tags of quoted calls before that rule is asked.
            let pclose = match pairing {
                QuotePairing::Off => param_end_v073(vstart),
                QuotePairing::On => param_end_paired(vstart, &mut pair, &ValueCx { tools, name: &name, key: &key, given: &args }),
            };
            let Some(pclose) = pclose else { break };
            // The template puts a newline after `>` and before `</parameter>`; they are delimiters,
            // not part of the value. Trim only those, so interior whitespace of a multi-line value
            // survives.
            let raw = vstart[..pclose].trim_matches('\n');
            args.insert(key.clone(), coerce(raw, schema.and_then(|s| s.get(&key))));
            cursor = body.len() - vstart.len() + pclose + "</parameter>".len();
        }
        // This call's text runs to the next function tag after the last consumed value span (or the end of the block).
        let end = find_function_tag_mode(&body[cursor..], wf_only).map(|(ti, _)| cursor + ti).unwrap_or(body.len());
        let after = &body[start + tlen..end];
        if args.is_empty() {
            let jb = after[gt + 1..].trim();
            let jb = jb.strip_suffix("</function>").unwrap_or(jb).trim();
            // A body that LOOKS like JSON is a JSON body: the Froggeric template's
            // `raw_tool_args_history` branch renders the model's own previous calls as a JSON
            // object inside the function tag, and the model imitates that while closing with the
            // `</parameter>` tag it also sees in the XML example. So the emitted shape is
            // `{…json…}\n</parameter>` — ONE stray trailing close tag. Strip it (and the
            // whitespace between) before parsing. Before this, every such body fell through to
            // `args` = {} and the harness invoked a real tool with no arguments: 6/6 lost
            // (phase3/EMPTY_ARGS_EVIDENCE.md, PHASE3_REPORT.md 4.3), while the 5/5 clean-JSON
            // bodies parsed. A non-empty body that still fails to yield arguments is ALARMED,
            // never silently dropped.
            let jsonish = jb.starts_with('{') || jb.starts_with('[');
            if jsonish {
                let jb = strip_stray_param_close(jb);
                // Phase-5 B1: `json_object_lenient` = strict parse, then the unescaped-quote
                // repair. An array, an unrepairable body or a truncated one still alarms.
                match json_object_lenient(jb) {
                    Some(o) => for (k, v) in o { args.insert(k, v); },
                    None => eprintln!("[tool-args-alarm] {jb}"),
                }
            } else if !jb.is_empty() {
                eprintln!("[tool-args-alarm] {jb}");
            }
        }
        calls.push(ToolCall {
            id: format!("call_{}", CALL_SEQ.fetch_add(1, Ordering::Relaxed)),
            kind: "function".to_string(),
            function: crate::tokenizer::FunctionCall {
                name,
                arguments: serde_json::to_string(&Value::Object(args)).unwrap_or_else(|_| "{}".into()),
            },
        });
        from = end;
    }
    calls
}

/// One JSON-body call. Strict `{"name": T, "arguments": …}` is the shipped form and is tried
/// first; the three MALFORMED shapes the Phase-1 ledger proved the model actually emits inside
/// `<tool_call>` are salvaged after it (LEDGER_REPORT.md 1.1.1):
///   * wrapper    `{"function": T, "arguments": {…}}` / `{"function": {"name": T, …}}`
///   * flattened  `{"function": T, "k": v, …}` — the remaining keys ARE the arguments
///   * name-first `{T, "k": "v"}` — not valid JSON; repaired, T is the tool name
/// The salvage shapes require the recovered name to be a tool of the request catalog (when the
/// catalog is known), so structured-output JSON with an unrelated `name` field is never promoted
/// to a call. The strict form keeps its original behaviour (no catalog check).
fn parse_json_call(body: &str, tools: Option<&[Value]>, idx: usize) -> Option<ToolCall> {
    let t = body.trim();
    if let Ok(v) = serde_json::from_str::<Value>(t) {
        return call_from_object(v.as_object()?, tools, idx);
    }
    // 2b. Phase-5 B1: valid JSON except for UNESCAPED quotes inside a string value —
    //     `{"language":"python","code":"print("correct")"}`. The NVFP4 lane emits this ~4x
    //     more often than FP8 (phase-4 alarm census); before the repair every argument was
    //     lost. Repair first, then run the SAME shape resolution as the strict path.
    if repair_quotes_enabled() {
        if let Some(rep) = repair_unescaped_quotes(t) {
            if let Ok(Value::Object(o)) = serde_json::from_str::<Value>(&rep) {
                if let Some(tc) = call_from_object(&o, tools, idx) {
                    return Some(tc);
                }
            }
        }
    }
    // 3. not valid JSON: `{T, "k": "v"}` (name-first, unquoted name) — repair.
    let (name, args) = salvage_name_first(t, tools)?;
    Some(json_call(name, args, idx))
}

/// Resolve ONE already-parsed JSON object into a call, using the four shapes the ledger proved
/// the model emits: strict OpenAI `{"name":T,"arguments":…}`, flattened `{"name":T,"k":v,…}`,
/// wrapper `{"function":T|{…},"arguments":…}` and flattened `{"function":T,"k":v,…}`. Returns
/// `None` for any object that is not one of them (never guesses a name).
fn call_from_object(o: &serde_json::Map<String, Value>, tools: Option<&[Value]>, idx: usize) -> Option<ToolCall> {
    // 1. strict OpenAI form.
    if let Some(name) = o.get("name").and_then(Value::as_str).filter(|s| !s.is_empty()) {
        if let Some(args) = o.get("arguments") {
            return Some(json_call(name.to_string(), args_to_string(args), idx));
        }
        // 1b. flattened name form: {"name": T, "k": v, …}
        if let Some(args) = flat_args(o, &["name"]) {
            return Some(json_call(name.to_string(), args, idx));
        }
    }
    // 2. function-key wrapper / flattened form.
    if let Some(f) = o.get("function") {
        let (name, args) = match f {
            Value::String(s) if !s.is_empty() => match o.get("arguments") {
                Some(a) => (s.clone(), args_to_string(a)),
                None => (s.clone(), flat_args(o, &["function"]).unwrap_or_else(|| "{}".into())),
            },
            Value::Object(fo) => {
                let name = fo.get("name").and_then(Value::as_str)?.to_string();
                let a = fo.get("arguments").or_else(|| o.get("arguments"));
                (name, a.map(args_to_string).unwrap_or_else(|| "{}".into()))
            }
            _ => return None,
        };
        if !name.is_empty() && known_tool(tools, &name) {
            return Some(json_call(name, args, idx));
        }
        return None;
    }
    None
}

/// Phase-5 B1 negative control switch. Production is ALWAYS on (`#[cfg(not(test))]` below
/// returns true); tests flip it off in their own thread to prove the repair — and not some
/// other salvage path — is what recovers the archived bodies.
#[cfg(not(test))]
fn repair_quotes_enabled() -> bool { true }

#[cfg(test)]
thread_local! {
    static REPAIR_QUOTES: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}
#[cfg(test)]
fn repair_quotes_enabled() -> bool { REPAIR_QUOTES.with(|c| c.get()) }
#[cfg(test)]
fn set_repair_quotes(on: bool) { REPAIR_QUOTES.with(|c| c.set(on)) }

/// Phase-5 B1 — repair UNESCAPED double quotes inside JSON string VALUES.
///
/// Lex the text and escape every `"` that is INSIDE a string but cannot be its closing
/// delimiter. A closing quote is followed (after optional whitespace) by one of `, } ] :` —
/// the only characters that can legally follow a string in JSON; a quote followed by anything
/// else is part of the value and must be escaped. A backslash escapes the next character, so
/// already-escaped input (`\"`) is copied verbatim and never re-escaped.
///
/// `None` means NOT REPAIRABLE: nothing changed, a string was left unterminated, or the
/// repaired text still does not parse as a JSON object. The caller then keeps the
/// `[tool-args-alarm]` — an ambiguous body is never guessed (and a TRUNCATED body, which this
/// never completes, stays alarmed by policy: see the phase-5 truncated-body decision).
fn repair_unescaped_quotes(s: &str) -> Option<String> {
    let mut out = String::with_capacity(s.len() + 8);
    let mut in_string = false;
    let mut changed = false;
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if !in_string {
            if c == '"' { in_string = true; }
            out.push(c);
            continue;
        }
        match c {
            '\\' => {
                out.push(c);
                if let Some(n) = it.next() { out.push(n); }
            }
            '"' => {
                let closes = it.clone()
                    .find(|ch| !ch.is_whitespace())
                    .map_or(true, |ch| matches!(ch, ',' | '}' | ']' | ':'));
                if closes {
                    in_string = false;
                    out.push('"');
                } else {
                    out.push_str("\\\"");
                    changed = true;
                }
            }
            _ => out.push(c),
        }
    }
    if !changed || in_string { return None; }
    match serde_json::from_str::<Value>(&out) {
        Ok(Value::Object(_)) => Some(out),
        _ => None,
    }
}

/// Parse a function body as a JSON arguments object, applying the unescaped-quote repair when
/// strict parsing fails. `None` => the caller emits the alarm.
fn json_object_lenient(jb: &str) -> Option<serde_json::Map<String, Value>> {
    if let Ok(Value::Object(o)) = serde_json::from_str::<Value>(jb) {
        return Some(o);
    }
    if repair_quotes_enabled() {
        if let Some(rep) = repair_unescaped_quotes(jb) {
            if let Ok(Value::Object(o)) = serde_json::from_str::<Value>(&rep) {
                return Some(o);
            }
        }
    }
    None
}

/// Strip ONE stray trailing `</parameter>` (plus the whitespace between the JSON and that tag)
/// from a function body that otherwise looks like JSON. The model emits
/// `{…json…}\n</parameter>` because it imitates the JSON body our render puts in the history
/// while closing with the XML tag the system prompt shows. Returns the input unchanged when the
/// tag is absent.
fn strip_stray_param_close(jb: &str) -> &str {
    let t = jb.trim_end();
    match t.strip_suffix("</parameter>") {
        Some(rest) => rest.trim_end(),
        None => t,
    }
}

/// The RAW function body the model emitted for `name`, verbatim (only outer whitespace
/// trimmed). Used by the empty-argument alarm; `None` when the tag is not in the text.
fn raw_function_body<'a>(raw: &'a str, name: &str) -> Option<&'a str> {
    let wf = format!("<function={name}>");
    let bare = format!("function={name}>");
    let i = raw.find(&wf).or_else(|| raw.find(&bare))?;
    let tlen = if raw[i..].starts_with(&wf) { wf.len() } else { bare.len() };
    let rest = &raw[i + tlen..];
    let end = ["</function>", "</tool_call>", "<function="].iter()
        .filter_map(|m| rest.find(m))
        .min()
        .unwrap_or(rest.len());
    Some(rest[..end].trim())
}

/// Phase-4 A2 — the empty-argument alarm.
///
/// For every parsed call whose arguments are EMPTY while the tool's schema declares parameters,
/// return the raw function body the model emitted, when that body is non-empty. Callers log each
/// returned body as `[tool-args-alarm] <raw body>`. This is the net that makes ANY future
/// regression of the silent-argument-drop class visible in every serving log, instead of
/// surfacing as a tool that "ran" and did nothing.
pub fn empty_arg_alarms<'a>(raw: &'a str, tools: Option<&[Value]>, calls: &[ToolCall]) -> Vec<&'a str> {
    let mut out = Vec::new();
    let Some(ts) = tools else { return out };
    for c in calls {
        let a = c.function.arguments.trim();
        if !(a.is_empty() || a == "{}") { continue; }
        let declares = param_schema(ts, &c.function.name).map_or(false, |p| !p.is_empty());
        if !declares { continue; }
        if let Some(body) = raw_function_body(raw, &c.function.name) {
            if !body.is_empty() { out.push(body); }
        }
    }
    out
}

fn json_call(name: String, arguments: String, idx: usize) -> ToolCall {
    ToolCall {
        id: format!("call_{:02x}", idx),
        kind: "function".to_string(),
        function: crate::tokenizer::FunctionCall { name, arguments },
    }
}

/// `arguments` value → the JSON string our ToolCall carries: a string is kept verbatim, an
/// object/array is re-serialized, null becomes `{}`.
fn args_to_string(a: &Value) -> String {
    match a {
        Value::String(s) => s.clone(),
        Value::Null => "{}".to_string(),
        other => other.to_string(),
    }
}

/// Every object key except `skip`, serialized as an arguments object (the flattened shape).
fn flat_args(o: &serde_json::Map<String, Value>, skip: &[&str]) -> Option<String> {
    let m: serde_json::Map<String, Value> = o.iter()
        .filter(|(k, _)| !skip.contains(&k.as_str()))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    if m.is_empty() { None } else { Some(Value::Object(m).to_string()) }
}

/// Is `name` a tool of the request catalog? Unknown/absent catalog → true (the call is inside a
/// `<tool_call>` block, i.e. the model already declared the intent).
fn known_tool(tools: Option<&[Value]>, name: &str) -> bool {
    match tools {
        Some(ts) if !ts.is_empty() => ts.iter().any(|t| {
            t.pointer("/function/name").and_then(Value::as_str) == Some(name)
                || t.get("name").and_then(Value::as_str) == Some(name)
        }),
        _ => true,
    }
}

/// Repair the invalid `{T, "k": "v"}` shape: first token is the tool name (quoted or bare), the
/// rest is the argument object once wrapped in braces.
fn salvage_name_first(t: &str, tools: Option<&[Value]>) -> Option<(String, String)> {
    let inner = t.strip_prefix('{')?.trim_start();
    let (name, rest) = if let Some(stripped) = inner.strip_prefix('"') {
        let end = stripped.find('"')?;
        (stripped[..end].to_string(), &stripped[end + 1..])
    } else {
        let end = inner.find(|c| c == ',' || c == '}')?;
        (inner[..end].trim().to_string(), &inner[end..])
    };
    if name.is_empty() || !known_tool(tools, &name) { return None; }
    let rest = rest.trim_start().trim_start_matches(',').trim();
    let rest = rest.strip_suffix('}').unwrap_or(rest).trim();
    if rest.is_empty() { return Some((name, "{}".to_string())); }
    let repaired = format!("{{{rest}}}");
    let v: Value = serde_json::from_str(&repaired).ok()?;
    if !v.is_object() { return None; }
    Some((name, v.to_string()))
}

/// `tools[i].function.parameters.properties` for the named function.
fn param_schema<'a>(tools: &'a [Value], name: &str) -> Option<&'a serde_json::Map<String, Value>> {
    tools.iter()
        .find(|t| t.pointer("/function/name").and_then(Value::as_str) == Some(name))
        .and_then(|t| t.pointer("/function/parameters/properties"))
        .and_then(Value::as_object)
}

/// The ONE final (content, tool_calls, finish_reason) decision for a completed generation,
/// shared by BOTH response modes — streaming SSE and non-streaming JSON.
///
/// This exists because the modes used to decide independently and could diverge (the
/// 2026-08-27 user report): with a call the parser could not read, streaming dropped the
/// block's text entirely while non-streaming returned it verbatim. Both modes now call this
/// on the same post-think-split answer text, so they can only ever agree.
///
/// Rules (the exact old non-streaming behavior, now canonical for both):
///  - at least one call parsed  -> content becomes the prose outside the calls (`None` when
///    empty) and finish becomes `"tool_calls"` — the flag every harness branches on;
///  - no call parsed            -> content is the model's literal answer text UNCHANGED
///    (recoverable by the operator) and `finish` passes through.
pub fn finalize(raw: &str, tools: Option<&[Value]>, finish: &str) -> (Option<String>, Vec<ToolCall>, String) {
    let parsed = parse(raw, tools);
    finalize_parsed(raw, parsed, finish)
}

/// Decision half of [`finalize`] for callers that already ran `parse` (e.g. to log the raw
/// output). Takes the parse result by value so call ids are minted exactly once.
pub fn finalize_parsed(
    raw: &str,
    parsed: ParsedOutput,
    finish: &str,
) -> (Option<String>, Vec<ToolCall>, String) {
    if parsed.tool_calls.is_empty() {
        (Some(raw.to_string()), Vec::new(), finish.to_string())
    } else {
        let c = if parsed.content.is_empty() { None } else { Some(parsed.content) };
        (c, parsed.tool_calls, "tool_calls".to_string())
    }
}

/// What the STREAMING mode must still surface after its incremental emission, so it never
/// silently drops text the non-streaming mode returns. `acc` is the full generated text and
/// `emitted` the byte offset already sent as content chunks (the hold-back in server.rs stops
/// at the first `<tool_call`, so everything after it sits in the remainder).
///
/// `None` when nothing is held back. Some(text) only when the held-back span actually contains
/// a tool-call marker: if the parser read the calls, the block is represented by the
/// `tool_calls` delta instead (and unparsed block text is dropped by BOTH modes identically —
/// that is `finalize_parsed`'s rule); if the parser read NOTHING from a block that looked like
/// a call, the raw text is surfaced as a final content chunk, exactly what non-streaming does.
///
/// S-B9-REL-SURFACE2B R5: the server's stream no longer calls this. It sends EVERY held tail at
/// the finish (`server::held_tail`): a tail that is only a marker prefix (`<`) was lost by this
/// marker-only rule. Kept for the integration test that models the marker case.
pub fn held_back_remainder<'a>(acc: &'a str, emitted: usize) -> Option<&'a str> {
    if emitted >= acc.len() {
        return None;
    }
    let rest = &acc[emitted..];
    if rest.contains(CALL_OPEN) || rest.contains(HY_CALL_OPEN) || rest.contains("<tool_calls") {
        Some(rest)
    } else {
        None
    }
}

/// Coerce one text value to the type its schema declares. Unknown/absent schema, or a value that does
/// not parse as the declared type, stays a string — never guess a type the schema did not ask for.
fn coerce(raw: &str, schema: Option<&Value>) -> Value {
    let ty = schema.and_then(|s| s.get("type")).and_then(Value::as_str);
    match ty {
        Some("integer") => raw.trim().parse::<i64>().map(Value::from).unwrap_or_else(|_| Value::String(raw.to_string())),
        Some("number")  => raw.trim().parse::<f64>().map(Value::from).unwrap_or_else(|_| Value::String(raw.to_string())),
        Some("boolean") => match raw.trim().to_ascii_lowercase().as_str() {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            _ => Value::String(raw.to_string()),
        },
        // The template serialises objects/arrays with `| tojson`, so they come back as JSON text.
        Some("object") | Some("array") =>
            serde_json::from_str(raw.trim()).unwrap_or_else(|_| Value::String(raw.to_string())),
        _ => Value::String(raw.to_string()),
    }
}

