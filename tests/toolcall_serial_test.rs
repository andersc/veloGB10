//! HOTFIX gate (2026-08-27): malformed tool-call tag + streaming/non-streaming divergence.
//!
//! User report on build 0.4.2 (dev b803b41): the model sometimes emits the function tag with
//! the `<` missing — `<tool_call>\nfunction=get_weather>` (live capture: 2 of 38 sampled tool
//! requests, temp 1.0) — and the engine then diverged by mode: the STREAMING serializer
//! silently dropped the whole block's text (it had been held back and nothing parsed), while
//! the NON-STREAMING serializer leaked the raw block into `content` (recoverable). The fix is
//! ONE shared decision (`tools::finalize_parsed`) for both modes and a parser repair
//! (`tools::find_function_tag`) that reads the bare `function=NAME>` form.
//!
//! WHAT THIS FILE TESTS, AND WHAT IT DOES NOT (S-B9-TOOLPARSE3 T8, 2026-10-10):
//! it tests the DECISION on a finished generation (`tools::parse`, `tools::finalize`,
//! `tools::finalize_parsed`) against the outcome every fixture must reach. It used to also
//! claim "streaming equals non-streaming" for the TEXT the client receives; that half modelled
//! the stream with `tools::held_back_remainder`, which the server stopped calling in
//! S-B9-REL-SURFACE2B R5 (the stream sends every held tail at the finish: `server::held_tail`),
//! so it asserted a hand simulation, not the server. The stream's emission rule lives in a
//! `pub(crate)` generator (`server::chat_sse_stream`) that an integration test cannot reach.
//! The streamed text of THESE SAME FIXTURES is therefore checked where the rule is reachable,
//! by driving the real generator, in the test below (at the end of `src/server.rs`'s
//! `benchfix_sse_tests` module; run it with `cargo test --release --lib the_serial_fixtures_stream`):
//! `server::benchfix_sse_tests::the_serial_fixtures_stream_what_the_non_streaming_response_returns`
//!
//! Expected wire form is the MODEL's own chat_template.jinja (NOT guessed):
//!   <tool_call>\n<function=NAME>\n<parameter=K>\nvalue\n</parameter>\n</function>\n</tool_call>
//! and hy_v3's `<tool_calls:opensource>` / `<tool_call:opensource>NAME<tool_sep:opensource>…`.
//! Run: `cargo test --release --test toolcall_serial_test`.

use gb10_inference::tools;
use serde_json::{json, Value};

/// The EXACT malformed raw text captured live from the model (seed-6 sweep, 2026-08-27):
/// `<tool_call>` intact, `<` of `<function=` missing. This is the case that tripped.
const RAW_MALFORMED_CAPTURED: &str = "<tool_call>\nfunction=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=units>\nc\n</parameter>\n</function>\n</tool_call>";

const RAW_WELLFORMED: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nBerlin\n</parameter>\n</function>\n</tool_call>";

const RAW_MULTI: &str = "I'll check both cities.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n</function>\n</tool_call>";

const RAW_NO_ARGS: &str = "<tool_call>\n<function=list_cities>\n</function>\n</tool_call>";

const RAW_UNICODE: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nZürich 🇨🇭 北çå\n</parameter>\n<parameter=note>\ntempérature — très élevée\n</parameter>\n</function>\n</tool_call>";

const RAW_HY3: &str = "Checking.<tool_calls:opensource>\n<tool_call:opensource>get_weather<tool_sep:opensource>\n<arg_key:opensource>city</arg_key:opensource>\n<arg_value:opensource>\nParis\n</arg_value:opensource>\n</tool_call:opensource>\n</tool_calls:opensource>";

/// A truncated block (no close tag) — parse drops it; BOTH modes must then surface the raw
/// text (the no-silent-drop rule).
const RAW_TRUNCATED: &str = "Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis";

const PLAIN: &str = "The sky is blue because Rayleigh scattering.";

/// One fixture and the outcome the decision must reach on it. `calls` = (name, arguments as
/// JSON), in order; `content` = the `content` field `finalize` returns when calls parsed
/// (None = empty), and when `calls` is empty the content is the raw text itself.
struct Case {
    name: &'static str,
    raw: &'static str,
    calls: Vec<(&'static str, Value)>,
    content: Option<&'static str>,
}

fn cases() -> Vec<Case> {
    vec![
        Case { name: "malformed captured", raw: RAW_MALFORMED_CAPTURED,
               calls: vec![("get_weather", json!({"city": "Paris", "units": "c"}))], content: None },
        Case { name: "well-formed", raw: RAW_WELLFORMED,
               calls: vec![("get_weather", json!({"city": "Berlin"}))], content: None },
        Case { name: "two calls with prose before", raw: RAW_MULTI,
               calls: vec![("get_weather", json!({"city": "Paris"})), ("get_weather", json!({"city": "Tokyo"}))],
               content: Some("I'll check both cities.") },
        Case { name: "no arguments", raw: RAW_NO_ARGS, calls: vec![("list_cities", json!({}))], content: None },
        Case { name: "unicode", raw: RAW_UNICODE,
               calls: vec![("get_weather", json!({"city": "Zürich 🇨🇭 北çå", "note": "température — très élevée"}))],
               content: None },
        Case { name: "hy_v3 opensource", raw: RAW_HY3,
               calls: vec![("get_weather", json!({"city": "Paris"}))], content: Some("Checking.") },
        Case { name: "truncated block", raw: RAW_TRUNCATED, calls: vec![], content: Some(RAW_TRUNCATED) },
        Case { name: "plain prose", raw: PLAIN, calls: vec![], content: Some(PLAIN) },
    ]
}

/// The request's `tools` array: none, so coercion stays string-typed (the request carries the
/// schema, not disk; this file used to read a tools fixture from an environment-named
/// directory and throw it away — removed: no environment variable configures anything).
fn no_schema() -> Option<Vec<Value>> {
    None
}

/// Strict wire well-formedness: the emitted call must round-trip as the OpenAI tool_calls
/// shape AND its `arguments` must be a JSON object (the harness feeds it to a real function).
fn assert_wire_wellformed(calls: &[gb10_inference::tokenizer::ToolCall]) {
    for c in calls {
        assert_eq!(c.kind, "function", "type must be \"function\" on the wire");
        assert!(!c.id.is_empty(), "call id must be non-empty");
        assert!(!c.function.name.is_empty(), "function name must be non-empty");
        assert!(
            !c.function.name.contains('<') && !c.function.name.contains('>') && !c.function.name.contains('\n'),
            "name must be a bare identifier, not tag fragments: {:?}",
            c.function.name
        );
        let args: Value = serde_json::from_str(&c.function.arguments)
            .unwrap_or_else(|e| panic!("arguments must be a JSON object, got {:?}: {}", c.function.arguments, e));
        assert!(args.is_object(), "arguments must deserialize to a JSON object");
    }
}

/// The finish-time decision of BOTH response modes on one fixture, against the stated outcome.
///  - non-streaming: `tools::finalize(raw)`;
///  - streaming: `chat_sse_stream` runs `tools::parse` on the post-think answer text and then
///    `tools::finalize_parsed` (src/server.rs, after the token loop) — the same two calls.
/// Every comparison is against the fixture's EXPECTED calls/content (a success signal: a
/// fixture that must yield two calls fails when it yields none), never only against the other
/// mode (two empty results would agree). Both finish reasons a generation can end with are run.
fn assert_decision(case: &Case) {
    let schema = no_schema();
    for finish_in in ["stop", "length"] {
        let (content, calls, finish) = tools::finalize(case.raw, schema.as_deref(), finish_in);
        let parsed = tools::parse(case.raw, schema.as_deref());
        let (s_content, s_calls, s_finish) = tools::finalize_parsed(case.raw, parsed, finish_in);

        // 1. The expected calls came out, with their arguments (not just a count).
        assert_eq!(calls.len(), case.calls.len(), "{}: call count (finish_in {finish_in})", case.name);
        for (got, (name, args)) in calls.iter().zip(&case.calls) {
            assert_eq!(got.function.name, *name, "{}: call name", case.name);
            let a: Value = serde_json::from_str(&got.function.arguments).unwrap();
            assert_eq!(a, *args, "{}: call arguments", case.name);
        }
        // 2. The text the client gets back, and the finish_reason.
        if case.calls.is_empty() {
            assert_eq!(content.as_deref(), Some(case.raw), "{}: nothing parsed: the raw text, unchanged", case.name);
            assert_eq!(finish, finish_in, "{}: finish passes through when nothing parsed", case.name);
        } else {
            assert_eq!(content.as_deref(), case.content, "{}: prose outside the calls", case.name);
            assert_eq!(finish, "tool_calls", "{}: the flag every harness branches on", case.name);
        }
        // 3. The streaming mode's finish decision is the same decision.
        assert_eq!(s_calls.len(), calls.len(), "{}: stream call count", case.name);
        for (s, j) in s_calls.iter().zip(&calls) {
            assert_eq!(s.function.name, j.function.name, "{}: stream call name", case.name);
            assert_eq!(s.function.arguments, j.function.arguments, "{}: stream call arguments", case.name);
        }
        assert_eq!(s_finish, finish, "{}: stream finish_reason", case.name);
        assert_eq!(s_content, content, "{}: stream content", case.name);
        // 4. Wire tag well-formed on whatever calls were emitted.
        assert_wire_wellformed(&calls);
    }
}

#[test]
fn every_fixture_reaches_its_stated_outcome_in_both_modes() {
    let all = cases();
    // the fixture set itself must carry both kinds of outcome, or the loop proves nothing
    assert!(all.iter().filter(|c| !c.calls.is_empty()).count() >= 6, "the call fixtures");
    assert!(all.iter().filter(|c| c.calls.is_empty()).count() >= 2, "the no-call fixtures");
    for c in &all {
        assert_decision(c);
    }
}

#[test]
fn failing_case_malformed_tag_parses_and_both_modes_agree() {
    // BEFORE the fix: parse failed -> streaming DROPPED the block (no calls, no content),
    // non-streaming LEAKED the raw block into content. Mode agreement was impossible.
    let parsed = tools::parse(RAW_MALFORMED_CAPTURED, no_schema().as_deref());
    assert_eq!(parsed.tool_calls.len(), 1, "the malformed (bare function=) form must parse");
    assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
    let args: Value = serde_json::from_str(&parsed.tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Paris");
    assert_eq!(args["units"], "c");
    assert_decision(&cases()[0]);
}

#[test]
fn known_passing_wellformed_call() {
    let parsed = tools::parse(RAW_WELLFORMED, no_schema().as_deref());
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
    let args: Value = serde_json::from_str(&parsed.tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Berlin");
    assert_decision(&cases()[1]);
}

#[test]
fn multi_tool_call() {
    let parsed = tools::parse(RAW_MULTI, no_schema().as_deref());
    assert_eq!(parsed.tool_calls.len(), 2, "both calls must parse");
    assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
    assert_eq!(parsed.tool_calls[1].function.name, "get_weather");
    let a0: Value = serde_json::from_str(&parsed.tool_calls[0].function.arguments).unwrap();
    let a1: Value = serde_json::from_str(&parsed.tool_calls[1].function.arguments).unwrap();
    assert_eq!(a0["city"], "Paris");
    assert_eq!(a1["city"], "Tokyo");
    assert_eq!(parsed.content, "I'll check both cities.");
    assert_decision(&cases()[2]);
}

#[test]
fn call_with_no_arguments() {
    let parsed = tools::parse(RAW_NO_ARGS, no_schema().as_deref());
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "list_cities");
    let args: Value = serde_json::from_str(&parsed.tool_calls[0].function.arguments).unwrap();
    assert_eq!(args.as_object().map(|m| m.len()), Some(0), "no-arg call must be the empty object");
    assert_decision(&cases()[3]);
}

#[test]
fn unicode_in_arguments() {
    let parsed = tools::parse(RAW_UNICODE, no_schema().as_deref());
    assert_eq!(parsed.tool_calls.len(), 1);
    let args: Value = serde_json::from_str(&parsed.tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Zürich 🇨🇭 北çå");
    assert_eq!(args["note"], "température — très élevée");
    assert_decision(&cases()[4]);
}

#[test]
fn hy_v3_opensource_form() {
    let parsed = tools::parse(RAW_HY3, no_schema().as_deref());
    assert_eq!(parsed.tool_calls.len(), 1, "hy_v3 :opensource form must parse");
    assert_eq!(parsed.tool_calls[0].function.name, "get_weather");
    let args: Value = serde_json::from_str(&parsed.tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Paris");
    assert_eq!(parsed.content, "Checking.");
    assert_decision(&cases()[5]);
}

#[test]
fn truncated_block_never_silently_dropped() {
    // No close tag -> the call is dropped by the parser (half a call is worse than none), but
    // BOTH modes must surface the raw text: non-streaming returns it as content; the stream
    // sends it as the held tail at the finish (`server::held_tail`; the text it sends is
    // asserted on the real generator in `src/server.rs`, not modelled here).
    let (content, calls, finish) = tools::finalize(RAW_TRUNCATED, no_schema().as_deref(), "stop");
    assert!(calls.is_empty(), "truncated call must not parse");
    assert_eq!(finish, "stop", "finish passes through when nothing parsed");
    let content = content.expect("raw text must be surfaced, not dropped");
    assert!(content.contains("<tool_call"), "the held-back block text must be recoverable");
    assert_eq!(content, RAW_TRUNCATED, "and it is the generated text, byte for byte");
    assert_decision(&cases()[6]);
}

#[test]
fn malformed_repair_prefers_wellformed_and_rejects_value_text() {
    // The well-formed tag always wins when present ...
    let both = "<tool_call>\n<function=real_fn>\n<parameter=q>\nfunction=decoy>\n</parameter>\n</function>\n</tool_call>";
    let parsed = tools::parse(both, no_schema().as_deref());
    assert_eq!(parsed.tool_calls.len(), 1);
    assert_eq!(parsed.tool_calls[0].function.name, "real_fn");
    let args: Value = serde_json::from_str(&parsed.tool_calls[0].function.arguments).unwrap();
    assert_eq!(args["q"], "function=decoy>", "the in-value occurrence must stay payload");
    // ... and the bare-form repair only fires at a line boundary, never mid-word.
    let midword = "<tool_call>\nsomefunction=x>\n</tool_call>";
    assert!(tools::parse(midword, no_schema().as_deref()).tool_calls.is_empty());
}

#[test]
fn no_drop_when_no_tools_at_all() {
    // Ordinary prose (no markers): finalize passes the text through untouched — the
    // serialization change must not alter plain text.
    let (content, calls, finish) = tools::finalize(PLAIN, no_schema().as_deref(), "stop");
    assert_eq!(calls.len(), 0);
    assert_eq!(content.as_deref(), Some(PLAIN));
    assert_eq!(finish, "stop");
    assert_decision(&cases()[7]);
}
