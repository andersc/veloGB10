//! The work bound of the cost model, on the shapes the independent review (2026-10-10, B1) measured.
//!
//! Part of the lib test suite: the first test asserts COUNTED work (work units and raw operations,
//! never a clock); the second prints units against the clock side by side and is run by hand
//! (`cargo test --release --lib work_units_table -- --ignored --nocapture`).

use serde_json::{json, Value};

use super::cost;
use super::{parse_rules, BlockEnd, QuotePairing, Rules};

/// The request catalogue of `n` (>= 4) tools: generated tools first, the four tools the shapes call
/// LAST (the order in which a linear scan of the list is dearest), the largest of them
/// `create_ticket` with 200 declared parameters.
pub(in crate::tools) fn catalogue(n: usize) -> Vec<Value> {
    let tool = |name: &str, props: &[String], req: &[&str]| json!({"type": "function", "function": {
        "name": name, "description": "a dummy tool of the bound tests",
        "parameters": {"type": "object",
            "properties": props.iter().map(|p| (p.clone(), json!({"type": "string"}))).collect::<serde_json::Map<_, _>>(),
            "required": req}}});
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let ticket: Vec<String> = (0..200).map(|i| format!("field_{i}")).collect();
    let tail = vec![
        tool("read_file", &s(&["path"]), &["path"]),
        tool("run_shell", &s(&["command"]), &["command"]),
        tool("write_file", &s(&["path", "content"]), &["path", "content"]),
        tool("create_ticket", &ticket, &["field_0", "field_1"]),
    ];
    let mut v: Vec<Value> = (0..n.saturating_sub(tail.len())).map(|i| {
        let props: Vec<String> = (0..3 + i % 4).map(|k| format!("arg_{k}")).collect();
        tool(&format!("tool_{i}"), &props, &["arg_0"])
    }).collect();
    v.extend(tail);
    v
}

fn block(content: &str) -> String {
    format!("<tool_call>\n<function=write_file>\n<parameter=path>\ndocs/x.md\n</parameter>\n<parameter=content>\n{content}\n</parameter>\n</function>\n</tool_call>")
}

const Q2: &str = "Example:\n\n<function=write_file>\n<parameter=path>\nnotes/todo.txt\n</parameter>\n<parameter=content>\nbuy milk\n</parameter>\n</function>\n\n";
const Q1: &str = "Then run:\n\n<function=run_shell>\n<parameter=command>\nls -la\n</parameter>\n</function>\n\n";
const QP: &str = "{%- if a %}\n<parameter={{ k }}>\n{{ v }}\n</parameter>\n{%- endif %}\n";
const QB: &str = "Example:\n\n<tool_call>\n<function=read_file>\n<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>\n\n";

/// What a shape is for: `Typical` shapes must be read by the cost model (not declined) at every
/// catalogue size; `Heavy` ones may be declined, and what is checked is that the work stays bounded.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::tools) enum Class { Typical, Heavy }

/// The shapes of the review's two benchmarks, named as it named them, and the degenerate ones.
pub(in crate::tools) fn shapes() -> Vec<(String, String, Class)> {
    use Class::*;
    let fill = |line: &str, bytes: usize| { let mut v = String::new(); while v.len() < bytes { v.push_str(line); } v };
    let mut v: Vec<(String, String, Class)> = Vec::new();
    v.push(("ordinary write_file, 2 KB".into(), block(&fill("a line of ordinary file content, 0123456789\n", 2_000)), Typical));
    v.push(("ordinary write_file, 60 KB".into(), block(&fill("a line of ordinary file content, 0123456789\n", 60_000)), Typical));
    for k in [1usize, 5, 10, 20] {
        v.push((format!("file quoting {k} two-parameter calls"), block(&Q2.repeat(k)), Typical));
    }
    for k in [40usize, 60, 100, 200] {
        v.push((format!("file quoting {k} two-parameter calls"), block(&Q2.repeat(k)), Heavy));
    }
    for k in [3usize, 10, 20] {
        v.push((format!("file quoting {k} one-parameter calls with prose"), block(&Q1.repeat(k)), Typical));
        v.push((format!("template with {k} quoted parameter pairs"), block(&QP.repeat(k)), Typical));
        v.push((format!("file quoting {k} complete blocks"), block(&QB.repeat(k)), Typical));
    }
    for k in [60usize, 200] {
        v.push((format!("file quoting {k} one-parameter calls with prose"), block(&Q1.repeat(k)), Heavy));
        v.push((format!("template with {k} quoted parameter pairs"), block(&QP.repeat(k)), Heavy));
        v.push((format!("file quoting {k} complete blocks"), block(&QB.repeat(k)), Heavy));
    }
    for p in [4usize, 8, 12, 16, 24, 32] {
        let mut b = String::from("<tool_call>\n<function=create_ticket>\n");
        for i in 0..p { b.push_str(&format!("<parameter=field_{i}>\nvalue {i}\n</parameter>\n")); }
        b.push_str("</function>\n</tool_call>");
        v.push((format!("create_ticket with {p} declared parameters, plain values"), b, Typical));
    }
    for p in [64usize, 200] {
        let mut b = String::from("<tool_call>\n<function=create_ticket>\n");
        for i in 0..p { b.push_str(&format!("<parameter=field_{i}>\nvalue {i}\n</parameter>\n")); }
        b.push_str("</function>\n</tool_call>");
        v.push((format!("create_ticket with {p} parameters, plain values"), b, Heavy));
    }
    // the degenerate responses of the review
    let b40 = block(&Q2.repeat(40));
    v.push(("4 blocks, each quoting 40 calls".into(), format!("{b40}\n{b40}\n{b40}\n{b40}"), Heavy));
    let b60 = block(&Q2.repeat(60));
    v.push(("24 blocks of 8 KB, each quoting 60 calls".into(), (0..24).map(|_| b60.clone()).collect::<Vec<_>>().join("\n"), Heavy));
    // a response at the caps of the shadow: 64 blocks, under 512 KiB
    let k = (super::SHADOW_MAX_BYTES / 64 - 256) / Q2.len();
    let one = block(&Q2.repeat(k));
    let at_caps = (0..64).map(|_| one.clone()).collect::<Vec<_>>().join("\n");
    assert!(at_caps.len() <= super::SHADOW_MAX_BYTES, "{}", at_caps.len());
    v.push(("at the caps: 64 blocks, under 512 KiB".into(), at_caps, Heavy));
    v.push(("64 blocks, one call of 20 parameters each".into(), (0..64).map(|_| {
        let mut b = String::from("<tool_call>\n<function=create_ticket>\n");
        for i in 0..20 { b.push_str(&format!("<parameter=field_{i}>\nvalue {i}\n</parameter>\n")); }
        b.push_str("</function>\n</tool_call>");
        b
    }).collect::<Vec<_>>().join("\n"), Heavy));
    v
}

/// One response through the cost rule as the SERVED rule: (units spent, raw operations, blocks declined).
pub(in crate::tools) fn run_cost(text: &str, tools: &[Value]) -> (usize, usize, usize, usize) {
    cost::OPS.with(|c| c.set(0));
    super::COST_USED.with(|c| c.set((0, 0)));
    let rules = Rules { pairing: QuotePairing::Cost, block: BlockEnd::First, ..Rules::V073 };
    let out = parse_rules(text, Some(tools), &rules, Some("stop"));
    let (units, declined) = super::COST_USED.with(|c| c.get());
    (units, cost::OPS.with(|c| c.get()), declined, out.tool_calls.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE BOUND, as counted work. For every shape of the review and catalogues of 4, 30 and 100
    /// tools: the cost model, as the served rule, spends at most `RESPONSE_UNITS` work units (plus
    /// the one charge that crosses the line, never more than the text's own walk); the raw
    /// operations it performs are covered by what it charged (so a loop that is added without a
    /// charge fails here); the shapes that are typical are READ by the model, not declined, whatever
    /// the catalogue; and the catalogue size does not change the count at all (it is a lookup).
    #[test]
    fn the_cost_model_work_is_bounded_on_the_review_shapes() {
        let mut typical_declined = Vec::new();
        for (name, text, class) in shapes() {
            let mut first: Option<(usize, usize)> = None;
            for n in [4usize, 30, 100] {
                let cat = catalogue(n);
                assert_eq!(cat.len(), n);
                let (units, ops, declined, calls) = run_cost(&text, &cat);
                let slack = text.len() / 8 + 2_000;
                assert!(units <= cost::RESPONSE_UNITS + slack, "{name} / {n} tools: {units} units");
                assert!(ops <= 32 * units + 2_000, "{name} / {n} tools: {ops} operations against {units} units charged");
                if class == Class::Typical && declined > 0 { typical_declined.push(format!("{name} / {n} tools")); }
                // the catalogue only changes what the model SAYS about calls to tools that are not
                // offered (none here): units and calls must not depend on its size
                match first {
                    None => first = Some((units, calls)),
                    Some((u0, c0)) => {
                        assert_eq!(calls, c0, "{name}: {n} tools read differently from 4");
                        assert!(units.abs_diff(u0) <= u0 / 20 + 64, "{name}: {units} units at {n} tools against {u0} at 4");
                    }
                }
            }
        }
        assert!(typical_declined.is_empty(), "typical shapes declined: {typical_declined:?}");
    }

    /// The same bound through the shadow: the comparison of a response that holds the heavy shapes,
    /// with the cost model switched on, spends the same budget and counts the blocks it skipped.
    #[test]
    fn a_shadowed_cost_model_declines_and_counts_instead_of_running_on() {
        let cat = catalogue(100);
        let (_, text, _) = shapes().into_iter().find(|s| s.0.starts_with("24 blocks")).unwrap();
        let sh = super::super::shadow_compare(&text, Some(&cat), &Rules::V073, Some("stop"));
        assert_eq!(sh.blocks, 24);
        assert!(sh.cost_declined > 0, "the response spends more than one response budget: {sh:?}");
        assert_eq!(sh.compared[super::super::Other::Cost.idx()] + sh.cost_declined, 24, "every block is compared or declined");
    }

    /// The server's entry as `parse_served` runs it, minus the metrics and the option read: the served
    /// parse with the block records, then the shadow comparison (the cheap rules; the cost model too
    /// when `with_cost`).
    fn served_with_shadow(text: &str, tools: &[Value], with_cost: bool) -> usize {
        let rules = Rules::V073;
        let mut recs = super::super::Recs::new(true);
        let out = super::super::parse_core(text, Some(tools), &rules, Some("stop"), Some(&mut recs));
        let sh = super::super::shadow_from(text, Some(tools), &rules, Some("stop"), &out, &recs.v, with_cost);
        out.tool_calls.len() + sh.diffs.len()
    }

    /// Units against the clock, the shapes against the v0.7.3 default, and the shadow (run by hand,
    /// release): `cargo test --release --lib work_units_table -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn work_units_table() {
        use std::time::Instant;
        let med_max = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); (v[v.len() / 2], v[v.len() - 1]) };
        println!("{:<58} {:>8} {:>5} | {:>8} {:>8} | {:>8} {:>8} | {:>8} {:>8} | {:>8} {:>8} | {:>8} {:>9} {:>4} {:>7}",
                 "shape", "bytes", "tools", "v073 med", "v073 max", "dflt med", "dflt max", "+cost md", "+cost mx", "cost med", "cost max",
                 "units", "ops", "decl", "ns/unit");
        for (name, text, _) in shapes() {
            for n in [4usize, 30, 100] {
                let cat = catalogue(n);
                let (mut a, mut d, mut dc, mut b) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                let mut stats = (0, 0, 0, 0);
                for _ in 0..9 {
                    let t = Instant::now();
                    std::hint::black_box(parse_rules(&text, Some(&cat), &Rules::V073, Some("stop")));
                    a.push(t.elapsed().as_secs_f64() * 1e3);
                    let t = Instant::now();
                    std::hint::black_box(served_with_shadow(&text, &cat, false));
                    d.push(t.elapsed().as_secs_f64() * 1e3);
                    let t = Instant::now();
                    std::hint::black_box(served_with_shadow(&text, &cat, true));
                    dc.push(t.elapsed().as_secs_f64() * 1e3);
                    let t = Instant::now();
                    stats = std::hint::black_box(run_cost(&text, &cat));
                    b.push(t.elapsed().as_secs_f64() * 1e3);
                }
                let (am, ax) = med_max(a);
                let (dm, dx) = med_max(d);
                let (cm, cx) = med_max(dc);
                let (bm, bx) = med_max(b);
                let (units, ops, declined, _) = stats;
                println!("{:<58} {:>8} {:>5} | {:>8.3} {:>8.3} | {:>8.3} {:>8.3} | {:>8.3} {:>8.3} | {:>8.3} {:>8.3} | {:>8} {:>9} {:>4} {:>7.1}",
                         name, text.len(), n, am, ax, dm, dx, cm, cx, bm, bx, units, ops, declined,
                         if units > 0 { bm * 1e6 / units as f64 } else { 0.0 });
            }
        }
    }
}
