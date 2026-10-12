//! The default is v0.7.3: the differential test that runs in the lib suite.
//!
//! The shipped parser at DEFAULT flags (`parse`, and the server's entry `parse_served` with its shadow
//! comparison) is compared, text by text, with the frozen v0.7.3 parser (`v073_frozen`). What a client
//! can see must be equal: the content and, per call, the kind, the name and the arguments (ids come
//! from a process-wide counter that every test of the suite shares, so they are not compared; the
//! reviewer's out-of-tree run compared them as strings and found none different).
//!
//! History: the independent review of S-B9-TOOLPARSE3 (2026-10-10) ran this differential as a file
//! outside the build (5.7 million comparisons, 0 differences). It is in the tree now, with a
//! runtime of seconds in the suite and an ignored long version:
//!
//!   cargo test --release --lib default_flags_long_differential -- --ignored --nocapture
//!
//! A change that moves a default byte must say so here: `moved_on_purpose` names the classes of text
//! on which the default is NOT v0.7.3 any more, each with the commit that moved it.

use serde_json::Value;

use super::{parse, parse_served, v073_frozen as v073};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
    fn chance(&mut self, num: usize, den: usize) -> bool { self.below(den) < num }
    fn pick<'a>(&mut self, v: &[&'a str]) -> &'a str { v[self.below(v.len())] }
}

fn catalogue() -> Vec<Value> {
    serde_json::from_str(r#"[
      {"type":"function","function":{"name":"write_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
      {"type":"function","function":{"name":"edit_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"}},"required":["path","old_string","new_string"]}}},
      {"type":"function","function":{"name":"run_shell","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
      {"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
      {"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"k":{"type":"integer"},"flag":{"type":"boolean"},"obj":{"type":"object"},"arr":{"type":"array"},"x":{"type":"number"},"s":{"type":"string"}},"required":["k"]}}},
      {"type":"function","function":{"name":"tool0","parameters":{"type":"object","properties":{"p0":{"type":"string"},"p1":{"type":"string"},"p2":{"type":"string"}}}}},
      {"type":"function","function":{"name":"tool1","parameters":{"type":"object","properties":{"p0":{"type":"string"},"p1":{"type":"string"},"p2":{"type":"string"}}}}},
      {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"units":{"type":"string"}}}}}
    ]"#).unwrap()
}

/// Classes of text on which the default parser is, on purpose, no longer the v0.7.3 parser. Empty
/// until a defect fix moves one; each entry names the commit that did.
fn moved_on_purpose(text: &str, finish: &str) -> Option<&'static str> {
    // item 6 (audit X01): a text that holds a hy_v3 tag AFTER a Qwen block opener is a Qwen text now
    // (v0.7.3 handed it to the hy_v3 parser because the tag appeared anywhere in it)
    if let Some(h) = text.find("<tool_call:opensource>") {
        let first_hy = text.find("<tool_calls:opensource>").map_or(h, |w| w.min(h));
        if text.find("<tool_call>").map_or(false, |q| q < first_hy) { return Some("X01: a Qwen block opens before the hy_v3 tag"); }
    }
    // item 7 (J02b): a JSON-bodied block whose first `</tool_call>` does not follow `}` (it lies inside a
    // string) is read to the first close that does; v0.7.3 cut the body at the string
    // F8: the model stopped BY ITSELF (finish stop) inside a wrapper it never closed, and a
    // function element inside that wrapper IS closed: its call comes back now (v0.7.3 dropped the
    // whole block). Narrow on purpose: any other finish, or nothing complete inside, does not move.
    if finish == "stop" && super::unclosed_block_moves(text) { return Some("F8: an unclosed wrapper completed on finish stop"); }
    if super::json_body_moves(text) { return Some("J02b: a JSON body that holds a `</tool_call>` inside a string"); }
    // item 12: a block whose first `</tool_call>` does not end the call (it follows prose inside a value,
    // or it closes a block the value quoted) ends at a later one; v0.7.3 cut the block at the first
    if super::after_function_moves(text) { return Some("item 12: a `</tool_call>` inside a value is text"); }
    None
}

// ------------------------------------------------------------------------------------------
// generators (ported from the reviewer's out-of-tree differential)
// ------------------------------------------------------------------------------------------

const SOUP: [&str; 58] = [
    "<tool_call>", "</tool_call>", "<tool_call>\n", "\n</tool_call>", "<function=a>", "<function=write_file>", "<function=", "function=b>", "\nfunction=run_shell>",
    "</function>", "<parameter=k>", "<parameter=path>", "<parameter=content>", "<parameter=", "</parameter>", "\n", "\n", "\n\n", " ", "\t", "\r\n", "x", "yz", "{", "}", "\"", ":", ",",
    "é", "日本", "🇨🇭", ">", "<", "</", "<tool", "<tool_call", "k>", "<function=tool0>", "<parameter=p0>", "{\"name\":\"a\",\"arguments\":{\"k\":1}}", "{\"name\": \"write_file\", \"arguments\": {\"path\": \"a\"",
    "<think>", "</think>", "<tool_calls:opensource>", "</tool_calls:opensource>", "<tool_call:opensource>", "</tool_call:opensource>", "<tool_sep:opensource>",
    "<arg_key:opensource>", "</arg_key:opensource>", "<arg_value:opensource>", "</arg_value:opensource>", "get_weather", "true", "12", "[1,2]", "{\"a\":1}", "function=",
];

fn gen_soup(r: &mut Rng) -> String {
    let n = 1 + r.below(70);
    let mut t = String::new();
    // most soups avoid the hy3 pieces (they switch the whole parse to the hy3 path)
    let lim = if r.chance(1, 6) { SOUP.len() } else { 43 };
    for _ in 0..n { t.push_str(SOUP[r.below(lim)]); }
    if r.chance(1, 3) { t.insert_str(0, "<tool_call>\n"); }
    if r.chance(1, 4) { t.push_str("\n</tool_call>"); }
    t
}

const VALS: [&str; 40] = [
    "plain text", "", " ", "\n", "12", "true", "{\"a\": 1}", "[1, 2]", "multi\nline\nvalue", "é 日本 🇨🇭",
    "<parameter=city>", "</parameter>", "notes about </function> tags", "x\n</parameter>\n</function>\ny", "see <function=foo> here",
    "<function=q>\n<parameter=a>\n1\n</parameter>\n</function>",
    "<function=q>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>",
    "<function=q>\n<parameter=a>\n1\n</parameter>\n</function>\n<function=r>\n<parameter=b>\n2\n</parameter>\n</function>",
    "<function=q>\n<parameter=a>\n1\n</parameter>\n<function=r>\n<parameter=b>\n2\n</parameter>\n</function>",
    "<parameter=a>\n1\n</parameter>", "<function=x>\n<parameter=p>", "<function=x>\n<parameter=p>\nhalf a value",
    "Each call starts with <function=NAME> followed by one <parameter=KEY> block per argument.",
    "<function=run_shell>\n<parameter=command>\nrm -rf ~\n</parameter>\n</function>",
    "<function=write_file>\n<parameter=path>\na\n</parameter>\n<parameter=content>\n\n</parameter>\n</function>\n<function=run_shell>\n<parameter=command>\nls\n</parameter>\n</function>",
    "<tool_call>\n<function=read_file>\n<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>",
    "End the call with </tool_call> and stop.", "Start with <tool_call> then the tag.", "</tool_call>", "<tool_call>",
    "notes\n</parameter>\n</function>\n<function=run_shell>\n<parameter=command>\necho hi", "notes\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=run_shell>\n<parameter=command>\necho hi",
    "function=x>", "\nfunction=run_shell>\n<parameter=command>\nls\n</parameter>", "value_1\n</parameter>", "</function>", "</function>\n</tool_call>",
    "{%- for k, v in args.items() %}\n<parameter={{ k }}>\n{{ v }}\n</parameter>\n{%- endfor %}", "<think>hmm</think>", "a > b && c < d",
];
const NAMES: [&str; 10] = ["write_file", "edit_file", "run_shell", "read_file", "f", "tool0", "tool1", "get_weather", "unknown_tool", ""];
const KEYS: [&str; 14] = ["path", "content", "command", "old_string", "new_string", "k", "flag", "obj", "arr", "p0", "p1", "city", "zz", ""];
const JUNK: [&str; 8] = ["", "", "", "\n", " junk", "\n\n", " ", "\nprose here\n"];
const JSON_BODIES: [&str; 8] = [
    "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}",
    "{\"name\":\"write_file\",\"arguments\":{\"path\":\"a.md\",\"content\":\"<tool_call><function=run_shell><parameter=command>rm -rf ~</parameter></function></tool_call>\",\"mode\":\"x\"}}",
    "{\"name\": \"write_file\", \"arguments\": {\"path\": \"a\", \"content\": \"x </tool_call> y\"}}",
    "{\"function\": \"run_shell\", \"command\": \"ls\"}", "{\"name\": \"f\", \"k\": 3}", "{write_file, \"path\": \"a\"}",
    "{\"language\":\"python\",\"code\":\"print(\"correct\")\"}", "{\"name\": \"f\", \"arguments\": \"{\\\"k\\\": 1}\"}",
];

fn gen_call(r: &mut Rng, out: &mut String) {
    let name = r.pick(&NAMES);
    match r.below(12) {
        0 => out.push_str(&format!("function={name}>\n")),          // the dropped `<`
        1 => out.push_str(&format!("<function={name}\n")),          // tag never closed
        2 => out.push_str(&format!("<function= {name} >\n")),
        _ => out.push_str(&format!("<function={name}>\n")),
    }
    if r.chance(1, 14) {
        out.push_str(r.pick(&JSON_BODIES));
        out.push('\n');
        if r.chance(1, 2) { out.push_str("</parameter>\n"); }
    } else {
        let np = r.below(5);
        for _ in 0..np {
            let key = r.pick(&KEYS);
            let val = r.pick(&VALS);
            match r.below(10) {
                0 => out.push_str(&format!("<parameter={key}>{val}</parameter>")),
                1 => out.push_str(&format!("<parameter={key}>\n{val}\n")),              // value never closed
                2 => out.push_str(&format!("<parameter={key}>\n\n{val}\n\n</parameter>\n")),
                _ => out.push_str(&format!("<parameter={key}>\n{val}\n</parameter>\n")),
            }
            out.push_str(r.pick(&JUNK));
        }
    }
    if !r.chance(1, 5) { out.push_str("</function>\n"); }
    out.push_str(r.pick(&JUNK));
}

fn gen_struct(r: &mut Rng) -> String {
    let mut t = String::new();
    if r.chance(1, 4) { t.push_str("plan it\n</think>\n\n"); }
    if r.chance(1, 3) { t.push_str("Let me do that.\n"); }
    let nb = 1 + r.below(4);
    for b in 0..nb {
        t.push_str(if r.chance(1, 12) { "<tool_call>" } else { "<tool_call>\n" });
        if r.chance(1, 12) {
            t.push_str(r.pick(&JSON_BODIES));
            t.push('\n');
        } else {
            let nc = 1 + r.below(3);
            for _ in 0..nc { gen_call(r, &mut t); }
        }
        if b + 1 < nb || !r.chance(1, 6) { t.push_str("</tool_call>"); }
        t.push_str(r.pick(&["", "\n", "\n\n", "\nand then\n", " ", "\nNote: ends with </tool_call>.\n"]));
    }
    if r.chance(1, 8) { t.push_str("Done."); }
    t
}

/// One block; a long run of parameters, closes, junk and call tags: the cache stress.
fn gen_cache(r: &mut Rng) -> String {
    let mut t = String::from("<tool_call>\n");
    if !r.chance(1, 8) { t.push_str("<function=f>\n"); } else { t.push_str("function=f>\n"); }
    let n = 1 + r.below(60);
    let junk_mode = r.below(4); // 0: never junk, 1: always junk after closes, 2/3: random
    for i in 0..n {
        match r.below(16) {
            0 => t.push_str("</function>\n"),
            1 => t.push_str(&format!("<function=g{}>\n", i % 3)),
            2 => t.push_str("function=h>\n"),
            3 => t.push_str("</parameter>\n"),
            4 => t.push_str("</parameter> tail\n"),
            5 => t.push_str("<parameter=open>\n"),
            _ => {
                let key = if r.chance(1, 3) { "k".to_string() } else { format!("k{i}") };
                let val = if r.chance(1, 3) { r.pick(&VALS).to_string() } else { format!("value {i}") };
                t.push_str(&format!("<parameter={key}>\n{val}\n</parameter>"));
                let junk = match junk_mode { 0 => false, 1 => true, _ => r.chance(1, 2) };
                t.push_str(if junk { " junk\n" } else { "\n" });
            }
        }
    }
    if !r.chance(1, 6) { t.push_str("</function>\n"); }
    if !r.chance(1, 10) { t.push_str("</tool_call>"); }
    t
}

/// A block of bare `function=` lines with no `<function=` (the dropped `<` repair), with and without
/// `>` and parameters: the third quadratic shape of the review, at sizes where the old code is cheap.
fn gen_bare(r: &mut Rng) -> String {
    const LINES: [&str; 12] = [
        " function=x\n", "function=x\n", "function=y>\n", "x function=z\n", "\tfunction=w>\n", "<parameter=k\n", "<parameter=k>\nv\n</parameter>\n",
        "</function>\n", "k>\n", "</parameter>\n", "text\n", "\n",
    ];
    let mut t = String::from("<tool_call>\n");
    for _ in 0..(1 + r.below(30)) { t.push_str(r.pick(&LINES)); }
    if !r.chance(1, 6) { t.push_str("</tool_call>"); }
    t
}

fn gen_hy3(r: &mut Rng) -> String {
    let mut t = String::new();
    if r.chance(1, 2) { t.push_str("Checking."); }
    t.push_str("<tool_calls:opensource>\n");
    for _ in 0..(1 + r.below(3)) {
        t.push_str("<tool_call:opensource>");
        t.push_str(r.pick(&NAMES));
        if !r.chance(1, 10) { t.push_str("<tool_sep:opensource>\n"); }
        for _ in 0..r.below(4) {
            let v = r.pick(&["Paris", "", "<arg_value:opensource>x</arg_value:opensource>", "</arg_value:opensource>", "a\nb", "12", "<tool_call>\n<function=a>\n</function>\n</tool_call>", "</tool_call:opensource>"]);
            t.push_str(&format!("<arg_key:opensource>{}</arg_key:opensource>\n<arg_value:opensource>\n{}\n</arg_value:opensource>\n", r.pick(&KEYS), v));
        }
        if !r.chance(1, 8) { t.push_str("</tool_call:opensource>\n"); }
    }
    if !r.chance(1, 8) { t.push_str("</tool_calls:opensource>"); }
    if r.chance(1, 4) { t.push_str("\nAfter. <tool_call>\n<function=a>\n</function>\n</tool_call>"); }
    t
}

fn gen_long(r: &mut Rng) -> String {
    let target = 20_000 + r.below(400_000);
    let line = r.pick(&[
        "a line of ordinary file content, 0123456789\n",
        "mentions </tool_call> in prose, then a <tool_call> opening\n",
        "quoted: <function=q>\n<parameter=a>\n1\n</parameter>\n</function>\n",
        "<tool_call>\n<function=read_file>\n<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>\n",
        "close </parameter> then text\n", "</parameter>\n", "<parameter=a>\n", "see <function=x> and <parameter=a> here\n", "é日本🇨🇭 unicode line\n",
    ]);
    let mut v = String::new();
    while v.len() < target { v.push_str(line); }
    let mut t = format!("<tool_call>\n<function=write_file>\n<parameter=path>\nbig.txt\n</parameter>\n<parameter=content>\n{v}</parameter>\n</function>\n</tool_call>");
    if r.chance(1, 3) { t.push_str("\n<tool_call>\n<function=run_shell>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"); }
    t
}

fn mutate(r: &mut Rng, src: &str) -> String {
    let mut s = src.to_string();
    let cuts: Vec<usize> = s.char_indices().map(|(i, _)| i).collect();
    if cuts.is_empty() { return s; }
    for _ in 0..(1 + r.below(3)) {
        let cuts: Vec<usize> = s.char_indices().map(|(i, _)| i).chain(std::iter::once(s.len())).collect();
        let a = cuts[r.below(cuts.len())];
        let b = cuts[r.below(cuts.len())];
        let (a, b) = (a.min(b), a.max(b));
        match r.below(6) {
            0 => { let e = *cuts.iter().filter(|&&c| c >= a && c <= b && c <= a + 200).last().unwrap_or(&a); s.replace_range(a..e, ""); } // delete a span
            1 => { let e = *cuts.iter().filter(|&&c| c >= a && c <= b && c <= a + 400).last().unwrap_or(&a); let dup = s[a..e].to_string(); s.insert_str(a, &dup); } // duplicate a span
            2 => { s.insert_str(a, SOUP[r.below(43)]); }                           // insert a tag piece
            3 => { s.truncate(a); }                                                // truncate
            4 => { s.insert_str(a, r.pick(&VALS)); }
            _ => { let piece = r.pick(&["</parameter>", "</function>", "</tool_call>", "<tool_call>", "<function=run_shell>", "<parameter=command>"]);
                   if let Some(p) = s[a..].find(piece) { s.replace_range(a + p..a + p + piece.len(), ""); } }
        }
        if s.is_empty() { break; }
    }
    s
}


// ------------------------------------------------------------------------------------------
// the comparison
// ------------------------------------------------------------------------------------------

#[derive(Default)]
struct Tally { parses: usize, calls: usize, moved: usize, mism: usize, shown: usize }

fn clip(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect();
    format!("{t:?}{}", if s.chars().count() > n { " ...(cut)" } else { "" })
}

/// Do the two calls lists differ ONLY in the newlines around string arguments (item 8: the default
/// removes one leading and one trailing newline, v0.7.3 removed every one)?
fn only_newlines_differ(old: &[(String, String)], new: &[(String, String)]) -> bool {
    if old.len() != new.len() { return false; }
    let norm = |v: &Value| -> Value {
        match v {
            Value::Object(o) => Value::Object(o.iter().map(|(k, x)| (k.clone(), match x {
                Value::String(s) => Value::String(s.trim_matches('\n').to_string()),
                other => other.clone(),
            })).collect()),
            other => other.clone(),
        }
    };
    old.iter().zip(new).all(|(a, b)| {
        a.0 == b.0 && match (serde_json::from_str::<Value>(&a.1), serde_json::from_str::<Value>(&b.1)) {
            (Ok(x), Ok(y)) => norm(&x) == norm(&y),
            _ => a.1 == b.1,
        }
    })
}

/// One comparison at one catalogue and finish reason through the entries of the default parser:
/// `parse` and the server's `parse_served` (the defaults: they may differ from v0.7.3 by the
/// newline rule, item 8, and only by that, or by a class `moved_on_purpose` names) and the escape
/// hatch `Rules::V073` (`--tool-value-trim all`, `--tool-block-end first`), which must be v0.7.3
/// except for `moved_on_purpose`.
fn compare(t: &mut Tally, gen: &str, text: &str, tools: Option<&[Value]>, finish: &str) {
    let m = v073::parse(text, tools);
    let sig_m: Vec<(&str, &str, &str)> = m.tool_calls.iter().map(|c| (c.kind.as_str(), c.function.name.as_str(), c.function.arguments.as_str())).collect();
    let own_m: Vec<(String, String)> = m.tool_calls.iter().map(|c| (c.function.name.clone(), c.function.arguments.clone())).collect();
    for entry in ["parse", "parse_served", "rules_v073"] {
        let b = match entry {
            "parse" => parse(text, tools),
            "parse_served" => parse_served(text, tools, finish),
            _ => super::parse_rules(text, tools, &super::Rules::V073, Some(finish)),
        };
        t.parses += 1;
        t.calls += m.tool_calls.len();
        let sig_b: Vec<(&str, &str, &str)> = b.tool_calls.iter().map(|c| (c.kind.as_str(), c.function.name.as_str(), c.function.arguments.as_str())).collect();
        if m.content == b.content && sig_m == sig_b { continue; }
        let own_b: Vec<(String, String)> = b.tool_calls.iter().map(|c| (c.function.name.clone(), c.function.arguments.clone())).collect();
        if entry != "rules_v073" && m.content == b.content && only_newlines_differ(&own_m, &own_b) { t.moved += 1; continue; }
        if let Some(_why) = moved_on_purpose(text, finish) { t.moved += 1; continue; }
        t.mism += 1;
        if t.shown < 5 {
            t.shown += 1;
            eprintln!("MISMATCH gen={gen} entry={entry} tools={} finish={finish}\n  text   = {}\n  v0.7.3 = content {} calls {:?}\n  now    = content {} calls {:?}",
                      tools.map_or("none".to_string(), |x| x.len().to_string()), clip(text, 500), clip(&m.content, 200), sig_m, clip(&b.content, 200), sig_b);
        }
    }
}

fn run_generators(seed: u64, n: usize) -> Tally {
    let cat = catalogue();
    let mut t = Tally::default();
    let mut r = Rng(seed);
    for i in 0..n {
        let (gen, text) = match i % 6 {
            0 | 1 => ("soup", gen_soup(&mut r)),
            2 | 3 => ("struct", gen_struct(&mut r)),
            4 => ("cache", gen_cache(&mut r)),
            _ => match r.below(9) {
                0 => ("hy3", gen_hy3(&mut r)),
                1 => ("long", gen_long(&mut r)),
                2 | 3 => ("bare", gen_bare(&mut r)),
                _ => ("struct+mutated", { let s = gen_struct(&mut r); mutate(&mut r, &s) }),
            },
        };
        let tools: Option<&[Value]> = if r.chance(1, 2) { Some(&cat) } else { None };
        let finish = ["stop", "length", "tool_calls"][r.below(3)];
        compare(&mut t, gen, &text, tools, finish);
        // two blocks glued together (an agent turn with a reasoning draft and a real call)
        if i % 5 == 0 {
            let glued = format!("{}\n{}", text, gen_struct(&mut r));
            compare(&mut t, "glued", &glued, tools, "stop");
        }
    }
    t
}

/// The exhaustive blocks of the value-terminator sets, wrapped as one block of a response.
fn run_exhaustive(step: usize) -> Tally {
    use super::tests as tt;
    let cat = tt::exhaustive_catalogue();
    let mut t = Tally::default();
    for (calls, k3) in tt::exhaustive_blocks(tt::FRAG.len()).iter().step_by(step) {
        let body = tt::exhaustive_render(calls, *k3);
        let text = format!("<tool_call>\n{body}</tool_call>");
        compare(&mut t, "exhaustive", &text, Some(&cat), "stop");
        compare(&mut t, "exhaustive", &text, None, "stop");
    }
    t
}

/// The real outputs of S-B9-REL-SURFACE2B (348 of them), when the lab's PLAN directory is present.
fn run_real_outputs() -> Option<Tally> {
    let dir = format!("{}/PLAN/REL_SURFACE2_evidence/surface2b/gates/A3.dump_tools.log", env!("CARGO_MANIFEST_DIR"));
    let bytes = std::fs::read(&dir).ok()?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let mut t = Tally::default();
    let mut n = 0;
    for line in text.lines() {
        let Some(at) = line.find("[dump-tools] ") else { continue };
        let Ok(v) = serde_json::from_str::<Value>(&line[at + 13..]) else { continue };
        let Some(raw) = v["raw"].as_str() else { continue };
        n += 1;
        for finish in ["stop", "length"] {
            compare(&mut t, "real", raw, None, finish);
            compare(&mut t, "real-mutated", &mutate(&mut Rng(0xC0FFEE + n as u64), raw), None, finish);
        }
    }
    assert!(n > 300, "the real outputs are read: {n}");
    Some(t)
}

fn report(name: &str, t: &Tally) {
    println!("default-identity {name}: {} parses, {} calls returned, {} moved on purpose, {} mismatches", t.parses, t.calls, t.moved, t.mism);
}

/// The default is v0.7.3, every generator, both entries, with and without a catalogue, under three
/// finish reasons. A runtime of seconds in the suite.
#[test]
fn default_flags_read_every_generated_text_as_v073_did() {
    let t = run_generators(0x5EED_0001, 9_000);
    report("generators", &t);
    assert_eq!(t.mism, 0, "the default parser differs from v0.7.3 (see MISMATCH above)");
    assert!(t.calls > 5_000, "the generators must produce calls: {}", t.calls);
}

#[test]
fn default_flags_read_the_exhaustive_blocks_as_v073_did() {
    let t = run_exhaustive(23);
    report("exhaustive blocks (every 23rd)", &t);
    assert_eq!(t.mism, 0);
    assert!(t.parses > 20_000, "{}", t.parses);
}

#[test]
fn default_flags_read_the_real_outputs_as_v073_did() {
    let Some(t) = run_real_outputs() else { return };
    report("real outputs and mutations of them", &t);
    assert_eq!(t.mism, 0);
}

/// Writes exhaustive set A (every block of the 13 fragments, wrapped as one block of a response) as
/// JSON lines for the offline differential against vLLM's own parser (`scripts/toolparse_vllm_diff.py`):
/// run by hand, `cargo test --release --lib dump_exhaustive_set_a -- --ignored`, then
/// `target/release/examples/toolparse_dump_calls --raw target/toolparse_exhaustive_a.jsonl --out ...`.
#[test]
#[ignore]
fn dump_exhaustive_set_a() {
    use super::tests as tt;
    use std::io::Write;
    let path = format!("{}/target/toolparse_exhaustive_a.jsonl", env!("CARGO_MANIFEST_DIR"));
    let mut w = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    let mut n = 0usize;
    for (calls, k3) in &tt::exhaustive_blocks(13) {
        let body = tt::exhaustive_render(calls, *k3);
        let text = format!("<tool_call>\n{body}</tool_call>");
        writeln!(w, "{}", serde_json::json!({"raw": text})).unwrap();
        n += 1;
    }
    println!("wrote {n} blocks to {path}");
}

/// The long version (run by hand, release): a few million comparisons.
#[test]
#[ignore]
fn default_flags_long_differential() {
    let mut mism = 0;
    for seed in 0..40u64 {
        let t = run_generators(0x5EED_1000 + seed * 7919, 100_000);
        report(&format!("generators seed {seed}"), &t);
        mism += t.mism;
    }
    let t = run_exhaustive(1);
    report("exhaustive blocks (all)", &t);
    mism += t.mism;
    assert_eq!(mism, 0);

}
