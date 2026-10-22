//! TOOLEVAL (P18, S-JOHNFB3): a flag-gated per-turn trace of the tool-call serving path.
//!
//! `--tool-trace <file>` arms a JSONL trace with one line per chat turn that involves tools —
//! the request offers tools, or its history carries a tool result. Each line holds the three
//! artifacts an engine-vs-quant attribution needs, byte for byte:
//!
//!   * `prompt_tail`  — the tail of the EXACT template-rendered prompt the model saw. A chat
//!                      template divergence (engine vs a rival stack) is visible here as a byte
//!                      diff;
//!   * `raw_full`     — the model's complete raw output, thinking included;
//!   * `parser_input` — the text the tool-call parser was given (raw minus the think block, the
//!                      same string `--dump-tools` prints);
//!   * `verdict`      — what the parser made of it: kept content, parsed tool calls (id, name,
//!                      arguments), and the finish reason handed to the client;
//!   * `meta`         — the request class: stream/non-stream, seed, resolved sampling,
//!                      penalties, thinking mode, tool count/choice, message roles. Every
//!                      number in a report that used this trace can name its request class.
//!
//! OFF by default. When off the whole module costs one registry read per request (`capture`
//! returns `None` and the call sites then do nothing). No served byte changes in either
//! state: the trace only READS the prompt, the raw output and the parse the server already
//! computed. The lines contain prompts and model output: treat the file as user data.

use std::io::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::tokenizer::ToolCall;

/// Prompt tail kept per turn: long enough to hold the tool-result region of an ordinary agent
/// turn (the generation prompt plus the last user/tool messages), bounded so a 262K-token
/// prompt cannot turn the trace file into the disk problem.
pub const TAIL_CHARS: usize = 6144;

/// The request-side half of one traced turn, captured where the prompt is rendered and the
/// sampling is resolved. `Send + 'static`: it rides into the SSE stream struct unchanged.
pub struct Turn {
    pub meta: serde_json::Value,
    pub prompt_tail: String,
}

/// The trace file, when the flag is set to one. Read per traced request (the value is a short
/// path string and this is once per request, not once per token), so tests can arm and disarm
/// the trace through the ordinary option store.
fn armed() -> Option<String> {
    crate::opts::get(crate::opt!("tool-trace"))
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty() && p != "off" && p != "0")
}

/// The last `n` BYTES of `s`, cut back to a char boundary (a multi-byte token can straddle
/// the cut; sliding back keeps every char whole).
pub fn tail_of(s: &str, n: usize) -> &str {
    if s.len() <= n {
        return s;
    }
    let mut cut = s.len() - n;
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    &s[cut..]
}

/// Capture the request side of a traced turn. `None` (cheap) when the flag is off.
pub fn capture(prompt: &str, meta: serde_json::Value) -> Option<Turn> {
    let path = armed()?;
    open_header(&path);
    Some(Turn { meta, prompt_tail: tail_of(prompt, TAIL_CHARS).to_string() })
}

/// Log the completed turn: what the model emitted and what the parser made of it.
/// `raw_full` is the complete generation (thinking included); `parser_input` is the text the
/// tool parser read after the think split; `content`/`calls`/`finish` are the finalized
/// verdict — exactly what the client was handed.
pub fn log(t: &Turn, raw_full: &str, parser_input: &str, content: Option<&str>, calls: &[ToolCall], finish: &str) {
    let Some(path) = armed() else { return };
    let line = serde_json::json!({
        "ev": "turn", "ts": chrono::Utc::now().to_rfc3339(),
        "meta": t.meta,
        "prompt_tail_chars": t.prompt_tail.chars().count(),
        "prompt_tail": t.prompt_tail,
        "raw_chars": raw_full.chars().count(),
        "raw_full": raw_full,
        "parser_input": parser_input,
        "verdict": {
            "content": content,
            "tool_calls": calls.iter().map(|c| serde_json::json!({
                "id": c.id, "name": c.function.name, "arguments": c.function.arguments,
            })).collect::<Vec<_>>(),
            "finish": finish,
        },
    });
    append(&path, &line.to_string());
}

/// One `{"ev":"open"}` line per process, so a trace file always names the run that wrote it.
fn open_header(path: &str) {
    static DONE: OnceLock<()> = OnceLock::new();
    DONE.get_or_init(|| {
        let line = serde_json::json!({
            "ev": "open", "ts": chrono::Utc::now().to_rfc3339(), "pid": std::process::id(),
        });
        append(path, &line.to_string());
    });
}

/// Serialize writes (requests are served concurrently) and never spam: the first write failure
/// disables the trace for the process with ONE log line — a diagnostic must not become the
/// outage.
fn append(path: &str, line: &str) {
    static LOCK: Mutex<()> = Mutex::new(());
    static BROKEN: AtomicBool = AtomicBool::new(false);
    if BROKEN.load(Ordering::Relaxed) {
        return;
    }
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let r = std::fs::OpenOptions::new().create(true).append(true).open(path)
        .and_then(|mut f| writeln!(f, "{}", line));
    if let Err(e) = r {
        BROKEN.store(true, Ordering::Relaxed);
        crate::reprintln!("[tool-trace] write to {} failed ({e}); tool-trace disabled for this process", path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_of_cuts_on_char_boundary() {
        assert_eq!(tail_of("abc", 0), "");
        assert_eq!(tail_of("abc", 2), "bc");
        assert_eq!(tail_of("abc", 10), "abc");
        // 'é' is 2 bytes: a byte cut at len-1 lands mid-char and must slide forward
        let s = "abé";
        assert_eq!(tail_of(s, 1), "é");
        // 4-byte emoji at the head of the tail
        let s = "a\u{1F600}b";
        assert_eq!(tail_of(s, 2), "\u{1F600}b");
    }

    #[test]
    fn tail_of_is_bounded_and_valid() {
        let s = "x".repeat(TAIL_CHARS * 2);
        let t = tail_of(&s, TAIL_CHARS);
        assert_eq!(t.chars().count(), TAIL_CHARS);
        assert!(t.len() <= TAIL_CHARS + 3); // at most one straddling char slides the cut
    }

    /// Armed through the ordinary option store; off unless the flag names a file. Restores the
    /// store so the (process-global) read cannot leak into other tests.
    #[test]
    fn capture_off_by_default_and_writes_when_armed() {
        let id = crate::opt!("tool-trace");
        let prev = crate::opts::var(id).ok();
        crate::opts::unset(id);
        assert!(crate::opts::get(id).is_none() || !armed().is_some(), "no flag = no trace");
        assert!(capture("prompt", serde_json::json!({"id": "t0"})).is_none());

        let path = std::env::temp_dir().join(format!("tool_trace_test_{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        crate::opts::set(id, path.to_str().unwrap());

        let t = capture("a prompt tail", serde_json::json!({"id": "t1", "seed": 7}))
            .expect("armed flag captures");
        assert_eq!(t.prompt_tail, "a prompt tail");
        log(&t, "raw <think>r</think> body", "body", Some("body"), &[], "stop");
        log(&t, "raw2", "raw2", None, &[ToolCall {
            id: "c1".into(), kind: "function".into(),
            function: crate::tokenizer::FunctionCall { name: "get_weather".into(), arguments: "{\"city\":\"Paris\"}".into() },
        }], "tool_calls");

        let txt = std::fs::read_to_string(&path).expect("trace file written");
        let lines: Vec<&str> = txt.lines().collect();
        assert_eq!(lines.len(), 3, "open header + two turns");
        let open: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(open["ev"], "open");
        let v: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(v["meta"]["seed"], 7);
        assert_eq!(v["verdict"]["finish"], "stop");
        assert_eq!(v["verdict"]["content"], "body");
        let v: serde_json::Value = serde_json::from_str(lines[2]).unwrap();
        assert_eq!(v["verdict"]["tool_calls"][0]["name"], "get_weather");
        assert_eq!(v["verdict"]["tool_calls"][0]["arguments"], "{\"city\":\"Paris\"}");
        assert_eq!(v["verdict"]["finish"], "tool_calls");
        let _ = std::fs::remove_file(&path);

        match prev {
            Some(v) => crate::opts::set(id, v),
            None => crate::opts::unset(id),
        }
    }
}
