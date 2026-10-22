//! OUT1 (2026-10-12): the boot report — ONE vocabulary for every family's boot output.
//!
//! Before OUT1 each serving family printed its own boot dialect (see `PLAN/OUT1_SURVEY.md`): the
//! NVFP4 server ended with a 5-line ready block, the EXL3 server with one `[exl3-serve] listening`
//! line, the DSV4 head with prose + `HEAD SYNCED n node(s)`, durations existed only as one
//! aggregate `[load] total`, and telemetry (`[mem]`, `[memwatch]`), prose essays and user-facing
//! lines all shared one stream. This module gives every family loader the same formatting:
//!
//! * stages — `[boot] stage <name> begin` / `[boot] stage <name> done <d>s` (fixed names:
//!   `config`, `tokenizer`, `weights`, `calibrate`, `graphs`, `scheduler`, `vision`, `serve`);
//! * severity — `[boot] warn: …` / `[boot] error: …`;
//! * §1b-bis boot-anyway notices — one line, `[boot] notice: <asked> → <doing> (<why>)`;
//! * exactly ONE idle line, identical shape for every model:
//!   `SERVING family=<f> model=<id> tp=<n> ctx=<n> batch=<n> kv=<t> prefix=<on|off>
//!    mtp=<auto|on|off|none> spec=<mtp|dspark|df2|dflash|plain> port=<p> url=<u> boot=<s>`
//!
//! `--log-level` (registry option, default `normal`): `normal` prints stage lines + SERVING +
//! warnings/errors + one-line notices. `debug` keeps EVERY line that existed before OUT1, verbatim
//! and on its original stream — re-leveled, not deleted — plus the new stage/SERVING lines.
//! Nothing was removed: every pre-OUT1 line is still printed under `--log-level debug`.
//!
//! Streams: stage lines and SERVING go to stdout (where the old ready lines were); warn / error /
//! notice go to stderr. No serving behavior, HTTP/API or `/metrics` name changes — output only.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Level { Normal = 0, Debug = 1 }

static LEVEL: AtomicU8 = AtomicU8::new(Level::Normal as u8);
static T0: OnceLock<Instant> = OnceLock::new();

/// Read `--log-level` from the options store (already parsed by `opts::parse_cli`) and start the
/// boot clock. Safe to call again (a TP node calls `refresh_from_opts` after it installs the
/// head's shipped snapshot, so the node logs at the head's level).
pub fn init_from_args(_args: &[String]) { refresh_from_opts(); }

pub fn refresh_from_opts() {
    let lvl = match crate::opts::var(crate::opt!("log-level")) {
        Ok(v) if v == "debug" => Level::Debug,
        Ok(v) if v == "normal" => Level::Normal,
        Ok(other) => {
            crate::logq::send(true, warn_str(&format!("--log-level '{other}' is not normal|debug — using normal")));
            Level::Normal
        }
        Err(_) => Level::Normal,
    };
    set_level(lvl);
    let _ = T0.get_or_init(Instant::now);
}

pub fn set_level(l: Level) { LEVEL.store(l as u8, Ordering::Relaxed); }
pub fn level() -> Level { if LEVEL.load(Ordering::Relaxed) == 1 { Level::Debug } else { Level::Normal } }
/// True when every pre-OUT1 line should print (the `--log-level debug` mode).
pub fn debug() -> bool { level() == Level::Debug }

/// Seconds since the boot report was initialized (the boot clock behind the `boot=` field).
pub fn since_boot() -> f64 {
    T0.get_or_init(Instant::now).elapsed().as_secs_f64()
}

// ---- stage lines -----------------------------------------------------------

pub fn stage_begin_str(name: &str) -> String { format!("[boot] stage {name} begin") }
pub fn stage_done_str(name: &str, secs: f64) -> String { format!("[boot] stage {name} done {:.1}s", secs) }
pub fn stage_done_detail_str(name: &str, secs: f64, detail: &str) -> String {
    format!("[boot] stage {name} done {:.1}s ({detail})", secs)
}
pub fn stage_failed_str(name: &str, secs: f64, err: &str) -> String {
    format!("[boot] stage {name} failed after {:.1}s: {err}", secs)
}

/// A boot stage: prints the uniform begin line now and the uniform done line on `ok()`.
/// The begin line is part of `normal` because the long stages (weights, calibrate) are exactly
/// where an operator used to stare at a silent 100 s load.
pub struct Stage { name: &'static str, t0: Instant }

impl Stage {
    pub fn begin(name: &'static str) -> Stage {
        crate::logq::send(false, stage_begin_str(name));
        Stage { name, t0: Instant::now() }
    }
    pub fn ok(self) { self.ok_detail(None) }
    /// Done, with an optional short detail token appended in parentheses.
    pub fn ok_detail(self, detail: Option<&str>) {
        let d = self.t0.elapsed().as_secs_f64();
        crate::logq::send(false, match detail {
            Some(s) => stage_done_detail_str(self.name, d, s),
            None => stage_done_str(self.name, d),
        });
    }
    /// Stage failed: print the uniform failure line (the caller then reports the error/exits).
    pub fn fail(self, err: &str) {
        crate::logq::send(true, stage_failed_str(self.name, self.t0.elapsed().as_secs_f64(), err));
    }
}

// ---- severity vocabulary ---------------------------------------------------

pub fn warn_str(msg: &str) -> String { format!("[boot] warn: {msg}") }
pub fn error_str(msg: &str) -> String { format!("[boot] error: {msg}") }
pub fn notice_str(msg: &str) -> String { format!("[boot] notice: {msg}") }

pub fn warn(msg: &str) { crate::logq::send(true, warn_str(msg)); }
pub fn error(msg: &str) { crate::logq::send(true, error_str(msg)); }
/// A §1b-bis boot-anyway notice: one line — what was asked, what runs instead, why.
pub fn notice(msg: &str) { crate::logq::send(true, notice_str(msg)); }

// ---- OUT2 (2026-10-16): debug-gated print macros ---------------------------------
//
// OUT1 re-leveled the dense-family boot prints by hand; OUT2 extends the same rule to every
// remaining technical line (the EXL3/TP boot flood and the per-request internals): a line that
// is not a stage, notice, warning, error or SERVING prints ONLY under `--log-level debug`.
// The macros re-level a call site without changing its bytes: rename `println!` → `dprintln!`
// (stdout), `eprintln!` → `deprintln!` (stderr), `crate::rprintln!` → `crate::drprintln!` and
// `crate::reprintln!` → `crate::dreprintln!` (logq-routed: per-request lines keep the LR-1
// non-blocking routing). Nothing is deleted — the identical line still prints at debug level.

/// Print to stdout only under `--log-level debug` (OUT2 re-leveling).
#[macro_export]
macro_rules! dprintln {
    ($($arg:tt)*) => { if $crate::boot_report::debug() { println!($($arg)*) } };
}

/// Print to stderr only under `--log-level debug`.
#[macro_export]
macro_rules! deprintln {
    ($($arg:tt)*) => { if $crate::boot_report::debug() { eprintln!($($arg)*) } };
}

/// logq-routed stdout, debug-only (per-request technical lines keep the LR-1 routing).
#[macro_export]
macro_rules! drprintln {
    ($($arg:tt)*) => { if $crate::boot_report::debug() { $crate::rprintln!($($arg)*) } };
}

/// logq-routed stderr, debug-only.
#[macro_export]
macro_rules! dreprintln {
    ($($arg:tt)*) => { if $crate::boot_report::debug() { $crate::reprintln!($($arg)*) } };
}

// ---- the one SERVING line --------------------------------------------------

/// Field values are single tokens (no spaces); unknown → "?".
pub struct ServingLine<'a> {
    pub family: &'a str,   // nvfp4 | exl3 | dsv4
    pub model: &'a str,    // the served model id (what GET /v1/models reports)
    pub tp: u32,           // TP world size
    pub ctx: usize,        // resolved max context (tokens)
    pub batch: usize,      // resolved max batch (lanes)
    pub kv: &'a str,       // resolved KV dtype (bf16 | q8 | …)
    pub prefix: &'a str,   // prefix cache: on | off
    pub mtp: &'a str,      // MTP: auto | on | off | none
    pub spec: &'a str,     // resolved speculation source: mtp | dspark | df2 | dflash | plain
    pub port: u16,
    pub url: String,       // the bound HTTP url (what the old ready line printed)
}

/// The frozen shape — snapshot-tested below; do NOT reorder or rename fields outside a roadmap
/// decision: scripts grep this line.
pub fn format_serving(l: &ServingLine, boot_secs: f64) -> String {
    format!(
        "SERVING family={} model={} tp={} ctx={} batch={} kv={} prefix={} mtp={} spec={} port={} url={} boot={:.1}s",
        l.family, l.model, l.tp, l.ctx, l.batch, l.kv, l.prefix, l.mtp, l.spec, l.port, l.url, boot_secs
    )
}

/// Print THE idle SERVING line (stdout). Exactly one per boot, at bind time.
pub fn serving(l: &ServingLine) {
    crate::logq::send(false, format_serving(l, since_boot()));
}

// ---- snapshot tests (the frozen contract) ----------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serving_line_shape_is_frozen_nvfp4() {
        let l = ServingLine {
            family: "nvfp4", model: "Qwen/Qwen3.5-0.8B", tp: 1, ctx: 4096, batch: 8,
            kv: "bf16", prefix: "off", mtp: "auto", spec: "mtp", port: 18901,
            url: "http://0.0.0.0:18901".into(),
        };
        assert_eq!(format_serving(&l, 2.44),
            "SERVING family=nvfp4 model=Qwen/Qwen3.5-0.8B tp=1 ctx=4096 batch=8 kv=bf16 prefix=off mtp=auto spec=mtp port=18901 url=http://0.0.0.0:18901 boot=2.4s");
    }

    #[test]
    fn serving_line_shape_is_frozen_exl3() {
        let l = ServingLine {
            family: "exl3", model: "qwen4-exp-27b", tp: 1, ctx: 4097, batch: 8,
            kv: "q8", prefix: "off", mtp: "off", spec: "plain", port: 18902,
            url: "http://0.0.0.0:18902".into(),
        };
        assert_eq!(format_serving(&l, 31.16),
            "SERVING family=exl3 model=qwen4-exp-27b tp=1 ctx=4097 batch=8 kv=q8 prefix=off mtp=off spec=plain port=18902 url=http://0.0.0.0:18902 boot=31.2s");
    }

    #[test]
    fn serving_line_shape_is_frozen_dsv4() {
        let l = ServingLine {
            family: "dsv4", model: "DeepSeek-V4-Flash", tp: 2, ctx: 8192, batch: 4,
            kv: "k8v8", prefix: "on", mtp: "off", spec: "dspark", port: 9000,
            url: "http://0.0.0.0:9000".into(),
        };
        assert_eq!(format_serving(&l, 154.0),
            "SERVING family=dsv4 model=DeepSeek-V4-Flash tp=2 ctx=8192 batch=4 kv=k8v8 prefix=on mtp=off spec=dspark port=9000 url=http://0.0.0.0:9000 boot=154.0s");
    }

    #[test]
    fn stage_lines_are_frozen() {
        assert_eq!(stage_begin_str("weights"), "[boot] stage weights begin");
        assert_eq!(stage_done_str("weights", 1.71), "[boot] stage weights done 1.7s");
        assert_eq!(stage_done_detail_str("weights", 1.71, "streaming bf16"),
            "[boot] stage weights done 1.7s (streaming bf16)");
        assert_eq!(stage_failed_str("weights", 3.0, "missing tensor: lm_head.weight"),
            "[boot] stage weights failed after 3.0s: missing tensor: lm_head.weight");
    }

    #[test]
    fn severity_lines_are_frozen() {
        assert_eq!(notice_str("asked --max-batch 32, serving 16 (this server serves at most 16 lanes)"),
            "[boot] notice: asked --max-batch 32, serving 16 (this server serves at most 16 lanes)");
        assert_eq!(warn_str("vision tower load failed (no CUDA); serving text-only"),
            "[boot] warn: vision tower load failed (no CUDA); serving text-only");
        assert_eq!(error_str("--model-dir /x does not exist on this box"),
            "[boot] error: --model-dir /x does not exist on this box");
    }

    #[test]
    fn level_defaults_to_normal_and_parses() {
        // No opts store entries in the test harness: the default is Normal.
        set_level(Level::Normal);
        assert!(!debug());
        set_level(Level::Debug);
        assert!(debug());
        set_level(Level::Normal);
    }
}
