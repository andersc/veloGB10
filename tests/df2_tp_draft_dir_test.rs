//! FLASHNEXT-DF2-C (ISSUES 116): under an EXL3 TP serve the drafter rides the sync; the shipped
//! config decides the draft dir on every rank, and the head ships only a DFlash2-class source's dir.
use gb10_inference::exl3_serve::{df2_draft_dir_for_boot, tp_head_draft_dir};
use gb10_inference::tp::TpConfig;

fn args(v: &[&str]) -> Vec<String> { v.iter().map(|s| s.to_string()).collect() }

#[test]
fn single_box_reads_argv() {
    let a = args(&["gb10_inference", "--server", "--draft-dir", "/d/flash-next"]);
    assert_eq!(df2_draft_dir_for_boot(&a, None).as_deref(), Some("/d/flash-next"));
    let mut c = TpConfig::from_opts(); c.mode_serve = true; c.exl3 = true; c.world = 1; c.df2_draft_dir = "/cache/x".into();
    assert_eq!(df2_draft_dir_for_boot(&a, Some(&c)).as_deref(), Some("/d/flash-next"), "world 1 is a single-box boot");
}

#[test]
fn tp_ranks_take_the_shipped_dir_or_nothing() {
    let a = args(&["gb10_inference", "--server", "--tp", "2", "--draft-dir", "/home/head/models/draft"]);
    let mut c = TpConfig::from_opts(); c.mode_serve = true; c.exl3 = true; c.world = 2;
    c.df2_draft_dir = "/home/node/.cache/gb10_tp/models/draft@x".into();
    assert_eq!(df2_draft_dir_for_boot(&a, Some(&c)).as_deref(), Some("/home/node/.cache/gb10_tp/models/draft@x"),
               "a node loads the blob-cache copy the sync assembled, never the head's path from argv");
    c.df2_draft_dir.clear();
    assert_eq!(df2_draft_dir_for_boot(&a, Some(&c)), None, "the head shipped no drafter: no local dir may stand in");
    let mut n = TpConfig::from_opts(); n.mode_serve = true; n.exl3 = false; n.world = 2; n.df2_draft_dir = "/cache/y".into();
    assert_eq!(df2_draft_dir_for_boot(&a, Some(&n)).as_deref(), Some("/home/head/models/draft"), "not an EXL3 serve config: argv");
}

#[test]
fn head_ships_only_a_df2_source_dir_as_an_absolute_path() {
    let d = std::env::temp_dir().join(format!("df2c_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let rel = d.to_string_lossy().to_string();
    let a = args(&["gb10_inference", "--server", "--tp", "2", "--spec-source", "dflash2", "--draft-dir", &rel]);
    let got = tp_head_draft_dir(&a).expect("a DF2 source with an existing dir ships");
    assert_eq!(got, std::fs::canonicalize(&d).unwrap().to_string_lossy());
    let a_mtp = args(&["gb10_inference", "--server", "--tp", "2", "--spec-source", "mtp", "--draft-dir", &rel]);
    assert_eq!(tp_head_draft_dir(&a_mtp), None, "a non-DF2 source ships no drafter");
    let a_none = args(&["gb10_inference", "--server", "--tp", "2"]);
    assert_eq!(tp_head_draft_dir(&a_none), None, "no --draft-dir: nothing to ship (the default dflash2-auto falls back at the gate)");
    let a_missing = args(&["gb10_inference", "--server", "--tp", "2", "--draft-dir", "/nonexistent/df2c"]);
    assert_eq!(tp_head_draft_dir(&a_missing), None, "an unreadable path ships nothing; the head's own load/gate reports it");
    std::fs::remove_dir_all(&d).ok();
}
