//! FLASHNEXT-DF2-B (ISSUES work[113]): unsent-field sampler defaults come from the pack's
//! generation_config.json; the family card / legacy triple only when the pack has none.
use gb10_inference::server::SamplingDefaults;

fn pack(dir: &std::path::Path, body: Option<&str>) -> String {
    std::fs::create_dir_all(dir).unwrap();
    if let Some(b) = body { std::fs::write(dir.join("generation_config.json"), b).unwrap(); }
    dir.to_string_lossy().to_string()
}

#[test]
fn ornith_card_triple_replaces_legacy_on_the_nvfp4_server() {
    let d = std::env::temp_dir().join(format!("sd_ornith_{}", std::process::id()));
    let dir = pack(&d, Some(r#"{"do_sample": true, "temperature": 0.6, "top_k": 20, "top_p": 0.95, "eos_token_id": [1, 2]}"#));
    let (sd, src) = SamplingDefaults::from_pack(&dir, SamplingDefaults::LEGACY);
    assert_eq!(src, "generation_config.json");
    assert_eq!(sd.think, (0.6, 0.95, 20));
    assert_eq!(sd.no_think, (0.6, 0.95, 20), "a non-Qwen3.8 fallback uses the file's triple for both modes");
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn qwen38_pack_is_identity_with_the_card_and_keeps_the_instruct_triple() {
    let d = std::env::temp_dir().join(format!("sd_q38_{}", std::process::id()));
    let dir = pack(&d, Some(r#"{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "do_sample": true}"#));
    let (sd, src) = SamplingDefaults::from_pack(&dir, SamplingDefaults::QWEN38_CARD);
    assert_eq!(src, "generation_config.json");
    assert_eq!(sd.think, SamplingDefaults::QWEN38_CARD.think);
    assert_eq!(sd.no_think, SamplingDefaults::QWEN38_CARD.no_think, "the card's 0.7/0.8/20 instruct values are not in the file and must survive");
    std::fs::remove_dir_all(&d).ok();
}

#[test]
fn missing_or_temperatureless_file_falls_back() {
    let d = std::env::temp_dir().join(format!("sd_none_{}", std::process::id()));
    let dir = pack(&d, None);
    let (sd, src) = SamplingDefaults::from_pack(&dir, SamplingDefaults::LEGACY);
    assert!(src.starts_with("built-in"), "{src}");
    assert_eq!(sd.think, SamplingDefaults::LEGACY.think);
    std::fs::write(d.join("generation_config.json"), r#"{"eos_token_id": 7}"#).unwrap();
    let (sd2, src2) = SamplingDefaults::from_pack(&dir, SamplingDefaults::QWEN38_CARD);
    assert!(src2.starts_with("built-in"), "{src2}");
    assert_eq!(sd2.think, SamplingDefaults::QWEN38_CARD.think);
    let _ = sd.describe(src); // the boot line renders
    std::fs::remove_dir_all(&d).ok();
}
