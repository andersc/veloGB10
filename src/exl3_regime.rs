//! S-B9-REL-QSAREGIME: `--probe-exl3-regime`, the mixed-regime decode-step instrument.
//!
//! A decode step that carries a row past `qsa_limit()` runs the sparse (QSA) attention kernels;
//! a row at or below the limit that shares the step used to take them too (the regime was chosen
//! once per step from the largest position). This probe measures what that does to the short row
//! and is the gate of the per-row regime fix. A child module of `exl3_forward` so it reaches the
//! served `forward_step` / `prefill_chunk` and the scratch logits without widening them.
//!
//! Nine slots, the SAME runner prompt in four of them, teacher-forced with the solo arm's argmax:
//!   A  slot 0          runner alone                       (m = 1; the reference)
//!   B  slots 1, 2      runner + a short partner           (m = 2, both rows at/below the limit: the control)
//!   D  slot 5          joiner alone                       (m = 1; the long row's reference)
//!   C  slots 3, 4      runner + the long joiner           (m = 2, mixed: the row under test)
//!   E  slots 6, 7, 8   runner + partner + joiner          (m = 3, mixed with TWO short rows)
//! Every arm is the served `forward_step` (graphs included), its rows ordered as the scheduler
//! orders them (`regime_order`: rows at/below the switch first; --regime-order long-first puts the
//! joiner first in C and E instead). Per step: B's runner row vs A and the joiner rows of C / E vs D
//! must be bit-identical (co-batching alone changes nothing; the long row does not depend on its
//! neighbours); C's runner row vs A is the measurement — bit differences, max |dlogit|, rel-L2,
//! KL(A || C), top-1 agreement and A's top-1/top-2 margin at every disagreement; E repeats it with
//! two short rows (runner vs A, partner vs its row in B).

use super::*;
use super::kvnll::{f16_lut, kl_rows, pct, read_ids, score_row};

pub struct RegimeArgs {
    pub dir: String,
    pub max_pos: usize,
    /// prefill chunk width C
    pub chunk: usize,
    /// runner prompt: raw text (tokenized here) or token ids (JSON array / whitespace)
    pub runner: String,
    /// joiner prompt, same formats; must be longer than qsa_limit() tokens
    pub joiner: String,
    /// teacher-forced decode steps
    pub steps: usize,
    /// per-step CSV
    pub out_csv: Option<String>,
    /// row order inside the mixed step: false = [runner, joiner], true = [joiner, runner]
    pub long_first: bool,
    /// after the comparison loop: time this many 4-row mixed steps (three short rows + the joiner)
    pub time_steps: usize,
}

/// One arm-vs-reference comparison of two f16 logits rows.
struct RowCmp { bits: usize, maxabs: f32, rel_l2: f64, kl: f64, agree: bool, top5: usize }

fn cmp_rows(r: &[u16], a: &[u16], lut: &[f32]) -> RowCmp {
    let (rs, as_) = (score_row(r, lut), score_row(a, lut));
    let (mut bits, mut maxabs, mut num, mut den) = (0usize, 0f32, 0f64, 0f64);
    for (&rb, &ab) in r.iter().zip(a) {
        if rb == ab { let v = lut[rb as usize] as f64; if v.is_finite() { den += v * v; } continue; }
        bits += 1;
        let (x, y) = (lut[rb as usize], lut[ab as usize]);
        if !x.is_finite() || !y.is_finite() { maxabs = f32::INFINITY; continue; }
        maxabs = maxabs.max((x - y).abs());
        num += ((x - y) as f64).powi(2);
        den += (x as f64).powi(2);
    }
    let kl = if bits == 0 { 0.0 } else { kl_rows(r, &rs, a, &as_, lut) };
    let top5 = rs.top5.iter().filter(|t| as_.top5.contains(t)).count();
    RowCmp { bits, maxabs, rel_l2: (num / den.max(1e-30)).sqrt(), kl, agree: rs.argmax == as_.argmax, top5 }
}

/// top-1 / top-2 logit gap of a row (the near-tie measure at a disagreement).
fn margin(row: &[u16], lut: &[f32]) -> f32 {
    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &x in row {
        let v = lut[x as usize];
        if !v.is_finite() { continue; }
        if v > a { b = a; a = v; } else if v > b { b = v; }
    }
    a - b
}

fn load_ids(path: &str, tok: &crate::tokenizer::QwenTokenizer) -> Result<Vec<u32>> {
    if let Ok(v) = read_ids(path) { if !v.is_empty() { return Ok(v); } }
    tok.encode(&std::fs::read_to_string(path).with_context(|| format!("read {path}"))?, false)
}

#[derive(Default)]
struct Agg { n: usize, bit_steps: usize, bits: usize, disagree: usize, kl: Vec<f64>, rel: Vec<f64>, maxabs: f32, top5: usize }

impl Agg {
    fn add(&mut self, c: &RowCmp) {
        self.n += 1;
        self.bit_steps += (c.bits > 0) as usize;
        self.bits += c.bits;
        self.disagree += !c.agree as usize;
        self.kl.push(c.kl);
        self.rel.push(c.rel_l2);
        self.maxabs = self.maxabs.max(c.maxabs);
        self.top5 += c.top5;
    }
    fn line(&mut self) -> String {
        if self.n == 0 { return "n=0".to_string(); }
        let n = self.n as f64;
        let klm = self.kl.iter().sum::<f64>() / n;
        let klx = self.kl.iter().cloned().fold(0.0, f64::max);
        let (k99, r50) = (pct(&mut self.kl, 0.99), pct(&mut self.rel, 0.5));
        let rx = self.rel.iter().cloned().fold(0.0, f64::max);
        format!("n={} steps_with_bit_diff={} logits_differing_mean={:.0} top1_disagree={} top1_agree={:.4} top5_overlap={:.4} \
                 kl_mean={klm:.3e} kl_p99={k99:.3e} kl_max={klx:.3e} rel_l2_median={r50:.3e} rel_l2_max={rx:.3e} max_abs_dlogit={:.4}",
                self.n, self.bit_steps, self.bits as f64 / n, self.disagree, 1.0 - self.disagree as f64 / n,
                self.top5 as f64 / (5.0 * n), self.maxabs)
    }
}

pub fn probe_regime(a: &RegimeArgs) -> Result<()> {
    const SLOTS: usize = 9;
    let out = |s: String| println!("{s}");
    let tok = crate::tokenizer::QwenTokenizer::from_file(&format!("{}/tokenizer.json", a.dir.trim_end_matches('/')))?;
    let runner = load_ids(&a.runner, &tok)?;
    let joiner = load_ids(&a.joiner, &tok)?;
    let model = FwdModel::load(&a.dir, SLOTS, a.max_pos)?;
    let lim = model.qsa_limit();
    let v = model.cfg.vocab_size;
    anyhow::ensure!(runner.len() >= 2 && joiner.len() >= 2, "empty prompt");
    anyhow::ensure!(runner.iter().chain(&joiner).all(|&t| (t as usize) < v), "prompt has token ids >= vocab {v}");
    let total = a.steps + a.time_steps + 32;
    anyhow::ensure!(joiner.len() + total < a.max_pos && runner.len() + total < a.max_pos,
                    "--max-seq-len {} too small for the prompts + {total} steps", a.max_pos);
    // the partner: a short dense row with other content (the head of the joiner document)
    let partner: Vec<u32> = joiner[..joiner.len().min(600).min(lim / 2)].to_vec();
    out(format!("REGIME probe: runner {} tok, joiner {} tok, partner {} tok, steps {}, qsa_limit {lim} (rows at pos > {lim} are sparse), \
                 qsa_live(joiner)={}, kv_fmt={}, C={}, mixed row order = {}",
                runner.len(), joiner.len(), partner.len(), a.steps, model.qsa_live(joiner.len() - 1),
                kv_fmt_name(model.kv_fmt), a.chunk, if a.long_first { "[joiner, runner]" } else { "[runner, joiner]" }));
    if !model.qsa_live(joiner.len() - 1) {
        out("REGIME note: the joiner is NOT past the sparse switch (short prompt, or --qsa-exl3 0): every step is dense".to_string());
    }
    let mut sc = Scratch::new(&model.dev, &model.cfg, SLOTS, model.cfg.rotary_dim, model.max_pos())?;
    let mut psc = model.prefill_scratch(a.chunk)?;
    // prefill: every slot gets its prompt minus the last token (the seam token is the first step's input)
    let roles: [&[u32]; SLOTS] = [&runner, &runner, &partner, &runner, &joiner, &joiner, &runner, &partner, &joiner];
    let t0 = std::time::Instant::now();
    for (slot, ids) in roles.iter().enumerate() {
        model.reset_slot(slot)?;
        let n = ids.len() - 1;
        let mut pos = 0;
        while pos < n {
            let end = (pos + a.chunk).min(n);
            let toks: Vec<i32> = ids[pos..end].iter().map(|&t| t as i32).collect();
            model.prefill_chunk(&mut psc, &toks, pos, slot, false, None)?;
            pos = end;
        }
    }
    model.dev.synchronize()?;
    out(format!("REGIME prefill of {SLOTS} slots: {:.1} s", t0.elapsed().as_secs_f64()));
    drop(psc);

    let lut = f16_lut();
    let mut csv = match &a.out_csv {
        Some(p) => {
            let mut w = std::io::BufWriter::new(std::fs::File::create(p).with_context(|| format!("create {p}"))?);
            writeln!(w, "step,runner_pos,runner_sparse,tok,ctl_bits,long_bits,bits,max_abs,rel_l2,kl,agree,top5,margin_ref,arm_argmax")?;
            Some(w)
        }
        None => None,
    };
    // (token, position) of the next input row of each stream
    let (mut rt, mut rp) = (*runner.last().unwrap() as i32, runner.len() - 1);
    let (mut pt, mut pp) = (*partner.last().unwrap() as i32, partner.len() - 1);
    let (mut jt, mut jp) = (*joiner.last().unwrap() as i32, joiner.len() - 1);
    let (mut ctl, mut long, mut below, mut above, mut all) = (Agg::default(), Agg::default(), Agg::default(), Agg::default(), Agg::default());
    let (mut e_run, mut e_par, mut e_long) = (Agg::default(), Agg::default(), Agg::default());
    let mut bucket = Agg::default();
    let mut flips: Vec<String> = Vec::new();
    let mut first_bit: Option<usize> = None;
    let mut gen: Vec<u32> = Vec::new();
    let mut c_ms: Vec<f64> = Vec::new();
    // One served step over `rows` = (token, position, slot), ordered as the scheduler orders a plain
    // step (rows at/below the switch first; `long_first` = the opposite, the unsorted caller).
    // Returns each row's (argmax id, logits row) in the GIVEN order, and the step's wall ms.
    let step_rows = |sc: &mut Scratch, rows: &[(i32, usize, usize)], long_first: bool| -> Result<(Vec<(i32, Vec<u16>)>, f64)> {
        let pos: Vec<usize> = rows.iter().map(|r| r.1).collect();
        let mut ord = regime_order(&pos, lim);
        if long_first { let k = ord.iter().take_while(|&&i| pos[i] <= lim).count(); ord.rotate_left(k); }
        let (t, p, sl): (Vec<i32>, Vec<usize>, Vec<usize>) =
            (ord.iter().map(|&i| rows[i].0).collect(), ord.iter().map(|&i| rows[i].1).collect(), ord.iter().map(|&i| rows[i].2).collect());
        let ts = std::time::Instant::now();
        let ids = model.forward_step(sc, &t, &p, &sl, None)?;
        let ms = ts.elapsed().as_secs_f64() * 1e3;
        let lg = model.dev.dtoh_sync_copy(&sc.logits)?;
        let mut out: Vec<(i32, Vec<u16>)> = vec![(0, Vec::new()); rows.len()];
        for (r, &i) in ord.iter().enumerate() { out[i] = (ids[r], lg[r * v..(r + 1) * v].to_vec()); }
        Ok((out, ms))
    };
    let t1 = std::time::Instant::now();
    for step in 0..a.steps {
        // A: the runner alone;  B: the control pair;  D: the joiner alone
        let (ra, _) = step_rows(&mut sc, &[(rt, rp, 0)], false)?;
        let (rb, _) = step_rows(&mut sc, &[(rt, rp, 1), (pt, pp, 2)], false)?;
        let (rd, _) = step_rows(&mut sc, &[(jt, jp, 5)], false)?;
        // C: the mixed step (runner + joiner);  E: two short rows + the joiner
        let (rc, ms_c) = step_rows(&mut sc, &[(rt, rp, 3), (jt, jp, 4)], a.long_first)?;
        let (re, _) = step_rows(&mut sc, &[(rt, rp, 6), (pt, pp, 7), (jt, jp, 8)], a.long_first)?;
        c_ms.push(ms_c);
        let (ia, la) = (vec![ra[0].0], &ra[0].1);
        let (ib, lb) = (vec![rb[0].0, rb[1].0], &rb[0].1);
        let (id, ld) = (vec![rd[0].0], &rd[0].1);
        let (lc, lj) = (&rc[0].1, &rc[1].1);
        // success signal: the logits rows read back are the step's own (their argmax is the returned id)
        let sa = score_row(la, &lut);
        anyhow::ensure!(sa.argmax as i32 == ia[0] && !sa.nonfinite,
                        "step {step}: logits row argmax {} != returned id {} (or non-finite logits)", sa.argmax, ia[0]);
        let (cb, cl, cc) = (cmp_rows(la, lb, &lut), cmp_rows(ld, lj, &lut), cmp_rows(la, lc, &lut));
        ctl.add(&cb);
        long.add(&cl);
        all.add(&cc);
        bucket.add(&cc);
        e_run.add(&cmp_rows(la, &re[0].1, &lut));
        e_par.add(&cmp_rows(&rb[1].1, &re[1].1, &lut));
        e_long.add(&cmp_rows(ld, &re[2].1, &lut));
        let sparse_own = model.qsa_live(rp);
        if sparse_own { above.add(&cc); } else { below.add(&cc); }
        if cc.bits > 0 && first_bit.is_none() { first_bit = Some(step); }
        let mg = margin(la, &lut);
        let arm_am = score_row(lc, &lut).argmax;
        if !cc.agree && flips.len() < 40 {
            flips.push(format!("step {step} pos {rp}: solo {} vs mixed {arm_am}, solo top1-top2 margin {mg:.4}, kl {:.3e}", sa.argmax, cc.kl));
        }
        if let Some(w) = csv.as_mut() {
            writeln!(w, "{step},{rp},{},{},{},{},{},{:.6},{:.4e},{:.4e},{},{},{mg:.5},{arm_am}",
                     sparse_own as u8, sa.argmax, cb.bits, cl.bits, cc.bits, cc.maxabs, cc.rel_l2, cc.kl, cc.agree as u8, cc.top5)?;
        }
        if (step + 1) % 100 == 0 || step + 1 == a.steps {
            out(format!("REGIME bucket steps {}..{} (runner pos ..{rp}): {}", step + 1 - bucket.n, step, bucket.line()));
            bucket = Agg::default();
        }
        gen.push(sa.argmax);
        // teacher forcing: every runner copy takes the solo arm's token; the joiners take the solo joiner's
        rt = ia[0]; rp += 1;
        pt = ib[1]; pp += 1;
        jt = id[0]; jp += 1;
    }
    if let Some(mut w) = csv { w.flush()?; }
    let secs = t1.elapsed().as_secs_f64();
    if a.time_steps > 0 {
        // The cost of a mixed step: the served 4-row step [runner, runner copy, partner, joiner]
        // (each row fed its own argmax), wall time per forward_step, the first 16 steps discarded
        // (graph capture, warm-up). Compare a run at the default with one at --qsa-row-regime 0.
        let (mut t4, mut p4) = ([rt, rt, pt, jt], [rp, rp, pp, jp]);
        let mut ms: Vec<f64> = Vec::new();
        let (mut md_lo, mut md_hi) = (usize::MAX, 0usize);
        for i in 0..a.time_steps + 16 {
            let md = if row_regime_on() { dense_prefix(&p4, lim).0 } else { 0 };
            let ts = std::time::Instant::now();
            let ids = model.forward_step(&mut sc, &t4, &p4, &[0, 1, 2, 4], None)?;
            if i >= 16 { ms.push(ts.elapsed().as_secs_f64() * 1e3); md_lo = md_lo.min(md); md_hi = md_hi.max(md); }
            for r in 0..4 { t4[r] = ids[r]; p4[r] += 1; }
        }
        let mean = ms.iter().sum::<f64>() / ms.len() as f64;
        out(format!("REGIME TIME 4-row mixed step (3 rows at/below {lim} + 1 past it; row regime {}; dense rows {md_lo}..{md_hi}): \
                     n={} median {:.3} ms  mean {mean:.3}  p10 {:.3}  p90 {:.3}  (runner pos {}..{}, joiner pos ..{})",
                    if row_regime_on() { "per row" } else { "per step" }, ms.len(), pct(&mut ms.clone(), 0.5),
                    pct(&mut ms.clone(), 0.1), pct(&mut ms.clone(), 0.9), rp, p4[0], p4[3]));
    }
    if a.steps == 0 { return Ok(()); }
    out(format!("REGIME {} steps in {secs:.1} s; solo runner text head: {:?}", a.steps,
                tok.decode(&gen[..gen.len().min(48)], true).unwrap_or_default()));
    out(format!("REGIME CONTROL  dense pair row 0 vs solo           : {}", ctl.line()));
    out(format!("REGIME LONG     joiner in the mixed step vs solo    : {}", long.line()));
    out(format!("REGIME SHORT    runner in the mixed step vs solo    : {}", all.line()));
    out(format!("REGIME SHORT    .. while the runner is at/below {lim} : {}", below.line()));
    out(format!("REGIME SHORT    .. after the runner passed {lim}      : {}", above.line()));
    out(format!("REGIME 3-ROW    runner in [runner, partner, joiner] vs solo        : {}", e_run.line()));
    out(format!("REGIME 3-ROW    partner in that step vs the partner of the dense pair: {}", e_par.line()));
    out(format!("REGIME 3-ROW    joiner in that step vs solo                         : {}", e_long.line()));
    if c_ms.len() >= 8 {
        // the lazy graph of the 2-row mixed step: call 0 runs eager and captures, call 1 is the first replay
        out(format!("REGIME CAPTURE  2-row mixed step wall ms: first call {:.1} (eager + capture), second {:.1} (first replay), third {:.1}, median of the rest {:.2}",
                    c_ms[0], c_ms[1], c_ms[2], pct(&mut c_ms[3..].to_vec(), 0.5)));
    }
    out(format!("REGIME first step with a bit difference in the mixed-step runner row: {}",
                first_bit.map_or("none".to_string(), |s| s.to_string())));
    for f in &flips { out(format!("REGIME FLIP {f}")); }
    let e_bad = e_run.bit_steps + e_par.bit_steps + e_long.bit_steps;
    let ok = ctl.bit_steps == 0 && long.bit_steps == 0 && all.bit_steps == 0 && e_bad == 0;
    out(format!("RESULT: {} (control {} / long {} / short {} / 3-row step {}+{}+{} of {} steps differ bitwise)",
                if ok { "REGIME_INVARIANT" } else { "REGIME_DEPENDENT" }, ctl.bit_steps, long.bit_steps, all.bit_steps,
                e_run.bit_steps, e_par.bit_steps, e_long.bit_steps, a.steps));
    Ok(())
}
