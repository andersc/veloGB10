#!/usr/bin/env python3
"""What did this server actually do for the traffic it saw? In plain words, from its own counters.

    scripts/owner_check.py --base http://HOST:9000
    scripts/owner_check.py --base http://HOST:9000 --window-seconds 120   # only what happens in the next 2 minutes
    scripts/owner_check.py --file saved_metrics.txt                       # a saved copy of /metrics

Read-only: one or two GETs of /metrics, nothing else. Flags only, no environment.

Why this exists (2026-10-10): a feature was reported "built, gated, x1.2 faster" and did nothing on the
owner's real setup; the server's own counters said so (0 packed rounds) and nobody had looked at them
there. This prints those counters as sentences, so "did it engage for MY requests" takes seconds to
answer and does not depend on anybody's summary.
"""
import argparse
import re
import sys
import time
import urllib.request

LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf|-Inf)\s*$')
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

# why a request stayed out of a packed round, in plain words (the reasons of src/lane_pack.rs)
SKIP_WORDS = {
    "sampled": "the request uses sampling (temperature above 0) and this boot keeps sampled requests out of packed rounds "
               "(several machines with --lane-pack-sampled-tp off, or an older build)",
    "penalty": "the request carries a repetition / presence / frequency penalty (for example --default-presence-penalty): "
               "penalized requests never join a packed round in this build",
    "few_packable": "it was the only request that could pack at that moment (the others were kept out for another reason)",
    "odd_lane": "an odd number of requests: this one was the odd one out for that round (the role rotates)",
    "prefill": "another request was still reading its prompt (prefill) in that step",
    "mtp_off": "speculation was turned off for this request (it was not paying), so it rode the plain batch",
    "control": "the request owed a control round (the engine re-measures plain speed now and then)",
    "probe": "the request was inside a re-probe window (speculation being tried again after it was turned off)",
    "image": "the request has images",
    "depth0": "no draft fitted in that round (end of the context window or of the token budget)",
    "mixed_regime": "the two requests were on different sides of the long-context attention switch",
    "regime_clamp": "the round was clamped at the long-context attention switch",
    "no_graph": "the packed graph for that shape was not captured at boot",
}


def parse(text):
    out = {}
    for raw in text.splitlines():
        if not raw or raw.startswith("#"):
            continue
        m = LINE.match(raw)
        if not m:
            continue
        labels = tuple(sorted(LABEL.findall(m.group(2) or "")))
        try:
            out[(m.group(1), labels)] = float(m.group(3))
        except ValueError:
            pass
    return out


def fetch(base, timeout):
    url = base.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    with urllib.request.urlopen(url + "/metrics", timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def val(m, name, **labels):
    return m.get((name, tuple(sorted(labels.items()))), 0.0)


def by_label(m, name, label):
    out = {}
    for (n, labels), v in m.items():
        if n == name:
            d = dict(labels)
            if label in d:
                out[d[label]] = out.get(d[label], 0.0) + v
    return out


def diff(a, b):
    """b - a for counters; gauges and build info are taken from b."""
    gauges = ("velogb10_build_info", "velogb10_uptime_seconds", "velogb10_max_batch",
              "velogb10_max_batch_requested", "velogb10_lane_pack_disabled", "velogb10_lane_pack_active",
              "velogb10_engine_alive", "velogb10_requests_running", "velogb10_requests_waiting")
    return {k: (v if k[0] in gauges else v - a.get(k, 0.0)) for k, v in b.items()}


def n(x):
    return f"{int(round(x)):,}"


def pct(part, whole):
    return f"{100.0 * part / whole:.0f}%" if whole > 0 else "-"


def report(m, where, window):
    L = []
    info = next((dict(labels) for (name, labels) in m if name == "velogb10_build_info"), {})
    build = ", ".join(f"{k} {v}" for k, v in sorted(info.items())) or "build info not exposed"
    up = val(m, "velogb10_uptime_seconds")
    L.append(f"Server {where}")
    L.append(f"  {build}; up {up / 60:.0f} min; --max-batch {int(val(m, 'velogb10_max_batch'))}")
    L.append(f"  counted over: {'the last ' + str(window) + ' s' if window else 'everything since the server started'}")
    cap_req, cap = val(m, "velogb10_max_batch_requested"), int(val(m, "velogb10_max_batch"))
    if cap_req > cap:
        L.append(f"  this boot asked for --max-batch {int(cap_req)}: the server serves at most 16 requests at the same "
                 f"time, so it booted with {cap} lanes and further requests wait in the queue (none is rejected)")
    if val(m, "velogb10_engine_alive") == 0 and ("velogb10_engine_alive", ()) in m:
        L.append("  *** THE ENGINE REPORTS ITSELF DEAD (velogb10_engine_alive 0) ***")

    fin = by_label(m, "velogb10_requests_total", "finish")
    done = sum(fin.values())
    toks = val(m, "velogb10_generation_tokens_total")
    L.append("")
    L.append(f"REQUESTS finished: {n(done)}" + (" (" + ", ".join(f"{k} {n(v)}" for k, v in sorted(fin.items()) if v) + ")" if done else "")
             + f"; tokens generated: {n(toks)}")
    if done == 0 and toks == 0:
        L.append("  Nothing was served in this period, so there is nothing to judge. Send your requests, then run this again.")
        return L
    dc, ds = val(m, "velogb10_decode_tokens_per_second_count"), val(m, "velogb10_decode_tokens_per_second_sum")
    if dc > 0:
        L.append(f"  average decode speed of a request: {ds / dc:.1f} tok/s over {n(dc)} requests "
                 f"(each request's own speed; with several at once the total is higher than this)")

    pt, pc = val(m, "velogb10_prompt_tokens_total"), val(m, "velogb10_prompt_tokens_cached_total")
    if pt > 0:
        L.append("")
        L.append("PROMPTS")
        L.append(f"  prompt tokens received: {n(pt)}; taken from the prompt cache instead of being read again: {pct(pc, pt)} ({n(pc)})")
        L.append("  (the cache is per lane: a conversation whose lane is busy or was reused by another one reads its whole prompt again,")
        L.append("   and every running stream stands still while a prompt is read)")
    ge = val(m, "velogb10_graph_cache_entries")
    if ("velogb10_graph_cache_entries", ()) in m:
        L.append("")
        L.append("GPU GRAPHS (pre-recorded launch sequences; a shape without one runs the slow way the first time)")
        L.append(f"  graphs recorded during this period: {n(ge)}" if window else
                 f"  graphs held now: {n(ge)} (boot records the greedy, unpenalized shapes; sampled and penalized ones are recorded during traffic)")

    rounds, emitted = val(m, "velogb10_spec_rounds_total"), val(m, "velogb10_spec_emitted_tokens_total")
    draft, acc = val(m, "velogb10_spec_draft_tokens_total"), val(m, "velogb10_spec_accepted_tokens_total")
    L.append("")
    L.append("SPECULATION (the engine drafts several tokens ahead and keeps the ones that check out; requests still running may not be counted yet)")
    if rounds > 0:
        L.append(f"  speculative rounds: {n(rounds)}; tokens produced per round: {emitted / rounds:.2f} "
                 f"(1.00 would mean speculation gains nothing)")
        L.append(f"  drafted tokens that were accepted: {pct(acc, draft)} ({n(acc)} of {n(draft)})")
        if toks > 0:
            L.append(f"  share of all generated tokens that came out of speculative rounds: {pct(emitted, toks)}")
    else:
        L.append("  NO speculative round ran in this period (speculation off, or every step was a shared plain step).")
    off = val(m, "velogb10_spec_mtp_off_events_total")
    if ("velogb10_spec_mtp_off_events_total", ()) not in m:
        L.append("  speculation switch-off events: NOT EXPORTED by this build (no statement about them)")
    if off > 0:
        L.append(f"  the engine turned speculation OFF for a request {n(off)} time(s) because it was not paying there; "
                 f"tried it again {n(val(m, 'velogb10_spec_mtp_reprobes_total'))} time(s), "
                 f"{n(val(m, 'velogb10_spec_mtp_reprobe_ok_total'))} of them successful")

    arms = by_label(m, "velogb10_lane_pack_rounds_total", "arm")
    total = sum(arms.values())
    # F1/F2/F3 (owner 2026-10-10): the boot's own decision about lane packing, from its gauge.
    lpa_rows = by_label(m, "velogb10_lane_pack_active", "reason")
    if lpa_rows:
        (reason, v), = lpa_rows.items()
        L.append("")
        if v >= 0.5:
            L.append("LANE PACKING IS ACTIVE on this boot (packed rounds can run)")
        elif reason == "--lane-pack off":
            L.append("LANE PACKING IS OFF on this boot (--lane-pack off was given; packing is on by default): every request runs its own round")
        else:
            L.append(f"LANE PACKING IS NOT ACTIVE on this boot (it is on by default): {reason}; requests are served without packed rounds")
    # F3: modules the wide trellis kernel cannot decode (widths outside 3..=5) served through the
    # reconstruct + cuBLASLt arm whatever --exl3-recon-min said (one boot line names it the first time).
    rf = by_label(m, "velogb10_exl3_recon_forced_total", "bits")
    if rf:
        det = ", ".join(f"{b}-bit: {n(v)}" for b, v in sorted(rf.items()))
        L.append(f"  this pack has modules the wide trellis kernel cannot decode; those GEMMs serve through the "
                 f"reconstruct arm (--exl3-recon-min overridden): {det}")
    L.append("")
    L.append("HOW ROUNDS WERE SERVED (every round counted in this period, ALSO the ones run while only one request was busy)")
    if total > 0:
        packed, serial, shared = arms.get("packed", 0), arms.get("serial", 0), arms.get("shared_plain", 0)
        L.append(f"  packed       {n(packed):>9}  {pct(packed, total):>4}   ONE round that served TWO requests (the lane feature); it counts once here")
        L.append(f"  serial       {n(serial):>9}  {pct(serial, total):>4}   one request's own speculative round: the others wait their turn, or it was running alone")
        L.append(f"  shared plain {n(shared):>9}  {pct(shared, total):>4}   all busy requests in one step, speculation OFF for that step")
    else:
        L.append("  (no round counted by arm: an older build, or nothing ran)")

    skips = {k: v for k, v in by_label(m, "velogb10_lane_pack_skip_total", "reason").items() if v > 0}
    L.append("")
    packed = arms.get("packed", 0)
    if packed > 0:
        L.append(f"LANE PACKING RAN: {n(packed)} packed rounds = {n(2 * packed)} request-rounds served two at a time, "
                 f"against {n(arms.get('serial', 0))} served one at a time (that figure includes every round of a request that ran alone, "
                 f"so this is NOT the share of the time two requests overlapped: the server does not count that).")
    else:
        L.append("LANE PACKING DID NOT RUN: 0 packed rounds." + ("" if skips or total else
                 " (Were two requests ever busy at once? Packing needs two busy requests; a single request never packs.)"))
    if val(m, "velogb10_lane_pack_disabled") > 0:
        L.append("  *** the packed arm switched itself OFF for the rest of this boot after an error in a packed round ***")
    if skips:
        L.append("  times a request was kept OUT of a packed round, by reason:")
        for k, v in sorted(skips.items(), key=lambda kv: -kv[1]):
            L.append(f"    {k:<13}{n(v):>8}   {SKIP_WORDS.get(k, '(no description: see src/lane_pack.rs)')}")
    # F7b (2026-10-11): unknown "--"-options ignored at boot (warn, never refuse)
    uo = val(m, "velogb10_unknown_options")
    if ("velogb10_unknown_options", ()) not in m:
        # found 2026-10-12 by the v0.7.4 cut: the builds up to and including the release candidate set the count but
        # never render it, so an absent series must not read as "all clear". The startup warning line is the signal.
        L.append("  unknown-option count: NOT EXPORTED by this build (the 'unknown option' warning lines at startup are the only signal; "
                 "absence of the series is not an all-clear)")
    elif uo == 0:
        L.append("  every flag on the command line was known (0 unknown options ignored)")
    else:
        L.append(f"  unknown options ignored: {n(uo)} - each got one '[opts] unknown option' line at boot (with up to three near names); a misspelt flag used to disappear silently")
    # the tool-call parser's shadow counting: the counters say what the OTHER rules
    # of the parser would have returned for the text it parsed; nothing here changes what is served
    blocks = val(m, "velogb10_toolcall_blocks_total")
    served = next((dict(labels) for (name, labels) in m if name == "velogb10_toolcall_rules_info"), None)
    if served is not None or blocks > 0:
        L.append("")
        L.append("TOOL-CALL PARSER (how the text of tool calls was read; the shadow counters only compare, they change nothing)")
        if served is not None:
            L.append(f"  served rules: values end by --tool-quote-pairing {served.get('pairing', '?')}, "
                     f"blocks end by --tool-block-end {served.get('block', '?')}, "
                     f"unclosed blocks by --tool-unclosed {served.get('unclosed', '?')}"
                     + (" (drop = the v0.7.3 reading, the default: a block the model never closed is dropped whole; "
                        "--tool-unclosed complete returns its closed function elements when the model stopped by itself)"
                        if served.get('unclosed') == 'drop' else ""))
        L.append(f"  <tool_call> blocks read: {n(blocks)}")
        unc = by_label(m, "velogb10_toolcall_unclosed_total", "outcome")
        if unc:
            parts = ", ".join(f"{k}={n(v)}" for k, v in sorted(unc.items()))
            L.append(f"  blocks never closed: {parts} (returned = completed on a self-stop; "
                     f"dropped_cut = self-stop with nothing complete; dropped_length = cut by budget/client/engine)")
        skipped = by_label(m, "velogb10_toolcall_shadow_skipped_total", "reason")
        if sum(skipped.values()) > 0:
            L.append("  the shadow comparison was NOT made or was cut short for some responses: " + ", ".join(
                f"{k} {n(v)}" for k, v in sorted(skipped.items()) if v) + " (size = text over 512 KiB, blocks = more than 64 blocks, "
                "time = the comparison's 100 ms wall-clock budget, spent between two rules); what was served is unaffected")
        declined = by_label(m, "velogb10_toolcall_cost_declined_total", "role")
        if declined.get("served", 0) > 0:
            L.append(f"  the cost model was SERVED and read {n(declined['served'])} block(s) by the v0.7.3 rule because they exceeded its work "
                     f"budget (60,000 work units per block, 200,000 per response, about 4 ms of one core): those blocks got the v0.7.3 reading")
        if declined.get("shadow", 0) > 0:
            L.append(f"  the cost model, read only to compare, skipped {n(declined['shadow'])} block(s) over its work budget "
                     f"(not compared; what was served is unaffected)")
        # how long the thread that parses a response was busy with it (on the EXL3 server that thread serves
        # every stream, /health and /metrics: this is how long they waited)
        edges = sorted((float(dict(l)["le"]), v) for (nm, l), v in m.items()
                       if nm == "velogb10_toolcall_parse_seconds_bucket" and dict(l).get("part") == "served" and dict(l).get("le") != "+Inf")
        for part in ("served", "shadow"):
            cnt = val(m, "velogb10_toolcall_parse_seconds_count", part=part)
            if cnt <= 0:
                continue
            bk = sorted((float(dict(l)["le"]), v) for (nm, l), v in m.items()
                        if nm == "velogb10_toolcall_parse_seconds_bucket" and dict(l).get("part") == part and dict(l).get("le") != "+Inf")
            def quant(q):
                for le, c in bk:
                    if c >= q * cnt:
                        return f"<= {le * 1e3:g} ms"
                return "> " + (f"{bk[-1][0] * 1e3:g} ms" if bk else "?")
            mean = val(m, "velogb10_toolcall_parse_seconds_sum", part=part) / cnt
            worst = val(m, "velogb10_toolcall_parse_seconds_max", part=part)
            L.append(f"  time in the parser, {part}: {n(cnt)} responses, mean {mean * 1e3:.3f} ms, median {quant(0.5)}, 99th {quant(0.99)}, "
                     f"worst {worst * 1e3:.3f} ms" + ("  (the thread that serves every stream waited this long)" if part == "served" else
                     "  (the comparison that is only counted)"))
        unclosed = {}
        for (name, labels), v in m.items():
            if name == "velogb10_toolcall_unclosed_total" and v > 0:
                d = dict(labels)
                unclosed[(d.get("finish"), d.get("ends_like_call"))] = v
        if unclosed:
            stop_call = unclosed.get(("stop", "yes"), 0)
            L.append(f"  responses that ENDED INSIDE a <tool_call> block the model never closed (dropped under every rule): "
                     f"{n(sum(unclosed.values()))}; of those {n(stop_call)} stopped by themselves with text that ends like a finished call "
                     f"(a forgotten close token, or a call drafted inside reasoning)")
        if val(m, "velogb10_toolcall_shadow_failed_total") > 0:
            L.append(f"  *** the shadow comparison panicked {n(val(m, 'velogb10_toolcall_shadow_failed_total'))} time(s) (a defect in a rule that is not served; "
                     f"what was served is unaffected) ***")
        if blocks == 0:
            L.append("  no block was read in this period (no request that offered tools produced a call).")
        else:
            words = {"off": "the v0.7.3 value rule (--tool-quote-pairing off)", "on": "quote pairing (--tool-quote-pairing on)",
                     "cost": "the cost model (--tool-quote-pairing cost)", "block_end": "the other block-end rule (--tool-block-end)"}
            comp = by_label(m, "velogb10_toolcall_shadow_compared_total", "other")
            dis = {}
            for (name, labels), v in m.items():
                if name == "velogb10_toolcall_parse_disagreement_total" and v > 0:
                    d = dict(labels)
                    dis[(d.get("other"), d.get("kind"), d.get("fewer"))] = v
            if comp.get("cost", 0) <= 0 and (served or {}).get("pairing") != "cost":
                L.append("  the cost model was NOT read in the shadow (--tool-shadow-cost off, the default: reading it costs the thread that "
                         "parses the response up to ~4 ms per response); the other rules below were")
            for other in ("off", "on", "cost", "block_end"):
                if comp.get(other, 0) <= 0:
                    continue
                tot = sum(v for (o, _, _), v in dis.items() if o == other)
                L.append(f"  also read under {words[other]}: {n(comp[other])}; it read the text differently in {n(tot)} ({pct(tot, comp[other])})")
                for (o, kind, fewer), v in sorted(dis.items()):
                    if o != other:
                        continue
                    what = {"call_count": "a different NUMBER of calls", "arg_boundary": "the same number of calls, an argument or name differs",
                            "content": "the same calls, the prose differs"}.get(kind, kind)
                    who = {"served": "the SERVED rule returned fewer calls", "other": "the OTHER rule returned fewer calls",
                           "same": "equal number of calls"}.get(fewer, fewer)
                    L.append(f"      {n(v):>7}  {what}; {who}")
            if dis:
                L.append("  A difference means the text was ambiguous under the rules (a value that quotes tool-call syntax). It does not say "
                         "which reading is right; the server log has one content-free line per difference ([toolcall-shadow], block hash and sizes only).")
            else:
                L.append("  every other rule read every block the same way: nothing in this traffic quoted tool-call syntax.")
    for name, words in (("velogb10_graph_capture_invalidated_total",
                         "CUDA graph captures lost to a CUDA call that is illegal during capture (another thread's, or the capturing thread's own; the step ran eagerly)"),
                        ("velogb10_graph_capture_failed_total",
                         "CUDA graph captures that produced no graph (BeginCapture, EndCapture, instantiate or the TP graph audit; the step ran eagerly)"),
                        ("velogb10_requests_rejected_total", "requests refused at admission"),
                        ("velogb10_streams_cancelled_backlog_total", "streams cancelled because the reader stalled")):
        if val(m, name) > 0:
            L.append(f"NOTE: {words}: {n(val(m, name))}")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--base", help="the server, e.g. http://192.0.2.11:9000 (a trailing /v1 is accepted)")
    src.add_argument("--file", help="a saved copy of the server's /metrics text")
    ap.add_argument("--window-seconds", type=int, default=0,
                    help="read the counters twice, this many seconds apart, and report only what happened in between")
    ap.add_argument("--timeout", type=float, default=5.0, help="seconds to wait for /metrics (default 5)")
    a = ap.parse_args()
    try:
        if a.file:
            if a.window_seconds:
                ap.error("--window-seconds needs --base")
            m, where = parse(open(a.file, encoding="utf-8", errors="replace").read()), a.file
        else:
            first = parse(fetch(a.base, a.timeout))
            if a.window_seconds > 0:
                print(f"counting for {a.window_seconds} s: send your requests now ...", file=sys.stderr)
                time.sleep(a.window_seconds)
                first = diff(first, parse(fetch(a.base, a.timeout)))
            m, where = first, a.base
    except OSError as e:
        print(f"cannot read the counters: {e}", file=sys.stderr)
        return 2
    if not any(k[0].startswith("velogb10_") for k in m):
        print("no velogb10_* series found: this is not a veloGB10 /metrics page", file=sys.stderr)
        return 2
    print("\n".join(report(m, where, a.window_seconds)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
