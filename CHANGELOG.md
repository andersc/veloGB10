# Changelog

High-level release notes for veloGB10. Minor bug fixes and small optimizations are grouped under
generic language where they aren't individually notable.

## v0.7.4 — speculation re-arms inside a request, per-row attention regime, vision-tower recovery, five community pull requests

This release fixes two defects of the EXL3 (Qwen3.8-Flash-Next) serving path that were found
after v0.7.3: speculative decoding could switch itself off for the rest of a long request, and a
request's greedy text could depend on what else was being served. It turns a GPU vision-tower
error from an outage into a per-request event (issue #20), ports five community pull requests
(#15, #16, #17, #21, #23), and closes a list of small defects. One new default changes output
bytes in a narrow case: see "Behaviour change" under the per-row attention regime.

<<CUT: release artifacts line: tarball name, binary sha256, release-source commit, dev commit>>

### Fixed — speculation could stay off for the rest of a long request

On the EXL3 server a per-request control compares the cost of speculative (MTP) rounds with
plain decoding and turns speculation off for a request when it stops paying. Two things were
wrong with it. The decision used a 2 % margin over a window of about 16 rounds, against a
plain-decoding sample that is refreshed only once every 128 rounds (or 1,024). And it was
final: nothing turned speculation back on for that request. In an engine log shared by a user,
a request that had averaged 18.3 ms per token with speculation was switched off at round 250 on
a short stretch measuring 27.84 ms per token against a plain sample of 27.22; the remaining
1,640 of its 2,592 tokens (63 %) ran at plain speed. A second user's log shows the same after
161 rounds (24.58 against 23.35 ms per token). Only generations of 64 rounds or more could be
affected.

What changed:

- The cost is averaged over about 64 rounds and the margin is 10 %.
- Before switching off, the engine takes a fresh plain-decoding measurement at the current
  context, and switches off only if the margin still holds.
- A request that was switched off keeps its draft head in step, and speculation is tried again
  for a short probe (at most 32 rounds; a probe that is not paying fails after 16) once 256
  plain steps have passed. Each failed probe doubles the wait, up to 4,096 steps; a successful
  probe resets it.
- `--mtp-reprobe <auto|off>` (default `auto`). `off` restores the previous one-way behaviour
  exactly.

Measured (3.05 bpw pack):

- Eight long sampled requests of the kind in the first log (up to 3,200 tokens each,
  temperature 0.7 / top_p 0.8 / top_k 20, one machine): 8 of 8 served with 0 off events; the
  six requests served with speculation ran 2.57–4.28 tokens per round at 10.3–15.4 ms per
  token.
- A workload where speculation does not pay (temperature 1.3, 8 × 3,000 tokens): speculation
  is switched off (for example after 207 rounds, 28.95 against 26.19 ms per token at 1.10
  tokens per round), a probe follows after 256 plain steps and fails, and the waits double:
  512, 1,024, 2,048. The same sequence runs on two machines (TP=2: 7 off events, 12 re-arms,
  11 failed probes and 1 successful one over 8 requests, all served), with the same lines in
  both ranks' logs.
- Greedy output is unchanged: six recorded identity cells (one machine and TP=2) give their
  recorded hashes while a test drill forces the off / re-arm cycle continuously (the head log
  of one TP=2 run holds 416 off events and 631 re-probes, with both ranks in step throughout).
- Cost when speculation is healthy: none measurable. Same binary with `auto` and with `off`,
  alternating boots, one request of 256 tokens: code 84.5 / 83.7 against 84.5 / 84.3 tok/s,
  prose 60.0 / 59.2 against 59.9 / 59.9 tok/s (inside the ±1 % run-to-run band).
- Cost of keeping the draft head in step while off: requests that ran almost entirely plain
  measured a median of 26.45 ms per token with it and 26.58 without, a difference below the
  request-to-request spread (about 4 ms per token).
- Upper bound of the probing cost on a request where speculation never pays (arithmetic, not a
  measurement): 32 of 256 steps (12.5 %) for one period after the first failed probe, falling
  to 32 of 4,096 (0.78 %) once the wait has reached its cap.

What does not change: when draft acceptance stays low for a whole request, the probes fail and
the request still runs mostly at plain-decoding speed. The fix is for a temporary trough that
used to cost the rest of the request.

Observability: the per-request `[mtp-stats]` log line now ends with
`off_events=N reprobes=N reprobe_ok=N`; `/metrics` adds `velogb10_spec_mtp_off_events_total`
(events, not requests: one request can go off several times), `velogb10_spec_mtp_reprobes_total`
and `velogb10_spec_mtp_reprobe_ok_total`; the log prints `MTP re-probe (slot N): back ON after
N plain steps`, `MTP re-probe FAILED (...next probe in N plain steps)` and `MTP re-probe OK`
next to the existing `MTP off for this request` line.

### Fixed — a request's greedy text could depend on what else was being served

A decode step chose dense or sparse attention once for all of its rows, from the longest
request in the step. The sparse path starts at 2,052 tokens of context on Flash-Next, so a
request at or below 2,051 tokens that shared a batched step with a longer one was computed by
the sparse kernels: the same attention, through a different rounding path. A second site of
the same rule: a speculative round whose rows straddled the 2,051-token switch.

Measured before the fix:

- Two running requests (short prompts, 1,500 generated tokens) keep their texts when they run
  alone; with a request of about 20,000 tokens joining, both texts changed in 3 of 3 runs. It
  reproduces on the released v0.7.3 binary.
- Two requests with prompts of about 1,500 tokens, each crossing the switch while generating,
  were not stable from run to run even without a third request (two different texts in three
  runs for one of them).
- Size of the effect: after one step the logits differ by one 16-bit-float unit (relative L2
  1.4e-3). Under teacher forcing the difference settles at a KL divergence of about 5e-4 to
  8e-4 nats with no trend over 1,600 positions, and the top-1 token differs at 1 to 1.5 % of
  positions, every one of them a near-tie. It is harmless for quality and real for
  reproducibility: a free-running greedy text departs at its first near-tie.

What changed (host side only; no kernel changed):

- In a mixed step the rows at or below the switch keep the dense kernels they run alone, and
  the scheduler orders such rows first.
- A speculative round stops at the switch instead of straddling it (a few shorter rounds, once
  per request that crosses 2,052 tokens).
- `--qsa-row-regime <on|off>` (default `on`). `0` restores the previous rule.

Measured on the fix (3.05 bpw pack unless stated):

- Bit level (`--probe-exl3-regime`, a new diagnostic): a short row inside a mixed step is
  bit-identical to the same row alone. 0 of 1,600 steps differ (before: 1,600 of 1,600), in
  two-row and in three-row mixed steps.
- Served: with a ~20,000-token request joining, 0 of 5 runs change a running request's text.
  The two crossing requests give the same texts alone, with a joiner, and with speculation off.
- Texts that were stable keep their bytes: the recorded identity cells are unchanged on one
  machine and at TP=2, and the requests' alone texts equal those of v0.7.3.
- Also checked on the release candidate: the 4.05 bpw pack's identity cell; 16 sampled requests
  crossing the switch, alone and next to long requests (16 of 16 served, no error);
  `--lane-order fcfs` (same texts alone and joined); TP=2 with and without speculation (same
  texts alone and joined, both ranks in step).
- Cost: about 0.8 % of aggregate tokens/s with four concurrent requests of mixed length (63.7
  to 63.2 tok/s, two boots per side); nothing at one request (61.8 / 61.5 against 61.9 / 61.7)
  or with four short requests (91.4 / 90.9 against 91.0 / 91.2). The first step of a new
  (rows, short rows) shape costs about 8 ms once per server lifetime (40.8 ms against 33.05 ms
  for a replay).

**Behaviour change.** With the default (`--qsa-row-regime` on), two kinds of greedy request
produce different bytes than on v0.7.3: a request whose context crosses about 2,051 tokens
while it generates, and a short request that used to share a decode step with a longer one.
The v0.7.3 bytes of those requests depended on timing and on the other requests in flight;
the v0.7.4 bytes do not. Requests that never come near the switch and never shared a step
with a longer request keep their bytes. `--qsa-row-regime 0` restores the old rule for
comparison.

New gauge: `velogb10_graph_cache_mixed_step_entries` (mixed-step CUDA graphs captured since
boot; they are captured on first use, one per step shape seen, and are also counted in
`velogb10_graph_cache_entries`). Found while testing pull request #23.

### Fixed — a GPU vision-tower error disabled images until restart (issue #20, pull request #16)

Reported by [@liorm0505](https://github.com/liorm0505) (issue #20); fix contributed by
[@seanlinmt](https://github.com/seanlinmt) (pull request #16), ported into this tree.

A cuBLAS error inside the GPU vision tower was a panic. The panic poisoned the tower's lock,
and every later image request failed until the server was restarted. Reproduced before the
port at TP=2 with two and three simultaneous images: 19 of 20 and 30 of 30 requests failed
with HTTP 400.

Now:

- Tower errors (matrix multiply, attention, readback, synchronisation) are returned as errors.
  The request is then served by the CPU tower, and the next request uses the GPU tower again.
- A poisoned lock is recovered, and a slow CPU fallback does not hold the lock: other image
  requests keep using the GPU tower meanwhile.
- The tower's cuBLAS handle is pinned to the tower's own stream and its per-image scratch is
  released (memory flat over 100 images: 69,869 to 69,903 MiB).
- A failed image preparation is an HTTP 400 with its cause, never a 200 without the image.

Measured: 202 image requests on one machine (100 one at a time, 102 three at a time), 0
failures; at TP=2, 100 requests in groups of two and three, 0 failures. With a failure
injected by a test drill, the request is answered by the CPU tower with the same text
(53.8 s on one machine) and the following requests return to the GPU tower (274 ms, then
104 ms). On the release candidate at TP=2: 1,138 image and 1,145 text requests over six
boots, no wrong answer, both ranks in step, six injected failures all answered by the CPU
tower, and 360 other image requests served by the GPU tower while those fallbacks ran.

Limits: failures of other kinds inside the tower (a kernel launch, a pool allocation, the
weight upload) still panic; they are caught by the worker's panic handler and the lock
recovery, loudly. A GPU fault that the CPU tower cannot cover is reported as HTTP 400 with the
error text, not as a 5xx. Two findings of the TP=2 run are under "Known limits".

### Fixed — EXL3 served windows above the trained length (pull request #21)

Reported and fixed by [@ccgauvin94](https://github.com/ccgauvin94) (pull request #21), ported
into this tree.

The EXL3 server did not clamp `--max-seq-len`. Above the trained window (262,144 tokens on
Flash-Next) a scratch buffer of the sparse attention kept its trained-window size while the
scorer wrote one block per served position, so above 262,147 a speculative round could write
past it (at `--max-seq-len 524288` every round of five or more rows did, from the first one).
v0.7.3 is affected. The defect is shown by the code path and its index arithmetic: in a 96-token probe
at a 524,288 window the v0.7.3 binary showed no visible symptom, and the later CUDA
illegal-address error described in the pull request was not reproduced here.

Now: the buffer follows the served window; the EXL3 server clamps `--max-seq-len` to the
trained window and prints the limit (`--max-seq-len N exceeds the model's trained window
262144 (max_position_embeddings) - clamping to 262144.`); the bit-packed launch arguments are
checked in release builds; the loader refuses a window they cannot carry.

Measured: a boot with `--max-seq-len 524288` clamps and serves; a request of 255,656 prompt
tokens plus 2,048 generated ends at position 257,704 of 262,144 with a correct answer (234 s);
the recorded identity cells are unchanged; at TP=2 both ranks clamp to the same window.

Behaviour changes: (1) an EXL3 `--max-seq-len` above the trained window is now clamped where
it used to boot at the requested size. (2) The same checks are active on the NVFP4 path, where
`--rope-yarn-factor 2 --max-seq-len 524288` used to overflow a packed field silently and now
stops at the first attention launch with `bs_packed field overflow … lower --max-seq-len`.
This second case was not run on a GPU.

### Fixed — TP network bring-up (pull request #15) and TP image payload order (pull request #17)

Contributed by [@seanlinmt](https://github.com/seanlinmt) (pull requests #15 and #17), ported
into this tree.

- **RoCE GID index.** The TP data plane used a fixed GID index (3). It is now detected: the
  first IPv4-mapped RoCE v2 GID in sysfs, printed at bring-up with the sysfs evidence
  (`[tp] gid_idx=3 auto-detected on <device> (sysfs type 'RoCE v2', gid ...)`); if none is
  found the old fixed value is used. `--gid-idx <N>` sets it explicitly for every rail; a
  wrong N fails at bind and is never re-scanned.
- **Bring-up.** The CUDA device is selected explicitly in both bring-up paths, and a failed
  pinned-memory allocation prints the CUDA error text and code.
- **Image payloads under TP.** All image payloads of a scheduler step are read right after the
  step frame, in admission order, before any prefill runs (a prefill uses the same control
  stream).
- **`--ple-headroom-per-lane-gb <F>`** (opt-in, default 0): extra headroom for the
  `--ple-ram auto` decision, per additional lane on top of the fixed 16 GiB (for example 2
  with `--max-batch 10` gives 16 + 2 × 9 = 34 GiB). The default reproduces the previous
  decision exactly.

Measured at TP=2: auto-detection picks index 3 on both ranks of the test pair (the value the
fixed setting used), the recorded identity cells are unchanged, a boot with `--gid-idx 3`
behaves the same, and after the head process is killed the node re-arms by itself (4 s) and
serves the next head. On the release candidate the node re-armed across six consecutive head
sessions, and 1,117 steps shipped image payloads (21 of them with two or more) with every
answer naming its own image.

Not ported: the environment-variable form of the GID option in #15 (the engine takes flags
only); `--served-model-name` from #15 and the telemetry wiring from #17 (both already in
v0.7.3); the 2 GiB-per-lane default of #17 (the option is there, the default stays 0); and a
transport-side fallback scan for the GID, which was removed in review so that the transport
binds exactly the index the engine resolved.

### Added — `--prefill-interleave <N>` (EXL3, one machine, default 0; pull request #23)

Contributed by [@adriangrassi](https://github.com/adriangrassi) (pull request #23), ported
into this tree.

A long prompt that joins while other requests are generating used to freeze those streams for
its whole prefill. With `--prefill-interleave N` the server runs up to N decode steps for the
running requests after each prefill chunk of the joining prompt except the last.

Measured (two running streams, a joining prompt of about 20,000 tokens, one machine):

| `--prefill-chunk` | interleave | longest stall of the running streams | joining request's time to first token |
|---|---|---|---|
| 4095 (default) | 0 | 14.1 s | 14.1 s |
| 4095 (default) | 1 | 5.4 s | 14.5 s |
| 2048 | 0 | 15.1 s | 15.1 s |
| 2048 | 1 | 2.7 s | 15.6 s |
| 1024 | 0 | 16.6–17.0 s | 16.75 s (mean of 3) |
| 1024 | 1 | 1.35–1.47 s | 18.21 s (mean of 3, +8.7 %) |
| 1024 | 2 | 1.35–1.55 s | 19.57 s (mean of 3, +17 %) |

The last chunk is never interleaved and the tail of a prompt is merged into it, so the longest
stall is the largest chunk of the grid: at the default chunk width the flag alone only brings
the stall to 5.4 s. The useful settings combine it with a smaller `--prefill-chunk`, which
prefills more slowly and changes the output of long prompts (as that flag is documented to
do). The prefill rate itself is not slowed (1,210 against 1,203 tok/s in the server's prefill
line); the added time to first token is the decode steps run for the other requests.

Identity: with the per-row attention regime of this release, the running requests' greedy
texts are the same in all nine joined runs and equal to their texts alone, and each joining
request's text is the same at interleave 0, 1 and 2 (three joining documents, chunk 1024, one
machine). Stress: 60 minutes at interleave 1 with long prompts joining, being cancelled during
prefill and during generation, and re-sent (334 joining requests): 0 server errors, 0 panics.

The option is ignored under TP, with a line on every rank
(`--prefill-interleave N IGNORED under TP (single-box only)`); with the flag set, TP=2 output
equals the recorded hashes. The server's prefill line excludes the interleaved decode time and
a new line reports it (`prefill-interleave slot=N: ... scheduler step(s) ... ms of decode`).
Not tested with the flag on: an image request as the joining request, and `--lane-order fcfs`.
The NVFP4 server does not read this flag (it has its own `--prefill-sched`).

### Changed

- **`--mtp off` is honoured on the EXL3 server.** It used to be ignored there (the boot banner
  kept saying `MTP chain-verify ON`). With `--mtp off` the banner reads `MTP chain-verify OFF:
  MTP OFF (--mtp off)`, no draft head runs, and the `[mtp-stats]` lines show `rounds=0
  drafted=0`. `--exl3-no-mtp` still works. On the EXL3 server `on` and `auto` mean the same
  (speculation on, with the per-request control); an invalid value is refused at startup.
- **`--help` says what the servers do.** `--max-tokens` is the default for a chat request that
  omits `max_tokens`, not a cap: unset, the EXL3 server has no limit and the NVFP4 server uses
  8192. `--default-presence-penalty` is 0.0 on the EXL3 server (1.5, or 2.0 on 2B, on NVFP4).
  The sampler defaults are stated per endpoint (see "Known limits"). `--lane-order` carries the
  measured numbers. Thirteen lines of `--help` that printed mis-encoded characters are fixed.
- **`--mtp-lanes` and `--prefill-sched` are listed in `--help`.** Both are NVFP4-server flags
  that were parsed but listed nowhere; they now refuse an unknown value at startup (before, an
  unknown value silently meant the default). `--prefill-sched` defaults to `cursor`.
- **TP launcher (`run_tp_server.sh`).** `--tp` now defaults to 1 + the number of `--node`
  flags, and an explicit `--tp` that disagrees with that count is an error (exit 2). Two
  `--node` flags are refused, because the engine serves a world of 2 or 4 only.
- **`[gen]` log line.** Three fields are appended: `stop_kind` (`eos`, `stop_string`, `length`,
  `loop`, `cancel`, `error`, `other`), and for a request that ends on a stop token `stop_p` and
  `stop_rank`: the probability the sampler gave that token and its rank among the logits (for a
  greedy request, the softmax probability at temperature 1). The cost is one read of a logits
  row, once per request, on its last step, and only for a stop-token finish. The two fields
  read `na` under TP, on the NVFP4 server, when the first generated token is the stop token,
  and for every other kind of finish. A request that runs out of tokens inside its thinking
  block is now counted as reasoning tokens in this line (it was counted as content).
- **Streaming `/v1/completions` sends a usage chunk** when the request carries
  `stream_options: {"include_usage": true}`: empty `choices`, top-level `usage` with
  `prompt_tokens_details.cached_tokens`, after the finish chunk and before `[DONE]`. Without
  the option the stream is byte-identical to before. (The chat endpoint sends its usage chunk
  unless told not to; this endpoint sends it only when asked.)
- **Benchmark and probe rounds** (`--bench-mtp` and two diagnostics) stop at the 2,051-token
  switch like the server, so a losslessness run across the switch measures what is served.

### Investigated — issue #18: no engine defect found

Reported by [@hatemismail](https://github.com/hatemismail) (issue #18), with a complete
reproducer: on one 33K-token request with three tool definitions, the model's reasoning ran to
the 8,192-token budget on veloGB10 while a reference stack finished in about 1K tokens.

Result, on the reporter's request, the 4.05 bpw pack, temperature 1.0 / top_p 0.95 / top_k 20:

| prompt fed as token ids | veloGB10 v0.7.3 | ExLlamaV3 1.4.7, same pack |
|---|---|---|
| the request as veloGB10 and Hugging Face transformers render it (33,415 tokens) | 25 of 35 requests reach the budget | 7 of 10 |
| the same request with the tool definitions serialised as the reference stack does (33,445 tokens, reconstructed from the reporter's description) | 0 of 15 | 0 of 10 |
| the same three tool definitions in reverse order (33,415 tokens) | 0 of 10 | 0 of 10 |

On identical token ids a second runtime on the same weights behaves as veloGB10 does, on each
rendering. No engine setting moved the rate on the original rendering (KV cache precision,
speculation on and off, the sampling rule, sparse against dense attention). The difference the
reporter saw is reproduced by the difference in how the two stacks serialise the tool
definitions; veloGB10 passes the client's tool JSON through the model's chat template
unchanged. Of ten renderings of the tools block tried (the original included), six reach the
budget often (4 to 8 of 10 requests), three almost never do (0 of 15, 0 of 10, 1 of 10) and
one is in between (3 of 10). Why these few tokens matter is not known, and no mechanism is
claimed. Not measured: the reporter's
reference weights (NVFP4) fed the original rendering. The issue stays open for that
measurement. The Flash-Next guide has a short note on this behaviour.

### Verification on two machines (TP=2)

- **The soak owed from v0.7.3.** The published v0.7.3 binary, in the configuration of the
  issue #10 report (one request at a time, ~16K-token sampled prompts, a client cancel every
  eighth request): 203,129 pre-verify rounds in 159.8 minutes, 0 aborts, 0 overwritten slots,
  the round counters equal on both ranks, 368 completed requests and 52 client cancels with 0
  request errors, memory flat.
- **The v0.7.4 release candidate** (a development build of the same sources; the release
  binary is built from the public tree and gated again): the six recorded TP=2 identity cells
  give their recorded hashes; the same configuration as the soak above for 45.5 minutes:
  58,057 pre-verify rounds, 0 aborts, 0 request errors; the speculation re-probe, the image
  robustness run, the prefill-interleave identity and the attention-regime checks are reported
  in their sections above.
- **No drift across releases.** The published v0.7.2 and v0.7.3 binaries and the candidate
  produce the same per-request texts in the 1-to-8-concurrent identity cell (`IDENT_ALL
  44fe40977a57e9d9`), at every concurrency level.
- **What the TP=2 checks do not show.** At default flags with three busy requests the TP=2
  scheduler kept serial speculation, so those runs exercise the round stop at the switch and
  not the mixed step; the mixed step at TP=2 was exercised with speculation off. More busy
  requests at `--max-batch 16` were not tried.

### Known limits

- **A GPU vision-tower failure when an image request coincides with a CUDA graph capture**
  (owner decision pending: dropped if F1 is fixed before the cut). Seen once in 1,138 image
  requests at TP=2, ten seconds after a boot, while the scheduler was capturing graphs for
  newly used step shapes. The request was answered correctly by the CPU tower; nothing else
  was affected. The CUDA error involved exists only while a graph is being captured, which is
  mostly in the first seconds after a boot and later whenever a new step shape is first used.
  The mechanism was read from the code and is not confirmed by an experiment.
- **The CPU-tower answer is slow at TP=2** (owner decision pending: dropped if F1 is fixed
  before the cut). 173 to 218 s for one 448 × 448 image on the head (53.8 s on one machine).
  Other requests and other images are not slowed, but a client with a 60 to 120 s timeout
  gives up before the answer arrives.
- **Not covered by the image verification:** image traffic on one machine (TP=1) against graph
  capture, a continuous image hour without restarts, and TP=4.
- **Sampler defaults differ between the two endpoints.** When a request omits the fields:
  `/v1/chat/completions` on the EXL3 server uses temperature 1.0 / top_p 0.95 / top_k 20 when
  the prompt opens a thinking block and 0.7 / 0.8 / 20 otherwise; on the NVFP4 server it uses
  0.7 / 0.8 / 20 in both cases. `/v1/completions` uses 0.7 / 0.8 / 20 on both servers. The
  penalties follow the `--default-*-penalty` flags on the chat endpoint (EXL3: 1.0 / 0.0 /
  0.0; NVFP4: 1.0 / 1.5, or 2.0 on 2B / 0.0) and are fixed at 1.0 / 0.0 / 0.0 on
  `/v1/completions`, which ignores those flags. `max_tokens`: the chat endpoint follows
  `--max-tokens` (EXL3 unlimited when unset, NVFP4 8192); `/v1/completions` uses 16 and ignores
  the flag. Greedy decoding needs an explicit `temperature: 0` on either endpoint. The chat
  values are constants in the engine; they are not read from the pack's
  `generation_config.json`.
- **A non-streaming chat request that ends inside its thinking block** returns that text in
  `content` with an empty `reasoning_content`; the streaming path sends the same text as
  `reasoning_content`.
- **The tuner's speculative rounds are not stopped at the 2,051-token switch.** The server and
  the benchmark rounds are; the four rounds inside the autotune harness are not.
- **`--prefill-interleave` at the default chunk width** leaves a stall of about 5.4 s for a
  20K-token joining prompt (see its section); it is ignored under TP.
- **TP agreement check, width field.** The per-step agreement word that the TP ranks compare
  carries the row count of a step in 4 bits, so a 15-row and a 16-row step are not
  distinguished by it. This predates v0.7.3 and no failure has been attributed to it; the
  wider field is planned and is not in this release.
- **Still open from earlier releases:** the legacy minimal DSV4 server does not report the
  newer usage fields; two and three concurrent requests on the EXL3 path still share the
  single-request aggregate rate (the packed multi-request verify is not wired into serving).

### Not in this release

YaRN for the EXL3 path (issue #19, pull request #22): planned; it builds on the window fix of
#21, which is in this release. KV pool occupancy gauges on `/metrics` (issue #24): planned for
the release after this one.

## v0.7.3 — reported-issue fixes: TP=2 pre-verify race, exllamav3 1.5.x pack loading, cached-token accounting; tool-call parsing; long-running hardening

This release closes the three open issues from the v0.7.2 feedback (#8, #9, #10), makes the
server tell the truth about cached prompt tokens, stops a stalled log reader (or a slow OTLP
receiver, or a client that stops reading a stream) from stalling the engine, and adds the
operational surface (gauges, admission bounds, an operations guide) for running the server for
hours at a time. The two tarball launcher scripts changed interface: they take flags now, not
environment variables.

### Fixed — TP=2 `TP pre-verify FAILED` (issue #10)

Reported by [@JashicTM](https://github.com/JashicTM) (issue #10).

At world size 2, the host-side control exchange staged its RDMA frames into the same ring the
device doorbell epochs use: when the exchange's slot counter happened to alias the slot holding
the last draft-pass epoch payload, the first bytes of that payload were overwritten and the
pre-verify hash compare fired — the `TP pre-verify FAILED` abort reported against
v0.7.0–v0.7.2, typically hours into a run. World sizes above 2 already used dedicated control
rings; world 2 now does too, which closes the whole aliasing class (not just the pre-verify
instance). The bug was confirmed with a deterministic fault-injection probe that fails on
v0.7.2 and passes on v0.7.3; outputs are bit-identical to the previous build
(`--exl3-tp-ident` hashes unchanged at TP=2 and TP=1), and decode speed is unchanged within
measurement noise. The `--tp-dh-shard 0` workaround is no longer needed. Cost: ~11.5 MB of
pinned host memory per rank. Verified on the release candidate: the fault-injection probe logs
0 clobbered slots over 142 aligned rounds (v0.7.2: 14), and a 30-minute stress checkpoint
(c=1, ~16K-prompt sampled requests, temperature 1) logged 38,273 pre-verify rounds with 0 aborts
and 0 clobbered slots on both ranks — that configuration runs ~1,270 rounds/min, so the longer
(~2 h / ~200,000-round) soak runs after this release and was not waited for. TP=2 output identity:
IDENT_ALL 871f083d58a54281 (code) / 38458d4ce6310c6f (prose) at 10/12/16 lanes, with r16 on,
`--exl3-r16 off` and `--router-coal 0` alike; decode medians unchanged (c=1 82.07 vs 82.10 tok/s,
4 concurrent 127.57 vs 127.60).

### Fixed — packs from exllamav3 1.5.x with the sharded n-gram sidecar (issue #9)

Reported by [@MushroomMan321](https://github.com/MushroomMan321) (issue #9).

exllamav3 1.5.x writes the n-gram table as 128 shard tensors in the sidecar, and most public
Flash-Next EXL3 packs now use that layout; some of them also omit the shards from
`model.safetensors.index.json`. v0.7.2 refused those packs at boot (`ple shard 0 not in
index`). v0.7.3 loads both layouts as shipped — single-tensor and 128-shard, K=5 and K=6,
index-listed or not (the loader falls back to the sidecar's own safetensors header and
validates the layout; a boot line names the one in use). No flags, no workarounds: the
header-rewrite workaround some users applied is no longer needed. Packs that loaded before are
byte-identical before and after (gated). A tested-packs table is in the Flash-Next setup guide.

### Server surface (issue #8)

Reported by [@liorm0505](https://github.com/liorm0505) (issue #8; the same report was first filed as #7 by [@liorzivsensors-cmd](https://github.com/liorzivsensors-cmd)).

- `/v1/models` and `/v1/models/{id}` now report `max_model_len` = the served `--max-seq-len`.
- `usage.prompt_tokens_details.cached_tokens` — the number of prompt tokens served from the
  prefix cache (prompt-end hit or checkpoint resume, whichever is deeper) — on the chat
  non-streaming response, the streaming usage chunk (`stream_options.include_usage`), and the
  legacy non-streaming `/v1/completions` response; EXL3 and NVFP4 paths alike. The matching
  counter is `velogb10_prompt_tokens_cached_total`.
- `--model-name` / `--served-model-name` are honored on the EXL3 server (they used to be
  silently ignored there); the resolution order is `--model-name` > `--served-model-name` >
  the model card's `base_model:` > the directory name. Model ids containing `/` work.
- Known gaps, stated plainly: the minimal DSV4 server does not report the new fields, and the
  legacy **streaming** `/v1/completions` path emits no usage chunk.

Release artifacts: `velogb10-v0.7.3-gb10-sm121.tar.gz` + `SHA256SUMS.txt` + `PROVENANCE.txt`
(binary sha256 `949cab4758ae8af04565a826f28290c4f4280b179d95c93257361c6e09af706f`; built from this
tree at the release-source commit `d7529bd` with a clean checkout; dev repo `rel/cutb` @ `3a4cab1`).

### Tool calls: literal `</parameter>` inside a value

A parameter value that itself contains the text `</parameter>` (an agent writing a file *about*
the tool format) used to be silently truncated at that point. The parser now ends a value only
at the *structural* close — the `</parameter>` followed by the next `<parameter=`, by the
`</function>` that closes the call, or by the end of the call block. Two residual ambiguities
are inherent to the format and documented in the Flash-Next guide, together with advice for
agent authors (prefer a diff-style edit tool, or break the closing sequence when quoting
tool-call syntax).

### `--keep-tools-when-tool-choice-none` (opt-in, default off)

vLLM-compatible handling of `tool_choice: "none"`: the tool definitions stay in the prompt
(byte-identical tools block, so the prefix cache and checkpoints survive a mid-conversation
summary/compaction turn), a do-not-call instruction is appended, and the reply comes back as
plain content. In the replayed reporter trace (309-message conversation, ~89K prompt tokens),
the compaction request and the following turn went from three full ~80 s re-prefills to
7.9 s and 1.3 s. The model still *sees* the tools and is merely told not to call them — a
stray call returns as visible text. The default stays off (byte-identical legacy behaviour);
`--max-batch 2` remains the no-flag alternative. Details in the Flash-Next guide.

### Telemetry on Flash-Next

`--otel-endpoint` now works on the EXL3 (Flash-Next) server — it was silently ignored there.
The `--otel-*` flags are shared by both servers; an unusable endpoint is refused before the
model load. Under TP, only the head emits (nodes have no HTTP hooks). The sender is fully
async: a slow or absent receiver delays nothing but itself.

### Long-running hardening

- Log lines no longer go straight to a locked stdout: a bounded non-blocking queue (4096
  lines, drop-and-count, one recovery summary) means a stalled log reader can no longer stall
  the scheduler (`velogb10_log_lines_dropped_total` counts the losses).
- `--max-waiting` (default 256): 503 + `Retry-After: 5` instead of an unbounded admit queue.
- `--stream-backlog-events` (default 65536 ≈ 11 min at 100 tok/s): a stream whose consumer
  stopped reading is cancelled like a client disconnect.
- New gauges (RSS, threads, fds, memory headroom, scheduler progress/busy/age, graph-cache
  entries, rejected requests, cancelled streams, dropped log lines, cached prompt tokens) and
  exit-path log flushing, plus `=`-form flag parsing on both servers.
- New `docs/OPERATIONS.md`: supervisor rationale, systemd units, log handling, the gauge and
  alert table (alert on `scheduler_busy == 1 AND age > 120` only), exit codes, soak watch-list.
- `--exit-on-fatal` is now the default on the EXL3 server (it was opt-in): on a sticky CUDA
  error or a scheduler-thread panic the in-flight request finishes with an error and the process
  exits 70 for a supervisor to restart, instead of staying up as a DEAD engine answering 503.
  `--exit-on-fatal off` restores the old behaviour; the systemd examples in docs/OPERATIONS.md
  assume the default. The NVFP4 server is unchanged (it never had the option).

### Kernels

New in v0.7.3: fast paths for 9–16-row router folds and HC mixes (v0.7.2 ran the legacy
kernel chain at those widths), live at default flags: bit-identical output, 8/8 identity suite.
In isolation the kernels measure hc ~7.2 → ~6 ms and router ~4.8 → ~1.3 ms at 10 rows. Measured
end to end on one box (TP=1, alternating boots): decode at c=1 is unchanged (91.2/90.1/92.0 vs
91.7/91.4/91.3 tok/s, inside the 1.9 tok/s run-to-run spread), and aggregate throughput at
`--max-batch 16` with 10 concurrent requests measures 145.4/148.8 vs 139.3/141.2 tok/s
(2 reps per side, +4.4 % / +5.4 %, beyond the spread) — the expected effect of the row-batched
router/HC kernels on that workload; no claim is made for TP ≥ 2 (expert-parallel ranks keep
the legacy router at 9–16 rows) or for other workloads.

### `--lane-order fcfs` / `--lane-quantum` (opt-in; default `rr` unchanged)

For several concurrent requests in the serial-speculation arm: `rr` (today's behaviour) gives
every busy lane a round per scheduler step — n requests each run at 1/n speed and all finish
late; `fcfs` runs one lane to completion at a time (first-come-first-served, bounded by
`--lane-quantum`, default 256 generated tokens). Aggregate throughput is unchanged. Measured:
with equal-length turns at TP=1, mean completion time drops 23% at 2 and 30% at 3 concurrent
requests; two 512-token turns at `--lane-quantum 1024` finish 20–24% sooner on the mean
(8.2–8.5 s vs 10.4–10.8 s). The cost is the later lanes' time-to-first-token, and with mixed
lengths `rr` wins the mean (4.24 s vs 4.72 s at quantum 256).
Greedy output is byte-identical in both orders; TP=2 output identity verified.

### Launcher interface changed (tarball scripts)

`run_tp_server.sh` and `run_tp_node.sh` took `MODEL_DIR=`, `PORT=`, `NODE=`… environment
variables and carried stale defaults. They are plain flag scripts now:
`run_tp_server.sh --model-dir DIR --node IP:29500 [--port 9000] [--max-seq-len N]
[--max-batch N] [--tp 2] [-- <engine flags>]` and `run_tp_node.sh [--port 29500]
[--rdma-dev DEV]`; `--help` and `--dry-run` on both, missing required flags fail loudly, and
the server always binds `--host 0.0.0.0`. The old environment overrides are gone.

### Not in this release

The packed multi-request speculative verify (two and three concurrent requests on the EXL3
path) is still probe-only: it exists as hidden `--probe-exl3-pack*` / `--pack3-*` diagnostics,
is not wired into serving, and is not a supported feature. Two and three concurrent requests
still share the single-request aggregate rate.

## v0.7.2 — concurrent requests on the EXL3 path, Qwen3.8-Flash-Next 4.05 bpw, `/metrics`, `--host`, and reported-issue fixes

This release answers a community benchmark of Flash-Next as a coding-agent executor (an apt test of
the engine), adds the higher-fidelity 4.05 bpw pack, and fixes what was found along
the way. It adds a **fourth PTX file** to the deploy set (`src/ptx/exl3_bench_k6.ptx`); copy the whole
`src/ptx/` directory as before — the release tarball already has it.

### Concurrent requests on the EXL3 path (Qwen3.8-Flash-Next)

Speculative (MTP) rounds run one request at a time, so with several busy requests the aggregate used
to stay flat at the single-request rate. Now each scheduling round picks the better of two arms by
estimated aggregate tokens/s: serial speculation, or **one shared batched step for every busy request**
(`--spec-lanes-max auto`, the default; `--spec-lanes-max N` shares the step only above N busy requests,
`0` keeps serial speculation). A single busy request always speculates exactly as before, and every
request's greedy output is byte-identical to running it alone, at every topology.

Aggregate generated tokens/s on a code workload (256-token requests, one lab box per node, untuned
build, indicative absolute values), v0.7.1 behaviour → v0.7.2, at 4 / 8 / 16 busy requests:

| | 4 requests | 8 requests | 16 requests |
|---|---:|---:|---:|
| TP=1 | 91 → 97 | 89 → 129 | 88 → 168 |
| TP=2 | 130 → 135 | 125 → 202 | 124 → 249 |
| TP=4 | 159 → 173 | 152 → 238 | 154 → 293 |

Prose (lower draft acceptance) gains earlier: at TP=1, 4 requests 67 → 89 and 8 requests 68 → 119. The
TP=2 and TP=4 figures are single passes.

**Two and three concurrent requests are not faster yet:** their aggregate stays at the single-request
rate (each request gets its turn). A packed multi-request speculative verify for that range is in
development and is the next item on the list.

- Under the hood: a K-chunked expert gate/up for 10–16-row calls, bit-identical to the paths it
  replaces (plain batched decode at 10–16 requests is ~20–25% faster); single-request speed and output
  are unchanged.
- Fixed on the way: at TP=2 with `--max-batch 16` the server could stop with `TP pre-verify FAILED ...`
  once several requests were busy (the two ranks could hold different sets of captured CUDA graphs).

### New: Qwen3.8-Flash-Next at 4.05 bpw (EXL3)

[`doth4580/Qwen3.8-Flash-Next-EXL3-4.05bpw`](https://huggingface.co/doth4580/Qwen3.8-Flash-Next-EXL3-4.05bpw)
(turboderp's `4.05bpw_h6_ng6`: 4-bit experts, 6-bit dense layers and head, a 6-bit n-gram table;
~108 GB) serves at **TP=1** (the n-gram table is read from SSD — see below), **TP=2 and TP=4**, with
image input. Untuned smoke-test figures (1,000 generated tokens / 2K-token prefill, single request):
TP=1 63 tok/s decode and 1,399 prefill; TP=2 89 and 2,612; TP=4 120 and 3,496 (the 3.05 pack measures
~73 / 102 / 122 decode on the same test). The pack was checked against the exllamav3 reference
implementation itself: every 4-, 5- and 6-bit weight tensor class (15.8M trellis blocks, including the
`lm_head`) and the 6-bit n-gram path decode bit-identically to exllamav3's. The model directory needs
`vision_tower_bf16.safetensors` for image input (it is in the Hugging Face repo).

### Memory: the n-gram table can live on SSD

`--ple-ram ram|ssd|auto` (default `auto`; `on`/`off` still work). `ssd` reads the n-gram table from
disk through the page cache and frees its 30–39 GB for requests and context: identical output, decode
~1–4% slower and cold prefill up to ~10% slower (measured on a PCIe Gen 5 drive). `auto` now decides
**after** the boot has allocated everything else (RAM if the table plus 16 GiB stay free, otherwise
SSD, logged with the reason), which fixes `--max-batch 8` at 131K being stopped by the memory watchdog
at boot on one Spark. A configuration that cannot fit at all is refused **before** the load, with the
arithmetic and the fix (`--ple-ram ssd`, a smaller `--max-batch` / `--max-seq-len`, or `--tp 2`).

### New flags and endpoints

- `--host <addr>` — the bind address of the HTTP API (default `0.0.0.0`; `127.0.0.1` = this machine only).
- `GET /metrics` — Prometheus text format, on by default, no measurable cost: requests by finish
  reason, prompt/generated tokens, requests running/waiting, time-to-first-token / request-duration /
  prefill and decode-rate histograms, speculation totals, engine alive, build info.

### Fixed

- **Tool-call history** is rendered the way the model writes it: `arguments` that are a JSON object are
  parsed before the chat template runs, even for templates with an `arguments is string` branch (they
  used to render `{"command": ...}` inside `<function=...>`, a mixed XML/JSON history the model copied
  in long sessions; issue #6, reported by [@JashicTM](https://github.com/JashicTM)).
- **Tool-call parsing:** text inside a parameter value that merely looks like a function tag
  (`<function=x>` or a bare `function=x>` line, for instance a file an agent is writing about this
  format) is payload and no longer produces a second, phantom tool call.
- **NVFP4 with DFlash2 or DSpark and `--max-batch 2`:** the scheduler could stop with
  `df2 ring nprev 512 != lane pos 713` when a second request arrived while the first was decoding (the
  drafter ring is shared). A request that loses the ring now continues on MTP.
- **NVFP4 `--kv-cache k8v8` now serves any `--max-batch`** (it was refused above one request). Plain
  batched decode runs the same attention kernel a lone request runs, so a request's text does not
  depend on how many others are busy; tree verification and `--mtp-lanes` use new int8 readers. KV
  compaction after a tree-verified step on k8v8 used the bf16 copy kernel on int8 data; fixed.
- A missing `--model-dir` is a clear error instead of a panic; TP shard shipping includes sidecar
  files the tensor index omits.

### Known limits

- Two and three concurrent requests on the EXL3 path do not scale yet (above).
- On the NVFP4 path, above one request the lanes run as a plain batch (the MTP head holds one request's
  state): two requests together are slower in aggregate than one speculating request, and a request
  that shared a batched step does not speculate again for the rest of that request.
- Seeded sampled requests are not byte-reproducible across runs. Video and audio parts return `400`.
- 4.05 bpw is untuned (no autotune decisions were found for it; it decodes ~0.86× as fast as 3.05 on
  one or two nodes).

## v0.7.1 — TP=4 on the EXL3 path, image input, automatic shard shipping

A fast follow to v0.7.0: the EXL3 path gains a fourth node and images, the node sync becomes
shard-level, and two defaults change that alter long-context output bytes.

### TP=4 on the EXL3 path

`--tp 4` now serves Qwen3.8-Flash-Next from EXL3 packs across three peer nodes plus the head:
attention, GDN and the KV cache split by head, the routed experts dealt across the ranks, the
vocab-parallel LM head, sharded MTP-head screening, a per-step lockstep agreement check and a
watchdog. Measured on pure code: **~221 tok/s decode** (min/max 173/240, p50/p90/p99 221/234/237) and
**~3.2K tok/s prefill** — TTFT 0.64 s at ~2.1K input tokens, 10.9 s at ~34.9K. Against a single GB10
that is **×1.6 decode and ×2.0–2.2 prefill**.

### Image input, on every topology

Images are served end to end on the EXL3 path at TP=1, TP=2 and TP=4: the pack's **original bf16
vision tower** is loaded (`vision_tower_bf16.safetensors`), each image is resized so its longer side
is at most `--image-max-edge` (default 1024, aspect preserved), 3-axis mrope is applied at all six
main-attention RoPE sites **and** the QSA indexer — so images work past the 2,051-token dense window —
and image rows ship to the TP nodes as a digest-checked binary payload. Parts are returned in client
order. **Video and audio parts return `400`** for now; video is planned.

### Automatic shard shipping

A multi-node run no longer needs the model copied to the peer machines. The head plans each rank's
shard and ships only that, through the TP blob cache: **~57.5 GiB per node at TP=2** and **~46.8 GiB
at TP=4** for the EXL3 pack, and 66.0 GiB for NVFP4 (was 98.5 replicated). TP=2 output is
token-identical to the previous path for both formats, and the live TP=4 bank is identical.
`--tp-sync-only` runs a node that syncs and exits.

### Default changes (output-changing on long context)

- **`--prefill-chunk 4095`**, up from 2048 — a large part of the TP=4 prefill gain. Ragged-tail
  folding is always on; `--prefill-chunk 2048` restores the previous grid.
- **`--qsa-key-rope full`** — the EXL3 indexer had left rotary dims 32..63 of every pooled key at
  zero since the sparse-attention work, while the reference fills them. `half` keeps the old bytes
  for A/B comparison only.

Plus minor bug fixes and optimizations, including a clear error instead of a panic when `--model-dir`
does not exist.

## v0.7.0 — EXL3 packs (Qwen3.8-Flash-Next), TP=2 serving, CLI-only options

A large release: a second quantization format with its own model family, two-node serving as a
shipped mode, and one breaking change to how the engine is configured.

### New: EXL3 packs are served directly

`--model-dir` now accepts an **EXL3 (ExLlamaV3 trellis)** pack alongside the NVFP4/FP8 families.
The model shipped on this path is **Qwen3.8-Flash-Next**: a 125B MoE with ~6B active parameters,
a Gated DeltaNet + Qwen Sparse Attention hybrid, 512 experts (top-10), a 51B n-gram embedding
table, an MTP draft head, and a 262,144-token context.

- The weights are published as
  **[doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw](https://huggingface.co/doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw)**
  — EXL3 3.05 bpw, about 85 GB including the 32.6 GB n-gram table.
- Native MTP speculative decoding is built in: greedy output is bitwise identical to
  non-speculative decoding, and sampled output is distribution-exact.
- **TP=1 and TP=2 only.** Any other world size exits with `--tp 2 only`.
- **Text only — no vision on the EXL3 path** (image requests are rejected). Vision remains on the
  NVFP4 / Qwen3.8 27B path added in v0.6.0.
- q8 KV by default, prefix cache with prefill checkpoints, penalties, streaming, loop detection.

### TP=2

TP=2 ships as a served mode: sequence-parallel prefill, a vocab-parallel LM head, and prefill
communication overlap — each on by default with an `off` value. Measured on two GB10 (DGX Spark)
over ConnectX-7, one request at a time, on the same engine code as this release but not on the
release binary itself; decode includes MTP speculation:

| | TP=2 | vs TP=1 |
|---|---|---|
| Decode, greedy (C code / Python / prose) | ~192 / ~160 / ~110 tok/s | ×1.38–1.42 |
| Decode, thinking-on, sampled | ~95 tok/s | ×1.36 |
| Prefill (2K / 32K / 128K / 256K) | 2,485 / 2,376 / 2,279 / 2,138 tok/s | ×1.56–1.69 |

### CLI-1: every option is a command-line flag (breaking)

- **The engine reads no environment variables.** Leaving a `GB10_*` variable set now refuses
  startup and names the replacement flag — for example `GB10_TP_GRAPH` suggests `--tp-graph`.
- `--print-config` prints every option's resolved value for a given command line; `--help-diag`
  lists the diagnostics and test drills.
- Migration table for every removed variable: **[docs/ENV_TO_FLAGS.md](docs/ENV_TO_FLAGS.md)**.
- Under TP every option is set on the head; the head ships its resolved options to the node, and a
  node's own command line carries only the per-box options.

### Speed

Prefill and decode levers on both paths, all bitwise: a pipelined expert kernel, a tensor-core GDN
chunk scan, hyper-connection fusion, router coalescing, sparse-attention selection, MoE gate/up
folding, and shared-expert overlap. On the EXL3 path, prefill TTFT is down roughly 27–30% against
the previous release. Autotune tables are now stamped with a build id and validated at load, so a
stale table cannot be applied to a changed build. NVFP4 W4A4 prefill kernels were contributed via
community PR #4 by [@kedric](https://github.com/kedric).

### Reliability

- **A node that fails during boot now fails the head loudly** instead of leaving it hanging
  (PACK-FIX). The EXL3 TP pack-manifest check was removed by owner decision — each box loads its
  own pack at the path the head names.
- Autotune's preflight no longer requires specific machine addresses; it checks only that the GPU
  is idle (no other engine, no resident compute applications).
- Streaming loop detection, and CPU-affinity pinning by default.

Plus minor bug fixes and optimizations.

## v0.6.1 — `response_format` served again (quality-regression hotfix)

- **`response_format` requests are served, not rejected.** v0.6.0's P13 W2 change turned any
  `response_format` request into a loud HTTP 400; that was a hard failure of requests that used to
  succeed and cost measured quality on the public benchmark (the hardmode leg fell 150/176 →
  85/100 because json_schema scenarios returned `[server error 400]` instead of a verdict). Such a
  request is now served again, and the truth is advertised instead of hidden: every reply for a
  schema request carries an **`x-json-schema-enforced: none`** header, and the reason is logged
  loudly once per distinct reason. Nothing is silently ignored, and nothing that used to work is
  refused. (Schema *enforcement* remains off — it is not yet proven, and the switch is documented as
  gated on evidence.)
- **JSON-schema FSM fix.** An INTEGER prefix that can never return into range is now refused at the
  digit rather than at termination (previously an unbounded digit run after `maximum` was allowed).
  Deliberately narrow: `Ty::Integer` only, so a float case like `70e-1 == 7` is never rejected.
- Phase-B decode-mask arming + schema-blind-lane routing groundwork (inert while no lane carries a
  schema); 4 new tests. Minor bug fixes and optimizations.

## v0.6.0 — DFlash v1 drafter lane, DSpark, tool-render parity, TP/FP8 correctness

A large release: a new drafter serving lane, a new draft-model family, reference-exact tool
rendering, and a broad sweep of TP/FP8/losslessness correctness fixes.

### Drafters & speculation
- **DFlash v1 drafter lane** (`--spec-source dflash`) — the v1 drafter is now config-driven and
  wired into the API serving path, including under **TP=2/TP=4** (SPMD lane on every rank + artifact
  shipping). The draft round's attention was DRAM-bound on ~144× redundant KV traffic; a tiled
  DFlash attention kernel fixes it.
- **DSpark drafter support** — new kernel module `gpu_dspark` and a native DSpark round, with a
  signed ring-identity guard for the prefix carry.
- **DF2 block-16** — runtime `--df2-block 8|16`. Measured at TP=4: **+23.03% pooled / +9.47%
  row-medians** on the mission prompt set (GO on that workload; NO-GO on the recipe's own payload).
  Opt-in — the default stays block 8.
- **DF2 carry across a prefix-cache hit** — keeps DFlash2 in the draft seat across a cache hit
  (flag-gated, default off; measured slower at TP=4).
- **One artifact flag for every drafter**: `--draft-dir`; `--spec-source` is the only selector.

### Tool calling & template rendering
- **Reference-exact tool rendering** — `serde_json` and `minijinja` now preserve key insertion order
  (`preserve_order`). Without it the `<tools>` block was re-sorted and diverged from the reference
  (transformers/vLLM) at byte 83 on every tool-using turn (−345/−1,346 tokens on 12/52 tools).
- JSON-schema handling made explicit (loud floor) and tool-call serialization reworked.

### Serving & scheduler
- **Two-lane prefill cursor** (`--prefill-sched <inline|cursor>`) — the prefill window loop moved out
  of `admit()` into the step loop.
- Scheduler fixes: a silent request drop in the TP head path, and slot exhaustion now fails loudly.
- k8v8 multi-lane fast-fail; thinking toggle; `/health` telemetry.

### TP correctness
- **MTP head kept replicated under TP** — F6 sharding broke the draft chain (a TP=2/TP=4 serving
  regression).
- **F9 fix**: warm prefix-reuse stream divergence (TP2 FP8 27B).
- Batch-invariant verify-attention partition under TP + a binv probe.
- Shard block-128 FP8 (`W::Fp8Blk`) on the in-place path.
- TP-mode GDN state probe (`StateTp`) + logits/extent probe fixes for the sharded path.

### FP8 & losslessness
- **pf8 prefill OOB class fixed** — the full-attention / batch-mixer prefill-width W8A8 lane output
  buffers are padded to tp64 (FIX #2, #2b, #3, #4).
- **`ignore_eos` / `min_new` honored on every spec-path emit loop** (12 sites) — the stop rule is a
  property of the request, not the serving path (fixes the d2/d4-vs-d0 early-stop ladder failures).
- Depth-2 attention key partition keyed on the request constant; residual depth-2 verify losslessness
  diagnostics.

### Safety & observability
- **Release-live tripwire** — a pool/OOB tripwire with `--pool-census` and `--tripwire-selftest`.
- **`DISPATCH_ASSERT`** — the wide-verify dispatch rule as a machine-checked test.
- New `tel` module + telemetry reporting (e.g. dflash2-tree as its own `/health` mode).

### Launchers
- TP=4 production launcher (world=4) + a DRYRUN-asserting test; an NVFP4 TP=2 production launcher
  with a boot-identity line on both lanes. Minor bug fixes and optimizations.

## v0.5.5 — FP8 prefill levers on, prompt-truncation + max_tokens fixes

- **FP8 prefill levers now on by default.** The tensor-core flash-attention prefill
  (`GB10_FA_PREFILL`) and the tensor-core chunked GDN scan (`GB10_GDN_CHUNK2`) are now default-on for
  the FP8 path (value-checked; `=0` restores the legacy path). Big prefill speedup on the FP8
  Qwen3.8 27B configuration. NVFP4/MXF4/other-model paths are untouched.
- **`truncate_prompt_tokens` on `/v1/chat/completions`.** vLLM's left-truncation field now works on
  the chat path (it previously only worked on `/v1/tokenize`, so an over-length chat couldn't be
  rescued). Keeps the LAST `n` prompt tokens instead of the over-length 400.
- **`max_tokens`-omitted fix.** The scheduler now clamps before rejecting instead of rejecting when
  `prompt_len + max_new + depth + 8 > kv_stride` — a request that omits `max_tokens` no longer
  terminates with 0 tokens. Minor bug fixes and optimizations.

## v0.5.4 — Built-in OpenTelemetry, FP8 support, DFlash2 tree mode

- **Built-in OpenTelemetry.** `--otel-endpoint <URL>` streams OTLP/HTTP-JSON generation telemetry
  (the actual SSE chunk bytes, with `model.id` / `topology` / `request.id` / `token.index` / `event`
  attributes), off by default, near-zero decode interference. `request.id` is now the conversation
  key so a reply's turns join one continuous session; `generation.id` stays per-POST. Companion
  flags: `--otel-batch-size`, `--otel-batch-interval-ms`, `--otel-include-tokens`,
  `--otel-model-id`, `--otel-topology`.
- **FP8 support expanded.** Direct load of Qwen fine-grained block-128 FP8 (`weight_scale_inv`);
  DFlash2 FP8 drafter bake (`--df2-bake-fp8`) plus weight-only NVFP4 (`--df2-bake-nvfp4`);
  `--df2-quant-fidelity` gate tool. Fixed FP8 kernel bugs across `gpu_batch.cu`.
- **DFlash2 tree verification.** New opt-in `--spec-source dflash2-tree` mode (additive; MTP and
  DFlash2 unchanged). Minor bug fixes and optimizations.

## v0.5.2 — vLLM-compatible tokenize / detokenize endpoints

- **`POST /v1/tokenize`** — vLLM-compatible tokenization: `{tokens, count, max_model_len}`, a pure
  tokenizer call (no forward / KV / GPU). Accepts a `prompt` string or a chat `messages` array; the
  `messages` mode renders exactly as the chat path so its count equals `usage.prompt_tokens`. Empty
  prompt → `{tokens: [], count: 0}`; over-length → `400 context_length_exceeded`.
- **`POST /v1/detokenize`** — vLLM-compatible decode half of the pair (`{model, prompt}`) for
  exact-N prompt building.
- Added so our engine can be benchmarked more correctly. Minor bug fixes and optimizations.

## v0.5.1 — Vision robustness, reasoning-effort, graceful-load fixes

- **Vision generalization + boot fix.** The GPU vision tower now bootstraps opportunistically: a
  non-vision or geometry-incompatible model serves text-only instead of crashing at startup (fixes a
  v0.5.0 boot crash on non-27B packs; issue #2, reported by [@npw1980](https://github.com/npw1980)). Vision is generalized across the Qwen3.5/3.8 VL family, so
  all vision-tower models serve images.
- **OpenAI `reasoning_effort`.** Full level table (`none/low/medium/high/xhigh/max`) with
  per-family normalization, plus `--reasoning-effort`; the `high` mapping no longer silently drops
  thinking (regression fix).
- **Tool-call + reasoning-mode fix.** Tool-call markup is held back in reasoning mode too, fixing a
  first-call double-emit leak.
- **Graceful model-load exit.** Corrupted / stale / wrong-format checkpoints exit with a clear
  actionable message instead of a panic/OOM/core-dump.
- **`--output-prompts [n]`** — human-readable chat-request logging; `--vision-cpu` now listed in
  `--help`. Minor bug fixes and optimizations.

## v0.5.0 — Vision support

- **Vision support.** Image input is now supported end-to-end on a GPU vision tower
  (`gpu_vision` kernels), with a `--vision-cpu` escape hatch to the CPU reference path. PNG/JPEG/WebP/GIF
  decoding added. The engine now ships and requires the `gpu_vision.ptx` kernel artifact in addition
  to the existing PTX set.
- **Better tool-call support.** A single canonical serializer now handles streaming and
  non-streaming tool-call output identically, repairs malformed tool-call tags, and no longer drops
  or leaks text around tool-call boundaries. New tool-call compliance and serializer test suites.
- **Prefill/TTFT optimizations.** New opt-in prefill levers (tensor-core flash-attention prefill,
  v2 W4A4 prefill GEMM, GDN tensor-core chunked scan), all env-gated **default off**, so the default
  serving path is unchanged. Minor bug fixes and optimizations.
- **Model-id fix.** `/v1/models` and responses now report the model card's `base_model`
  (e.g. `Qwen/Qwen3.8-27B`) instead of a local directory fragment. `--model-name` still overrides.

## v0.4.2

- Fix: accept OpenAI multipart `content` (string | array | null) to unblock agent clients that send
  content parts; request-schema only.

## v0.4.1

- Fix: `--draft-dir` is now mandatory only when `--spec-source` explicitly names a DFlash2 mode;
  plain-MTP launches no longer require it.

## v0.4.0

- **Qwen3.8 27B NVFP4** support with native **DFlash 2** speculative decoding, full 256K context.
- **TP=4** serving (plus TP=2 and single-node).
- New DSV4 / DFlash2 / DSpark / MXFP4 kernel set.
- README Update section with the Qwen3.8 27B performance table and live throughput traces; new
  `QWEN_27B_SETUP.md` and `MANAGING_CACHE.md` docs.

## v0.3.1

- **KAT-Coder** model support; supported-models table in the README.

## v0.3.0

- README generalization and load-pipeline features. Minor bug fixes and optimizations.

## v0.2.0

- **Tencent Hy3 (hy_v3)** family support, 4-bit KV cache, FR-Spec draft head, model-name family fix.

## v0.1.0

- Initial public release: from-scratch Rust + CUDA engine for Qwen3.5/3.6 on single and TP=2 GB10,
  with NVFP4/FP8 quantization, MTP speculative decoding, an OpenAI-compatible server, and prebuilt
  release binaries.
