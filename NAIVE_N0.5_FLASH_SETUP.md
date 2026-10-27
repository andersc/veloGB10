# Naive N0.5 Flash EXL3 — step-by-step setup

This guide walks through bringing up **Naive-N0.5-Flash** from an **EXL3 pack** on veloGB10, at
**TP=4 and TP=2**. It covers the exact launch commands, the log lines you should expect, the
measured loading times, and the measured performance.

- **Target model:** two EXL3 packs of the same model — **use either one**, they differ only in
  precision (and therefore in speed and in how much disk they need):
  - **[doth4580/Naive-N0.5-Flash-EXL3-3.5bpw](https://huggingface.co/doth4580/Naive-N0.5-Flash-EXL3-3.5bpw)**
    — 3.5 bpw, ~110.8 GB of artifacts. The examples below use it.
  - **[doth4580/Naive-N0.5-Flash-EXL3-4bpw](https://huggingface.co/doth4580/Naive-N0.5-Flash-EXL3-4bpw)**
    — 4 bpw. Higher fidelity; larger on disk.
- **Drafter:** the FP8 **S8 block-7 drafter** (a small 5-layer draft model, ~1.3 GB) rides next to
  the pack as `Naive-N0.5-Flash-FP8-Draft` and is passed with `--draft-dir`. It ships to every
  node automatically — you never copy it by hand.
- **Full context:** launch with `--max-seq-len 262144` (the model's full 256K).

> **The head ships the weights each node needs — you do not copy the pack to the node machines.**
> The head plans every rank's shard and ships it through the TP blob cache (`~/.cache/gb10_tp`), so
> an EXL3 node receives only its own rank's share — roughly **65.3 GiB at TP=2** and **34.4 GiB at
> TP=4** — rather than the whole pack. A staged decode cache under
> `~/.cache/gb10_tp/naive-load/` then makes every later boot warm (see §4).

---

## 1. One-time preparation

On the head machine, download the pack and the drafter (paths here are examples — any local path
works; the commands below assume both sit side by side):

```bash
huggingface-cli download doth4580/Naive-N0.5-Flash-EXL3-3.5bpw --local-dir /path/to/models/Naive-N0.5-Flash-exl3-3.5bpw-vllm
# plus the FP8 drafter next to it:
#   /path/to/models/Naive-N0.5-Flash-FP8-Draft
```

On **every node machine**, start the resident node supervisor once (it survives reboots of the
head and re-arms itself after every session):

```bash
./gb10_inference --node
```

That is all a node needs — no model files, no config.

---

## 2. Launch commands

### TP=4 (recommended: head + 3 nodes)

On the head:

```bash
./gb10_inference --server \
  --model-dir /path/to/models/Naive-N0.5-Flash-exl3-3.5bpw-vllm \
  --draft-dir  /path/to/models/Naive-N0.5-Flash-FP8-Draft \
  --host 0.0.0.0 --port 9000 \
  --tp 4 --nodes 192.0.2.12,192.0.2.13,192.0.2.14 \
  --max-seq-len 262144 --max-batch 4 --max-tokens 65536 \
  --prefix-cache on --log-level debug
```

### TP=2 (head + 1 node)

```bash
./gb10_inference --server \
  --model-dir /path/to/models/Naive-N0.5-Flash-exl3-3.5bpw-vllm \
  --draft-dir  /path/to/models/Naive-N0.5-Flash-FP8-Draft \
  --host 0.0.0.0 --port 9000 \
  --tp 2 --nodes 192.0.2.12 \
  --max-seq-len 262144 --max-batch 4 --max-tokens 65536 \
  --prefix-cache on --log-level debug
```

Notes:

- **TP=4 and TP=2 shard differently** (tensor-parallel expert slices vs interleaved whole
  experts), so switching topologies re-ships the node cache **once** per topology — the caches
  then coexist; flipping back is warm.
- The API is OpenAI-compatible at `http://<head-ip>:9000/v1` once serving.

---

## 3. What the boot looks like

The head plans, ships and syncs every node (times below are for the first-ever boot):

```
[head] B32 plan rank 1/4 (naive, deal tp): 79 files, 34.39 GiB
...
[head] manifest 'Naive-N0.5-Flash-exl3-3.5bpw-vllm': 237 artifacts, 110.79 GB (B32: per-rank shards through the blob cache)
[head] 192.0.2.12 (rank 1) READY — model at ~/.cache/gb10_tp/models/...@naive-w4-r1-tp (36.93 GB in 371.3s = 0.10 GB/s)
[head] all 3 node(s) synced.
```

Then the data-plane link and the load:

```
[tp] rank 0/4 — link UP
[tp] rank 0/4 — data-plane all-reduce link SANE
[naive-load] cache ~/.cache/gb10_tp/naive-load/Naive-N0.5-Flash-exl3-3.5bpw-vllm-w4r0L48: cold, building (verify on, 4 workers)
[naive] layer 0 (Dsa, dense) loaded, 11.0 s
[naive] layer 1 (Swa, moe) loaded, 42.8 s
...
[naive-load] 48/48 layers staged in 378.0 s (warm 0 / built 48; cache ...)
[naive-serve] S8 drafter loaded from .../Naive-N0.5-Flash-FP8-Draft: 5 layers, taps [1, 7, 14, 20, 26, 32, 39, 45], block 7, SHARDED
[naive-serve] node rank 1 READY
[naive-serve] node rank 2 READY
[naive-serve] node rank 3 READY
```

And the server is up when you see:

```
SERVING family=naive model=doth4580/Naive-N0.5-Flash-EXL3-3.5bpw tp=4 ctx=262144 batch=4 kv=f16 prefix=on spec=dspark:block7 port=9000 url=http://0.0.0.0:9000 boot=388.1s
[naive-serve] decode graph captured in 0.04 s
```

> **A first cold boot is slow — let it run.** Nodes bring their RDMA link up only after the head
> finished shipping every rank, and they **wait patiently** (they log
> `nway connect ...: still retrying (60 s — cold-cache head still shipping?)` once a minute and
> abort only if the head process itself dies). Do not restart anything: the one-time cost is the
> weight ship, and every later boot is much faster.

---

## 4. Loading times (measured)

Measured on the shipping topology (head + nodes, EXL3 3.5 bpw pack, `--log-level debug`, server
up to `/health = 200`):

| Boot | What it pays | TP=4 | TP=2 |
|---|---|---:|---:|
| **First-ever (empty caches)** | ship 3 × ~37 GB to nodes + staged decode of every layer | **~20.5 min** (serving at 1230 s) | ~15 min (ship 70 GB ≈ 7.5 min + staged decode) |
| **Staged-cold** (weights cached, decode cache cold) | EXL3 decode of all 48 layers, 4 workers | **~6.5 min** (boot 388 s; head stages in 378 s, nodes in under a minute each) | ~7 min (boot 877 s, head-staged 418 s after the node re-ship) |
| **Warm** (everything cached) | staged reads + H2D upload only | **~1.8 min** (boot 106 s; 48/48 layers staged warm in 95 s) | **~3.5 min** (boot 210 s; staged warm in 192 s) |

Reading the table:

- The **first cold boot is ship-bound**: 3 × ~37 GB at ~0.10 GB/s sequential is ~19 minutes on
  its own. This is paid exactly once per node per topology.
- The **staged-cold** row is the per-layer EXL3 decode into the `naive-load` cache — also paid
  once; after that the cache holds all 48 decoded layers and the boot is **warm**, essentially
  host-to-device upload time plus link bring-up.
- Switching **TP=4 ⇄ TP=2** re-ships once (different shard plans) and re-stages once; afterwards
  both topologies boot warm.

---

## 5. Performance

Measured with **VeloBenchmark 0.3.0** against the served model, one request at a time, reasoning
effort low, pure-code regime (ANSI C sorting), sampled defaults (no sampler fields sent):

| | TP=4 | TP=2 |
|---|---:|---:|
| Decode median | **82.4 tok/s** | 51.9 tok/s |
| Decode min / max | 43.7 / 93.0 | 44.2 / 79.6 |
| Decode p50 / p90 / p99 | 82.4 / 90.0 / 91.6 | 51.9 / 72.8 / 79.6 |
| TTFT (short prompt) | 0.52 s | 0.58 s |
| Spec acceptance / depth | 77 % / 4.9 | 38 % / 3.3 |
| Stability (sustain / peak) | 98 % | 84 % |
| Output tokens in cell | 3,726 (77 % code, 10 % reasoning) | 439 (cancelled early — a short probe, treat as indicative) |

The TP=4 cell is a full ~50 s pure-code session (single request, 3,726 generated tokens, 91 steps,
27 speculation rounds at 2.00 accepted tokens per round). The TP=2 cell is a single cancelled
9-second request and under-reports the topology's sustained rate; it is listed for orientation
only.

---

## 6. Notes

- **Speculation** is the S8 block drafter (`spec=dspark:block7`): the drafter context is
  per-lane, so a single request in the batch still speculates even while others are served
  (`--naive-lane-ctx on`, default).
- **Prefix cache** (`--prefix-cache on`): a prompt that repeats an earlier one resumes prefill at
  the last chunk boundary (one conversation per KV slot — 4 at `--max-batch 4`).
- **Expert sharding** at TP=4 is tensor-parallel (every rank holds a slice of every expert — no
  per-layer expert-count skew); at TP=2 it is expert-parallel (whole experts per rank). Both are
  automatic; `--naive-moe-shard` overrides.
- The decode step is **CUDA-graph captured** per batch width at boot (the
  `step graph captured: m 1..7` lines), with GPU sampling and head-authoritative tokens over the
  control stream.
