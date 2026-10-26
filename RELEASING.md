# RELEASING.md — how veloGB10 public releases are cut

**Rule: this public repository is updated ONLY from stable builds.** Development happens in the
private repo. When a stable build is cut there and is green on every gate, the curated file set
below is synced here, reviewed, committed (the commit message names the stable build), and pushed.
There are no WIP commits on this repo's `main`.

**Second rule: the dev repo is never modified for release cosmetics.** The public tree carries a
small, documented scrub delta (§"Public-side scrub") applied to the synced copies only.

## What this repo contains — the curated set, and only the curated set

- `Cargo.toml`, `Cargo.lock`, `build.rs`
- `src/` (all `*.rs`) + `src/ptx/{gpu_batch,gpu_kernels}.ptx` (checked-in kernels for nvcc-less builds)
- `kernels/*.cu` (**the full set build.rs compiles** — syncing only `gpu_{batch,kernels}.cu` once shipped a public build whose embedded `exl3_bench.ptx` was compiled from a stale `exl3_bench.cu` and failed at module load; found cutting v0.7.5, 2026-10-22), `native/{net_shim.c,tp_doorbell.h}`
- `benches/`
- `scripts/{convert_model.py, quantize_nvfp4.py, derive_gdn4.sh, gate.py, profile.sh, run_tp_node.sh, run_tp_server.sh}`
- `README.md`, `assets/`, `.gitignore`, this file

Deliberately **not** included: model weights or quantized artifacts (distributed separately),
`target/`, `tests/` (their fixtures point at lab-local model paths — a candidate for a later
env-var fixture cleanup), internal working docs (AGENTS.md, HANDOFF.md, ROADMAP.md, session
notes), lab-specific run presets and TP gate scripts, experimental subprojects (`mtp_finetune/`,
`tp_prototypes/`, `quant/`, `ref/`), and **binaries** (they ship as Release assets, see below —
never committed to git history).

## Cutting a source release (runbook)

1. Stable build in the dev repo is green: full gate suite + smoke test.
2. Sync the curated set (from the dev repo root):

   ```bash
   PUB=/path/to/veloGB10   # the public repo working copy
   cp Cargo.toml Cargo.lock build.rs "$PUB/"
   rsync -a --delete --exclude='ptx/' src/ "$PUB/src/"
   rsync -a src/ptx/gpu_batch.ptx src/ptx/gpu_kernels.ptx "$PUB/src/ptx/"
   rsync -a --delete benches/ "$PUB/benches/"
   cp kernels/*.cu "$PUB/kernels/"
   cp native/net_shim.c native/tp_doorbell.h "$PUB/native/"
   cp scripts/{convert_model.py,quantize_nvfp4.py,derive_gdn4.sh,gate.py,profile.sh,run_tp_node.sh,run_tp_server.sh} "$PUB/scripts/"
   ```

   (`rsync -a --delete` on src/benches keeps this repo an exact mirror of the curated set;
   README.md, RELEASING.md, LICENSE and assets/ are maintained here, not synced.)
3. **Public-side scrub** — the dev repo's `src/main.rs` help text/comments carry lab-specific
   examples; the public copy replaces them. Re-apply after every sync (currently 4 lines):
   - `--model-dir /home/user/models/...` → `/models/...` (two help examples)
   - `--nodes <peer-ip>:29500` → `--nodes <peer-ip>:29500` (help example + two code comments)
4. Re-scrub check: `grep -rn "192\.168\.\|/home/" "$PUB" -l` must return nothing.
5. Review: `git -C "$PUB" status && git -C "$PUB" diff --stat`.
6. Commit as `release: <stable-build-sha> — <one-line headline>` and push. Tag to match the
   binary release (`vX.Y.Z`).

## Build-host paths in the binary (v0.7.4)

A release binary must not carry the build host's directories (panic locations of the cargo
registry dependencies, and of our own modules, are absolute paths: `<cargo home>/registry/src/...`
and `<checkout>/src/...`). Stable cargo has no `trim-paths`, and a committed
`--remap-path-prefix=<absolute path>=...` would put that very path into the public tree, so the
public tree carries two small files instead (public side only; the dev source is not changed):

- `.cargo/config.toml`: `[build] rustc-wrapper = "scripts/rustc_remap.sh"` (a path relative to the
  directory that holds `.cargo`);
- `scripts/rustc_remap.sh`: execs the real rustc with one extra `--remap-path-prefix`, derived from
  the directory cargo runs rustc in (no environment variable is read): a crates.io dependency runs
  in `<registry>/src/<index>/<crate>-<version>`, so `<registry>/src/<index>` becomes
  `/cargo/registry`; a workspace crate runs in the checkout root, which becomes empty (`src/x.rs`).

Nothing else about the build changes (same flags, same code; `--remap-path-prefix` does not
enter the binary's behaviour). Check on every release binary, and on the built binary rather than
the tree:

```bash
strings -a target/release/gb10_inference | grep -c -E "/home/|192\.168\.|<host>|/wt/"   # must print 0
```

(`rust_infer` in `"owned_by":"rust_infer"` of `/v1/models` is an API field and `/tmp/...` names
are diagnostic dump files; neither is a build-host path.) A user's own `RUSTC_WRAPPER` (sccache
and the like) overrides the config key: the remap is then absent, which is cosmetic only.

## Binary releases (GitHub Releases)

Prebuilt binaries ship as **Release assets** (the GitHub-native way — git history stays
source-only). Per release, three files:

- `velogb10-vX.Y.Z-gb10-sm121.tar.gz` — `gb10_inference` + `run_tp_{node,server}.sh`.
  Kernels are EMBEDDED in the binary (EMB1): nothing at runtime reads PTX from disk
  (strace-verified 2026-10-26: 420 opens on a boot from a binary-only dir, 0 .ptx).
- `SHA256SUMS.txt` — checksum of the tarball (`sha256sum -c` to verify)
- `PROVENANCE.txt` — build date/toolchain, source stable-build sha, gate status, contents

Cut from the deployed stable build:

```bash
REL=velogb10-releases/vX.Y.Z
cp /path/to/stable_build/gb10_inference "$REL/velogb10/"
cp scripts/run_tp_{node,server}.sh "$REL/velogb10/"
(cd "$REL" && tar czf velogb10-vX.Y.Z-gb10-sm121.tar.gz velogb10 \
   && sha256sum velogb10-vX.Y.Z-gb10-sm121.tar.gz > SHA256SUMS.txt)
# write PROVENANCE.txt, then attach all three to the GitHub Release
```

Upload via the repo's **Releases → Draft a new release** page (or `gh release create` where the
CLI exists). Optional hardening for later: sign `SHA256SUMS.txt` (GPG/cosign) and/or GitHub
artifact attestations once Actions are in use.

## Access

Pushes use a repo-scoped SSH **deploy key** (write access), not a personal account. The key on the
release machine lives at `~/.ssh/velogb10_deploy` with the `github-velogb10` host alias in
`~/.ssh/config`; the remote is `git@github-velogb10:<owner>/veloGB10.git`.
