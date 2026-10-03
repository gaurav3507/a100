#!/usr/bin/env bash
# Create the A100 home for the ranktest-diagnostics lane.
#
# Idempotent: safe to re-run. Creates directories, writes README.md only if it
# is absent (pass --force to overwrite), and NEVER touches
# /workspace/meridian-identifiability/, which is a separate project.
#
#   bash causalbench/scripts/setup_a100.sh
#   bash causalbench/scripts/setup_a100.sh --force     # rewrite README.md

set -euo pipefail

ROOT="/workspace/ranktest-diagnostics"
FORBIDDEN="/workspace/meridian-identifiability"
FORCE="${1:-}"

# --- guard: this script must never write inside the other project ------------
case "$ROOT" in
  "$FORBIDDEN"|"$FORBIDDEN"/*)
    echo "[fatal] refusing to write inside $FORBIDDEN" >&2; exit 1;;
esac

echo "[setup] target: $ROOT"
mkdir -p "$ROOT/scripts" "$ROOT/results" "$ROOT/logs"

README="$ROOT/README.md"
if [ -e "$README" ] && [ "$FORCE" != "--force" ]; then
  echo "[setup] README.md exists, leaving it alone (use --force to overwrite)"
else
  cat > "$README" <<'MD'
# ranktest-diagnostics

Descriptive profiling and diagnostic gates for the multi-environment CRL
assumption bundle. Descriptives only here: no assumption verdicts.

## Claude Code cannot run on this machine

University policy. Every script in `scripts/` is written on the Mac, pushed to
`origin/ranktest-lane`, pulled here, and run by hand. If something needs
changing, change it on the Mac and pull again. Do not edit in place here, or
the A100 copy and the repo will silently diverge.

## Environment

Use the `cb` venv (numpy / scipy / sklearn / anndata / scanpy):

    source /workspace/venvs/cb/bin/activate

Do NOT use `dvae` (torch, numpy < 2) for anything in this lane.

## Launch convention

There is no tmux on this box. Long runs go through `nohup`, and `python -u` so
the log is not lost to buffering if the job is killed:

    cd /workspace/ranktest-diagnostics
    nohup python -u scripts/85_dataset_descriptives.py --dataset k562 \
        > logs/k562_desc.log 2>&1 &

Check progress with `tail -f logs/<name>.log`.

## Results

`results/INDEX.md` is the index: one row per artefact, with its statistic,
key numbers, commit and status. Quote only rows marked CURRENT.

Every results file carries a mandatory `meta` block recording the statistic,
git commit, timestamp, full config, package versions AND platform. The
platform block exists so a number that moves between this machine and the Mac
can be attributed to the environment rather than mistaken for a finding: the
BLAS backend differs, and SVD-derived quantities depend on it.

To check the two machines agree before trusting any cross-machine comparison:

    python -u scripts/85_dataset_descriptives.py --selftest
    python -u scripts/85_dataset_descriptives.py --overlap-check abide

## Not this project

`/workspace/meridian-identifiability/` is a separate project. Nothing here
writes to it. Its data is read-only input for some loaders, never a target.
MD
  echo "[setup] wrote $README"
fi

# --- confirm the other project was not touched -------------------------------
if [ -d "$FORBIDDEN" ]; then
  echo "[setup] $FORBIDDEN present and untouched (no writes attempted)"
fi

echo "[setup] layout:"
find "$ROOT" -maxdepth 2 -print | sed "s|^$ROOT|  .|"
echo "[setup] done. Activate the venv with: source /workspace/venvs/cb/bin/activate"
