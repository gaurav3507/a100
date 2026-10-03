# Manual A100 runbook

The repository deliberately separates the oracle gate from the transformed grid. Do not launch the grid in the first server session. The prediction table must be generated and committed after all 30 oracle cells and before any transformed result.

## 1. Fetch, verify, and launch the oracle gate

```bash
cd /workspace
test ! -e /workspace/newidea
git clone --recurse-submodules https://github.com/gaurav3507/new-idea.git /workspace/newidea
cd /workspace/newidea
test "$(pwd)" = "/workspace/newidea"
git checkout main
git pull --ff-only origin main
git submodule update --init --recursive
export UV_INSTALL_DIR=/workspace/newidea/.tools
export UV_CACHE_DIR=/workspace/newidea/.uv-cache
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/workspace/newidea/.tools:$PATH
uv --version
scripts/setup_a100.sh
scripts/launch_oracle_nohup.sh
scripts/check_oracle.sh
```

Re-run `scripts/check_oracle.sh` later. Continue only when it reports `oracle_cells=30/30`, the process is no longer live, the log has no traceback, and `results/oracle_summary.json` exists.

## 2. Preregister predictions and manifest

These commands assert that no transformed cell exists, generate the table from the predictions already encoded in the pushed script, and commit the oracle gate and prediction table before the manifest.

```bash
cd /workspace/newidea
export PATH=/workspace/newidea/.tools:$PATH
export UV_CACHE_DIR=/workspace/newidea/.uv-cache
export PYTHONPATH="$PWD/src"
.venv/bin/python scripts/write_prediction_table.py
test -z "$(find results/cells -type f -name '*.json' -print -quit 2>/dev/null)"
git config user.name "Gaurav Goyal"
git config user.email "gaurav3507@users.noreply.github.com"
git add prediction_table.md results/oracle results/oracle_summary.json
git commit -m "Record oracle gate and preregister compatibility predictions"
git push origin main
git fetch origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
PREDICTION_COMMIT="$(git rev-parse HEAD)"
.venv/bin/python scripts/build_manifest.py
git add results/manifest.csv
git commit -m "Freeze Phase 1 transformed-grid manifest"
git push origin main
git fetch origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
echo "prediction_table_commit=$PREDICTION_COMMIT"
```

If `build_manifest.py` reports no eligible methods, stop. The oracle gate has globally blocked a meaningful transformed audit.

## 3. Launch and monitor the full grid

```bash
cd /workspace/newidea
scripts/launch_phase1_nohup.sh
scripts/check_phase1.sh
```

The launcher refuses a duplicate live PID, verifies both pinned scientific submodules, requires a committed prediction table and manifest, records the current commit and manifest row count, and writes an atomic PID file.

## 4. Validate and summarize after completion

```bash
cd /workspace/newidea
export PYTHONPATH="$PWD/src"
.venv/bin/python -c "from pathlib import Path; from crlcd_audit.manifest import validate_results; print(validate_results(Path('results/manifest.csv')))"
.venv/bin/python scripts/summarize_phase1.py
git add results/phase1_summary.csv results/phase1_summary.json
git commit -m "Record Phase 1 compatibility summary"
git push origin main
```

Do not add ignored per-cell JSON, cached latent arrays, or logs. Review observed-vs-nominal mismatches against the fixed materiality thresholds before writing a survival or kill conclusion.
