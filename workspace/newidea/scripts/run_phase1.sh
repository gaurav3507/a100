#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

test -x .venv/bin/python
test "$(git -C vendor/GCaRL rev-parse HEAD)" = "$(tr -d '[:space:]' < vendor/GCaRL_COMMIT.txt)"
test "$(git -C vendor/notears rev-parse HEAD)" = "$(tr -d '[:space:]' < vendor/NOTEARS_COMMIT.txt)"
test -f results/manifest.csv
test -f prediction_table.md
PREDICTION_COMMIT="$(git log -1 --format=%H -- prediction_table.md)"
test -n "$PREDICTION_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
export PYTHONPATH="$PROJECT_DIR/src"
.venv/bin/python -c "from pathlib import Path; from crlcd_audit.manifest import validate_results; print(validate_results(Path('results/manifest.csv'), allow_incomplete=True))"
exec .venv/bin/python scripts/run_phase1.py --manifest results/manifest.csv
