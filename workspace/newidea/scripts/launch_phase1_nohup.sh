#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PID_FILE="results/phase1.pid"
mkdir -p results/logs
test -z "$(git status --porcelain --untracked-files=no)"
git ls-files --error-unmatch prediction_table.md results/manifest.csv >/dev/null
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
  echo "Refusing launch: HEAD does not match origin/main" >&2
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "Refusing duplicate launch: Phase 1 PID $EXISTING_PID is live" >&2
    exit 1
  fi
  rm -f "$PID_FILE"
fi

COMMIT="$(git rev-parse HEAD)"
EXPECTED_GCARL="$(tr -d '[:space:]' < vendor/GCaRL_COMMIT.txt)"
ACTUAL_GCARL="$(git -C vendor/GCaRL rev-parse HEAD)"
test "$EXPECTED_GCARL" = "$ACTUAL_GCARL"
EXPECTED_NOTEARS="$(tr -d '[:space:]' < vendor/NOTEARS_COMMIT.txt)"
ACTUAL_NOTEARS="$(git -C vendor/notears rev-parse HEAD)"
test "$EXPECTED_NOTEARS" = "$ACTUAL_NOTEARS"
test -f results/manifest.csv
export PYTHONPATH="$PROJECT_DIR/src"
.venv/bin/python -c "from pathlib import Path; from crlcd_audit.manifest import validate_results; print(validate_results(Path('results/manifest.csv'), allow_incomplete=True))"
LOG="results/logs/phase1_$(date -u +%Y%m%dT%H%M%SZ).log"

echo "commit=$COMMIT" > "$LOG"
echo "gcarl_commit=$ACTUAL_GCARL" >> "$LOG"
echo "notears_commit=$ACTUAL_NOTEARS" >> "$LOG"
echo "manifest_rows=$(($(wc -l < results/manifest.csv) - 1))" >> "$LOG"
nohup scripts/run_phase1.sh >> "$LOG" 2>&1 &
NEW_PID="$!"
TMP_PID="${PID_FILE}.tmp.$$"
printf '%s\n' "$NEW_PID" > "$TMP_PID"
mv "$TMP_PID" "$PID_FILE"
echo "PID=$NEW_PID"
echo "log=$LOG"
