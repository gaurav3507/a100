#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"

if [[ -f results/phase1.pid ]]; then
  PHASE1_PID="$(tr -d '[:space:]' < results/phase1.pid)"
  if [[ "$PHASE1_PID" =~ ^[0-9]+$ ]] && kill -0 "$PHASE1_PID" 2>/dev/null; then
    echo "RUNNING PID=$PHASE1_PID"
  else
    echo "NOT RUNNING stale PID=$PHASE1_PID"
  fi
else
  echo "NOT RUNNING no PID file"
fi

.venv/bin/python -c "from pathlib import Path; from crlcd_audit.manifest import validate_results; print(validate_results(Path('results/manifest.csv'), allow_incomplete=True))"
LATEST_LOG=""
if [[ -d results/logs ]]; then
  LATEST_LOG="$(find results/logs -type f -name 'phase1_*.log' -print | sort | tail -n 1)"
fi
if [[ -n "$LATEST_LOG" ]]; then
  echo "latest_log=$LATEST_LOG"
  tail -n 20 "$LATEST_LOG"
fi
