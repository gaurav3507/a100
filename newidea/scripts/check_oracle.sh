#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ -d results/oracle ]]; then
  COUNT="$(find results/oracle -maxdepth 1 -type f -name '*.json' | wc -l)"
else
  COUNT=0
fi
echo "oracle_cells=$COUNT/30"
if [[ -f results/oracle.pid ]]; then
  ORACLE_PID="$(tr -d '[:space:]' < results/oracle.pid)"
  if [[ "$ORACLE_PID" =~ ^[0-9]+$ ]] && kill -0 "$ORACLE_PID" 2>/dev/null; then
    echo "RUNNING PID=$ORACLE_PID"
  else
    echo "NOT RUNNING stale PID=$ORACLE_PID"
  fi
fi
LATEST_LOG=""
if [[ -d results/logs ]]; then
  LATEST_LOG="$(find results/logs -type f -name 'oracle_*.log' -print | sort | tail -n 1)"
fi
if [[ -n "$LATEST_LOG" ]]; then
  echo "latest_log=$LATEST_LOG"
  tail -n 30 "$LATEST_LOG"
fi
