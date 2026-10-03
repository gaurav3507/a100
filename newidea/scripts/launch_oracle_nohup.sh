#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PID_FILE="results/oracle.pid"
mkdir -p results/logs
test -z "$(git status --porcelain --untracked-files=no)"
if [[ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]]; then
  echo "Refusing launch: HEAD does not match origin/main" >&2
  exit 1
fi
if find results/cells -type f -name '*.json' -print -quit 2>/dev/null | grep -q .; then
  echo "Refusing oracle launch after transformed-grid results exist" >&2
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(tr -d '[:space:]' < "$PID_FILE")"
  if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "Refusing duplicate launch: oracle PID $EXISTING_PID is live" >&2
    exit 1
  fi
  rm -f "$PID_FILE"
fi

LOG="results/logs/oracle_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "commit=$(git rev-parse HEAD)" > "$LOG"
echo "gcarl_commit=$(git -C vendor/GCaRL rev-parse HEAD)" >> "$LOG"
echo "notears_commit=$(git -C vendor/notears rev-parse HEAD)" >> "$LOG"
nohup scripts/run_oracle.sh >> "$LOG" 2>&1 &
NEW_PID="$!"
TMP_PID="${PID_FILE}.tmp.$$"
printf '%s\n' "$NEW_PID" > "$TMP_PID"
mv "$TMP_PID" "$PID_FILE"
echo "PID=$NEW_PID"
echo "log=$LOG"
