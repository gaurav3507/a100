#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
test -x .venv/bin/python
test "$(git -C vendor/GCaRL rev-parse HEAD)" = "$(tr -d '[:space:]' < vendor/GCaRL_COMMIT.txt)"
test "$(git -C vendor/notears rev-parse HEAD)" = "$(tr -d '[:space:]' < vendor/NOTEARS_COMMIT.txt)"
export PYTHONPATH="$PROJECT_DIR/src"
exec .venv/bin/python scripts/run_oracle.py

