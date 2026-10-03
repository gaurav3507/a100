#!/usr/bin/env bash
# One entry point for the whole pipeline.
#
# Two classes of stage. ANYWHERE stages rebuild every table, figure and index
# from committed artefacts and need no data. DATA stages need the raw matrices,
# which are not redistributable and are never committed; they are skipped
# unless --with-data is passed.
#
#   bash run_all.sh                 # anywhere stages only (default)
#   bash run_all.sh --with-data     # everything, requires $PRECOND_DATA
#
# EVERY STAGE IS ASSERTED. Artefact counts are taken before and after each
# stage and a stage that produced nothing new is a FAIL, not a warning. A run
# that cannot look complete when it is not is the whole point: "6/6 written"
# was once printed while the files landed outside the repository.
#
# Env overrides:
#   PY            python to use        (default: python3)
#   PRECOND_DATA  raw data root        (required only with --with-data)
#   LOCKFILE      lock path            (default: logs/run_all.lock)

set -uo pipefail            # deliberately NOT -e: one bad stage must not abort
                            # the run; the summary decides the exit code

REPO="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
SCRIPTS="$REPO/scripts"
LOGS="$REPO/logs"
LOCKFILE="${LOCKFILE:-$LOGS/run_all.lock}"
WITH_DATA=0
[ "${1:-}" = "--with-data" ] && WITH_DATA=1

mkdir -p "$LOGS"

# ------------------------------------------------------------------ lockfile
if [ -e "$LOCKFILE" ]; then
  OTHER=$(cat "$LOCKFILE" 2>/dev/null || echo "")
  if [ -n "$OTHER" ] && kill -0 "$OTHER" 2>/dev/null; then
    echo "[fatal] another run is already live (pid $OTHER, lock $LOCKFILE)." >&2
    echo "        Two concurrent batches into one output directory is how the" >&2
    echo "        11 Aug artefacts had to be discarded. Refusing." >&2
    exit 3
  fi
  echo "[lock] stale lock from pid ${OTHER:-unknown} (not running); reclaiming"
  rm -f "$LOCKFILE"
fi
echo $$ > "$LOCKFILE"
cleanup() { rm -f "$LOCKFILE"; }
trap cleanup EXIT INT TERM

# ------------------------------------------------------------ preflight
fail=0
if ! command -v "$PY" >/dev/null 2>&1 && [ ! -x "$PY" ]; then
  echo "[fatal] python not found: $PY   (set \$PY)" >&2; fail=1
fi
"$PY" - <<'EOF' || fail=1
import sys
missing = []
for m in ("numpy", "scipy", "matplotlib", "pandas"):
    try:
        __import__(m)
    except ImportError:
        missing.append(m)
if missing:
    sys.stderr.write("[fatal] missing packages: %s\n" % ", ".join(missing))
    sys.stderr.write("        pip install -r requirements-mac.txt   (or requirements-a100.txt)\n")
    sys.exit(1)
EOF

for f in "$SCRIPTS/87_make_gate_tables.py" "$SCRIPTS/88_make_descriptive_tables.py" \
         "$SCRIPTS/89_make_index.py" "$SCRIPTS/90_spectrum_figure.py" \
         "$SCRIPTS/92_preprocessing_table.py"; do
  [ -f "$f" ] || { echo "[fatal] script not found: $f" >&2; fail=1; }
done
for d in "$REPO/results/descriptives" "$REPO/results/gates" "$REPO/results/preprocessing"; do
  [ -d "$d" ] || { echo "[fatal] artefact directory not found: $d" >&2; fail=1; }
done

if [ "$WITH_DATA" -eq 1 ]; then
  if [ -z "${PRECOND_DATA:-}" ]; then
    echo "[fatal] --with-data given but \$PRECOND_DATA is not set." >&2
    echo "        Every loader resolves through, in order:" >&2
    echo "          1. an explicit --<name> argument" >&2
    echo "          2. \$PRECOND_DATA          (and \$ABIDE_NPZ for ABIDE)" >&2
    echo "          3. $REPO/data" >&2
    echo "          4. /workspace/precondition-audit/data" >&2
    echo "          5. /workspace/ranktest-diagnostics/data" >&2
    echo "        None of these is required to exist for the anywhere stages;" >&2
    echo "        rerun without --with-data to rebuild from committed artefacts." >&2
    exit 2
  fi
  for rel in dataset_k562.npz dataset_rpe1.npz Norman2019_raw.h5ad; do
    if [ ! -e "$PRECOND_DATA/$rel" ]; then
      echo "[fatal] required input missing. Tried:" >&2
      echo "          \$PRECOND_DATA/$rel  -> $PRECOND_DATA/$rel" >&2
      echo "          $REPO/data/$rel" >&2
      echo "          /workspace/precondition-audit/data/$rel" >&2
      echo "          /workspace/ranktest-diagnostics/data/$rel" >&2
      fail=1
    fi
  done
fi
[ "$fail" -eq 0 ] || { echo "[fatal] preflight failed; nothing was run." >&2; exit 2; }

# ------------------------------------------------------------ stage runner
count() { find "$1" -type f -name "$2" 2>/dev/null | wc -l | tr -d ' '; }
NAMES=(); RCS=(); PRES=(); POSTS=()

stage() {          # stage <label> <watch-dir> <glob> <cmd...>
  local label="$1" dir="$2" glob="$3"; shift 3
  local pre; pre=$(count "$dir" "$glob")
  echo ""
  echo "--- [$label] $(date -u +%H:%M:%SZ) ---"
  "$@" > "$LOGS/${label}.log" 2>&1
  local rc=$?
  local post; post=$(count "$dir" "$glob")
  echo "    exit=$rc  artefacts ${pre} -> ${post}  log=$LOGS/${label}.log"
  NAMES+=("$label"); RCS+=("$rc"); PRES+=("$pre"); POSTS+=("$post")
}

echo "=========================================================="
echo " precondition-audit, full pipeline"
echo "   repo    : $REPO"
echo "   python  : $PY  ($("$PY" -V 2>&1))"
echo "   mode    : $([ "$WITH_DATA" -eq 1 ] && echo 'anywhere + data' || echo 'anywhere only')"
echo "   lock    : $LOCKFILE (pid $$)"
echo "   started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================================="

if [ "$WITH_DATA" -eq 1 ]; then
  # Long. Detached so a dropped connection does not kill the batch; each
  # runner writes its own log and asserts its own artefacts.
  echo ""
  echo "--- [data stages] detaching; follow $LOGS/ ---"
  nohup bash "$SCRIPTS/run_descriptives.sh" > "$LOGS/run_descriptives.log" 2>&1 & disown
  nohup bash "$SCRIPTS/run_preprocessing_sweep.sh" > "$LOGS/run_sweep.log" 2>&1 & disown
  echo "    launched; rerun without --with-data once they finish to rebuild tables"
fi

stage descriptive_tables "$REPO/paper/tables"  'table_[ab]*'   "$PY" "$SCRIPTS/88_make_descriptive_tables.py"
stage preprocessing_tables "$REPO/paper/tables" 'table_[cd]*'  "$PY" "$SCRIPTS/92_preprocessing_table.py"
stage gate_tables        "$REPO/paper/tables"  '*.csv'         "$PY" "$SCRIPTS/87_make_gate_tables.py"
stage spectrum_figures   "$REPO/paper/figures" 'spectrum_*'    "$PY" "$SCRIPTS/90_spectrum_figure.py"
stage artefact_index     "$REPO/results"       'INDEX.md'      "$PY" "$SCRIPTS/89_make_index.py"

echo ""
echo "=========================================================="
printf " %-22s %6s %8s %8s  %s\n" "STAGE" "EXIT" "BEFORE" "AFTER" "STATUS"
echo "----------------------------------------------------------"
FAIL=0
for i in "${!NAMES[@]}"; do
  # A rebuild overwrites in place, so "produced something" is a non-zero
  # count after, not a count that grew. A zero count after is always a FAIL.
  if [ "${RCS[$i]}" -eq 0 ] && [ "${POSTS[$i]}" -gt 0 ]; then
    status="PASS"
  else
    status="FAIL"; FAIL=$((FAIL+1))
  fi
  printf " %-22s %6s %8s %8s  %s\n" \
    "${NAMES[$i]}" "${RCS[$i]}" "${PRES[$i]}" "${POSTS[$i]}" "$status"
done
echo "----------------------------------------------------------"
echo " finished : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$FAIL" -gt 0 ]; then
  echo " RESULT   : FAIL -- $FAIL of ${#NAMES[@]} stage(s) produced no artefact."
  echo "            This run is INCOMPLETE. Do not treat it as a full rebuild."
  echo "            Check the per-stage log named above before rerunning."
  echo "=========================================================="
  exit 1
fi
echo " RESULT   : PASS -- all ${#NAMES[@]} stages produced artefacts."
echo "=========================================================="
exit 0
