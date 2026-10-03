#!/usr/bin/env bash
# Run the preprocessing sweep across the six Perturb-seq control matrices.
#
# Same conventions as run_descriptives.sh: SCRIPTS resolves next to this file,
# absolute logs and results, one log per job, and an artefact assertion at the
# end so a partial run cannot be mistaken for a complete one.
#
# PID LOCKFILE. Two concurrent batches wrote into the same output directory on
# the morning of 11 Aug and the artefacts had to be discarded. This refuses to
# start while another instance is alive, and clears a stale lock whose PID is
# gone.
#
#   nohup bash causalbench/scripts/run_preprocessing_sweep.sh > logs/sweep.log 2>&1 &
#
# Env overrides:
#   PY       python to use      (default: the A100 cb venv)
#   PROJ     project root       (default: /workspace/ranktest-diagnostics)
#   SCRIPTS  script directory   (default: this script's own dir)
#   RESULTS  artefact directory (default: causalbench/results/ranktest in repo)
#   HVG      Frangieh gene cap  (default: unset, all genes)
#   LOCKFILE lock path          (default: $PROJ/logs/preprocessing_sweep.lock)

set -uo pipefail            # deliberately NOT -e: one bad job must not abort
                            # the batch; the summary decides the exit code

PY="${PY:-python3}"
PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="${SCRIPTS:-$SELF_DIR}"
LOGS="$PROJ/logs"
SCRIPT="$SCRIPTS/91_preprocessing_sweep.py"
RESULTS="${RESULTS:-$(cd "$(dirname "$0")/../results" 2>/dev/null && pwd)}"
ART_DIR="$RESULTS/preprocessing"
LOCKFILE="${LOCKFILE:-$LOGS/preprocessing_sweep.lock}"

mkdir -p "$LOGS" "$ART_DIR"

# ------------------------------------------------------------------ lockfile
if [ -e "$LOCKFILE" ]; then
  OTHER=$(cat "$LOCKFILE" 2>/dev/null || echo "")
  if [ -n "$OTHER" ] && kill -0 "$OTHER" 2>/dev/null; then
    echo "[fatal] another sweep is already running (pid $OTHER, lock $LOCKFILE)." >&2
    echo "        Two concurrent batches into one output directory is exactly" >&2
    echo "        how the 11 Aug artefacts had to be discarded. Refusing." >&2
    exit 3
  fi
  echo "[lock] stale lock from pid ${OTHER:-unknown} (not running); reclaiming"
  rm -f "$LOCKFILE"
fi
echo $$ > "$LOCKFILE"
cleanup() { rm -f "$LOCKFILE"; }
trap cleanup EXIT INT TERM

if [ ! -f "$SCRIPT" ]; then
  echo "[fatal] script not found: $SCRIPT" >&2; exit 2
fi
if [ ! -x "$PY" ] && ! command -v "$PY" >/dev/null 2>&1; then
  echo "[fatal] python not found: $PY" >&2; exit 2
fi

JOBS=(k562 rpe1 norman frangieh_coculture frangieh_control frangieh_ifng)

HVG_ARG=""
[ -n "${HVG:-}" ] && HVG_ARG="--hvg $HVG"

echo "=========================================================="
echo " preprocessing sweep, four arms at matched n=2000"
echo "   python  : $PY"
echo "   script  : $SCRIPT"
echo "   logs    : $LOGS"
echo "   results : $ART_DIR"
echo "   lock    : $LOCKFILE (pid $$)"
echo "   started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================================="

declare -A PRE
for name in "${JOBS[@]}"; do
  PRE["$name"]=$(ls -1 "$ART_DIR"/*__"$name".json 2>/dev/null | wc -l | tr -d ' ')
done

declare -A RC
for name in "${JOBS[@]}"; do
  extra=""
  case "$name" in frangieh_*) extra="$HVG_ARG";; esac
  echo ""
  echo "--- [$name] $(date -u +%H:%M:%SZ) ---"
  # shellcheck disable=SC2086
  "$PY" -u "$SCRIPT" --dataset "$name" $extra > "$LOGS/sweep_${name}.log" 2>&1
  RC["$name"]=$?
  echo "    exit=${RC[$name]}  log=$LOGS/sweep_${name}.log"
done

echo ""
echo "=========================================================="
printf " %-22s %6s  %-9s %s\n" "JOB" "EXIT" "ARTEFACT" "STATUS"
echo "----------------------------------------------------------"
FAIL=0
for name in "${JOBS[@]}"; do
  now=$(ls -1 "$ART_DIR"/*__"$name".json 2>/dev/null | wc -l | tr -d ' ')
  if [ "$now" -gt "${PRE[$name]}" ]; then art="yes"; else art="NO"; fi
  if [ "${RC[$name]}" -eq 0 ] && [ "$art" = "yes" ]; then
    status="PASS"
  else
    status="FAIL"; FAIL=$((FAIL+1))
  fi
  printf " %-22s %6s  %-9s %s\n" "$name" "${RC[$name]}" "$art" "$status"
done
echo "----------------------------------------------------------"
echo " finished : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$FAIL" -gt 0 ]; then
  echo " RESULT   : FAIL -- $FAIL of ${#JOBS[@]} job(s) produced no new artefact."
  echo "            This run is INCOMPLETE. Do not treat it as a full sweep."
  echo "            A step-0 gate failure exits 1 and is the likeliest cause;"
  echo "            check the per-job log before rerunning."
  echo "=========================================================="
  exit 1
fi
echo " RESULT   : PASS -- all ${#JOBS[@]} jobs produced a new artefact."
echo "=========================================================="
exit 0
