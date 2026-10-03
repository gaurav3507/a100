#!/usr/bin/env bash
# Drive the two regeneration jobs and REFUSE to look complete when they are not.
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
#   nohup bash scripts/run_regen.sh > logs/regen.log 2>&1 &
#
# Jobs, in order. The audit runs first: it is the cheaper of the two and its
# step-0 gate is the one that catches a bad $PRECOND_DATA.
#   control_pool  94_control_pool_audit.py    needs $PRECOND_DATA
#   power_soft    93_regen_power_curves.py    simulator only, no data
#   power_hard    93_regen_power_curves.py    simulator only, no data
#
# Env overrides:
#   PY            python to use      (default: python3)
#   PROJ          project root       (default: this script's parent)
#   SCRIPTS       script directory   (default: this script's own dir)
#   RESULTS       artefact directory (default: <repo>/results)
#   RNG_ORDER     full | isolated    (default: full, see 93's header)
#   HVG           Frangieh gene cap  (default: unset, all genes)
#   LOCKFILE      lock path          (default: $PROJ/logs/run_regen.lock)
#   SKIP_AUDIT    set to 1 to run the two power arms only

set -uo pipefail            # deliberately NOT -e: one bad job must not abort
                            # the batch; the summary decides the exit code

PY="${PY:-python3}"
PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="${SCRIPTS:-$SELF_DIR}"
LOGS="$PROJ/logs"
RESULTS="${RESULTS:-$(cd "$(dirname "$0")/../results" 2>/dev/null && pwd)}"
REGEN_DIR="$RESULTS/regen"
RNG_ORDER="${RNG_ORDER:-full}"
LOCKFILE="${LOCKFILE:-$LOGS/run_regen.lock}"

AUDIT="$SCRIPTS/94_control_pool_audit.py"
POWER="$SCRIPTS/93_regen_power_curves.py"

mkdir -p "$LOGS" "$REGEN_DIR"

# ------------------------------------------------------------------ lockfile
if [ -e "$LOCKFILE" ]; then
  OTHER=$(cat "$LOCKFILE" 2>/dev/null || echo "")
  if [ -n "$OTHER" ] && kill -0 "$OTHER" 2>/dev/null; then
    echo "[fatal] another regen batch is already running (pid $OTHER, lock $LOCKFILE)." >&2
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

# ----------------------------------------------------------------- preflight
for f in "$AUDIT" "$POWER"; do
  [ -f "$f" ] || { echo "[fatal] script not found: $f" >&2; exit 2; }
done
if [ ! -x "$PY" ] && ! command -v "$PY" >/dev/null 2>&1; then
  echo "[fatal] python not found: $PY" >&2; exit 2
fi
# 92 needs the data; the two power arms do not. Fail here rather than after the
# power arms have already burned an hour.
if [ "${SKIP_AUDIT:-0}" != "1" ] && [ -z "${PRECOND_DATA:-}" ]; then
  echo "[fatal] \$PRECOND_DATA is not set, and 94_control_pool_audit.py needs it." >&2
  echo "        Candidate roots 92 will try, in order:" >&2
  echo "          1. \$PRECOND_DATA                              (unset)" >&2
  echo "          2. $PROJ/data" >&2
  echo "          3. /workspace/precondition-audit/data" >&2
  echo "          4. /workspace/ranktest-diagnostics/data" >&2
  echo "        Export it, or set SKIP_AUDIT=1 to run the power arms only." >&2
  exit 2
fi

HVG_ARG=""
[ -n "${HVG:-}" ] && HVG_ARG="--hvg $HVG"

# job name | output subdir | command
JOBS=()
[ "${SKIP_AUDIT:-0}" != "1" ] && JOBS+=("control_pool|control_pool|$PY -u $AUDIT $HVG_ARG")
JOBS+=("power_soft|soft|$PY -u $POWER --arm soft --rng-order $RNG_ORDER")
JOBS+=("power_hard|hard|$PY -u $POWER --arm hard --rng-order $RNG_ORDER")

echo "=========================================================="
echo " precondition-audit regeneration batch"
echo "   python    : $PY"
echo "   scripts   : $SCRIPTS"
echo "   logs      : $LOGS"
echo "   results   : $REGEN_DIR"
echo "   rng-order : $RNG_ORDER"
echo "   data      : ${PRECOND_DATA:-<unset, power arms only>}"
echo "   lock      : $LOCKFILE (pid $$)"
echo "   started   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================================="

# artefacts that already existed, so "present" means THIS run produced one.
# Each job also rm -rf's its own subdir at the top, so these should read 0.
declare -A PRE
for spec in "${JOBS[@]}"; do
  name="${spec%%|*}"; rest="${spec#*|}"; sub="${rest%%|*}"
  PRE["$name"]=$(find "$REGEN_DIR/$sub" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
done

declare -A RC
for spec in "${JOBS[@]}"; do
  name="${spec%%|*}"; rest="${spec#*|}"; cmd="${rest#*|}"
  echo ""
  echo "--- [$name] $(date -u +%H:%M:%SZ) ---"
  echo "    $cmd"
  # shellcheck disable=SC2086
  $cmd > "$LOGS/regen_${name}.log" 2>&1
  RC["$name"]=$?
  echo "    exit=${RC[$name]}  log=$LOGS/regen_${name}.log"
done

# ---------------------------------------------------------------- summary
echo ""
echo "=========================================================="
printf " %-16s %6s  %-9s %s\n" "JOB" "EXIT" "ARTEFACT" "STATUS"
echo "----------------------------------------------------------"
FAIL=0
for spec in "${JOBS[@]}"; do
  name="${spec%%|*}"; rest="${spec#*|}"; sub="${rest%%|*}"
  now=$(find "$REGEN_DIR/$sub" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$now" -gt "${PRE[$name]}" ]; then art="yes"; else art="NO"; fi
  if [ "${RC[$name]}" -eq 0 ] && [ "$art" = "yes" ]; then
    status="PASS"
  else
    status="FAIL"; FAIL=$((FAIL+1))
  fi
  printf " %-16s %6s  %-9s %s\n" "$name" "${RC[$name]}" "$art" "$status"
done
echo "----------------------------------------------------------"
echo " finished : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$FAIL" -gt 0 ]; then
  echo " RESULT   : FAIL -- $FAIL of ${#JOBS[@]} job(s) produced no new artefact."
  echo "            This run is INCOMPLETE. Do not treat it as a full regen."
  echo "            A step-0 gate failure exits 1 or 2 and is the likeliest"
  echo "            cause; check the per-job log before rerunning."
  echo "=========================================================="
  exit 1
fi
echo " RESULT   : PASS -- all ${#JOBS[@]} jobs produced a new artefact."
echo "=========================================================="
exit 0
