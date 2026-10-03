#!/usr/bin/env bash
# Run the full descriptives batch, serially, and REFUSE to look complete when
# it is not.
#
# The 10 Aug batch silently produced no Frangieh artefacts: the driver passed
# --dataset frangieh_coculture and friends, which 85_dataset_descriptives.py
# did not accept. Individual jobs failed, the loop carried on, and nothing in
# the output said so. Hence the artefact assertion at the end: a partial run
# now exits non-zero and prints which jobs are missing.
#
#   nohup bash causalbench/scripts/run_descriptives.sh > logs/batch.log 2>&1 &
#
# Env overrides:
#   PY        python to use          (default: the A100 cb venv)
#   PROJ      project root           (default: /workspace/ranktest-diagnostics)
#   SCRIPTS   script directory       (default: this script's own dir)
#   RESULTS   artefact directory     (default: causalbench/results/ranktest under the repo)
#   HVG       Frangieh gene cap      (default: unset, all genes)
#   ABIDE_NPZ path to the ABIDE npz (default: the A100 data dir)

set -uo pipefail            # deliberately NOT -e: one bad job must not abort
                            # the batch, the summary is what decides the exit

PY="${PY:-python3}"
PROJ="${PROJ:-$(cd "$(dirname "$0")/.." && pwd)}"
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
# NOT $PROJ/scripts: that path does not exist on the A100 and the
# 15:36Z launch died on it. Resolve next to this script, like RESULTS.
SCRIPTS="${SCRIPTS:-$SELF_DIR}"
LOGS="$PROJ/logs"
SCRIPT="$SCRIPTS/85_dataset_descriptives.py"
RESULTS="${RESULTS:-$(cd "$(dirname "$0")/../results" 2>/dev/null && pwd)}"
ART_DIR="$RESULTS/descriptives"

mkdir -p "$LOGS"

if [ ! -f "$SCRIPT" ]; then
  echo "[fatal] script not found: $SCRIPT" >&2; exit 2
fi
if [ ! -x "$PY" ] && ! command -v "$PY" >/dev/null 2>&1; then
  echo "[fatal] python not found: $PY" >&2; exit 2
fi

# job name  ->  arguments.  Frangieh is one job PER ARM; arms are never pooled.
JOBS=(
  "k562|--dataset k562"
  "rpe1|--dataset rpe1"
  "norman|--dataset norman"
  "frangieh_coculture|--dataset frangieh --arm coculture"
  "frangieh_control|--dataset frangieh --arm control"
  "frangieh_ifng|--dataset frangieh --arm ifng"
  "hcp|--dataset hcp"
  "abide|--dataset abide"
)

HVG_ARG=""
[ -n "${HVG:-}" ] && HVG_ARG="--hvg $HVG"

# Passed explicitly. The Python default used to be a hardcoded Mac
# path, which failed every A100 batch.
ABIDE_NPZ="${ABIDE_NPZ:-${PRECOND_DATA:-$PROJ/data}/abide_harmonized.npz}"

echo "=========================================================="
echo " descriptives batch"
echo "   python  : $PY"
echo "   script  : $SCRIPT"
echo "   logs    : $LOGS"
echo "   results : $ART_DIR"
echo "   abide   : $ABIDE_NPZ"
echo "   started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================================="

# artefacts that already existed, so "present" means THIS run produced one
declare -A PRE
for spec in "${JOBS[@]}"; do
  name="${spec%%|*}"
  PRE["$name"]=$(ls -1 "$ART_DIR"/*__"$name".json 2>/dev/null | wc -l | tr -d ' ')
done

declare -A RC
for spec in "${JOBS[@]}"; do
  name="${spec%%|*}"; args="${spec#*|}"
  extra=""
  case "$name" in
    frangieh_*) extra="$HVG_ARG";;
    abide)      extra="--abide-npz $ABIDE_NPZ";;
  esac
  echo ""
  echo "--- [$name] $(date -u +%H:%M:%SZ) ---"
  # shellcheck disable=SC2086
  "$PY" -u "$SCRIPT" $args $extra > "$LOGS/${name}.log" 2>&1
  RC["$name"]=$?
  echo "    exit=${RC[$name]}  log=$LOGS/${name}.log"
done

# ---------------------------------------------------------------- summary
echo ""
echo "=========================================================="
printf " %-22s %6s  %-9s %s\n" "JOB" "EXIT" "ARTEFACT" "STATUS"
echo "----------------------------------------------------------"
FAIL=0
for spec in "${JOBS[@]}"; do
  name="${spec%%|*}"
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
  echo "            This run is INCOMPLETE. Do not treat it as a full batch."
  echo "=========================================================="
  exit 1
fi
echo " RESULT   : PASS -- all ${#JOBS[@]} jobs produced a new artefact."
echo "=========================================================="
exit 0
