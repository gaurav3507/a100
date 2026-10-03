#!/usr/bin/env bash
# Tier 2 experiments on the A100: Experiment A (E4: Backbone C, the second backbone,
# then the covariance readout as a negative control), then Experiment B (E5:
# split-control gate). No tmux; launch with nohup from the repo root:
#
#   export GRD_DATA_ROOT=/workspace/external        # see DATA.md
#   export CB_ACTIVATE=/path/to/cb/bin/activate     # the cb venv (optional if active)
#   nohup bash experiments/run_tier2_a100.sh > /dev/null 2>&1 &
#   tail -f "$(cat results/logs/tier2_latest.txt)"
#
# All output goes to results/logs/tier2_<date>_<time>.log. Each stage prints one
# PASS or FAIL line with its artifact count. FAIL means the stage crashed or its
# artifact is missing or malformed. A scientific outcome (for example Backbone C
# behaving differently from the precision backbone, or RPE1's BH count dropping
# under split control) is a finding recorded in the JSON, never a FAIL.
#
# Backbone C runs only if its population check passes (declared rule: population
# MCC mean >= 0.99 and every seed >= 0.98 at the E2 reference point); otherwise its
# sweeps are skipped and reported, and the rest of the run continues.
#
# The preflight re-screens RPE1 seed 0 with the unchanged E3 gate and must
# reproduce results/e3_stability and results/e3_decisions exactly; if it does not,
# the run stops before any Tier 2 stage (set TIER2_ALLOW_PREFLIGHT_FAIL=1 to go on).
# TIER2_SKIP_DONE=1 skips stages whose artifact already validates (resume).
#
# This script never installs anything. Do not pip install -U anything that touches
# numpy in the cb venv.

set -u -o pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="results/logs/tier2_${STAMP}.log"
LOCK="results/logs/tier2.pid"
echo "$LOG" > results/logs/tier2_latest.txt
exec >>"$LOG" 2>&1

if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
    echo "another Tier 2 run is active (pid $(cat "$LOCK")); exiting"
    exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

if [ -n "${CB_ACTIVATE:-}" ]; then
    # shellcheck disable=SC1090
    source "$CB_ACTIVATE"
fi
PY=${GRD_PYTHON:-python}
FAILED=()

echo "=== Tier 2 run ${STAMP} on $(hostname), repo $(pwd)"
echo "=== commit $(git rev-parse HEAD 2>/dev/null || echo 'not a git checkout') \
branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
echo "=== GRD_DATA_ROOT=${GRD_DATA_ROOT:-<unset, legacy /workspace/external>}"

if ! "$PY" - <<'EOF'
import importlib, sys
print("python", sys.version.split()[0], sys.executable)
for name in ("numpy", "scipy", "anndata"):
    module = importlib.import_module(name)
    print(name, module.__version__)
try:
    import matplotlib
    print("matplotlib", matplotlib.__version__)
except ImportError:
    print("matplotlib missing: the figure stage will be skipped")
sys.path.insert(0, "experiments")
from data_paths import real_data_inputs
missing = [f"{k}: {v}" for k, v in real_data_inputs().items()
           if k in ("K562", "RPE1", "Norman") and not v.exists()]
if missing:
    sys.exit("missing Perturb-seq inputs: " + "; ".join(missing))
EOF
then
    echo "FAIL [preflight-env] python environment or data inputs"
    exit 1
fi

stage() {
    # stage LABEL CHECK COMMAND...   (CHECK is a tier2_summary.py --check name or -)
    local label=$1 chk=$2 t0 rc dt msg
    shift 2
    if [ "${TIER2_SKIP_DONE:-0}" = "1" ] && [ "$chk" != "-" ] \
        && msg=$("$PY" experiments/tier2_summary.py --check "$chk" 2>/dev/null); then
        echo "PASS [$label] already done, skipped: $msg"
        return 0
    fi
    t0=$(date +%s)
    echo ""
    echo "=== [$label] start $(date '+%Y-%m-%d %H:%M:%S') :: $*"
    "$@"
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    if [ $rc -ne 0 ]; then
        echo "FAIL [$label] exit code $rc (${dt}s)"
        FAILED+=("$label")
        return 1
    fi
    if [ "$chk" = "-" ]; then
        echo "PASS [$label] (${dt}s)"
        return 0
    fi
    if msg=$("$PY" experiments/tier2_summary.py --check "$chk"); then
        echo "PASS [$label] $msg (${dt}s)"
    else
        echo "FAIL [$label] $msg (${dt}s)"
        FAILED+=("$label")
        return 1
    fi
}

# ---------------------------------------------------------------- preflight
if ! stage preflight-e3-reproduction - \
        "$PY" experiments/tier2_screen.py --dataset rpe1 --seed 0; then
    if [ "${TIER2_ALLOW_PREFLIGHT_FAIL:-0}" != "1" ]; then
        echo "stopping: the unchanged E3 gate did not reproduce the committed RPE1 decisions"
        exit 1
    fi
    echo "continuing despite preflight failure (TIER2_ALLOW_PREFLIGHT_FAIL=1)"
fi

# ---------------------------------------------------------------- Experiment A
# Backbone C (second backbone): the population check must pass before any sweep.
DATASETS="k562:K562 rpe1:RPE1 norman:Norman"
if stage A0-e4-C-population-check e4_popcheck \
        "$PY" experiments/e4_second_backbone.py population-check; then
    stage A1-e4-C-calibration e4_calibration \
        "$PY" experiments/e4_second_backbone.py calibration --backbone C
    stage A2-e4-C-starvation e4_starvation \
        "$PY" experiments/e4_second_backbone.py starvation --backbone C
    for pair in $DATASETS; do
        ds=${pair%%:*}
        stage "A3-e4-C-real-${ds}" "e4_real:${pair#*:}" \
            "$PY" experiments/e4_second_backbone.py real --backbone C --dataset "$ds"
    done
else
    echo "SKIP [A1-A3 Backbone C] population check did not pass; no Backbone C sweep runs"
    FAILED+=("A1-A3-backbone-C-skipped")
fi

# Covariance readout, relabelled a negative control (headline: population MCC).
stage A4-e4-negctrl-calibration e4nc_calibration \
    "$PY" experiments/e4_second_backbone.py calibration --backbone covneg
stage A5-e4-negctrl-starvation e4nc_starvation \
    "$PY" experiments/e4_second_backbone.py starvation --backbone covneg
for pair in $DATASETS; do
    ds=${pair%%:*}
    stage "A6-e4-negctrl-real-${ds}" "e4nc_real:${pair#*:}" \
        "$PY" experiments/e4_second_backbone.py real --backbone covneg --dataset "$ds"
done

# ---------------------------------------------------------------- Experiment B
for pair in $DATASETS; do
    ds=${pair%%:*}
    stage "B1-e5-real-${ds}" "e5_real:${pair#*:}" \
        "$PY" experiments/e5_split_control.py real --dataset "$ds"
done
stage B2-e5-rpe1-confound e5_confound \
    "$PY" experiments/e5_split_control.py rpe1-confound
stage B3-e5-poscontrol e5_poscontrol \
    "$PY" experiments/e5_split_control.py poscontrol

# ---------------------------------------------------------------- figures + summary
if "$PY" -c "import matplotlib" 2>/dev/null; then
    stage F-figure7 - "$PY" paper/make_all_figures.py --figure figure7
    stage F-figure8 - "$PY" paper/make_all_figures.py --figure figure8
else
    echo "SKIP [figures] matplotlib not installed; render figure7/figure8 on the Mac"
fi

echo ""
"$PY" experiments/tier2_summary.py | tee "results/logs/tier2_${STAMP}_summary.txt"
echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "=== Tier 2 finished $(date '+%Y-%m-%d %H:%M:%S'): all stages PASS"
    exit 0
fi
echo "=== Tier 2 finished $(date '+%Y-%m-%d %H:%M:%S'): FAILED stages: ${FAILED[*]}"
exit 1
