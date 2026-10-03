#!/usr/bin/env bash
# Overnight baseline runner. Executes every baseline variant sequentially,
# logs each to its own file, and keeps going if one step fails.
#
# Usage (inside tmux, from the repo root, venv active):
#   bash scripts/run_baselines_overnight.sh
#
# Safe alongside Track A: forces CPU-only and modest thread counts.
set -u  # unset vars are errors; deliberately NOT set -e (one failure must not kill the queue)

cd "$(dirname "$0")/.." || { echo "FATAL: cannot cd to repo root"; exit 1; }
[ -f "scripts/train_baselines.py" ] || { echo "FATAL: scripts/train_baselines.py not found -- run from the repo"; exit 1; }

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

LOGDIR="data/processed/baseline_results/logs"
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/master.log"

run_step () {
    local name="$1"; shift
    local log="$LOGDIR/${name}.log"
    local t0=$SECONDS
    echo "[$(date '+%H:%M:%S')] START $name" | tee -a "$MASTER"
    if python "$@" > "$log" 2>&1; then
        local dt=$(( SECONDS - t0 ))
        echo "[$(date '+%H:%M:%S')] OK    $name (${dt}s)" | tee -a "$MASTER"
    else
        local dt=$(( SECONDS - t0 ))
        echo "[$(date '+%H:%M:%S')] FAIL  $name (${dt}s) -- see $log" | tee -a "$MASTER"
    fi
}

echo "============================================================" | tee -a "$MASTER"
echo "Baseline queue started $(date)" | tee -a "$MASTER"
echo "============================================================" | tee -a "$MASTER"

# --- 1. Trivial floors (seconds each) --------------------------------------
run_step "majority"        scripts/train_baselines.py --model majority
run_step "stratified"      scripts/train_baselines.py --model stratified

# --- 2. Paper-config classifiers, all 3 nodes ------------------------------
run_step "rf_all"          scripts/train_baselines.py --model rf
run_step "svm_all"         scripts/train_baselines.py --model svm

# --- 3. Paper's closest replication: + PCA(90%) ----------------------------
run_step "rf_all_pca"      scripts/train_baselines.py --model rf  --pca
run_step "svm_all_pca"     scripts/train_baselines.py --model svm --pca

# --- 4. Unimodal ablation on both classifiers -------------------------------
for mod in vision bio grip; do
    run_step "rf_${mod}"   scripts/train_baselines.py --model rf  --modality "$mod"
    run_step "svm_${mod}"  scripts/train_baselines.py --model svm --modality "$mod"
done

# --- 5. Final summary table --------------------------------------------------
python - <<'PY' 2>&1 | tee -a "$MASTER"
import json
from pathlib import Path

d = Path("data/processed/baseline_results")
rows = []
for f in sorted(d.glob("*.json")):
    try:
        r = json.loads(f.read_text())
        rows.append((f.stem,
                     r["accuracy"]["mean"], r["accuracy"]["std"],
                     r["f1_macro"]["mean"], r["f1_macro"]["std"],
                     r["n_runs"]))
    except (json.JSONDecodeError, KeyError) as e:
        rows.append((f.stem + "  <unreadable: " + str(e) + ">", 0, 0, 0, 0, 0))

print()
print("=" * 78)
print("FINAL BASELINE SUMMARY (LOSO, 3-class)")
print("=" * 78)
print(f"{'variant':<22}{'accuracy':>16}{'f1_macro':>16}{'folds':>7}")
print("-" * 78)
for name, am, asd, fm, fsd, n in rows:
    print(f"{name:<22}{am*100:>9.2f} ± {asd*100:<4.1f}{fm*100:>9.2f} ± {fsd*100:<4.1f}{n:>7}")
print("=" * 78)
print("Reference points: LightGBM 48.03% / MLP 41.79% / GRU 41.52% (track_b)")
PY

echo "[$(date '+%H:%M:%S')] Queue finished." | tee -a "$MASTER"
