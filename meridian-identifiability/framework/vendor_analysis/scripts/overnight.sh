#!/bin/bash
set -e
REPO=/workspace/external/discrepancy_vae
OUT=/workspace/meridian-identifiability/dvae_repro
export WANDB_MODE=disabled

echo "===== DISK ====="; df -h /workspace | tail -1

echo "===== 1. DOWNLOAD ====="
mkdir -p $REPO/datasets && cd $REPO/datasets
if [ ! -f cpa_binaries.tar ]; then
  wget -c https://dl.fbaipublicfiles.com/dlp/cpa_binaries.tar
else
  echo "tar already present"
fi
[ -f Norman2019_raw.h5ad ] || tar -xvf cpa_binaries.tar
# it may extract into a subfolder
if [ ! -f Norman2019_raw.h5ad ]; then
  found=$(find . -name "Norman2019_raw.h5ad" | head -1)
  [ -n "$found" ] && mv "$found" ./Norman2019_raw.h5ad
fi
ls -lh Norman2019_raw.h5ad || { echo "FAIL: Norman2019_raw.h5ad not found"; exit 1; }

echo "===== 2. PATCH HARDCODED PATHS ====="
cd $REPO/src
grep -rn "jzhang" *.py || echo "(none left)"
sed -i "s|/home/jzhang/discrepancy_vae/identifiable_causal_vae|$REPO|g" *.py
sed -i "s|/home/jzhang/discrepancy_vae|$REPO|g" *.py
echo "-- after patch --"
grep -rn "jzhang" *.py && { echo "FAIL: paths remain"; exit 1; } || echo "clean"
grep -rn "h5ad" *.py

echo "===== 3. VERIFY DATA LOADS ====="
python - << 'PY'
import scanpy as sc
a = sc.read_h5ad("/workspace/external/discrepancy_vae/datasets/Norman2019_raw.h5ad")
print("shape:", a.shape)
print("obs cols:", list(a.obs.columns)[:15])
for c in a.obs.columns:
    if 'pert' in c.lower() or 'guide' in c.lower() or 'cond' in c.lower():
        print(f"  {c}: {a.obs[c].nunique()} unique")
PY

echo "===== 4. TOY SEED 12 (their exact seed, 1 min) ====="
cd $REPO/src
sed -i 's/for seed in \[12\]/for seed in [12]/' run_simu_seeds.py 2>/dev/null || true
python run_simu_seeds.py 2>&1 | tail -8 || echo "(toy skipped)"

echo "===== 5. BIOLOGICAL TRAINING (the long one) ====="
mkdir -p $OUT/results/bio
cd $REPO/src
python run.py --device cuda:0 -s $OUT/results/bio/

echo "===== ALL DONE ====="
