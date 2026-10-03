"""CausalBench .npz -> AnnData for discrepancy_vae, plus THE canonical zero-shot split.

The split file written here is the single source of truth. Baselines and model MUST
both read it. Generating splits independently in two scripts gives different held-out
sets even with the same seed, because the perturbation list gets built in different
orders -- which would silently invalidate every baseline comparison.

Their contract (external/discrepancy_vae/src/dataset.py, SCDataset):
  adata.X                sparse, cells x genes
  adata.obs['guide_ids'] '' for control, 'GENE' for a single perturbation
"""
import os, json
import numpy as np, anndata as ad, pandas as pd
from scipy.sparse import csr_matrix

D    = "/workspace/meridian-identifiability/causalbench/data"
OUT  = "/workspace/external/discrepancy_vae/datasets"
SPL  = "/workspace/meridian-identifiability/framework/results/splits"
CTRL = "non-targeting"
NMIN = 200
SEED = 0
os.makedirs(OUT, exist_ok=True)
os.makedirs(SPL, exist_ok=True)

for ds in ["k562", "rpe1"]:
    d  = np.load(os.path.join(D, f"dataset_{ds}.npz"), allow_pickle=True)
    X  = d["expression_matrix"].astype(np.float32)
    iv = np.asarray(d["interventions"]).astype(str)
    vn = [str(v) for v in d["var_names"]]

    keep = iv != "excluded"
    X, iv = X[keep], iv[keep]
    guide = np.where(iv == CTRL, "", iv)

    vs = set(vn)
    tg = sorted(set(guide) - {""})
    counts = {g: int((guide == g).sum()) for g in tg}
    usable = sorted([g for g in tg if counts[g] >= NMIN])
    missing = [g for g in tg if g not in vs]

    print(f"\n===== {ds} =====", flush=True)
    print(f"  cells {X.shape[0]}  genes {len(vn)}  control {(guide=='').sum()}", flush=True)
    print(f"  perturbations {len(tg)}  usable (>={NMIN} cells) {len(usable)}", flush=True)
    print(f"  targets missing from feature columns: {len(missing)}  (expect 0)", flush=True)
    print(f"  X: min {X.min():.3f} max {X.max():.3f} mean {X.mean():.3f}", flush=True)
    assert X.min() >= 0, "expected non-negative log1p values"
    assert not missing, f"{len(missing)} targets absent from features"

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(usable))
    n_te = int(0.2 * len(usable))
    held  = sorted(usable[i] for i in perm[:n_te])
    train = sorted(usable[i] for i in perm[n_te:])

    sp = os.path.join(SPL, f"{ds}_zeroshot_split.json")
    json.dump({"dataset": ds, "nmin": NMIN, "seed": SEED,
               "n_usable": len(usable), "n_train": len(train), "n_heldout": len(held),
               "cells_per_perturbation": counts,
               "train_perturbations": train,
               "heldout_perturbations": held},
              open(sp, "w"), indent=2)
    print(f"  split: {len(train)} train / {len(held)} held out", flush=True)
    print(f"  -> {sp}", flush=True)

    dst = os.path.join(OUT, f"causalbench_{ds}.h5ad")
    if os.path.exists(dst):
        chk = ad.read_h5ad(dst, backed="r")
        if chk.shape == X.shape:
            print(f"  SKIP {dst} (exists, shape matches)", flush=True)
            continue
        print(f"  STALE {dst} {chk.shape} != {X.shape}, rewriting", flush=True)

    ad.AnnData(
        X=csr_matrix(X),
        obs=pd.DataFrame({"guide_ids": pd.Categorical(guide)},
                         index=[f"c{i}" for i in range(X.shape[0])]),
        var=pd.DataFrame(index=vn)
    ).write_h5ad(dst)
    print(f"  wrote {dst}", flush=True)

print("\nALL DONE", flush=True)
