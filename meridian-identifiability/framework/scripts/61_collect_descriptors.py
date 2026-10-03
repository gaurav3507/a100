"""Collect the numbers that only exist on the analysis machine (A100).

Everything here needs either the raw .npz / .h5ad, the HCP pipeline outputs, or
the third-party source tree. Nothing is written into /workspace/external/ --
that tree is opened read-only.

Emits paper/descriptors.json plus a printed report.

Usage (A100):
    source /workspace/venvs/cb/bin/activate
    python scripts/61_collect_descriptors.py
"""
import json
import os
import re
from pathlib import Path

import numpy as np

FW = Path(__file__).resolve().parent.parent
DATA = Path("/workspace/meridian-identifiability/causalbench/data")
EXT = Path("/workspace/external/discrepancy_vae")
HCP = Path("/workspace/meridian-identifiability/hcp")
NORMAN = EXT / "datasets/Norman2019_raw.h5ad"
CB_H5AD = EXT / "datasets/causalbench_k562.h5ad"

out = {}


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------- CausalBench npz
hdr("SECTION 7a -- CausalBench K562 / RPE1 descriptors (from dataset_*.npz)")
for ds in ("k562", "rpe1"):
    p = DATA / f"dataset_{ds}.npz"
    if not p.exists():
        print(f"  {ds}: MISSING {p}")
        continue
    d = np.load(p, allow_pickle=True)
    X = d["expression_matrix"]
    iv = np.asarray(d["interventions"]).astype(str)
    vn = [str(v) for v in d["var_names"]]

    n_cells_raw, n_feat = X.shape
    n_excluded = int((iv == "excluded").sum())
    keep = iv != "excluded"
    iv_k = iv[keep]
    n_ctrl = int((iv_k == "non-targeting").sum())
    pert = iv_k[iv_k != "non-targeting"]
    genes, counts = np.unique(pert, return_counts=True)
    n_at200 = int((counts >= 200).sum())
    n_at100 = int((counts >= 100).sum())

    Xn = np.asarray(X, dtype=np.float64)
    rec = dict(
        n_cells_raw=int(n_cells_raw), n_cells_after_excluded=int(keep.sum()),
        n_features=int(n_feat), n_excluded_cells=n_excluded,
        n_control_cells=n_ctrl, n_perturbed_cells=int(pert.size),
        n_unique_perturbations=int(genes.size),
        n_perturbations_ge_200=n_at200, n_perturbations_ge_100=n_at100,
        cells_per_pert_min=int(counts.min()),
        cells_per_pert_p10=float(np.percentile(counts, 10)),
        cells_per_pert_median=float(np.median(counts)),
        cells_per_pert_p90=float(np.percentile(counts, 90)),
        cells_per_pert_max=int(counts.max()),
        value_min=float(Xn.min()), value_max=float(Xn.max()),
        value_mean=float(Xn.mean()),
        frac_exactly_zero=float((Xn == 0).sum() / Xn.size),
        all_integers=bool(np.allclose(Xn, np.round(Xn))),
        targets_in_var_names=int(sum(1 for g in genes if g in set(vn))),
    )
    out[ds] = rec
    print(f"\n  {ds}:")
    for k, v in rec.items():
        print(f"    {k:<32} {v}")

# ------------------------------------------------------------ Norman h5ad
hdr("SECTION 7b -- Norman 2019 descriptors (from Norman2019_raw.h5ad)")
try:
    import anndata as ad
    if NORMAN.exists():
        a = ad.read_h5ad(NORMAN, backed="r")
        g = np.asarray(a.obs["guide_ids"]).astype(str)
        ctrl = g == ""
        dbl = np.array([("," in x) and (x != "") for x in g])
        sgl = np.array([("," not in x) and (x != "") for x in g])
        labels = g[sgl]
        uniq, counts = np.unique(labels, return_counts=True)
        vn = set(str(v) for v in a.var_names)
        rec = dict(
            n_cells_total=int(g.size), n_features=int(a.shape[1]),
            n_control_cells=int(ctrl.sum()),
            n_single_perturbation_cells=int(sgl.sum()),
            n_double_perturbation_cells=int(dbl.sum()),
            n_unique_single_targets=int(uniq.size),
            n_targets_ge_200=int((counts >= 200).sum()),
            n_targets_ge_100=int((counts >= 100).sum()),
            cells_per_pert_min=int(counts.min()),
            cells_per_pert_median=float(np.median(counts)),
            cells_per_pert_max=int(counts.max()),
            targets_in_var_names=int(sum(1 for u in uniq if u in vn)),
            obs_columns=list(map(str, a.obs.columns)),
        )
        out["norman"] = rec
        print()
        for k, v in rec.items():
            print(f"    {k:<32} {v}")
    else:
        print(f"  MISSING {NORMAN}")
except ImportError:
    print("  anndata not importable in this venv")

# ------------------------------------------------------ CausalBench h5ad (loader)
hdr("SECTION 7c -- causalbench_k562.h5ad (the file cb_data.py actually reads)")
try:
    import anndata as ad
    if CB_H5AD.exists():
        a = ad.read_h5ad(CB_H5AD, backed="r")
        gi = np.asarray(a.obs["guide_ids"]).astype(str)
        rec = dict(n_cells=int(a.shape[0]), n_features=int(a.shape[1]),
                   n_control_cells_empty_string=int((gi == "").sum()),
                   n_unique_guide_ids=int(np.unique(gi).size),
                   obs_columns=list(map(str, a.obs.columns)))
        out["causalbench_k562_h5ad"] = rec
        print()
        for k, v in rec.items():
            print(f"    {k:<32} {v}")
    else:
        print(f"  MISSING {CB_H5AD}")
except ImportError:
    pass

# ------------------------------------------------------------- HCP descriptors
hdr("SECTION 7d -- HCP descriptors (from hcp/results/*.json)")
for fn in ("mean_shift_LR.json", "mean_shift_RL.json", "mean_shift_v2.json"):
    p = HCP / "results" / fn
    if not p.exists():
        print(f"  MISSING {p}")
        continue
    j = json.loads(p.read_text())
    rec = {k: v for k, v in j.items() if k != "results"}
    rec["result_keys"] = sorted(j.get("results", {}).keys())
    inner = j.get("results", {})
    first = inner.get(sorted(inner)[0]) if inner else {}
    rec["per_d_keys"] = sorted(first.keys()) if isinstance(first, dict) else []
    out[f"hcp_{fn}"] = rec
    print(f"\n  {fn}:")
    for k, v in rec.items():
        print(f"    {k:<32} {v}")

# --------------------------------------------- HCP script vs 03_screen.py
hdr("SECTION 1 flag -- hcp/scripts/mean_shift_v2.py construction")
hs = HCP / "scripts/mean_shift_v2.py"
if hs.exists():
    src = hs.read_text()
    print(f"  {hs}  ({len(src.splitlines())} lines)")
    print("\n  --- full source (read-only) ---")
    for i, line in enumerate(src.splitlines(), 1):
        print(f"  {i:>4} | {line}")
    out["hcp_mean_shift_v2_py"] = dict(n_lines=len(src.splitlines()),
                                        has_pairs="pairs" in src,
                                        has_step0="step0" in src.lower())
else:
    print(f"  MISSING {hs}")

# ------------------------------------------------- third-party hyperparameters
hdr("SECTION 8 -- discrepancy_vae hyperparameters (READ-ONLY from their tree)")
for rel in ("src/train.py", "src/model.py"):
    p = EXT / rel
    if not p.exists():
        print(f"  MISSING {p}")
        continue
    src = p.read_text()
    print(f"\n  --- {rel} ---")
    for m in re.finditer(r"add_argument\(\s*'--?([\w-]+)'[^)]*?default\s*=\s*([^,)]+)", src):
        print(f"    --{m.group(1):<16} default={m.group(2).strip()}")
    if rel.endswith("train.py"):
        m = re.search(r"def loss_function\((.*?)\):", src, re.S)
        if m:
            print(f"    loss_function signature: ({' '.join(m.group(1).split())})")
        for kw in ("MMD_loss", "fix_sigma", "kernel_num", "kernel_mul"):
            n = src.count(kw)
            if n:
                print(f"    mentions {kw}: {n}x")
    if rel.endswith("model.py"):
        m = re.search(r"def dag\(self,(.*?)\):", src, re.S)
        if m:
            print(f"    CMVAE.dag signature: ({' '.join(m.group(1).split())})")
        m2 = re.search(r"torch\.inverse\([^)]*\)", src)
        if m2:
            print(f"    inverse call: {m2.group(0)}")

# ------------------------------------------- our training used their loss verbatim
hdr("SECTION 8 -- confirm our loop imports their loss_function unmodified")
for rel in ("model/cb_train.py", "scripts/27_no_causal_ablation.py"):
    p = FW / rel
    if not p.exists():
        print(f"  MISSING {p}")
        continue
    src = p.read_text()
    print(f"  {rel}:")
    print(f"    'from train import loss_function' : "
          f"{'from train import loss_function' in src}")
    print(f"    defines its own loss_function     : {'def loss_function' in src}")
    print(f"    calls loss_function(...)          : {src.count('loss_function(')}")

# ------------------------------------------------------------------------ write
p = FW / "paper/descriptors.json"
p.parent.mkdir(parents=True, exist_ok=True)
tmp = str(p) + ".tmp"
with open(tmp, "w") as f:
    json.dump(out, f, indent=2, default=str)
os.rename(tmp, p)
print(f"\n\n[write] {p}")
