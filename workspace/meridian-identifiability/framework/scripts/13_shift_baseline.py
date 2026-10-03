"""The trivially strong baseline: predict A->B if B moved when A was perturbed in TRAIN.

The statistical evaluator asks exactly this question on HELD-OUT cells. So this baseline
measures how reproducible perturbation effects are between train and test. If it scores
very high, the benchmark's statistical metric is largely a reproducibility test, and every
published number on it must be read against this bar -- not against random.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, "/workspace/external/causalbench_repo")
from causalscbench.evaluation import statistical_evaluation

D, CTRL, BUDGET, SEED = "/workspace/meridian-identifiability/causalbench/data", "non-targeting", 5000, 0
OUT = "/workspace/meridian-identifiability/framework/results/eval_calibration"
rng = np.random.default_rng(SEED)

for ds in ["k562", "rpe1"]:
    d = np.load(os.path.join(D, f"dataset_{ds}.npz"), allow_pickle=True)
    X = d["expression_matrix"].astype(np.float64)
    iv = np.asarray(d["interventions"]).astype(str)
    genes = [str(v) for v in d["var_names"]]
    keep = iv != "excluded"; X, iv = X[keep], iv[keep]

    n = X.shape[0]; perm = rng.permutation(n)
    te, tr = perm[:n//5], perm[n//5:]
    ev = statistical_evaluation.Evaluator(X[te], list(iv[te]), genes)
    Xtr, ivtr = X[tr], iv[tr]
    ctrl_mu = Xtr[ivtr == CTRL].mean(0)
    gidx = {g: i for i, g in enumerate(genes)}
    perturbed_te = set(iv[te]) - {CTRL}

    scored = []
    for g in set(ivtr) - {CTRL}:
        if g not in perturbed_te: continue
        rows = np.where(ivtr == g)[0]
        if len(rows) < 100: continue
        v = np.abs(Xtr[rows].mean(0) - ctrl_mu)
        if g in gidx: v[gidx[g]] = 0.0          # exclude the target itself
        for j in np.argsort(-v)[:60]:
            scored.append((v[j], g, genes[j]))
    scored.sort(reverse=True)
    net = [(a, b) for _, a, b in scored[:BUDGET] if a != b]

    out = ev.evaluate_network(net, max_path_length=1)
    tp = out["output_graph"]["true_positives"]; fp = out["output_graph"]["false_positives"]
    print(f"{ds}: empirical_shift  edges={len(net)}  TP={tp:.0f} FP={fp:.0f} "
          f"precision={tp/(tp+fp):.4f}", flush=True)
    json.dump({"n_edges": len(net), "tp": float(tp), "fp": float(fp),
               "precision": float(tp/(tp+fp))},
              open(f"{OUT}/{ds}_empirical_shift.json", "w"), indent=2)
