"""Canonical zero-shot baselines. These are the numbers the model must beat.

Fixes two flaws in scripts/15_zeroshot.py:

  1. SPLIT MISMATCH. 15_zeroshot generated its own held-out set. Same seed, different
     construction order => a different 77 perturbations than the model would see, so
     the comparison was invalid. This reads results/splits/<ds>_zeroshot_split.json,
     which is the single source of truth.

  2. CEILING NOT COMPARABLE. Baselines predicted the FULL-data truth while the ceiling
     predicted a half-sample from another half-sample -- different targets, different
     noise. Here every method predicts the SAME target: half B of the held-out
     perturbation's cells. The ceiling predicts B from half A. Now they are on one scale.

Also: all available cells are used. Subsampling to 200 crushed the ceiling and
understated headroom ~6x.

The targeted gene's own column is zeroed throughout -- "silencing g lowers g" is
trivially true and would inflate every method equally. Downstream effects only.
"""
import os, json
import numpy as np
from sklearn.linear_model import Ridge

D     = "/workspace/meridian-identifiability/causalbench/data"
SPL   = "/workspace/meridian-identifiability/framework/results/splits"
OUT   = "/workspace/meridian-identifiability/framework/results/zeroshot_canonical"
CTRL  = "non-targeting"
D_LAT = 15
SEED  = 0
os.makedirs(OUT, exist_ok=True)


def r2(pred, true):
    """R2 against the 'no effect' null, which is the honest reference for a shift."""
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = (true ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def run(ds):
    out_f = os.path.join(OUT, f"{ds}.json")
    if os.path.exists(out_f):
        print(f"SKIP {ds}"); return

    split = json.load(open(os.path.join(SPL, f"{ds}_zeroshot_split.json")))
    train_p, held_p = split["train_perturbations"], split["heldout_perturbations"]

    d  = np.load(os.path.join(D, f"dataset_{ds}.npz"), allow_pickle=True)
    X  = d["expression_matrix"].astype(np.float64)
    iv = np.asarray(d["interventions"]).astype(str)
    genes = [str(v) for v in d["var_names"]]
    keep = iv != "excluded"; X, iv = X[keep], iv[keep]
    gidx = {g: i for i, g in enumerate(genes)}
    p = len(genes)

    ctrl = np.where(iv == CTRL)[0]
    ctrl_mu = X[ctrl].mean(0)
    rng = np.random.default_rng(SEED)

    print(f"\n===== {ds} =====  {len(train_p)} train / {len(held_p)} held out  "
          f"| {p} genes", flush=True)

    def shift(rows, t):
        v = X[rows].mean(0) - ctrl_mu
        v = v.copy(); v[t] = 0.0
        return v

    # control-cell correlation profile: the feature vector for each gene
    Cc = np.nan_to_num(np.corrcoef((X[ctrl] - ctrl_mu).T))

    # ---- fit on TRAIN perturbations only, all their cells --------------------
    V_tr, F_tr = [], []
    for g in train_p:
        t = gidx[g]
        V_tr.append(shift(np.where(iv == g)[0], t))
        F_tr.append(Cc[t])
    V_tr, F_tr = np.array(V_tr), np.array(F_tr)

    gm = V_tr.mean(0)                                   # generic response
    _, _, Vt = np.linalg.svd(V_tr, full_matrices=False)
    W = Vt[:D_LAT].T
    ridge = Ridge(alpha=100.0).fit(F_tr, V_tr @ W)
    typ = float(np.median(np.linalg.norm(V_tr, axis=1)))

    # ---- evaluate: every method predicts half B of the held-out cells --------
    rows = []
    for g in held_p:
        t = gidx[g]
        cells = rng.permutation(np.where(iv == g)[0])
        h = len(cells) // 2
        A, B = shift(cells[:h], t), shift(cells[h:], t)   # B is the shared target

        preds = {
            "zero":        np.zeros(p),
            "global_mean": gm,
            "corr_prop":   -Cc[t] * (typ / (np.linalg.norm(Cc[t]) + 1e-12)),
            "nn_corr":     V_tr[int(np.argmax(
                               F_tr @ Cc[t] / (np.linalg.norm(F_tr, axis=1)
                                               * np.linalg.norm(Cc[t]) + 1e-12)))],
            "ridge_basis": ridge.predict(Cc[t][None])[0] @ W.T,
            "CEILING":     A,                              # same-perturbation half
        }
        for name, pr in preds.items():
            pr = pr.copy(); pr[t] = 0.0
            rows.append((name, g, r2(pr, B), cos(pr, B)))

    order = ["zero", "global_mean", "corr_prop", "nn_corr", "ridge_basis", "CEILING"]
    res = {}
    print(f"\n  {'method':<14}{'R2 med':>9}{'R2 mean':>9}{'cos med':>9}"
          f"{'frac>gmean':>12}", flush=True)
    gm_by_g = {g: v for n, g, v, _ in rows if n == "global_mean" for g in [g]}
    for name in order:
        rr = np.array([x[2] for x in rows if x[0] == name])
        cc = np.array([x[3] for x in rows if x[0] == name])
        frac = np.mean([v > gm_by_g[g] for n, g, v, _ in rows if n == name])
        res[name] = dict(r2_median=float(np.median(rr)), r2_mean=float(np.mean(rr)),
                         cos_median=float(np.median(cc)), frac_beats_gmean=float(frac))
        tag = "  <- reference" if name == "global_mean" else \
              "  <- upper bound" if name == "CEILING" else ""
        print(f"  {name:<14}{np.median(rr):>9.4f}{np.mean(rr):>9.4f}"
              f"{np.median(cc):>9.4f}{frac:>12.2f}{tag}", flush=True)

    g0, c0 = res["global_mean"]["r2_median"], res["CEILING"]["r2_median"]
    head = c0 - g0
    print(f"\n  HEADROOM (ceiling - global_mean) = {head:.4f} R2 units", flush=True)
    print(f"  {'method':<14}{'gain':>9}{'% of headroom':>16}", flush=True)
    for name in ["corr_prop", "nn_corr", "ridge_basis"]:
        gain = res[name]["r2_median"] - g0
        print(f"  {name:<14}{gain:>+9.4f}{100*gain/head if head>0 else float('nan'):>15.1f}%",
              flush=True)
    res["_headroom"] = float(head)

    json.dump({"dataset": ds, "d_latent": D_LAT,
               "n_train": len(train_p), "n_heldout": len(held_p),
               "split_file": f"{ds}_zeroshot_split.json", "results": res,
               "per_perturbation": [{"method": n, "gene": g, "r2": v, "cos": c}
                                    for n, g, v, c in rows]},
              open(out_f + ".tmp", "w"), indent=2)
    os.rename(out_f + ".tmp", out_f)


if __name__ == "__main__":
    for ds in ["k562", "rpe1"]:
        try: run(ds)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"FAIL {ds}: {e}", flush=True)
    print("\nALL DONE", flush=True)
