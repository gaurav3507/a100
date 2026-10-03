"""Zero-shot intervention prediction: can you predict the effect of a perturbation
you have NEVER seen?

Why this metric and not the others:
  - statistical evaluator  -> gamed by differential expression (0.966), and cannot
                              separate direct from indirect effects
  - STRING                 -> gamed by correlation (0.639), and is partly built from
                              co-expression evidence, so the comparison is circular
  - ChIP-seq               -> every method loses to random, including random itself
                              beating correlation 6x
None of those can demonstrate that a causal method works. This one can: differential
expression REQUIRES having seen the perturbation, so it is structurally unavailable.

Protocol: hold out 20% of perturbations ENTIRELY. Predict their shift vectors. Score
against the truth, with a noise ceiling from split-half reproducibility of the same
held-out perturbation.

The target gene's own column is excluded -- "silencing g reduces g" is trivially true
and would inflate every method equally. We want DOWNSTREAM effects.

Baselines, weakest to strongest:
  zero          predict no effect at all
  global_mean   predict the average training shift (the generic stress response).
                THE KEY REFERENCE -- beating this requires gene-SPECIFIC signal.
  corr_prop     shift proportional to -corr(g, .), scaled to typical magnitude
  nn_corr       copy the shift of the training gene with the most similar
                correlation profile
  ridge_basis   regress d=15 shift-basis coefficients on the target gene's
                correlation profile. This is the learnable version, and the
                framework would be a better one.

Read: does ANYTHING beat global_mean, and how far is it from the noise ceiling.
"""
import os, json
import numpy as np
from sklearn.linear_model import Ridge

D     = "/workspace/meridian-identifiability/causalbench/data"
OUT   = "/workspace/meridian-identifiability/framework/results/zeroshot"
CTRL  = "non-targeting"
NMIN  = int(os.environ.get("NMIN", 200))
D_LAT = 15
SEED  = 0
os.makedirs(OUT, exist_ok=True)


def load(ds):
    d = np.load(os.path.join(D, f"dataset_{ds}.npz"), allow_pickle=True)
    X  = d["expression_matrix"].astype(np.float64)
    iv = np.asarray(d["interventions"]).astype(str)
    vn = [str(v) for v in d["var_names"]]
    keep = iv != "excluded"
    return X[keep], iv[keep], vn


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = (true ** 2).sum()          # against "no effect", the honest reference
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def run(ds):
    out_f = os.path.join(OUT, f"{ds}.json")
    if os.path.exists(out_f):
        print(f"SKIP {ds}"); return

    rng = np.random.default_rng(SEED)
    X, iv, genes = load(ds)
    gidx = {g: i for i, g in enumerate(genes)}
    p = len(genes)
    ctrl = np.where(iv == CTRL)[0]
    ctrl_mu = X[ctrl].mean(0)

    us, cs = np.unique(iv, return_counts=True)
    envs = [(g, np.where(iv == g)[0]) for g, c in zip(us, cs)
            if g != CTRL and c >= NMIN and g in gidx]
    print(f"\n===== {ds} =====  {len(envs)} perturbations, {p} genes", flush=True)

    # correlation profile of every gene, from CONTROL cells only -> features
    Cc = np.corrcoef((X[ctrl] - ctrl_mu).T)
    Cc = np.nan_to_num(Cc)

    def shift_of(rows, tgt):
        v = X[rows].mean(0) - ctrl_mu
        v = v.copy(); v[tgt] = 0.0          # drop the trivially-predictable column
        return v

    # split perturbations, not cells
    idx = rng.permutation(len(envs))
    ntr = int(0.8 * len(envs))
    tr = [envs[i] for i in idx[:ntr]]
    te = [envs[i] for i in idx[ntr:]]
    print(f"  train {len(tr)} perturbations | held out {len(te)} (never seen)", flush=True)

    V_tr = np.array([shift_of(r, gidx[g]) for g, r in tr])   # all cells
    F_tr = np.array([Cc[gidx[g]] for g, _ in tr])

    gm = V_tr.mean(0)
    _, _, Vt = np.linalg.svd(V_tr, full_matrices=False)
    W = Vt[:D_LAT].T

    ridge = Ridge(alpha=100.0).fit(F_tr, V_tr @ W)

    typ = float(np.median(np.linalg.norm(V_tr, axis=1)))

    rows = []
    for g, r in te:
        t = gidx[g]
        sub = rng.permutation(r)              # ALL cells -- subsampling to NMIN
        h = len(sub) // 2                     # crushed the ceiling and understated
        truth = shift_of(sub, t)              # the headroom ~6x

        # noise ceiling: one half of this perturbation's own cells predicting the other
        a, b = shift_of(sub[:h], t), shift_of(sub[h:], t)

        preds = {
            "zero":        np.zeros(p),
            "global_mean": gm,
            "corr_prop":   -Cc[t] * (typ / (np.linalg.norm(Cc[t]) + 1e-12)),
            "nn_corr":     V_tr[int(np.argmax(F_tr @ Cc[t] /
                             (np.linalg.norm(F_tr, axis=1) * np.linalg.norm(Cc[t]) + 1e-12)))],
            "ridge_basis": (ridge.predict(Cc[t][None])[0]) @ W.T,
        }
        for name, pr in preds.items():
            pr = pr.copy(); pr[t] = 0.0
            rows.append((name, r2(pr, truth), cos(pr, truth)))
        rows.append(("CEILING_splithalf", r2(a, b), cos(a, b)))

    res = {}
    print(f"\n  {'method':<20}{'R2 median':>11}{'R2 mean':>10}{'cos median':>12}", flush=True)
    for name in ["zero", "global_mean", "corr_prop", "nn_corr", "ridge_basis",
                 "CEILING_splithalf"]:
        rr = np.array([x[1] for x in rows if x[0] == name])
        cc = np.array([x[2] for x in rows if x[0] == name])
        res[name] = dict(r2_median=float(np.median(rr)), r2_mean=float(np.mean(rr)),
                         cos_median=float(np.median(cc)))
        mark = "  <-- reference" if name == "global_mean" else \
               "  <-- upper bound" if name.startswith("CEILING") else ""
        print(f"  {name:<20}{np.median(rr):>11.4f}{np.mean(rr):>10.4f}"
              f"{np.median(cc):>12.4f}{mark}", flush=True)

    gmr = res["global_mean"]["r2_median"]
    print(f"\n  gain over global_mean (median R2):", flush=True)
    for name in ["corr_prop", "nn_corr", "ridge_basis"]:
        print(f"    {name:<14}{res[name]['r2_median'] - gmr:+.4f}", flush=True)

    json.dump({"dataset": ds, "n_train": len(tr), "n_heldout": len(te),
               "d_latent": D_LAT, "results": res},
              open(out_f + ".tmp", "w"), indent=2)
    os.rename(out_f + ".tmp", out_f)


if __name__ == "__main__":
    for ds in ["k562", "rpe1"]:
        try: run(ds)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"FAIL {ds}: {e}", flush=True)
    print("\nALL DONE", flush=True)
