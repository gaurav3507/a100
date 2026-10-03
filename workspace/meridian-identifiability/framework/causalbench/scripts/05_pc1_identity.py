import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

D = "/workspace/meridian-identifiability/causalbench/data"
CTRL, NMIN, SEED = "non-targeting", 200, 0
SHEET = {"k562": "TabB_K562_day6_summary_stat", "rpe1": "TabC_RPE1_summary_statistic"}

for ds in ["k562", "rpe1"]:
    d0 = np.load(os.path.join(D, f"dataset_{ds}.npz"), allow_pickle=True)
    X = d0["expression_matrix"].astype(np.float64)
    iv = np.asarray(d0["interventions"])
    vn = [str(v) for v in d0["var_names"]]; gidx = {g:i for i,g in enumerate(vn)}
    rng = np.random.default_rng(SEED)

    ctrl = np.where(iv == CTRL)[0]
    mu = X[ctrl].mean(0)
    _, _, Vt = np.linalg.svd(X[ctrl] - mu, full_matrices=False)
    W = Vt[:20].T
    us, cs = np.unique(iv, return_counts=True)
    envs = [(g, np.where(iv==g)[0], gidx.get(g))
            for g,n in zip(us,cs) if g not in (CTRL,"excluded") and n >= NMIN]

    ref_mu = ((X[ctrl] - mu) @ W).mean(0)
    rows, names = [], []
    for g, r, t in envs:
        Wm = W.copy()
        if t is not None: Wm[t,:] = 0.0
        rows.append(((X[rng.choice(r,NMIN,replace=False)] - mu) @ Wm).mean(0) - ref_mu)
        names.append(g)
    M = np.array(rows)

    U, S, _ = np.linalg.svd(M - M.mean(0), full_matrices=False)
    load1 = U[:,0]*S[0]

    ss = pd.read_excel(os.path.join(D,"summary_stats.xlsx"), sheet_name=SHEET[ds])
    ss.columns = [c.strip() for c in ss.columns]
    key = [c for c in ss.columns if "genetic perturbation" in c.lower()][0]
    ss["_g"] = ss[key].astype(str).str.split("_").str[-1]
    ss = ss.drop_duplicates("_g").set_index("_g")

    print(f"\n===== {ds} =====")
    print("summary_stats cols:", list(ss.columns)[:12])
    for col in ss.columns:
        lc = col.lower()
        if "deg" in lc or "knockdown" in lc:
            v = pd.to_numeric(ss.reindex(names)[col], errors="coerce").values
            ok = ~np.isnan(v)
            if ok.sum() > 30:
                r_s, p_s = spearmanr(np.abs(load1[ok]), v[ok])
                print(f"  |PC1 loading| vs {col!r}: rho={r_s:+.3f} p={p_s:.2e} n={ok.sum()}")
    print("  sign of loadings: %.0f%% share one sign" %
          (100*max((load1>0).mean(), (load1<0).mean())))
