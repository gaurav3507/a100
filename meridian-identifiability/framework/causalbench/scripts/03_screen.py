import os, json, sys
import numpy as np

D = "/workspace/meridian-identifiability/causalbench/data"
R = "/workspace/meridian-identifiability/causalbench/results/screen"
os.makedirs(R, exist_ok=True)

RIDGE   = 1.0
N_PAIRS = 300
SEED    = 0
CTRL    = "non-targeting"

def load(ds, filt):
    name = f"dataset_{ds}" + ("_filtered" if filt else "") + ".npz"
    d = np.load(os.path.join(D, name), allow_pickle=True)
    return (d["expression_matrix"].astype(np.float64),
            np.asarray(d["interventions"]),
            [str(v) for v in d["var_names"]])

def fit_pca(Xc, d):
    mu = Xc.mean(0)
    _, _, Vt = np.linalg.svd(Xc - mu, full_matrices=False)
    return mu, Vt[:d].T                      # p x d

def project(X, mu, W, drop):
    Wm = W.copy()
    if drop: Wm[list(drop), :] = 0.0         # kill targeted gene's contribution
    return (X - mu) @ Wm

def coefs(Z):
    """Static analogue of the VAR matrix: regress each latent on the others."""
    Z = Z - Z.mean(0); d = Z.shape[1]
    B = np.zeros((d, d))
    for j in range(d):
        o = [k for k in range(d) if k != j]
        A = Z[:, o]
        B[j, o] = np.linalg.solve(A.T @ A + RIDGE*np.eye(d-1), A.T @ Z[:, j])
    return B

def offdiag(B):
    d = B.shape[0]
    return B[~np.eye(d, dtype=bool)]

def run(ds, filt, nmin, d, step0=False):
    tag = f"{ds}_filt{int(filt)}_n{nmin}_d{d}" + ("_STEP0" if step0 else "")
    out = os.path.join(R, tag + ".json")
    if os.path.exists(out):
        print(f"SKIP {tag}", flush=True); return json.load(open(out))

    rng = np.random.default_rng(SEED)
    X, iv, vn = load(ds, filt)
    gidx = {g: i for i, g in enumerate(vn)}
    half = nmin // 2

    ctrl_rows = np.where(iv == CTRL)[0]
    mu, W = fit_pca(X[ctrl_rows], d)          # basis from CONTROLS ONLY

    if step0:
        # pseudo-environments carved from the control pool: all same condition
        rng.shuffle(ctrl_rows)
        k = len(ctrl_rows) // nmin
        envs = [(f"ctrl{i}", ctrl_rows[i*nmin:(i+1)*nmin], None) for i in range(k)]
        ref_pool = ctrl_rows[:0]              # reference drawn from envs themselves
    else:
        us, cs = np.unique(iv, return_counts=True)
        keep = [(g, n) for g, n in zip(us, cs)
                if g not in (CTRL, "excluded") and n >= nmin]
        envs = [(g, np.where(iv == g)[0], gidx.get(g)) for g, _ in keep]
        ref_pool = ctrl_rows

    if len(envs) < 4:
        print(f"ABORT {tag}: only {len(envs)} environments"); return None

    within_c, within_m, between_c, between_m = [], [], [], []
    for g, rows, drop in envs:
        rows = rng.choice(rows, nmin, replace=False)
        drops = {drop} if drop is not None else set()
        Z = project(X[rows], mu, W, drops)
        Za, Zb = Z[:half], Z[half:nmin]
        within_c.append(np.abs(offdiag(coefs(Za)) - offdiag(coefs(Zb))))
        within_m.append(np.linalg.norm(Za.mean(0) - Zb.mean(0)))

        ref = rng.choice(ref_pool if len(ref_pool) else rows, half, replace=False)
        Zr = project(X[ref], mu, W, drops)
        between_c.append(np.abs(offdiag(coefs(Za)) - offdiag(coefs(Zr))))
        between_m.append(np.linalg.norm(Za.mean(0) - Zr.mean(0)))

    # environment-vs-environment pairs (analogue of HCP's task pairs)
    pair_c, pair_m = [], []
    for _ in range(min(N_PAIRS, len(envs)*(len(envs)-1)//2)):
        i, j = rng.choice(len(envs), 2, replace=False)
        (ga, ra, da), (gb, rb, db) = envs[i], envs[j]
        drops = {x for x in (da, db) if x is not None}
        Za = project(X[rng.choice(ra, half, replace=False)], mu, W, drops)
        Zb = project(X[rng.choice(rb, half, replace=False)], mu, W, drops)
        pair_c.append(np.abs(offdiag(coefs(Za)) - offdiag(coefs(Zb))))
        pair_m.append(np.linalg.norm(Za.mean(0) - Zb.mean(0)))

    wc = np.concatenate(within_c); bc = np.concatenate(between_c)
    pc = np.concatenate(pair_c) if pair_c else np.array([np.nan])
    thr = np.percentile(wc, 95)

    res = dict(tag=tag, dataset=ds, filter=filt, nmin=nmin, d=d,
               step0=step0, n_envs=len(envs), n_fit=half,
               coef_ratio_vs_ctrl=float(np.median(bc)/np.median(wc)),
               coef_ratio_pairs=float(np.median(pc)/np.median(wc)),
               shift_frac_vs_ctrl=float((bc > thr).mean()),
               shift_frac_pairs=float((pc > thr).mean()),
               shift_frac_within=float((wc > thr).mean()),
               mean_ratio_vs_ctrl=float(np.median(between_m)/np.median(within_m)),
               mean_ratio_pairs=float(np.median(pair_m)/np.median(within_m)))

    tmp = out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=2); os.rename(tmp, out)
    print(json.dumps(res, indent=2), flush=True)
    return res

if __name__ == "__main__":
    print("### STEP 0 GATE — control pseudo-environments, expect ratios ~1.00 ###", flush=True)
    for ds in ["k562", "rpe1"]:
        for d in [5, 10]:
            run(ds, False, 200, d, step0=True)

    print("\n### REAL SCREEN ###", flush=True)
    for ds in ["k562", "rpe1"]:
        for filt in [False, True]:
            for nmin in [200, 100]:
                for d in [5, 10]:
                    try: run(ds, filt, nmin, d)
                    except Exception as e:
                        print(f"FAIL {ds} filt={filt} n={nmin} d={d}: {e}", flush=True)
    print("ALL DONE", flush=True)
