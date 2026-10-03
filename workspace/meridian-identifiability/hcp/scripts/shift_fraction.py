"""How much does cognitive condition shift latent VAR dynamics on HCP task fMRI?

Measures K/d: the fraction of latent VAR(1) coefficients that differ between task
pairs BEYOND within-task split-half estimation noise. The synthetic ceiling probes
established identifiability needs K/d <~ 0.4 with d >= 20, so this asks whether real
task contrasts land inside that regime.

The split-half null is load-bearing: without it every coefficient looks "shifted"
because VAR fits from a few hundred frames are noisy. Differences are only counted
as condition-driven if they exceed the same-task split-half difference distribution.

Usage: python shift_fraction.py [--enc LR] [--dims 10,20,30,50]
"""
import argparse, glob, itertools, json, os
import numpy as np

TS = "/workspace/hcp/ts"
OUT = "/workspace/hcp/results"
TASKS = ["WM", "GAMBLING", "MOTOR", "LANGUAGE", "SOCIAL", "RELATIONAL", "EMOTION"]
RIDGE = 1e-2

def load(subj, task, enc):
    p = os.path.join(TS, f"{subj}_{task}_{enc}.npy")
    return np.load(p) if os.path.exists(p) else None

def znorm(x):
    return (x - x.mean(0)) / (x.std(0) + 1e-8)

def fit_var(z, ridge=RIDGE):
    """z: (T, d) -> B: (d, d) predicting z[t] from z[t-1]."""
    X, Y = z[:-1], z[1:]
    G = X.T @ X + ridge * len(X) * np.eye(z.shape[1])
    return np.linalg.solve(G, X.T @ Y).T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="LR")
    ap.add_argument("--dims", default="10,20,30,50")
    a = ap.parse_args()
    dims = [int(x) for x in a.dims.split(",")]
    os.makedirs(OUT, exist_ok=True)

    subs = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(f"{TS}/*.npy")})
    subs = [s for s in subs if all(load(s, t, a.enc) is not None for t in TASKS)]
    print(f"{len(subs)} subjects with all {len(TASKS)} tasks (enc {a.enc})")

    # shared PCA basis across all tasks/subjects so latent coords are comparable
    pool = np.concatenate([znorm(load(s, t, a.enc)) for s in subs for t in TASKS], 0)
    pool -= pool.mean(0)
    _, _, Vt = np.linalg.svd(pool, full_matrices=False)
    print(f"pooled data {pool.shape}, PCA basis built")

    results = {}
    for d in dims:
        W = Vt[:d].T                                   # (200, d)
        between, within = [], []
        for s in subs:
            Bs, Bh = {}, {}
            for t in TASKS:
                z = znorm(load(s, t, a.enc)) @ W
                Bs[t] = fit_var(z)
                h = len(z) // 2
                Bh[t] = (fit_var(z[:h]), fit_var(z[h:]))   # split-half null
            # SAMPLE-SIZE MATCHED: both null and signal use half-length fits.
            # Comparing full-run between-task fits against half-run within-task fits
            # inflates the null with estimation noise and is invalid.
            for t in TASKS:
                within.append(np.abs(Bh[t][0] - Bh[t][1]).ravel())
            for t1, t2 in itertools.combinations(TASKS, 2):
                between.append(np.abs(Bh[t1][0] - Bh[t2][0]).ravel())
                between.append(np.abs(Bh[t1][1] - Bh[t2][1]).ravel())
        within = np.concatenate(within)
        between = np.concatenate(between)
        thr = np.percentile(within, 95)                 # null 95th pct
        frac = float((between > thr).mean())
        results[d] = dict(shift_fraction=frac, null_thr=float(thr),
                          within_median=float(np.median(within)),
                          between_median=float(np.median(between)),
                          ratio=float(np.median(between) / (np.median(within) + 1e-12)))
        verdict = ("INSIDE identifiable regime (K/d < 0.4)" if frac < 0.4 else
                   "OUTSIDE: tasks shift too much" if frac > 0.6 else "BORDERLINE")
        print(f"  d={d:3d}: shift fraction {frac:.3f}  "
              f"(between/within median ratio {results[d]['ratio']:.2f})  -> {verdict}")

    with open(os.path.join(OUT, f"shift_fraction_{a.enc}.json"), "w") as f:
        json.dump(dict(enc=a.enc, n_subjects=len(subs), tasks=TASKS, results=results), f, indent=2)
    print(f"\nwrote {OUT}/shift_fraction_{a.enc}.json")
    print("READ: shift fraction is K/d on real data. Synthetic work needs < ~0.4 at d >= 20.")
    print("Ratio near 1.0 = tasks differ no more than split-half noise (suspicious, check pipeline).")

if __name__ == "__main__":
    main()
