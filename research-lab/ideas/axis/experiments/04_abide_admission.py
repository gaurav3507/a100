"""AXIS real-data run: ABIDE site admission test (Option A, pool-within-site).

Hypothesis: acquisition sites shift the MEASUREMENT axis (diagonal VAR coeffs),
not the MECHANISM axis (off-diagonal). So dissociate() should return a LOW
offdiag_share (change concentrated on the diagonal).

Method:
  - match sample size: cap both sites to equal subject counts (fixed seed)
  - per-region z-score WITHIN each site (removes trivial scale diff; raw
    per-site variance ratio ~2.8 would otherwise fake a diagonal difference)
  - pool subjects into one series per site, DROPPING the seam at each subject
    boundary so no VAR(1) pair spans two subjects
  - fit one VAR(1) per site (well-determined: ~9k rows vs 200 predictors)
  - bootstrap SE by resampling whole subjects (block = one subject)
  - hand both matrices to the VALIDATED dissociate() (same code the gates used)

CAVEAT (printed): two sites differ in many ways at once. A diagonal-dominant
result is consistent with a measurement-axis shift but does not prove only
measurement shifted. Evidence, not proof.
"""
import sys, os, json, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from dissociation import dissociate

NPZ = "/workspace/ranktest-diagnostics/data/abide_harmonized.npz"
SITE_A, SITE_B = "NYU", "UM_1"
SEED = 0
N_BOOT = 500
THRESHOLD = 3.0

def zscore_per_region(X):                 # X: (n_subj, T, d)
    mu = X.mean(axis=(0, 1), keepdims=True)
    sd = X.std(axis=(0, 1), keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd

def pooled_design_drop_seams(Xs):         # Xs: (n_subj, T, d) -> (Z, Y)
    Zs, Ys = [], []
    for s in range(Xs.shape[0]):
        x = Xs[s]
        Zs.append(x[:-1]); Ys.append(x[1:])   # seam dropped between subjects
    return np.vstack(Zs), np.vstack(Ys)

def fit_A(Xs):
    Z, Y = pooled_design_drop_seams(Xs)
    B, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
    return B.T

def fit_site(Xs):
    A = fit_A(Xs)
    rng = np.random.default_rng(SEED)
    n = Xs.shape[0]
    boots = [fit_A(Xs[rng.integers(0, n, n)]) for _ in range(N_BOOT)]
    A_se = np.std(np.stack(boots), axis=0)
    return A, A_se

def main():
    d = np.load(NPZ, allow_pickle=True)
    X, site = d["X"], d["site_ids"]
    h = hashlib.sha256(X.tobytes()).hexdigest()[:16]

    XA, XB = X[site == SITE_A], X[site == SITE_B]
    n = min(len(XA), len(XB))
    rng = np.random.default_rng(SEED)
    XA = XA[rng.permutation(len(XA))[:n]]
    XB = XB[rng.permutation(len(XB))[:n]]
    print(f"{SITE_A}: {n} subjects | {SITE_B}: {n} subjects | matched (subjects)")

    XA, XB = zscore_per_region(XA), zscore_per_region(XB)

    AA, AA_se = fit_site(XA)
    AB, AB_se = fit_site(XB)

    # matched n for dissociate: equal subjects -> equal pooled rows
    nrows = XA.shape[0] * (XA.shape[1] - 1)
    ratio, summary = dissociate(AA, AA_se, nrows, AB, AB_se, nrows, threshold=THRESHOLD)

    print(f"threshold={THRESHOLD}  d={summary['d']}  matched_rows={nrows}")
    print(f"  diag rejections   : {summary['diag_reject']}/{summary['diag_total']}   rate={summary['diag_rate']:.4f}")
    print(f"  offdiag rejections: {summary['offdiag_reject']}/{summary['offdiag_total']}   rate={summary['offdiag_rate']:.4f}")
    print(f"  offdiag_share     : {ratio}   (0=measurement/diagonal, 1=mechanism/off-diagonal)")
    print(f"  offdiag/diag rate ratio: {summary['offdiag_to_diag_rate_ratio']}")
    if ratio is None:
        verdict = "NO SIGNAL (no rejections at all)"
    elif ratio < 0.34:
        verdict = "DIAGONAL-DOMINANT -> consistent with MEASUREMENT-axis shift (hypothesis supported)"
    elif ratio > 0.66:
        verdict = "OFF-DIAGONAL-DOMINANT -> MECHANISM shift (hypothesis NOT supported)"
    else:
        verdict = "MIXED -> inconclusive for this pair"
    print(f"  VERDICT: {verdict}")
    print("  CAVEAT: two sites differ in many ways at once; evidence, not proof.")

    out = {"sites": [SITE_A, SITE_B], "seed": SEED, "n_boot": N_BOOT,
           "matched_subjects": int(n), "npz": NPZ, "X_hash": h,
           "verdict": verdict, **summary}
    op = os.path.join(HERE, "..", "results", "04_abide_admission.json")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f: json.dump(out, f, indent=2)
    print(f"wrote {op}")

if __name__ == "__main__":
    main()
