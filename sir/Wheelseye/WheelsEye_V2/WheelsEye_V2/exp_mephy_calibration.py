"""Tier-1 #3: replicate the calibration effect on a second dataset (MePhy).
Per-subject baseline from the subject's REST windows (label == --rest-label), first k windows only
(enrollment) or all rest windows; LOSO over users. Requires the MePhy feature table built by the
first-generation pipeline (default path below; override with --features-path).
Usage: python exp_mephy_calibration.py [--features-path PATH] [--rest-label 0] [--k-enroll 24]
"""
import argparse, numpy as np, pandas as pd
from tier1_common import fit_predict, metrics, clean, save, META

ap = argparse.ArgumentParser()
ap.add_argument("--features-path", default="../../WheelsEye/data/processed/mephy_features.parquet")
ap.add_argument("--rest-label", type=int, default=0); ap.add_argument("--k-enroll", type=int, default=24)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
df = pd.read_parquet(a.features_path)
ucol = next(c for c in ("user", "subject", "participant") if c in df.columns)
ocol = next((c for c in ("window_id", "window_start", "start", "t0") if c in df.columns), None)
feats = [c for c in df.columns if c not in META | {ucol, "condition", ocol} and pd.api.types.is_numeric_dtype(df[c])]
n_classes = int(df.label.max()) + 1
print(f"MePhy: {len(df)} windows, {df[ucol].nunique()} users, {len(feats)} features, {n_classes} classes; "
      f"label distribution {df.label.value_counts().sort_index().to_dict()}; rest label = {a.rest_label}")
users = sorted(u for u, g in df.groupby(ucol) if (g.label == a.rest_label).sum() >= 2)
print(f"evaluating {len(users)} users with rest windows")
print("rest windows per user (label==rest):", df[df.label == a.rest_label].groupby(ucol).size().to_dict())
print("VERIFY the rest label against the v1 mapping: grep -n 'rest\\|label' ../../WheelsEye/mephy_repro/dataset.py")

def calibrated(mode, mean_only=False):
    out = df.copy(); out[feats] = out[feats].astype("float64")
    if mode == "none": return out
    for u, g in df.groupby(ucol):
        src = g[g.label == a.rest_label]
        if ocol: src = src.sort_values(ocol)
        if mode == "enroll": src = src.head(a.k_enroll)
        if len(src) < 2: continue
        mean = src[feats].mean().fillna(0.0); std = src[feats].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        if mean_only: std = std * 0 + 1.0
        m = (out[ucol] == u).to_numpy(); out.loc[m, feats] = ((out.loc[m, feats] - mean) / std).to_numpy()
    return out

for mode, mo, tag in (("none", False, "mephy_loso_none"), ("enroll", False, f"mephy_loso_rest_enroll{a.k_enroll}"),
                      ("enroll", True, f"mephy_loso_rest_enroll{a.k_enroll}_mean"), ("all_rest", True, "mephy_loso_rest_all_mean")):
    fr = calibrated(mode, mo); per_fold = []
    for i, u in enumerate(users):
        tr, te = fr[fr[ucol] != u], fr[fr[ucol] == u]
        y_pred = fit_predict(clean(tr[feats]), tr.label.to_numpy(), clean(te[feats]), a.seed)
        per_fold.append({"fold": i, "label": f"held_out={u}", "metrics": metrics(te.label.to_numpy(), y_pred, n_classes, names=[f"c{k}" for k in range(n_classes)])})
    save(tag, {"experiment": "mephy_calibration", "variant": mode + ("_mean" if mo else ""), "k_enroll": a.k_enroll, "rest_label": a.rest_label, "model": "lightgbm"}, per_fold)
print("\nNext: python paired_tests.py --ref data/processed/results/mephy_loso_rest_enroll%d.json data/processed/results/mephy_loso_none.json" % a.k_enroll)
