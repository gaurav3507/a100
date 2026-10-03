"""Tier-1 #2: calibration baselines (LightGBM, LOSO). Variants:
  none | enroll (mean+std, first k Awake) | enroll_mean (mean only) | whole_awake |
  both_sessions (transductive z-score on all of the subject's windows) | coral (test-time covariance
  alignment, transductive) | supervised_k (labeled enrollment: first k windows of BOTH sessions join
  training with labels -- upper bound; those windows are excluded from scoring)
Usage: python exp_calibration_baselines.py [--k-enroll 96]
"""
import argparse, numpy as np, pandas as pd
from tier1_common import resolve_groups, loso_plan, calibrate, coral_align, fit_predict, metrics, clean, save

ap = argparse.ArgumentParser()
ap.add_argument("--features-path", default="data/processed/uldd_features.parquet")
ap.add_argument("--k-enroll", type=int, default=96); ap.add_argument("--groups-json"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
df = pd.read_parquet(a.features_path); groups, ctx = resolve_groups(df, a.groups_json)
feats = sum(groups.values(), []) + ctx; plan = loso_plan(df)
print(f"CALIBRATION BASELINES -- LOSO ({len(plan)} folds), {len(feats)} features, k={a.k_enroll}")

def run(tag, frame, extra=None, exclude_enroll=False, coral=False):
    per_fold = []
    for p in plan:
        tr, te = frame[frame.subject != p["held_out"]], frame[frame.subject == p["held_out"]]
        if exclude_enroll:  # supervised: first k of each test-subject session join training WITH labels
            enroll_idx = te.sort_values("window_id").groupby("session").head(a.k_enroll).index
            tr = pd.concat([tr, te.loc[enroll_idx]]); te = te.drop(enroll_idx)
        Xtr, Xte = clean(tr[feats]), clean(te[feats])
        if coral: Xte = coral_align(Xtr, Xte).astype(np.float32)
        y_pred = fit_predict(Xtr, tr.label.to_numpy(), Xte, a.seed)
        per_fold.append({"fold": p["fold"], "label": p["label"], "metrics": metrics(te.label.to_numpy(), y_pred)})
    save(tag, {"experiment": "calibration_baselines", "variant": tag, "k_enroll": a.k_enroll, "model": "lightgbm", **(extra or {})}, per_fold)

run("tier3_calib_none", calibrate(df, feats, "none"))
run(f"tier3_calib_enroll{a.k_enroll}_meanstd", calibrate(df, feats, "enroll", a.k_enroll))
run(f"tier3_calib_enroll{a.k_enroll}_meanonly", calibrate(df, feats, "enroll_mean", a.k_enroll))
run("tier3_calib_whole_awake", calibrate(df, feats, "whole_awake"))
run("tier3_calib_both_sessions_transductive", calibrate(df, feats, "both_sessions"))
run("tier3_calib_coral_transductive", calibrate(df, feats, "none"), coral=True)
run(f"tier3_calib_supervised_enroll{a.k_enroll}_upperbound", calibrate(df, feats, "enroll", a.k_enroll), exclude_enroll=True)
print("\nNext: python paired_tests.py --ref data/processed/results/tier3_calib_enroll%d_meanstd.json --all data/processed/results | grep calib" % a.k_enroll)
