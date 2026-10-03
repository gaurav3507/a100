"""Decisive run: MEAN-ONLY per-driver calibration under V2's convention (sensor features calibrated,
context/drift features untouched). Produces: (i) a reconciliation check against V2's reported
mean+std numbers, (ii) the mean-only enrollment curve incl. whole-session, (iii) the transductive
mean-only baseline, (iv) the supervised labeled-enrollment upper bound on the mean-only base.
Usage: python exp_meanonly.py [--k-enroll 96]
"""
import argparse, pandas as pd
from tier1_common import resolve_groups, loso_plan, calibrate, fit_predict, metrics, clean, save

ap = argparse.ArgumentParser()
ap.add_argument("--features-path", default="data/processed/uldd_features.parquet")
ap.add_argument("--k-enroll", type=int, default=96); ap.add_argument("--groups-json"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--curve", type=int, nargs="*", default=[12, 24, 48, 96, 144, 240, 360, 420, 460])
a = ap.parse_args()
df = pd.read_parquet(a.features_path); groups, ctx = resolve_groups(df, a.groups_json)
sensor = sum(groups.values(), []); feats = sensor + ctx; plan = loso_plan(df)
print(f"MEAN-ONLY CALIBRATION -- LOSO ({len(plan)} folds); {len(sensor)} sensor features calibrated, {len(ctx)} context features untouched")

def run(tag, frame, extra, supervised=False):
    per_fold = []
    for p in plan:
        tr, te = frame[frame.subject != p["held_out"]], frame[frame.subject == p["held_out"]]
        if supervised:
            idx = te.sort_values("window_id").groupby("session").head(a.k_enroll).index
            tr = pd.concat([tr, te.loc[idx]]); te = te.drop(idx)
        y_pred = fit_predict(clean(tr[feats]), tr.label.to_numpy(), clean(te[feats]), a.seed)
        per_fold.append({"fold": p["fold"], "label": p["label"], "metrics": metrics(te.label.to_numpy(), y_pred)})
    save(tag, {"experiment": "meanonly", "model": "lightgbm", "calibrated_features": "sensor_only", **extra}, per_fold)

print("\n-- (i) reconciliation with V2 (expected: enroll96 mean+std ~65.67, whole-session mean+std ~73.69) --")
run(f"tier3_cal2_meanstd_enroll{a.k_enroll}", calibrate(df, feats, "enroll", a.k_enroll, sensor), {"variant": "meanstd_enroll", "k": a.k_enroll})
run("tier3_cal2_meanstd_whole", calibrate(df, feats, "whole_awake", cal_feats=sensor), {"variant": "meanstd_whole"})
print("\n-- (ii) mean-only enrollment curve --")
for k in a.curve:
    run(f"tier3_cal2_mean_enroll{k}", calibrate(df, feats, "enroll_mean", k, sensor), {"variant": "mean_enroll", "k": k})
run("tier3_cal2_mean_whole", calibrate(df, feats, "whole_awake_mean", cal_feats=sensor), {"variant": "mean_whole"})
print("\n-- (iii) transductive mean-only baseline (both sessions) --")
run("tier3_cal2_mean_both_sessions", calibrate(df, feats, "both_sessions_mean", cal_feats=sensor), {"variant": "mean_both_sessions_transductive"})
print("\n-- (iv) supervised labeled-enrollment upper bound on the mean-only base --")
run(f"tier3_cal2_supervised_mean_enroll{a.k_enroll}", calibrate(df, feats, "enroll_mean", a.k_enroll, sensor), {"variant": "supervised_upperbound_mean", "k": a.k_enroll}, supervised=True)
print("\nNext: python paired_tests.py --ref data/processed/results/tier3_cal2_mean_enroll%d.json data/processed/results/tier3_cal2_meanstd_enroll%d.json data/processed/results/tier3_calib_none.json data/processed/results/tier3_cal2_meanstd_whole.json data/processed/results/tier3_i2m2_v2_cal_smooth3_seed0.json data/processed/results/tier3_i2m2_v2_seed0.json" % (a.k_enroll, a.k_enroll))
