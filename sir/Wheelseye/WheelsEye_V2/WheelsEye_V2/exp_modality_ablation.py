"""Tier-1 #1: modality ablation UNDER calibration (LightGBM, LOSO, 8-min enrollment).
Answers: does vision or grip add anything over bio once per-driver calibration is applied?
Usage: python exp_modality_ablation.py [--k-enroll 96] [--groups-json groups.json]
"""
import argparse, itertools
import pandas as pd
from tier1_common import resolve_groups, loso_plan, calibrate, fit_predict, metrics, clean, save

ap = argparse.ArgumentParser()
ap.add_argument("--features-path", default="data/processed/uldd_features.parquet")
ap.add_argument("--k-enroll", type=int, default=96)
ap.add_argument("--groups-json"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cal-mode", choices=["meanstd", "mean"], default="mean", help="enrollment calibration variant (sensor features only)")
a = ap.parse_args()

df = pd.read_parquet(a.features_path)
groups, ctx = resolve_groups(df, a.groups_json)
print("groups:", {k: len(v) for k, v in groups.items()}, "context:", len(ctx))
names = [g for g in groups if groups[g]]
subsets = {g: groups[g] for g in names}
for r in range(2, len(names)):
    for combo in itertools.combinations(names, r):
        subsets["+".join(combo)] = sum((groups[g] for g in combo), [])
subsets["all"] = sum((groups[g] for g in names), [])
subsets["all+context"] = subsets["all"] + ctx

plan = loso_plan(df)
all_feats = sorted(set(sum(subsets.values(), [])))
sensor_all = sum((groups[g] for g in names), [])
cal = calibrate(df, all_feats, "enroll" if a.cal_mode == "meanstd" else "enroll_mean", a.k_enroll, cal_feats=sensor_all)  # sensor-only, once
print(f"\nMODALITY ABLATION -- LOSO ({len(plan)} folds), {a.k_enroll}-window enrollment calibration ({a.cal_mode}, sensor-only)")
for name, feats in subsets.items():
    per_fold = []
    for p in plan:
        tr, te = cal[cal.subject != p["held_out"]], cal[cal.subject == p["held_out"]]
        y_pred = fit_predict(clean(tr[feats]), tr.label.to_numpy(), clean(te[feats]), a.seed)
        per_fold.append({"fold": p["fold"], "label": p["label"], "metrics": metrics(te.label.to_numpy(), y_pred)})
    save(f"tier3_ablation_{name.replace('+','_')}_enroll{a.k_enroll}" + ("_mean" if a.cal_mode == "mean" else ""),
         {"experiment": "modality_ablation", "subset": name, "n_features": len(feats), "k_enroll": a.k_enroll, "cal_mode": a.cal_mode, "model": "lightgbm"}, per_fold)
print("\nNext: python paired_tests.py --ref data/processed/results/tier3_ablation_all_context_enroll%d.json data/processed/results/tier3_ablation_bio_enroll%d.json" % (a.k_enroll, a.k_enroll))
