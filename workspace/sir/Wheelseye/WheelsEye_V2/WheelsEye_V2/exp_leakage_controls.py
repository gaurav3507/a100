"""Tier-1 #4: controlled leakage experiments (LightGBM).
(a) Block-permuted labels: each 4-min block gets a random label (constant within block) -> under
    window-mixed folds (R1) a model that memorises temporal neighbours still scores high; under
    blocked folds (R2) it must fall to chance.  (b) Subject re-identification from a single window.
Usage: python exp_leakage_controls.py
"""
import argparse, numpy as np, pandas as pd
from tier1_common import resolve_groups, window_folds, block_folds, fit_predict, metrics, clean, save

ap = argparse.ArgumentParser()
ap.add_argument("--features-path", default="data/processed/uldd_features.parquet")
ap.add_argument("--groups-json"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-splits", type=int, default=5)
a = ap.parse_args()
df = pd.read_parquet(a.features_path).reset_index(drop=True); groups, ctx = resolve_groups(df, a.groups_json)
feats = sum(groups.values(), []) + ctx; X = clean(df[feats]); rng = np.random.default_rng(a.seed)

# permuted labels: one random label per (subject, session, 4-min block), constant within the block
blk = df.sort_values(["subject", "session", "window_id"]).groupby(["subject", "session"]).cumcount() // 48
key = df["subject"].astype(str) + "|" + df["session"].astype(str) + "|" + blk.reindex(df.index).astype(str)
perm_map = {k: int(v) for k, v in zip(key.unique(), rng.integers(0, 3, size=key.nunique()))}
y_perm = key.map(perm_map).to_numpy(); y_true = df.label.to_numpy()
subj_codes, y_subj = np.unique(df.subject, return_inverse=True)

def cv(fold, y, tag, n_classes, extra):
    per_fold = []
    for k in sorted(set(fold) - {-1}):
        tr, te = fold != k, fold == k
        if k == -1: continue
        tr &= fold != -1
        y_pred = fit_predict(X[tr], y[tr], X[te], a.seed)
        per_fold.append({"fold": int(k), "label": f"fold={k}", "metrics": metrics(y[te], y_pred, n_classes, names=[str(i) for i in range(n_classes)])})
    save(tag, {"experiment": "leakage_controls", **extra}, per_fold)

f1 = window_folds(df, a.n_splits, a.seed); f2 = block_folds(df, a.n_splits)
print("LEAKAGE CONTROLS (LightGBM)\n-- (a) true labels vs block-permuted labels --")
cv(f1, y_true, "leak_R1_true_labels", 3, {"regime": "R1_window_mixed", "labels": "true"})
cv(f1, y_perm, "leak_R1_permuted_labels", 3, {"regime": "R1_window_mixed", "labels": "block_permuted"})
cv(f2, y_true, "leak_R2_true_labels", 3, {"regime": "R2_blocked", "labels": "true"})
cv(f2, y_perm, "leak_R2_permuted_labels", 3, {"regime": "R2_blocked", "labels": "block_permuted"})
print("-- (b) subject re-identification from one window (%d subjects; chance=%.1f%%) --" % (len(subj_codes), 100 / len(subj_codes)))
cv(f1, y_subj, "leak_R1_subject_reid", len(subj_codes), {"regime": "R1_window_mixed", "target": "subject_id"})
cv(f2, y_subj, "leak_R2_subject_reid", len(subj_codes), {"regime": "R2_blocked", "target": "subject_id"})
print("\nReading: high accuracy on PERMUTED labels under R1 = temporal-neighbour memorisation; it must drop to ~33% under R2.")
