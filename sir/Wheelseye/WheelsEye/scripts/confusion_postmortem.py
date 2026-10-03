"""Confusion-matrix post-mortem for a baseline under LOSO.

Re-runs the chosen model with scripts/train_baselines.py's own (tested)
prepare_fold/build_model, pooling every fold's test predictions, then prints
the pooled confusion matrix, per-class precision/recall, and the worst
per-fold collapses. Read it to answer: WHERE do the ~43% errors of the
champion (svm --modality bio) actually go?

Usage:
    python scripts/confusion_postmortem.py --model svm --modality bio
    python scripts/confusion_postmortem.py --model lightgbm
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.dataset import DEFAULT_FEATURES_PATH
from common.metrics import compute_metrics
from common.splits import load_folds
from scripts.train_baselines import build_model, prepare_fold, select_features

CLASS_NAMES = ["Low", "Medium", "High"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="svm")
    p.add_argument("--modality", default="bio")
    p.add_argument("--pca", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    args = p.parse_args()

    features = select_features(args.modality)
    folds = load_folds()
    all_ids = [f["fold"] for f in folds]
    df = pd.read_parquet(args.features_path)
    df = df[df["fold"].isin(all_ids + [-1])]          # Option A cohort

    y_true_all, y_pred_all, per_fold = [], [], []
    for f in folds:
        held = f["held_out_subject"]
        train_df = df[df["subject"] != held]
        test_df = df[df["subject"] == held]
        X_tr, X_te = prepare_fold(train_df, test_df, features, args.pca, args.seed)
        model = build_model(args.model, args.seed, args.n_jobs)
        t0 = time.time()
        model.fit(X_tr, train_df["label"].to_numpy())
        y_pred = model.predict(X_te)
        y_true = test_df["label"].to_numpy()
        m = compute_metrics(y_true, y_pred)
        per_fold.append((held, m.accuracy, np.bincount(y_true, minlength=3),
                         np.bincount(y_pred, minlength=3)))
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)
        print(f"  fold {f['fold']} ({held}): acc={m.accuracy:.3f} ({time.time()-t0:.0f}s)")

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    pooled = compute_metrics(y_true, y_pred)

    print(f"\npooled LOSO accuracy: {pooled.accuracy:.4f}   f1_macro: {pooled.f1_macro:.4f}")
    print(f"\nPooled confusion matrix (rows = true, cols = predicted):")
    cm = np.asarray(pooled.confusion)
    hdr = "".join(f"{c:>9}" for c in CLASS_NAMES)
    print(f"{'':>9}{hdr}{'row_acc':>9}")
    for i, cname in enumerate(CLASS_NAMES):
        row = "".join(f"{cm[i, j]:>9d}" for j in range(len(CLASS_NAMES)))
        ra = cm[i, i] / cm[i].sum() if cm[i].sum() else float("nan")
        print(f"{cname:>9}{row}{ra:>9.3f}")

    print("\nPer-class precision / recall / F1:")
    for i, cname in enumerate(CLASS_NAMES):
        print(f"  {cname:<8} P={pooled.per_class_precision[i]:.3f}  "
              f"R={pooled.per_class_recall[i]:.3f}  F1={pooled.per_class_f1[i]:.3f}")

    print("\nWorst folds (by accuracy) with true vs predicted class counts:")
    for held, acc, tc, pc in sorted(per_fold, key=lambda r: r[1])[:4]:
        print(f"  {held}: acc={acc:.3f}  true L/M/H={tc.tolist()}  pred L/M/H={pc.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
