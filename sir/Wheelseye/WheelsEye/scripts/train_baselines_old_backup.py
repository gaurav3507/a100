"""Classical LOSO baselines: majority, stratified, SVM (RBF), Random Forest.

Model configurations are taken verbatim from the UL-DD dataset paper's own
technical validation (Bodaghi et al., Sci Data 13:506, 2026, "Validation of
Multimodal Data"):

    SVM : RBF kernel, C=0.1, gamma='auto', class_weight='balanced'
    RF  : max_depth=5, n_estimators=100, min_samples_leaf=10

The paper evaluated these under 5-fold *stratified* cross-validation, which
places windows from the same subject in both train and test. This script
re-runs the identical configurations under this project's subject-independent
LOSO folds. That protocol difference is the entire point of the comparison --
any accuracy gap is attributable to subject-independence, not to the model.

Preprocessing deliberately differs from the paper. They resampled raw signals
to 4 Hz, interpolated, MinMax-scaled and applied PCA (90% variance). We feed
this project's 10s-window engineered features instead, so the comparison
isolates the classifier rather than the representation. MinMax scaling is kept
(it matches the paper and is monotonic per-feature, so tree results are
unaffected); --pca additionally restores their PCA step.

Every scaler, imputer and PCA basis is fit on the training subjects ONLY --
fitting on the full table would leak the held-out subject.

Usage:
    python scripts/train_baselines.py --model majority
    python scripts/train_baselines.py --model rf
    python scripts/train_baselines.py --model svm --modality vision
    python scripts/train_baselines.py --model all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.dataset import DEFAULT_FEATURES_PATH, feature_groups
from common.metrics import Metrics, aggregate, compute_metrics
from common.splits import load_folds

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "baseline_results"

MODELS = ("majority", "stratified", "svm", "rf")


def build_model(name: str, seed: int, n_jobs: int):
    """Instantiate a baseline. SVM/RF configs are the UL-DD paper's."""
    if name == "majority":
        from sklearn.dummy import DummyClassifier
        return DummyClassifier(strategy="most_frequent")
    if name == "stratified":
        from sklearn.dummy import DummyClassifier
        return DummyClassifier(strategy="stratified", random_state=seed)
    if name == "svm":
        from sklearn.svm import SVC
        return SVC(kernel="rbf", C=0.1, gamma="auto", class_weight="balanced",
                   cache_size=1000, random_state=seed)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(max_depth=5, n_estimators=100,
                                      min_samples_leaf=10, random_state=seed,
                                      n_jobs=n_jobs)
    raise ValueError(f"unknown model {name!r}")


def select_features(modality: str) -> list[str]:
    """Feature columns for a modality, or all three nodes concatenated.

    Telemetry is excluded throughout: no deployed sensor produces it (see
    PROGRESS.md), so including it would make these baselines incomparable
    to Track B.
    """
    groups = feature_groups(include_telemetry=False)
    if modality == "all":
        return [c for cols in groups.values() for c in cols]
    if modality not in groups:
        raise ValueError(f"unknown modality {modality!r}, expected one of {list(groups)} or 'all'")
    return list(groups[modality])


def prepare_fold(train_df, test_df, features, use_pca: bool, seed: int):
    """Impute -> scale -> (optionally) PCA, all fit on training rows only."""
    from sklearn.preprocessing import MinMaxScaler

    # Median impute, matching track_b_light/train.py's LightGBM path. A
    # column that is entirely NaN across the training subjects has a NaN
    # median; fall back to 0.0 so the array stays finite.
    medians = train_df[features].median()
    fill = {c: (0.0 if pd.isna(medians[c]) else float(medians[c])) for c in features}

    X_train = train_df[features].fillna(value=fill).to_numpy(dtype=np.float64)
    X_test = test_df[features].fillna(value=fill).to_numpy(dtype=np.float64)

    # Guard against any residual non-finite values (e.g. inf from a ratio
    # feature) that fillna would not catch.
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = MinMaxScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    if use_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.90, random_state=seed).fit(X_train)
        X_train, X_test = pca.transform(X_train), pca.transform(X_test)

    return X_train, X_test


def run_fold(model_name, held_out_subject, df, features, args) -> tuple[Metrics, int]:
    train_df = df[df["subject"] != held_out_subject]
    test_df = df[df["subject"] == held_out_subject]
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(f"empty split for held-out subject {held_out_subject!r}")

    X_train, X_test = prepare_fold(train_df, test_df, features, args.pca, args.seed)
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    model = build_model(model_name, args.seed, args.n_jobs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return compute_metrics(y_test, y_pred), X_train.shape[1]


def run_model(model_name: str, df, features, folds, args) -> dict:
    print(f"\n=== {model_name}  (modality={args.modality}, {len(features)} features"
          f"{', PCA 90%' if args.pca else ''}) ===")
    fold_metrics = []
    for f in folds:
        t0 = time.time()
        metrics, n_dims = run_fold(model_name, f["held_out_subject"], df, features, args)
        fold_metrics.append(metrics)
        print(f"  fold {f['fold']} (held out {f['held_out_subject']}): "
              f"acc={metrics.accuracy:.4f} f1_macro={metrics.f1_macro:.4f} "
              f"dims={n_dims} ({time.time()-t0:.1f}s)")

    agg = aggregate(fold_metrics)
    print(f"{model_name} aggregate over {len(fold_metrics)} folds: "
          f"acc={agg['accuracy']['mean']:.4f}+/-{agg['accuracy']['std']:.4f} "
          f"f1_macro={agg['f1_macro']['mean']:.4f}+/-{agg['f1_macro']['std']:.4f}")

    agg["model"] = model_name
    agg["modality"] = args.modality
    agg["n_features"] = len(features)
    agg["pca"] = args.pca
    return agg


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=[*MODELS, "all"], required=True)
    parser.add_argument("--modality", default="all",
                        help="all | vision | bio | grip")
    parser.add_argument("--folds", type=int, nargs="*", default=None,
                        help="subset of fold ids to REPORT; training always uses every other subject")
    parser.add_argument("--pca", action="store_true",
                        help="apply PCA (90%% variance) after scaling, as the UL-DD paper did")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=8,
                        help="sklearn parallelism; keep modest when a GPU job shares the node")
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH)
    args = parser.parse_args()

    features = select_features(args.modality)

    all_folds = load_folds()
    folds = [f for f in all_folds if f["fold"] in args.folds] if args.folds else all_folds
    if not folds:
        raise SystemExit(f"no folds matched {args.folds}; available: {[f['fold'] for f in all_folds]}")
    all_fold_ids = [f["fold"] for f in all_folds]

    # Load ALL foldable subjects regardless of --folds. Each fold's training
    # set must be every other subject; filtering the table to the requested
    # folds would silently starve training (the bug NOTES.md records for the
    # three train.py entry points).
    df = pd.read_parquet(args.features_path)
    df = df[df["fold"].isin(all_fold_ids)]

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise SystemExit(f"feature table is missing {len(missing)} columns: {missing[:5]}")

    print(f"features: {args.features_path}")
    print(f"rows: {len(df)}  subjects: {df['subject'].nunique()}  folds reported: {len(folds)}")

    names = list(MODELS) if args.model == "all" else [args.model]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in names:
        agg = run_model(name, df, features, folds, args)
        summary[name] = agg
        suffix = f"_{args.modality}" if args.modality != "all" else ""
        suffix += "_pca" if args.pca else ""
        (RESULTS_DIR / f"{name}{suffix}.json").write_text(json.dumps(agg, indent=2))

    if len(names) > 1:
        print(f"\n{'Model':<14}{'Accuracy':>11}{'F1 (macro)':>13}{'n_folds':>9}")
        print("-" * 47)
        for name, agg in summary.items():
            print(f"{name:<14}{agg['accuracy']['mean']*100:>10.2f}%"
                  f"{agg['f1_macro']['mean']*100:>12.2f}%{agg['n_runs']:>9}")

    print(f"\nSaved to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
