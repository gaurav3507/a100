"""Phase-1 protocol anchor (plan §3, Phase 1): reproduce the UL-DD paper's
own SVM and RF baselines (Table 8, Bio+Beh+Fac: SVM 83.75%, RF 75.14%)
BEFORE any of our models run on the Tier-1 protocol.

Grounded against the dataset's own Codes/ folder (read 2026-08-19:
4.Multimodal_Analysis.ipynb + 1.Combine_Data_and_Change_Frequency.ipynb +
2_1.Concatenate_Biometric_and_Labeling.ipynb), which reveals four protocol
facts the paper's text does not state:

  1. **Their MinMaxScaler AND PCA are fit on the ENTIRE dataset before
     cross-validation** (notebook 4: `scaler.fit_transform(X)` then
     `pca.fit_transform` then `cross_val_score`). The published baselines
     therefore contain preprocessing leakage -- the same defect class our
     v1 fixed (defect #2). `--prep paper` replicates this exactly (the
     "do we hit their number" arm); `--prep clean` fits scaler/PCA on
     train folds only. The delta between the two arms MEASURES the leakage
     in the published baseline -- a reportable finding.
  2. Their resampling is row decimation / repetition, not interpolation.
     (We keep time-grid linear interpolation -- documented deviation; for
     fixed-rate signals the two nearly coincide.)
  3. IBI is never used by their pipeline (biometrics = ACC, BVP, EDA, HR,
     TEMP, O2M only) -- so it is excluded here too.
  4. Their CV is StratifiedKFold(5, shuffle=True, random_state=42); label
     binning is 0 if KSS<4, 1 if KSS<7, else 2 (identical to ours).

Model hyperparameters copied from their notebook 4 verbatim:
  SVM: RBF, C=0.1, gamma='auto', class_weight='balanced', random_state=42
  RF:  n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42

Other implementation notes: missing whole files (A/Awake telemetry) become
NaN columns for that session, median-imputed (global medians in --prep
paper, train-fold medians in --prep clean) -- their notebook pads A/Alert
telemetry with NaN rows and imputes in an intermediate step not shipped in
Codes/. Memory: ~340k rows x ~290 float32 cols ~= 400 MB.

Usage:
  python scripts/reproduce_uldd_baselines.py --prep paper   # expect ~83.75/75.14
  python scripts/reproduce_uldd_baselines.py --prep clean   # the honest variant
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from common import loaders
from common.labels import available_sessions, get_label
from common.paths import SUBJECTS
from common.windowing import session_duration_seconds

GRID_HZ = 4.0
ANCHORS = {"svm": 83.75, "rf": 75.14}  # UL-DD Table 8, Bio+Beh+Fac rows

# Matches the union used by their notebook 4 (Bio+Beh+Fac). NO IBI -- their
# pipeline never touches it (see docstring fact #3).
MODALITY_PLAN = (
    ("HR", "hr"), ("EDA", "eda"), ("TEMP", "temp"), ("BVP", "bvp"),
    ("ACC", "acc"), ("O2M", "o2m"),
    ("LGP", "lgp"), ("RGP", "rgp"), ("Telemetry", "tel"),
    ("FL", "fl"), ("PL", "pl"), ("FAU", "fau"),
)


def resample_session(subject: str, session: str) -> pd.DataFrame:
    duration = session_duration_seconds(subject, session)
    grid = np.arange(0.0, duration, 1.0 / GRID_HZ)
    frame = pd.DataFrame({"subject": subject, "session": session, "t": grid})

    for modality, prefix in MODALITY_PLAN:
        try:
            times, values, columns = loaders.load(subject, session, modality)
        except FileNotFoundError:
            times, values, columns = np.empty(0), np.empty((0, 1)), [modality.lower()]
        for j, col in enumerate(columns):
            name = f"{prefix}_{col}"
            if len(times) < 2:
                frame[name] = np.nan
                continue
            v = values[:, j].astype(np.float32)
            good = np.isfinite(v)
            if good.sum() < 2:
                frame[name] = np.nan
                continue
            # linear interpolation, edge-clamped (equivalent to fwd/bwd fill
            # outside the covered span) -- documented deviation from their
            # row-decimation resampling
            frame[name] = np.interp(grid, times[good], v[good]).astype(np.float32)

    frame["label"] = [get_label(subject, session, t / 60.0) for t in grid]
    return frame


def build_4hz_table():
    parts = []
    for subject in SUBJECTS:
        for session in available_sessions(subject):
            t0 = time.time()
            part = resample_session(subject, session)
            parts.append(part)
            print(f"  {subject}/{session}: {len(part)} rows ({time.time() - t0:.1f}s)")
    table = pd.concat(parts, ignore_index=True)
    feature_cols = [c for c in table.columns if c not in ("subject", "session", "t", "label")]
    print(f"4 Hz table: {len(table)} rows x {len(feature_cols)} feature columns")
    return table, feature_cols


def _impute(x: np.ndarray, medians: np.ndarray) -> np.ndarray:
    x = x.copy()
    nan_mask = np.isnan(x)
    x[nan_mask] = np.take(medians, np.where(nan_mask)[1])
    return x


def run_protocol(table: pd.DataFrame, feature_cols: list[str], seed: int, prep: str) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVC

    X_raw = table[feature_cols].to_numpy(dtype=np.float32)
    # source CSVs contain inf / float32-overflow values (the authors' own
    # notebook 1 does this same inf->NaN replacement for O2M) -- missing, not data
    X_raw[~np.isfinite(X_raw)] = np.nan
    y = table["label"].to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    if prep == "paper":
        # THEIR EXACT ORDER (notebook 4): impute+scale+PCA on ALL rows,
        # folds split afterwards. Deliberately leaky -- it reproduces the
        # published number and quantifies that leakage vs --prep clean.
        medians = np.nanmedian(X_raw, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        X_all = MinMaxScaler().fit_transform(_impute(X_raw, medians))
        pca = PCA(n_components=0.90, random_state=seed)
        X_all = pca.fit_transform(X_all)
        print(f"  [paper prep] PCA dims: {pca.n_components_} (fit on all rows)")

    results = {name: [] for name in ANCHORS}
    for fold, (tr, te) in enumerate(skf.split(X_raw, y)):
        if prep == "paper":
            X_tr, X_te = X_all[tr], X_all[te]
        else:
            medians = np.nanmedian(X_raw[tr], axis=0)
            medians = np.where(np.isnan(medians), 0.0, medians)
            scaler = MinMaxScaler().fit(_impute(X_raw[tr], medians))
            pca = PCA(n_components=0.90, random_state=seed)
            X_tr = pca.fit_transform(scaler.transform(_impute(X_raw[tr], medians)))
            X_te = pca.transform(scaler.transform(_impute(X_raw[te], medians)))

        models = {
            "svm": SVC(kernel="rbf", C=0.1, gamma="auto", class_weight="balanced",
                       cache_size=200, random_state=seed),
            "rf": RandomForestClassifier(
                n_estimators=100, max_depth=5, min_samples_leaf=10,
                random_state=seed, n_jobs=-1,
            ),
        }
        for name, clf in models.items():
            clf.fit(X_tr, y[tr])
            pred = clf.predict(X_te)
            acc = float(np.mean(pred == y[te]))
            p, r, f, _ = precision_recall_fscore_support(
                y[te], pred, average="macro", zero_division=0
            )
            results[name].append({"fold": fold, "accuracy": acc,
                                  "precision": float(p), "recall": float(r), "f1": float(f)})
            print(f"  fold {fold} {name}: acc={acc:.4f}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42,
                        help="42 = the authors' own random_state (notebook 4)")
    parser.add_argument("--prep", choices=["paper", "clean"], default="paper",
                        help="paper: scaler/PCA fit on ALL data (their exact, leaky "
                             "protocol -- the anchor); clean: fit on train folds only")
    args = parser.parse_args()

    table, feature_cols = build_4hz_table()
    results = run_protocol(table, feature_cols, args.seed, args.prep)

    print(f"\nprep={args.prep}  seed={args.seed}")
    print(f"{'Model':6s} {'Ours':>8s} {'Paper':>8s} {'Delta':>8s}")
    print("-" * 34)
    ok = True
    for name, folds in results.items():
        mean_acc = float(np.mean([f["accuracy"] for f in folds])) * 100
        delta = mean_acc - ANCHORS[name]
        print(f"{name:6s} {mean_acc:7.2f}% {ANCHORS[name]:7.2f}% {delta:+7.2f}")
        if args.prep == "paper" and abs(delta) > 5.0:
            ok = False
    if args.prep == "paper":
        print("\n" + ("Protocol anchor OK: Tier-1 implementation is validated."
                      if ok else
                      "WARNING: >5 pp from the published anchor under their own "
                      "prep -- check preprocessing against Codes/ before Tier 1."))
    else:
        print("\n[clean prep] Expected to land below the paper numbers -- the gap "
              "vs --prep paper measures the preprocessing leakage in the "
              "published baseline (reportable finding).")

    out_path = (Path(__file__).resolve().parent.parent / "data" / "processed" /
                "results" / f"uldd_baseline_reproduction_{args.prep}_seed{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"prep": args.prep, "seed": args.seed, "anchors": ANCHORS, "results": results}, indent=2
    ))
    print(f"Saved {out_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
