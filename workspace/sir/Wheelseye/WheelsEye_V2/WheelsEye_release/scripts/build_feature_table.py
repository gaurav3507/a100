"""Task 2.5 / plan §4 - assemble the final windowed feature table: one row
per (subject, session, window) with the full v2 feature set -- v1's vision +
bio + grip (+ telemetry) columns PLUS the v2 additions: FAU action units,
PRV-spectral cardiac features (from a 120s trailing BVP context), wrist-ACC
movement, posture, O2M motion, and the session-context columns
(time-on-task + drift). This exact table is the single input every model in
every tier consumes.

Usage: python scripts/build_feature_table.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from common.context_features import add_context_features
from common.features_acc import extract_acc_features
from common.features_bio import extract_bio_features
from common.features_cardiac_spectral import BVP_CONTEXT_SECONDS, extract_bvp_spectral_features
from common.features_fau import extract_fau_features
from common.features_grip import extract_grip_features
from common.features_posture import extract_posture_features
from common.features_telemetry import extract_telemetry_features
from common.features_vision import extract_vision_features
from common.labels import available_sessions
from common.splits import DEFAULT_FOLDS_PATH, assign_fold_column, loso_folds, save_folds
from common.paths import NO_DROWSY_SUBJECTS, SUBJECTS
from common.windowing import WINDOW_SECONDS, iter_windows

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "uldd_features.parquet"


def rows_for_subject_session(subject: str, session: str):
    # BVP gets a trailing 120s context slice instead of the bare 10s window:
    # its PRV features need to resolve the 0.04-0.15 Hz LF band, which a 10s
    # slice physically cannot (see features_cardiac_spectral.py docstring).
    for w in iter_windows(subject, session, context_slices={"BVP": BVP_CONTEXT_SECONDS}):
        row = {
            "subject": subject,
            "session": session,
            "window_id": w.window_id,
            "t_start": w.t_start,
            "t_end": w.t_end,
            "kss": w.kss,
            "label": w.label,
        }
        row.update(extract_vision_features(w.signals["FL"], w.signals["PL"], WINDOW_SECONDS))
        row.update(
            extract_bio_features(
                w.signals["HR"], w.signals["IBI"], w.signals["EDA"], w.signals["O2M"], w.signals["TEMP"]
            )
        )
        row.update(extract_grip_features(w.signals["LGP"], w.signals["RGP"]))
        row.update(extract_telemetry_features(w.signals["Telemetry"]))
        # v2 additions (plan §4)
        row.update(extract_fau_features(w.signals["FAU"]))
        row.update(extract_bvp_spectral_features(w.signals["BVP"]))
        row.update(extract_acc_features(w.signals["ACC"]))
        row.update(extract_posture_features(w.signals["PL"]))
        yield row


def main() -> int:
    subjects = [s for s in SUBJECTS]
    all_rows = []
    t0 = time.time()

    for subject in subjects:
        for session in available_sessions(subject):
            n_before = len(all_rows)
            all_rows.extend(rows_for_subject_session(subject, session))
            print(f"  {subject}/{session}: {len(all_rows) - n_before} windows")

    df = pd.DataFrame(all_rows)
    print(f"\nTotal rows: {len(df)} in {time.time() - t0:.1f}s")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

    # v2: session-context columns (minutes_elapsed + per-feature drift from
    # the session's opening 5 minutes) -- computed on the assembled table,
    # before fold assignment. Causal at inference: the current drive's own
    # opening minutes are always observable.
    df = add_context_features(df)

    folds = loso_folds()
    save_folds(folds, DEFAULT_FOLDS_PATH)

    # C/F/L have no Drowsy session (tasks.md 1.1) and are excluded from LOSO
    # folds entirely (tasks.md 1.4). Keep their rows in the table for
    # completeness/inspection but mark them fold=-1 so downstream train/eval
    # code can filter them out explicitly rather than accidentally including
    # an awake-only subject in a fold.
    no_drowsy_mask = df["subject"].isin(NO_DROWSY_SUBJECTS)
    df_foldable = assign_fold_column(df[~no_drowsy_mask], folds)
    df_excluded = df[no_drowsy_mask].copy()
    df_excluded["fold"] = -1
    df = pd.concat([df_foldable, df_excluded], ignore_index=True).sort_values(
        ["subject", "session", "window_id"]
    ).reset_index(drop=True)

    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts):
        print(f"\nColumns with NaN values (expected for sparse IBI / missing Telemetry):")
        print(nan_counts)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")

    excluded = sorted(NO_DROWSY_SUBJECTS)
    print(f"Subjects excluded from folds (no Drowsy session): {excluded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
