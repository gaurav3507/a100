"""Assemble the MePhy windowed feature table for the FatigueNet reproduction
sanity check (tasks.md 1.2/3.1): one row per (user, condition, window) with
ECG/EDA/EMG/blink features and the 4-class condition label (rest=0,
cognitive-fatigue=1, physical-fatigue=2, combo-fatigue=3).

Restricted to the 19 users with all four modalities (user0..user18) --
FatigueNet's GNN needs all four modality nodes per sample.

Usage: .venv/Scripts/python.exe scripts/build_mephy_feature_table.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from mephy_repro.features import extract_mephy_features
from mephy_repro.paths import CONDITIONS, FULL_COVERAGE_USERS
from mephy_repro.windowing import iter_windows

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "mephy_features.parquet"
FOLDS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "mephy_folds.json"


def rows_for(user: str, condition: str):
    for w in iter_windows(user, condition):
        row = {"user": user, "condition": condition, "window_id": w.window_id,
               "t_start": w.t_start, "t_end": w.t_end, "label": w.label}
        row.update(extract_mephy_features(
            w.signals["ECG"], w.signals["EDA"], w.signals["EMG"], w.signals["EyeBlinking"]
        ))
        yield row


def loso_folds(users):
    return [
        {"fold": i, "held_out_user": u, "train_users": [x for x in users if x != u]}
        for i, u in enumerate(users)
    ]


def main() -> int:
    users = list(FULL_COVERAGE_USERS)
    all_rows = []
    t0 = time.time()
    for user in users:
        for condition in CONDITIONS:
            n_before = len(all_rows)
            all_rows.extend(rows_for(user, condition))
            print(f"  {user}/{condition}: {len(all_rows) - n_before} windows")

    df = pd.DataFrame(all_rows)
    print(f"\nTotal rows: {len(df)} in {time.time() - t0:.1f}s")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

    folds = loso_folds(users)
    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FOLDS_PATH.write_text(json.dumps(folds, indent=2))

    fold_by_user = {f["held_out_user"]: f["fold"] for f in folds}
    df["fold"] = df["user"].map(fold_by_user)

    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if len(nan_counts):
        print(f"\nColumns with NaN values:\n{nan_counts}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.2f} MB), {len(users)} folds (LOSO)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
