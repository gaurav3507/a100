"""Subject-independent CV fold generation, shared by both tracks (tasks.md
Task 1.4).

Leave-one-subject-out (LOSO): each fold holds out exactly one subject for
validation and trains on every other subject. No random splitting is used
anywhere in this pipeline (tasks.md Section 0) because random splits leak
within-subject correlation across train/test.

Subjects with no Drowsy session (C, F, L) are excluded by default since a
fold cannot be evaluated on a class the subject never exhibits, and their
presence in a training fold with only two of three classes would also be
inconsistent with every other fold.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from common.paths import NO_DROWSY_SUBJECTS, SUBJECTS

DEFAULT_FOLDS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "folds.json"


def loso_folds(
    subjects: list[str] | None = None, *, exclude_no_drowsy: bool = True
) -> list[dict]:
    """Return one fold per subject: [{'fold', 'held_out_subject', 'train_subjects'}]."""
    pool = list(subjects) if subjects is not None else list(SUBJECTS)
    if exclude_no_drowsy:
        pool = [s for s in pool if s not in NO_DROWSY_SUBJECTS]
    return [
        {
            "fold": i,
            "held_out_subject": held_out,
            "train_subjects": [s for s in pool if s != held_out],
        }
        for i, held_out in enumerate(pool)
    ]


def assign_fold_column(
    df: pd.DataFrame, folds: list[dict], subject_col: str = "subject"
) -> pd.DataFrame:
    """Add a 'fold' column: index of the fold in which this row's subject is held out."""
    subject_to_fold = {f["held_out_subject"]: f["fold"] for f in folds}
    out = df.copy()
    out["fold"] = out[subject_col].map(subject_to_fold)
    missing = out["fold"].isna()
    if missing.any():
        bad_subjects = sorted(out.loc[missing, subject_col].unique())
        raise ValueError(f"Subjects not covered by any fold: {bad_subjects}")
    out["fold"] = out["fold"].astype(int)
    return out


def save_folds(folds: list[dict], path: Path = DEFAULT_FOLDS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(folds, indent=2))


def load_folds(path: Path = DEFAULT_FOLDS_PATH) -> list[dict]:
    return json.loads(Path(path).read_text())
