"""CV fold generation for the v2 three-tier evaluation protocol (plan §3),
plus the original LOSO generator (tasks.md Task 1.4).

Tier 1 (primary)  stratified_window_folds -- window-level stratified 5-fold,
                  matching the UL-DD paper's own published protocol (Table 8:
                  SVM 83.75 / RF 75.14 / Transformer 86.39 / I2M2 88.03), so
                  our headline numbers are directly comparable to theirs.
Tier 2            block_folds -- contiguous time blocks per session (default
                  one 4-minute KSS interval per block) dealt to folds, with
                  boundary-straddling windows purged, so overlapping 10s/5s
                  windows never span train/test. Our defensive middle number.
Tier 3            loso_folds -- unseen-driver evaluation. v2 adds
                  include_awake_only_in_train so C/F/L's Low/Medium windows
                  are training data (18-subject train pools, 16 eval folds)
                  instead of being silently discarded (v1 defect #4).

Fold assignments are persisted to disk (scripts/build_v2_folds.py) so every
model consumes identical folds -- the apples-to-apples guarantee.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.paths import NO_DROWSY_SUBJECTS, SUBJECTS

_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
DEFAULT_FOLDS_PATH = _PROCESSED / "folds.json"
LOSO_V2_FOLDS_PATH = _PROCESSED / "folds_v2.json"
TIER1_FOLDS_PATH = _PROCESSED / "tier1_folds.json"
TIER2_FOLDS_PATH = _PROCESSED / "tier2_folds.json"

WINDOW_KEY = ["subject", "session", "window_id"]


def loso_folds(
    subjects: list[str] | None = None,
    *,
    exclude_no_drowsy: bool = True,
    include_awake_only_in_train: bool = False,
) -> list[dict]:
    """Return one fold per subject: [{'fold', 'held_out_subject', 'train_subjects'}].

    `include_awake_only_in_train=True` (Tier 3 v2) adds the awake-only
    subjects (C/F/L) to every fold's training pool -- they are still never
    held out (a fold can't be evaluated on a subject with no Drowsy
    session), but their thousands of Low/Medium windows are real training
    data v1 silently discarded (defect #4)."""
    pool = list(subjects) if subjects is not None else list(SUBJECTS)
    awake_only = sorted(s for s in pool if s in NO_DROWSY_SUBJECTS)
    if exclude_no_drowsy:
        pool = [s for s in pool if s not in NO_DROWSY_SUBJECTS]
    extra = awake_only if include_awake_only_in_train else []
    return [
        {
            "fold": i,
            "held_out_subject": held_out,
            "train_subjects": [s for s in pool if s != held_out] + extra,
        }
        for i, held_out in enumerate(pool)
    ]


def stratified_window_folds(
    df: pd.DataFrame, n_splits: int = 5, seed: int = 0, label_col: str = "label"
) -> pd.DataFrame:
    """Tier 1: window-level stratified k-fold over ALL rows (all 19 subjects,
    C/F/L included -- the paper pooled everyone), stratified by class label,
    exactly the UL-DD paper's protocol ("5-fold cross-validation (k=5) ...
    stratified to maintain the distribution of drowsiness levels").

    Returns a DataFrame with columns [subject, session, window_id, fold]
    assigning every window to exactly one *test* fold. Same-driver windows
    land in both train and test by construction -- that is the published
    protocol, and it is reported as such (plan §3), never mixed with the
    other tiers."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_col = np.full(len(df), -1, dtype=int)
    for i, (_, test_idx) in enumerate(skf.split(np.zeros(len(df)), df[label_col].to_numpy())):
        fold_col[test_idx] = i
    out = df[WINDOW_KEY].copy().reset_index(drop=True)
    out["fold"] = fold_col
    if (out["fold"] < 0).any():
        raise RuntimeError("StratifiedKFold left rows unassigned -- should be impossible")
    return out


def block_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    block_seconds: float = 240.0,
    purge_boundary: bool = True,
) -> pd.DataFrame:
    """Tier 2: contiguous-block k-fold. Each session's timeline is cut into
    `block_seconds` blocks (default 240s = one 4-minute KSS interval, the
    natural unit: windows sharing a block share a label reading and most of
    their signal overlap). Blocks are dealt round-robin to folds, with the
    deal offset rotated per session so fold contents vary across sessions.

    `purge_boundary=True` marks any window whose [t_start, t_end) span
    crosses a block edge as fold=-1 (dropped from train AND test): with a
    10s window / 5s step that's ~1 window in 48, and it is the only way an
    overlapping window could straddle two folds -- purging it makes the
    "no overlap between train and test" guarantee exact, not approximate."""
    out = df[WINDOW_KEY + ["t_start", "t_end"]].copy().reset_index(drop=True)
    sessions = sorted(out[["subject", "session"]].drop_duplicates().itertuples(index=False))
    offset_by_session = {(s.subject, s.session): i % n_splits for i, s in enumerate(sessions)}

    block_idx = (out["t_start"] // block_seconds).astype(int)
    offsets = out.apply(lambda r: offset_by_session[(r["subject"], r["session"])], axis=1)
    fold = (block_idx + offsets) % n_splits

    if purge_boundary:
        # a window straddles iff its end falls in a later block than its start
        # (t_end is exclusive, so a window ending exactly on an edge is clean)
        end_block = ((out["t_end"] - 1e-9) // block_seconds).astype(int)
        fold = fold.where(end_block == block_idx, -1)

    out = out[WINDOW_KEY].copy()
    out["fold"] = fold.astype(int)
    return out


def save_window_folds(assignments: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(assignments.to_dict(orient="records"), indent=None))


def load_window_folds(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(Path(path).read_text()))


def merge_window_folds(
    df: pd.DataFrame, assignments: pd.DataFrame, column: str = "exp_fold"
) -> pd.DataFrame:
    """Attach a persisted window-level fold assignment to a feature table as
    `column`. Rows without an assignment (shouldn't happen for a matching
    table) fail loudly rather than silently becoming train data."""
    a = assignments.rename(columns={"fold": column})
    out = df.merge(a, on=WINDOW_KEY, how="left", validate="one_to_one")
    if out[column].isna().any():
        n = int(out[column].isna().sum())
        raise ValueError(
            f"{n} rows in the feature table have no fold assignment -- "
            f"regenerate fold files (scripts/build_v2_folds.py) against this table"
        )
    out[column] = out[column].astype(int)
    return out


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
