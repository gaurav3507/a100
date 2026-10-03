"""Shared utilities for the Tier-1 experiment scripts (WheelsEye_V2).

Self-contained: derives the LOSO plan from the feature table itself (subjects
with both sessions are evaluated; awake-only subjects join every training
pool), implements every calibration variant locally, and writes results in the
same JSON schema as scripts/run_experiment.py so paired_tests.py and
collect_v2.py consume them unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

META = {"subject", "session", "window_id", "label", "fold", "t3_fold", "window_start", "window_end", "kss", "block"}
RESULTS = Path("data/processed/results")


# ---------------------------------------------------------------- features
def resolve_groups(df: pd.DataFrame, groups_json: str | None = None) -> tuple[dict[str, list[str]], list[str]]:
    """Return (modality groups, context features). Prefers the repo's V2 definitions."""
    if groups_json:
        g = json.loads(Path(groups_json).read_text())
        ctx = g.pop("context", [])
        return {k: [c for c in v if c in df.columns] for k, v in g.items()}, [c for c in ctx if c in df.columns]
    try:
        import common.dataset as ds  # type: ignore
        groups = getattr(ds, "V2_FEATURE_GROUPS", None) or getattr(ds, "CORE_FEATURE_GROUPS")
        groups = {k: [c for c in v if c in df.columns] for k, v in dict(groups).items()}
        ctx = []
        try:
            from common.context_features import CONTEXT_FEATURES  # type: ignore
            ctx = [c for c in CONTEXT_FEATURES if c in df.columns]
        except Exception:
            ctx = [c for c in df.columns if c not in META and c.startswith(("ctx_", "drift_"))]
        if sum(len(v) for v in groups.values()):
            return groups, ctx
    except Exception:
        pass
    # fallback: prefix heuristics, printed so the user can verify
    cols = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    groups = {"vision": [c for c in cols if c.startswith(("ear", "mar", "perclos", "blink", "yawn", "head", "fau", "au", "pose", "post"))],
              "bio": [c for c in cols if c.startswith(("hr", "hrv", "ibi", "eda", "scl", "scr", "spo2", "pr_", "temp", "acc", "bvp", "lf", "hf", "rmssd", "sdnn"))],
              "grip": [c for c in cols if c.startswith(("grip", "lgp", "rgp"))]}
    ctx = [c for c in cols if c.startswith(("ctx_", "drift_"))]
    print("WARNING: feature groups inferred from column prefixes -- verify:", {k: len(v) for k, v in groups.items()})
    return groups, ctx


# ---------------------------------------------------------------- folds
def loso_plan(df: pd.DataFrame) -> list[dict]:
    """Evaluate every subject having both sessions; all others train in every fold."""
    have_both = sorted(s for s, g in df.groupby("subject") if {"A", "D"} <= set(g["session"]))
    return [{"fold": i, "held_out": s, "label": f"held_out={s}"} for i, s in enumerate(have_both)]


def window_folds(df: pd.DataFrame, n_splits: int, seed: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold
    fold = np.empty(len(df), dtype=int)
    for k, (_, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(df, df["label"])):
        fold[te] = k
    return fold


def block_folds(df: pd.DataFrame, n_splits: int, block_windows: int = 48, purge: int = 2) -> np.ndarray:
    """Contiguous 4-min blocks per session round-robin to folds; boundary windows purged (-1)."""
    fold = np.full(len(df), -1, dtype=int)
    for _, g in df.groupby(["subject", "session"]):
        g = g.sort_values("window_id")
        pos = np.arange(len(g)); blk = pos // block_windows
        f = blk % n_splits
        edge = (pos % block_windows < purge) | (pos % block_windows >= block_windows - purge)
        f = np.where(edge & (blk > 0) | edge & (pos % block_windows >= block_windows - purge), -1, f)
        fold[df.index.get_indexer(g.index)] = f
    return fold


# ---------------------------------------------------------------- calibration
def _stats(g: pd.DataFrame, feats: list[str]):
    mean = g[feats].mean().fillna(0.0)
    std = g[feats].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    return mean, std


def calibrate(df: pd.DataFrame, feats: list[str], mode: str, k: int = 96,
              cal_feats: list[str] | None = None) -> pd.DataFrame:
    """Per-driver calibration applied to `cal_feats` (default: all `feats`). Pass the SENSOR features
    only to match V2's convention (context/drift columns are already relative and are left untouched).
    mode: none | enroll | enroll_mean | whole_awake | whole_awake_mean | both_sessions | both_sessions_mean
    """
    cal_feats = list(feats) if cal_feats is None else [c for c in cal_feats if c in feats]
    out = df.copy(); out[feats] = out[feats].astype("float64")
    if mode == "none":
        return out
    mean_only = mode.endswith("_mean")
    base = mode[:-5] if mean_only else mode
    for subj, g in df.groupby("subject"):
        if base in ("enroll", "whole_awake"):
            src = g[g["session"] == "A"].sort_values("window_id")
            if base == "enroll": src = src.head(k)
        elif base == "both_sessions":
            src = g
        else:
            raise ValueError(f"unknown calibration mode {mode!r}")
        if len(src) < 2:
            raise ValueError(f"subject {subj}: not enough windows for calibration mode {mode}")
        mean, std = _stats(src, cal_feats)
        if mean_only: std = pd.Series(1.0, index=cal_feats)
        m = (out["subject"] == subj).to_numpy()
        out.loc[m, cal_feats] = ((out.loc[m, cal_feats] - mean) / std).to_numpy()
    return out


def coral_align(X_train: np.ndarray, X_test: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Test-time CORAL: align the held-out subject's feature covariance (and mean) to the training pool.
    Label-free but transductive (uses all test windows). Returns aligned X_test."""
    def sqrtm_psd(C, inv=False):
        w, V = np.linalg.eigh(C); w = np.clip(w, 1e-8, None)
        return (V * (w ** (-0.5 if inv else 0.5))) @ V.T
    mu_s, mu_t = X_train.mean(0), X_test.mean(0)
    Cs = np.cov(X_train - mu_s, rowvar=False) + eps * np.eye(X_train.shape[1])
    Ct = np.cov(X_test - mu_t, rowvar=False) + eps * np.eye(X_test.shape[1])
    return ((X_test - mu_t) @ sqrtm_psd(Ct, inv=True) @ sqrtm_psd(Cs)) + mu_s


# ---------------------------------------------------------------- model + metrics
def fit_predict(X_tr, y_tr, X_te, seed: int = 0, n_estimators: int = 200):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=n_estimators, verbosity=-1, random_state=seed)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def metrics(y_true, y_pred, n_classes: int = 3, names=("Low", "Medium", "High")) -> dict:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    names = list(names)[:n_classes] if len(names) >= n_classes else [f"c{i}" for i in labels]
    return {"accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
            "precision_macro": float(p.mean()), "recall_macro": float(r.mean()), "f1_macro": float(f.mean()),
            "confusion_matrix": cm.tolist(),
            "per_class": {n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])} for i, n in enumerate(names)}}


def clean(X: pd.DataFrame) -> np.ndarray:
    return np.nan_to_num(X.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def save(tag: str, config: dict, per_fold: list[dict]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    acc = np.array([f["metrics"]["accuracy"] for f in per_fold]); f1 = np.array([f["metrics"]["f1_macro"] for f in per_fold])
    out = {"config": config, "per_fold": per_fold,
           "aggregate": {"accuracy": {"mean": float(acc.mean()), "std": float(acc.std())},
                         "f1_macro": {"mean": float(f1.mean()), "std": float(f1.std())}, "n_runs": len(per_fold)}}
    p = RESULTS / f"{tag}.json"; p.write_text(json.dumps(out, indent=2))
    print(f"  {tag:<48} acc={acc.mean()*100:6.2f}+/-{acc.std()*100:4.1f}  f1={f1.mean()*100:6.2f}")
    return p
