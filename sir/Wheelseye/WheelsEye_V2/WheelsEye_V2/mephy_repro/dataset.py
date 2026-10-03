"""PyTorch Dataset wrapping the processed MePhy feature table, producing
per-modality raw feature sequences shaped for common/fatiguenet_model.py's
FatigueNetModel (which expects {modality: (batch, T, raw_dim)}).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DEFAULT_FEATURES_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "mephy_features.parquet"
)

ECG_FEATURES = ["hr_mean", "hr_std", "hrv_rmssd_ms", "hrv_sdnn_ms"]
EDA_FEATURES = ["eda_tonic_mean", "eda_phasic_mean", "eda_phasic_std", "eda_phasic_max"]
EMG_FEATURES = ["emg_rms", "emg_mav", "emg_zero_crossing_rate", "emg_waveform_length"]
BLINK_FEATURES = ["perclos", "blink_count", "blink_rate_per_min"]

MODALITY_FEATURES = {
    "ECG": ECG_FEATURES,
    "EDA": EDA_FEATURES,
    "EMG": EMG_FEATURES,
    "EyeBlinking": BLINK_FEATURES,
}
MODALITY_DIMS = {m: len(cols) for m, cols in MODALITY_FEATURES.items()}


class MePhySequenceDataset(Dataset):
    """One sample = the last `context_len` windows (same user+condition,
    zero-padded at the start of a recording) ending at a given window,
    grouped per modality -- matches FatigueNetModel's expected input."""

    def __init__(
        self,
        source=DEFAULT_FEATURES_PATH,
        folds: list[int] | None = None,
        context_len: int = 12,
        impute: str = "median",
        standardize: bool = True,
    ):
        df = source if isinstance(source, pd.DataFrame) else pd.read_parquet(source)
        if folds is not None:
            df = df[df["fold"].isin(folds)]
        self.df = df.sort_values(["user", "condition", "window_id"]).reset_index(drop=True)
        self.context_len = context_len
        self.standardize = standardize

        all_cols = [c for cols in MODALITY_FEATURES.values() for c in cols]
        self._fill = {c: 0.0 for c in all_cols}
        if impute == "median":
            for c in all_cols:
                med = self.df[c].median()
                self._fill[c] = 0.0 if pd.isna(med) else float(med)

        # Raw feature scales differ by up to 6 orders of magnitude within
        # this table (e.g. emg_waveform_length ~1e4-1e5 vs perclos ~1e-2),
        # which is fatal for a neural net fed raw values through nn.Linear:
        # confirmed by direct diagnosis (a no-GNN ablation NaN'd out
        # immediately on unstandardized inputs; the GNN's forced L2-norm
        # only masked the problem for the full model, at the cost of
        # destroying signal -- accuracy plateaued near chance). Z-score
        # each feature using this dataset's own mean/std before it reaches
        # the model. LightGBM (track_b_light) is scale-invariant and
        # doesn't need this -- it's specific to the neural architectures.
        self._mean = {c: float(self.df[c].mean()) for c in all_cols}
        self._std = {c: float(self.df[c].std()) or 1.0 for c in all_cols}

        self._group_start = self.df.groupby(["user", "condition"]).apply(
            lambda g: g.index.min(), include_groups=False
        ).to_dict()

    def __len__(self) -> int:
        return len(self.df)

    def _row_group(self, row: pd.Series, columns: list[str]) -> np.ndarray:
        vals = [self._fill[c] if pd.isna(row[c]) else float(row[c]) for c in columns]
        if self.standardize:
            vals = [(v - self._mean[c]) / self._std[c] for v, c in zip(vals, columns)]
        return np.asarray(vals, dtype=np.float32)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        key = (row["user"], row["condition"])
        group_start = self._group_start[key]
        lo = max(group_start, idx - self.context_len + 1)

        per_modality_seqs = {m: [] for m in MODALITY_FEATURES}
        for i in range(lo, idx + 1):
            r = self.df.iloc[i]
            for m, cols in MODALITY_FEATURES.items():
                per_modality_seqs[m].append(self._row_group(r, cols))

        pad_len = self.context_len - (idx - lo + 1)
        result = {}
        for m, seq in per_modality_seqs.items():
            if pad_len > 0:
                seq = [np.zeros_like(seq[0])] * pad_len + seq
            result[m] = torch.from_numpy(np.stack(seq))  # (context_len, dim_m)

        return {
            "nodes": result,
            "label": int(row["label"]),
            "user": row["user"],
            "condition": row["condition"],
            "window_id": int(row["window_id"]),
        }


def collate(batch: list[dict]) -> dict:
    nodes = {m: torch.stack([b["nodes"][m] for b in batch]) for m in MODALITY_FEATURES}
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"nodes": nodes, "label": labels}
