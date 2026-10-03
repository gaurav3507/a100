"""PyTorch Dataset wrapping the processed UL-DD feature table. Shared by
Track A and Track B so both consume identical features/splits/labels --
only the model architecture differs (tasks.md Section 0).

Feature columns are grouped into "nodes" (vision / bio / grip) matching
Track A's GNN, which treats each modality as one node the way FatigueNet
treats ECG/EDA/EMG/blink (tasks.md 3.1) -- except the node set here is our
own three sensor modalities, not the paper's four, and not arbitrary:

  vision -> camera feed (Jetson + IR/RGB camera, UL-DD's FL/PL landmarks)
  bio    -> physiological sensor (Empatica E4 + Checkme O2 Max in UL-DD;
            an ESP32-S3 PPG+GSR breakout board in the deployed hardware,
            tasks.md 5.1)
  grip   -> pressure sensor (left/right FSR grip sensors, both in UL-DD
            and the deployed ESP32-S3 hardware, tasks.md 5.1)

These three are exactly the sensors the actual product has. UL-DD's
Telemetry columns (driving-simulator heading/speed/rpm/gear) are
deliberately *not* one of the default nodes: there is no vehicle-telemetry
sensor anywhere in the ESP32-S3/Jetson hardware spec (Phases 5-6), so a
model trained to depend on it could never run on the real system. The
columns still exist in the feature table and in `TELEMETRY_FEATURES`
below, reachable via `include_telemetry=True`, for a Track-A-only "what if
we also had simulator telemetry" ceiling comparison -- never for Track B,
which is the actual deployment candidate and must only ever see features
producible by the real sensor set.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DEFAULT_FEATURES_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "uldd_features.parquet"
)

META_COLUMNS = ("subject", "session", "window_id", "t_start", "t_end", "kss", "label", "fold")

VISION_FEATURES = [
    "ear_mean", "ear_std", "mar_mean", "mar_std", "perclos",
    "blink_count", "blink_rate_per_min", "yawn_count", "yawn_rate_per_min",
    "head_pitch_mean", "head_pitch_std", "head_yaw_mean", "head_yaw_std",
    "head_roll_mean", "head_roll_std", "head_nod_rate_per_min", "head_tilt_rate_per_min",
]
BIO_FEATURES = [
    "hr_mean", "hr_std", "hrv_rmssd", "hrv_sdnn",
    "eda_tonic_mean", "eda_phasic_mean", "eda_phasic_std", "eda_phasic_max",
    "spo2_mean", "spo2_std", "spo2_min", "pulse_rate_mean", "pulse_rate_std",
    "temp_mean", "temp_slope",
]
GRIP_FEATURES = [
    "grip_left_mean", "grip_left_std", "grip_right_mean", "grip_right_std", "grip_asymmetry",
]
TELEMETRY_FEATURES = ["speed_mean", "speed_std", "heading_change_rate"]

# Our three real deployed sensor modalities -- the default for both tracks.
CORE_FEATURE_GROUPS = {
    "vision": VISION_FEATURES,
    "bio": BIO_FEATURES,
    "grip": GRIP_FEATURES,
}
CORE_FEATURES = [c for group in CORE_FEATURE_GROUPS.values() for c in group]

# Backwards-compatible aliases: default (no telemetry) node set / flat list.
FEATURE_GROUPS = CORE_FEATURE_GROUPS
ALL_FEATURES = CORE_FEATURES


def feature_groups(include_telemetry: bool = False) -> dict[str, list[str]]:
    """Node groups for a dataset instance. `include_telemetry=True` is for
    a Track-A-only ceiling ablation (tasks.md 3.6-style "what's the extra
    headroom from a signal we can't deploy") -- never use it for Track B."""
    groups = dict(CORE_FEATURE_GROUPS)
    if include_telemetry:
        groups["telemetry"] = TELEMETRY_FEATURES
    return groups


def _load_df(source) -> pd.DataFrame:
    return source if isinstance(source, pd.DataFrame) else pd.read_parquet(source)


class ULDDWindowDataset(Dataset):
    """One sample per window: grouped per-modality node tensors + a flat
    concatenation, for single-window classifiers (Track B MLP/LightGBM) or
    as the per-timestep input to a sequence model (Track A Transformer,
    Track B GRU -- see ULDDSequenceDataset)."""

    def __init__(
        self,
        source=DEFAULT_FEATURES_PATH,
        folds: list[int] | None = None,
        impute: str = "median",
        standardize: bool = True,
        include_telemetry: bool = False,
        stats_subjects: list[str] | None = None,
        include_no_drowsy: bool = True,
    ):
        df = _load_df(source)
        if folds is not None:
            keep = list(folds) + ([-1] if include_no_drowsy else [])
            df = df[df["fold"].isin(keep)]
        self.df = df.reset_index(drop=True)
        self.standardize = standardize
        self.feature_groups = feature_groups(include_telemetry)
        self.all_features = [c for group in self.feature_groups.values() for c in group]

        # Normalization statistics come from stats_subjects ONLY (the
        # training subjects of the current fold). Computing them over the
        # full table -- as this class previously did -- lets the held-out
        # subject's distribution shape every z-score, a mild but real
        # leak. When stats_subjects is None (e.g. quick inspection), fall
        # back to all loaded rows, but training code must always pass it.
        stats_df = (
            self.df[self.df["subject"].isin(stats_subjects)]
            if stats_subjects is not None else self.df
        )
        if len(stats_df) == 0:
            raise ValueError("stats_subjects matched no rows -- check subject codes")

        self._fill = {c: 0.0 for c in self.all_features}
        if impute == "median":
            for c in self.all_features:
                med = stats_df[c].median()
                self._fill[c] = 0.0 if pd.isna(med) else float(med)

        # Raw feature scales span ~3-4 orders of magnitude (e.g. hrv_rmssd
        # ~0.05 vs blink_rate_per_min ~124, head_yaw_mean ~-87) -- fatal for
        # the neural models (Track A's GNN+Transformer, Track B's MLP/GRU)
        # fed raw values through nn.Linear: confirmed on mephy_repro's
        # identical architecture, where unstandardized inputs caused a
        # no-GNN ablation to NaN outright and the full model to plateau
        # near chance accuracy (see NOTES.md). LightGBM is scale-invariant
        # and unaffected either way.
        self._mean = {c: float(stats_df[c].mean()) for c in self.all_features}
        self._std = {c: float(stats_df[c].std()) or 1.0 for c in self.all_features}

    def set_stats_subjects(self, subjects: list[str]) -> None:
        """Recompute imputation/normalization statistics from `subjects` only.

        Trainers call this once per LOSO fold with that fold's TRAINING
        subjects, so the held-out subject never contributes to the
        statistics that transform its own features at test time."""
        stats_df = self.df[self.df["subject"].isin(subjects)]
        if len(stats_df) == 0:
            raise ValueError("stats subjects matched no rows -- check subject codes")
        for c in self.all_features:
            med = stats_df[c].median()
            self._fill[c] = 0.0 if pd.isna(med) else float(med)
            self._mean[c] = float(stats_df[c].mean())
            self._std[c] = float(stats_df[c].std()) or 1.0

    def __len__(self) -> int:
        return len(self.df)

    def _row_group(self, row: pd.Series, columns: list[str]) -> np.ndarray:
        vals = [self._fill[c] if pd.isna(row[c]) else float(row[c]) for c in columns]
        if self.standardize:
            vals = [(v - self._mean[c]) / self._std[c] for v, c in zip(vals, columns)]
        return np.asarray(vals, dtype=np.float32)

    def row_features(self, row: pd.Series) -> dict[str, np.ndarray]:
        return {name: self._row_group(row, cols) for name, cols in self.feature_groups.items()}

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        nodes = self.row_features(row)
        flat = np.concatenate(list(nodes.values()))
        return {
            "nodes": {k: torch.from_numpy(v) for k, v in nodes.items()},
            "flat": torch.from_numpy(flat),
            "label": int(row["label"]),
            "subject": row["subject"],
            "session": row["session"],
            "window_id": int(row["window_id"]),
        }


class ULDDSequenceDataset(ULDDWindowDataset):
    """Last `context_len` windows (same subject+session, zero-padded at the
    start of a session) as one sequence sample, for Track A's Transformer
    (tasks.md 3.1: 12-24 windows of context) or Track B's optional GRU
    (tasks.md 4.1 Option 3: 6-12 windows)."""

    def __init__(self, source=DEFAULT_FEATURES_PATH, folds: list[int] | None = None, impute: str = "median",
                 standardize: bool = True, context_len: int = 12, include_telemetry: bool = False):
        super().__init__(source, folds, impute, standardize, include_telemetry)
        self.context_len = context_len
        self.df = self.df.sort_values(["subject", "session", "window_id"]).reset_index(drop=True)
        self._group_start = self.df.groupby(["subject", "session"]).apply(
            lambda g: g.index.min(), include_groups=False
        ).to_dict()

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        key = (row["subject"], row["session"])
        group_start = self._group_start[key]
        lo = max(group_start, idx - self.context_len + 1)

        node_seqs = {name: [] for name in self.feature_groups}
        flat_feats = []
        for i in range(lo, idx + 1):
            r = self.df.iloc[i]
            nodes = self.row_features(r)
            for name, vec in nodes.items():
                node_seqs[name].append(vec)
            flat_feats.append(np.concatenate(list(nodes.values())))

        pad_len = self.context_len - len(flat_feats)
        if pad_len > 0:
            flat_feats = [np.zeros_like(flat_feats[0])] * pad_len + flat_feats
            for name in node_seqs:
                node_seqs[name] = [np.zeros_like(node_seqs[name][0])] * pad_len + node_seqs[name]

        sequence = torch.from_numpy(np.stack(flat_feats))  # (context_len, n_features), for Track B's GRU
        nodes = {
            name: torch.from_numpy(np.stack(seq)) for name, seq in node_seqs.items()
        }  # {modality: (context_len, dim_m)}, for Track A's FatigueNetModel
        return {
            "sequence": sequence,
            "nodes": nodes,
            "label": int(row["label"]),
            "subject": row["subject"],
            "session": row["session"],
            "window_id": int(row["window_id"]),
        }
