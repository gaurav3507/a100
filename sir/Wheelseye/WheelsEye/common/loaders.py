"""Raw per-modality loaders for UL-DD.

Every loader returns a `(times_seconds, values, [column_names])` triple so
`common/windowing.py` can slice any modality uniformly by a time range,
regardless of whether the underlying file is fixed-rate (ACC/BVP/EDA/...),
event-based (IBI: one row per detected heartbeat), or frame-indexed
(FL/PL/FAU: keyed by a video frame number that can skip frames when face
detection fails, so time must be derived from the frame number, not row
position).

Sample rates below were cross-checked against Info.xlsx and confirmed by
dividing each file's row count by the 2400s (40 min) nominal session length.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.paths import csv_path, feature_path

FIXED_RATES_HZ = {
    "ACC": 32.0,
    "BVP": 64.0,
    "EDA": 4.0,
    "HR": 1.0,
    "LGP": 3.0,
    "RGP": 3.0,
    "O2M": 0.5,
    "TEMP": 4.0,
}
O2M_COLUMNS = ("spo2", "pulse_rate", "motion")
ACC_COLUMNS = ("acc_x", "acc_y", "acc_z")
VIDEO_FPS = 60.0


def _fixed_rate_series(subject: str, session: str, modality: str):
    """A few UL-DD O2M.csv files use the literal sentinel '--' for a dropped
    sensor reading (e.g. F/Awake row 60: '--,--,0' -- SpO2/pulse missing,
    motion still recorded). Parsed as NaN; feature functions must use
    NaN-aware reductions (nanmean etc.), not plain mean/std."""
    path = csv_path(subject, session, modality)
    values = pd.read_csv(path, header=None, na_values=["--"]).to_numpy(dtype=float)
    rate = FIXED_RATES_HZ[modality]
    times = np.arange(len(values)) / rate
    return times, values


def load_acc(subject: str, session: str):
    times, values = _fixed_rate_series(subject, session, "ACC")
    return times, values, list(ACC_COLUMNS)


def load_bvp(subject: str, session: str):
    times, values = _fixed_rate_series(subject, session, "BVP")
    return times, values, ["bvp"]


def load_eda(subject: str, session: str):
    times, values = _fixed_rate_series(subject, session, "EDA")
    return times, values, ["eda"]


def load_hr(subject: str, session: str):
    times, values = _fixed_rate_series(subject, session, "HR")
    return times, values, ["hr"]


def load_lgp(subject: str, session: str):
    times, values = _fixed_rate_series(subject, session, "LGP")
    return times, values, ["grip_left"]


def load_rgp(subject: str, session: str):
    times, values = _fixed_rate_series(subject, session, "RGP")
    return times, values, ["grip_right"]


def load_o2m(subject: str, session: str):
    times, values = _fixed_rate_series(subject, session, "O2M")
    return times, values, list(O2M_COLUMNS)


def load_temp(subject: str, session: str):
    times, values = _fixed_rate_series(subject, session, "TEMP")
    return times, values, ["temp"]


def load_ibi(subject: str, session: str):
    """IBI.csv columns: (t_offset_seconds, ibi_seconds), one row per detected
    beat. A handful of UL-DD sessions ship a genuinely 0-byte IBI file (see
    data/raw/ULDD_VERIFICATION_NOTES.md, subject J/Awake) -- treated the same
    as "no beats detected in this session" rather than an error."""
    path = csv_path(subject, session, "IBI")
    if path.stat().st_size == 0:
        return np.empty(0), np.empty((0, 1)), ["ibi"]
    df = pd.read_csv(path, header=None)
    if df.empty:
        return np.empty(0), np.empty((0, 1)), ["ibi"]
    times = df[0].to_numpy(dtype=float)
    values = df[[1]].to_numpy(dtype=float)
    return times, values, ["ibi"]


def load_telemetry(subject: str, session: str):
    path = csv_path(subject, session, "Telemetry")
    df = pd.read_csv(path)
    t0 = df["timestamp[us]"].iloc[0]
    times = (df["timestamp[us]"] - t0).to_numpy(dtype=float) / 1e6
    data_cols = [c for c in df.columns if c != "timestamp[us]"]
    values = df[data_cols].to_numpy(dtype=float)
    return times, values, data_cols


def _video_feature_series(subject: str, session: str, modality: str, frame_col: str):
    path = feature_path(subject, session, modality)
    df = pd.read_csv(path)
    times = df[frame_col].to_numpy(dtype=float) / VIDEO_FPS
    data_cols = [c for c in df.columns if c != frame_col]
    values = df[data_cols].to_numpy(dtype=float)
    return times, values, data_cols


def load_fl(subject: str, session: str):
    """68-pt Dlib facial landmarks, columns X1,Y1..X68,Y68 (pixel coords)."""
    return _video_feature_series(subject, session, "FL", "Frame_Number")


def load_pl(subject: str, session: str):
    """33-pt MediaPipe Pose landmarks, columns {0..32}_{x,y,z} (normalized coords)."""
    return _video_feature_series(subject, session, "PL", "frame")


def load_fau(subject: str, session: str):
    """30 facial action unit intensities."""
    return _video_feature_series(subject, session, "FAU", "Frame")


LOADERS = {
    "ACC": load_acc,
    "BVP": load_bvp,
    "EDA": load_eda,
    "HR": load_hr,
    "LGP": load_lgp,
    "RGP": load_rgp,
    "O2M": load_o2m,
    "TEMP": load_temp,
    "IBI": load_ibi,
    "Telemetry": load_telemetry,
    "FL": load_fl,
    "PL": load_pl,
    "FAU": load_fau,
}


def load(subject: str, session: str, modality: str):
    """Dispatch to the right loader. Returns (times_seconds, values, column_names)."""
    return LOADERS[modality](subject, session)
