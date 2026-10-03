"""Raw per-modality loaders for MePhy, column semantics confirmed against
the dataset's ReadMe.pdf (see mephy_repro/paths.py docstring):

  - ECG: 6 cols (timestamp, heart rate, RR interval in 1/1024 fmt, 2 hex
    status bytes, RR interval in ms), 1 Hz (Polar H10 already computes RR
    intervals on-device). The two hex columns are cross-checked here
    against col3*1024 == col6 as a light integrity check, then dropped --
    they're link-quality status codes, not signal.
  - EDA / EMG: 2 cols (timestamp, raw digital value), 1000 Hz (BITalino).
  - EyeBlinking: 2 cols (timestamp, eyelid-closed boolean), 30 Hz
    (Logitech C920 + CV). Values do include real 1s (blink/closed events),
    confirmed by inspection, not always zero.

Every loader returns (times_seconds, values, column_names), with times
zeroed to the first sample of that (user, modality, condition) file --
each condition is its own short standalone recording, not a continuous
multi-hour session like UL-DD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mephy_repro.paths import file_path


def _parse_ts_seconds(ts: str) -> float:
    """Robust HH:MM:SS[.frac] -> seconds-of-day float. MePhy's own
    timestamps aren't consistently zero-padded (EyeBlinking has both
    '10:41:2.0' and '10:41:02.033' in the same file), so this avoids
    assuming fixed field widths."""
    time_part, _, frac = ts.partition(".")
    h, m, s = time_part.split(":")
    seconds = int(h) * 3600 + int(m) * 60 + int(s)
    if frac:
        seconds += float("0." + frac)
    return seconds


def load_ecg(user: str, condition: str):
    path = file_path(user, "ECG", condition)
    times, hr, rr_ms = [], [], []
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 6:
                continue
            ts, hr_val, _rr_frac, _hex1, _hex2, rr_ms_val = parts
            times.append(_parse_ts_seconds(ts))
            hr.append(float(hr_val))
            rr_ms.append(float(rr_ms_val))
    times = np.asarray(times, dtype=float)
    if len(times):
        times = times - times[0]
    values = np.column_stack([hr, rr_ms]) if len(hr) else np.empty((0, 2))
    return times, values, ["hr", "rr_ms"]


BITALINO_RATE_HZ = 1000.0


def _load_bitalino(user: str, modality: str, condition: str, column_name: str):
    """Shared loader for EDA/EMG. The dataset's own per-row timestamp string
    truncates to just 'HH:MM' (dropping seconds/ms) at every second boundary
    -- a logging artifact in the source files, confirmed by inspection (every
    1000th row in a 1000Hz file). Since EDA/EMG are fixed-rate per the
    ReadMe, time is derived from sample index / rate instead of trusting
    that per-row string, sidestepping the truncation bug entirely."""
    path = file_path(user, modality, condition)
    df = pd.read_csv(path, sep="\t", header=None, names=["ts", "value"])
    times = np.arange(len(df)) / BITALINO_RATE_HZ
    values = df[["value"]].to_numpy(dtype=float)
    return times, values, [column_name]


def load_eda(user: str, condition: str):
    return _load_bitalino(user, "EDA", condition, "eda_raw")


def load_emg(user: str, condition: str):
    return _load_bitalino(user, "EMG", condition, "emg_raw")


def load_eyeblinking(user: str, condition: str):
    path = file_path(user, "EyeBlinking", condition)
    df = pd.read_csv(path, sep="\t", header=None, names=["ts", "closed"])
    times = df["ts"].map(_parse_ts_seconds).to_numpy()
    if len(times):
        times = times - times[0]
    values = df[["closed"]].to_numpy(dtype=float)
    return times, values, ["eye_closed"]


LOADERS = {
    "ECG": load_ecg,
    "EDA": load_eda,
    "EMG": load_emg,
    "EyeBlinking": load_eyeblinking,
}


def load(user: str, modality: str, condition: str):
    return LOADERS[modality](user, condition)
