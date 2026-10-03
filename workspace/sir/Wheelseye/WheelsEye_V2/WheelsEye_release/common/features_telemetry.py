"""Driving telemetry features (tasks.md Task 2.4, optional auxiliary signal).
Only available for a subset of subject/sessions -- callers must treat a
Signal with zero samples as "not available for this window", not an error.
"""
from __future__ import annotations

import numpy as np

from common.windowing import Signal

# Telemetry.csv columns (after dropping timestamp[us]):
# raw rendering ts, raw simulation ts, raw paused simulation ts,
# heading[deg], pitch[deg], roll[deg], speed[m/s], rpm, gear
_NAN_FEATURES = {
    "speed_mean": float("nan"), "speed_std": float("nan"),
    "heading_change_rate": float("nan"),
}


def extract_telemetry_features(telemetry: Signal) -> dict[str, float]:
    if len(telemetry.times) < 2:
        return dict(_NAN_FEATURES)

    heading = telemetry.values[:, telemetry.columns.index("heading[deg]")]
    speed = telemetry.values[:, telemetry.columns.index("speed[m/s]")]

    # headings wrap at 360deg; wrap diffs into (-180, 180] before treating
    # them as a continuous "rate of turn" signal
    heading_diff = np.diff(heading)
    heading_diff = (heading_diff + 180.0) % 360.0 - 180.0
    duration_s = telemetry.times[-1] - telemetry.times[0]

    return {
        "speed_mean": float(np.mean(speed)),
        "speed_std": float(np.std(speed)),
        "heading_change_rate": float(np.sum(np.abs(heading_diff)) / duration_s) if duration_s > 0 else float("nan"),
    }
