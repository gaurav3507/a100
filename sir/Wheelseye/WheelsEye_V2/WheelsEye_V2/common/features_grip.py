"""Grip pressure features (tasks.md Task 2.3). Asymmetry between left/right
grip is called out in the UL-DD paper as a drowsiness-relevant behavioral
cue -- a driver losing engagement tends to grip unevenly or one-handed.
"""
from __future__ import annotations

import numpy as np

from common.windowing import Signal


def extract_grip_features(lgp: Signal, rgp: Signal) -> dict[str, float]:
    if len(lgp.times) == 0 or len(rgp.times) == 0:
        return {
            "grip_left_mean": float("nan"), "grip_left_std": float("nan"),
            "grip_right_mean": float("nan"), "grip_right_std": float("nan"),
            "grip_asymmetry": float("nan"),
        }
    left = lgp.values[:, 0]
    right = rgp.values[:, 0]
    return {
        "grip_left_mean": float(np.mean(left)),
        "grip_left_std": float(np.std(left)),
        "grip_right_mean": float(np.mean(right)),
        "grip_right_std": float(np.std(right)),
        "grip_asymmetry": float(abs(np.mean(left) - np.mean(right))),
    }
