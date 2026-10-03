"""Posture features from the full 33-point MediaPipe Pose landmarks (plan §4
"Posture": v1 used 5 of 33 PL landmarks, and only for head pose; the UL-DD
paper explicitly names slouching / head tilting / leaning to one side as
fatigue cues in its Behavioral Data section).

Coordinates are MediaPipe-normalized (x, y in [0, 1] of the frame, y grows
DOWNWARD). All features are therefore in normalized-image units; relative
changes are what carry signal, and downstream standardization (and the
Tier-3 alert-baseline calibration) handles the per-camera offset.

Landmark indices reuse common/features_vision.py's constants (nose 0, eye
outer corners 3/6, shoulders 11/12).
"""
from __future__ import annotations

import numpy as np

from common.features_vision import (
    PL_LEFT_EYE_OUTER, PL_LEFT_SHOULDER, PL_NOSE, PL_RIGHT_EYE_OUTER, PL_RIGHT_SHOULDER,
)
from common.windowing import Signal

POSTURE_FEATURES = (
    "posture_shoulder_angle_mean", "posture_shoulder_angle_std",
    "posture_head_drop_mean", "posture_head_drop_std",
    "posture_lean_mean", "posture_lean_std",
    "posture_motion_mean",
)

_NAN = {k: float("nan") for k in POSTURE_FEATURES}


def _xy(values: np.ndarray, columns: list[str], idx: int):
    return values[:, columns.index(f"{idx}_x")], values[:, columns.index(f"{idx}_y")]


def extract_posture_features(pl: Signal) -> dict[str, float]:
    """One dict of scalar posture features for a single window's PL slice.

    - shoulder_angle: tilt of the shoulder line vs horizontal, degrees --
      a drooping/uneven shoulder line reads as postural collapse.
    - head_drop: vertical eye-line-to-shoulder-line distance (normalized
      units). Shrinks as the head sinks toward the shoulders (slouch).
    - lean: nose x-offset from the shoulder midpoint -- leaning to one side.
    - motion: mean frame-to-frame displacement of nose + shoulder midpoint,
      a general body-restlessness measure (drops as the driver disengages).
    """
    if len(pl.times) < 2:
        return dict(_NAN)

    lx, ly = _xy(pl.values, pl.columns, PL_LEFT_SHOULDER)
    rx, ry = _xy(pl.values, pl.columns, PL_RIGHT_SHOULDER)
    elx, ely = _xy(pl.values, pl.columns, PL_LEFT_EYE_OUTER)
    erx, ery = _xy(pl.values, pl.columns, PL_RIGHT_EYE_OUTER)
    nx, ny = _xy(pl.values, pl.columns, PL_NOSE)

    shoulder_angle = np.degrees(np.arctan2(ry - ly, rx - lx + 1e-9))
    sh_mid_x, sh_mid_y = (lx + rx) / 2.0, (ly + ry) / 2.0
    eye_mid_y = (ely + ery) / 2.0
    head_drop = np.abs(sh_mid_y - eye_mid_y)  # y grows downward; abs = vertical gap
    lean = nx - sh_mid_x

    nose_step = np.sqrt(np.diff(nx) ** 2 + np.diff(ny) ** 2)
    sh_step = np.sqrt(np.diff(sh_mid_x) ** 2 + np.diff(sh_mid_y) ** 2)
    motion = (nose_step + sh_step) / 2.0

    return {
        "posture_shoulder_angle_mean": float(np.mean(shoulder_angle)),
        "posture_shoulder_angle_std": float(np.std(shoulder_angle)),
        "posture_head_drop_mean": float(np.mean(head_drop)),
        "posture_head_drop_std": float(np.std(head_drop)),
        "posture_lean_mean": float(np.mean(lean)),
        "posture_lean_std": float(np.std(lean)),
        "posture_motion_mean": float(np.mean(motion)),
    }
