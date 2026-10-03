"""Vision features from UL-DD's pre-extracted FL (68-pt Dlib) and PL (33-pt
MediaPipe Pose) landmark CSVs (tasks.md Task 2.1). No video decoding needed.

Landmark indices follow the standard iBUG 300-W 68-point scheme for FL
(1-indexed, matching the FL CSV's X1..X68/Y1..Y68 columns) and the standard
MediaPipe Pose 33-point scheme for PL (0-indexed, matching the PL CSV's
{i}_x/{i}_y/{i}_z columns).

Head pose here is a lightweight geometric proxy (eye-line tilt for roll,
ear/eye depth asymmetry for yaw, nose-vs-shoulder offset for pitch), not a
camera-calibrated solvePnP estimate -- UL-DD doesn't ship camera intrinsics.
It's monotonic in actual head rotation, which is sufficient for the
nodding/tilt *frequency* signal tasks.md 2.1 asks for.
"""
from __future__ import annotations

import numpy as np

from common.windowing import Signal

# --- FL (68-pt Dlib), 1-indexed per the CSV's X{n}/Y{n} columns ---
FL_RIGHT_EYE = (37, 38, 39, 40, 41, 42)
FL_LEFT_EYE = (43, 44, 45, 46, 47, 48)
FL_MOUTH_LEFT_CORNER = 49
FL_MOUTH_RIGHT_CORNER = 55
FL_MOUTH_TOP_INNER = (51, 53)
FL_MOUTH_BOTTOM_INNER = (59, 57)

EAR_THRESHOLD = 0.21  # standard Soukupova & Cech closed-eye threshold
MAR_THRESHOLD = 0.5  # yawn threshold, typical for this MAR formula

# --- PL (33-pt MediaPipe Pose), 0-indexed per the CSV's {i}_x/{i}_y/{i}_z ---
PL_NOSE = 0
PL_LEFT_EYE_OUTER = 3
PL_RIGHT_EYE_OUTER = 6
PL_LEFT_SHOULDER = 11
PL_RIGHT_SHOULDER = 12


def _fl_point(values: np.ndarray, columns: list[str], idx: int):
    xi = columns.index(f"X{idx}")
    yi = columns.index(f"Y{idx}")
    return values[:, xi], values[:, yi]


def _dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def eye_aspect_ratio(values: np.ndarray, columns: list[str], eye_points: tuple[int, ...]):
    p1x, p1y = _fl_point(values, columns, eye_points[0])
    p2x, p2y = _fl_point(values, columns, eye_points[1])
    p3x, p3y = _fl_point(values, columns, eye_points[2])
    p4x, p4y = _fl_point(values, columns, eye_points[3])
    p5x, p5y = _fl_point(values, columns, eye_points[4])
    p6x, p6y = _fl_point(values, columns, eye_points[5])
    vertical = _dist(p2x, p2y, p6x, p6y) + _dist(p3x, p3y, p5x, p5y)
    horizontal = 2.0 * _dist(p1x, p1y, p4x, p4y) + 1e-9
    return vertical / horizontal


def mouth_aspect_ratio(values: np.ndarray, columns: list[str]):
    lx, ly = _fl_point(values, columns, FL_MOUTH_LEFT_CORNER)
    rx, ry = _fl_point(values, columns, FL_MOUTH_RIGHT_CORNER)
    t1x, t1y = _fl_point(values, columns, FL_MOUTH_TOP_INNER[0])
    b1x, b1y = _fl_point(values, columns, FL_MOUTH_BOTTOM_INNER[0])
    t2x, t2y = _fl_point(values, columns, FL_MOUTH_TOP_INNER[1])
    b2x, b2y = _fl_point(values, columns, FL_MOUTH_BOTTOM_INNER[1])
    vertical = _dist(t1x, t1y, b1x, b1y) + _dist(t2x, t2y, b2x, b2y)
    horizontal = 2.0 * _dist(lx, ly, rx, ry) + 1e-9
    return vertical / horizontal


def _count_runs_below(series: np.ndarray, threshold: float) -> int:
    """Count contiguous runs where series < threshold (e.g. blinks, from EAR dips)."""
    below = series < threshold
    if len(below) == 0:
        return 0
    edges = np.diff(below.astype(int))
    return int(np.sum(edges == 1)) + (1 if below[0] else 0)


def _count_runs_above(series: np.ndarray, threshold: float) -> int:
    above = series > threshold
    if len(above) == 0:
        return 0
    edges = np.diff(above.astype(int))
    return int(np.sum(edges == 1)) + (1 if above[0] else 0)


def _pl_point(values: np.ndarray, columns: list[str], idx: int):
    xi, yi, zi = columns.index(f"{idx}_x"), columns.index(f"{idx}_y"), columns.index(f"{idx}_z")
    return values[:, xi], values[:, yi], values[:, zi]


def head_pose(values: np.ndarray, columns: list[str]):
    """Return (pitch, yaw, roll) in degrees, per frame. See module docstring
    for the geometric-proxy caveat."""
    lx, ly, lz = _pl_point(values, columns, PL_LEFT_EYE_OUTER)
    rx, ry, rz = _pl_point(values, columns, PL_RIGHT_EYE_OUTER)
    nx, ny, nz = _pl_point(values, columns, PL_NOSE)
    slx, sly, _ = _pl_point(values, columns, PL_LEFT_SHOULDER)
    srx, sry, _ = _pl_point(values, columns, PL_RIGHT_SHOULDER)

    roll = np.degrees(np.arctan2(ry - ly, rx - lx + 1e-9))
    yaw = np.degrees(np.arctan2(rz - lz, rx - lx + 1e-9))
    eye_mid_y = (ly + ry) / 2.0
    shoulder_mid_y = (sly + sry) / 2.0
    torso = np.abs(shoulder_mid_y - eye_mid_y) + 1e-6
    pitch = np.degrees(np.arctan2(ny - eye_mid_y, torso))
    return pitch, yaw, roll


def extract_vision_features(fl: Signal, pl: Signal, window_seconds: float) -> dict[str, float]:
    """One dict of scalar features for a single window's FL + PL slices."""
    out: dict[str, float] = {}
    minutes = window_seconds / 60.0

    if len(fl.times) > 0:
        ear_r = eye_aspect_ratio(fl.values, fl.columns, FL_RIGHT_EYE)
        ear_l = eye_aspect_ratio(fl.values, fl.columns, FL_LEFT_EYE)
        ear = (ear_r + ear_l) / 2.0
        mar = mouth_aspect_ratio(fl.values, fl.columns)

        out["ear_mean"] = float(np.mean(ear))
        out["ear_std"] = float(np.std(ear))
        out["mar_mean"] = float(np.mean(mar))
        out["mar_std"] = float(np.std(mar))
        out["perclos"] = float(np.mean(ear < EAR_THRESHOLD))
        out["blink_count"] = float(_count_runs_below(ear, EAR_THRESHOLD))
        out["blink_rate_per_min"] = out["blink_count"] / minutes
        out["yawn_count"] = float(_count_runs_above(mar, MAR_THRESHOLD))
        out["yawn_rate_per_min"] = out["yawn_count"] / minutes
    else:
        for key in (
            "ear_mean", "ear_std", "mar_mean", "mar_std", "perclos",
            "blink_count", "blink_rate_per_min", "yawn_count", "yawn_rate_per_min",
        ):
            out[key] = float("nan")

    if len(pl.times) > 0:
        pitch, yaw, roll = head_pose(pl.values, pl.columns)
        out["head_pitch_mean"] = float(np.mean(pitch))
        out["head_pitch_std"] = float(np.std(pitch))
        out["head_yaw_mean"] = float(np.mean(yaw))
        out["head_yaw_std"] = float(np.std(yaw))
        out["head_roll_mean"] = float(np.mean(roll))
        out["head_roll_std"] = float(np.std(roll))
        # nod/tilt "events": large frame-to-frame pitch/roll swings, a proxy
        # for head-nodding and head-tilt frequency called out in tasks.md 2.1
        out["head_nod_rate_per_min"] = float(np.sum(np.abs(np.diff(pitch)) > 5.0)) / minutes
        out["head_tilt_rate_per_min"] = float(np.sum(np.abs(np.diff(roll)) > 5.0)) / minutes
    else:
        for key in (
            "head_pitch_mean", "head_pitch_std", "head_yaw_mean", "head_yaw_std",
            "head_roll_mean", "head_roll_std", "head_nod_rate_per_min", "head_tilt_rate_per_min",
        ):
            out[key] = float("nan")

    return out
