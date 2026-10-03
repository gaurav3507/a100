"""Task 8.1 - unit tests for common/ feature functions against small,
hand-computed synthetic examples, verified before trusting these functions
on real UL-DD data.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from common.features_bio import hrv_features, spo2_features, temp_features
from common.features_grip import extract_grip_features
from common.features_telemetry import extract_telemetry_features
from common.features_vision import eye_aspect_ratio, mouth_aspect_ratio, head_pose, FL_RIGHT_EYE
from common.labels import bin_kss, HIGH, LOW, MEDIUM
from common.splits import loso_folds
from common.windowing import Signal


def _signal(times, values, columns):
    return Signal(times=np.asarray(times, dtype=float), values=np.asarray(values, dtype=float), columns=columns)


# --- labels.py ---

def test_bin_kss_boundaries():
    assert bin_kss(1) == LOW
    assert bin_kss(3.9) == LOW
    assert bin_kss(4) == MEDIUM
    assert bin_kss(6) == MEDIUM
    assert bin_kss(6.1) == HIGH
    assert bin_kss(9) == HIGH


# --- splits.py ---

def test_loso_folds_exclude_no_drowsy_and_full_coverage():
    folds = loso_folds()
    assert len(folds) == 16  # 19 subjects - {C, F, L}
    held_out = {f["held_out_subject"] for f in folds}
    assert held_out.isdisjoint({"C", "F", "L"})
    for f in folds:
        assert len(f["train_subjects"]) == 15
        assert f["held_out_subject"] not in f["train_subjects"]


# --- features_bio.py ---

def test_hrv_rmssd_sdnn_hand_computed():
    # IBI values in seconds; RMSSD/SDNN computed by hand below.
    ibi_values = [0.8, 0.9, 0.7, 1.0]
    ibi = _signal(times=[0, 1, 2, 3], values=[[v] for v in ibi_values], columns=["ibi"])

    diffs = [0.1, -0.2, 0.3]
    expected_rmssd = math.sqrt(sum(d * d for d in diffs) / len(diffs))  # 0.21602468994692867
    mean = sum(ibi_values) / len(ibi_values)  # 0.85
    expected_sdnn = math.sqrt(sum((v - mean) ** 2 for v in ibi_values) / len(ibi_values))  # 0.11180339887498948

    result = hrv_features(ibi)
    assert result["hrv_rmssd"] == pytest.approx(expected_rmssd, abs=1e-9)
    assert result["hrv_sdnn"] == pytest.approx(expected_sdnn, abs=1e-9)


def test_hrv_features_nan_when_fewer_than_two_beats():
    empty = _signal(times=[], values=np.empty((0, 1)), columns=["ibi"])
    one_beat = _signal(times=[0], values=[[0.8]], columns=["ibi"])
    for sig in (empty, one_beat):
        result = hrv_features(sig)
        assert math.isnan(result["hrv_rmssd"])
        assert math.isnan(result["hrv_sdnn"])


def test_temp_slope_hand_computed_linear_ramp():
    # temp rises exactly 0.01 degC per second over 10s -> slope should recover 0.01
    times = list(range(10))
    values = [[29.0 + 0.01 * t] for t in times]
    temp = _signal(times=times, values=values, columns=["temp"])
    result = temp_features(temp)
    assert result["temp_slope"] == pytest.approx(0.01, abs=1e-9)
    assert result["temp_mean"] == pytest.approx(29.0 + 0.01 * 4.5, abs=1e-9)


def test_spo2_features_nan_sentinel_rows_ignored():
    # mimics UL-DD's '--' sentinel rows, parsed upstream as NaN
    o2m = _signal(
        times=[0, 1, 2],
        values=[[97, 84, 0], [np.nan, np.nan, 0], [95, 80, 0]],
        columns=["spo2", "pulse_rate", "motion"],
    )
    result = spo2_features(o2m)
    assert result["spo2_mean"] == pytest.approx(96.0)
    assert result["spo2_min"] == pytest.approx(95.0)
    assert result["pulse_rate_mean"] == pytest.approx(82.0)


# --- features_grip.py ---

def test_grip_asymmetry_hand_computed():
    lgp = _signal(times=[0, 1, 2], values=[[2.0], [3.0], [4.0]], columns=["grip_left"])
    rgp = _signal(times=[0, 1, 2], values=[[5.0], [5.0], [5.0]], columns=["grip_right"])
    result = extract_grip_features(lgp, rgp)
    assert result["grip_left_mean"] == pytest.approx(3.0)
    assert result["grip_right_mean"] == pytest.approx(5.0)
    assert result["grip_asymmetry"] == pytest.approx(2.0)


# --- features_vision.py ---

def test_ear_hand_computed_geometric():
    # p1=37=(0,0) p2=38=(1,-1) p3=39=(3,-1) p4=40=(4,0) p5=41=(3,1) p6=42=(1,1)
    # vertical = dist(p2,p6) + dist(p3,p5) = 2 + 2 = 4
    # horizontal = 2 * dist(p1,p4) = 2 * 4 = 8 -> EAR = 4/8 = 0.5
    columns = [f"{ax}{i}" for i in FL_RIGHT_EYE for ax in ("X", "Y")]
    points = {37: (0, 0), 38: (1, -1), 39: (3, -1), 40: (4, 0), 41: (3, 1), 42: (1, 1)}
    row = []
    for i in FL_RIGHT_EYE:
        row.extend(points[i])
    values = np.array([row], dtype=float)
    ear = eye_aspect_ratio(values, columns, FL_RIGHT_EYE)
    assert ear[0] == pytest.approx(0.5, abs=1e-9)


def test_mar_hand_computed_geometric():
    # p49=(0,0) p55=(4,0) horizontal=4; p51=(1,-1) p59=(1,1) vertical1=2;
    # p53=(3,-1) p57=(3,1) vertical2=2 -> MAR = (2+2)/(2*4) = 0.5
    idxs = [49, 51, 53, 55, 57, 59]
    columns = [f"{ax}{i}" for i in idxs for ax in ("X", "Y")]
    points = {49: (0, 0), 51: (1, -1), 53: (3, -1), 55: (4, 0), 57: (3, 1), 59: (1, 1)}
    row = []
    for i in idxs:
        row.extend(points[i])
    values = np.array([row], dtype=float)
    mar = mouth_aspect_ratio(values, columns)
    assert mar[0] == pytest.approx(0.5, abs=1e-9)


def test_head_pose_hand_computed():
    columns = []
    points = {
        3: (0, 0, 0), 6: (1, 0, 1),  # left/right eye outer: roll=atan2(0,1)=0, yaw=atan2(1,1)=45deg
        0: (0.5, -0.5, 0),  # nose
        11: (0, -1, 0), 12: (1, -1, 0),  # shoulders -> eye_mid_y=0, shoulder_mid_y=-1, torso=1
    }
    for i in (3, 6, 0, 11, 12):
        columns += [f"{i}_x", f"{i}_y", f"{i}_z"]
    row = []
    for i in (3, 6, 0, 11, 12):
        row.extend(points[i])
    values = np.array([row], dtype=float)

    pitch, yaw, roll = head_pose(values, columns)
    assert roll[0] == pytest.approx(0.0, abs=1e-6)
    assert yaw[0] == pytest.approx(45.0, abs=1e-6)
    # pitch = atan2(nose.y - eye_mid_y, torso) = atan2(-0.5 - 0, 1) = -26.565...deg
    # (torso has a +1e-6 epsilon in the implementation to avoid div-by-zero)
    assert pitch[0] == pytest.approx(math.degrees(math.atan2(-0.5, 1.0 + 1e-6)), abs=1e-6)


# --- features_telemetry.py ---

def test_heading_change_rate_wraps_correctly():
    # heading goes 350 -> 10deg: true change is +20deg, not -340deg
    telemetry = _signal(
        times=[0, 1],
        values=[[0, 0, 0, 350.0, 0.0, 0.0, 10.0, 1000, 3], [0, 0, 0, 10.0, 0.0, 0.0, 10.0, 1000, 3]],
        columns=["raw rendering ts", "raw simulation ts", "raw paused simulation ts",
                 "heading[deg]", "pitch[deg]", "roll[deg]", "speed[m/s]", "rpm", "gear"],
    )
    result = extract_telemetry_features(telemetry)
    assert result["heading_change_rate"] == pytest.approx(20.0, abs=1e-6)
    assert result["speed_mean"] == pytest.approx(10.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
