"""Unit tests for the v2 additions (same Task-8.1 standard as
tests/test_features.py: small hand-computed examples for every new feature
function, plus structural invariants for the new fold generators -- the
cross-fold invariants v1's test suite lacked, which let defect #3 through).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from common.calibration import apply_alert_baseline
from common.context_features import add_context_features
from common.features_acc import extract_acc_features
from common.features_cardiac_spectral import extract_bvp_spectral_features
from common.features_fau import FAU_COLUMNS, KEY_AUS, extract_fau_features
from common.features_posture import extract_posture_features
from common.smoothing import causal_majority_smooth, hysteresis_alert_stream, smooth_by_session
from common.splits import (
    block_folds, loso_folds, merge_window_folds, stratified_window_folds,
)
from common.windowing import Signal


def _signal(times, values, columns):
    return Signal(times=np.asarray(times, dtype=float),
                  values=np.asarray(values, dtype=float), columns=list(columns))


# --- features_fau.py ---

def test_fau_mean_std_max_hand_computed():
    values = np.zeros((2, 30))
    j = FAU_COLUMNS.index("lid_droop")
    values[:, j] = [1.0, 3.0]
    fau = _signal([0, 1], values, FAU_COLUMNS)
    out = extract_fau_features(fau)
    assert out["fau_lid_droop_mean"] == pytest.approx(2.0)
    assert out["fau_lid_droop_std"] == pytest.approx(1.0)
    assert out["fau_lid_droop_max"] == pytest.approx(3.0)
    assert out["fau_inner_brow_raiser_mean"] == pytest.approx(0.0)
    assert len(out) == 30 * 2 + len(KEY_AUS)


def test_fau_empty_window_is_nan():
    fau = _signal([], np.empty((0, 30)), FAU_COLUMNS)
    out = extract_fau_features(fau)
    assert all(math.isnan(v) for v in out.values())


def test_fau_wrong_columns_fail_loudly():
    fau = _signal([0], np.zeros((1, 30)), [f"au_{i}" for i in range(30)])
    with pytest.raises(ValueError):
        extract_fau_features(fau)


# --- features_acc.py ---

def test_acc_hand_computed():
    # magnitudes: |(1,2,2)|=3, |(2,3,6)|=7 -> mean 5, std 2; jerk |7-3|=4
    acc = _signal([0, 1], [[1, 2, 2], [2, 3, 6]], ["acc_x", "acc_y", "acc_z"])
    out = extract_acc_features(acc)
    assert out["acc_mag_mean"] == pytest.approx(5.0)
    assert out["acc_mag_std"] == pytest.approx(2.0)
    assert out["acc_jerk_mean"] == pytest.approx(4.0)
    assert out["acc_still_fraction"] == pytest.approx(0.0)  # 4 >= threshold 3


def test_acc_single_sample_is_nan():
    acc = _signal([0], [[1, 1, 1]], ["acc_x", "acc_y", "acc_z"])
    assert all(math.isnan(v) for v in extract_acc_features(acc).values())


# --- features_posture.py ---

def test_posture_hand_computed():
    # two identical frames: level shoulders at y=1, eyes at y=0.4, nose
    # offset +0.1 from shoulder midpoint -> angle 0, drop 0.6, lean 0.1,
    # zero stds and zero motion
    points = {11: (0.0, 1.0), 12: (1.0, 1.0), 3: (0.3, 0.4), 6: (0.7, 0.4), 0: (0.6, 0.5)}
    columns, row = [], []
    for idx, (x, y) in points.items():
        columns += [f"{idx}_x", f"{idx}_y"]
        row += [x, y]
    pl = _signal([0, 1], [row, row], columns)
    out = extract_posture_features(pl)
    assert out["posture_shoulder_angle_mean"] == pytest.approx(0.0, abs=1e-6)
    assert out["posture_head_drop_mean"] == pytest.approx(0.6)
    assert out["posture_lean_mean"] == pytest.approx(0.1)
    assert out["posture_shoulder_angle_std"] == pytest.approx(0.0, abs=1e-9)
    assert out["posture_motion_mean"] == pytest.approx(0.0, abs=1e-9)


# --- features_cardiac_spectral.py ---

def test_bvp_beat_rate_on_synthetic_pulse_train():
    # 1 Hz sinusoid over 120s at 64 Hz: one peak per second -> ~60 beats/min
    t = np.arange(0, 120, 1 / 64.0)
    bvp = _signal(t, np.sin(2 * np.pi * 1.0 * t)[:, None], ["bvp"])
    out = extract_bvp_spectral_features(bvp)
    assert 55.0 <= out["bvp_beat_rate"] <= 65.0
    # perfectly regular rhythm -> IBI variability ~0 -> negligible band power
    assert out["bvp_lf_power"] == pytest.approx(0.0, abs=1e-6)
    assert out["bvp_amp_std"] == pytest.approx(1 / math.sqrt(2), abs=0.05)


def test_bvp_short_context_is_nan():
    t = np.arange(0, 10, 1 / 64.0)  # 10s << MIN_CONTEXT_SECONDS
    bvp = _signal(t, np.sin(2 * np.pi * t)[:, None], ["bvp"])
    assert all(math.isnan(v) for v in extract_bvp_spectral_features(bvp).values())


# --- calibration.py ---

def _calib_df():
    return pd.DataFrame({
        "subject": ["X", "X", "X", "Y", "Y", "Y"],
        "session": ["A", "A", "D", "A", "A", "D"],
        "f": [1.0, 3.0, 5.0, 2.0, 2.0, 4.0],
    })


def test_alert_baseline_hand_computed():
    # X alert: mean 2, std(ddof=0) 1 -> D row (5-2)/1 = 3
    # Y alert: std 0 -> fallback 1     -> D row (4-2)/1 = 2
    out = apply_alert_baseline(_calib_df(), ["f"])
    assert out.loc[2, "f"] == pytest.approx(3.0)
    assert out.loc[5, "f"] == pytest.approx(2.0)
    assert out.loc[0, "f"] == pytest.approx(-1.0)  # alert rows calibrate too


def test_alert_baseline_append_mode_keeps_raw():
    out = apply_alert_baseline(_calib_df(), ["f"], mode="append")
    assert out.loc[2, "f"] == pytest.approx(5.0)
    assert out.loc[2, "f_cal"] == pytest.approx(3.0)


def test_alert_baseline_requires_awake_session():
    df = pd.DataFrame({"subject": ["Z"], "session": ["D"], "f": [1.0]})
    with pytest.raises(ValueError):
        apply_alert_baseline(df, ["f"])


# --- context_features.py ---

def test_context_drift_hand_computed():
    df = pd.DataFrame({
        "subject": ["X"] * 3, "session": ["A"] * 3,
        "t_start": [0.0, 100.0, 400.0], "t_end": [10.0, 110.0, 410.0],
        "hr_mean": [10.0, 20.0, 30.0],
    })
    out = add_context_features(df, drift_features=("hr_mean",))
    # opening (< 300s) mean = 15 -> drifts -5, +5, +15; minutes = midpoint/60
    assert out["hr_mean_drift"].tolist() == pytest.approx([-5.0, 5.0, 15.0])
    assert out.loc[0, "minutes_elapsed"] == pytest.approx(5.0 / 60.0)


# --- smoothing.py ---

def test_causal_majority_hand_traces():
    assert causal_majority_smooth(np.array([2, 0, 2, 2]), k=2).tolist() == [2, 0, 2, 2]
    assert causal_majority_smooth(np.array([0, 2, 2, 0, 2]), k=3).tolist() == [0, 2, 2, 2, 2]


def test_hysteresis_alert_stream():
    alert = hysteresis_alert_stream(np.array([2, 2, 0, 2, 2, 2]), consecutive=2)
    assert alert.tolist() == [False, True, False, False, True, True]


def test_smooth_by_session_is_per_stream_and_ordered():
    meta = pd.DataFrame({
        "subject": ["X", "X", "X", "Y"], "session": ["A"] * 4,
        "window_id": [1, 0, 2, 0],  # deliberately out of order
    })
    preds = np.array([0, 2, 2, 1])
    out = smooth_by_session(meta, preds, k=3)
    # X stream in window_id order is [2, 0, 2]: i0 -> 2; i1 window [2,0] ties
    # toward the current raw pred (0); i2 window [2,0,2] -> majority 2.
    # Smoothed stream [2, 0, 2] maps back to rows as w1=0, w0=2, w2=2.
    assert out.tolist() == [0, 2, 2, 1]  # Y's single window untouched


# --- splits.py (cross-fold invariants) ---

def _windows_df(n_sessions=2, n_windows=100):
    rows = []
    for s in range(n_sessions):
        for w in range(n_windows):
            rows.append({
                "subject": chr(ord("A") + s), "session": "A", "window_id": w,
                "t_start": w * 5.0, "t_end": w * 5.0 + 10.0,
                "label": w % 3,
            })
    return pd.DataFrame(rows)


def test_stratified_window_folds_partition():
    df = _windows_df()
    folds = stratified_window_folds(df, n_splits=5, seed=0)
    assert len(folds) == len(df)
    assert set(folds["fold"].unique()) == {0, 1, 2, 3, 4}
    # every fold sees every class (stratification)
    merged = merge_window_folds(df, folds)
    for f in range(5):
        assert set(merged[merged["exp_fold"] == f]["label"].unique()) == {0, 1, 2}


def test_block_folds_purge_and_contiguity():
    df = _windows_df(n_sessions=1, n_windows=100)  # t_start 0..495
    folds = block_folds(df, n_splits=5, block_seconds=240.0)
    merged = df.merge(folds, on=["subject", "session", "window_id"])
    by_start = merged.set_index("t_start")["fold"]
    assert by_start[0.0] == by_start[230.0]        # same block, same fold
    assert by_start[235.0] == -1                    # straddles 240s edge: purged
    assert by_start[240.0] != by_start[230.0]       # next block, different fold
    # no non-purged window's span crosses a block edge
    kept = merged[merged["fold"] >= 0]
    assert ((kept["t_start"] // 240) == ((kept["t_end"] - 1e-9) // 240)).all()


def test_merge_window_folds_fails_on_missing_assignment():
    df = _windows_df()
    folds = stratified_window_folds(df, n_splits=5).iloc[:-1]  # drop one row
    with pytest.raises(ValueError):
        merge_window_folds(df, folds)


def test_loso_awake_only_in_train():
    folds = loso_folds(include_awake_only_in_train=True)
    assert len(folds) == 16
    for f in folds:
        assert len(f["train_subjects"]) == 18  # 15 foldable + C, F, L
        assert {"C", "F", "L"} <= set(f["train_subjects"])
        assert f["held_out_subject"] not in ("C", "F", "L")
        assert f["held_out_subject"] not in f["train_subjects"]


# --- dataset.py train-only statistics ---

def test_dataset_stats_source_uses_train_statistics_only():
    torch = pytest.importorskip("torch")  # noqa: F841
    from common.dataset import ULDDWindowDataset

    meta = {"session": "A", "window_id": 0}
    train = pd.DataFrame([
        {"subject": "X", "label": 0, "f1": 0.0, "f2": 4.0, "f3": np.nan, **meta},
        {"subject": "X", "label": 1, "f1": 2.0, "f2": 6.0, "f3": np.nan, **meta},
    ])
    test = pd.DataFrame([
        {"subject": "Y", "label": 2, "f1": 1.0, "f2": np.nan, "f3": np.nan, **meta},
    ])
    ds = ULDDWindowDataset(test, groups={"g": ["f1", "f2", "f3"]}, stats_source=train)
    flat = ds[0]["flat"].numpy()
    # f1: (1 - mean 1) / std sqrt(2) = 0
    # f2: NaN -> train median 5 -> (5 - mean 5) / std = 0
    # f3: all-NaN stats column -> fill 0, mean->0, std->1 -> 0 (no NaN leak)
    assert flat == pytest.approx([0.0, 0.0, 0.0])
    assert not np.isnan(flat).any()


# --- track_a_heavy/i2m2.py forward shape ---

def test_i2m2_forward_shapes_and_loss():
    torch = pytest.importorskip("torch")
    from track_a_heavy.i2m2 import I2M2Model, i2m2_loss

    model = I2M2Model({"a": 3, "b": 2}, num_classes=3)
    model.eval()
    nodes = {"a": torch.zeros(4, 3), "b": torch.zeros(4, 2)}
    with torch.no_grad():
        out = model(nodes)
    assert out["logits"].shape == (4, 3)
    assert out["expert_logits"]["a"].shape == (4, 3)
    loss, parts = i2m2_loss(out, torch.tensor([0, 1, 2, 0]))
    assert loss.ndim == 0 and "expert_a" in parts


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
