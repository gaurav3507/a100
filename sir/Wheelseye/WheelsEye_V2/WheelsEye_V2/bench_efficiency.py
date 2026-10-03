"""Efficiency benchmark for the deployment model (LightGBM + per-driver calibration).

Trains the deploy model on all subjects (the configuration a shipped device
would carry), then measures what the "lightweight, real-time" claim needs:
model size on disk, parameter/tree counts, single-window inference latency
(the deployment pattern: one 10 s window every 5 s), batch latency, and --
if onnxmltools/onnxruntime are installed -- ONNX export size and latency.
The calibration transform ((x - mean) / std per feature) is included in the
timed path so the number is end-to-end from features to alert level.

Run from the WheelsEye_V2 root (venv active):
    python bench_efficiency.py                      # defaults
    python bench_efficiency.py --n-estimators 200 --repeats 2000
Optional ONNX path:  pip install onnxmltools onnxruntime skl2onnx
Writes data/processed/results/efficiency_lightgbm.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

META = {"subject", "session", "window_id", "label", "fold", "t3_fold", "window_start", "window_end", "kss"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Prefer the repo's V2 feature list; fall back to all numeric non-meta columns."""
    try:
        from common.dataset import V2_FEATURES  # type: ignore
        try:
            from common.context_features import CONTEXT_FEATURES  # type: ignore
            cols = list(V2_FEATURES) + list(CONTEXT_FEATURES)
        except Exception:
            cols = list(V2_FEATURES)
        cols = [c for c in cols if c in df.columns]
        if cols:
            return cols
    except Exception:
        pass
    return [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]


def calibrate_all(df: pd.DataFrame, feats: list[str], n_enroll: int) -> pd.DataFrame:
    """Per-driver baseline from first n_enroll Awake windows (0 = whole Awake session)."""
    out = df.copy(); out[feats] = out[feats].astype("float64")
    for subj, g in df[df["session"] == "A"].groupby("subject"):
        g = g.sort_values("window_id"); g = g.head(n_enroll) if n_enroll else g
        mean = g[feats].mean().fillna(0.0); std = g[feats].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        m = (out["subject"] == subj).to_numpy()
        out.loc[m, feats] = ((out.loc[m, feats] - mean) / std).to_numpy()
    return out


def time_calls(fn, repeats: int) -> dict:
    for _ in range(50): fn()                      # warm-up
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return dict(mean_ms=statistics.fmean(ts), p50_ms=ts[len(ts)//2], p95_ms=ts[int(len(ts)*.95)], max_ms=ts[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-path", type=Path, default=Path("data/processed/uldd_features.parquet"))
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--n-enroll", type=int, default=96, help="enrollment windows for calibration (0 = whole session)")
    ap.add_argument("--repeats", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("data/processed/results/efficiency_lightgbm.json"))
    args = ap.parse_args()

    import lightgbm as lgb
    df = pd.read_parquet(args.features_path)
    feats = feature_columns(df)
    df = calibrate_all(df, feats, args.n_enroll)
    X = np.nan_to_num(df[feats].to_numpy(dtype=np.float32)); y = df["label"].to_numpy()
    print(f"rows={len(df)}  features={len(feats)}  classes={sorted(set(y))}")

    t0 = time.perf_counter()
    model = lgb.LGBMClassifier(n_estimators=args.n_estimators, verbosity=-1, random_state=0).fit(X, y)
    train_s = time.perf_counter() - t0
    booster = model.booster_

    # --- size ---
    pkl_bytes = len(pickle.dumps(model))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.txt"; booster.save_model(str(p)); native_bytes = p.stat().st_size
    n_trees = booster.num_trees()
    n_leaves = int(sum(booster.dump_model()["tree_info"][i]["num_leaves"] for i in range(n_trees)))

    # --- latency: end-to-end single window (calibration arithmetic + predict) ---
    mean_v = np.zeros(len(feats), dtype=np.float32); std_v = np.ones(len(feats), dtype=np.float32)
    x1 = X[:1].copy(); x16 = X[:16].copy()
    def one_window():
        xc = (x1 - mean_v) / std_v
        return model.predict(xc)
    def batch16():
        xc = (x16 - mean_v) / std_v
        return model.predict(xc)
    lat1 = time_calls(one_window, args.repeats); lat16 = time_calls(batch16, max(200, args.repeats // 5))

    res = dict(config=dict(n_estimators=args.n_estimators, n_enroll=args.n_enroll, n_features=len(feats), n_rows=int(len(df))),
               train_seconds=train_s, n_trees=n_trees, n_leaves_total=n_leaves,
               size_bytes=dict(pickle=pkl_bytes, lightgbm_native=native_bytes),
               latency_single_window_ms=lat1, latency_batch16_ms=lat16,
               window_step_seconds=5.0, realtime_budget_used_pct=lat1["p95_ms"] / 5000 * 100)

    # --- optional ONNX ---
    try:
        import onnxmltools, onnxruntime as ort
        from onnxmltools.convert.common.data_types import FloatTensorType
        onx = onnxmltools.convert_lightgbm(model, initial_types=[("x", FloatTensorType([None, len(feats)]))], zipmap=False)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.onnx"; p.write_bytes(onx.SerializeToString()); onnx_bytes = p.stat().st_size
            sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        ref = model.predict(X[:256]); got = np.asarray(sess.run(None, {name: X[:256]})[0]).reshape(-1)
        agree = float((ref == got).mean())
        lat_onnx = time_calls(lambda: sess.run(None, {name: x1}), args.repeats)
        res["onnx"] = dict(size_bytes=onnx_bytes, single_window_ms=lat_onnx, label_agreement_256=agree)
    except Exception as ex:  # library missing or export unsupported -- report, don't fail
        res["onnx"] = dict(skipped=str(ex)[:120])

    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(res, indent=2))

    print("\nEFFICIENCY -- LightGBM deploy model (all subjects, calibrated features)")
    print(f"  trees {n_trees}  |  leaves {n_leaves:,}  |  features {len(feats)}  |  train {train_s:.1f}s")
    print(f"  size: native {native_bytes/1024:.0f} KB  |  pickle {pkl_bytes/1024:.0f} KB")
    print(f"  latency, 1 window (calib+predict): mean {lat1['mean_ms']:.3f} ms  p95 {lat1['p95_ms']:.3f} ms  max {lat1['max_ms']:.2f} ms")
    print(f"  latency, batch of 16:               mean {lat16['mean_ms']:.3f} ms")
    print(f"  real-time budget used (p95 / 5 s step): {res['realtime_budget_used_pct']:.4f} %")
    o = res["onnx"]
    if "skipped" in o:
        onnx_line = "skipped -- " + o["skipped"]
    else:
        onnx_line = "%.0f KB, %.3f ms p95, label agreement %.3f" % (
            o["size_bytes"] / 1024, o["single_window_ms"]["p95_ms"], o["label_agreement_256"])
    print("  ONNX: " + onnx_line)
    print(f"  saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
