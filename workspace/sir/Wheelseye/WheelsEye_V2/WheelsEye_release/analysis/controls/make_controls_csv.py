"""Convert the WheelsEye V2 feature parquet into the CSV schema that
wheelseye_revision_controls.py expects.

Mapping applied:
    session   'A' -> 'alert', 'D' -> 'drowsy'
    t         window_id * step_seconds   (default step 5 s)
    label     0 -> 'Low', 1 -> 'Medium', 2 -> 'High'
    features  all remaining numeric columns pass through unchanged; the
              controls script separates drift/context columns by its own regex.

Rows whose fold marks them as non-evaluable are kept: the controls script
decides evaluability itself (subjects that have a drowsy session).

Usage (from the WheelsEye_V2 root):
    python make_controls_csv_v1.py                     # writes v2_windows.csv
    python make_controls_csv_v1.py --step 5 --out v2_windows.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SESSION_MAP = {"A": "alert", "D": "drowsy", "alert": "alert", "drowsy": "drowsy"}
LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=Path("data/processed/uldd_features.parquet"))
    ap.add_argument("--out", type=Path, default=Path("v2_windows.csv"))
    ap.add_argument("--step", type=float, default=5.0, help="window step in seconds (t = window_id * step)")
    a = ap.parse_args()

    df = pd.read_parquet(a.parquet)
    missing = [c for c in ("subject", "session", "window_id", "label") if c not in df.columns]
    if missing:
        raise SystemExit(f"parquet is missing required columns: {missing}")

    out = df.copy()
    out["session"] = out["session"].map(SESSION_MAP)
    if out["session"].isna().any():
        bad = sorted(df.loc[out["session"].isna(), "session"].unique())
        raise SystemExit(f"unmapped session values: {bad}")

    if pd.api.types.is_numeric_dtype(out["label"]):
        out["label"] = out["label"].map(LABEL_MAP)
    if out["label"].isna().any():
        raise SystemExit("unmapped label values; expected 0/1/2 or Low/Medium/High")

    out["t"] = out["window_id"].astype(float) * a.step
    out = out.drop(columns=[c for c in ("window_id", "fold", "t3_fold", "block") if c in out.columns])

    # columns the controls script treats as metadata must come first for readability
    front = [c for c in ("subject", "session", "t", "label", "kss") if c in out.columns]
    out = out[front + [c for c in out.columns if c not in front]]

    out.to_csv(a.out, index=False)
    feats = [c for c in out.columns if c not in ("subject", "session", "t", "label", "kss")
             and pd.api.types.is_numeric_dtype(out[c])]
    drift = [c for c in feats if any(s in c.lower() for s in ("drift", "ctx", "context"))]
    print(f"wrote {a.out}: {len(out)} windows, {out.subject.nunique()} subjects, "
          f"{len(feats)} numeric features ({len(drift)} matched as drift/context)")
    print(f"  sessions: {out.session.value_counts().to_dict()}")
    print(f"  labels:   {out.label.value_counts().to_dict()}")
    print(f"  t range per session: {out.groupby('session')['t'].max().round(0).to_dict()} s")
    evaluable = sorted(set(out.loc[out.session == 'drowsy', 'subject']))
    print(f"  evaluable subjects (have a drowsy session): {len(evaluable)}")
    if drift:
        print(f"  drift/context columns: {drift[:6]}{' ...' if len(drift) > 6 else ''}")
    else:
        print("  WARNING: no drift/context columns matched -- pass --drift-regex to the controls script")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
