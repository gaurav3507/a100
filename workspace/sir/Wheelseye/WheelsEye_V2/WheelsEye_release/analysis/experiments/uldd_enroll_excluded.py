"""UL-DD: enrollment-excluded calibration run (protocol-matched to the RLDD analysis).

The headline UL-DD figure calibrates each driver against the first k windows of their
Awake session and then SCORES those windows too. Because calibration drives the
enrollment windows toward zero, scoring them inflates accuracy. This script reports
both variants so the paper can state the gap and use the deployment-faithful number.

Usage (from the WheelsEye_V2 root, venv active):
    python uldd_enroll_excluded_v1.py                 # k=96, mean-only and z-score, both variants
    python uldd_enroll_excluded_v1.py --k 48 144
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

RESULTS = Path("data/processed/results")


def groups_and_context(df):
    try:
        import common.dataset as ds
        g = dict(getattr(ds, "V2_FEATURE_GROUPS", None) or getattr(ds, "CORE_FEATURE_GROUPS"))
        sensor = [c for v in g.values() for c in v if c in df.columns]
    except Exception:
        meta = {"subject","session","window_id","label","fold","kss","block"}
        sensor = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    try:
        from common.context_features import CONTEXT_FEATURES
        ctx = [c for c in CONTEXT_FEATURES if c in df.columns]
    except Exception:
        ctx = [c for c in df.columns if c.startswith(("ctx_", "drift_"))]
    return sensor, [c for c in ctx if c not in sensor]


def calibrate(df, sensor, mode, k):
    """Returns (calibrated df, enrollment mask). Baseline = first k Awake windows per subject."""
    out = df.copy(); out[sensor] = out[sensor].astype("float64")
    enroll = np.zeros(len(df), dtype=bool)
    if mode == "none":
        return out, enroll
    for s, g in df.groupby("subject"):
        src = g[g["session"] == "A"].sort_values("window_id")
        if k: src = src.head(k)
        if len(src) < 2: continue
        enroll[out.index.get_indexer(src.index)] = True
        mu = src[sensor].mean().fillna(0.0)
        m = (out["subject"] == s).to_numpy()
        vals = out.loc[m, sensor] - mu
        if mode == "zscore":
            sd = src[sensor].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
            vals = vals / sd
        out.loc[m, sensor] = vals.to_numpy()
    return out, enroll


def run(df, feats, enroll, exclude, seed=0):
    import lightgbm as lgb
    from sklearn.metrics import f1_score
    subs = sorted(s for s, g in df.groupby("subject") if {"A","D"} <= set(g["session"]))
    X = np.nan_to_num(df[feats].to_numpy(np.float32)); y = df["label"].to_numpy()
    sarr = df["subject"].to_numpy(); folds = []
    for s in subs:
        te = sarr == s; tr = ~te
        if exclude: te = te & ~enroll
        if te.sum() == 0 or len(np.unique(y[te])) < 2: continue
        m = lgb.LGBMClassifier(n_estimators=200, verbosity=-1, random_state=seed).fit(X[tr], y[tr])
        p = m.predict(X[te])
        folds.append(dict(subject=s, accuracy=float((p == y[te]).mean()),
                          f1_macro=float(f1_score(y[te], p, average="macro", zero_division=0)),
                          n=int(te.sum())))
    return folds


def paired(a, b, key="accuracy"):
    da = {f["subject"]: f[key] for f in a}; db = {f["subject"]: f[key] for f in b}
    c = sorted(set(da) & set(db)); x = np.array([da[s] for s in c]); z = np.array([db[s] for s in c])
    d = x - z; nz = d[d != 0]
    rng = np.random.default_rng(0); boot = rng.choice(d, size=(20000, len(d)), replace=True).mean(axis=1)
    return dict(a=x.mean(), b=z.mean(), delta=d.mean(),
                ci=(np.percentile(boot,2.5), np.percentile(boot,97.5)),
                p=float(stats.wilcoxon(nz).pvalue) if len(nz) >= 5 else float("nan"),
                wins=int((d>0).sum()), losses=int((d<0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-path", type=Path, default=Path("data/processed/uldd_features.parquet"))
    ap.add_argument("--k", type=int, nargs="*", default=[96])
    ap.add_argument("--out", type=Path, default=RESULTS / "uldd_enrollment_excluded.json")
    a = ap.parse_args()

    df = pd.read_parquet(a.features_path)
    sensor, ctx = groups_and_context(df)
    feats = sensor + ctx
    print(f"UL-DD: {len(df)} windows, {df.subject.nunique()} subjects, {len(sensor)} sensor + {len(ctx)} context features")

    out, store = {}, {}
    for mode in ("none", "mean", "zscore"):
        for k in ([0] if mode == "none" else a.k):
            cal, enroll = calibrate(df, sensor, mode, k)
            for excl in (False, True):
                if mode == "none" and excl: continue
                f = run(df if mode == "none" else cal, feats, enroll, excl)
                tag = mode + (f"_k{k}" if k else "") + ("_excl" if excl else "")
                store[tag] = f
                acc = np.array([x["accuracy"] for x in f]); f1 = np.array([x["f1_macro"] for x in f])
                out[tag] = dict(accuracy=float(acc.mean()), sd=float(acc.std()), f1=float(f1.mean()), folds=len(f))
                print(f"  {tag:<18} acc={acc.mean()*100:6.2f}+/-{acc.std()*100:4.1f}  f1={f1.mean()*100:6.2f}  folds={len(f)}")

    print("\nENROLLMENT-SCORING INFLATION (scored minus excluded)")
    for base in [t for t in store if not t.endswith("_excl") and t != "none"]:
        e = base + "_excl"
        if e in store:
            r = paired(store[base], store[e])
            print(f"  {base:<16} scored {r['a']*100:6.2f}  excluded {r['b']*100:6.2f}  "
                  f"inflation {r['delta']*100:+5.2f} pp  CI[{r['ci'][0]*100:+.1f},{r['ci'][1]*100:+.1f}]  p={r['p']:.4f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(summary=out, per_fold=store), indent=2, default=float))
    print(f"\nsaved {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
