#!/usr/bin/env python3
"""
wheelseye_revision_controls.py
==============================

Reviewer-requested controls for the WheelsEye manuscript, computed on the V2
window-level feature table with the paper's own protocol (R3 LOSO over the
evaluable subjects, all other subjects in the training pool, LightGBM with 200
boosting rounds per class, mean subtraction of the SENSOR features from the
driver's first k alert windows, drift-context features left untouched).

Analyses
--------
  A. class counts, majority-class baseline, prior-matched random baseline
  B. time-on-task controls: sensor-only, sensor+drift (headline), sensor+elapsed,
     sensor+drift(permuted within session), elapsed-only, drift-only
  C. within-drowsy-session Medium-vs-High discrimination, uncalibrated vs calibrated
  D. wrong-state enrollment (baseline from the first k windows of the DROWSY session)
  E. single-pipeline per-fold logging of the headline configuration
  F. warning-level metrics from the headline predictions
     (causal three-window majority, alarm = High)

Input CSV schema (one row per 10-s window)
------------------------------------------
  subject : str                     e.g. 'A' ... 'S'
  session : 'alert' | 'drowsy'
  t       : float, seconds from session start (window start)
  kss     : int 1..9                (optional if 'label' is present)
  label   : 'Low' | 'Medium' | 'High' (optional if 'kss' is present)
  every remaining numeric column is a feature. Drift-context columns are
  identified by --drift-regex (default matches 'drift' or 'ctx' or 'context').

Usage
-----
  python wheelseye_revision_controls.py --table v2_windows.csv --out results/
  python wheelseye_revision_controls.py --demo --out demo_results/      # synthetic smoke test

Outputs (in --out)
------------------
  results.json           all numbers
  per_subject.csv        per-fold accuracies of the headline configuration
  predictions.csv        window-level predictions of the headline configuration
  controls_snippets.tex  table rows ready to paste into main.tex
"""
import argparse, json, os, re, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    import lightgbm as lgb
except ImportError:
    sys.exit("lightgbm is required: pip install lightgbm")

CLASSES = ["Low", "Medium", "High"]
CIDX = {c: i for i, c in enumerate(CLASSES)}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def kss_to_label(k):
    return "Low" if k < 4 else ("Medium" if k <= 6 else "High")


def load_table(path, drift_regex):
    df = pd.read_csv(path)
    need = {"subject", "session", "t"}
    if not need.issubset(df.columns):
        sys.exit(f"table must contain columns {need}")
    if "label" not in df.columns:
        if "kss" not in df.columns:
            sys.exit("table needs 'label' or 'kss'")
        df["label"] = df["kss"].apply(kss_to_label)
    df["y"] = df["label"].map(CIDX)
    meta = {"subject", "session", "t", "kss", "label", "y"}
    feats = [c for c in df.columns if c not in meta and np.issubdtype(df[c].dtype, np.number)]
    drift = [c for c in feats if re.search(drift_regex, c, flags=re.I)]
    sensor = [c for c in feats if c not in drift]
    df = df.sort_values(["subject", "session", "t"]).reset_index(drop=True)
    return df, sensor, drift


def evaluable_subjects(df):
    has_drowsy = set(df.loc[df.session == "drowsy", "subject"])
    return sorted(has_drowsy)


def enrollment_index(df, subj, k, session="alert"):
    """Row indices of the first k windows (by t) of the given session of subj."""
    rows = df.index[(df.subject == subj) & (df.session == session)]
    return rows[:k]


def calibrate(df, sensor, k, mode="mean", enroll_session_for=None):
    """
    Return a copy of df with SENSOR columns calibrated per subject from that
    subject's first k ALERT windows (mean-only or z-score). enroll_session_for
    may map a subject -> session to draw the baseline from a different session
    (used by the wrong-state control). Also returns the set of enrollment rows.
    """
    out = df.copy()
    enroll_rows = set()
    for s in out.subject.unique():
        sess = (enroll_session_for or {}).get(s, "alert")
        rows = enrollment_index(out, s, k, sess)
        if len(rows) == 0:
            continue
        base = out.loc[rows, sensor]
        mu = base.mean(axis=0)
        srows = out.index[out.subject == s]
        if mode == "mean":
            out.loc[srows, sensor] = out.loc[srows, sensor].values - mu.values
        elif mode == "z":
            sd = base.std(axis=0, ddof=0).replace(0, 1.0)
            out.loc[srows, sensor] = (out.loc[srows, sensor].values - mu.values) / sd.values
        enroll_rows.update(rows.tolist())
    return out, enroll_rows


def add_elapsed(df):
    out = df.copy()
    out["elapsed_s"] = out["t"]
    T = out.groupby(["subject", "session"])["t"].transform("max").replace(0, 1.0)
    out["elapsed_frac"] = out["t"] / T
    return out


def permute_drift_within_session(df, drift, seed=0):
    out = df.copy()
    rng = np.random.default_rng(seed)
    for _, idx in out.groupby(["subject", "session"]).groups.items():
        idx = np.array(list(idx))
        perm = rng.permutation(len(idx))
        out.loc[idx, drift] = out.loc[idx[perm], drift].values
    return out


def fit_predict(Xtr, ytr, Xte, n_classes, rounds=200, seed=0, threads=1):
    params = {"verbosity": -1, "seed": seed, "num_threads": threads, "deterministic": True}
    if n_classes > 2:
        params.update({"objective": "multiclass", "num_class": n_classes})
        bst = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=rounds)
        return bst.predict(Xte).argmax(axis=1)
    params.update({"objective": "binary"})
    bst = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=rounds)
    return (bst.predict(Xte) >= 0.5).astype(int)


def summarize(y_true, y_pred, n_classes, fold_accs):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for a, b in zip(y_true, y_pred):
        cm[a, b] += 1
    recall = np.array([cm[c, c] / cm[c].sum() if cm[c].sum() else np.nan for c in range(n_classes)])
    prec = np.array([cm[c, c] / cm[:, c].sum() if cm[:, c].sum() else 0.0 for c in range(n_classes)])
    f1 = np.where(recall + prec > 0, 2 * recall * prec / (recall + prec + 1e-12), 0.0)
    return {
        "acc_mean_over_folds": float(np.mean(fold_accs)) * 100,
        "acc_sd_over_folds": float(np.std(fold_accs, ddof=1)) * 100 if len(fold_accs) > 1 else 0.0,
        "acc_pooled": float((y_true == y_pred).mean()) * 100,
        "balanced_acc_pooled": float(np.nanmean(recall)) * 100,
        "macro_f1_pooled": float(np.nanmean(f1)) * 100,
        "recall_pooled": [float(r) * 100 for r in recall],
        "n_test_windows": int(len(y_true)),
    }


# --------------------------------------------------------------------------- #
# generic LOSO runner
# --------------------------------------------------------------------------- #
def run_loso(df, feature_cols, subjects, k, calib="mean", exclude_enroll=False,
             test_filter=None, train_filter=None, label_map=None, n_classes=3,
             wrong_state_for_test=False, rounds=200, seed=0, threads=1, want_predictions=False):
    """
    df           : raw table (uncalibrated)
    calib        : None | 'mean' | 'z'
    test_filter  : callable(row_df) -> boolean mask restricting the test windows
    train_filter : callable(row_df) -> boolean mask restricting the training windows
    label_map    : dict original y -> new y (e.g. binary Medium/High), rows not in map dropped
    """
    sensor = [c for c in feature_cols if c in SENSOR_COLS]
    fold_accs, yt_all, yp_all, per_subj, preds = [], [], [], {}, []
    # calibration of the training subjects never depends on the held-out subject,
    # so calibrate the whole table once unless the wrong-state control is requested
    if calib and not wrong_state_for_test:
        dcal_all, enroll_all = calibrate(df, sensor, k, calib)
    for s in subjects:
        if wrong_state_for_test:
            dcal, enroll_rows = calibrate(df, sensor, k, calib, {s: "drowsy"})
        elif calib:
            dcal, enroll_rows = dcal_all, enroll_all
        else:
            dcal, enroll_rows = df, set(enrollment_index(df, s, k).tolist())
        tr = dcal[dcal.subject != s]
        te = dcal[dcal.subject == s]
        if exclude_enroll or wrong_state_for_test:
            te = te[~te.index.isin(enroll_rows)]
        if train_filter is not None:
            tr = tr[train_filter(tr)]
        if test_filter is not None:
            te = te[test_filter(te)]
        ytr, yte = tr["y"].values, te["y"].values
        if label_map is not None:
            mtr = np.isin(ytr, list(label_map)); mte = np.isin(yte, list(label_map))
            tr, te = tr[mtr], te[mte]
            ytr = np.vectorize(label_map.get)(tr["y"].values)
            yte = np.vectorize(label_map.get)(te["y"].values)
        if len(te) == 0 or len(np.unique(ytr)) < 2:
            continue
        yp = fit_predict(tr[feature_cols].values, ytr, te[feature_cols].values, n_classes, rounds, seed, threads)
        acc = float((yp == yte).mean())
        fold_accs.append(acc); per_subj[s] = acc * 100
        yt_all.extend(yte.tolist()); yp_all.extend(yp.tolist())
        if want_predictions:
            preds.append(pd.DataFrame({"subject": te.subject.values, "session": te.session.values,
                                       "t": te.t.values, "y": yte, "pred": yp,
                                       "kss": te["kss"].values if "kss" in te.columns else np.nan}))
    res = summarize(yt_all, yp_all, n_classes, fold_accs)
    res["per_subject_acc"] = per_subj
    if want_predictions:
        res["_predictions"] = pd.concat(preds, ignore_index=True)
    return res


# --------------------------------------------------------------------------- #
# A. class counts and trivial baselines
# --------------------------------------------------------------------------- #
def class_counts(df, subjects):
    ev = df[df.subject.isin(subjects)]
    counts = ev["label"].value_counts().reindex(CLASSES).fillna(0).astype(int)
    per_session = ev.groupby(["session", "label"]).size().unstack(fill_value=0).reindex(columns=CLASSES)
    shares = counts / counts.sum()
    # majority-class baseline: per fold, majority of the training pool applied to the test subject
    fold_accs = []
    for s in subjects:
        maj = df[df.subject != s]["label"].value_counts().idxmax()
        te = ev[ev.subject == s]
        fold_accs.append(float((te["label"] == maj).mean()))
    return {
        "counts_evaluable_subjects": counts.to_dict(),
        "shares": {c: float(shares[c]) for c in CLASSES},
        "counts_per_session": {k: v.to_dict() for k, v in per_session.iterrows()},
        "majority_class": counts.idxmax(),
        "majority_acc_mean_over_folds": float(np.mean(fold_accs)) * 100,
        "majority_acc_pooled": float(shares.max()) * 100,
        "prior_matched_random_acc": float((shares ** 2).sum()) * 100,
        "balanced_chance": 100.0 / 3,
    }


# --------------------------------------------------------------------------- #
# F. warning-level metrics
# --------------------------------------------------------------------------- #
def causal_majority(pred, w=3):
    out = pred.copy()
    for i in range(len(pred)):
        seg = pred[max(0, i - w + 1): i + 1]
        vals, cnt = np.unique(seg, return_counts=True)
        if cnt.max() > 1:
            out[i] = vals[cnt.argmax()]
    return out


def alarm_metrics(preds, step_s=5.0, kss_onsets=(7, 8), detect_within_min=(1, 2, 5)):
    """preds: DataFrame subject, session, t, y, pred, kss (window-level, headline config)."""
    HIGH = CIDX["High"]; LOW = CIDX["Low"]
    fa_low, low_hours = 0, 0.0
    fa_alert_sess, alert_hours = 0, 0.0
    low_windows_alarmed, low_windows = 0, 0
    high_windows_alarmed, high_windows = 0, 0
    latencies = {th: [] for th in kss_onsets}
    missed = {th: 0 for th in kss_onsets}
    sessions_with_onset = {th: 0 for th in kss_onsets}
    early_alarm_sessions = 0
    for (s, sess), g in preds.groupby(["subject", "session"]):
        g = g.sort_values("t")
        sm = causal_majority(g["pred"].values.astype(int))
        alarm = (sm == HIGH).astype(int)
        y = g["y"].values; t = g["t"].values
        edges = np.where(np.diff(np.concatenate([[0], alarm])) == 1)[0]
        # window-level rates
        low_mask = y == LOW; high_mask = y == HIGH
        low_windows += low_mask.sum(); low_windows_alarmed += (alarm[low_mask] == 1).sum()
        high_windows += high_mask.sum(); high_windows_alarmed += (alarm[high_mask] == 1).sum()
        low_hours += low_mask.sum() * step_s / 3600.0
        fa_low += sum(1 for e in edges if y[e] == LOW)
        if sess == "alert":
            alert_hours += len(g) * step_s / 3600.0
            fa_alert_sess += len(edges)
        if sess == "drowsy":
            have_kss = "kss" in g.columns and not g["kss"].isna().all()
            for th in kss_onsets:
                onset_mask = (g["kss"].values >= th) if have_kss else (y == HIGH if th == 7 else np.zeros_like(y, bool))
                if not onset_mask.any():
                    continue
                sessions_with_onset[th] += 1
                onset_t = t[onset_mask][0]
                after = np.where((alarm == 1) & (t >= onset_t))[0]
                if len(after) == 0:
                    missed[th] += 1
                else:
                    latencies[th].append((t[after[0]] - onset_t) / 60.0)
            if len(edges) and (t[edges[0]] < (t[y == HIGH][0] if (y == HIGH).any() else np.inf)):
                early_alarm_sessions += 1
    out = {
        "false_alarm_episodes_per_hour_on_Low_windows": fa_low / low_hours if low_hours else np.nan,
        "false_alarm_episodes_per_alert_session_hour": fa_alert_sess / alert_hours if alert_hours else np.nan,
        "Low_windows_under_alarm_pct": 100.0 * low_windows_alarmed / max(low_windows, 1),
        "High_windows_under_alarm_pct": 100.0 * high_windows_alarmed / max(high_windows, 1),
        "drowsy_sessions_with_alarm_before_first_High": early_alarm_sessions,
    }
    for th in kss_onsets:
        L = np.array(latencies[th]); n = sessions_with_onset[th]
        out[f"onset_kss>={th}"] = {
            "sessions_with_onset": n,
            "median_latency_min": float(np.median(L)) if len(L) else np.nan,
            "missed_sessions": missed[th],
            **{f"detected_within_{m}_min_pct": 100.0 * float((L <= m).sum()) / n if n else np.nan for m in detect_within_min},
        }
    return out


# --------------------------------------------------------------------------- #
# synthetic demo data (smoke test only; NOT real data)
# --------------------------------------------------------------------------- #
def make_demo(path, n_subjects=19, n_alert_only=3, windows=60, step=5.0, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    subs = [chr(ord("A") + i) for i in range(n_subjects)]
    for si, s in enumerate(subs):
        off = rng.normal(0, 1.5, 122)              # per-driver sensor offset
        sessions = ["alert"] if si < n_alert_only else ["alert", "drowsy"]
        for sess in sessions:
            for w in range(windows):
                t = w * step
                frac = w / (windows - 1)
                if sess == "alert":
                    kss = 2 if frac < 0.7 else (3 if rng.random() < 0.6 else 4)
                else:
                    kss = 6 if frac < 0.3 else (7 if frac < 0.6 else (8 if frac < 0.85 else 9))
                drows = (kss - 2) / 7.0
                sensor = off + drows * rng.normal(1.0, 0.3, 122) + rng.normal(0, 1.0, 122)
                drift = frac * np.ones(13) * (1.5 if sess == "drowsy" else 0.3) + rng.normal(0, 0.5, 13)
                rows.append([s, sess, t, kss] + sensor.tolist() + drift.tolist())
    cols = ["subject", "session", "t", "kss"] + [f"f{i:03d}" for i in range(122)] + [f"drift_ctx_{i:02d}" for i in range(13)]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# LaTeX snippet writer
# --------------------------------------------------------------------------- #
def f1(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


def write_snippets(R, path):
    L = []
    A = R["A_class_counts"]
    c = A["counts_evaluable_subjects"]; sh = A["shares"]
    L.append("% ---- Table II extra rows (paste under the R3 uncalibrated row) ----")
    L.append(f"\\rthree & majority class ({A['majority_class']}, training pool) & {f1(A['majority_acc_mean_over_folds'])} & n/a & trivial \\\\")
    L.append(f"\\rthree & prior-matched random guessing & {f1(A['prior_matched_random_acc'])} & n/a & trivial \\\\")
    L.append("% ---- class counts sentence ----")
    L.append(f"% Evaluable-subject windows: Low {c['Low']} ({100*sh['Low']:.1f}\\%), Medium {c['Medium']} ({100*sh['Medium']:.1f}\\%), High {c['High']} ({100*sh['High']:.1f}\\%).")
    L.append("% ---- Controls table rows (tab:controls2) ----")
    B = R["B_time_on_task"]
    for key, name in [("sensor_only", "Sensor features only (122)"), ("sensor_drift", "Sensor + drift context (full model)"),
                      ("sensor_elapsed", "Sensor + elapsed time"), ("sensor_drift_permuted", "Sensor + drift context permuted within session"),
                      ("elapsed_only", "Elapsed time only"), ("drift_only", "Drift context only")]:
        r = B[key]
        L.append(f"{name} & {f1(r['acc_mean_over_folds'])} & {f1(r['balanced_acc_pooled'])} & {f1(r['macro_f1_pooled'])} \\\\")
    C = R["C_within_drowsy_session"]
    L.append(f"Within drowsy session, Medium vs.\\ High, uncalibrated & {f1(C['uncalibrated']['acc_mean_over_folds'])} & {f1(C['uncalibrated']['balanced_acc_pooled'])} & {f1(C['uncalibrated']['macro_f1_pooled'])} \\\\")
    L.append(f"Within drowsy session, Medium vs.\\ High, calibrated & {f1(C['calibrated']['acc_mean_over_folds'])} & {f1(C['calibrated']['balanced_acc_pooled'])} & {f1(C['calibrated']['macro_f1_pooled'])} \\\\")
    L.append(f"% within-session majority baseline (mean over folds): {f1(C['majority_acc_mean_over_folds'])}")
    D = R["D_wrong_state_enrollment"]
    L.append(f"Enrollment from the drowsy session (wrong state) & {f1(D['acc_mean_over_folds'])} & {f1(D['balanced_acc_pooled'])} & {f1(D['macro_f1_pooled'])} \\\\")
    L.append("% ---- Table VIII replacement (single pipeline, headline configuration) ----")
    E = R["E_per_fold_headline"]
    subs = sorted(E["uncalibrated"].keys())
    for s in subs:
        u = E["uncalibrated"][s]; m = E["mean_only_scored"][s]
        L.append(f"{s} & {u:.1f} & {m:.1f} & {m-u:+.1f} \\\\")
    L.append("% ---- Warning-level table rows (tab:alarm) ----")
    F = R["F_warning_level"]
    L.append(f"False-alarm episodes per hour on Low-labelled windows & {f1(F['false_alarm_episodes_per_hour_on_Low_windows'])} \\\\")
    L.append(f"False-alarm episodes per alert-session hour & {f1(F['false_alarm_episodes_per_alert_session_hour'])} \\\\")
    L.append(f"Low windows under alarm (\\%) & {f1(F['Low_windows_under_alarm_pct'])} \\\\")
    L.append(f"High windows under alarm (\\%) & {f1(F['High_windows_under_alarm_pct'])} \\\\")
    for th in (7, 8):
        o = F.get(f"onset_kss>={th}", {})
        if o:
            L.append(f"Median latency to first alarm after KSS $\\ge {th}$ (min) & {f1(o['median_latency_min'])} \\\\")
            L.append(f"Drowsy sessions detected within 5 min of KSS $\\ge {th}$ (\\%) & {f1(o['detected_within_5_min_pct'])} \\\\")
    open(path, "w").write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
SENSOR_COLS = []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", help="V2 window-level feature table (CSV)")
    ap.add_argument("--out", default="revision_results")
    ap.add_argument("--k", type=int, default=96, help="enrollment windows (96 = 8 min at 5-s step)")
    ap.add_argument("--step", type=float, default=5.0, help="window step in seconds")
    ap.add_argument("--drift-regex", default=r"drift|ctx|context")
    ap.add_argument("--rounds", type=int, default=200, help="boosting rounds per class")
    ap.add_argument("--threads", type=int, default=1, help="LightGBM threads (raise on a multi-core machine)")
    ap.add_argument("--within-train", choices=["drowsy", "both"], default="drowsy",
                    help="training windows for the within-drowsy-session control")
    ap.add_argument("--demo", action="store_true", help="generate synthetic data and run (smoke test)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.demo:
        args.table = os.path.join(args.out, "demo_synthetic.csv")
        make_demo(args.table)
        print("[demo] synthetic table written (NOT real data):", args.table)

    global SENSOR_COLS
    df, sensor, drift = load_table(args.table, args.drift_regex)
    SENSOR_COLS = sensor
    subjects = evaluable_subjects(df)
    print(f"windows={len(df)}  subjects={df.subject.nunique()}  evaluable={len(subjects)}  "
          f"sensor features={len(sensor)}  drift features={len(drift)}")
    R = {"config": {"k": args.k, "step_s": args.step, "rounds_per_class": args.rounds,
                    "n_sensor": len(sensor), "n_drift": len(drift), "evaluable_subjects": subjects}}
    kw = dict(k=args.k, rounds=args.rounds, threads=args.threads)

    # A ------------------------------------------------------------------
    R["A_class_counts"] = class_counts(df, subjects)
    print("[A] class shares:", {c: round(v, 3) for c, v in R["A_class_counts"]["shares"].items()},
          "| majority acc (mean over folds):", round(R["A_class_counts"]["majority_acc_mean_over_folds"], 2))

    # B ------------------------------------------------------------------
    dfe = add_elapsed(df)
    dfp = permute_drift_within_session(df, drift)
    B = {}
    B["sensor_only"] = run_loso(df, sensor, subjects, calib="mean", **kw)
    B["sensor_drift"] = run_loso(df, sensor + drift, subjects, calib="mean", **kw)
    B["sensor_elapsed"] = run_loso(dfe, sensor + ["elapsed_s", "elapsed_frac"], subjects, calib="mean", **kw)
    B["sensor_drift_permuted"] = run_loso(dfp, sensor + drift, subjects, calib="mean", **kw)
    B["elapsed_only"] = run_loso(dfe, ["elapsed_s", "elapsed_frac"], subjects, calib=None, **kw)
    B["drift_only"] = run_loso(df, drift, subjects, calib=None, **kw)
    R["B_time_on_task"] = B
    for k_, v in B.items():
        print(f"[B] {k_:24s} acc={v['acc_mean_over_folds']:.2f}  bal={v['balanced_acc_pooled']:.2f}  F1={v['macro_f1_pooled']:.2f}")

    # C ------------------------------------------------------------------
    MH = {CIDX["Medium"]: 0, CIDX["High"]: 1}
    te_f = lambda d: (d.session == "drowsy")
    tr_f = (lambda d: (d.session == "drowsy")) if args.within_train == "drowsy" else None
    C = {}
    C["uncalibrated"] = run_loso(df, sensor + drift, subjects, calib=None, test_filter=te_f, train_filter=tr_f,
                                 label_map=MH, n_classes=2, **kw)
    C["calibrated"] = run_loso(df, sensor + drift, subjects, calib="mean", test_filter=te_f, train_filter=tr_f,
                               label_map=MH, n_classes=2, **kw)
    accs = []
    for s in subjects:
        tr = df[(df.subject != s) & (df.session == "drowsy") & (df.label.isin(["Medium", "High"]))]
        te = df[(df.subject == s) & (df.session == "drowsy") & (df.label.isin(["Medium", "High"]))]
        if len(tr) and len(te):
            accs.append(float((te.label == tr.label.value_counts().idxmax()).mean()))
    C["majority_acc_mean_over_folds"] = float(np.mean(accs)) * 100 if accs else np.nan
    R["C_within_drowsy_session"] = C
    print(f"[C] within-drowsy Medium/High: uncal={C['uncalibrated']['acc_mean_over_folds']:.2f} "
          f"cal={C['calibrated']['acc_mean_over_folds']:.2f} majority={C['majority_acc_mean_over_folds']:.2f}")

    # D ------------------------------------------------------------------
    R["D_wrong_state_enrollment"] = run_loso(df, sensor + drift, subjects, calib="mean", wrong_state_for_test=True, **kw)
    print(f"[D] wrong-state enrollment acc={R['D_wrong_state_enrollment']['acc_mean_over_folds']:.2f}")

    # E ------------------------------------------------------------------
    unc = run_loso(df, sensor + drift, subjects, calib=None, **kw)
    scored = run_loso(df, sensor + drift, subjects, calib="mean", want_predictions=True, **kw)
    excl = run_loso(df, sensor + drift, subjects, calib="mean", exclude_enroll=True, **kw)
    preds = scored.pop("_predictions")
    R["E_per_fold_headline"] = {"uncalibrated": unc["per_subject_acc"], "mean_only_scored": scored["per_subject_acc"],
                                "mean_only_excluded": excl["per_subject_acc"],
                                "summary": {"uncalibrated": {k_: v for k_, v in unc.items() if k_ != "per_subject_acc"},
                                            "mean_only_scored": {k_: v for k_, v in scored.items() if k_ != "per_subject_acc"},
                                            "mean_only_excluded": {k_: v for k_, v in excl.items() if k_ != "per_subject_acc"}}}
    pd.DataFrame({"subject": subjects,
                  "uncalibrated": [unc["per_subject_acc"].get(s, np.nan) for s in subjects],
                  "mean_only_scored": [scored["per_subject_acc"].get(s, np.nan) for s in subjects],
                  "mean_only_excluded": [excl["per_subject_acc"].get(s, np.nan) for s in subjects]}
                 ).to_csv(os.path.join(args.out, "per_subject.csv"), index=False)
    print(f"[E] headline: uncal={unc['acc_mean_over_folds']:.2f}  mean-only scored={scored['acc_mean_over_folds']:.2f}  "
          f"excluded={excl['acc_mean_over_folds']:.2f}")

    # F ------------------------------------------------------------------
    preds.to_csv(os.path.join(args.out, "predictions.csv"), index=False)
    R["F_warning_level"] = alarm_metrics(preds, step_s=args.step)
    print("[F] warning-level:", json.dumps({k_: (round(v, 3) if isinstance(v, float) else v)
                                           for k_, v in R["F_warning_level"].items() if not isinstance(v, dict)}))

    json.dump(R, open(os.path.join(args.out, "results.json"), "w"), indent=2, default=float)
    write_snippets(R, os.path.join(args.out, "controls_snippets.tex"))
    print("written:", os.path.join(args.out, "results.json"), "and controls_snippets.tex")


if __name__ == "__main__":
    main()
