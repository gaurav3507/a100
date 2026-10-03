"""RLDD calibration replication (binary alert vs drowsy, LOSO over subjects).

Tests whether the per-driver mean-only calibration that lifted UL-DD LOSO
accuracy also helps on a third dataset: 60 subjects of real self-recorded
webcam video, 24 facial features, 30-second non-overlapping windows.

Design mirrors the UL-DD protocol:
  - baseline from each subject's own ALERT windows (label 0), features only
  - mean-only (offset removal) vs z-score, and an enrollment-length sweep
  - leave-one-subject-out; subjects lacking either class are excluded
  - paired Wilcoxon + bootstrap CI + Cohen's d + Cliff's delta over folds

Reads features_v2/rldd_newfolds.h5 (keys: f4, subjects, labels_3class,
labels_binary, window_idx, feature_names).

Usage:
    python rldd_calibration_v1.py                       # full study
    python rldd_calibration_v1.py --quick               # k=10 only, fast check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import stats

ALERT, DROWSY = 0, 2


def load(path: Path):
    with h5py.File(path, "r") as h:
        X = h["f4"][:].astype(np.float64)
        y3 = h["labels_3class"][:]
        subj = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in h["subjects"][:]])
        widx = h["window_idx"][:]
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in h["feature_names"][:]]
    y = (y3 == DROWSY).astype(int)                      # binary: alert=0, drowsy=1
    return X, y, y3, subj, widx, names


def calibrate(X, y3, subj, mode: str, k: int):
    """Per-subject baseline from that subject's ALERT windows (lowest window_idx first).
    mode: none | mean | zscore ; k = number of alert windows (0 = all alert windows).
    Returns (calibrated X, enrollment_mask) -- the mask marks the windows that FORMED the
    baseline; scoring them inflates accuracy because calibration drives them toward zero."""
    enroll = np.zeros(len(X), dtype=bool)
    if mode == "none":
        return X.copy(), enroll
    out = X.copy()
    for s in np.unique(subj):
        m = subj == s
        alert = m & (y3 == ALERT)
        if alert.sum() < 2:
            continue                                     # cannot enroll; left uncalibrated (subject is excluded anyway)
        idx = np.where(alert)[0]
        if k:
            idx = idx[:k]                                # first k alert windows = enrollment
        enroll[idx] = True
        mu = X[idx].mean(axis=0)
        out[m] = X[m] - mu
        if mode == "zscore":
            sd = X[idx].std(axis=0, ddof=0)
            sd[sd == 0] = 1.0
            out[m] = out[m] / sd
    return out, enroll


def loso(X, y, subj, eval_subjects, seed=0, enroll_mask=None, exclude_enrollment=False):
    """Per-fold accuracy/F1/per-class recall. With exclude_enrollment, the held-out subject's
    enrollment windows are dropped from SCORING (they still form that subject's baseline);
    training folds are unaffected."""
    import lightgbm as lgb
    from sklearn.metrics import f1_score, recall_score
    folds = []
    for s in eval_subjects:
        te = subj == s
        tr = ~te
        if exclude_enrollment and enroll_mask is not None:
            te = te & ~enroll_mask
            if te.sum() == 0 or len(np.unique(y[te])) < 2:
                continue                                  # nothing left to score for this subject
        m = lgb.LGBMClassifier(n_estimators=200, verbosity=-1, random_state=seed)
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        folds.append(dict(subject=s, n=int(te.sum()),
                          accuracy=float((p == y[te]).mean()),
                          f1_macro=float(f1_score(y[te], p, average="macro", zero_division=0)),
                          recall_alert=float(recall_score(y[te], p, pos_label=0, zero_division=0)),
                          recall_drowsy=float(recall_score(y[te], p, pos_label=1, zero_division=0))))
    return folds


def paired(a_folds, b_folds, key="accuracy"):
    da = {f["subject"]: f[key] for f in a_folds}; db = {f["subject"]: f[key] for f in b_folds}
    common = sorted(set(da) & set(db))
    if len(common) < 5:
        return dict(mean_a=float("nan"), mean_b=float("nan"), delta=float("nan"), ci=(float("nan"), float("nan")),
                    wilcoxon_p=float("nan"), d_z=float("nan"), cliff=float("nan"), wins=0, losses=0, n=len(common))
    a = np.array([da[s] for s in common]); b = np.array([db[s] for s in common])
    d = a - b
    nz = d[d != 0]
    w_p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 5 else float("nan")
    rng = np.random.default_rng(0)
    boot = rng.choice(d, size=(20000, len(d)), replace=True).mean(axis=1)
    gt = sum((x > y) for x in a for y in b); lt = sum((x < y) for x in a for y in b)
    return dict(mean_a=a.mean(), mean_b=b.mean(), delta=d.mean(),
                ci=(float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
                wilcoxon_p=w_p, d_z=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan"),
                cliff=float((gt - lt) / (len(a) * len(b))), wins=int((d > 0).sum()), losses=int((d < 0).sum()), n=int(len(common)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, default=Path("features_v2/rldd_newfolds.h5"))
    ap.add_argument("--out", type=Path, default=Path("features_v2/rldd_calibration_results.json"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--exclude-enrollment", action="store_true",
                    help="drop each held-out subject's enrollment windows from scoring (deployment-faithful)")
    ap.add_argument("--both", action="store_true",
                    help="report every configuration twice: enrollment scored and enrollment excluded")
    args = ap.parse_args()

    X, y, y3, subj, widx, names = load(args.h5)
    subs = np.unique(subj)
    # a subject is evaluable only if it has BOTH classes and >=2 alert windows to enroll from
    evaluable = np.array([s for s in subs
                          if len(np.unique(y[subj == s])) == 2 and ((subj == s) & (y3 == ALERT)).sum() >= 2])
    print(f"RLDD: {X.shape[0]} windows, {len(subs)} subjects, {X.shape[1]} features "
          f"({(y == 0).sum()} alert / {(y == 1).sum()} drowsy)")
    print(f"evaluable subjects (both classes + enrollable): {len(evaluable)}")
    dropped = sorted(set(subs) - set(evaluable))
    if dropped:
        print(f"excluded: {dropped}")

    configs = [("none", 0)] if args.quick else [("none", 0)]
    ks = [10] if args.quick else [5, 10, 15, 20, 0]      # 0 = all alert windows
    configs += [("mean", k) for k in ks] + [("zscore", k) for k in ([10] if args.quick else [10, 0])]

    variants = [(False, ""), (True, "_excl")] if args.both else [(args.exclude_enrollment, "_excl" if args.exclude_enrollment else "")]
    results, folds_by_cfg = {}, {}
    for mode, k in configs:
        Xc, enroll = calibrate(X, y3, subj, mode, k)
      
        for excl, suffix in variants:
            if excl and mode != "none":
                # exclusion is only meaningful if most subjects retain scorable alert windows
                remain = [((subj == s) & (y3 == ALERT) & ~enroll).sum() for s in evaluable]
                if np.mean(np.array(remain) >= 3) < 0.9:
                    print(f"  {mode}{('_k'+str(k)) if k else '_all'}_excl   SKIPPED: enrollment consumes the alert "
                          f"class for {(np.array(remain) < 3).sum()}/{len(evaluable)} subjects")
                    continue
            f = loso(Xc, y, subj, evaluable, enroll_mask=enroll, exclude_enrollment=excl)
            tag = f"{mode}" + (f"_k{k}" if k else ("_all" if mode != "none" else "")) + suffix
            folds_by_cfg[tag] = f
            acc = np.array([x["accuracy"] for x in f]); f1 = np.array([x["f1_macro"] for x in f])
            ra = np.array([x["recall_alert"] for x in f]); rd = np.array([x["recall_drowsy"] for x in f])
            results[tag] = dict(accuracy=float(acc.mean()), accuracy_sd=float(acc.std()),
                                f1_macro=float(f1.mean()), recall_alert=float(ra.mean()),
                                recall_drowsy=float(rd.mean()), n_folds=len(f),
                                enrollment_scored=(not excl))
            print(f"  {tag:<18} acc={acc.mean()*100:6.2f}+/-{acc.std()*100:4.1f}  f1={f1.mean()*100:6.2f}  "
                  f"recall alert={ra.mean()*100:5.1f} drowsy={rd.mean()*100:5.1f}  folds={len(f)}")

    print("\nPAIRED COMPARISONS (accuracy, percentage points)")
    print(f"{'comparison':<34}{'ref':>7}{'other':>7}{'delta':>8}{'95% CI':>18}{'p':>9}{'W/L':>8}{'d_z':>7}{'Cliff':>7}{'n':>5}")
    print("-" * 105)
    pref = [t for t in ("mean_k10_excl", "mean_k10") if t in folds_by_cfg]
    ref_tag = pref[0] if pref else list(folds_by_cfg)[1]
    stats_out = {}
    for tag in folds_by_cfg:
        if tag == ref_tag:
            continue
        r = paired(folds_by_cfg[ref_tag], folds_by_cfg[tag])
        stats_out[f"{ref_tag}_vs_{tag}"] = r
        star = " *" if (not np.isnan(r["wilcoxon_p"]) and r["wilcoxon_p"] < 0.05) else ""
        print(f"{ref_tag+' vs '+tag:<34}{r['mean_a']*100:>7.2f}{r['mean_b']*100:>7.2f}{r['delta']*100:>+8.2f}"
              f"{'[%+.1f, %+.1f]' % (r['ci'][0]*100, r['ci'][1]*100):>18}{r['wilcoxon_p']:>9.4f}"
              f"{r['wins']:>5}/{r['losses']}{r['d_z']:>7.2f}{r['cliff']:>7.2f}{r.get('n',0):>5}{star}")

    args.out.write_text(json.dumps(dict(summary=results, paired=stats_out,
                                        per_fold=folds_by_cfg, n_evaluable=len(evaluable),
                                        excluded=dropped, features=names), indent=2, default=float))
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
