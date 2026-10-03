"""Independent re-verification of every manuscript number.

For each quantity we report STORED (what the results file says) and RECOMPUTED
(what we get by re-deriving it from the underlying per-item arrays in that same
file). Any mismatch is printed as a DISCREPANCY line and counted.

This script recomputes only from data already inside the results files; it never
touches the raw .npz / .h5ad. It therefore runs identically on the Mac clone and
on the A100. Files that are absent on a given machine are reported as MISSING
rather than silently skipped.

Usage:
    python scripts/60_verify_evidence.py
    python scripts/60_verify_evidence.py --json    # machine-readable dump
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

FW = Path(__file__).resolve().parent.parent
CB = FW / "causalbench/results"

# HCP lives outside this repository (separate fMRI pipeline, A100 only).
HCP_DIR = Path("/workspace/meridian-identifiability/hcp/results")

TOL = 1e-6

n_traced = 0
n_recomputed = 0
discrepancies = []
missing = []


def trace(label, value, source, keypath, config=""):
    global n_traced
    n_traced += 1
    cfg = f"  [{config}]" if config else ""
    print(f"  {label:<44} = {value}")
    print(f"      source: {source}")
    print(f"      key   : {keypath}{cfg}")
    return value


def check(label, stored, recomputed, source, tol=TOL):
    """Compare a stored scalar against an independently recomputed one."""
    global n_recomputed
    n_recomputed += 1
    if stored is None or recomputed is None:
        status = "SKIP (one side None)"
        agree = None
    else:
        agree = abs(float(stored) - float(recomputed)) <= tol
        status = "OK" if agree else "*** DISCREPANCY ***"
    print(f"  {label:<44} stored={stored!s:<24} recomputed={recomputed!s:<24} {status}")
    if agree is False:
        discrepancies.append(dict(
            label=label, stored=float(stored), recomputed=float(recomputed),
            delta=float(stored) - float(recomputed), source=str(source)))
    return agree


def load(p):
    p = Path(p)
    if not p.exists():
        missing.append(str(p))
        return None
    with open(p) as f:
        return json.load(f)


def hdr(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ----------------------------------------------------------------- SECTION 1
def section1_screen_primary():
    hdr("SECTION 1 -- screen, primary metric mean_ratio_pairs (filt0, NMIN=200, d=10)")

    print("\n-- CausalBench K562 / RPE1 (03_screen.py output) --")
    for ds in ("k562", "rpe1"):
        p = CB / f"screen/{ds}_filt0_n200_d10.json"
        d = load(p)
        if d is None:
            print(f"  {ds}: MISSING {p}")
            continue
        trace(f"{ds} mean_ratio_pairs", d["mean_ratio_pairs"], p,
              "$.mean_ratio_pairs", "filt0, n=200, d=10, seed=0")
        trace(f"{ds} mean_ratio_vs_ctrl (secondary)", d["mean_ratio_vs_ctrl"], p,
              "$.mean_ratio_vs_ctrl", "filt0, n=200, d=10, seed=0")
        trace(f"{ds} n_envs surviving NMIN", d["n_envs"], p, "$.n_envs",
              "filt0, n=200, d=10")
        trace(f"{ds} n_fit (cells per half)", d["n_fit"], p, "$.n_fit", "")

        s0p = CB / f"screen/{ds}_filt0_n200_d10_STEP0.json"
        s0 = load(s0p)
        if s0 is not None:
            trace(f"{ds} STEP-0 gate mean_ratio_pairs", s0["mean_ratio_pairs"], s0p,
                  "$.mean_ratio_pairs", "step0=True, filt0, n=200, d=10")
            trace(f"{ds} STEP-0 mean_ratio_vs_ctrl (degenerate)",
                  s0["mean_ratio_vs_ctrl"], s0p, "$.mean_ratio_vs_ctrl",
                  "step0=True -- ref_pool empty, see 03_screen.py:63")

    print("\n-- Norman 2019 (40_screen_norman.py output) --")
    p = CB / "screen/norman.json"
    d = load(p)
    if d is not None:
        prim = next((r for r in d["screen"]
                     if r.get("nmin") == 200 and r.get("d") == 10), None)
        if prim:
            trace("norman mean_ratio_pairs", prim["mean_ratio_pairs"], p,
                  "$.screen[nmin=200,d=10].mean_ratio_pairs",
                  "filt n/a, n=200, d=10, seed=0")
            trace("norman mean_ratio_vs_ctrl (secondary)",
                  prim["mean_ratio_vs_ctrl"], p,
                  "$.screen[nmin=200,d=10].mean_ratio_vs_ctrl", "")
            trace("norman n_envs surviving NMIN", prim["n_envs"], p,
                  "$.screen[nmin=200,d=10].n_envs", "")
        g0 = next((r for r in d["step0_gate"]
                   if r.get("nmin") == 200 and r.get("d") == 10), None)
        if g0:
            trace("norman STEP-0 gate mean_ratio_pairs", g0["mean_ratio_pairs"], p,
                  "$.step0_gate[nmin=200,d=10].mean_ratio_pairs", "")
            trace("norman STEP-0 mean_ratio_vs_ctrl (degenerate)",
                  g0["mean_ratio_vs_ctrl"], p,
                  "$.step0_gate[nmin=200,d=10].mean_ratio_vs_ctrl", "")
        # internal consistency of the summary block
        summ = d.get("summary", {})
        if prim:
            check("norman summary.primary_mean_ratio_vs_ctrl",
                  summ.get("primary_mean_ratio_vs_ctrl"),
                  prim["mean_ratio_vs_ctrl"], p)
            check("norman summary.primary_mean_ratio_pairs",
                  summ.get("primary_mean_ratio_pairs"),
                  prim["mean_ratio_pairs"], p)
        gate_pairs = [r["mean_ratio_pairs"] for r in d["step0_gate"]
                      if "mean_ratio_pairs" in r]
        stored_pass = summ.get("step0_pass")
        recomputed_pass = all(0.9 <= x <= 1.1 for x in gate_pairs)
        print(f"  {'norman summary.step0_pass':<44} "
              f"stored={stored_pass!s:<24} recomputed={recomputed_pass!s:<24} "
              f"{'OK' if stored_pass == recomputed_pass else '*** DISCREPANCY ***'}")
        global n_recomputed
        n_recomputed += 1
        if stored_pass != recomputed_pass:
            discrepancies.append(dict(
                label="norman summary.step0_pass", stored=stored_pass,
                recomputed=recomputed_pass, delta=None, source=str(p)))

    print("\n-- HCP (hcp/scripts/mean_shift_v2.py -- DIFFERENT CONSTRUCTION) --")
    for tag, fn in (("LR", "mean_shift_LR.json"), ("RL", "mean_shift_RL.json")):
        p = HCP_DIR / fn
        d = load(p)
        if d is None:
            print(f"  HCP {tag}: MISSING {p}")
            continue
        r = d["results"].get("10")
        if r:
            trace(f"HCP {tag} mean_ratio (d=10)", r["mean_ratio"], p,
                  '$.results["10"].mean_ratio', f"enc={tag}, d=10")
            trace(f"HCP {tag} within_median", r["within_median"], p,
                  '$.results["10"].within_median', "")
            trace(f"HCP {tag} between_median", r["between_median"], p,
                  '$.results["10"].between_median', "")
            rec = r["between_median"] / r["within_median"]
            check(f"HCP {tag} mean_ratio = between/within", r["mean_ratio"], rec, p,
                  tol=1e-9)
    print("\n  NOTE: HCP has NO mean_ratio_pairs and NO step-0 gate field.")
    print("  See DISCREPANCIES section on construction comparability.")


# ----------------------------------------------------------------- SECTION 2
def section2_screen_sensitivity():
    hdr("SECTION 2 -- screen sensitivity across configurations (mean_ratio_pairs)")
    print(f"  {'dataset':<8}{'filt':<6}{'nmin':<7}{'d':<5}"
          f"{'pairs':>10}{'vs_ctrl':>10}{'n_envs':>9}")
    rows = []
    for ds in ("k562", "rpe1"):
        for filt in (0, 1):
            for nmin in (200, 100):
                for dd in (5, 10):
                    p = CB / f"screen/{ds}_filt{filt}_n{nmin}_d{dd}.json"
                    j = load(p)
                    if j is None:
                        print(f"  {ds:<8}{filt:<6}{nmin:<7}{dd:<5}"
                              f"{'MISSING':>10}")
                        continue
                    global n_traced
                    n_traced += 2
                    print(f"  {ds:<8}{filt:<6}{nmin:<7}{dd:<5}"
                          f"{j['mean_ratio_pairs']:>10.3f}"
                          f"{j['mean_ratio_vs_ctrl']:>10.3f}"
                          f"{j['n_envs']:>9}")
                    rows.append((ds, filt, nmin, dd,
                                 j["mean_ratio_pairs"], j["mean_ratio_vs_ctrl"]))
    if rows:
        for ds in ("k562", "rpe1"):
            vals = [r[4] for r in rows if r[0] == ds]
            if vals:
                print(f"\n  {ds} mean_ratio_pairs range across configs: "
                      f"[{min(vals):.3f}, {max(vals):.3f}]  "
                      f"spread={max(vals)-min(vals):.3f}")


# ----------------------------------------------------------------- SECTION 3
def section3_effective_dim():
    hdr("SECTION 3 -- effective dimensionality (dims above 2x noise, participation)")

    def verify_spectrum(label, j, source, keypath):
        if j is None:
            return
        s_sig = np.asarray(j["sing_signal"], dtype=float)
        s_noi = np.asarray(j["sing_noise"], dtype=float)
        ratio = np.asarray(j["sing_ratio"], dtype=float)
        n_cmp = min(len(s_sig), len(s_noi))
        rec_ratio = s_sig[:n_cmp] / s_noi[:n_cmp]
        trace(f"{label} n_dims_above_2x", j["n_dims_above_2x"], source,
              f"{keypath}.n_dims_above_2x", "")
        trace(f"{label} n_dims_above_noise", j["n_dims_above_noise"], source,
              f"{keypath}.n_dims_above_noise", "")
        trace(f"{label} participation_ratio", round(j["participation_ratio"], 4),
              source, f"{keypath}.participation_ratio", "")
        check(f"{label} sing_ratio[0] recomputed", ratio[0], rec_ratio[0], source,
              tol=1e-9)
        check(f"{label} n_dims_above_2x recomputed", j["n_dims_above_2x"],
              int((rec_ratio > 2.0).sum()), source)
        check(f"{label} n_dims_above_noise recomputed", j["n_dims_above_noise"],
              int((rec_ratio > 1.0).sum()), source)
        rec_pr = float(s_sig.sum() ** 2 / (s_sig ** 2).sum())
        check(f"{label} participation_ratio recomputed",
              j["participation_ratio"], rec_pr, source, tol=1e-6)
        check(f"{label} s2_over_s1 recomputed", j["s2_over_s1"],
              float(s_sig[1] / s_sig[0]), source, tol=1e-9)
        print(f"      len(sing_signal)={len(s_sig)}  len(sing_noise)={len(s_noi)}"
              f"  d={j.get('d')}  n_envs={j.get('n_envs')}  n_null={j.get('n_null')}")

    for ds in ("k562", "rpe1"):
        for dd in (10, 20, 50):
            p = CB / f"spectrum/{ds}_filt0_n200_d{dd}.json"
            j = load(p)
            if j is None:
                print(f"  {ds} d={dd}: MISSING {p}")
                continue
            print()
            verify_spectrum(f"{ds} d={dd}", j, p, "$")

    p = CB / "spectrum/norman.json"
    j = load(p)
    if j is not None:
        for r in j["runs"]:
            print()
            verify_spectrum(f"norman d={r['d']}", r, p, f"$.runs[d={r['d']}]")

    print("\n-- HCP spectrum (from mean_shift_LR/RL.json, different pipeline) --")
    for tag, fn in (("LR", "mean_shift_LR.json"), ("RL", "mean_shift_RL.json")):
        p = HCP_DIR / fn
        d = load(p)
        if d is None:
            print(f"  HCP {tag}: MISSING {p}")
            continue
        for dd in sorted(d["results"], key=int):
            r = d["results"][dd]
            sr = np.asarray(r["sing_ratio"], dtype=float)
            trace(f"HCP {tag} d={dd} n_dims_above_2x", r["n_dims_above_2x"], p,
                  f'$.results["{dd}"].n_dims_above_2x', f"enc={tag}")
            check(f"HCP {tag} d={dd} n_dims_above_2x recomputed",
                  r["n_dims_above_2x"], int((sr > 2.0).sum()), p)
            trace(f"HCP {tag} d={dd} participation_ratio",
                  round(r["participation_ratio"], 4), p,
                  f'$.results["{dd}"].participation_ratio', "")
            print(f"      max_possible_dims={r.get('max_possible_dims')} "
                  f"len(sing_ratio)={len(sr)}")


# ----------------------------------------------------------------- SECTION 4
def section4_subspace():
    hdr("SECTION 4 -- intervention subspace vs control PCA (held-out R2)")
    p = FW / "results/cf_estimator/k562.json"
    d = load(p)
    if d is None:
        print(f"  MISSING {p}")
        return
    trace("n_envs", d["n_envs"], p, "$.n_envs", f"nmin={d['nmin']}")
    trace("n_stable", d["n_stable"], p, "$.n_stable", "")
    trace("gate_cos_median (control-cell gate)", d["gate_cos_median"], p,
          "$.gate_cos_median", "expect ~0 -- control halves share no direction")
    trace("gate_cos_mean", d["gate_cos_mean"], p, "$.gate_cos_mean", "")
    trace("stab_cos_median (split-half stability)", d["stab_cos_median"], p,
          "$.stab_cos_median", "")
    trace("stab_cos_q10", d["stab_cos_q10"], p, "$.stab_cos_q10", "")
    trace("stab_cos_q90", d["stab_cos_q90"], p, "$.stab_cos_q90", "")
    trace("frac_cos_gt_0p3", d["frac_cos_gt_0p3"], p, "$.frac_cos_gt_0p3", "")
    trace("rank1_median", d["rank1_median"], p, "$.rank1_median", "")
    trace("shift_norm_median", d["shift_norm_median"], p, "$.shift_norm_median", "")
    print()
    print(f"  {'d':<5}{'shift_basis_r2':>16}{'control_pca_r2':>16}"
          f"{'gain(stored)':>14}{'gain(recomp)':>14}")
    for k in sorted(d["basis"], key=int):
        v = d["basis"][k]
        rec = v["shift_basis_r2"] - v["control_pca_r2"]
        print(f"  {k:<5}{v['shift_basis_r2']:>16.4f}{v['control_pca_r2']:>16.4f}"
              f"{v['gain']:>14.4f}{rec:>14.4f}")
        global n_traced
        n_traced += 3
        check(f"cf_estimator d={k} gain", v["gain"], rec, p, tol=1e-9)


# ----------------------------------------------------------------- SECTION 5
def section5_baselines():
    hdr("SECTION 5 -- zero-shot baselines on the canonical split")
    p = FW / "results/zeroshot_canonical/k562.json"
    d = load(p)
    if d is None:
        print(f"  MISSING {p}")
        return
    trace("split_file", d.get("split_file"), p, "$.split_file", "")
    trace("n_train", d.get("n_train"), p, "$.n_train", "")
    trace("n_heldout", d.get("n_heldout"), p, "$.n_heldout", "")
    trace("d_latent", d.get("d_latent"), p, "$.d_latent", "")

    sp = FW / "results/splits/k562_zeroshot_split.json"
    s = load(sp)
    if s is not None:
        check("n_train matches split file", d.get("n_train"),
              s.get("n_train"), sp, tol=0)
        check("n_heldout matches split file", d.get("n_heldout"),
              s.get("n_heldout"), sp, tol=0)
        check("len(heldout_perturbations)", s.get("n_heldout"),
              len(s["heldout_perturbations"]), sp, tol=0)
        check("len(train_perturbations)", s.get("n_train"),
              len(s["train_perturbations"]), sp, tol=0)
        check("n_usable = n_train + n_heldout", s.get("n_usable"),
              s.get("n_train") + s.get("n_heldout"), sp, tol=0)
        trace("split seed", s.get("seed"), sp, "$.seed", "")
        trace("split nmin", s.get("nmin"), sp, "$.nmin", "")

    rows = d.get("per_perturbation", [])
    if not rows:
        print("  per_perturbation array absent -- cannot recompute")
        return
    print(f"\n  recomputing from $.per_perturbation ({len(rows)} rows)")

    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    gm_by_gene = {r["gene"]: r["r2"] for r in by_method.get("global_mean", [])}

    print(f"\n  {'method':<14}{'r2_med(st)':>12}{'r2_med(re)':>12}"
          f"{'r2_mean(st)':>13}{'r2_mean(re)':>13}"
          f"{'fbg(st)':>10}{'fbg(re)':>10}{'n':>5}")
    for m in ("zero", "global_mean", "corr_prop", "nn_corr",
              "ridge_basis", "CEILING"):
        st = d["results"].get(m)
        rr = by_method.get(m, [])
        if st is None or not rr:
            print(f"  {m:<14} MISSING")
            continue
        vals = np.asarray([x["r2"] for x in rr], dtype=float)
        cosv = np.asarray([x["cos"] for x in rr], dtype=float)
        rec_med = float(np.median(vals))
        rec_mean = float(np.mean(vals))
        rec_cos = float(np.median(cosv))
        rec_fbg = float(np.mean([x["r2"] > gm_by_gene[x["gene"]] for x in rr]))
        print(f"  {m:<14}{st['r2_median']:>12.4f}{rec_med:>12.4f}"
              f"{st['r2_mean']:>13.4f}{rec_mean:>13.4f}"
              f"{st['frac_beats_gmean']:>10.3f}{rec_fbg:>10.3f}{len(rr):>5}")
        global n_traced
        n_traced += 4
        check(f"{m} r2_median", st["r2_median"], rec_med, p)
        check(f"{m} r2_mean", st["r2_mean"], rec_mean, p)
        check(f"{m} cos_median", st["cos_median"], rec_cos, p)
        check(f"{m} frac_beats_gmean", st["frac_beats_gmean"], rec_fbg, p)

    hd = d["results"].get("_headroom")
    rec_hd = (d["results"]["CEILING"]["r2_median"]
              - d["results"]["global_mean"]["r2_median"])
    check("_headroom = CEILING - global_mean", hd, rec_hd, p)


# ----------------------------------------------------------------- SECTION 6
def section6_model():
    hdr("SECTION 6 -- model results, four arms")
    arms = {
        "full_shift":   FW / "results/model/zeroshot_shift.json",
        "full_random":  FW / "results/model/zeroshot_random.json",
        "nodag_shift":  FW / "results/model/zeroshot_nodag_shift.json",
        "nodag_random": FW / "results/model/zeroshot_nodag_random.json",
    }
    print(f"  {'arm':<14}{'best(st)':>11}{'final_med(st)':>15}{'final_med(re)':>15}"
          f"{'final_mn(st)':>14}{'final_mn(re)':>14}{'n':>5}")
    per_arm = {}
    for arm, p in arms.items():
        d = load(p)
        if d is None:
            print(f"  {arm:<14} MISSING {p}")
            continue
        per_arm[arm] = (d, p)
        rows = d.get("per_gene", [])
        if rows:
            vals = np.asarray([x["r2"] for x in rows], dtype=float)
            rec_med, rec_mean = float(np.median(vals)), float(np.mean(vals))
        else:
            rec_med = rec_mean = None
        print(f"  {arm:<14}{d['best_median_r2']:>11.4f}"
              f"{d['final_median_r2']:>15.4f}"
              f"{(rec_med if rec_med is not None else float('nan')):>15.4f}"
              f"{d['final_mean_r2']:>14.4f}"
              f"{(rec_mean if rec_mean is not None else float('nan')):>14.4f}"
              f"{len(rows):>5}")
        global n_traced
        n_traced += 5
        if rec_med is not None:
            check(f"{arm} final_median_r2", d["final_median_r2"], rec_med, p)
            check(f"{arm} final_mean_r2", d["final_mean_r2"], rec_mean, p)
            # frac_beats_* in cb_train.py compare against the CONSTANTS 0.129 / 0.226
            rec_fbg = float(np.mean(vals > 0.129))
            rec_fbr = float(np.mean(vals > 0.226))
            check(f"{arm} frac_beats_gmean (const 0.129)",
                  d["frac_beats_gmean"], rec_fbg, p)
            check(f"{arm} frac_beats_ridge (const 0.226)",
                  d["frac_beats_ridge"], rec_fbr, p)
            print(f"      per_gene r2 range [{vals.min():.4f}, {vals.max():.4f}] "
                  f"n>0.129: {int((vals>0.129).sum())}  "
                  f"n>0.226: {int((vals>0.226).sum())}")

    print("\n-- SUMMARY.json cross-check against the four source files --")
    sp = FW / "results/model/SUMMARY.json"
    S = load(sp)
    if S is not None:
        for arm, (d, p) in per_arm.items():
            m = S["models"].get(arm)
            if m is None:
                print(f"  {arm}: absent from SUMMARY.json")
                continue
            for k in ("best_median_r2", "final_median_r2", "final_mean_r2",
                      "frac_beats_gmean", "frac_beats_ridge"):
                check(f"SUMMARY {arm}.{k}", m.get(k), d.get(k), sp)
        zc = load(FW / "results/zeroshot_canonical/k562.json")
        if zc is not None:
            for m in ("zero", "global_mean", "corr_prop", "nn_corr",
                      "ridge_basis", "CEILING"):
                check(f"SUMMARY baselines.{m}.r2_median",
                      S["baselines"][m]["r2_median"],
                      zc["results"][m]["r2_median"], sp)
            check("SUMMARY headroom", S.get("headroom"),
                  zc["results"].get("_headroom"), sp)
            g = zc["results"]["global_mean"]["r2_median"]
            r = zc["results"]["ridge_basis"]["r2_median"]
            h = zc["results"]["_headroom"]
            check("SUMMARY ridge_capture_fraction",
                  S.get("ridge_capture_fraction"), (r - g) / h, sp)

    print("\n-- DAG condition number (results/model/dag_stability.csv) --")
    cp = FW / "results/model/dag_stability.csv"
    if not cp.exists():
        missing.append(str(cp))
        print(f"  MISSING {cp}")
    else:
        with open(cp) as f:
            rd = list(csv.DictReader(f))
        cond = np.asarray([float(r["cond2"]) for r in rd if r["cond2"]], dtype=float)
        inv = np.asarray([float(r["inv_norm_2"]) for r in rd if r["inv_norm_2"]],
                          dtype=float)
        gfr = np.asarray([float(r["G_frobenius"]) for r in rd if r["G_frobenius"]],
                          dtype=float)
        eps = sorted({int(r["epoch"]) for r in rd})
        trace("dag_stability n_rows", len(rd), cp, "csv rows", "")
        trace("cond2 min", round(float(cond.min()), 6), cp, "col cond2", "")
        trace("cond2 max", round(float(cond.max()), 6), cp, "col cond2", "")
        trace("cond2 final", round(float(cond[-1]), 6), cp, "col cond2 last row", "")
        trace("inv_norm_2 min", round(float(inv.min()), 6), cp, "col inv_norm_2", "")
        trace("inv_norm_2 max", round(float(inv.max()), 6), cp, "col inv_norm_2", "")
        trace("G_frobenius min", round(float(gfr.min()), 6), cp, "col G_frobenius", "")
        trace("G_frobenius max", round(float(gfr.max()), 6), cp, "col G_frobenius", "")
        trace("epochs covered", f"{min(eps)}..{max(eps)}", cp, "col epoch", "")
        print(f"      NaNs: cond2={int(np.isnan(cond).sum())} "
              f"inv={int(np.isnan(inv).sum())} G={int(np.isnan(gfr).sum())}")

    print("\n-- epoch-10 value per arm (needs results/model/trajectories/) --")
    for arm in arms:
        tp = FW / f"results/model/trajectories/{arm}.json"
        t = load(tp)
        if t is None:
            print(f"  {arm}: MISSING {tp}  (parse from logs/ on the A100)")
            continue
        pts = {p["epoch"]: p for p in t["points"]}
        e10 = pts.get(10)
        if e10:
            trace(f"{arm} epoch-10 r2_median", e10["r2_median"], tp,
                  "$.points[epoch=10].r2_median", "")
        vals = [p["r2_median"] for p in t["points"]]
        trace(f"{arm} trajectory min", min(vals), tp, "$.points[*].r2_median", "")
        d, p = per_arm.get(arm, (None, None))
        if d is not None:
            check(f"{arm} best_median_r2 = max(trajectory)",
                  d["best_median_r2"], max(vals), tp, tol=1e-4)


# ----------------------------------------------------------------- SECTION 7
def section7_descriptors():
    hdr("SECTION 7 -- dataset descriptors")
    for ds in ("k562", "rpe1"):
        p = FW / f"results/splits/{ds}_zeroshot_split.json"
        s = load(p)
        if s is None:
            print(f"  {ds}: MISSING {p}")
            continue
        cpp = s.get("cells_per_perturbation", {})
        vals = np.asarray(list(cpp.values()), dtype=float) if cpp else None
        trace(f"{ds} n_usable environments", s.get("n_usable"), p, "$.n_usable",
              f"nmin={s.get('nmin')}, seed={s.get('seed')}")
        trace(f"{ds} n_train", s.get("n_train"), p, "$.n_train", "")
        trace(f"{ds} n_heldout", s.get("n_heldout"), p, "$.n_heldout", "")
        if vals is not None and vals.size:
            trace(f"{ds} cells/env median", float(np.median(vals)), p,
                  "$.cells_per_perturbation (recomputed median)", "")
            trace(f"{ds} cells/env min", float(vals.min()), p,
                  "$.cells_per_perturbation", "")
            trace(f"{ds} cells/env max", float(vals.max()), p,
                  "$.cells_per_perturbation", "")
            trace(f"{ds} n entries in cells_per_perturbation", int(vals.size), p,
                  "len($.cells_per_perturbation)", "")
    print("\n  Cells / features / control cells / batches are NOT in any results")
    print("  file -- they require reading the raw .npz / .h5ad on the A100.")
    print("  See the A100 collection script (scripts/61_collect_descriptors.py).")


# ----------------------------------------------------------------- SECTION 8
def section8_provenance():
    hdr("SECTION 8 -- third-party provenance")
    for label, p in (("discrepancy_vae", FW / "vendor_analysis/DVAE_COMMIT.txt"),
                      ("causalbench_repo", FW / "vendor_analysis/CAUSALBENCH_COMMIT.txt"),
                      ("COMMITS.txt (public mirror)", FW / "COMMITS.txt")):
        if p.exists():
            txt = p.read_text().strip()
            trace(label, txt.replace("\n", " | "), p, "file contents", "")
        else:
            missing.append(str(p))
            print(f"  {label}: MISSING {p}")
    pp = FW / "vendor_analysis/my_changes.patch"
    if pp.exists():
        txt = pp.read_text()
        n_hunks = txt.count("@@ ")
        files = [l.split()[-1] for l in txt.splitlines()
                 if l.startswith("+++ b/")]
        trace("my_changes.patch hunks", n_hunks, pp, "count of '@@ '", "")
        trace("my_changes.patch files touched", ", ".join(files) or "(none)", pp,
              "'+++ b/' lines", "")
    print("\n  Model hyperparameters come from model/cb_train.py argparse defaults;")
    print("  the loss itself is discrepancy_vae's loss_function, imported unmodified.")
    tp = FW / "model/cb_train.py"
    if tp.exists():
        src = tp.read_text()
        import re
        for m in re.finditer(r'p\.add_argument\("--(\w+)".*?default=([^,)]+)', src):
            print(f"      --{m.group(1):<12} default={m.group(2).strip()}")
        print(f"      imports loss_function: "
              f"{'from train import loss_function' in src}")
        print(f"      redefines loss_function: "
              f"{'def loss_function' in src}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true",
                    help="dump machine-readable summary at the end")
    a = ap.parse_args()

    print("EVIDENCE VERIFICATION -- stored vs recomputed")
    print(f"repo root: {FW}")
    print(f"hcp dir  : {HCP_DIR}  (exists={HCP_DIR.exists()})")

    section1_screen_primary()
    section2_screen_sensitivity()
    section3_effective_dim()
    section4_subspace()
    section5_baselines()
    section6_model()
    section7_descriptors()
    section8_provenance()

    hdr("VERIFICATION TOTALS")
    print(f"  numbers traced      : {n_traced}")
    print(f"  numbers recomputed  : {n_recomputed}")
    print(f"  discrepancies found : {len(discrepancies)}")
    print(f"  files missing       : {len(missing)}")
    if discrepancies:
        print("\n  *** DISCREPANCIES ***")
        for x in discrepancies:
            print(f"    {x['label']}")
            print(f"      stored={x['stored']}  recomputed={x['recomputed']}  "
                  f"delta={x['delta']}")
            print(f"      {x['source']}")
    if missing:
        print("\n  MISSING FILES (not present on this machine):")
        for x in missing:
            print(f"    {x}")

    if a.json:
        print("\n---JSON---")
        print(json.dumps(dict(n_traced=n_traced, n_recomputed=n_recomputed,
                              discrepancies=discrepancies, missing=missing),
                          indent=2, default=str))
    return 1 if discrepancies else 0


if __name__ == "__main__":
    sys.exit(main())
