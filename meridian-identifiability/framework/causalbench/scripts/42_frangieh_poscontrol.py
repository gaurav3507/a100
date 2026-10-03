"""Positive control for the Frangieh environment-validity screen.

CONTROL ONLY -- this script deliberately inflates the between-env contrast
by using the experimental ARM (Co-culture / Control / IFNγ) as the
environment label instead of the perturbation target. Arms have well-known
large transcriptional differences (IFNγ vs Co-culture in particular). If
the screen pipeline cannot detect a difference under this construction,
the pipeline is broken and no other Frangieh number is trustworthy.

The output is labeled `label = "POSITIVE_CONTROL_ONLY"` and must never
appear as a scientific result.

Design
------
* Same loader, same MOI==1 filter, same PCA machinery, same null
  construction as 41_screen_frangieh.py.
* Basis: sgRNA non-targeting cells pooled across ALL three arms.
* Environments: cells within each arm are split into contiguous NMIN-cell
  pseudo-envs labeled "{arm}_chunk_{i:04d}". Both perturbed and control
  cells within an arm are eligible (the arm effect is present in both).
  Pooling into pseudo-envs is required because 03_screen.py aborts at
  fewer than 4 envs, and we have only 3 arms.
* The "non-targeting" label is preserved for the PCA-basis cells only, so
  they are not double-counted as envs.

Expected outcome
----------------
mean_ratio_pairs well above 1.0 (say, >= 1.5) for a healthy pipeline. If
the number lands near the null band [0.90, 1.10] the pipeline is broken.

Usage (A100):
    python causalbench/scripts/42_frangieh_poscontrol.py
    python causalbench/scripts/42_frangieh_poscontrol.py --nmin 200
"""
import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

# 41_screen_frangieh.py lives next to this script (in the framework tree);
# 03_screen.py lives in the separate causalbench source tree and is loaded
# by 41 via its own hardcoded path. So we resolve 41 relative to __file__,
# not to any hardcoded tree root.
SCRIPTS_DIR = Path(__file__).resolve().parent
FRANGIEH_SCRIPT = SCRIPTS_DIR / "41_screen_frangieh.py"
# Results dir: 41 also writes here, so we use its constant post-import.


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Re-use 41's loader / regex / metadata parsing and 03's screen kernel by
# importing 41 (which itself imports 03).
FR = _load_module(FRANGIEH_SCRIPT, "_frangieh41")
load_metadata = FR.load_metadata
load_expression = FR.load_expression
screen_run = FR.screen_run
ARMS = FR.ARMS
CTRL_LABEL = FR.CTRL_LABEL
EXPR_CSV = FR.EXPR_CSV
META_CSV = FR.META_CSV
SEED = FR.SEED
SCREEN_DIR = FR.SCREEN_DIR


def build_arm_chunk_iv(iv_all, cond_all, nmin, seed, chunks_per_arm=None):
    """Return a new iv array where:
      - cells originally labeled non-targeting (basis cells) stay as
        CTRL_LABEL. They are pooled across arms and used ONLY for the
        PCA basis and reference pool.
      - every other cell gets label "{arm}_chunk_{i:04d}", chunked from
        that arm's cells into NMIN-size pseudo-envs.
      - leftover cells that do not complete an NMIN chunk get label
        "excluded" (screen_run's excluded-name path).

    chunks_per_arm caps how many pseudo-envs each arm contributes.

    WHY THE CAP MATTERS. `screen_run` samples random env PAIRS. A pair drawn
    from the SAME arm has Delta = 0 and contributes pure noise; only
    CROSS-arm pairs carry the arm effect this control is meant to measure.
    Because the ratio is a MEDIAN over pairs, a large within-arm fraction
    drags the median toward the null and ATTENUATES the positive control.

    Uncapped at NMIN=100 the split is 1113 chunks (407/267/439) giving
    P(within-arm) = 0.347. Capping at 2 per arm gives 6 envs and
    P(within-arm) = 0.200, which is closer to the per-target screens where
    essentially every pair is a genuine cross-perturbation contrast. Neither
    reaches 0 because only 3 arms exist and `screen_run` aborts below 4
    environments. Both configurations are therefore run and reported, with
    the within-arm fraction stated, rather than picking one silently.
    """
    rng = np.random.default_rng(seed)
    iv_new = np.array(iv_all, dtype=object).copy()
    is_ctrl = iv_all == CTRL_LABEL
    for arm in ARMS:
        arm_mask = (cond_all == arm) & (~is_ctrl)
        arm_idx = np.where(arm_mask)[0]
        rng.shuffle(arm_idx)
        n_chunks = len(arm_idx) // nmin
        if chunks_per_arm is not None:
            n_chunks = min(n_chunks, int(chunks_per_arm))
        for i in range(n_chunks):
            chunk_rows = arm_idx[i * nmin:(i + 1) * nmin]
            iv_new[chunk_rows] = f"{arm}_chunk_{i:04d}"
        leftover = arm_idx[n_chunks * nmin:]
        iv_new[leftover] = "excluded"
    return iv_new


def within_arm_pair_fraction(iv_pos):
    """Fraction of distinct env pairs that fall inside one arm (Delta = 0).
    Reported so the attenuation of the positive control is explicit.
    """
    labels = [g for g in np.unique(iv_pos)
              if g not in (CTRL_LABEL, "excluded")]
    per_arm = {}
    for g in labels:
        arm = g.rsplit("_chunk_", 1)[0]
        per_arm[arm] = per_arm.get(arm, 0) + 1
    n = sum(per_arm.values())
    if n < 2:
        return None
    total = n * (n - 1) // 2
    within = sum(k * (k - 1) // 2 for k in per_arm.values())
    return float(within / total)


def atomic_write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.rename(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hvg", type=int, default=None)
    ap.add_argument("--nmin", type=int, default=100,
                    help="(legacy, ignored; use --rungs)")
    ap.add_argument("--rungs", default="100,200",
                    help="comma-separated NMIN rungs; c_hat n-invariance is "
                         "checked across these on the positive control too")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed used for the reported env-chunk layout")
    ap.add_argument("--out", default=str(
        SCREEN_DIR / "frangieh_poscontrol.json"))
    a = ap.parse_args()

    out_path = Path(a.out)
    if out_path.exists():
        print(f"[skip] {out_path} already exists", flush=True)
        return

    md, meta = load_metadata()
    X, vn, cell_order = load_expression(EXPR_CSV,
                                         list(md["NAME"].astype(str)),
                                         hvg=a.hvg)
    md = md.set_index(md["NAME"].astype(str)).reindex(cell_order)
    ok = md["iv"].notna().to_numpy()
    X, md = X[ok], md.loc[ok]
    iv_orig = md["iv"].to_numpy().astype(object)
    cond_all = md["condition"].to_numpy().astype(str)
    print(f"[align] {X.shape[0]} cells aligned", flush=True)

    def c_hat_of(rec, nmin):
        """c_hat = (ratio^2 - 1)/NMIN. Proportional to ||Delta||^2/tr(Sigma),
        the squared standardized between-environment separation per cell.
        Constant of proportionality is ~1/3.5 rather than the 1/4 the
        expectation algebra gives, because 03_screen.py:111 takes a MEDIAN OF
        NORMS over heterogeneous pairs rather than an RMS. Verified against
        the code path with known Delta and Sigma; scales exactly as
        (tau/sigma)^2. Near-invariant in NMIN, not an exact identity.
        """
        if "mean_ratio_pairs" not in rec:
            return None
        r = rec["mean_ratio_pairs"]
        return float(max(r * r - 1.0, 0.0) / nmin)

    rungs = tuple(int(x) for x in a.rungs.split(","))
    seeds = tuple(int(x) for x in a.seeds.split(","))
    print(f"\n[config] rungs={rungs}  seeds={seeds}  d={a.d}", flush=True)

    def _stats(runs, field):
        vals = [r[field] for r in runs if r.get(field) is not None]
        if not vals:
            return None, None, None
        m, s = float(np.mean(vals)), float(np.std(vals))
        ci = None
        if len(vals) > 1:
            se = s / np.sqrt(len(vals))
            ci = [float(m - 1.96 * se), float(m + 1.96 * se)]
        return m, s, ci

    # Two chunk configurations. `capped` keeps the within-arm (Delta = 0)
    # pair fraction low so the arm effect is not attenuated by null pairs;
    # `all` uses every cell. Both reported.
    chunk_configs = [("capped2", 2), ("all", None)]

    per_rung = {}
    for nmin in rungs:
        for cfg_name, cpa in chunk_configs:
            key = f"{nmin}:{cfg_name}"
            print(f"\n{'-' * 70}\nNMIN={nmin}  chunks_per_arm={cpa or 'all'}"
                  f"\n{'-' * 70}", flush=True)
            iv_ref = build_arm_chunk_iv(iv_orig, cond_all, nmin, a.seed,
                                         chunks_per_arm=cpa)
            us, cs = np.unique(iv_ref, return_counts=True)
            per_arm_chunk_counts = {
                arm: int(sum(1 for g in us if g.startswith(f"{arm}_chunk_")))
                for arm in ARMS}
            n_ctrl = int(cs[us == CTRL_LABEL].sum()) if CTRL_LABEL in us else 0
            n_excl = int(cs[us == "excluded"].sum()) if "excluded" in us else 0
            n_env_labels = int(sum(1 for g in us
                                    if g not in (CTRL_LABEL, "excluded")))
            wa_frac = within_arm_pair_fraction(iv_ref)
            print(f"[env] pseudo-envs per arm: {per_arm_chunk_counts}   "
                  f"total envs={n_env_labels}", flush=True)
            print(f"[env] within-arm pair fraction (Delta=0, attenuates): "
                  f"{wa_frac if wa_frac is None else round(wa_frac, 3)}",
                  flush=True)
            print(f"[env] basis cells: {n_ctrl}   excluded: {n_excl}",
                  flush=True)

            gate_runs, pos_runs = [], []
            for seed in seeds:
                iv_s = build_arm_chunk_iv(iv_orig, cond_all, nmin, seed,
                                           chunks_per_arm=cpa)
                g = screen_run(X, iv_s, vn, nmin, a.d, step0=True,
                                seed=seed, no_drop=False)
                p = screen_run(X, iv_s, vn, nmin, a.d, step0=False,
                                seed=seed, no_drop=False)
                g["c_hat"] = c_hat_of(g, nmin)
                p["c_hat"] = c_hat_of(p, nmin)
                gate_runs.append(g)
                pos_runs.append(p)

            g_r, g_r_sd, g_r_ci = _stats(gate_runs, "mean_ratio_pairs")
            g_c, g_c_sd, g_c_ci = _stats(gate_runs, "c_hat")
            p_r, p_r_sd, p_r_ci = _stats(pos_runs, "mean_ratio_pairs")
            p_c, p_c_sd, p_c_ci = _stats(pos_runs, "c_hat")

            print(f"[step0]      ratio="
                  f"{g_r if g_r is None else round(g_r, 4)}"
                  f" +/- {g_r_sd if g_r_sd is None else round(g_r_sd, 4)}   "
                  f"c_hat={g_c if g_c is None else format(g_c, '.4e')}",
                  flush=True)
            print(f"[poscontrol] ratio="
                  f"{p_r if p_r is None else round(p_r, 4)}"
                  f" +/- {p_r_sd if p_r_sd is None else round(p_r_sd, 4)}   "
                  f"c_hat={p_c if p_c is None else format(p_c, '.4e')}",
                  flush=True)

            per_rung[key] = dict(
                nmin=int(nmin), chunks_per_arm=cpa, config=cfg_name,
                arm_chunk_counts=per_arm_chunk_counts,
                within_arm_pair_fraction=wa_frac,
                n_basis_cells=n_ctrl, n_excluded_leftover=n_excl,
                n_environments=n_env_labels,
                step0=dict(runs=gate_runs, ratio_mean=g_r, ratio_sd=g_r_sd,
                           ratio_ci95=g_r_ci, c_hat_mean=g_c,
                           c_hat_sd=g_c_sd, c_hat_ci95=g_c_ci),
                poscontrol=dict(runs=pos_runs, ratio_mean=p_r,
                                ratio_sd=p_r_sd, ratio_ci95=p_r_ci,
                                c_hat_mean=p_c, c_hat_sd=p_c_sd,
                                c_hat_ci95=p_c_ci),
            )

    # c_hat n-invariance on the positive control, within each chunk config.
    c_hat_cv_by_config = {}
    for cfg_name, _ in chunk_configs:
        vals = [per_rung[f"{n}:{cfg_name}"]["poscontrol"]["c_hat_mean"]
                for n in rungs
                if per_rung.get(f"{n}:{cfg_name}", {})
                .get("poscontrol", {}).get("c_hat_mean") is not None]
        c_hat_cv_by_config[cfg_name] = (
            float(np.std(vals) / np.mean(vals))
            if len(vals) > 1 and np.mean(vals) > 0 else None)
    c_hat_cv = c_hat_cv_by_config.get("capped2")

    # Reference values from 48_nmin_ladder_all.py (per-TARGET screens).
    PER_TARGET_C_HAT = {"k562": 9.33e-02, "norman": 6.51e-02, "rpe1": 1.45e-02,
                        "frangieh:Co-culture": 2.19e-03,
                        "frangieh:IFNg": 1.80e-03,
                        "frangieh:Control": "NOT_ESTIMABLE"}

    # Reference = the capped config at the first rung: lowest within-arm
    # (Delta = 0) pair fraction, so least attenuated.
    REF_KEY = f"{rungs[0]}:capped2"
    ref_r = per_rung[REF_KEY]["poscontrol"]["ratio_mean"]
    ref_c = per_rung[REF_KEY]["poscontrol"]["c_hat_mean"]
    verdict = "PIPELINE_BROKEN"
    if ref_r is not None:
        if ref_r >= 1.5:
            verdict = "PIPELINE_INTACT_CLEAR"
        elif ref_r >= 1.2:
            verdict = "PIPELINE_INTACT_MARGINAL"
        elif ref_r >= 1.1:
            verdict = "PIPELINE_WEAK"

    out = dict(
        label="POSITIVE_CONTROL_ONLY",
        dataset="frangieh",
        source_metadata=str(META_CSV),
        source_expression=str(EXPR_CSV),
        seeds=list(seeds), rungs=list(rungs), d=int(a.d),
        env_label_source="arm-chunk pseudo-envs of NMIN cells each",
        per_rung=per_rung,
        c_hat_cv_across_rungs=c_hat_cv,
        c_hat_cv_by_config=c_hat_cv_by_config,
        reference_config=REF_KEY,
        chunk_configs=[dict(name=n, chunks_per_arm=c)
                       for n, c in chunk_configs],
        c_hat_note=("c_hat is proportional to ||Delta||^2/tr(Sigma) with "
                     "constant ~1/3.5 (not 1/4) because the estimator uses a "
                     "median of norms over heterogeneous pairs. NEAR-invariant "
                     "in NMIN, not an exact identity. CV across rungs is an "
                     "n-invariance diagnostic, NOT a sampling error bar; use "
                     "the seed-level ci95 fields for uncertainty."),
        per_target_c_hat_reference=PER_TARGET_C_HAT,
        calibration_reading=(
            "The arm effect is large and published. Its c_hat on THIS loader "
            "is the calibration reference that makes the per-target Frangieh "
            "values (Co-culture 2.19e-03, IFNg 1.80e-03, Control NOT "
            "ESTIMABLE) interpretable as genuinely small rather than merely "
            "small-looking."),
        expected_ratio_note=("arms have well-known large transcriptional "
                              "differences (IFNγ vs Co-culture in particular); "
                              "a healthy pipeline is expected to give "
                              "mean_ratio_pairs >= 1.5 here."),
        verdict=verdict,
        do_not_report_as_result=(
            "This value is a deliberate positive control constructed by "
            "labelling cells by arm rather than perturbation. It is NOT a "
            "detectability estimate for perturbations and must never appear "
            "in a paper table as a Frangieh dataset result."),
        **meta,
    )
    atomic_write_json(out_path, out)

    print("\n" + "=" * 78, flush=True)
    print("FRANGIEH POSITIVE CONTROL (CONTROL ONLY, never report as result)",
          flush=True)
    print("=" * 78, flush=True)
    def _fmt(v, e=False):
        if v is None:
            return "ABORT"
        return format(v, ".4e") if e else f"{v:.4f}"

    print(f"  {'rung':<6}{'cfg':<9}{'envs':>6}{'wArm':>7}"
          f"{'step0 ratio':>14}{'step0 c_hat':>14}"
          f"{'POS ratio':>14}{'POS c_hat':>14}")
    for n in rungs:
        for cfg_name, _ in chunk_configs:
            R = per_rung[f"{n}:{cfg_name}"]
            g, p = R["step0"], R["poscontrol"]
            wa = R.get("within_arm_pair_fraction")
            print(f"  {n:<6}{cfg_name:<9}{R['n_environments']:>6}"
                  f"{('-' if wa is None else f'{wa:.3f}'):>7}"
                  f"{_fmt(g['ratio_mean']):>14}"
                  f"{_fmt(g['c_hat_mean'], True):>14}"
                  f"{_fmt(p['ratio_mean']):>14}"
                  f"{_fmt(p['c_hat_mean'], True):>14}")
    print("\n  seed-level 95% CIs on POS c_hat (this IS the error bar):")
    for n in rungs:
        for cfg_name, _ in chunk_configs:
            p = per_rung[f"{n}:{cfg_name}"]["poscontrol"]
            if p.get("c_hat_ci95"):
                print(f"    n={n:<5} {cfg_name:<9} "
                      f"[{p['c_hat_ci95'][0]:.4e}, {p['c_hat_ci95'][1]:.4e}]")
    print("\n  c_hat CV across rungs (n-invariance diagnostic, NOT an "
          "error bar):")
    for cfg_name, cv in c_hat_cv_by_config.items():
        print(f"    {cfg_name:<9} "
              f"{'-' if cv is None else f'{cv * 100:.1f}%'}")
    print(f"\n  NOTE: 'wArm' is the fraction of env pairs drawn from the SAME "
          f"arm.\n        Those have Delta=0 and ATTENUATE the control. "
          f"'capped2' is the\n        reference config because it minimises "
          f"this; 3 arms cannot reach 0\n        without tripping the >=4 env "
          f"abort in screen_run().")

    print(f"\n  CALIBRATION -- arm effect vs per-target c_hat "
          f"(from 48_nmin_ladder_all.py):")
    if ref_c:
        for k, v in PER_TARGET_C_HAT.items():
            if isinstance(v, str):
                print(f"    {k:<22} {v}")
            else:
                print(f"    {k:<22} {v:.3e}   "
                      f"arm-effect is {ref_c / v:>7.1f}x larger")
    print(f"\n  VERDICT: {verdict}")
    if verdict == "PIPELINE_BROKEN":
        print("  STOP -- the pipeline cannot detect a large published effect "
              "on this loader. Every Frangieh number is void.")
    print(f"\n[write] {out_path}", flush=True)


if __name__ == "__main__":
    main()
