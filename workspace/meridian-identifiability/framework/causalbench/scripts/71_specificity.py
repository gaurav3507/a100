"""Specificity statistic for the environment-validity screen.

The existing screen measures shift MAGNITUDE (mean_ratio_vs_ctrl, whether
perturbations differ from control) and DIMENSIONALITY (dims_above_2x). It does
not directly report SPECIFICITY -- whether perturbations differ FROM EACH
OTHER. This script adds two specificity readouts alongside the existing metric
and runs the gates the task spec requires.

Two definitions, both reported:

  specificity_ratio = median( ||mean(Z_a) - mean(Z_b)|| ) / median(within_m)
      across distinct perturbation pairs (a, b). Same PCA basis (control cells),
      same d, same target-column zeroing, same within-perturbation split-half
      denominator (within_m) as 03_screen.py. In arithmetic this equals
      mean_ratio_pairs from 03_screen.py by construction; it is retained as a
      self-check against the existing screen output.

  specificity_angular = median( 1 - cos(shift_a, shift_b) ) across pairs,
      where shift_i = env_mean - control_ref (both in projected latent).
      Magnitude-independent -- this is what makes the measurement new.
      Range [0, 2]: ~0 when all shifts point the same way, ~1 under isotropic
      noise, ~2 under antiparallel signals.

The two together cover the possibility that inter-perturbation dispersion is
present without direction diversity (same-direction shifts of different
magnitudes) and vice versa.

Gates
-----
(a) Step-0 gate. Control cells shuffled into NMIN pseudo-envs. Both statistics
    must return ~1.0 (tolerance [0.9, 1.1]). If either does not, the statistic
    is mis-normalised and the script aborts.
(b) Same-direction synthetic. All perturbations shifted by the same vector.
    specificity_angular must come back LOW.
    mean_ratio_vs_ctrl_recomp must come back HIGH.
    specificity_ratio will return near 1.0 (its null level) and is reported
    for context. A different-direction synthetic runs as a positive control:
    both spec_angular and vs_ctrl must be HIGH.
    If the same-direction synthetic does NOT drop spec_angular, the statistic
    is tracking magnitude rather than direction and the script aborts.

Inputs and pipeline: same as 03_screen.py. Metric helpers (fit_pca, project,
coefs, offdiag) are imported from 03_screen.py via importlib exactly as
40_screen_norman.py does it. 03_screen.py is not modified.

Usage (A100, cb venv):
    python causalbench/scripts/71_specificity.py            # writes if absent
    python causalbench/scripts/71_specificity.py --clean    # overwrite
    python causalbench/scripts/71_specificity.py --gates-only
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

CAUSALBENCH = Path("/workspace/meridian-identifiability/causalbench")
OUT_DIR = Path(__file__).resolve().parents[1] / "results/screen"
OUT_JSON = OUT_DIR / "specificity.json"

SEED = 0
NMIN = 200
D = 10
N_PAIRS_SPEC = 500
STEP0_TOL = (0.9, 1.1)
DATASETS = ("k562", "rpe1", "norman")


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Byte-identical import path used by 40_screen_norman.py; 03_screen.py is not
# modified by this script.
SCREEN_MOD = _load_module(CAUSALBENCH / "scripts/03_screen.py", "_screen03")
fit_pca = SCREEN_MOD.fit_pca
project = SCREEN_MOD.project
coefs = SCREEN_MOD.coefs
offdiag = SCREEN_MOD.offdiag
CTRL = SCREEN_MOD.CTRL

NORMAN_MOD = _load_module(CAUSALBENCH / "scripts/40_screen_norman.py",
                           "_norman40")


# --------------------------------------------------------------- data loading
def load_dataset(name, filt=False):
    if name == "norman":
        X, iv, vn, _ = NORMAN_MOD.load_norman()
        return X, iv, vn
    X, iv, vn = SCREEN_MOD.load(name, filt)
    return X, iv, vn


# ---------------------------------------------------- core specificity compute
def compute_specificities(X, iv, vn, nmin, d, step0=False,
                          seed=SEED, n_pairs_spec=N_PAIRS_SPEC):
    """Return a dict with specificity_ratio, specificity_angular, and the
    same-construction mean_ratio_vs_ctrl for reference. Uses the identical
    PCA basis, target-column zeroing, and within-perturbation split-half
    denominator as 03_screen.py.
    """
    rng = np.random.default_rng(seed)
    half = nmin // 2
    gidx = {g: i for i, g in enumerate(vn)}

    ctrl_rows = np.where(iv == CTRL)[0]
    if len(ctrl_rows) < nmin:
        return dict(aborted="fewer_control_cells_than_nmin",
                    n_control_cells=int(len(ctrl_rows)))

    mu, W = fit_pca(X[ctrl_rows], d)
    # Fixed control reference in latent for the angular shift definition.
    # No target-column drop is applied here (there is no per-env target).
    ctrl_ref = project(X[ctrl_rows], mu, W, set()).mean(0)

    if step0:
        rng.shuffle(ctrl_rows)
        k = len(ctrl_rows) // nmin
        envs = [(f"ctrl{i}", ctrl_rows[i * nmin:(i + 1) * nmin], None)
                for i in range(k)]
    else:
        us, cs = np.unique(iv, return_counts=True)
        keep = [(g, n) for g, n in zip(us, cs)
                if g not in (CTRL, "excluded") and n >= nmin]
        envs = [(g, np.where(iv == g)[0], gidx.get(g)) for g, _ in keep]

    if len(envs) < 4:
        return dict(aborted="fewer_than_4_envs", n_envs=len(envs))

    env_means = []
    within_m = []
    for g, rows, drop in envs:
        chosen = rng.choice(rows, nmin, replace=False)
        drops = {drop} if drop is not None else set()
        Z = project(X[chosen], mu, W, drops)
        env_means.append(Z.mean(0))
        Za, Zb = Z[:half], Z[half:nmin]
        within_m.append(np.linalg.norm(Za.mean(0) - Zb.mean(0)))
    env_means = np.array(env_means)

    n = len(envs)
    max_pairs = n * (n - 1) // 2
    n_pairs = min(n_pairs_spec, max_pairs)
    pair_m, cos_sims = [], []
    for _ in range(n_pairs):
        i, j = rng.choice(n, 2, replace=False)
        v_i, v_j = env_means[i], env_means[j]
        pair_m.append(float(np.linalg.norm(v_i - v_j)))
        s_i = v_i - ctrl_ref
        s_j = v_j - ctrl_ref
        denom = np.linalg.norm(s_i) * np.linalg.norm(s_j) + 1e-12
        cos_sims.append(float(s_i @ s_j / denom))

    between_m = [float(np.linalg.norm(m - ctrl_ref)) for m in env_means]

    return dict(
        step0=bool(step0), nmin=int(nmin), d=int(d), seed=int(seed),
        n_envs=int(n), n_pairs=int(n_pairs),
        specificity_ratio=float(np.median(pair_m) / np.median(within_m)),
        specificity_angular=float(1.0 - np.median(cos_sims)),
        median_cos_pairs=float(np.median(cos_sims)),
        mean_ratio_vs_ctrl_recomp=float(np.median(between_m) / np.median(within_m)),
        median_within_m=float(np.median(within_m)),
        median_pair_m=float(np.median(pair_m)),
        median_between_m=float(np.median(between_m)),
    )


# ==================================================================== SYNTHETIC
def make_synthetic(n_perts=30, n_cells=250, n_features=50,
                   shift_size=3.0, direction="same", seed=SEED):
    """Return (X, iv, vn) with n_perts perturbations plus one control class.

    direction="same":      every perturbation shifts by the SAME unit vector
    direction="different": each perturbation shifts by an independent vector
    Shift magnitude is `shift_size` in every case; within-perturbation noise
    is unit-variance isotropic Gaussian.
    """
    rng = np.random.default_rng(seed)

    if direction == "same":
        d_hat = rng.normal(0, 1, n_features)
        d_hat /= (np.linalg.norm(d_hat) + 1e-12)
        shifts = np.tile(d_hat * shift_size, (n_perts, 1))
    elif direction == "different":
        shifts = rng.normal(0, 1, (n_perts, n_features))
        shifts /= (np.linalg.norm(shifts, axis=1, keepdims=True) + 1e-12)
        shifts *= shift_size
    else:
        raise ValueError(direction)

    ctrl_cells = rng.normal(0, 1, (n_cells, n_features))
    X_parts = [ctrl_cells]
    iv_parts = [np.array([CTRL] * n_cells, dtype=object)]
    for i in range(n_perts):
        pert = rng.normal(0, 1, (n_cells, n_features)) + shifts[i]
        X_parts.append(pert)
        iv_parts.append(np.array([f"pert{i:02d}"] * n_cells, dtype=object))
    X = np.concatenate(X_parts, 0)
    iv = np.concatenate(iv_parts, 0)
    vn = [f"g{i:03d}" for i in range(n_features)]
    return X, iv, vn


def synthetic_gate():
    """Gate (b): same-direction synthetic must give spec_angular LOW while
    mean_ratio_vs_ctrl_recomp is HIGH. Different-direction synthetic runs as
    a positive control (both should be HIGH).
    """
    same = compute_specificities(*make_synthetic(direction="same"),
                                  nmin=NMIN, d=D, step0=False)
    diff = compute_specificities(*make_synthetic(direction="different"),
                                  nmin=NMIN, d=D, step0=False)

    same_pass = (same["specificity_angular"] < 0.3
                 and same["mean_ratio_vs_ctrl_recomp"] > 2.0)
    diff_pass = (diff["specificity_angular"] > 0.5
                 and diff["mean_ratio_vs_ctrl_recomp"] > 2.0)

    return dict(
        same_direction=dict(**same,
                             expected="spec_angular LOW, vs_ctrl HIGH",
                             pass_=bool(same_pass)),
        different_direction=dict(**diff,
                                  expected="spec_angular HIGH, vs_ctrl HIGH",
                                  pass_=bool(diff_pass)),
        overall_pass=bool(same_pass and diff_pass),
    )


# ================================================================= STEP-0 GATE
def step0_gate_for(dataset):
    X, iv, vn = load_dataset(dataset, filt=False)
    res = compute_specificities(X, iv, vn, nmin=NMIN, d=D, step0=True)
    if "aborted" in res:
        return dict(**res, pass_=False)
    lo, hi = STEP0_TOL
    ratio_ok = lo <= res["specificity_ratio"] <= hi
    ang_ok = lo <= res["specificity_angular"] <= hi
    return dict(**res, tolerance=list(STEP0_TOL),
                spec_ratio_pass=bool(ratio_ok),
                spec_angular_pass=bool(ang_ok),
                pass_=bool(ratio_ok and ang_ok))


# ====================================================================== REAL
def real_run_for(dataset):
    X, iv, vn = load_dataset(dataset, filt=False)
    res = compute_specificities(X, iv, vn, nmin=NMIN, d=D, step0=False)

    # Cross-check specificity_ratio against the existing screen's
    # mean_ratio_pairs. They must match to numerical precision because the
    # arithmetic is identical.
    ref = None
    if dataset == "norman":
        p = OUT_DIR / "norman.json"
        if p.exists():
            j = json.load(open(p))
            entry = next((r for r in j.get("screen", [])
                          if r.get("nmin") == NMIN and r.get("d") == D), None)
            if entry:
                ref = float(entry["mean_ratio_pairs"])
    else:
        p = OUT_DIR / f"{dataset}_filt0_n{NMIN}_d{D}.json"
        if p.exists():
            j = json.load(open(p))
            ref = float(j.get("mean_ratio_pairs"))

    self_check = None
    if ref is not None and "specificity_ratio" in res:
        delta = abs(res["specificity_ratio"] - ref)
        self_check = dict(
            existing_mean_ratio_pairs=ref,
            recomputed_specificity_ratio=res["specificity_ratio"],
            delta=float(delta),
            match_within_1e_3=bool(delta < 1e-3),
        )

    return dict(**res, self_check=self_check)


# ----------------------------------------------------------------------- util
def atomic_write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.rename(tmp, path)


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true",
                    help="rm the output JSON before running")
    ap.add_argument("--gates-only", action="store_true",
                    help="run only the gates; skip real datasets")
    args = ap.parse_args()

    if args.clean and OUT_JSON.exists():
        print(f"[clean] rm {OUT_JSON}", flush=True)
        OUT_JSON.unlink()
    if OUT_JSON.exists():
        print(f"[skip] {OUT_JSON} exists; --clean to overwrite", flush=True)
        return

    # -------- GATE (b): synthetic
    print("[gate] synthetic (same-direction + different-direction control)",
          flush=True)
    synth = synthetic_gate()
    s = synth["same_direction"]
    d = synth["different_direction"]
    print(f"       same-dir : spec_angular={s['specificity_angular']:.4f}  "
          f"spec_ratio={s['specificity_ratio']:.4f}  "
          f"vs_ctrl={s['mean_ratio_vs_ctrl_recomp']:.4f}  "
          f"-> {'PASS' if s['pass_'] else 'FAIL'}", flush=True)
    print(f"       diff-dir : spec_angular={d['specificity_angular']:.4f}  "
          f"spec_ratio={d['specificity_ratio']:.4f}  "
          f"vs_ctrl={d['mean_ratio_vs_ctrl_recomp']:.4f}  "
          f"-> {'PASS' if d['pass_'] else 'FAIL'}", flush=True)
    if not synth["overall_pass"]:
        print("[gate] SYNTHETIC FAIL -- statistic is not measuring what it claims",
              flush=True)
        atomic_write_json(OUT_JSON, dict(synthetic_gate=synth,
                                          aborted="synthetic_gate_failed"))
        sys.exit(1)
    print("[gate] synthetic PASS", flush=True)

    # -------- GATE (a): step-0 on each dataset
    print("[gate] step-0 on real datasets", flush=True)
    step0 = {}
    for ds in DATASETS:
        print(f"       {ds}...", flush=True)
        step0[ds] = step0_gate_for(ds)
        r = step0[ds]
        if r.get("aborted"):
            print(f"       {ds}: ABORTED ({r['aborted']})", flush=True)
        else:
            print(f"       {ds}: spec_ratio={r['specificity_ratio']:.4f}  "
                  f"spec_angular={r['specificity_angular']:.4f}  "
                  f"-> {'PASS' if r['pass_'] else 'FAIL'}", flush=True)
    step0_pass = all(s.get("pass_") for s in step0.values())
    if not step0_pass:
        print("[gate] STEP-0 FAIL -- statistic mis-normalised on real data",
              flush=True)
        atomic_write_json(OUT_JSON, dict(synthetic_gate=synth,
                                          step0_gate=step0,
                                          aborted="step0_gate_failed"))
        sys.exit(1)
    print("[gate] step-0 PASS on all datasets", flush=True)

    if args.gates_only:
        atomic_write_json(OUT_JSON, dict(synthetic_gate=synth,
                                          step0_gate=step0,
                                          note="gates_only run"))
        print("[done] --gates-only requested", flush=True)
        return

    # -------- REAL: compute specificity per dataset
    print("[real] running on real datasets", flush=True)
    real = {}
    for ds in DATASETS:
        print(f"       {ds}...", flush=True)
        real[ds] = real_run_for(ds)

    out = dict(
        config=dict(nmin=int(NMIN), d=int(D), seed=int(SEED),
                    n_pairs_spec=int(N_PAIRS_SPEC),
                    step0_tolerance=list(STEP0_TOL)),
        datasets=list(DATASETS),
        source_screen=("mean_ratio_pairs from causalbench/results/screen/"
                        "{k562,rpe1}_filt0_n200_d10.json and norman.json"),
        synthetic_gate=synth,
        step0_gate=step0,
        real=real,
    )
    atomic_write_json(OUT_JSON, out)
    print(f"\n[write] {OUT_JSON}", flush=True)

    # Summary
    print("\n" + "=" * 78, flush=True)
    print("SPECIFICITY SUMMARY (NMIN=200, d=10, filt0; primary metric: pairs)",
          flush=True)
    print("=" * 78, flush=True)
    print(f"  {'dataset':<10}{'mean_ratio_pairs':>20}{'spec_ratio':>14}"
          f"{'spec_angular':>16}{'match':>8}")
    for ds in DATASETS:
        r = real[ds]
        ref = r.get("self_check", {}).get("existing_mean_ratio_pairs") \
                if r.get("self_check") else None
        matched = r.get("self_check", {}).get("match_within_1e_3", False) \
                if r.get("self_check") else False
        print(f"  {ds:<10}"
              f"{(f'{ref:.3f}' if ref is not None else 'N/A'):>20}"
              f"{r['specificity_ratio']:>14.3f}"
              f"{r['specificity_angular']:>16.4f}"
              f"{('OK' if matched else '-'):>8}")
    print(f"\n  synthetic gate: {'PASS' if synth['overall_pass'] else 'FAIL'}")
    print(f"  step-0 gate:    {'PASS' if step0_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
