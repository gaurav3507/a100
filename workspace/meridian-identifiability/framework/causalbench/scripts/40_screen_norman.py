"""Environment-validity screen on Norman 2019 (predictive-validity check).

Runs the SAME metric functions from causalbench/scripts/03_screen.py and
04_spectrum.py against Norman 2019 (Zhang et al.'s CRL-succeeds dataset), so
the Norman number is directly comparable to CausalBench K562.

The metric helpers (fit_pca, project, coefs, offdiag) are imported from
03_screen.py via importlib.util; only the outer loop is duplicated. This means
the arithmetic is byte-identical to what the CausalBench screen would compute
if it could be fed Norman-shaped arrays.

Norman-specific differences vs CausalBench, all handled here:
    * Loaded from Norman2019_raw.h5ad (CausalBench is npz).
    * CRISPR activation (genes UP) not CRISPRi (genes DOWN); the metric doesn't
      care about sign.
    * obs['guide_ids']: '' = control, 'gene' = single-pert, 'A,B' = double-pert.
      We map to CausalBench's convention: 'non-targeting' / gene / 'excluded'.
    * 0/105 target genes appear in the 5000 HVG columns. So the target-column
      drop in project() is a no-op for every Norman perturbation (same code
      path, no rewrite). Flagged as target_column_drop_is_noop=True in JSON.

Output:
    causalbench/results/screen/norman.json
    causalbench/results/spectrum/norman.json

Usage:
    python causalbench/scripts/40_screen_norman.py
"""
import os, sys, json, importlib.util
from pathlib import Path

import numpy as np
import anndata as ad

NORMAN_H5AD = Path("/workspace/external/discrepancy_vae/datasets/Norman2019_raw.h5ad")
CAUSALBENCH = Path("/workspace/meridian-identifiability/causalbench")
SCREEN_DIR  = CAUSALBENCH / "results/screen"
SPECTRUM_DIR = CAUSALBENCH / "results/spectrum"

SEED = 0
CTRL_LABEL = "non-targeting"

NMIN_SET       = (200, 100)
D_SET_SCREEN   = (5, 10)
D_SET_SPECTRUM = (10, 20, 50)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCREEN_MOD = _load_module(CAUSALBENCH / "scripts/03_screen.py", "_screen03")
fit_pca = SCREEN_MOD.fit_pca
project = SCREEN_MOD.project
coefs = SCREEN_MOD.coefs
offdiag = SCREEN_MOD.offdiag
N_PAIRS = SCREEN_MOD.N_PAIRS

SPECTRUM_MOD = _load_module(CAUSALBENCH / "scripts/04_spectrum.py", "_spectrum04")


def load_norman():
    print(f"[load] {NORMAN_H5AD}", flush=True)
    a = ad.read_h5ad(NORMAN_H5AD)
    Xany = a.X
    X = np.asarray(Xany.toarray() if hasattr(Xany, "toarray") else Xany,
                    dtype=np.float64)
    vn = [str(v) for v in a.var_names]
    g = np.asarray(a.obs["guide_ids"]).astype(str)
    iv = np.array([
        CTRL_LABEL if x == "" else
        ("excluded" if "," in x else x)
        for x in g
    ], dtype=object)
    keep = iv != "excluded"
    X, iv = X[keep], iv[keep]
    ctrl_n = int((iv == CTRL_LABEL).sum())
    single_n = int((iv != CTRL_LABEL).sum())
    uniq_genes = sorted(set(iv[iv != CTRL_LABEL]))
    in_cols = sum(1 for u in uniq_genes if u in set(vn))
    print(f"[load] kept {X.shape[0]:>7} cells x {X.shape[1]} genes "
          f"(controls={ctrl_n}, single-pert={single_n}); "
          f"{len(uniq_genes)} unique targets, {in_cols} in var_names",
          flush=True)
    meta = dict(
        n_cells_kept=int(X.shape[0]),
        n_genes=int(X.shape[1]),
        n_control_cells=ctrl_n,
        n_single_perturbation_cells=single_n,
        n_single_perturbation_genes=len(uniq_genes),
        n_targets_in_var_names=in_cols,
        target_column_drop_is_noop=(in_cols == 0),
    )
    return X, iv, vn, meta


def screen_run(X, iv, vn, nmin, d, step0):
    """Mirror of 03_screen.py run(): identical arithmetic, Norman as input."""
    rng = np.random.default_rng(SEED)
    half = nmin // 2
    gidx = {g: i for i, g in enumerate(vn)}
    ctrl_rows = np.where(iv == CTRL_LABEL)[0]
    mu, W = fit_pca(X[ctrl_rows], d)

    if step0:
        rng.shuffle(ctrl_rows)
        k = len(ctrl_rows) // nmin
        envs = [(f"ctrl{i}", ctrl_rows[i*nmin:(i+1)*nmin], None) for i in range(k)]
        ref_pool = ctrl_rows[:0]
    else:
        us, cs = np.unique(iv, return_counts=True)
        keep = [(g, n) for g, n in zip(us, cs)
                if g not in (CTRL_LABEL, "excluded") and n >= nmin]
        envs = [(g, np.where(iv == g)[0], gidx.get(g)) for g, _ in keep]
        ref_pool = ctrl_rows

    if len(envs) < 4:
        return dict(step0=step0, nmin=nmin, d=d, n_envs=len(envs),
                    aborted="fewer_than_4_envs")

    within_c, within_m, between_c, between_m = [], [], [], []
    for g, rows, drop in envs:
        rows = rng.choice(rows, nmin, replace=False)
        drops = {drop} if drop is not None else set()
        Z = project(X[rows], mu, W, drops)
        Za, Zb = Z[:half], Z[half:nmin]
        within_c.append(np.abs(offdiag(coefs(Za)) - offdiag(coefs(Zb))))
        within_m.append(np.linalg.norm(Za.mean(0) - Zb.mean(0)))

        ref = rng.choice(ref_pool if len(ref_pool) else rows, half, replace=False)
        Zr = project(X[ref], mu, W, drops)
        between_c.append(np.abs(offdiag(coefs(Za)) - offdiag(coefs(Zr))))
        between_m.append(np.linalg.norm(Za.mean(0) - Zr.mean(0)))

    pair_c, pair_m = [], []
    for _ in range(min(N_PAIRS, len(envs) * (len(envs) - 1) // 2)):
        i, j = rng.choice(len(envs), 2, replace=False)
        (ga, ra, da), (gb, rb, db) = envs[i], envs[j]
        drops = {x for x in (da, db) if x is not None}
        Za = project(X[rng.choice(ra, half, replace=False)], mu, W, drops)
        Zb = project(X[rng.choice(rb, half, replace=False)], mu, W, drops)
        pair_c.append(np.abs(offdiag(coefs(Za)) - offdiag(coefs(Zb))))
        pair_m.append(np.linalg.norm(Za.mean(0) - Zb.mean(0)))

    wc = np.concatenate(within_c)
    bc = np.concatenate(between_c)
    pc = np.concatenate(pair_c) if pair_c else np.array([np.nan])
    thr = np.percentile(wc, 95)

    return dict(
        step0=step0, nmin=nmin, d=d, n_envs=len(envs), n_fit=half,
        coef_ratio_vs_ctrl=float(np.median(bc) / np.median(wc)),
        coef_ratio_pairs=float(np.median(pc) / np.median(wc)),
        shift_frac_vs_ctrl=float((bc > thr).mean()),
        shift_frac_pairs=float((pc > thr).mean()),
        shift_frac_within=float((wc > thr).mean()),
        mean_ratio_vs_ctrl=float(np.median(between_m) / np.median(within_m)),
        mean_ratio_pairs=float(np.median(pair_m) / np.median(within_m)),
    )


def spectrum_run(X, iv, vn, nmin, d):
    """Mirror of 04_spectrum.py run(): identical arithmetic, Norman as input."""
    rng = np.random.default_rng(SEED)
    gidx = {g: i for i, g in enumerate(vn)}
    ctrl = np.where(iv == CTRL_LABEL)[0]
    mu = X[ctrl].mean(0)
    _, _, Vt = np.linalg.svd(X[ctrl] - mu, full_matrices=False)
    W = Vt[:d].T

    def proj(rows, drop):
        Wm = W.copy()
        if drop is not None:
            Wm[drop, :] = 0.0
        return (X[rows] - mu) @ Wm

    us, cs = np.unique(iv, return_counts=True)
    envs = [(g, np.where(iv == g)[0], gidx.get(g))
            for g, n in zip(us, cs)
            if g not in (CTRL_LABEL, "excluded") and n >= nmin]

    cref = rng.permutation(ctrl)
    ref, rest = cref[:len(cref) // 2], cref[len(cref) // 2:]
    ref_mu = proj(ref, None).mean(0)

    M = np.array([proj(rng.choice(r, nmin, replace=False), t).mean(0) - ref_mu
                  for _, r, t in envs])
    k = len(M)
    N = np.array([proj(rng.choice(rest, nmin, replace=False), None).mean(0) - ref_mu
                  for _ in range(k)])

    s_sig = np.linalg.svd(M, compute_uv=False) / np.sqrt(len(M))
    s_noi = np.linalg.svd(N, compute_uv=False) / np.sqrt(len(N))
    n_cmp = min(len(s_sig), len(s_noi))
    ratio = s_sig[:n_cmp] / s_noi[:n_cmp]

    return dict(
        n_envs=len(envs), n_null=k, d=d, nmin=nmin,
        sing_signal=s_sig.tolist(),
        sing_noise=s_noi.tolist(),
        sing_ratio=ratio.tolist(),
        n_dims_above_noise=int((ratio > 1.0).sum()),
        n_dims_above_2x=int((ratio > 2.0).sum()),
        participation_ratio=float(s_sig.sum() ** 2 / (s_sig ** 2).sum()),
        s2_over_s1=float(s_sig[1] / s_sig[0]),
    )


def atomic_write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.rename(tmp, path)


def main():
    screen_json = SCREEN_DIR / "norman.json"
    spectrum_json = SPECTRUM_DIR / "norman.json"
    if screen_json.exists() and spectrum_json.exists():
        print(f"[skip] {screen_json.name} and {spectrum_json.name} both exist",
              flush=True)
        return

    X, iv, vn, meta = load_norman()

    step0_runs, screen_runs = [], []
    print("\n### STEP-0 GATE (control pseudo-envs; expect mean_ratio ~1.00) ###",
          flush=True)
    for nmin in NMIN_SET:
        for d in D_SET_SCREEN:
            r = screen_run(X, iv, vn, nmin, d, step0=True)
            step0_runs.append(r)
            if "aborted" in r:
                print(f"  [step0 n={nmin} d={d}] ABORTED: {r['aborted']}",
                      flush=True)
            else:
                print(f"  [step0 n={nmin} d={d}] "
                      f"mean_ratio_vs_ctrl={r['mean_ratio_vs_ctrl']:.3f} "
                      f"pairs={r['mean_ratio_pairs']:.3f} n_envs={r['n_envs']}",
                      flush=True)

    print("\n### REAL SCREEN (Norman single-perturbation envs) ###", flush=True)
    for nmin in NMIN_SET:
        for d in D_SET_SCREEN:
            r = screen_run(X, iv, vn, nmin, d, step0=False)
            screen_runs.append(r)
            if "aborted" in r:
                print(f"  [n={nmin} d={d}] ABORTED: {r['aborted']}", flush=True)
            else:
                print(f"  [n={nmin} d={d}] "
                      f"mean_ratio_vs_ctrl={r['mean_ratio_vs_ctrl']:.3f} "
                      f"pairs={r['mean_ratio_pairs']:.3f} "
                      f"coef_vs_ctrl={r['coef_ratio_vs_ctrl']:.3f} "
                      f"n_envs={r['n_envs']}", flush=True)

    print("\n### EFFECTIVE DIMENSIONALITY (spectrum) ###", flush=True)
    spectrum_runs = []
    for d in D_SET_SPECTRUM:
        s = spectrum_run(X, iv, vn, 200, d)
        spectrum_runs.append(s)
        print(f"  [d={d}] n_dims_above_2x={s['n_dims_above_2x']}/{d}  "
              f"above_noise={s['n_dims_above_noise']}/{d}  "
              f"participation_ratio={s['participation_ratio']:.2f}  "
              f"s2/s1={s['s2_over_s1']:.3f}", flush=True)

    def _pick(step0_flag, nmin, d):
        for r in (step0_runs if step0_flag else screen_runs):
            if r.get("nmin") == nmin and r.get("d") == d:
                return r
        return None

    step0_ratios = [r["mean_ratio_pairs"] for r in step0_runs
                    if "mean_ratio_pairs" in r]
    step0_pass = all(0.9 <= x <= 1.1 for x in step0_ratios) if step0_ratios else False

    primary = _pick(False, 200, 10) or _pick(False, 200, 5) or {}

    screen_out = dict(
        dataset="norman",
        source_h5ad=str(NORMAN_H5AD),
        **meta,
        step0_gate=step0_runs,
        screen=screen_runs,
        summary=dict(
            step0_mean_ratios=step0_ratios,
            step0_pass=step0_pass,
            primary_nmin=200, primary_d=10,
            primary_mean_ratio_vs_ctrl=primary.get("mean_ratio_vs_ctrl"),
            primary_mean_ratio_pairs=primary.get("mean_ratio_pairs"),
            primary_coef_ratio_vs_ctrl=primary.get("coef_ratio_vs_ctrl"),
        ),
        reference=dict(
            causalbench_k562_mean_ratio_range=[2.0, 5.6],
            hcp_mean_ratio=1.0,
            norman_workable_threshold=2.0,
        ),
    )
    atomic_write_json(screen_json, screen_out)
    print(f"\n[write] {screen_json}", flush=True)

    primary_spec = next((s for s in spectrum_runs if s.get("d") == 50),
                        spectrum_runs[-1] if spectrum_runs else {})
    spectrum_out = dict(
        dataset="norman",
        source_h5ad=str(NORMAN_H5AD),
        **{k: v for k, v in meta.items() if k in
           ("n_cells_kept", "n_control_cells", "n_single_perturbation_genes",
            "target_column_drop_is_noop")},
        runs=spectrum_runs,
        summary=dict(
            primary_d=50,
            primary_n_dims_above_2x=primary_spec.get("n_dims_above_2x"),
            primary_n_dims_above_noise=primary_spec.get("n_dims_above_noise"),
            primary_participation_ratio=primary_spec.get("participation_ratio"),
        ),
        reference=dict(
            causalbench_k562_effective_dim="~15",
            hcp_effective_dim="low",
            norman_workable_range=[7, 15],
        ),
    )
    atomic_write_json(spectrum_json, spectrum_out)
    print(f"[write] {spectrum_json}", flush=True)

    print("\n" + "=" * 78, flush=True)
    print("REFERENCE COMPARISON (Norman vs CausalBench K562 vs HCP fMRI)",
          flush=True)
    print("=" * 78, flush=True)
    print(f"  {'metric':<32}{'CausalBench K562':>20}{'HCP fMRI':>12}"
          f"{'Norman':>12}")
    print(f"  {'mean_ratio_vs_ctrl':<32}{'2.0 - 5.6':>20}{'1.0':>12}"
          f"{screen_out['summary']['primary_mean_ratio_vs_ctrl']:>12.3f}")
    print(f"  {'effective dim (dims_above_2x)':<32}{'~15':>20}{'low':>12}"
          f"{spectrum_out['summary']['primary_n_dims_above_2x']!s:>12}")
    print(f"  {'step-0 gate':<32}{'0.96 - 1.05':>20}"
          f"{'passed':>12}"
          f"{'PASS' if step0_pass else 'FAIL':>12}")
    if not step0_pass:
        print(f"  step-0 ratios: {step0_ratios}", flush=True)


if __name__ == "__main__":
    main()
