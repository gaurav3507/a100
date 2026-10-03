"""NMIN ladder for every screened dataset: characterise the metric's
n-dependence and extract the slope diagnostic.

WHY THIS EXISTS
---------------
`EVIDENCE_PACK.md` 2.1 already records that `mean_ratio_pairs` is not
scale-free in NMIN (K562 filt0 goes 2.228 at n=100 to 4.270 at n=200). This
script confirms and QUANTIFIES that; it does not discover it.

The construction explains the behaviour analytically. Write the denominator
as a control split-half distance and the numerator as a between-environment
distance, both computed on NMIN/2 cells per mean:

    within_m  = || mean(Za) - mean(Zb) ||     Za, Zb halves of ONE env
    pair_m    = || mean(Za) - mean(Zb) ||     Za, Zb from DIFFERENT envs

With Sigma the per-cell latent covariance and Delta the true between-env
shift, taking expectations of the squared quantities:

    E[within_m^2] = 4 tr(Sigma) / NMIN
    E[pair_m^2]   = ||Delta||^2 + 4 tr(Sigma) / NMIN

so the ratio obeys

    ratio^2 = 1 + c * NMIN ,     c = ||Delta||^2 / (4 tr Sigma)          (*)

Three consequences, all tested here:

  1. SLOPE. d log(ratio) / d log(NMIN) tends to 0.5 when signal is present
     (c*NMIN >> 1) and to 0 when it is not (c*NMIN << 1). Intermediate SNR
     gives an intermediate slope, so the slope is a graded readout, not a
     binary one. The slope is the primary instrument because it is a
     statement about how the statistic MOVES, which no single-n level can
     provide.

  2. n-INVARIANT EFFECT SIZE. Rearranging (*),

         c_hat = (ratio^2 - 1) / NMIN

     is, under the model, constant across rungs. It is reported at every
     rung precisely so its constancy can be checked. Where c_hat holds
     steady the model fits and c_hat is a defensible scale-free effect size;
     where it drifts, the model is wrong for that dataset and that must be
     said rather than papered over. A fixed threshold on `ratio` is
     indefensible without stating NMIN; a threshold on c_hat would not be.

  3. SELECTION CONFOUND. Raising NMIN also changes WHICH perturbations
     survive the >= NMIN cell filter, biasing toward high-population targets.
     K562's observed 1.9x for a 2x NMIN change exceeds the sqrt(2) = 1.41
     that the pure n-effect predicts, which is what a selection effect looks
     like. Every dataset is therefore run TWICE per rung:

       native       env set recomputed at each rung (confounds n + selection)
       intersected  env set FIXED to those surviving at the dataset's highest
                    reachable rung, held constant across all rungs

     Only the intersected series isolates the n-effect. Slopes are fitted on
     the intersected series only.

ARITHMETIC PROVENANCE
---------------------
`fit_pca`, `project`, `coefs`, `offdiag`, `N_PAIRS`, `RIDGE` are imported from
`03_screen.py` via importlib, exactly as `40_screen_norman.py` does it.
`03_screen.py` is NOT modified. The screen kernel below is a transcription of
`03_screen.py:74-111` with three additions, none of which touch the
arithmetic: a configurable seed, an optional environment whitelist, and the
pairs-based step-0 gate computed inline (NOT patched into the JSON afterwards,
which was the `40b_fix_step0_gate.py` mistake).

SAMPLE-SIZE MATCHING
--------------------
Numerator and denominator both average NMIN/2 cells per mean, at every rung.
This is asserted at runtime in `screen_once` (see MATCHED_N assert). This is
the failure that has bitten four times; it is checked, not assumed.

PRIMARY METRIC
--------------
`mean_ratio_pairs` everywhere. `mean_ratio_vs_ctrl` is retained in the
per-run records as supplementary and never enters a summary table.

Usage (A100):
    python causalbench/scripts/48_nmin_ladder_all.py --clean
    python causalbench/scripts/48_nmin_ladder_all.py --datasets k562,rpe1,norman
    python causalbench/scripts/48_nmin_ladder_all.py --datasets frangieh --hvg 5000
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
CAUSALBENCH = Path("/workspace/meridian-identifiability/causalbench")
OUT_DIR = SCRIPTS_DIR.parent / "results/nmin_ladder"
OUT_JSON = OUT_DIR / "ladder.json"

RUNGS = (50, 100, 200, 400)
SEEDS = (0, 1, 2, 3, 4)
D = 10
FILT = False                      # filt0 is the headline configuration
GATE_BAND = (0.9, 1.1)

ALL_DATASETS = ("k562", "rpe1", "norman",
                "frangieh:Co-culture", "frangieh:Control", "frangieh:IFNγ")


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Byte-identical arithmetic: helpers come from 03_screen.py, unmodified.
SCREEN_MOD = _load_module(CAUSALBENCH / "scripts/03_screen.py", "_screen03")
fit_pca = SCREEN_MOD.fit_pca
project = SCREEN_MOD.project
coefs = SCREEN_MOD.coefs
offdiag = SCREEN_MOD.offdiag
N_PAIRS = SCREEN_MOD.N_PAIRS
CTRL = SCREEN_MOD.CTRL


# ============================================================ screen kernel
def surviving_targets(iv, nmin):
    us, cs = np.unique(iv, return_counts=True)
    return {g for g, n in zip(us, cs)
            if g not in (CTRL, "excluded") and n >= nmin}


def screen_once(X, iv, vn, nmin, d, seed, step0=False, env_whitelist=None,
                pca=None):
    """Transcription of 03_screen.py:74-111 with a configurable seed and an
    optional fixed environment set. `pca` caches (mu, W) across calls; it is
    a deterministic function of (X[ctrl_rows], d) so caching cannot alter
    any number.
    """
    rng = np.random.default_rng(seed)
    half = nmin // 2
    gidx = {g: i for i, g in enumerate(vn)}
    ctrl_rows = np.where(iv == CTRL)[0]

    if len(ctrl_rows) < nmin:
        return dict(aborted="fewer_control_cells_than_nmin",
                    n_control_cells=int(len(ctrl_rows)),
                    nmin=int(nmin), d=int(d), seed=int(seed), step0=bool(step0))

    mu, W = pca if pca is not None else fit_pca(X[ctrl_rows], d)

    if step0:
        cr = ctrl_rows.copy()
        rng.shuffle(cr)
        k = len(cr) // nmin
        envs = [(f"ctrl{i}", cr[i * nmin:(i + 1) * nmin], None)
                for i in range(k)]
        ref_pool = cr[:0]
    else:
        us, cs = np.unique(iv, return_counts=True)
        keep = [(g, n) for g, n in zip(us, cs)
                if g not in (CTRL, "excluded") and n >= nmin]
        if env_whitelist is not None:
            keep = [(g, n) for g, n in keep if g in env_whitelist]
        envs = [(g, np.where(iv == g)[0], gidx.get(g)) for g, _ in keep]
        ref_pool = ctrl_rows

    if len(envs) < 4:
        return dict(aborted="fewer_than_4_envs", n_envs=int(len(envs)),
                    nmin=int(nmin), d=int(d), seed=int(seed), step0=bool(step0))

    within_c, within_m, between_c, between_m = [], [], [], []
    n_per_mean_seen = set()
    for g, rows, drop in envs:
        rows = rng.choice(rows, nmin, replace=False)
        drops = {drop} if drop is not None else set()
        Z = project(X[rows], mu, W, drops)
        Za, Zb = Z[:half], Z[half:nmin]
        n_per_mean_seen.add(len(Za))
        n_per_mean_seen.add(len(Zb))
        within_c.append(np.abs(offdiag(coefs(Za)) - offdiag(coefs(Zb))))
        within_m.append(np.linalg.norm(Za.mean(0) - Zb.mean(0)))

        ref = rng.choice(ref_pool if len(ref_pool) else rows, half,
                          replace=False)
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
        n_per_mean_seen.add(len(Za))
        n_per_mean_seen.add(len(Zb))
        pair_c.append(np.abs(offdiag(coefs(Za)) - offdiag(coefs(Zb))))
        pair_m.append(np.linalg.norm(Za.mean(0) - Zb.mean(0)))

    # MATCHED_N: every mean entering the numerator and every mean entering
    # the denominator is an average over exactly NMIN/2 cells. Operational
    # lesson 2; asserted rather than assumed.
    assert n_per_mean_seen == {half}, (
        f"sample sizes not matched at nmin={nmin}: means over "
        f"{sorted(n_per_mean_seen)} cells, expected all == {half}")

    wc = np.concatenate(within_c)
    bc = np.concatenate(between_c)
    pc = np.concatenate(pair_c) if pair_c else np.array([np.nan])
    thr = np.percentile(wc, 95)

    ratio = float(np.median(pair_m) / np.median(within_m))
    # n-invariant effect size implied by ratio^2 = 1 + c*NMIN. Constant
    # across rungs iff the model holds; reported per rung so that can be
    # checked rather than assumed. Clipped at 0 because ratio < 1 (pure
    # noise, unlucky draw) would otherwise give a negative c_hat.
    c_hat = float(max(ratio * ratio - 1.0, 0.0) / nmin)

    return dict(
        step0=bool(step0), nmin=int(nmin), d=int(d), seed=int(seed),
        n_envs=int(len(envs)), n_per_mean=int(half),
        n_control_cells=int(len(ctrl_rows)),
        env_set="intersected" if env_whitelist is not None else "native",
        # PRIMARY
        mean_ratio_pairs=ratio,
        c_hat=c_hat,
        coef_ratio_pairs=float(np.median(pc) / np.median(wc)),
        shift_frac_pairs=float((pc > thr).mean()),
        median_within_m=float(np.median(within_m)),
        median_pair_m=float(np.median(pair_m)),
        # supplementary only; never enters a summary table
        mean_ratio_vs_ctrl=float(np.median(between_m) / np.median(within_m)),
        coef_ratio_vs_ctrl=float(np.median(bc) / np.median(wc)),
    )


# ============================================================ data loading
def load_dataset(name, hvg=None, _frangieh_cache={}):
    """Yield (X, iv, vn) for a dataset key. Frangieh is loaded once and
    sliced per arm; the cache keeps the parsed matrix alive across the three
    arm keys within one process.
    """
    if name in ("k562", "rpe1"):
        X, iv, vn = SCREEN_MOD.load(name, FILT)
        return X, iv, vn

    if name == "norman":
        NORMAN = _load_module(SCRIPTS_DIR / "40_screen_norman.py", "_norman40")
        X, iv, vn, _ = NORMAN.load_norman()
        return X, iv, vn

    if name.startswith("frangieh:"):
        arm = name.split(":", 1)[1]
        if "parsed" not in _frangieh_cache:
            FR = _load_module(SCRIPTS_DIR / "41_screen_frangieh.py",
                              "_frangieh41")
            md, meta = FR.load_metadata()
            X, vn, cell_order = FR.load_expression(
                FR.EXPR_CSV, list(md["NAME"].astype(str)), hvg=hvg)
            md = md.set_index(md["NAME"].astype(str)).reindex(cell_order)
            ok = md["iv"].notna().to_numpy()
            X, md = X[ok], md.loc[ok]
            _frangieh_cache["parsed"] = (
                X, md["iv"].to_numpy().astype(object),
                md["condition"].to_numpy().astype(str), vn)
        X, iv_all, cond_all, vn = _frangieh_cache["parsed"]
        sel = cond_all == arm
        if sel.sum() == 0:
            raise SystemExit(f"[fatal] no cells for arm {arm!r}")
        return np.ascontiguousarray(X[sel]), iv_all[sel], vn

    raise SystemExit(f"[fatal] unknown dataset {name!r}")


# ============================================================ slope fitting
def fit_loglog_slope(nmins, ratios):
    """Least-squares slope of log(ratio) on log(NMIN). Returns None if fewer
    than 2 usable points or any ratio is non-positive.
    """
    pts = [(n, r) for n, r in zip(nmins, ratios)
           if r is not None and r > 0]
    if len(pts) < 2:
        return None
    x = np.log(np.array([p[0] for p in pts], dtype=float))
    y = np.log(np.array([p[1] for p in pts], dtype=float))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = A @ np.array([slope, intercept])
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(slope=float(slope), intercept=float(intercept),
                r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
                n_points=len(pts))


def atomic_write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.rename(tmp, path)


# ==================================================================== ORACLE
def oracle(rungs=(50, 100, 200, 400), seed=0, n_env=40, n_cells=500,
           p=60, d=10):
    """Known-answer test for the SLOPE diagnostic itself.

    Synthesises environments whose true between-env shift magnitude is set
    by `shift_scale`, runs the real ladder machinery, and asserts the fitted
    log-log slope lands where the model (*) says it must:

        shift_scale = 0     -> no signal      -> slope ~ 0    (FLAT)
        shift_scale = 0.5   -> strong signal  -> slope ~ 0.5  (GROWS)

    Without this the slope readout is unvalidated and a flat result cannot be
    distinguished from a broken ladder. Verified locally before commit:
    0.00 -> -0.055, 0.02 -> -0.005, 0.05 -> 0.056, 0.15 -> 0.270, 0.50 -> 0.454.
    """
    scenarios = [
        ("no_signal",     0.0,  (-0.15, 0.15)),
        ("weak_signal",   0.05, (-0.10, 0.25)),
        ("strong_signal", 0.50, (0.35, 0.60)),
    ]
    results = {}
    for label, shift_scale, band in scenarios:
        ratios, c_hats = [], []
        for nmin in rungs:
            rng = np.random.default_rng(seed + 7919)
            ctrl = rng.normal(0, 1, (3000, p))
            mu, W = fit_pca(ctrl, d)
            shifts = (rng.normal(0, shift_scale, (n_env, p))
                      if shift_scale > 0 else np.zeros((n_env, p)))
            envs = [rng.normal(0, 1, (n_cells, p)) + shifts[i]
                    for i in range(n_env)]
            half = nmin // 2
            within = []
            for E in envs:
                rows = rng.choice(len(E), nmin, replace=False)
                z = project(E[rows], mu, W, set())
                within.append(np.linalg.norm(
                    z[:half].mean(0) - z[half:nmin].mean(0)))
            pair = []
            for _ in range(N_PAIRS):
                i, j = rng.choice(n_env, 2, replace=False)
                za = project(envs[i][rng.choice(n_cells, half, replace=False)],
                             mu, W, set())
                zb = project(envs[j][rng.choice(n_cells, half, replace=False)],
                             mu, W, set())
                pair.append(np.linalg.norm(za.mean(0) - zb.mean(0)))
            r = float(np.median(pair) / np.median(within))
            ratios.append(r)
            c_hats.append(max(r * r - 1.0, 0.0) / nmin)
        fit = fit_loglog_slope(list(rungs), ratios)
        s = fit["slope"] if fit else None
        ok = bool(s is not None and band[0] <= s <= band[1])
        results[label] = dict(shift_scale=shift_scale,
                              ratios=[float(x) for x in ratios],
                              c_hat=[float(x) for x in c_hats],
                              slope=s, expected_band=list(band), ok=ok)
    passed = all(v["ok"] for v in results.values())
    return passed, dict(
        scenarios=results,
        note=("Validates that the log-log slope separates signal from noise "
              "on data with a KNOWN between-environment shift magnitude, "
              "using the same ladder arithmetic as the real run."))


# ==================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true",
                    help="rm -rf the results dir before running")
    ap.add_argument("--datasets", default=",".join(ALL_DATASETS),
                    help="comma-separated; 'frangieh' expands to all 3 arms")
    ap.add_argument("--rungs", default=",".join(str(r) for r in RUNGS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--d", type=int, default=D)
    ap.add_argument("--hvg", type=int, default=None,
                    help="Frangieh only: cap to N most variable genes")
    ap.add_argument("--oracle-only", action="store_true",
                    help="run only the slope oracle; touch no real data")
    a = ap.parse_args()

    if a.clean and OUT_DIR.exists():
        print(f"[clean] rm -rf {OUT_DIR}", flush=True)
        shutil.rmtree(OUT_DIR)
    if OUT_JSON.exists() and not a.oracle_only:
        print(f"[skip] {OUT_JSON} exists; --clean to overwrite", flush=True)
        return

    # ---- ORACLE FIRST: an unvalidated slope cannot distinguish a flat
    # dataset from a broken ladder.
    print("[oracle] slope diagnostic on known-answer synthetic", flush=True)
    o_pass, o_meta = oracle()
    for label, r in o_meta["scenarios"].items():
        print(f"[oracle] {label:<15s} shift={r['shift_scale']:<5} "
              f"slope={r['slope']:.4f} "
              f"expect=[{r['expected_band'][0]:.2f},"
              f"{r['expected_band'][1]:.2f}] -> "
              f"{'OK' if r['ok'] else 'FAIL'}", flush=True)
    if not o_pass:
        print("[oracle] FAIL -- slope readout unvalidated; refusing to run",
              flush=True)
        atomic_write_json(OUT_JSON, dict(oracle=o_meta,
                                          aborted="oracle_failed"))
        sys.exit(1)
    print("[oracle] PASS", flush=True)
    if a.oracle_only:
        print("[done] --oracle-only requested", flush=True)
        return

    rungs = tuple(int(r) for r in a.rungs.split(","))
    seeds = tuple(int(s) for s in a.seeds.split(","))
    wanted = []
    for tok in a.datasets.split(","):
        tok = tok.strip()
        if tok == "frangieh":
            wanted += [d for d in ALL_DATASETS if d.startswith("frangieh:")]
        elif tok:
            wanted.append(tok)

    print(f"[config] rungs={rungs} seeds={seeds} d={a.d} filt={FILT}",
          flush=True)
    print(f"[config] datasets={wanted}", flush=True)

    out = dict(
        config=dict(rungs=list(rungs), seeds=list(seeds), d=int(a.d),
                    filt=bool(FILT), hvg=a.hvg,
                    primary_metric="mean_ratio_pairs",
                    gate_band=list(GATE_BAND),
                    model="ratio^2 = 1 + c*NMIN; c_hat=(ratio^2-1)/NMIN",
                    arithmetic_source="03_screen.py via importlib (unmodified)"),
        oracle=dict(passed=o_pass, **o_meta),
        datasets={},
    )

    for ds in wanted:
        print(f"\n{'=' * 78}\n{ds}\n{'=' * 78}", flush=True)
        try:
            X, iv, vn = load_dataset(ds, hvg=a.hvg)
        except SystemExit as e:
            print(f"  [fatal] {e}", flush=True)
            out["datasets"][ds] = dict(aborted=str(e))
            continue
        print(f"  loaded X={X.shape}  n_ctrl="
              f"{int((iv == CTRL).sum())}", flush=True)

        # Cache the control-cell PCA once per dataset: deterministic in
        # (X[ctrl_rows], d), so this cannot change any number.
        ctrl_rows = np.where(iv == CTRL)[0]
        pca = fit_pca(X[ctrl_rows], a.d) if len(ctrl_rows) else None

        # Which rungs are reachable, and the intersected env set.
        reach, dropped = [], {}
        for r in rungs:
            n_env = len(surviving_targets(iv, r))
            n_ctrl_env = len(ctrl_rows) // r
            if n_env >= 4 and n_ctrl_env >= 4:
                reach.append(r)
            else:
                dropped[r] = (f"n_envs={n_env} (need>=4), "
                              f"n_ctrl_pseudo_envs={n_ctrl_env} (need>=4)")
        if not reach:
            print(f"  [abort] no reachable rungs: {dropped}", flush=True)
            out["datasets"][ds] = dict(aborted="no_reachable_rungs",
                                        dropped_rungs=dropped)
            continue
        top = max(reach)
        whitelist = surviving_targets(iv, top)
        print(f"  reachable rungs: {reach}   dropped: {dropped or 'none'}",
              flush=True)
        print(f"  intersected env set = targets with >= {top} cells "
              f"({len(whitelist)} envs), held fixed across all rungs",
              flush=True)

        rows = []
        for r in reach:
            for seed in seeds:
                g = screen_once(X, iv, vn, r, a.d, seed, step0=True, pca=pca)
                nat = screen_once(X, iv, vn, r, a.d, seed, step0=False,
                                   env_whitelist=None, pca=pca)
                itx = screen_once(X, iv, vn, r, a.d, seed, step0=False,
                                   env_whitelist=whitelist, pca=pca)
                rows.append(dict(nmin=r, seed=seed, gate=g,
                                  native=nat, intersected=itx))
            # per-rung console line, seed 0
            r0 = next(x for x in rows if x["nmin"] == r
                      and x["seed"] == seeds[0])
            def _f(rec, k="mean_ratio_pairs"):
                return (f"{rec[k]:.3f}" if k in rec
                        else f"ABORT({rec.get('aborted','?')})")
            print(f"  NMIN={r:<4} gate={_f(r0['gate'])}  "
                  f"native={_f(r0['native'])} "
                  f"(envs={r0['native'].get('n_envs','-')})  "
                  f"intersected={_f(r0['intersected'])} "
                  f"(envs={r0['intersected'].get('n_envs','-')})", flush=True)

        # Slope on the INTERSECTED series, per seed, then aggregate.
        per_seed_slopes = []
        for seed in seeds:
            ns, rs = [], []
            for r in reach:
                rec = next((x["intersected"] for x in rows
                            if x["nmin"] == r and x["seed"] == seed), None)
                if rec and "mean_ratio_pairs" in rec:
                    ns.append(r)
                    rs.append(rec["mean_ratio_pairs"])
            fit = fit_loglog_slope(ns, rs)
            if fit:
                per_seed_slopes.append(fit["slope"])
        slope_mean = float(np.mean(per_seed_slopes)) if per_seed_slopes else None
        slope_sd = float(np.std(per_seed_slopes)) if per_seed_slopes else None
        slope_ci = None
        if per_seed_slopes and len(per_seed_slopes) > 1:
            se = slope_sd / np.sqrt(len(per_seed_slopes))
            slope_ci = [float(slope_mean - 1.96 * se),
                        float(slope_mean + 1.96 * se)]

        verdict = "UNKNOWN"
        if slope_mean is not None:
            if slope_mean >= 0.30:
                verdict = "GROWS (signal present)"
            elif slope_mean <= 0.15:
                verdict = "FLAT (noise-dominated)"
            else:
                verdict = "INTERMEDIATE"

        out["datasets"][ds] = dict(
            n_cells=int(X.shape[0]), n_genes=int(X.shape[1]),
            n_control_cells=int(len(ctrl_rows)),
            reachable_rungs=reach, dropped_rungs=dropped,
            intersected_env_count=len(whitelist),
            intersected_defined_at_nmin=top,
            rows=rows,
            slope_intersected=dict(per_seed=per_seed_slopes,
                                    mean=slope_mean, sd=slope_sd,
                                    ci95=slope_ci, verdict=verdict),
        )
        print(f"  log-log slope (intersected) = "
              f"{slope_mean if slope_mean is None else round(slope_mean, 4)}"
              f" +/- {slope_sd if slope_sd is None else round(slope_sd, 4)}"
              f"   -> {verdict}", flush=True)

        del X, iv
        if not ds.startswith("frangieh:"):
            pca = None

    atomic_write_json(OUT_JSON, out)
    print(f"\n[write] {OUT_JSON}", flush=True)
    _summary(out, rungs, seeds)


# ================================================================= summary
def _agg(rows, rung, key, field="mean_ratio_pairs"):
    vals = [r[key][field] for r in rows
            if r["nmin"] == rung and field in r[key]]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def _summary(out, rungs, seeds):
    print("\n" + "=" * 78, flush=True)
    print("NMIN LADDER  --  primary metric mean_ratio_pairs", flush=True)
    print("=" * 78, flush=True)

    print("\nA. INTERSECTED env set (isolates the n-effect); mean +/- sd "
          "over seeds")
    hdr = f"  {'dataset':<22}" + "".join(f"{'n=' + str(r):>16}" for r in rungs)
    print(hdr)
    for ds, D_ in out["datasets"].items():
        if "rows" not in D_:
            print(f"  {ds:<22}  ABORTED ({D_.get('aborted')})")
            continue
        line = f"  {ds:<22}"
        for r in rungs:
            m, s = _agg(D_["rows"], r, "intersected")
            line += (f"{m:>10.3f}+-{s:<4.3f}" if m is not None
                     else f"{'-':>16}")
        print(line)

    print("\nB. NATIVE env set (confounds n with target selection)")
    print(hdr)
    for ds, D_ in out["datasets"].items():
        if "rows" not in D_:
            continue
        line = f"  {ds:<22}"
        for r in rungs:
            m, s = _agg(D_["rows"], r, "native")
            line += (f"{m:>10.3f}+-{s:<4.3f}" if m is not None
                     else f"{'-':>16}")
        print(line)

    print("\nC. STEP-0 GATE at every rung (must sit in "
          f"[{GATE_BAND[0]}, {GATE_BAND[1]}])")
    print(hdr)
    gate_fail = []
    for ds, D_ in out["datasets"].items():
        if "rows" not in D_:
            continue
        line = f"  {ds:<22}"
        for r in rungs:
            m, s = _agg(D_["rows"], r, "gate")
            if m is None:
                line += f"{'-':>16}"
            else:
                bad = not (GATE_BAND[0] <= m <= GATE_BAND[1])
                if bad:
                    gate_fail.append((ds, r, m))
                line += f"{m:>9.3f}{'!' if bad else ' '}+-{s:<4.3f}"
        print(line)
    if gate_fail:
        print("\n  GATE FAILURES (marked ! above):")
        for ds, r, m in gate_fail:
            print(f"    {ds:<22} NMIN={r:<5} gate={m:.4f} OUTSIDE "
                  f"[{GATE_BAND[0]}, {GATE_BAND[1]}]")
        print("  A gate that drifts with NMIN means the null is not matched "
              "and every level above it is suspect.")
    else:
        print("\n  all gates within band at all rungs")

    print("\nD. n-INVARIANT EFFECT SIZE c_hat = (ratio^2 - 1)/NMIN, "
          "intersected")
    print("   constant across rungs iff ratio^2 = 1 + c*NMIN holds")
    print(hdr)
    for ds, D_ in out["datasets"].items():
        if "rows" not in D_:
            continue
        line = f"  {ds:<22}"
        for r in rungs:
            m, s = _agg(D_["rows"], r, "intersected", field="c_hat")
            line += (f"{m:>16.3e}" if m is not None else f"{'-':>16}")
        print(line)

    print("\nE. LOG-LOG SLOPE on the intersected series")
    print(f"  {'dataset':<22}{'slope':>10}{'sd':>9}{'95% CI':>22}"
          f"   verdict")
    for ds, D_ in out["datasets"].items():
        sl = D_.get("slope_intersected")
        if not sl or sl.get("mean") is None:
            print(f"  {ds:<22}{'-':>10}")
            continue
        ci = sl.get("ci95")
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "-"
        print(f"  {ds:<22}{sl['mean']:>10.4f}{sl['sd']:>9.4f}{ci_s:>22}"
              f"   {sl['verdict']}")
    print("\n  reference: 0.5 = pure sqrt(NMIN) growth (signal present)")
    print("             0.0 = flat (numerator noise-dominated)")

    grows = [d for d, D_ in out["datasets"].items()
             if (D_.get("slope_intersected") or {}).get("verdict", "")
             .startswith("GROWS")]
    flat = [d for d, D_ in out["datasets"].items()
            if (D_.get("slope_intersected") or {}).get("verdict", "")
            .startswith("FLAT")]
    mid = [d for d, D_ in out["datasets"].items()
           if (D_.get("slope_intersected") or {}).get("verdict", "")
           == "INTERMEDIATE"]
    print(f"\n  GROWS        : {grows or 'none'}")
    print(f"  INTERMEDIATE : {mid or 'none'}")
    print(f"  FLAT         : {flat or 'none'}")


if __name__ == "__main__":
    main()
