"""Synthetic s* vs d sweep: where does the ranktest start detecting nonlinearity?

WHAT THIS MEASURES. The detection threshold s* is the smallest s at which the
rejection rate crosses out of level control, reported as a function of the
latent dimension d.

TWO REGIMES, DEFAULT LINEAR (--mixing). The headline runs under LINEAR mixing,
the regime where the LFC test holds its level (gate0, gate1). There s does NOT
enter the mixing: it is a grid label only, and the artefact records
s_enters_mixing = false, so the flat curve reads as validity rather than a
swept threshold. Under NONLINEAR mixing (--mixing nonlinear) s is gate2's
leaky-ReLU nonlinearity scale and the mixing manufactures apparent rank
structure, but that is the blocked Phase-B regime where rejection is confounded
with mixing curvature, so an s* measured there is contaminated. Both regimes
are kept; the mixing is recorded top-level in every artefact so no reader can
mistake one for the other.

NOTHING IS REIMPLEMENTED. The statistic, the projection and the standardiser are
imported from 80_ranktest_core.py; the simulator and the step-0 oracle gate are
imported from 81_ranktest_oracle.py. Both are loaded through importlib in the
same pattern 40_screen_norman.py uses for 03_screen.py, so the arithmetic is
byte-identical to the committed gates. A divergent second copy of either is the
defect this lane has already paid for once, so there is not one here. Neither 80
nor 81 is modified, and the md5 of 80 is recorded in every artefact.

THE DECISION RULE IS 80'S, NOT INVENTED. Each draw is decided by
reject_rank2_cf from 80.rank_diagnostic, which is the LFC test at r = 2 and
level alpha: reject when the statistic exceeds the (1 - alpha) bootstrap
critical value. See _lfc_rank_test and rank_diagnostic in 80.

THE s* CROSSING LEVEL IS 81'S, NOT INVENTED. The rejection RATE is bracketed
against the same over-rejection boundary gate0 uses for its verdict,
alpha + 2 Monte-Carlo standard errors at this draw count. A rate above it is
what gate0 already calls loss of level control. The raw per-cell rates are
written too, so the brackets can be recomputed at any other level without
rerunning.

COARSE GRID, HONESTLY REPORTED. Five s points is a coarse grid. s* is reported
as the bracketing interval [s_lo, s_hi] it falls in, left- or right-censored
when the crossing sits outside the grid. No point estimate is interpolated and
no law is fitted through five brackets. Any trend number is labelled a coarse
trend estimate and no R^2 is reported.

PURELY SYNTHETIC. This reads and writes nothing under results/descriptives or
results/preprocessing. Its only output is results/decay/.

Usage (A100). Long-running batch job; launch detached:

    nohup python -u scripts/82_ranktest_decay.py > logs/decay.log 2>&1 &

    python scripts/82_ranktest_decay.py --smoke                  # linear (default)
    python scripts/82_ranktest_decay.py --smoke --mixing linear
    python scripts/82_ranktest_decay.py --mixing nonlinear       # Phase-B regime

No em dashes anywhere in this file, by lane style rule.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_DIR = REPO / "results" / "decay"
LOCKFILE = REPO / "logs" / "decay.lock"

CORE_PY = HERE / "80_ranktest_core.py"
ORACLE_PY = HERE / "81_ranktest_oracle.py"
RESULTS_IO_PY = HERE / "84_results_io.py"

# The grid, fixed by the handoff and not revised after seeing any number.
D_LATENT_GRID = (5, 10, 20, 40, 80)
S_GRID = (0.005, 0.01, 0.02, 0.05, 0.1)
N_E = 2000                      # per environment, mirrors gate2's n
DRAWS_PER_CELL = 400
SCALINGS = ("raw", "standardise")   # the two the lane uses; standardise is 80's
MIXINGS = ("linear", "nonlinear")   # default linear: the valid regime
D_OBS = 200                     # observed dimension D, mirrors gate2

# --- iv-scale sweep (--iv-scale-sweep): a real swept axis in the VALID regime.
# Intervention magnitude replaces the (linear-inert) nonlinearity scale s. The
# grid is deliberately reduced so the run is affordable.
IV_SCALE_GRID = (0.5, 1.0, 2.0, 4.0)
IV_D_LATENT_GRID = (5, 20, 80)
IV_SCALINGS = ("standardise",)
IVSCALE_OUT_DIR = REPO / "results" / "decay_ivscale"

# --- k sweep (--k-sweep): vary the NUMBER of intervened nodes at fixed
# magnitude, linear regime. k=1,2 sit inside H0(2) (rank <= 2), k>=3 exceeds
# it, so this axis crosses the null boundary and has a real detection
# threshold; gate1's own power regime is k>=3 (81:321-322). Node selection
# matches gate1 (rng_iv.choice size=k). Magnitude is FIXED at
# K_IV_SCALE_FIXED, a deliberate deviation from gate1's random U(2,4) draw so
# that k is the ONLY thing varying; recorded in the artefact.
K_GRID = (1, 2, 3, 5)
K_D_LATENT_GRID = (5, 20, 80)
K_SCALINGS = ("standardise",)
K_IV_SCALE_FIXED = 2.0
KSWEEP_OUT_DIR = REPO / "results" / "decay_ksweep"

# Recorded so the artefact names the rules it used rather than implying them.
DECISION_RULE_ID = ("reject_rank2_cf from 80_ranktest_core.rank_diagnostic: "
                    "LFC test of H0 rank<=2 at level alpha, reject when the "
                    "statistic exceeds the (1-alpha) bootstrap critical value "
                    "(_lfc_rank_test)")
GATE_ID = ("81_ranktest_oracle.gate0: level-control oracle, true rank 0 under "
           "linear mixing, PASS when no cell's pooled reject rate exceeds "
           "alpha + 2 Monte-Carlo SE")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    real = os.makedirs
    os.makedirs = lambda *a, **k: None     # no directory creation at import
    try:
        spec.loader.exec_module(mod)
    finally:
        os.makedirs = real
    return mod


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H-%M-%SZ")


def crossing_level(n_draws):
    """gate0's over-rejection boundary at this draw count. Not invented here."""
    alpha = ORACLE.ALPHA
    se = (alpha * (1.0 - alpha) / n_draws) ** 0.5
    return alpha + 2.0 * se


# =============================================================== one draw
def one_draw(d_latent, D, n, s, scaling, mixing, main_seed, iv_seed, iv_scale=None, k_nodes=1):
    """One synthetic realisation, decided by 80's reject_rank2_cf.

    mixing="linear"    : the proper full-rank Gaussian linear map gate0 and
                         gate1 use, X = mix_linear(Z, A) with
                         A = standard_normal((D, d_latent)), one A reused for
                         the control and the environment. This is the regime
                         where the test holds its level. s does NOT enter the
                         mixing here; it is carried only as a grid label, and
                         the artefact records s_enters_mixing = false so the
                         flat curve is never mistaken for a swept one.
    mixing="nonlinear" : gate2's 2-layer leaky-ReLU mix_mlp at scale s, the
                         blocked Phase-B regime, unchanged. s IS the
                         nonlinearity scale.

    The two generators mirror gate2's split of the mixing stream from the
    intervention stream. Nothing here is a new statistic or a new simulator:
    make_scm, sample_latent, mix_linear, mix_mlp, add_obs_noise and split_pools
    are 81's; standardise, fit_pca, project, null_band_from_pool and
    rank_diagnostic are 80's.

    iv_scale (default None) is forwarded ONLY to the hard-intervention
    environment draw. None keeps 81's random U(2,4) magnitude byte-for-byte, so
    the default and --mixing paths are unchanged; a float fixes the intervened
    node's variance, which is what --iv-scale-sweep sweeps. The control draw
    takes no intervention and never sees it.
    """
    d = min(10, d_latent)                  # projection dimension, as in gate2
    n_total = 10 * n
    basis_i, pool_i, env_i = ORACLE.split_pools(n_total, n)

    rng = np.random.default_rng(main_seed)
    B, nv, _ = ORACLE.make_scm(d_latent, rng)

    if mixing == "linear":
        # gate0/gate1's proper linear map. Full-rank Gaussian, not degenerate,
        # one A shared by control and environment. s is not consumed.
        A = rng.standard_normal((D, d_latent))

        def mix(Z):
            return ORACLE.mix_linear(Z, A)
    elif mixing == "nonlinear":
        H = 4 * d_latent
        A1 = rng.standard_normal((H, d_latent)) / np.sqrt(d_latent)
        A2 = rng.standard_normal((D, H)) / np.sqrt(H)

        def mix(Z):
            return ORACLE.mix_mlp(Z, A1, A2, s)
    else:
        raise ValueError(f"unknown mixing {mixing!r}; use linear or nonlinear")

    Zc = ORACLE.sample_latent(B, nv, n_total, rng)
    Xc = mix(Zc)
    sd = 0.1 * float(np.mean(Xc.std(0)))
    Xc = ORACLE.add_obs_noise(Xc, sd, rng)
    Xc_raw = Xc
    if scaling == "standardise":
        Xc = CORE.standardise(Xc, Xc[basis_i])
    Xb, Xp = Xc[basis_i], Xc[pool_i]

    mu, W = CORE.fit_pca(Xb, d)
    Yp = CORE.project(Xp, mu, W, ())
    band = CORE.null_band_from_pool(Yp, n, ORACLE.B_NULL, ORACLE.ALPHA, rng)

    rng_iv = np.random.default_rng(iv_seed)
    if k_nodes == 1:
        # Default and existing modes (plain sweep, --iv-scale-sweep): unchanged.
        nodes = rng_iv.choice(d_latent, size=1, replace=False)
    else:
        # --k-sweep only: gate1's own selection pattern at size=k
        # (81_ranktest_oracle.gate1: nodes = rng_iv.choice(dl, size=k,
        # replace=False)). Identical call shape, k instead of 1.
        nodes = rng_iv.choice(d_latent, size=k_nodes, replace=False)
    Ze = ORACLE.sample_latent(B, nv, n, rng, kind="hard", nodes=nodes,
                              rng_iv=rng_iv, iv_scale=iv_scale)
    Xe = ORACLE.add_obs_noise(mix(Ze), sd, rng)
    if scaling == "standardise":
        Xe = CORE.standardise(Xe, Xc_raw[basis_i])

    r = CORE.rank_diagnostic(Xe, Xb, Xp, d, n, ORACLE.B_NULL, ORACLE.ALPHA, rng,
                             basis_idx=basis_i, ref_pool_idx=pool_i,
                             null_band=band)
    return bool(r["reject_rank2_cf"])


def cell_seeds(mixing_idx, scaling_idx, d_idx, s_idx, draw):
    """Distinct reproducible seeds per (mixing, cell, draw). No draw special-cased."""
    off = (mixing_idx * 2_000_000 + scaling_idx * 300_000
           + d_idx * 50_000 + s_idx * 8_000 + draw)
    return 1_000_000 + off, 20_000_000 + off


def run_cell(mixing, mixing_idx, scaling, scaling_idx, d_idx, d_latent,
             s_idx, s, n_draws):
    rejects = []
    for draw in range(n_draws):
        ms, ivs = cell_seeds(mixing_idx, scaling_idx, d_idx, s_idx, draw)
        rejects.append(one_draw(d_latent, D_OBS, N_E, s, scaling, mixing, ms, ivs))
    rate = float(np.mean(rejects)) if rejects else None
    return dict(mixing=mixing, d_latent=int(d_latent),
                d_projection=int(min(10, d_latent)),
                s=float(s), scaling=scaling, n_draws=int(n_draws),
                reject_rate=rate,
                n_reject=int(np.sum(rejects)))


# =============================================================== s* bracket
def bracket_s_star(rows, level):
    """The interval [s_lo, s_hi] of adjacent grid points where the rate crosses.

    rows: the cells for one (d, scaling), any order. Returns a bracket with
    honest censoring; no interpolation.
    """
    rows = sorted(rows, key=lambda r: r["s"])
    s = [r["s"] for r in rows]
    rate = [r["reject_rate"] for r in rows]
    above = [rt is not None and rt > level for rt in rate]

    if above[0]:
        return dict(kind="left_censored",
                    s_lo=None, s_hi=float(s[0]),
                    note=(f"LEFT-CENSORED: rate {rate[0]:.4f} already exceeds "
                          f"the level {level:.4f} at the smallest s={s[0]}; "
                          f"s_star <= {s[0]}"),
                    rates=list(map(float, rate)), s_grid=list(map(float, s)))
    for i in range(1, len(s)):
        if above[i]:
            return dict(kind="bracketed",
                        s_lo=float(s[i - 1]), s_hi=float(s[i]),
                        note=(f"rate crosses the level {level:.4f} between "
                              f"s={s[i - 1]} (rate {rate[i - 1]:.4f}) and "
                              f"s={s[i]} (rate {rate[i]:.4f}); "
                              f"s_star in ({s[i - 1]}, {s[i]}]"),
                        rates=list(map(float, rate)),
                        s_grid=list(map(float, s)))
    return dict(kind="right_censored",
                s_lo=float(s[-1]), s_hi=None,
                note=(f"RIGHT-CENSORED: rate never exceeds the level "
                      f"{level:.4f} within the grid (max rate "
                      f"{max(r for r in rate if r is not None):.4f} at "
                      f"s={s[-1]}); s_star > {s[-1]}"),
                rates=list(map(float, rate)), s_grid=list(map(float, s)))


def coarse_trend(brackets_by_d):
    """Describe how s* moves with d. Coarse, censoring-aware, no law, no R^2."""
    reps = []           # one representative s* proxy per d, with its kind
    for d in sorted(brackets_by_d):
        b = brackets_by_d[d]
        if b["kind"] == "bracketed":
            proxy, bound = float(b["s_hi"]), "point_in_bracket"
        elif b["kind"] == "left_censored":
            proxy, bound = float(b["s_hi"]), "upper_bound"      # s_star <= s_hi
        else:
            proxy, bound = float(b["s_lo"]), "lower_bound"      # s_star > s_lo
        reps.append(dict(d_latent=int(d), s_star_proxy=proxy, bound=bound,
                         kind=b["kind"]))
    proxies = [r["s_star_proxy"] for r in reps]
    ds = [r["d_latent"] for r in reps]
    monotone_down = all(b <= a for a, b in zip(proxies, proxies[1:]))
    slope = None
    if len(ds) >= 2 and proxies[0] != proxies[-1]:
        slope = float((proxies[-1] - proxies[0]) / (ds[-1] - ds[0]))
    return dict(
        representative_points=reps,
        direction=("s_star decreases with d" if monotone_down
                   else "s_star not monotone in d over this grid"),
        monotone_decreasing=bool(monotone_down),
        coarse_trend_slope_s_star_per_d=slope,
        label=("coarse trend estimate, 5-point grid; representative s* proxies "
               "mix bracket points with censoring bounds, so this is a "
               "direction and rough rate only, not a fitted law; no R^2"))


# =============================================================== oracle gate
def run_step0_gate(smoke):
    """Call 81.gate0 and assert its known answer passes. Never invents a gate."""
    if smoke:
        # Light Monte-Carlo so the smoke path stays fast; gate0's own logic,
        # fewer splits and one d. Labelled so it is never mistaken for the
        # full-run gate.
        cfg = dict(n_splits=40, d_set=(5,), d_latent=10, D=D_OBS, n_env=300)
    else:
        cfg = dict()            # gate0 defaults: the full level-control gate
    print(f"[gate0] running with {cfg or 'defaults'} ...", flush=True)
    res = ORACLE.gate0(**cfg)
    verdict = res.get("verdict")
    print(f"[gate0] verdict={verdict}  {res.get('pass_criterion', '')}",
          flush=True)
    return verdict == "PASS", verdict, res.get("pass_criterion"), cfg


def provenance(gate_cfg):
    RIO = _load(RESULTS_IO_PY, "_results_io")
    return dict(
        core_path=str(CORE_PY.relative_to(REPO)),
        core_md5=md5(CORE_PY),
        oracle_path=str(ORACLE_PY.relative_to(REPO)),
        oracle_md5=md5(ORACLE_PY),
        git_commit=RIO.git_commit(),
        timestamp=utc_stamp(),
        decision_rule=DECISION_RULE_ID,
        oracle_gate=GATE_ID,
        oracle_gate_config=gate_cfg,
        versions=RIO.versions(),
        platform_tag=RIO.platform_tag(),
    )


def full_config(mixing):
    return dict(mixing=mixing,
               s_enters_mixing=bool(mixing == "nonlinear"),
               d_latent_grid=list(D_LATENT_GRID), s_grid=list(S_GRID),
               n_e=N_E, draws_per_cell=DRAWS_PER_CELL, D=D_OBS,
               scalings=list(SCALINGS), alpha=ORACLE.ALPHA,
               B_null=ORACLE.B_NULL,
               s_star_crossing_level=crossing_level(DRAWS_PER_CELL),
               s_star_crossing_level_id=("alpha + 2 Monte-Carlo SE at "
                                         "draws_per_cell, reused from "
                                         "81.gate0's over-rejection boundary"))


def write_scaling(scaling, cells, gate_cfg, outdir, mixing):
    level = crossing_level(DRAWS_PER_CELL)
    brackets_by_d = {}
    for d in D_LATENT_GRID:
        rows = [c for c in cells if c["d_latent"] == d]
        if len(rows) == len(S_GRID):
            brackets_by_d[d] = bracket_s_star(rows, level)
    payload = dict(
        mixing=mixing,
        scaling=scaling,
        config=full_config(mixing),
        s_star_crossing_level=level,
        cells=cells,
        s_star_bracket_per_d={str(d): b for d, b in brackets_by_d.items()},
        coarse_trend=coarse_trend(brackets_by_d) if brackets_by_d else None,
        n_cells=len(cells),
        n_cells_expected=len(D_LATENT_GRID) * len(S_GRID),
        provenance=provenance(gate_cfg),
        contains_no_test_on_real_data=True,
        disclaimer=("SYNTHETIC ONLY. No real dataset was touched and no "
                    "assumption verdict is expressed or implied."),
    )
    path = outdir / f"{utc_stamp()}__decay_{mixing}_{scaling}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path, len(cells)


# =============================================================== smoke
def run_smoke(outdir, mixing, mixing_idx):
    print("=" * 70, flush=True)
    print(f" SMOKE ({mixing} mixing): one d=80 cell, s=0.01, both scalings, "
          f"{DRAWS_PER_CELL} draws, plus the step-0 gate", flush=True)
    print("=" * 70, flush=True)

    ok, verdict, crit, gate_cfg = run_step0_gate(smoke=True)
    print(f"step0={'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        print(f"[fatal] step-0 gate did not pass (verdict={verdict!r}); "
              f"writing nothing.", file=sys.stderr)
        sys.exit(1)

    d_latent = 80
    d_idx = D_LATENT_GRID.index(80)
    s = 0.01
    s_idx = S_GRID.index(0.01)

    cells, timings = [], {}
    for scaling_idx, scaling in enumerate(SCALINGS):
        t0 = time.time()
        cell = run_cell(mixing, mixing_idx, scaling, scaling_idx, d_idx,
                        d_latent, s_idx, s, DRAWS_PER_CELL)
        secs = time.time() - t0
        timings[scaling] = secs
        cells.append(cell)
        print(f"[cell] d80 s=0.01 {scaling:<12} rate={cell['reject_rate']:.4f} "
              f"seconds={secs:.1f}", flush=True)

    # The number that decides GPU vs Mac. Timed on the raw arm explicitly and
    # printed even on a fast path.
    d80 = timings["raw"]
    print(f"d80_cell_seconds={d80:.3f}", flush=True)

    outdir.mkdir(parents=True, exist_ok=True)
    payload = dict(
        mode="smoke",
        mixing=mixing,
        cell=dict(d_latent=80, s=0.01, draws_per_cell=DRAWS_PER_CELL,
                  mixing=mixing),
        cells=cells,
        d80_cell_seconds=d80,
        d80_cell_seconds_by_scaling={k: float(v) for k, v in timings.items()},
        step0_pass=True, step0_verdict=verdict, step0_criterion=crit,
        provenance=provenance(gate_cfg),
        contains_no_test_on_real_data=True,
        disclaimer=("SYNTHETIC SMOKE. Confirms the write path and times the "
                    "most expensive cell; not a full sweep."),
    )
    path = outdir / f"{utc_stamp()}__decay_smoke.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[write] {path}", flush=True)
    print("SMOKE COMPLETE", flush=True)


# =========================================================== iv-scale sweep
def iv_cell_seeds(d_idx, iv_idx, draw):
    """Distinct reproducible seeds for the iv-scale sweep, in their own band so
    they never collide with the s-sweep's cell_seeds. No draw is special-cased."""
    off = 900_000_000 + d_idx * 5_000_000 + iv_idx * 800_000 + draw
    return off, off + 100_000_000


def run_iv_cell(d_idx, d_latent, iv_idx, iv_scale, n_draws):
    """One (d, iv_scale) cell: linear mixing, standardise, hard k=1 at a FIXED
    intervention magnitude iv_scale. s is inert under linear mixing and is not
    swept here; iv_scale is."""
    rejects = []
    for draw in range(n_draws):
        ms, ivs = iv_cell_seeds(d_idx, iv_idx, draw)
        rejects.append(one_draw(d_latent, D_OBS, N_E, 0.0, "standardise",
                                "linear", ms, ivs, iv_scale=iv_scale))
    rate = float(np.mean(rejects)) if rejects else None
    return dict(swept_axis="iv_scale", mixing="linear", scaling="standardise",
                d_latent=int(d_latent), d_projection=int(min(10, d_latent)),
                iv_scale=float(iv_scale),
                s=float(iv_scale),   # s-slot holds iv_scale so bracket_s_star reuses cleanly
                n_draws=int(n_draws), reject_rate=rate,
                n_reject=int(np.sum(rejects)))


def write_ivscale(cells, gate_cfg, outdir):
    level = crossing_level(DRAWS_PER_CELL)
    brackets_by_d = {}
    for d in IV_D_LATENT_GRID:
        rows = [c for c in cells if c["d_latent"] == d]
        if len(rows) == len(IV_SCALE_GRID):
            # bracket_s_star brackets over the value in each row's "s" slot,
            # which for these cells holds iv_scale. The s* logic is reused, the
            # axis is iv_scale.
            brackets_by_d[d] = bracket_s_star(rows, level)
    payload = dict(
        swept_axis="iv_scale",
        mixing="linear",
        iv_scale_enters_intervention=True,
        scaling="standardise",
        config=dict(
            swept_axis="iv_scale",
            mixing="linear",
            iv_scale_enters_intervention=True,
            iv_scale_grid=list(IV_SCALE_GRID),
            d_latent_grid=list(IV_D_LATENT_GRID),
            scalings=list(IV_SCALINGS),
            n_e=N_E, draws_per_cell=DRAWS_PER_CELL, D=D_OBS,
            alpha=ORACLE.ALPHA, B_null=ORACLE.B_NULL,
            crossing_level=level,
            crossing_level_id=("alpha + 2 Monte-Carlo SE at draws_per_cell, "
                               "reused from 81.gate0's over-rejection boundary"),
            bracket_field_semantics=("iv_scale_bracket_per_d reuses the s* "
                                     "bracket helper; its s_lo, s_hi and s_grid "
                                     "fields hold iv_scale values, not the "
                                     "nonlinearity scale s"),
        ),
        crossing_level=level,
        cells=cells,
        iv_scale_bracket_per_d={str(d): b for d, b in brackets_by_d.items()},
        coarse_trend=coarse_trend(brackets_by_d) if brackets_by_d else None,
        n_cells=len(cells),
        n_cells_expected=len(IV_D_LATENT_GRID) * len(IV_SCALE_GRID),
        provenance=provenance(gate_cfg),
        contains_no_test_on_real_data=True,
        disclaimer=("SYNTHETIC ONLY. Intervention-magnitude sweep under linear "
                    "mixing, the valid regime. No real dataset was touched and "
                    "no assumption verdict is expressed or implied."),
    )
    path = outdir / f"{utc_stamp()}__decay_ivscale_standardise.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path, len(cells)


def run_ivscale_sweep(outdir, gate_cfg):
    if outdir.exists():
        print(f"[clean] rm -rf {outdir}", flush=True)
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("[config] swept_axis=iv_scale mixing=linear "
          "iv_scale_enters_intervention=true "
          f"iv_scale_grid={list(IV_SCALE_GRID)} "
          f"d_latent_grid={list(IV_D_LATENT_GRID)} scalings={list(IV_SCALINGS)}",
          flush=True)
    cells = []
    for d_idx, d_latent in enumerate(IV_D_LATENT_GRID):
        for iv_idx, iv_scale in enumerate(IV_SCALE_GRID):
            print(f"[cell] standardise d_latent={d_latent:<3} "
                  f"iv_scale={iv_scale} ...", flush=True)
            cell = run_iv_cell(d_idx, d_latent, iv_idx, iv_scale, DRAWS_PER_CELL)
            cells.append(cell)
            print(f"       rate={cell['reject_rate']:.4f} "
                  f"({cell['n_reject']}/{cell['n_draws']})", flush=True)
    want = len(IV_D_LATENT_GRID) * len(IV_SCALE_GRID)     # 3 x 4 x 1 = 12
    path, n = write_ivscale(cells, gate_cfg, outdir)
    print(f"[write] {path}  ({n} cells)", flush=True)
    print("\n" + "=" * 70)
    print(f" cells: {n} of {want}")
    print("=" * 70)
    if n != want:
        print(f"[fatal] SHORTFALL: {n} cells for {want} requested.",
              file=sys.stderr)
        sys.exit(1)
    print(f" RESULT: PASS, all {want} iv_scale cells produced.", flush=True)


# ================================================================ k sweep
def k_cell_seeds(d_idx, k_idx, draw):
    """Distinct reproducible seeds for the k sweep, in their own band so they
    never collide with cell_seeds or iv_cell_seeds. No draw is special-cased."""
    off = 1_500_000_000 + d_idx * 5_000_000 + k_idx * 800_000 + draw
    return off, off + 100_000_000


def run_k_cell(d_idx, d_latent, k_idx, k_nodes, n_draws):
    """One (d, k) cell: linear mixing, standardise, hard intervention on
    k_nodes nodes at FIXED magnitude K_IV_SCALE_FIXED. k is the only swept
    variable; magnitude and everything else are held."""
    rejects = []
    for draw in range(n_draws):
        ms, ivs = k_cell_seeds(d_idx, k_idx, draw)
        rejects.append(one_draw(d_latent, D_OBS, N_E, 0.0, "standardise",
                                "linear", ms, ivs,
                                iv_scale=K_IV_SCALE_FIXED, k_nodes=k_nodes))
    rate = float(np.mean(rejects)) if rejects else None
    return dict(swept_axis="k_nodes", mixing="linear", scaling="standardise",
                intervention_kind="hard",
                d_latent=int(d_latent), d_projection=int(min(10, d_latent)),
                k_nodes=int(k_nodes), iv_scale_fixed=float(K_IV_SCALE_FIXED),
                s=float(k_nodes),   # s-slot holds k so bracket_s_star reuses cleanly
                n_draws=int(n_draws), reject_rate=rate,
                n_reject=int(np.sum(rejects)))


def write_ksweep(cells, gate_cfg, outdir):
    level = crossing_level(DRAWS_PER_CELL)
    brackets_by_d = {}
    for d in K_D_LATENT_GRID:
        rows = [c for c in cells if c["d_latent"] == d]
        if len(rows) == len(K_GRID):
            # bracket_s_star brackets over each row's "s" slot, which here
            # holds k. The bracket helper is reused; the axis is k_nodes.
            brackets_by_d[d] = bracket_s_star(rows, level)
    payload = dict(
        swept_axis="k_nodes",
        mixing="linear",
        intervention_kind="hard",
        iv_scale_fixed=float(K_IV_SCALE_FIXED),
        k_grid=list(K_GRID),
        scaling="standardise",
        config=dict(
            swept_axis="k_nodes",
            mixing="linear",
            intervention_kind="hard",
            iv_scale_fixed=float(K_IV_SCALE_FIXED),
            iv_scale_fixed_note=("deliberate deviation from gate1, which draws "
                                 "each node magnitude from U(2,4); fixed here "
                                 "so k is the only varying axis"),
            node_selection=("rng_iv.choice(d_latent, size=k, replace=False), "
                            "gate1's own pattern"),
            k_grid=list(K_GRID),
            d_latent_grid=list(K_D_LATENT_GRID),
            scalings=list(K_SCALINGS),
            n_e=N_E, draws_per_cell=DRAWS_PER_CELL, D=D_OBS,
            alpha=ORACLE.ALPHA, B_null=ORACLE.B_NULL,
            crossing_level=level,
            crossing_level_id=("alpha + 2 Monte-Carlo SE at draws_per_cell, "
                               "reused from 81.gate0's over-rejection boundary"),
            bracket_field_semantics=("k_bracket_per_d reuses the s* bracket "
                                     "helper; its s_lo, s_hi and s_grid fields "
                                     "hold k values, not the nonlinearity "
                                     "scale s"),
            null_geometry=("H0: rank<=2. k=1 hard is rank 1 or 2 (inside or "
                           "on the null boundary); k>=3 exceeds it. The "
                           "crossing in k is the real detection threshold."),
        ),
        crossing_level=level,
        cells=cells,
        k_bracket_per_d={str(d): b for d, b in brackets_by_d.items()},
        coarse_trend=coarse_trend(brackets_by_d) if brackets_by_d else None,
        n_cells=len(cells),
        n_cells_expected=len(K_D_LATENT_GRID) * len(K_GRID),
        provenance=provenance(gate_cfg),
        contains_no_test_on_real_data=True,
        disclaimer=("SYNTHETIC ONLY. Intervened-node-count sweep under linear "
                    "mixing, the valid regime. No real dataset was touched and "
                    "no assumption verdict is expressed or implied."),
    )
    path = outdir / f"{utc_stamp()}__decay_ksweep_standardise.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path, len(cells)


def run_ksweep(outdir, gate_cfg):
    if outdir.exists():
        print(f"[clean] rm -rf {outdir}", flush=True)
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("[config] swept_axis=k_nodes mixing=linear intervention_kind=hard "
          f"iv_scale_fixed={K_IV_SCALE_FIXED} k_grid={list(K_GRID)} "
          f"d_latent_grid={list(K_D_LATENT_GRID)} scalings={list(K_SCALINGS)}",
          flush=True)
    cells = []
    for d_idx, d_latent in enumerate(K_D_LATENT_GRID):
        for k_idx, k_nodes in enumerate(K_GRID):
            print(f"[cell] standardise d_latent={d_latent:<3} "
                  f"k={k_nodes} ...", flush=True)
            cell = run_k_cell(d_idx, d_latent, k_idx, k_nodes, DRAWS_PER_CELL)
            cells.append(cell)
            print(f"       rate={cell['reject_rate']:.4f} "
                  f"({cell['n_reject']}/{cell['n_draws']})", flush=True)
    want = len(K_D_LATENT_GRID) * len(K_GRID)     # 3 x 4 x 1 = 12
    path, n = write_ksweep(cells, gate_cfg, outdir)
    print(f"[write] {path}  ({n} cells)", flush=True)
    print("\n" + "=" * 70)
    print(f" cells: {n} of {want}")
    print("=" * 70)
    if n != want:
        print(f"[fatal] SHORTFALL: {n} cells for {want} requested.",
              file=sys.stderr)
        sys.exit(1)
    print(f" RESULT: PASS, all {want} k cells produced.", flush=True)


# =============================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="one d=80 s=0.01 cell both scalings, gate, timing")
    ap.add_argument("--mixing", choices=MIXINGS, default="linear",
                    help="linear (default, the valid regime) or nonlinear")
    ap.add_argument("--iv-scale-sweep", action="store_true",
                    help="sweep intervention magnitude iv_scale (linear regime, "
                         "reduced grid) instead of nonlinearity scale s")
    ap.add_argument("--k-sweep", action="store_true",
                    help="sweep the NUMBER of intervened nodes k at fixed "
                         "magnitude (linear regime, reduced grid); k crosses "
                         "the H0 rank<=2 boundary at k=3")
    ap.add_argument("--outdir", default=str(OUT_DIR))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    mixing = args.mixing
    mixing_idx = MIXINGS.index(mixing)

    if args.iv_scale_sweep and args.k_sweep:
        sys.exit("[fatal] --iv-scale-sweep and --k-sweep are mutually "
                 "exclusive: one swept axis at a time.")

    if args.smoke:
        run_smoke(outdir, mixing, mixing_idx)
        return

    # ---- lockfile
    LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
    if LOCKFILE.exists():
        other = LOCKFILE.read_text().strip()
        alive = False
        if other.isdigit():
            try:
                os.kill(int(other), 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            sys.exit(f"[fatal] another decay run is live (pid {other}, lock "
                     f"{LOCKFILE}). Two concurrent batches into one output "
                     f"directory is how the 11 Aug artefacts had to be "
                     f"discarded. Refusing.")
        print(f"[lock] stale lock from pid {other or 'unknown'}; reclaiming",
              flush=True)
        LOCKFILE.unlink()
    LOCKFILE.write_text(str(os.getpid()))

    try:
        # ---- step-0 gate BEFORE any grid number is trusted.
        ok, verdict, crit, gate_cfg = run_step0_gate(smoke=False)
        if not ok:
            print(f"[fatal] step-0 gate FAILED (verdict={verdict!r}); the "
                  f"known answer did not pass, so no grid number is "
                  f"trustworthy. Writing nothing.", file=sys.stderr)
            sys.exit(1)

        # ---- iv-scale sweep: a real swept axis in the valid regime. The
        # gate above has already run; this branch replaces the S_GRID sweep and
        # writes to results/decay_ivscale/, not results/decay/.
        if args.iv_scale_sweep:
            run_ivscale_sweep(IVSCALE_OUT_DIR, gate_cfg)
            return

        # ---- k sweep: vary intervened-node count at fixed magnitude. The
        # gate above has already run; writes to results/decay_ksweep/ only.
        if args.k_sweep:
            run_ksweep(KSWEEP_OUT_DIR, gate_cfg)
            return

        # ---- fresh output directory.
        if outdir.exists():
            print(f"[clean] rm -rf {outdir}", flush=True)
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        print(f"[config] mixing={mixing} "
              f"(s_enters_mixing={mixing == 'nonlinear'})", flush=True)
        written, total_cells = [], 0
        for scaling_idx, scaling in enumerate(SCALINGS):
            cells = []
            for d_idx, d_latent in enumerate(D_LATENT_GRID):
                for s_idx, s in enumerate(S_GRID):
                    print(f"[cell] {scaling:<12} d_latent={d_latent:<3} "
                          f"s={s} ...", flush=True)
                    cell = run_cell(mixing, mixing_idx, scaling, scaling_idx,
                                    d_idx, d_latent, s_idx, s, DRAWS_PER_CELL)
                    cells.append(cell)
                    print(f"       rate={cell['reject_rate']:.4f} "
                          f"({cell['n_reject']}/{cell['n_draws']})", flush=True)
            path, n = write_scaling(scaling, cells, gate_cfg, outdir, mixing)
            written.append(path)
            total_cells += n
            print(f"[write] {path}  ({n} cells)", flush=True)

        # ---- PASS/FAIL: 5 x 5 x 2 = 50 cells across the two artefacts.
        want = len(D_LATENT_GRID) * len(S_GRID) * len(SCALINGS)
        print("\n" + "=" * 70)
        print(f" artefacts: {len(written)} of {len(SCALINGS)} scalings")
        print(f" cells    : {total_cells} of {want}")
        for p in written:
            print(f"   {p}")
        print("=" * 70)
        if len(written) != len(SCALINGS) or total_cells != want:
            print(f"[fatal] SHORTFALL: {total_cells} cells for {want} "
                  f"requested, {len(written)} artefacts for {len(SCALINGS)} "
                  f"scalings.", file=sys.stderr)
            sys.exit(1)
        print(" RESULT: PASS, all 50 cells produced.", flush=True)
    finally:
        if LOCKFILE.exists():
            LOCKFILE.unlink()


# Loaded once at import so every helper sees the same statistic and simulator.
CORE = _load(CORE_PY, "_ranktest_core")
ORACLE = _load(ORACLE_PY, "_ranktest_oracle")


if __name__ == "__main__":
    main()
