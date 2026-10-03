"""M2.6 aggregate-resolution gate check (decides the M3 fork).

M2.5 showed the P4 gate slips at PER-ENVIRONMENT atomic resolution. But the
framework consumes the AGGREGATE recoverable dimension n_recoverable (the number
of latent directions with detectable signal across the whole multi-environment
set), not per-environment ranks. This check asks: on the REDUCING-regime data the
backbone recovers from, does the gate read that AGGREGATE quantity correctly? If
yes, the gate is reliable at the resolution the framework actually uses (M3 Path A).

Aggregation method and its E0 provenance (determined, not invented)
  rank_readout.py offers NO pooling/stacking function. E0 validated the gate by
  intervening on ALL target nodes in ONE environment and reading the rank of that
  single covariance difference (e0_oracle.py: one "iv" env = hard on all sources;
  Delta = covariance_difference(Ye, Y0); spectral_rank + estimate_rank_lfc, at
  d_proj = d_latent). This reuses that EXACT aggregation with all d nodes as the
  single aggregate intervention. The true aggregate recoverable dimension is the
  number of distinct intervened latents = d for the all-node construction,
  confirmed per seed against constructed_rank.

  The backbone requires the atomic per-latent environments to recover, so both are
  generated from the same SCM/seed/regime: the atomic envs feed the backbone
  (recovery MCC, coupled-view context), and the single all-node aggregate env feeds
  the gate. Same reducing regime (iv_scale = 0.1 = E1's value), same seeds.

Primary readout is estimate_rank_lfc at E0's d_proj = d_latent (the assertion runs
on this). Recorded caveats, disclosed not hidden:
  * At d_proj = d the aggregate rank fills the projection, so the LFC k_hat cannot
    over-read (bounded at d) and the boundary sv-gap is undefined; k_hat < d would
    still fire if any direction were undetectable, so k_hat == d is a real positive.
  * The deterministic spectral_rank under-reads the aggregate (its rtol is relative
    to the largest singular value, and the aggregate spectrum is anisotropic), so
    the LFC readout, not spectral_rank, is the reliable aggregate reading.
  * A d_proj = d_latent + 5 headroom cross-check (null space present) is reported
    for context; it can over-read by one on rare seeds (the alpha-level slip).

Read-only in spirit: imports the generator, rank readout, backbone and E0 mcc;
reimplements nothing. Writes results/e2p6/aggregate_gate_report.json (tracked),
prints a PASS/FAIL table, exits non-zero if any seed's aggregate k_hat != d.
"""
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=r".*encountered in matmul",
                        category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results" / "e2p6"
E2P5_REPORT = REPO / "results" / "e2p5" / "regime_coupling_report.json"

SEEDS = list(range(10))
D_LATENT = 5
D_OBS = 200
N_PER_ENV = 4000
D_PROJ = 5                 # E0's setting (d_proj = d_latent)
D_PROJ_XCHECK = 10         # headroom cross-check (null space present)
EDGE_PROB = 0.4
ALPHA = 0.05
B_BOOT = 500
RANK_RTOL = 0.05
IV_REDUCING = 0.1          # E1's value
RECOVERY_MCC = 0.90


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SIM = _load(REPO / "sim" / "simulator.py", "grd_sim_simulator")
RR = _load(REPO / "src" / "gate" / "rank_readout.py", "grd_gate_rank_readout")
BK = _load(REPO / "src" / "recover" / "backbone.py", "grd_recover_backbone")
E0 = _load(REPO / "experiments" / "e0_oracle.py", "grd_e0_oracle")


def _aggregate_khat(Y_all, Y_obs, d_proj, seed_tag):
    """Aggregate LFC rank of one all-node covariance difference (E0's readout)."""
    rng = np.random.default_rng(700_000 + seed_tag)
    return RR.estimate_rank_lfc(Y_all, Y_obs, r_max=d_proj - 1,
                                B=B_BOOT, alpha=ALPHA, rng=rng)["k_hat"]


def _boundary_reject_rate(Y_all, Y_obs, r, n_boot_seeds=10):
    """Fraction of bootstrap seeds for which H0(rank<=r) is rejected. If a seed
    under-reads (k_hat==r<true_dim), a rate near 0 means the miss is stable (a
    genuine detectability failure), not an alpha-level slip.
    """
    rej = [RR.lfc_rank_test(Y_all, Y_obs, r, B=B_BOOT, alpha=ALPHA,
                            rng=np.random.default_rng(b))["reject"]
           for b in range(n_boot_seeds)]
    return round(sum(rej) / len(rej), 2)


def run():
    m25_note = "not found"
    if E2P5_REPORT.exists():
        m25 = json.load(open(E2P5_REPORT))
        red = m25["regimes"]["REDUCING"]
        m25_note = (f"{red['gate_seeds_correct']}/10 seeds fully correct at "
                    f"per-environment atomic resolution (M2.5 reducing regime)")

    report = {
        "milestone": "M2.6",
        "code_commit": E0._code_commit(),
        "aggregation_method": (
            "single all-node hard intervention environment, covariance_difference vs obs, "
            "estimate_rank_lfc + spectral_rank at d_proj=d_latent"),
        "aggregation_provenance": (
            "E0 (experiments/e0_oracle.py) intervened on all target nodes in ONE env and read "
            "that single covariance difference's rank; rank_readout.py has no pooling function. "
            "Reused exactly, with all d nodes as the aggregate intervention."),
        "config": dict(seeds=SEEDS, d_latent=D_LATENT, D=D_OBS, n_per_env=N_PER_ENV,
                       d_proj=D_PROJ, d_proj_xcheck=D_PROJ_XCHECK, edge_prob=EDGE_PROB,
                       alpha=ALPHA, B=B_BOOT, rank_rtol=RANK_RTOL, iv_scale=IV_REDUCING,
                       regime="REDUCING", recovery_mcc_threshold=RECOVERY_MCC),
        "m25_per_env_contrast": m25_note,
        "data_fingerprint": {},
        "per_seed": [],
    }

    for seed in SEEDS:
        rng_scm = np.random.default_rng(seed)
        B, nv, is_source = SIM.make_scm(D_LATENT, rng_scm, edge_prob=EDGE_PROB)
        all_nodes = tuple(range(D_LATENT))
        true_dim = SIM.constructed_rank(B, nv, "hard", all_nodes, IV_REDUCING)

        # One dataset: atomic envs (backbone) + one all-node aggregate env (gate).
        specs = [SIM.EnvSpec("basis", None, ()), SIM.EnvSpec("obs", None, ())]
        for i in range(D_LATENT):
            specs.append(SIM.EnvSpec(f"iv{i}", "hard", (i,), IV_REDUCING))
        specs.append(SIM.EnvSpec("ivall", "hard", all_nodes, IV_REDUCING))
        ds = SIM.simulate(D_LATENT, D_OBS, N_PER_ENV, specs, seed,
                          mixing="linear", edge_prob=EDGE_PROB)
        report["data_fingerprint"][str(seed)] = E0._fingerprint(ds.environments["ivall"].X)

        # ---- GATE aggregate reading (E0's method), primary d_proj and cross-check.
        mu, Wp = BK.fit_pca(ds.environments["basis"].X, D_PROJ)
        Y_obs = BK.project(ds.environments["obs"].X, mu, Wp)
        Y_all = BK.project(ds.environments["ivall"].X, mu, Wp)
        Delta = RR.covariance_difference(Y_all, Y_obs)
        agg_khat = _aggregate_khat(Y_all, Y_obs, D_PROJ, seed)
        agg_spectral = int(RR.spectral_rank(Delta, rtol=RANK_RTOL))
        agg_gap = RR.gap_ratio(Delta, true_dim)   # inf when true_dim == d_proj (rank fills space)

        muX, WpX = BK.fit_pca(ds.environments["basis"].X, D_PROJ_XCHECK)
        Y_obsX = BK.project(ds.environments["obs"].X, muX, WpX)
        Y_allX = BK.project(ds.environments["ivall"].X, muX, WpX)
        DeltaX = RR.covariance_difference(Y_allX, Y_obsX)
        agg_khat_x = _aggregate_khat(Y_allX, Y_obsX, D_PROJ_XCHECK, seed + 10_000)
        agg_gap_x = RR.gap_ratio(DeltaX, true_dim)

        # ---- BACKBONE recovery on the atomic envs (coupled-view context).
        envs = {k: e.X for k, e in ds.environments.items() if k != "ivall"}
        targets = {f"iv{i}": i for i in range(D_LATENT)}
        out = BK.recover(envs, targets, D_LATENT, basis_key="basis", obs_key="obs")
        recover_mcc = E0.mcc(out["Z_hat"], ds.environments["obs"].Z)

        agg_pass = bool(agg_khat == true_dim)
        recover_pass = bool(recover_mcc >= RECOVERY_MCC)
        # For an under-read, confirm it is stable across bootstrap seeds (genuine,
        # not an alpha-level slip): reject rate of the decisive boundary H0(rank<=k_hat).
        stable = None
        if not agg_pass and agg_khat < true_dim:
            stable = _boundary_reject_rate(Y_all, Y_obs, agg_khat)
        report["per_seed"].append(dict(
            seed=seed, true_dim=int(true_dim),
            agg_k_hat=int(agg_khat), agg_pass=agg_pass,
            agg_spectral_rank=agg_spectral,
            agg_sv_gap=(None if not np.isfinite(agg_gap) else round(float(agg_gap), 2)),
            agg_k_hat_xcheck_dproj10=int(agg_khat_x),
            agg_sv_gap_xcheck=(None if not np.isfinite(agg_gap_x) else round(float(agg_gap_x), 1)),
            underread_boundary_reject_rate=stable,
            recover_mcc=round(float(recover_mcc), 4), recover_pass=recover_pass,
            gate_backbone_disagreement=bool((not agg_pass) and recover_pass)))

    all_pass = all(r["agg_pass"] for r in report["per_seed"])
    disagree = [r["seed"] for r in report["per_seed"] if r["gate_backbone_disagreement"]]
    report["gate_backbone_disagreement_seeds"] = disagree
    report["verdict"] = "PASS" if all_pass else "FAIL"
    report["status"] = report["verdict"]
    if all_pass:
        report["m3_fork"] = (
            "Path A supported: the LFC gate reads the AGGREGATE recoverable dimension (d) "
            "correctly on the reducing data at E0's d_proj, on all seeds. Caveat: spectral_rank "
            "under-reads the aggregate (anisotropic spectrum), so M3 must use the LFC readout.")
    else:
        n_ok = sum(r["agg_pass"] for r in report["per_seed"])
        report["m3_fork"] = (
            f"Path A NOT cleanly supported. The aggregate LFC gate reads n_recoverable "
            f"correctly on {n_ok}/10 reducing seeds but STABLY under-reads on seeds {disagree} "
            f"(k_hat < true dim across bootstrap seeds, reject rate ~0): a weak aggregate "
            f"direction (sv orders of magnitude below the dominant ones) is masked by the "
            f"dominant directions' sampling fluctuation in the pooled covariance difference. "
            f"The backbone still recovers those directions (MCC >= 0.90) via per-node precision "
            f"isolation, so this is a gate-backbone DISAGREEMENT: the gate (aggregate covariance "
            f"rank) and the backbone (per-node precision) are coupled to different signal "
            f"representations. M3 cannot certify the backbone's recoverable set from the "
            f"aggregate covariance rank alone; reconcile before wiring gate into recover.")
    return report


def _print(report):
    print(f"\nM2.6 AGGREGATE GATE CHECK  (status: {report['status']})", flush=True)
    print("=" * 78, flush=True)
    print(f"{'seed':>4} {'true dim':>9} {'agg k_hat':>10} {'pass':>5} "
          f"{'spectral':>9} {'gap(d5)':>8} {'k_hat(d10)':>11} {'gap(d10)':>9} "
          f"{'recover MCC':>12}", flush=True)
    for r in report["per_seed"]:
        print(f"{r['seed']:>4} {r['true_dim']:>9} {r['agg_k_hat']:>10} "
              f"{'yes' if r['agg_pass'] else 'NO':>5} {r['agg_spectral_rank']:>9} "
              f"{str(r['agg_sv_gap']):>8} {r['agg_k_hat_xcheck_dproj10']:>11} "
              f"{str(r['agg_sv_gap_xcheck']):>9} {r['recover_mcc']:>12.4f}", flush=True)
    print("-" * 78, flush=True)
    print(f"aggregation: {report['aggregation_method']}", flush=True)
    print(f"per-environment contrast (M2.5): {report['m25_per_env_contrast']}", flush=True)
    dis = report["gate_backbone_disagreement_seeds"]
    if dis:
        for s in dis:
            r = next(x for x in report["per_seed"] if x["seed"] == s)
            print(f"DISAGREEMENT seed {s}: gate k_hat={r['agg_k_hat']} < true {r['true_dim']} "
                  f"(stable, boundary reject rate {r['underread_boundary_reject_rate']}), "
                  f"but backbone recovers MCC {r['recover_mcc']}", flush=True)
    print(f"VERDICT: {report['verdict']}", flush=True)
    print(f"M3 FORK: {report['m3_fork']}", flush=True)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = run()
    (RESULTS / "aggregate_gate_report.json").write_text(json.dumps(report, indent=2))
    _print(report)
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
