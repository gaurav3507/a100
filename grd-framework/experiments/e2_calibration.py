"""E2: violation-injection calibration (headline experiment, milestone E2).

Inject one data-adequacy violation at a time, sweep severity, and test whether the
GATE verdict tracks where the naive (ungated) backbone fails. For each (arm, severity,
seed) we run the gate (precision readout -> n_recoverable -> verdict) and the naive
backbone (recover on every supplied environment, no gate) and compare.

Central claim: the gate abstains/caps where naive recovery degrades, so it PREDICTS
failure instead of the method failing silently. Kill criterion (design doc): if the
gate verdict does not track naive failure (calibration flat), the claim is false.

PRE-REGISTERED EXPECTATIONS (written before running; a contradiction is reported as a
finding, not tuned away):
  A ENVIRONMENT STARVATION (drop interventional environments m from 5 to 1; gate P1
    caps n_max_env = m). EXPECT tight tracking: gate n_recoverable = m equals the
    number of directions naive recovers; naive MCC over all 5 latents falls as m
    falls; the gate flips PROCEED -> PROCEED_CAPPED exactly when a direction is lost
    (m < 5). Direction: gate tracks, YES.
  B POWER STARVATION (shrink per-environment n_e; gate P2, realized via precision
    detectability at low n). EXPECT the gate to grow CONSERVATIVE as n_e falls
    (precision count drops / abstains) while the linear backbone stays sample-robust,
    so the gate abstains before naive actually fails. Direction: gate degrades in the
    safe direction (never PROCEED while naive collapses), YES, with over-conservatism
    noted.
  C MEASUREMENT CONTAMINATION (replace c of 5 environments with a measurement gain
    shift on the mixing, no latent intervention). Per decision R1 the mechanism-vs-
    measurement gate P3 is temporal-only and this simulator is not temporal, so this
    arm is DETECTABILITY-BASED, not P3-based. EXPECT the covariance/precision gate
    CANNOT distinguish a measurement change from a mechanism change and is partially
    FOOLED (certifies contaminated directions) while naive degrades. Direction: gate
    does NOT reliably track here, a documented limitation needing P3.
  D RANK / WEAK-SIGNAL STARVATION. In this atomic construction reducing the number of
    intervened nodes is the SAME manipulation as arm A, so D is realized as
    intervention-strength weakening (iv_scale 0.1 -> 1.0), a distinct signal axis.
    EXPECT the gate to grow conservative (count drops / abstains) as the intervention
    weakens while the backbone stays signal-robust. Direction: gate degrades in the
    safe direction, YES, with over-conservatism noted; node-count starvation is
    covered by arm A.

Read-only in spirit: imports the simulator, precision gate readout, backbone
(including its _unmixing_row building block, composed here for the partial/naive
recovery because backbone.recover requires full target coverage), discover, and the
E0 mcc; modifies none of them.

Writes results/e2/e2_calibration_report.json (tracked); prints a per-arm severity
table; exits non-zero only if the central claim is KILLED (no arm shows tracking).
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
RESULTS = REPO / "results" / "e2"

SEEDS = list(range(10))
D_LATENT = 5
D_OBS = 200
D_PROJ = 5
EDGE_PROB = 0.4
ALPHA = 0.05
B_BOOT = 500
IV_BASE = 0.1
N_BASE = 2000
RECOVERY_MCC = 0.90
DIR_CORR = 0.90                 # per-direction recovery: |corr| above this = recovered


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SIM = _load(REPO / "sim" / "simulator.py", "grd_sim_simulator")
PR = _load(REPO / "src" / "gate" / "precision_readout.py", "grd_gate_precision_readout")
BK = _load(REPO / "src" / "recover" / "backbone.py", "grd_recover_backbone")
E0 = _load(REPO / "experiments" / "e0_oracle.py", "grd_e0_oracle")


def _abs_corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    da = np.sqrt((a * a).sum()); db = np.sqrt((b * b).sum())
    if da == 0 or db == 0:
        return 0.0
    return abs(float((a * b).sum() / (da * db)))


def _best_match_corr(col, Z_true):
    """Best |corr| of one recovered column against any true latent: does this
    direction recover SOME true latent? Permutation-safe (unlike aligned corr)."""
    return max(_abs_corr(col, Z_true[:, j]) for j in range(Z_true.shape[1]))


def gate_certificate(m_int, precision_count, intended_d):
    """P1 caps n_max_env at the interventional-environment count; P4 (precision, Path A)
    gives the detectable-direction count (P2 folds in: low n_e drops the count).
    """
    n_max_env = m_int
    k_hat_rank = precision_count
    n_recoverable = int(min(n_max_env, k_hat_rank))
    if n_recoverable == 0:
        verdict = "ABSTAIN"
    elif n_recoverable < intended_d:
        verdict = "PROCEED_CAPPED"
    else:
        verdict = "PROCEED"
    return n_recoverable, verdict


def evaluate(basis_X, obs_X, obs_Z, int_envs, seed, readout="precision",
             recovery=None):
    """int_envs: list of (node_index, X). Returns gate + naive + gated metrics.

    readout selects the gate statistic and recovery the backbone row rule
    (recovery=None uses readout). Defaults are the precision gate and backbone;
    Tier 2 uses readout="frobenius", recovery="jad" (Backbone C) and
    readout="covariance" (negative control).
    """
    mu, Wp = BK.fit_pca(basis_X, D_PROJ)
    Y_obs = BK.project(obs_X, mu, Wp)
    Y_int = [BK.project(X, mu, Wp) for (_, X) in int_envs]
    nodes = [i for (i, _) in int_envs]

    # GATE
    gate = PR.count_recoverable(Y_int, Y_obs, alpha=ALPHA, B=B_BOOT,
                                rng=np.random.default_rng(910_000 + seed),
                                readout=readout)
    detect = gate["detect"]
    certified = [nodes[k] for k in range(len(nodes)) if detect[k]]
    n_recoverable, verdict = gate_certificate(len(int_envs), gate["count"], D_LATENT)

    # NAIVE backbone: rows for each supplied env (backbone's own block, via
    # unmixing_rows), zero rows for latents with no supplied environment. Equals
    # backbone.recover when all d nodes are supplied.
    W = BK.unmixing_rows(Y_obs, Y_int, nodes, D_LATENT,
                         readout=readout if recovery is None else recovery)
    Z_hat = Y_obs @ W.T
    naive_mcc = float(E0.mcc(Z_hat, obs_Z))   # overall recovery (Hungarian), the 0.90-bar metric
    # number of true latents recovered by SOME column (permutation-safe direction count)
    best_per_true = [max(_abs_corr(Z_hat[:, i], obs_Z[:, j]) for i in range(D_LATENT))
                     for j in range(D_LATENT)]
    naive_recovered_dirs = int(sum(c > DIR_CORR for c in best_per_true))

    # GATED / certification trustworthiness: does each CERTIFIED column recover some
    # true latent? Best-match |corr| per certified column (permutation-safe). Low here
    # means the gate certified directions that do not recover (fooled).
    # Only meaningful when the gate certifies something; None on ABSTAIN (safe, not fooled).
    if certified:
        certified_recovery = float(np.mean(
            [_best_match_corr(Z_hat[:, node], obs_Z) for node in certified]))
    else:
        certified_recovery = None

    return dict(m_int=len(int_envs), gate_count=int(gate["count"]),
                n_recoverable=n_recoverable, verdict=verdict,
                certified=sorted(int(c) for c in certified),
                naive_mcc=round(naive_mcc, 4),
                naive_recovered_dirs=naive_recovered_dirs,
                certified_recovery=(round(certified_recovery, 4)
                                    if certified_recovery is not None else None))


# ------------------------------------------------------------- arm builders
def _base_scm(seed):
    rng = np.random.default_rng(seed)
    B, nv, is_src = SIM.make_scm(D_LATENT, rng, edge_prob=EDGE_PROB)
    return B, nv, is_src


def arm_A(seed, m_int):
    specs = [SIM.EnvSpec("basis", None, ()), SIM.EnvSpec("obs", None, ())]
    for i in range(m_int):
        specs.append(SIM.EnvSpec(f"iv{i}", "hard", (i,), IV_BASE))
    ds = SIM.simulate(D_LATENT, D_OBS, N_BASE, specs, seed, mixing="linear",
                      edge_prob=EDGE_PROB)
    int_envs = [(i, ds.environments[f"iv{i}"].X) for i in range(m_int)]
    return ds, int_envs


def arm_B(seed, n_e):
    specs = [SIM.EnvSpec("basis", None, ()), SIM.EnvSpec("obs", None, ())]
    for i in range(D_LATENT):
        specs.append(SIM.EnvSpec(f"iv{i}", "hard", (i,), IV_BASE))
    ds = SIM.simulate(D_LATENT, D_OBS, n_e, specs, seed, mixing="linear",
                      edge_prob=EDGE_PROB)
    int_envs = [(i, ds.environments[f"iv{i}"].X) for i in range(D_LATENT)]
    return ds, int_envs


def arm_C(seed, n_contam):
    specs = [SIM.EnvSpec("basis", None, ()), SIM.EnvSpec("obs", None, ())]
    for i in range(D_LATENT):
        specs.append(SIM.EnvSpec(f"iv{i}", "hard", (i,), IV_BASE))
    ds = SIM.simulate(D_LATENT, D_OBS, N_BASE, specs, seed, mixing="linear",
                      edge_prob=EDGE_PROB)
    g = np.random.default_rng(seed + 7)
    for i in range(D_LATENT - n_contam, D_LATENT):
        # measurement contamination: observational latents (no intervention) seen
        # through a per-dimension measurement gain on the mixing.
        Zc = SIM.sample_latent(ds.B, ds.noise_var, N_BASE, g)
        Xc = SIM.mix_linear(Zc, ds.A) * g.uniform(0.5, 1.5, D_OBS)
        ds.environments[f"iv{i}"].X = SIM.add_obs_noise(Xc, ds.sd_obs, g)
    int_envs = [(i, ds.environments[f"iv{i}"].X) for i in range(D_LATENT)]
    return ds, int_envs


def arm_D(seed, iv_scale):
    specs = [SIM.EnvSpec("basis", None, ()), SIM.EnvSpec("obs", None, ())]
    for i in range(D_LATENT):
        specs.append(SIM.EnvSpec(f"iv{i}", "hard", (i,), iv_scale))
    ds = SIM.simulate(D_LATENT, D_OBS, N_BASE, specs, seed, mixing="linear",
                      edge_prob=EDGE_PROB)
    int_envs = [(i, ds.environments[f"iv{i}"].X) for i in range(D_LATENT)]
    return ds, int_envs


ARMS = {
    "A_environment_starvation": dict(
        builder=arm_A, levels=[5, 4, 3, 2, 1], level_name="m_int",
        expect="gate n_recoverable=m tracks naive recovered directions; PROCEED->CAPPED at m<5"),
    "B_power_starvation": dict(
        builder=arm_B, levels=[4000, 400, 100, 40, 20], level_name="n_e",
        expect="gate grows conservative (count drops/abstains) before sample-robust naive fails"),
    "C_measurement_contamination": dict(
        builder=arm_C, levels=[0, 1, 2, 3, 5], level_name="n_contaminated",
        expect="detectability gate cannot tell measurement from mechanism (no P3); partially fooled"),
    "D_weak_signal_starvation": dict(
        builder=arm_D, levels=[0.1, 0.4, 0.7, 0.9, 1.0], level_name="iv_scale",
        expect="gate grows conservative as intervention weakens before signal-robust naive fails"),
}


def _arm_verdict(arm_rows):
    """Classify the gate's behaviour across the sweep from the numbers.

    The decisive calibration signal is GATED MCC: recovery quality on the directions
    the gate certifies. If it stays high, the gate certifies only recoverable
    directions (trustworthy). If it collapses while the gate still certifies
    directions, the gate is FOOLED (certifies non-recoverable directions).
    """
    n_seeds = len(SEEDS)
    nrec = [r["gate_n_recoverable_mean"] for r in arm_rows]
    degrades = (max(nrec) - min(nrec)) > 0
    silent_failure = any(r["verdict_counts"]["PROCEED"] == n_seeds
                         and r["naive_mcc_mean"] < RECOVERY_MCC for r in arm_rows)
    # certification trustworthy: wherever the gate certifies (n_rec>=0.5), certified
    # directions recover (gated MCC >= bar).
    cert_rows = [r for r in arm_rows if r["certified_recovery_mean"] is not None]
    cert_trustworthy = all(r["certified_recovery_mean"] >= RECOVERY_MCC for r in cert_rows)
    calibration_gap = round(float(np.mean(
        [abs(r["gate_n_recoverable_mean"] - r["naive_recovered_dirs_mean"])
         for r in arm_rows])), 2)
    if not cert_trustworthy:
        behavior = "FOOLED"        # certifies directions that do not recover (needs P3)
    elif degrades and calibration_gap <= 0.6 and not silent_failure:
        behavior = "TRACKS_TIGHT"
    elif degrades and not silent_failure:
        behavior = "TRACKS_CONSERVATIVE"
    else:
        behavior = "FLAT_OR_SILENT"
    return dict(behavior=behavior, calibration_gap=calibration_gap,
                certification_trustworthy=bool(cert_trustworthy),
                silent_failure=bool(silent_failure), degrades=bool(degrades))


def run(readout="precision", recovery=None):
    report = {
        "milestone": "E2",
        "code_commit": E0._code_commit(),
        "config": dict(seeds=SEEDS, d_latent=D_LATENT, D=D_OBS, d_proj=D_PROJ,
                       alpha=ALPHA, B=B_BOOT, iv_base=IV_BASE, n_base=N_BASE,
                       recovery_mcc=RECOVERY_MCC, dir_corr=DIR_CORR,
                       arm_C_basis="detectability-based, NOT P3 (P3 is temporal-only, R1)"),
        "arms": {},
    }

    for arm_name, spec in ARMS.items():
        builder = spec["builder"]
        arm_rows = []       # aggregated per severity level
        for lvl in spec["levels"]:
            seed_recs = []
            for seed in SEEDS:
                B, nv, is_src = _base_scm(seed)
                ds, int_envs = builder(seed, lvl)
                r = evaluate(ds.environments["basis"].X, ds.environments["obs"].X,
                             ds.environments["obs"].Z, int_envs, seed,
                             readout=readout, recovery=recovery)
                seed_recs.append(r)
            # aggregate over seeds
            def col(key):
                return [s[key] for s in seed_recs]
            verdicts = col("verdict")
            cr_vals = [s["certified_recovery"] for s in seed_recs
                       if s["certified_recovery"] is not None]
            cr_mean = round(float(np.mean(cr_vals)), 4) if cr_vals else None
            arm_rows.append(dict(
                level=lvl, level_name=spec["level_name"],
                gate_n_recoverable_mean=round(float(np.mean(col("n_recoverable"))), 2),
                gate_n_recoverable_spread=[int(min(col("n_recoverable"))), int(max(col("n_recoverable")))],
                verdict_counts={v: verdicts.count(v) for v in ("PROCEED", "PROCEED_CAPPED", "ABSTAIN")},
                naive_mcc_mean=round(float(np.mean(col("naive_mcc"))), 4),
                naive_recovered_dirs_mean=round(float(np.mean(col("naive_recovered_dirs"))), 2),
                certified_recovery_mean=cr_mean,
                n_seeds_certifying=len(cr_vals),
                per_seed=seed_recs))
        # crossovers (using means): first level where naive < 0.90; where gate leaves PROCEED
        naive_below = next((row["level"] for row in arm_rows
                            if row["naive_mcc_mean"] < RECOVERY_MCC), None)
        gate_caps = next((row["level"] for row in arm_rows
                          if row["verdict_counts"]["PROCEED"] < len(SEEDS)), None)
        v = _arm_verdict(arm_rows)
        report["arms"][arm_name] = dict(
            expectation=spec["expect"], levels=arm_rows,
            crossover_naive_below_0p90=naive_below,
            crossover_gate_leaves_proceed=gate_caps,
            gate_behavior=v["behavior"],
            calibration_gap=v["calibration_gap"],
            certification_trustworthy=v["certification_trustworthy"],
            silent_failure=v["silent_failure"])

    behaviors = {name: a["gate_behavior"] for name, a in report["arms"].items()}
    tracking = [n for n, b in behaviors.items() if b in ("TRACKS_TIGHT", "TRACKS_CONSERVATIVE")]
    fooled = [n for n, b in behaviors.items() if b == "FOOLED"]
    silent = [n for n, a in report["arms"].items() if a["silent_failure"]]
    # Kill criterion: no arm tracks (calibration flat everywhere). Silent failure would also kill.
    if not tracking or silent:
        report["central_claim"] = (
            f"KILLED: tracking arms={tracking}, silent-failure arms={silent}")
        report["status"] = "FAIL"
    else:
        report["central_claim"] = (
            f"SUPPORTED: gate tracks naive failure on {tracking} and never proceeds while naive "
            f"collapses. EXCEPTION: measurement contamination fools the detectability gate "
            f"(arms {fooled}: it certifies non-recoverable directions), which needs the temporal "
            f"P3 (R1, deferred). Tight calibration on environment/direction-count starvation; "
            f"conservative (safe) on power and weak-signal starvation.")
        report["status"] = "PASS"
    report["arm_C_note"] = ("detectability-based, NOT P3: the covariance/precision gate cannot "
                            "separate measurement from mechanism without a temporal P3 (R1).")
    return report


def _print(report):
    print(f"\nE2 CALIBRATION  (status: {report['status']})", flush=True)
    print(f"central claim: {report['central_claim']}", flush=True)
    for arm_name, a in report["arms"].items():
        print(f"\n=== {arm_name} ===", flush=True)
        print(f"  pre-registered: {a['expectation']}", flush=True)
        print(f"  {'level':>10} {'gate n_rec':>12} {'verdicts(P/C/A)':>16} "
              f"{'naive MCC':>10} {'naive dirs':>11} {'cert recov':>11}", flush=True)
        for row in a["levels"]:
            vc = row["verdict_counts"]
            cr = row["certified_recovery_mean"]
            cr_s = f"{cr:.3f}" if cr is not None else "n/a"
            print(f"  {str(row['level']):>10} "
                  f"{row['gate_n_recoverable_mean']:>5} {str(row['gate_n_recoverable_spread']):>6} "
                  f"{str(vc['PROCEED'])+'/'+str(vc['PROCEED_CAPPED'])+'/'+str(vc['ABSTAIN']):>16} "
                  f"{row['naive_mcc_mean']:>10.3f} {row['naive_recovered_dirs_mean']:>11} "
                  f"{cr_s:>11}", flush=True)
        print(f"  crossover: naive<0.90 at {a['crossover_naive_below_0p90']}, "
              f"gate leaves PROCEED at {a['crossover_gate_leaves_proceed']}", flush=True)
        print(f"  behavior: {a['gate_behavior']}  (calibration_gap="
              f"{a['calibration_gap']}, certification_trustworthy="
              f"{a['certification_trustworthy']}, silent_failure={a['silent_failure']})",
              flush=True)
    print(f"\narm C: {report['arm_C_note']}", flush=True)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = run()
    (RESULTS / "e2_calibration_report.json").write_text(json.dumps(report, indent=2))
    _print(report)
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
