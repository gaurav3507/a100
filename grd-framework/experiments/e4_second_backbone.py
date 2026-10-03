"""E4 (Tier 2, Experiment A): the Gate-Recover-Discover contract with a second backbone.

Two readouts run through the same experiments, chosen with --backbone:

  C (default)  THE SECOND BACKBONE. Recovery by joint approximate diagonalization
               of all precision differences Delta_e = Prec(Y_e) - Prec(Y_obs)
               (backbone.jad_rows: FFDiag, rows matched to the known targets). Gate
               statistic ||Delta_e||_F (precision_readout readout="frobenius").
  covneg       A NEGATIVE CONTROL, not a backbone: the covariance-difference readout
               (gate lambda_max(Cov(Y_0) - Cov(Y_e)); rows v_max of the same). It was
               first run as "Backbone B". The Mac preview of 2026-09-26 showed that its
               recovery rule reaches only MCC 0.898 even with exact population
               covariances, and it was relabelled a negative control AFTER that number
               was seen. Its headline is that population decomposition.

For both, the null (size-matched, disjoint split when n_e <= n_0/2), BH, d_rec, the
verdict rule and every experiment design are unchanged and reached through the
existing entry points; no experiment logic is copied.

Parts:
  population-check  Backbone C only, and before any Backbone C sweep. Population MCC
                    at the E2 reference point must meet the rule declared below (mean
                    >= 0.99 and every seed >= 0.98) or no Backbone C part runs.
  calibration       the four E2 arms (e2_calibration.run), 10 seeds, B=500, plus the
                    same arms with the precision readout on the same code as the
                    comparator, and the population decomposition of all row rules.
  starvation        the E2c random-subset starvation series (e2c_starvation), 10 seeds.
  attribution       Backbone C only, run on the Mac: the five e2b attribution cases
                    with the Frobenius detection statistic (e2b_attribution.run),
                    written to e4_attribution.json in the e2b schema.
  real              gate only (no recovery) on K562 / RPE1 / Norman, corrected
                    disjoint null, BH within family, NMIN=200, d_proj=10, five gate
                    seeds, plus the E3 random pure-control family.

Artifacts: Backbone C writes e4_*.json; the negative control writes e4_negctrl_*.json.

PRE-REGISTERED EXPECTATION (Tier 2 brief): the second backbone's gate restricts before
its own recovery MCC crosses 0.90 under starvation, abstains under power and
weak-signal starvation, and is fooled by measurement contamination. A difference is
recorded, not tuned. A contract difference never makes this script exit non-zero;
only a crash, a malformed artifact, or a missing or failed population check does.

Usage:
    python experiments/e4_second_backbone.py population-check
    python experiments/e4_second_backbone.py attribution
    python experiments/e4_second_backbone.py calibration [--backbone covneg]
    python experiments/e4_second_backbone.py starvation [--backbone covneg]
    python experiments/e4_second_backbone.py real --dataset k562 [--backbone covneg]
    python experiments/e4_second_backbone.py real --dataset k562 --selftest --out-dir /tmp/x
"""

import argparse
import importlib.util
import json
import time
import warnings
from pathlib import Path

import numpy as np

import tier2_common as T2

warnings.filterwarnings("ignore", message=r".*encountered in matmul",
                        category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results" / "e4_second_backbone"
NULL_TEXT = "size-matched; disjoint split when n_e <= n_0/2, else two-resample"
BACKBONES = {
    "C": dict(
        name="Backbone C", role="second backbone", prefix="e4",
        gate_readout="frobenius", recovery="jad",
        gate_statistic="||Prec(Y_e) - Prec(Y_0)||_F",
        recovery_rule=("joint approximate diagonalization (FFDiag, Ziehe et al. 2004) "
                       "of all Delta_e = Prec(Y_e) - Prec(Y_0); unmixing rows are the "
                       "columns of V^-1, matched to the known targets (Hungarian)"),
        null=NULL_TEXT),
    "covneg": dict(
        name="covariance readout", role="negative control", prefix="e4_negctrl",
        gate_readout="covariance", recovery="covariance",
        gate_statistic="lambda_max(Cov(Y_0) - Cov(Y_e))",
        recovery_rule="w_i = v_max(Cov(Y_0) - Cov(Y_e(i))), Z_i = Y_0 w_i",
        null=NULL_TEXT,
        relabel_note=("Run first as 'Backbone B'. Relabelled a negative control on "
                      "2026-09-26 after the Mac preview showed its recovery rule "
                      "reaches MCC 0.898 with exact population covariances; the "
                      "decision was made after that number was seen.")),
}

# Backbone C must recover at the population level before any sweep runs. Declared
# before the first evaluation (2026-09-26) and not changed after it.
POP_CHECK = dict(mean_at_least=0.99, every_seed_at_least=0.98)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_population_check(backbone, out_dir):
    """Stop any Backbone C part unless its population check exists and passed."""
    if backbone != "C":
        return None
    path = Path(out_dir) / "e4_population_check.json"
    if not path.exists():
        raise SystemExit(f"[stop] {path} missing: run the population-check part first")
    check = json.loads(path.read_text())
    if not check.get("passed"):
        raise SystemExit(
            f"[stop] Backbone C population check failed (mean "
            f"{check.get('population_mcc_mean')}, min {check.get('population_mcc_min')}"
            f"; rule {check.get('rule')}); no Backbone C sweep may run")
    return dict(source=str(path.name), sha256=T2.sha256_file(path),
                population_mcc_mean=check["population_mcc_mean"],
                population_mcc_min=check["population_mcc_min"], rule=check["rule"])


def _public(spec):
    return {k: v for k, v in spec.items() if k != "prefix"}


# ------------------------------------------------------------------ population
def _population_setup(E2, seed):
    """Exact projected-space covariances for arm A (m = d) at one seed."""
    SIM, BK = E2.SIM, E2.BK
    ds, int_envs = E2.arm_A(seed, E2.D_LATENT)
    mu, Wp = BK.fit_pca(ds.environments["basis"].X, E2.D_PROJ)
    R = Wp.T @ ds.A
    noise = ds.sd_obs ** 2 * np.eye(E2.D_PROJ)
    C0 = R @ SIM.population_latent_cov(ds.B, ds.noise_var) @ R.T + noise
    nodes = [i for i, _ in int_envs]
    Cs = [R @ SIM.population_latent_cov(ds.B, ds.noise_var, "hard", (i,), E2.IV_BASE)
          @ R.T + noise for i in nodes]
    Y_obs = BK.project(ds.environments["obs"].X, mu, Wp)
    Y_int = [BK.project(X, mu, Wp) for _, X in int_envs]
    return ds, R, C0, Cs, nodes, Y_obs, Y_int


def population_check(E2):
    """Backbone C population MCC at the E2 reference point, plus the JAD diagnostic.

    For each seed: exact population precisions of the projected data, joint
    diagonalization of the population Delta_e, Hungarian MCC of the resulting
    unmixing on the observational sample. The off-diagonal ratio at the true
    unmixing (V = R^T) against the ratio at the JAD solution separates a model
    mismatch (the truth is not the joint-diagonal optimum) from an optimizer
    failure (the solution is worse than the truth).
    """
    BK, E0 = E2.BK, E2.E0
    d = E2.D_LATENT
    rows = []
    for seed in E2.SEEDS:
        ds, R, C0, Cs, nodes, Y_obs, _ = _population_setup(E2, seed)
        P0 = np.linalg.inv(C0)
        Ps = [np.linalg.inv(C) for C in Cs]
        W, info = BK.jad_rows(P0, Ps, nodes, d)
        deltas = [P - P0 for P in Ps]
        rows.append(dict(
            seed=seed,
            population_mcc=round(float(E0.mcc(Y_obs @ W.T, ds.environments["obs"].Z)), 4),
            offdiag_ratio_at_solution=float(info["offdiag_ratio"]),
            offdiag_ratio_at_truth=float(BK.offdiag_ratio(R.T, deltas)),
            n_iter=info["n_iter"], converged=info["converged"],
            n_parents_per_node=[int(v) for v in (np.abs(ds.B) > 0).sum(1)],
        ))
    mccs = [r["population_mcc"] for r in rows]
    passed = bool(np.mean(mccs) >= POP_CHECK["mean_at_least"]
                  and min(mccs) >= POP_CHECK["every_seed_at_least"])
    return dict(
        setting="arm A, m=5, n=2000, iv_scale=0.1, seeds 0-9, exact population precisions",
        rule=POP_CHECK,
        population_mcc_mean=round(float(np.mean(mccs)), 4),
        population_mcc_min=min(mccs),
        passed=passed,
        jad=dict(algorithm="FFDiag (Ziehe et al. 2004)", theta=BK.JAD_THETA,
                 tol=BK.JAD_TOL, max_iter=BK.JAD_MAX_ITER,
                 init="eigenvectors of sum_e Delta_e"),
        diagnostic=("offdiag_ratio_at_solution < offdiag_ratio_at_truth means the "
                    "truth is not the exact joint-diagonal optimum (model mismatch "
                    "from the removed-edge term), not an optimizer failure"),
        per_seed=rows,
    )


def run_population_check(out_dir):
    E2 = _load(REPO / "experiments" / "e2_calibration.py", "grd_e2_for_e4_pop")
    started = time.time()
    check = population_check(E2)
    check["experiment"] = "e4_backbone_c_population_check"
    check["provenance"] = T2.provenance(
        __file__, extra_files=["experiments/e2_calibration.py"])
    check["wall_seconds"] = round(time.time() - started, 1)
    out = Path(out_dir) / "e4_population_check.json"
    T2.write_json(out, check)
    print("\nBACKBONE C POPULATION CHECK (declared: mean >= "
          f"{POP_CHECK['mean_at_least']}, every seed >= "
          f"{POP_CHECK['every_seed_at_least']})", flush=True)
    for r in check["per_seed"]:
        print(f"  seed {r['seed']}: MCC {r['population_mcc']:.4f}  off-diagonal ratio "
              f"solution {r['offdiag_ratio_at_solution']:.2e} vs truth "
              f"{r['offdiag_ratio_at_truth']:.2e}  iters {r['n_iter']} "
              f"converged {r['converged']}", flush=True)
    print(f"  mean {check['population_mcc_mean']}  min {check['population_mcc_min']}  "
          f"-> {'PASS' if check['passed'] else 'FAIL: Backbone C sweeps must not run'}",
          flush=True)
    print(f"written {out}", flush=True)
    if not check["passed"]:
        raise SystemExit(1)


def population_decomposition(E2):
    """Spec-versus-implementation split for all three row rules at the E2 reference.

    Arm A with all five environments (n=2000, iv_scale=0.1), seeds 0-9. Each rule is
    applied to the EXACT population covariances of the projected data
    (Cov(Y_e) = R Sigma_e R^T + sd_obs^2 I, R = W_pca^T A) and to the sample
    covariances, as the backbone does; both are scored with the same E0 Hungarian
    MCC on the same observational sample. Population below the 0.90 bar means the
    rule itself falls short (a specification property); sample far below population
    would mean an estimation or implementation problem.
    """
    BK, E0 = E2.BK, E2.E0
    d = E2.D_LATENT
    rows = []
    for seed in E2.SEEDS:
        ds, R, C0, Cs, nodes, Y_obs, Y_int = _population_setup(E2, seed)
        Z = ds.environments["obs"].Z
        W = {k: np.zeros((d, d)) for k in ("pop_cov", "pop_prec", "smp_cov", "smp_prec")}
        for i, Ce, Y_env in zip(nodes, Cs, Y_int):
            vals, vecs = np.linalg.eigh(C0 - Ce)
            W["pop_cov"][i] = vecs[:, int(np.argmax(vals))]
            vals, vecs = np.linalg.eigh(np.linalg.inv(Ce) - np.linalg.inv(C0))
            W["pop_prec"][i] = vecs[:, int(np.argmax(vals))]
            W["smp_cov"][i] = BK._unmixing_row(Y_obs, Y_env, readout="covariance")
            W["smp_prec"][i] = BK._unmixing_row(Y_obs, Y_env, readout="precision")
        W["pop_jad"], _ = BK.jad_rows(np.linalg.inv(C0),
                                      [np.linalg.inv(C) for C in Cs], nodes, d)
        W["smp_jad"] = BK.unmixing_rows(Y_obs, Y_int, nodes, d, readout="jad")
        rows.append(dict(seed=seed, **{k: round(float(E0.mcc(Y_obs @ w.T, Z)), 4)
                                       for k, w in W.items()}))

    def stat(key):
        v = [r[key] for r in rows]
        return dict(mean=round(float(np.mean(v)), 4), min=min(v), max=max(v))
    return dict(
        setting="arm A, m=5, n=2000, iv_scale=0.1, seeds 0-9",
        jad_rule=dict(population=stat("pop_jad"), sample=stat("smp_jad")),
        covariance_rule=dict(population=stat("pop_cov"), sample=stat("smp_cov")),
        precision_rule=dict(population=stat("pop_prec"), sample=stat("smp_prec")),
        per_seed=rows,
    )


# ------------------------------------------------------------------ calibration
def _arm_contract(arm):
    """Did this arm keep the precision-gate contract? Descriptive, from the numbers."""
    return dict(
        gate_behavior=arm["gate_behavior"],
        silent_failure=bool(arm["silent_failure"]),
        certification_trustworthy=bool(arm["certification_trustworthy"]),
        crossover_gate_leaves_proceed=arm["crossover_gate_leaves_proceed"],
        crossover_naive_below_0p90=arm["crossover_naive_below_0p90"],
    )


def run_calibration(backbone, out_dir):
    spec = BACKBONES[backbone]
    pop = _require_population_check(backbone, out_dir)
    E2 = _load(REPO / "experiments" / "e2_calibration.py", "grd_e2_for_e4")
    started = time.time()
    report = E2.run(readout=spec["gate_readout"], recovery=spec["recovery"])
    precision = E2.run(readout="precision")
    decomposition = population_decomposition(E2)

    behaviors = {n: a["gate_behavior"] for n, a in report["arms"].items()}
    precision_behaviors = {n: a["gate_behavior"] for n, a in precision["arms"].items()}
    tracking = [n for n, b in behaviors.items()
                if b in ("TRACKS_TIGHT", "TRACKS_CONSERVATIVE")]
    fooled = [n for n, b in behaviors.items() if b == "FOOLED"]
    silent = [n for n, a in report["arms"].items() if a["silent_failure"]]
    differs = [n for n in behaviors if behaviors[n] != precision_behaviors[n]]

    power = report["arms"]["B_power_starvation"]
    weak = report["arms"]["D_weak_signal_starvation"]
    contam = report["arms"]["C_measurement_contamination"]
    expectation = dict(
        power_starvation_restricts_without_silent_failure=bool(
            power["gate_behavior"] in ("TRACKS_TIGHT", "TRACKS_CONSERVATIVE")
            and not power["silent_failure"]),
        weak_signal_restricts_without_silent_failure=bool(
            weak["gate_behavior"] in ("TRACKS_TIGHT", "TRACKS_CONSERVATIVE")
            and not weak["silent_failure"]),
        measurement_contamination_fools_gate=bool(
            contam["gate_behavior"] == "FOOLED"),
    )

    head = dict(role=spec["role"])
    if backbone == "covneg":
        cov = decomposition["covariance_rule"]
        prec = decomposition["precision_rule"]
        head["headline"] = dict(
            statement=(
                f"Negative control. The covariance recovery rule reaches MCC "
                f"{cov['population']['mean']} with exact population covariances "
                f"({cov['sample']['mean']} from samples), against "
                f"{prec['population']['mean']} for the precision rule, so it sits "
                f"{'below' if cov['population']['mean'] < E2.RECOVERY_MCC else 'at or above'}"
                f" the {E2.RECOVERY_MCC} recovery bar before any sampling error. Its "
                f"arm verdicts measure that bias, not gate calibration."),
            covariance_rule=cov, precision_rule=prec,
            relabel_note=spec["relabel_note"])
    else:
        head["population_check"] = pop
    report = head | report
    report["milestone"] = "E4"
    report["experiment"] = f"{spec['prefix']}_calibration"
    report["config"]["readout"] = spec["gate_readout"]
    report["config"]["recovery"] = spec["recovery"]
    report["config"]["backbone"] = _public(spec)
    # "status" keeps its E2 meaning (PASS unless no arm tracks or an arm fails
    # silently). A FAIL is a finding about this readout, so the script still exits
    # 0; the launcher checks the artifact, not the verdict.
    report["central_claim"] = (
        f"{spec['name']} ({spec['role']}) on the E2 arms: behaviors {behaviors}; "
        f"tracking arms {tracking}; fooled arms {fooled}; silent-failure arms "
        f"{silent}. Same-code precision behaviors {precision_behaviors}. Arms where "
        f"it differs from precision: {differs} (recorded, not tuned).")
    report["preregistered_expectation"] = dict(
        statement=("gate abstains or caps under power and weak-signal starvation "
                   "and is fooled by measurement contamination, as the precision "
                   "gate is"),
        observed=expectation,
        all_held=bool(all(expectation.values())),
    )
    report["precision_same_code"] = dict(
        note=("precision readout run in this job on the same code; the comparator "
              "for every arm"),
        contract_status=precision["status"],
        arms={n: _arm_contract(a) for n, a in precision["arms"].items()},
        level_means={
            n: [dict(level=lv["level"],
                     gate_n_recoverable_mean=lv["gate_n_recoverable_mean"],
                     naive_mcc_mean=lv["naive_mcc_mean"],
                     certified_recovery_mean=lv["certified_recovery_mean"])
                for lv in a["levels"]]
            for n, a in precision["arms"].items()},
    )
    report["comparison"] = {
        n: dict(this_readout=_arm_contract(report["arms"][n]),
                precision=_arm_contract(precision["arms"][n]),
                same_behavior=behaviors[n] == precision_behaviors[n])
        for n in behaviors
    }
    report["population_decomposition"] = decomposition
    report["provenance"] = T2.provenance(
        __file__, extra_files=["experiments/e2_calibration.py"])
    report["wall_seconds"] = round(time.time() - started, 1)
    out = Path(out_dir) / f"{spec['prefix']}_calibration_report.json"
    T2.write_json(out, report)

    print(f"\nE4 CALIBRATION ({spec['name']}, {spec['role']})", flush=True)
    for n in behaviors:
        a = report["arms"][n]
        print(f"  {n:30s} this={behaviors[n]:20s} precision="
              f"{precision_behaviors[n]:20s} gate leaves PROCEED at "
              f"{a['crossover_gate_leaves_proceed']}, naive<0.90 at "
              f"{a['crossover_naive_below_0p90']}", flush=True)
        for lv in a["levels"]:
            vc = lv["verdict_counts"]
            print(f"      level {str(lv['level']):>6}: gate n_rec {lv['gate_n_recoverable_mean']:>4} "
                  f"{lv['gate_n_recoverable_spread']}  P/C/A {vc['PROCEED']}/"
                  f"{vc['PROCEED_CAPPED']}/{vc['ABSTAIN']}  naive MCC "
                  f"{lv['naive_mcc_mean']:.4f}  cert {lv['certified_recovery_mean']}",
                  flush=True)
    print(f"  expectation held: {expectation}", flush=True)
    for rule in ("jad_rule", "covariance_rule", "precision_rule"):
        dd = decomposition[rule]
        print(f"  {rule}: MCC with population covariances {dd['population']['mean']} "
              f"| with sample covariances {dd['sample']['mean']}", flush=True)
    print(f"written {out} ({report['wall_seconds']}s)", flush=True)


# ------------------------------------------------------------------ starvation
def run_starvation(backbone, out_dir, smoke=False):
    spec = BACKBONES[backbone]
    pop = _require_population_check(backbone, out_dir)
    E2C = _load(REPO / "experiments" / "e2c_starvation.py", "grd_e2c_for_e4")
    seeds = E2C.SMOKE_SEEDS if smoke else E2C.FULL_SEEDS
    B = E2C.SMOKE_BOOT if smoke else E2C.FULL_BOOT
    started = time.time()
    rows = []
    for seed in seeds:
        rows.append(E2C.evaluate_seed(seed, B, readout=spec["gate_readout"],
                                      recovery=spec["recovery"]))
        print(f"[seed {seed}] done ({time.time() - started:.1f}s)", flush=True)
    agg = E2C.aggregate(rows)
    wiring_pass = bool(
        agg["estimator_checks"]["full_rank_all"]
        and agg["estimator_checks"]["zero_or_rank_deficient_output_count"] == 0
        and np.isfinite([r["random_subset_control"]["mcc_mean"]
                         for r in agg["levels"]]).all())

    reference = {}
    ref_path = REPO / "results" / "e2c" / "starvation_report.json"
    if ref_path.exists() and not smoke:
        ref = json.loads(ref_path.read_text())["aggregate"]
        reference = dict(
            source=str(ref_path.relative_to(REPO)),
            sha256=T2.sha256_file(ref_path),
            note=("precision readout; current code reproduces its decisions exactly "
                  "(floating-point values agree to about 1e-11 across OS/BLAS "
                  "versions)"),
            crossover_random_subset_mcc_below_0p90=ref[
                "crossover_random_subset_mcc_below_0p90"],
            crossover_gate_first_restricts=ref["crossover_gate_first_restricts"],
            gate_restricts_at_or_before_mcc_crossover=ref[
                "gate_restricts_at_or_before_mcc_crossover"],
            random_subset_mcc_mean=[r["random_subset_control"]["mcc_mean"]
                                    for r in ref["levels"]],
        )

    report = dict(
        role=spec["role"],
        experiment=f"{spec['prefix']}_starvation",
        mode="smoke" if smoke else "full",
        result_eligible=not smoke,
        status="PASS" if wiring_pass else "FAIL",
        scientific_conclusion=(
            "Reduced-power wiring check only; no paper conclusion." if smoke else
            "See measured crossovers; a difference from the precision backbone is "
            "a finding, not tuned."),
        config=dict(
            seeds=seeds, B=B, alpha=E2C.ALPHA, d_latent=E2C.D_LATENT,
            D_observed=E2C.D_OBS, d_projected=E2C.D_PROJ,
            n_per_environment=E2C.N_BASE, iv_scale=E2C.IV_BASE,
            edge_prob=E2C.EDGE_PROB, starvation_levels=E2C.STARVATION_LEVELS,
            max_random_subsets=E2C.MAX_SUBSETS, mcc_threshold=E2C.MCC_THRESHOLD,
            readout=spec["gate_readout"], recovery=spec["recovery"],
            backbone=_public(spec)),
        estimator=(
            f"{spec['name']} rows for the supplied targets"
            + (" (joint fit refitted on each supplied subset)"
               if spec["recovery"] == "jad" else "")
            + " plus the unchanged E2c absolute-eigenvalue spectral completion of the "
            "pooled interventional-minus-control covariance in their orthogonal "
            "complement."),
        primary_metric="Hungarian MCC; permutation and sign safe, axis-sensitive.",
        preregistered_expectation=dict(
            statement="gate restricts no later than the first m where mean "
                      "random-subset MCC falls below 0.90",
            observed=agg["gate_restricts_at_or_before_mcc_crossover"]),
        population_check=pop,
        headline_source=(f"{BACKBONES['covneg']['prefix']}_calibration_report.json "
                         "headline" if backbone == "covneg" else None),
        precision_reference=reference or None,
        aggregate=agg,
        per_seed=rows,
        provenance=T2.provenance(
            __file__, extra_files=["experiments/e2c_starvation.py"]),
        wall_seconds=round(time.time() - started, 1),
    )
    suffix = "starvation_smoke.json" if smoke else "starvation_report.json"
    out = Path(out_dir) / f"{spec['prefix']}_{suffix}"
    T2.write_json(out, report)
    print(f"\nE4 STARVATION ({spec['name']}, {spec['role']})", flush=True)
    print("m  gate-n  MCC-mean  MCC-seed-SD  verdicts(P/C/A)", flush=True)
    for row in agg["levels"]:
        c = row["random_subset_control"]
        vc = c["verdict_counts"]
        print(f"{row['m']:d}  {c['gate_n_recoverable_mean']:6.2f}  {c['mcc_mean']:8.4f}"
              f"  {c['mcc_seed_sd']:11.4f}  {vc['PROCEED']}/{vc['PROCEED_CAPPED']}/"
              f"{vc['ABSTAIN']}", flush=True)
    print(f"crossover: gate first restricts at m={agg['crossover_gate_first_restricts']}; "
          f"MCC first below 0.90 at m={agg['crossover_random_subset_mcc_below_0p90']}; "
          f"restricts at or before: {agg['gate_restricts_at_or_before_mcc_crossover']}",
          flush=True)
    print(f"status: {report['status']}  written {out}", flush=True)
    if not wiring_pass:
        raise SystemExit(1)


# ------------------------------------------------------------------ real panel
def run_real(backbone, key, out_dir, seeds, B, selftest=False):
    spec = BACKBONES[backbone]
    readout = spec["gate_readout"]
    pop = None if selftest else _require_population_check(backbone, out_dir)
    started = time.time()
    panel = T2.make_selftest_panel(key) if selftest else T2.load_panel(key)
    name = panel["name"]
    Xc, perts = panel["Xc"], panel["perts"]
    proj = T2.control_projection(Xc)
    Yobs = proj(Xc)
    Yperts = [proj(X) for X in panel["Xperts"]]
    sizes = [int(len(Y)) for Y in Yperts]
    print(f"{name}: control {len(Xc)} | powered perts {len(perts)} | "
          f"{spec['name']} ({spec['role']})", flush=True)

    stab = None if selftest else T2.e3_stability_sets(name)
    dec = None if selftest else T2.e3_decision_set(name)

    per_seed = []
    for seed in seeds:
        res = T2.screen(Yperts, Yobs, seed, readout=readout, B=B)
        fake_labels, Yfake = T2.random_controls(Xc, proj, sizes, seed, T2.N_FAKE)
        fres = T2.screen(Yfake, Yobs, T2.FAKE_SEED_OFFSET + seed,
                         readout=readout, B=B)
        bh_ids = T2.selected(perts, res["bh_detect"])
        raw_ids = T2.selected(perts, res["raw_detect"])
        precision_bh = (stab["per_seed"].get(seed, {}).get("bh")
                        if stab else None)
        per_seed.append(dict(
            seed=seed,
            perturbation_screen=T2.summarize(res),
            negative_control_screen=T2.summarize(fres),
            bh_selected_ids=bh_ids,
            raw_selected_ids=raw_ids,
            jaccard_bh_vs_precision=dict(
                e3_decisions_seed0=(T2.jaccard(bh_ids, dec["bh"]) if dec else None),
                e3_stability_same_seed=(T2.jaccard(bh_ids, precision_bh)
                                        if precision_bh is not None else None),
                precision_bh_count_same_seed=(len(precision_bh)
                                              if precision_bh is not None else None),
            ),
            per_perturbation=T2.design_records(perts, Yperts, {readout: res}),
            negative_control_records=T2.design_records(
                fake_labels, Yfake, {readout: fres}),
        ))
        print(f"  seed {seed}: raw {res['raw_count']}/{len(perts)}  BH "
              f"{res['bh_count']}  | random controls raw {fres['raw_count']}/"
              f"{T2.N_FAKE} BH {fres['bh_count']}  ({time.time() - started:.0f}s)",
              flush=True)

    bh_sets = [set(r["bh_selected_ids"]) for r in per_seed]
    block = dict(
        dataset=name,
        role=spec["role"],
        source=panel["source"],
        n_cells=panel["n_cells"],
        n_genes=panel["n_genes"],
        control_label=panel["control_label"],
        n_control=int(len(Xc)),
        n_environments=len(perts),
        seeds=list(seeds),
        raw_count_per_seed=[r["perturbation_screen"]["raw_count"] for r in per_seed],
        bh_count_per_seed=[r["perturbation_screen"]["bh_count"] for r in per_seed],
        bh_selected_ids_per_seed={str(r["seed"]): r["bh_selected_ids"]
                                  for r in per_seed},
        bh_selected_all_seeds=sorted(set.intersection(*bh_sets)) if bh_sets else [],
        bh_selected_any_seed=sorted(set.union(*bh_sets)) if bh_sets else [],
        raw_fraction=T2.seed_sd([r["perturbation_screen"]["raw_fraction"]
                                 for r in per_seed]),
        bh_fraction=T2.seed_sd([r["perturbation_screen"]["bh_fraction"]
                                for r in per_seed]),
        negative_control_raw_count_per_seed=[
            r["negative_control_screen"]["raw_count"] for r in per_seed],
        negative_control_bh_count_per_seed=[
            r["negative_control_screen"]["bh_count"] for r in per_seed],
        precision_reference=dict(
            e3_decisions=({k: dec[k] for k in ("source", "sha256")}
                          | dict(bh_count=len(dec["bh"]),
                                 labels_match=dec["labels"] == perts)) if dec else None,
            e3_stability=({k: stab[k] for k in ("source", "sha256")}
                          | dict(labels_match=all(
                              v["labels"] == perts for v in stab["per_seed"].values())))
            if stab else None,
        ),
        population_check=pop,
        selftest=bool(selftest),
        per_seed=per_seed,
        provenance=T2.provenance(__file__),
        wall_seconds=round(time.time() - started, 1),
    )
    top = dict(
        experiment=f"{spec['prefix']}_real_panel",
        role=spec["role"],
        description=(f"{spec['name']} ({spec['role']}) gate only, no recovery: "
                     f"{spec['gate_statistic']}, corrected disjoint null, BH within "
                     "each family, raw alpha decisions kept for audit."),
        config=dict(readout=readout, backbone=_public(spec), d_proj=T2.D_PROJ,
                    nmin=T2.NMIN, alpha=T2.ALPHA, q=T2.Q, B=B,
                    null="corrected_disjoint", design="shared_reference",
                    seeds=list(seeds), n_random_controls=T2.N_FAKE,
                    bh_families=("BH applied separately to powered perturbations "
                                 "and random controls, within each seed"),
                    jaccard_convention="None when both BH sets are empty"),
    )
    out = Path(out_dir) / f"{spec['prefix']}_real_panel.json"
    T2.update_dataset_block(out, top, name, block)
    print(f"written {out} [{name}] ({block['wall_seconds']}s)", flush=True)


# ------------------------------------------------------------------ attribution
# Backbone C on the five e2b attribution cases. Only the detection statistic
# differs from e2b (||Delta_e||_F instead of lambda_max); the subspace attribution
# step, the environments and every RNG stream are the same. Expectations written
# before the run (2026-09-27). ||.||_F is two-sided, so the lambda_max
# one-sidedness result (Prop 1: variance inflation is invisible to the screen) does
# not carry over.
ATTRIBUTION_EXPECT_FROBENIUS = {
    "mechanism_hard_iv": ("detected ~1.0; UNATTRIBUTED|detected as in e2b "
                          "(attribution step unchanged)"),
    "mixed_gain_u05_15": "detected high; UNATTRIBUTED|detected ~1.0 as in e2b",
    "uniform_gain_0.7": ("detected ~1.0; verdict as in e2b (attribution step "
                         "unchanged)"),
    "uniform_gain_1.0": "detected ~alpha (identity sanity)",
    "uniform_gain_1.4": ("detected ~1.0, NOT ~0 as for lambda_max: ||Delta||_F is "
                         "two-sided, so inflation is visible; the Prop 1 "
                         "one-sidedness does not carry over"),
}


def run_attribution(out_dir):
    spec = BACKBONES["C"]
    E2 = _load(REPO / "experiments" / "e2_calibration.py", "grd_e2_for_e4_attr")
    check = population_check(E2)
    if not check["passed"]:
        raise SystemExit("[stop] Backbone C population check failed; attribution "
                         "check not run")
    E2B = _load(REPO / "experiments" / "e2b_attribution.py", "grd_e2b_for_e4")
    started = time.time()
    report = E2B.run(readout=spec["gate_readout"])

    for name, s in report["summary"].items():
        s["expected_precision_e2b"] = s["expected"]
        s["expected"] = ATTRIBUTION_EXPECT_FROBENIUS[name]

    comparison = None
    ref_path = REPO / "results" / "e2b" / "attribution_report.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        ref_rows = {(r["seed"], r["case"]): r for r in ref["rows"]}
        per_case = {}
        for name, s in report["summary"].items():
            rs = [r for r in report["rows"] if r["case"] == name]
            both = [r for r in rs if r["detected"]
                    and ref_rows[(r["seed"], r["case"])]["detected"]]
            per_case[name] = dict(
                detected_rate=dict(frobenius=s["detected_rate"],
                                   precision=ref["summary"][name]["detected_rate"]),
                unattributed_rate_given_detected=dict(
                    frobenius=s["unattributed_rate_given_detected"],
                    precision=ref["summary"][name]["unattributed_rate_given_detected"]),
                n_detected_by_both=len(both),
                verdict_agreement_when_both_detected=(
                    sum(r["verdict"] == ref_rows[(r["seed"], r["case"])]["verdict"]
                        for r in both)),
            )
        comparison = dict(source=str(ref_path.relative_to(REPO)),
                          sha256=T2.sha256_file(ref_path), per_case=per_case)

    report = dict(
        role=spec["role"],
        gate_readout=spec["gate_readout"],
        backbone=_public(spec),
        two_sided_note=("||Delta_e||_F is two-sided: it reacts to precision "
                        "increases and decreases alike, so the lambda_max "
                        "one-sidedness result (variance inflation invisible, "
                        "Prop 1) does not carry over to Backbone C."),
        rows_note=("rows[].precision_crit keeps its e2b key name but holds the "
                   "gate threshold of gate_readout"),
        population_check=dict(population_mcc_mean=check["population_mcc_mean"],
                              population_mcc_min=check["population_mcc_min"],
                              passed=check["passed"], rule=check["rule"]),
        comparison_to_precision_e2b=comparison,
    ) | report
    report["experiment"] = "e4_attribution"
    report["provenance"] = T2.provenance(
        __file__, extra_files=["experiments/e2b_attribution.py",
                               "experiments/e2_calibration.py"])
    report["wall_seconds"] = round(time.time() - started, 1)
    out = Path(out_dir) / "e4_attribution.json"
    T2.write_json(out, report)
    print(f"\nE4 ATTRIBUTION (Backbone C, gate readout {spec['gate_readout']})",
          flush=True)
    E2B.print_table(report)
    if comparison:
        print("\ncase                    detected F/P   unatt|det F/P   verdict "
              "agreement when both detected", flush=True)
        for name, c in comparison["per_case"].items():
            print(f"{name:22s}  {c['detected_rate']['frobenius']}/"
                  f"{c['detected_rate']['precision']}   "
                  f"{c['unattributed_rate_given_detected']['frobenius']}/"
                  f"{c['unattributed_rate_given_detected']['precision']}   "
                  f"{c['verdict_agreement_when_both_detected']}/"
                  f"{c['n_detected_by_both']}", flush=True)
    print(f"written {out} ({report['wall_seconds']}s)", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="part", required=True)
    sub.add_parser("population-check").add_argument("--out-dir", default=str(RESULTS))
    sub.add_parser("attribution").add_argument("--out-dir", default=str(RESULTS))
    for part in ("calibration", "starvation", "real"):
        sp = sub.add_parser(part)
        sp.add_argument("--backbone", choices=sorted(BACKBONES), default="C",
                        help="C: second backbone (default); covneg: negative control")
        sp.add_argument("--out-dir", default=str(RESULTS))
        if part == "starvation":
            sp.add_argument("--smoke", action="store_true",
                            help="three seeds, B=100; writes a *_starvation_smoke.json")
        if part == "real":
            sp.add_argument("--dataset", choices=sorted(T2.DATASETS), required=True)
            sp.add_argument("--selftest", action="store_true",
                            help="synthetic panel, code path only; needs --out-dir")
            sp.add_argument("--seeds", type=int, nargs="+", default=T2.SEEDS)
            sp.add_argument("--B", type=int, default=T2.B_BOOT)
    return p.parse_args()


def main():
    args = parse_args()
    if getattr(args, "selftest", False) and Path(args.out_dir).resolve() == RESULTS:
        raise SystemExit("--selftest must not write under results/; pass --out-dir")
    if args.part == "population-check":
        run_population_check(args.out_dir)
    elif args.part == "attribution":
        run_attribution(args.out_dir)
    elif args.part == "calibration":
        run_calibration(args.backbone, args.out_dir)
    elif args.part == "starvation":
        run_starvation(args.backbone, args.out_dir, smoke=args.smoke)
    else:
        run_real(args.backbone, args.dataset, args.out_dir, args.seeds, args.B,
                 selftest=args.selftest)


if __name__ == "__main__":
    main()
