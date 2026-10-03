"""E2e: finite-sample feasibility of the recovery-direction certificate.

This experiment asks whether the Gaussian/Wishart plus Davis-Kahan certificate
in ``src/recover/certificate.py`` is informative at GRD's current synthetic
sample size. It is a feasibility study, not a new assumption-free claim.

Design fixed before the archived full run (after a small implementation pilot):

* linear-Gaussian simulator, d=5, D=200, PCA fitted on an independent basis;
* reduced-variance hard interventions on every latent, as in E1;
* balanced per-environment n in 200, 500, 1000, 2000, 4000, 8000, 16000;
* 10 seeds, five directions per seed, gate B=500, familywise delta=0.05;
* one cached gate null per seed/sample size because all five environments have
  the same size and reference distribution;
* the certificate jointly covers the six distinct covariance matrices (one
  control and five interventions) within each seed/sample-size family.

The reference operating point is n=2000, matching E1/E2. The certificate is
called useful there only if: at least 80 percent of directions pass the gate,
at least 80 percent of those have a nonvacuous bound, nonvacuous empirical
coverage is at least 95 percent, and the median nonvacuous angle bound is at
most 15 degrees. Failure is reported as VACUOUS_AT_REFERENCE_N, not as a code
failure and not as a reason to tune the theorem after seeing the result.

The theorem's target is the population precision-difference eigenvector. The
script also reports its angle to the true unmixing row as model bias. The latter
is not covered by Davis-Kahan unless an exact population-alignment assumption is
added.

Use ``--smoke`` for three seeds, n in {2000, 8000}, and B=100. Smoke output is
not decision-eligible and is not written into the full result path.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np


warnings.filterwarnings(
    "ignore", message=r".*encountered in matmul", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = REPO / "results" / "e2e_certificate"

FULL_SEEDS = list(range(10))
SMOKE_SEEDS = list(range(3))
FULL_SAMPLE_SIZES = [200, 500, 1000, 2000, 4000, 8000, 16000]
SMOKE_SAMPLE_SIZES = [2000, 8000]
FULL_BOOT = 500
SMOKE_BOOT = 100
D_LATENT = 5
D_OBS = 200
D_PROJ = 5
EDGE_PROB = 0.4
OBS_NOISE_FRAC = 0.1
IV_SCALE = 0.1
ALPHA = 0.05
FAMILYWISE_DELTA = 0.05
SIMULTANEOUS_MATRICES = D_LATENT + 1
REFERENCE_N = 2000

USEFULNESS_CRITERIA = dict(
    gate_pass_rate_min=0.80,
    nonvacuous_among_gate_min=0.80,
    nonvacuous_coverage_min=0.95,
    median_angle_bound_deg_max=15.0,
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SIM = _load(REPO / "sim" / "simulator.py", "grd_sim_e2e")
PR = _load(REPO / "src" / "gate" / "precision_readout.py", "grd_gate_e2e")
BK = _load(REPO / "src" / "recover" / "backbone.py", "grd_backbone_e2e")
CERT = _load(
    REPO / "src" / "recover" / "certificate.py", "grd_certificate_e2e")


def projective_angle_deg(a, b):
    """Acute angle between one-dimensional subspaces in degrees."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cosine = float(np.clip(abs(a @ b), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_family(seed, n_per_env, B_boot):
    specs = [
        SIM.EnvSpec("basis", None, ()),
        SIM.EnvSpec("obs", None, ()),
    ] + [
        SIM.EnvSpec(f"iv{target}", "hard", (target,), IV_SCALE)
        for target in range(D_LATENT)
    ]
    dataset = SIM.simulate(
        D_LATENT,
        D_OBS,
        n_per_env,
        specs,
        seed,
        mixing="linear",
        edge_prob=EDGE_PROB,
        obs_noise_frac=OBS_NOISE_FRAC,
    )
    mu, projection = BK.fit_pca(dataset.environments["basis"].X, D_PROJ)
    Y_obs = BK.project(dataset.environments["obs"].X, mu, projection)
    Y_envs = [
        BK.project(dataset.environments[f"iv{target}"].X, mu, projection)
        for target in range(D_LATENT)
    ]

    # All intervention environments have the same n and share a control. Reuse
    # one Monte Carlo null rather than estimating the same quantile five times.
    gate_threshold = PR.null_threshold(
        Y_obs,
        alpha=ALPHA,
        B=B_boot,
        rng=np.random.default_rng(700_000_000 + seed * 100_000 + n_per_env),
        n_env=n_per_env,
    )

    mixing_projected = projection.T @ dataset.A
    true_unmixing = np.linalg.inv(mixing_projected)
    identity = np.eye(D_PROJ)
    covariance_obs = (
        mixing_projected
        @ dataset.environments["obs"].pop_cov
        @ mixing_projected.T
        + dataset.sd_obs ** 2 * identity
    )
    precision_obs = np.linalg.inv(covariance_obs)
    sample_precision_obs = np.linalg.inv(np.cov(Y_obs, rowvar=False))

    rows = []
    for target, Y_env in enumerate(Y_envs):
        signal = PR.precision_signal(Y_env, Y_obs)
        gate_pass = bool(signal > gate_threshold)
        certificate = CERT.precision_direction_certificate(
            Y_env,
            Y_obs,
            delta=FAMILYWISE_DELTA,
            simultaneous_matrices=SIMULTANEOUS_MATRICES,
        )

        covariance_env = (
            mixing_projected
            @ dataset.environments[f"iv{target}"].pop_cov
            @ mixing_projected.T
            + dataset.sd_obs ** 2 * identity
        )
        population_difference = np.linalg.inv(covariance_env) - precision_obs
        precision_env = np.linalg.inv(covariance_env)
        sample_precision_env = np.linalg.inv(np.cov(Y_env, rowvar=False))
        obs_precision_error = float(np.linalg.norm(
            sample_precision_obs - precision_obs, ord=2))
        env_precision_error = float(np.linalg.norm(
            sample_precision_env - precision_env, ord=2))
        difference_error = float(np.linalg.norm(
            (sample_precision_env - sample_precision_obs)
            - population_difference,
            ord=2,
        ))
        _, population_vectors = np.linalg.eigh(population_difference)
        population_direction = population_vectors[:, -1]
        sample_direction = certificate["direction"]
        latent_direction = true_unmixing[target]

        sample_population_angle = projective_angle_deg(
            sample_direction, population_direction)
        population_latent_bias = projective_angle_deg(
            population_direction, latent_direction)
        sample_latent_angle = projective_angle_deg(
            sample_direction, latent_direction)
        angle_bound = float(certificate["angle_bound_deg"])
        oracle_total_bound = float(min(
            90.0, angle_bound + population_latent_bias))

        rows.append(dict(
            seed=int(seed),
            n_per_env=int(n_per_env),
            target=int(target),
            gate_signal=float(signal),
            gate_threshold=float(gate_threshold),
            gate_pass=gate_pass,
            certificate_status=certificate["status"],
            gap_separated=bool(certificate["gap_separated"]),
            nonvacuous=bool(certificate["nonvacuous"]),
            empirical_gap=float(certificate["empirical_gap"]),
            perturbation_bound=_finite_or_none(
                certificate["perturbation_bound"]),
            obs_precision_error=obs_precision_error,
            env_precision_error=env_precision_error,
            precision_events_covered=bool(
                obs_precision_error <= certificate["obs_error_bound"] + 1e-10
                and env_precision_error
                <= certificate["env_error_bound"] + 1e-10),
            difference_error=difference_error,
            difference_error_covered=bool(
                difference_error
                <= certificate["perturbation_bound"] + 1e-10),
            population_gap_lower=float(certificate["population_gap_lower"]),
            raw_sin_bound=_finite_or_none(certificate["raw_sin_bound"]),
            angle_bound_deg=angle_bound,
            sample_population_angle_deg=sample_population_angle,
            sample_population_covered=bool(
                sample_population_angle <= angle_bound + 1e-10),
            population_latent_model_bias_deg=population_latent_bias,
            sample_latent_angle_deg=sample_latent_angle,
            oracle_total_bound_deg=oracle_total_bound,
            sample_latent_oracle_covered=bool(
                sample_latent_angle <= oracle_total_bound + 1e-10),
            obs_rho=float(certificate["obs_rho"]),
            env_rho=float(certificate["env_rho"]),
        ))
    return rows


def _mean(flags):
    return float(np.mean(np.asarray(flags, dtype=float)))


def _median_or_none(values):
    values = list(values)
    return float(np.median(values)) if values else None


def summarize_n(rows):
    gate_rows = [row for row in rows if row["gate_pass"]]
    nonvacuous = [row for row in gate_rows if row["nonvacuous"]]
    seeds = sorted(set(row["seed"] for row in rows))
    family_coverage = []
    for seed in seeds:
        seed_rows = [row for row in rows if row["seed"] == seed]
        family_coverage.append(all(
            row["sample_population_covered"] for row in seed_rows))

    return dict(
        n_per_env=int(rows[0]["n_per_env"]),
        n_seeds=len(seeds),
        n_directions=len(rows),
        gate_pass_rate=_mean([row["gate_pass"] for row in rows]),
        gap_separated_among_gate=(
            _mean([row["gap_separated"] for row in gate_rows])
            if gate_rows else None),
        nonvacuous_among_gate=(
            _mean([row["nonvacuous"] for row in gate_rows])
            if gate_rows else None),
        median_nonvacuous_angle_bound_deg=_median_or_none(
            row["angle_bound_deg"] for row in nonvacuous),
        nonvacuous_sample_population_coverage=(
            _mean([row["sample_population_covered"] for row in nonvacuous])
            if nonvacuous else None),
        precision_event_coverage=_mean([
            row["precision_events_covered"] for row in rows]),
        difference_error_coverage=_mean([
            row["difference_error_covered"] for row in rows]),
        simultaneous_family_coverage=_mean(family_coverage),
        median_sample_population_angle_deg=float(np.median([
            row["sample_population_angle_deg"] for row in rows])),
        median_population_latent_model_bias_deg=float(np.median([
            row["population_latent_model_bias_deg"] for row in rows])),
        median_sample_latent_angle_deg=float(np.median([
            row["sample_latent_angle_deg"] for row in rows])),
    )


def usefulness_decision(summary):
    angle = summary["median_nonvacuous_angle_bound_deg"]
    coverage = summary["nonvacuous_sample_population_coverage"]
    checks = dict(
        gate_pass_rate=bool(
            summary["gate_pass_rate"]
            >= USEFULNESS_CRITERIA["gate_pass_rate_min"]),
        nonvacuous_among_gate=bool(
            summary["nonvacuous_among_gate"] is not None
            and summary["nonvacuous_among_gate"]
            >= USEFULNESS_CRITERIA["nonvacuous_among_gate_min"]),
        nonvacuous_coverage=bool(
            coverage is not None
            and coverage >= USEFULNESS_CRITERIA["nonvacuous_coverage_min"]),
        median_angle_bound=bool(
            angle is not None
            and angle <= USEFULNESS_CRITERIA["median_angle_bound_deg_max"]),
    )
    return checks, bool(all(checks.values()))


def print_table(summaries):
    print(
        "N      GATE   GAP|GATE  CERT|GATE  MED-BOUND  "
        "COVER|CERT  SAMP-POP  MODEL-BIAS",
        flush=True,
    )
    for row in summaries:
        bound = row["median_nonvacuous_angle_bound_deg"]
        coverage = row["nonvacuous_sample_population_coverage"]
        print(
            f"{row['n_per_env']:5d}  {row['gate_pass_rate']:6.3f}  "
            f"{row['gap_separated_among_gate']:8.3f}  "
            f"{row['nonvacuous_among_gate']:9.3f}  "
            f"{bound if bound is not None else float('nan'):9.3f}  "
            f"{coverage if coverage is not None else float('nan'):10.3f}  "
            f"{row['median_sample_population_angle_deg']:8.3f}  "
            f"{row['median_population_latent_model_bias_deg']:10.3f}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true",
        help="run a non-decisional reduced-power wiring check")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = SMOKE_SEEDS if args.smoke else FULL_SEEDS
    sample_sizes = SMOKE_SAMPLE_SIZES if args.smoke else FULL_SAMPLE_SIZES
    B_boot = SMOKE_BOOT if args.smoke else FULL_BOOT
    started = time.time()
    all_rows = []

    for n_per_env in sample_sizes:
        for seed in seeds:
            all_rows.extend(evaluate_family(seed, n_per_env, B_boot))
            print(
                f"[n={n_per_env}] seed {seed} done "
                f"({time.time() - started:.1f}s)",
                flush=True,
            )

    summaries = [
        summarize_n([
            row for row in all_rows if row["n_per_env"] == n_per_env
        ])
        for n_per_env in sample_sizes
    ]
    print_table(summaries)

    decision_eligible = not args.smoke
    reference = next(
        (row for row in summaries if row["n_per_env"] == REFERENCE_N), None)
    checks = None
    useful = False
    if reference is not None:
        checks, useful = usefulness_decision(reference)
    decision = (
        "SMOKE_ONLY" if args.smoke
        else ("USEFUL_AT_REFERENCE_N" if useful
              else "VACUOUS_AT_REFERENCE_N")
    )
    first_n_80pct = next((
        row["n_per_env"] for row in summaries
        if row["nonvacuous_among_gate"] is not None
        and row["nonvacuous_among_gate"] >= 0.80
    ), None)

    script_path = Path(__file__).resolve()
    certificate_path = REPO / "src" / "recover" / "certificate.py"
    report = dict(
        experiment="e2e_direction_certificate",
        mode="smoke" if args.smoke else "full",
        decision=decision,
        decision_eligible=decision_eligible,
        reference_n=REFERENCE_N,
        usefulness_criteria=USEFULNESS_CRITERIA,
        reference_checks=checks,
        first_n_with_80pct_nonvacuous_among_gate=first_n_80pct,
        design=dict(
            seeds=seeds,
            sample_sizes=sample_sizes,
            gate_bootstraps=B_boot,
            d_latent=D_LATENT,
            D_observed=D_OBS,
            d_projected=D_PROJ,
            edge_probability=EDGE_PROB,
            observation_noise_fraction=OBS_NOISE_FRAC,
            intervention_variance=IV_SCALE,
            gate_alpha=ALPHA,
            familywise_delta=FAMILYWISE_DELTA,
            simultaneous_matrices=SIMULTANEOUS_MATRICES,
            gate_null_cached_by_seed_and_sample_size=True,
        ),
        theorem_target=(
            "population leading eigenvector of the precision difference"),
        causal_alignment_caveat=(
            "population-to-latent angle is model bias and is not covered by "
            "the Davis-Kahan sampling bound"),
        summaries=summaries,
        rows=all_rows,
        provenance=dict(
            git_head_before_result=_git_head(),
            script_sha256=_sha256(script_path),
            certificate_sha256=_sha256(certificate_path),
            elapsed_seconds=float(time.time() - started),
        ),
    )

    if args.smoke:
        print(f"DECISION {decision}", flush=True)
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / "certificate_report.json"
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"DECISION {decision}", flush=True)
    print(f"WROTE {output.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
