"""E2c: non-tautological environment-starvation calibration.

The original E2 arm A writes zero rows for latents whose intervention environment
is absent. Its MCC therefore drops mechanically as environments are removed. This
experiment leaves that historical result untouched and evaluates an ungated
estimator that always returns d nonzero, linearly independent directions.

Estimator (spectral covariance-difference completion):

1. Fit the committed control-only PCA and recover one precision-difference row
   for every supplied intervention with ``backbone._unmixing_row``.
2. Form the pooled interventional-minus-control covariance difference.
3. Restrict that symmetric matrix to the orthogonal complement of the recovered
   row span and use its leading absolute-eigenvalue directions to complete the
   missing rows.

This is a full-rank spectral completion, not a zero-row penalty. It uses no true
latent values, and Hungarian MCC makes its evaluation permutation and sign safe.
For each starvation level, the canonical first-m subset is reported alongside a
fixed-seed, equal-weight subset control. Since d=5, drawing up to ten unique
subsets exhausts every possible subset at every level.

PRE-REGISTERED EXPECTATION:

Hungarian MCC should degrade non-mechanically as m falls, rather than in forced
1/d steps, and the gate should cap recovery no later than the first m at which
mean subset-control MCC falls below 0.90. A contradiction is reported rather than
tuned away.

E0.subspace_r2 is retained as a diagnostic, but it is NOT a starvation endpoint:
with D_PROJ=d and a full-rank d-by-d completion, every estimate is an invertible
transformation of the same projected observations. Consequently full-dimensional
linear subspace R2 is mathematically invariant to the completion. The experiment
checks and reports this invariance instead of claiming an impossible decline.

The default full run uses 10 seeds and B=500. ``--smoke`` uses three seeds and
B=100, writes a separate reduced-power report, and cannot establish the paper
result. Existing experiments and results are never modified.
"""

import argparse
import hashlib
import importlib.util
import itertools
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
RESULTS = REPO / "results" / "e2c"

FULL_SEEDS = list(range(10))
SMOKE_SEEDS = list(range(3))
D_LATENT = 5
D_OBS = 200
D_PROJ = 5
EDGE_PROB = 0.4
ALPHA = 0.05
FULL_BOOT = 500
SMOKE_BOOT = 100
IV_BASE = 0.1
N_BASE = 2000
MCC_THRESHOLD = 0.90
STARVATION_LEVELS = [5, 4, 3, 2, 1]
MAX_SUBSETS = 10


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SIM = _load(REPO / "sim" / "simulator.py", "grd_sim_simulator_e2c")
PR = _load(
    REPO / "src" / "gate" / "precision_readout.py",
    "grd_gate_precision_readout_e2c",
)
BK = _load(REPO / "src" / "recover" / "backbone.py", "grd_backbone_e2c")
E0 = _load(REPO / "experiments" / "e0_oracle.py", "grd_e0_oracle_e2c")


def gate_certificate(m_int, precision_count):
    """The unchanged E2 P1/P4 certificate for an m-environment subset."""
    n_recoverable = int(min(m_int, precision_count))
    if n_recoverable == 0:
        verdict = "ABSTAIN"
    elif n_recoverable < D_LATENT:
        verdict = "PROCEED_CAPPED"
    else:
        verdict = "PROCEED"
    return n_recoverable, verdict


def build_dataset(seed):
    specs = [
        SIM.EnvSpec("basis", None, ()),
        SIM.EnvSpec("obs", None, ()),
    ] + [
        SIM.EnvSpec(f"iv{i}", "hard", (i,), IV_BASE)
        for i in range(D_LATENT)
    ]
    return SIM.simulate(
        D_LATENT,
        D_OBS,
        N_BASE,
        specs,
        seed,
        mixing="linear",
        edge_prob=EDGE_PROB,
    )


def subset_control(seed, m):
    """Fixed-seed unique subsets, exhaustive here because C(5,m) <= 10."""
    subsets = list(itertools.combinations(range(D_LATENT), m))
    rng = np.random.default_rng(620_000 + 100 * seed + m)
    order = rng.permutation(len(subsets))[:min(MAX_SUBSETS, len(subsets))]
    return [subsets[int(index)] for index in order]


def _covariance_difference(Y_interventions, Y_obs):
    pooled = np.vstack(Y_interventions)
    return np.cov(pooled, rowvar=False) - np.cov(Y_obs, rowvar=False)


def complete_unmixing(Y_obs, Y_interventions, recovered_rows, subset):
    """Return a full-rank d-by-d unmixing via spectral complement completion."""
    subset = tuple(int(i) for i in subset)
    missing = tuple(i for i in range(D_LATENT) if i not in subset)
    W = np.zeros((D_LATENT, D_LATENT))
    for node in subset:
        W[node] = recovered_rows[node]

    recovered = np.vstack([recovered_rows[node] for node in subset])
    _, singular_values, right_vectors = np.linalg.svd(
        recovered, full_matrices=True)
    tolerance = (
        max(recovered.shape) * np.finfo(float).eps * singular_values[0]
    )
    recovered_rank = int(np.sum(singular_values > tolerance))
    if recovered_rank != len(subset):
        raise np.linalg.LinAlgError(
            f"recovered row rank {recovered_rank} != supplied count "
            f"{len(subset)} for subset {subset}"
        )

    completion_eigenvalues = []
    if missing:
        complement = right_vectors[len(subset):].T
        delta = _covariance_difference(
            [Y_interventions[node] for node in subset], Y_obs)
        restricted = complement.T @ delta @ complement
        eigenvalues, eigenvectors = np.linalg.eigh(restricted)
        order = np.argsort(np.abs(eigenvalues))[::-1]
        fallback = (complement @ eigenvectors[:, order]).T
        completion_eigenvalues = [
            float(eigenvalues[index]) for index in order
        ]
        for node, row in zip(missing, fallback):
            W[node] = row

    full_rank = int(np.linalg.matrix_rank(W))
    if full_rank != D_LATENT:
        raise np.linalg.LinAlgError(
            f"completion returned rank {full_rank}, expected {D_LATENT}"
        )
    if np.any(np.linalg.norm(W, axis=1) <= 0.0):
        raise RuntimeError("completion returned a zero row")

    return W, dict(
        supplied_nodes=list(subset),
        completed_nodes=list(missing),
        recovered_row_rank=recovered_rank,
        full_rank=full_rank,
        completion_eigenvalues=completion_eigenvalues,
    )


def evaluate_subset(
        subset, Y_obs, Z_true, Y_interventions, recovered_rows, detect):
    W, completion = complete_unmixing(
        Y_obs, Y_interventions, recovered_rows, subset)
    Z_hat = Y_obs @ W.T
    gate_count = int(sum(bool(detect[node]) for node in subset))
    n_recoverable, verdict = gate_certificate(len(subset), gate_count)
    return dict(
        subset=list(subset),
        gate_count=gate_count,
        gate_n_recoverable=n_recoverable,
        gate_verdict=verdict,
        mcc=float(E0.mcc(Z_hat, Z_true)),
        subspace_r2=float(E0.subspace_r2(Z_hat, Z_true)),
        estimator=completion,
    )


def evaluate_seed(seed, B, readout="precision", recovery=None):
    """readout selects the gate statistic and recovery the supplied-row rule
    (recovery=None uses readout). Per-environment rules (precision, covariance)
    compute each row once; the joint rule "jad" (Backbone C) is refitted on every
    supplied subset, since its rows depend on which environments are supplied. The
    spectral completion of missing rows is the same for every rule.
    """
    recovery = readout if recovery is None else recovery
    ds = build_dataset(seed)
    basis = ds.environments["basis"].X
    obs = ds.environments["obs"]
    mu, projection = BK.fit_pca(basis, D_PROJ)
    Y_obs = BK.project(obs.X, mu, projection)
    Y_interventions = {
        node: BK.project(ds.environments[f"iv{node}"].X, mu, projection)
        for node in range(D_LATENT)
    }
    gate = PR.count_recoverable(
        [Y_interventions[node] for node in range(D_LATENT)],
        Y_obs,
        alpha=ALPHA,
        B=B,
        rng=np.random.default_rng(910_000 + seed),
        readout=readout,
    )
    if recovery == "jad":
        def rows_for(subset):
            W = BK.unmixing_rows(Y_obs, [Y_interventions[n] for n in subset],
                                 list(subset), D_LATENT, readout="jad")
            return {int(n): W[n] for n in subset}
    else:
        recovered_rows = {
            node: BK._unmixing_row(Y_obs, Y_interventions[node], readout=recovery)
            for node in range(D_LATENT)
        }

        def rows_for(subset):
            return recovered_rows

    levels = []
    for m in STARVATION_LEVELS:
        canonical_subset = tuple(range(m))
        canonical = evaluate_subset(
            canonical_subset,
            Y_obs,
            obs.Z,
            Y_interventions,
            rows_for(canonical_subset),
            gate["detect"],
        )
        control_subsets = subset_control(seed, m)
        controls = [
            evaluate_subset(
                subset,
                Y_obs,
                obs.Z,
                Y_interventions,
                rows_for(subset),
                gate["detect"],
            )
            for subset in control_subsets
        ]
        levels.append(dict(
            m=m,
            canonical=canonical,
            subset_control=controls,
        ))

    return dict(
        seed=seed,
        gate_all_environments=gate,
        levels=levels,
    )


def _mean(values):
    return float(np.mean(np.asarray(values, dtype=float)))


def _sd(values):
    return float(np.std(np.asarray(values, dtype=float)))


def aggregate(seed_rows):
    summaries = []
    for m in STARVATION_LEVELS:
        level_rows = [
            level
            for seed_row in seed_rows
            for level in seed_row["levels"]
            if level["m"] == m
        ]
        canonical = [level["canonical"] for level in level_rows]
        per_seed_control = []
        for level in level_rows:
            controls = level["subset_control"]
            per_seed_control.append(dict(
                mcc=_mean([row["mcc"] for row in controls]),
                subspace_r2=_mean([row["subspace_r2"] for row in controls]),
                gate_n_recoverable=_mean([
                    row["gate_n_recoverable"] for row in controls
                ]),
                gate_verdicts=[row["gate_verdict"] for row in controls],
            ))
        control_mcc = [row["mcc"] for row in per_seed_control]
        control_r2 = [row["subspace_r2"] for row in per_seed_control]
        control_gate = [
            row["gate_n_recoverable"] for row in per_seed_control
        ]
        verdicts = [
            verdict
            for row in per_seed_control
            for verdict in row["gate_verdicts"]
        ]
        summaries.append(dict(
            m=m,
            canonical=dict(
                mcc_mean=_mean([row["mcc"] for row in canonical]),
                mcc_sd=_sd([row["mcc"] for row in canonical]),
                subspace_r2_mean=_mean([
                    row["subspace_r2"] for row in canonical
                ]),
                subspace_r2_sd=_sd([
                    row["subspace_r2"] for row in canonical
                ]),
                gate_n_recoverable_mean=_mean([
                    row["gate_n_recoverable"] for row in canonical
                ]),
            ),
            random_subset_control=dict(
                subsets_per_seed=len(level_rows[0]["subset_control"]),
                mcc_mean=_mean(control_mcc),
                mcc_seed_sd=_sd(control_mcc),
                subspace_r2_mean=_mean(control_r2),
                subspace_r2_seed_sd=_sd(control_r2),
                gate_n_recoverable_mean=_mean(control_gate),
                gate_n_recoverable_seed_sd=_sd(control_gate),
                verdict_counts={
                    verdict: verdicts.count(verdict)
                    for verdict in ("PROCEED", "PROCEED_CAPPED", "ABSTAIN")
                },
            ),
        ))

    below = next(
        (row["m"] for row in summaries
         if row["random_subset_control"]["mcc_mean"] < MCC_THRESHOLD),
        None,
    )
    gate_restriction = next(
        (row["m"] for row in summaries
         if row["random_subset_control"]["verdict_counts"]["PROCEED"]
         < (len(seed_rows) * row["random_subset_control"]["subsets_per_seed"])),
        None,
    )
    gate_at_or_before = (
        None if below is None else bool(
            gate_restriction is not None and gate_restriction >= below)
    )

    all_estimators = [
        result
        for seed_row in seed_rows
        for level in seed_row["levels"]
        for result in [level["canonical"], *level["subset_control"]]
    ]
    full_rank_all = all(
        row["estimator"]["full_rank"] == D_LATENT for row in all_estimators)
    zero_row_count = sum(
        row["estimator"]["full_rank"] < D_LATENT for row in all_estimators)
    all_r2 = [row["subspace_r2"] for row in all_estimators]
    within_seed_r2_ranges = []
    for seed_row in seed_rows:
        seed_r2 = [
            result["subspace_r2"]
            for level in seed_row["levels"]
            for result in [level["canonical"], *level["subset_control"]]
        ]
        within_seed_r2_ranges.append(float(max(seed_r2) - min(seed_r2)))

    return dict(
        levels=summaries,
        crossover_random_subset_mcc_below_0p90=below,
        crossover_gate_first_restricts=gate_restriction,
        gate_restricts_at_or_before_mcc_crossover=gate_at_or_before,
        estimator_checks=dict(
            full_rank_all=bool(full_rank_all),
            zero_or_rank_deficient_output_count=int(zero_row_count),
        ),
        subspace_r2_invariance=dict(
            max_within_seed_range=float(max(within_seed_r2_ranges)),
            across_seed_and_completion_range=float(max(all_r2) - min(all_r2)),
            expected_invariant=True,
            explanation=(
                "A full-rank 5x5 transform preserves the column span of the "
                "5D projected observations, so full-dimensional linear "
                "subspace R2 cannot diagnose axis recovery."
            ),
        ),
    )


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the non-decisional three-seed/B=100 check",
    )
    return parser.parse_args()


def print_table(report):
    print("\nE2C NON-TAUTOLOGICAL STARVATION", flush=True)
    print(
        "m  gate-n  MCC-mean  MCC-seed-SD  subspace-R2  verdicts(P/C/A)",
        flush=True,
    )
    for row in report["aggregate"]["levels"]:
        control = row["random_subset_control"]
        counts = control["verdict_counts"]
        verdict_text = (
            f"{counts['PROCEED']}/{counts['PROCEED_CAPPED']}/"
            f"{counts['ABSTAIN']}"
        )
        print(
            f"{row['m']:d}  {control['gate_n_recoverable_mean']:6.2f}  "
            f"{control['mcc_mean']:8.4f}  "
            f"{control['mcc_seed_sd']:11.4f}  "
            f"{control['subspace_r2_mean']:11.6f}  {verdict_text}",
            flush=True,
        )
    aggregate_result = report["aggregate"]
    print(
        "crossover: gate first restricts at m="
        f"{aggregate_result['crossover_gate_first_restricts']}; "
        "MCC first below 0.90 at m="
        f"{aggregate_result['crossover_random_subset_mcc_below_0p90']}",
        flush=True,
    )
    print(
        "full-rank outputs: "
        f"{aggregate_result['estimator_checks']['full_rank_all']}; "
        "max within-seed subspace-R2 range: "
        f"{aggregate_result['subspace_r2_invariance']['max_within_seed_range']:.3e}",
        flush=True,
    )
    print(f"status: {report['status']}", flush=True)


def main():
    args = parse_args()
    seeds = SMOKE_SEEDS if args.smoke else FULL_SEEDS
    B = SMOKE_BOOT if args.smoke else FULL_BOOT
    started = time.time()
    seed_rows = []
    for seed in seeds:
        seed_rows.append(evaluate_seed(seed, B))
        print(f"[seed {seed}] done ({time.time() - started:.1f}s)", flush=True)

    aggregate_result = aggregate(seed_rows)
    wiring_pass = bool(
        aggregate_result["estimator_checks"]["full_rank_all"]
        and aggregate_result["estimator_checks"]
        ["zero_or_rank_deficient_output_count"] == 0
        and np.isfinite([
            row["random_subset_control"]["mcc_mean"]
            for row in aggregate_result["levels"]
        ]).all()
    )
    status = "PASS" if wiring_pass else "FAIL"
    script_path = Path(__file__).resolve()
    report = dict(
        experiment="e2c_non_tautological_starvation",
        mode="smoke" if args.smoke else "full",
        result_eligible=not args.smoke,
        status=status,
        scientific_conclusion=(
            "Reduced-power wiring check only; no paper conclusion."
            if args.smoke else
            "See measured MCC crossover; contradictions to the pre-registered "
            "expectation are retained."
        ),
        config=dict(
            seeds=seeds,
            B=B,
            alpha=ALPHA,
            d_latent=D_LATENT,
            D_observed=D_OBS,
            d_projected=D_PROJ,
            n_per_environment=N_BASE,
            iv_scale=IV_BASE,
            edge_prob=EDGE_PROB,
            starvation_levels=STARVATION_LEVELS,
            max_random_subsets=MAX_SUBSETS,
            mcc_threshold=MCC_THRESHOLD,
        ),
        estimator=(
            "Backbone precision rows for supplied targets plus absolute-eigenvalue "
            "spectral completion of the pooled interventional-minus-control "
            "covariance in their orthogonal complement."
        ),
        primary_metric=(
            "Hungarian MCC; permutation and sign safe, axis-sensitive."
        ),
        aggregate=aggregate_result,
        per_seed=seed_rows,
        provenance=dict(
            git_head=_git_head(),
            script_sha256=_sha256(script_path),
            numpy_version=np.__version__,
        ),
        wall_seconds=round(time.time() - started, 1),
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    filename = "starvation_smoke.json" if args.smoke else "starvation_report.json"
    output = RESULTS / filename
    output.write_text(json.dumps(report, indent=2) + "\n")
    print_table(report)
    print(f"written {output} ({report['wall_seconds']}s)", flush=True)
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
