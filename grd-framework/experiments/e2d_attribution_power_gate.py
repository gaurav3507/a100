"""E2d: synthetic power gate for projected subspace attribution.

This experiment decides whether a lower-dimensional random-projection version
of the E3 subspace test has earned a real-data follow-up. It does not read any
real dataset and it does not tune projections or thresholds from outcomes.

PRE-REGISTERED DESIGN (fixed before the full run):

* latent dimension d=10, observed dimension D=651, projected dimension q=40;
* K=5 fixed Gaussian orthogonal projections with seeds 7300,...,7304;
* 20 independent simulator seeds and B=500 null draws per projection;
* RPE1-like (n_control=2000, n_env=250) and small-n
  (n_control=100, n_env=94) regimes;
* exact E3 principal-angle statistic and exact E3 null branch rule;
* projection p-values combined as min(1, K * min_k p_k), which controls the
  within-environment search over the five fixed projections;
* no upstream detectability filter: this calibrates the attributor itself.

The rescue is a GO only if BOTH regimes satisfy all four criteria:

1. independent-null false-positive rate <= 0.10;
2. hard-mechanism false-attribution rate <= 0.10;
3. mixed feature-gain attribution power >= 0.80;
4. range of the five projection-specific mixed-gain rejection rates <= 0.25.

Uniform signal shrinkage by 0.7 is reported descriptively because finite-sample
SNR changes can rotate its estimated subspace even though scalar gain preserves
the population signal subspace. A failed scientific criterion is a NO_GO result,
not a program error and not a reason to change the specification.

Use --smoke for a three-seed/B=100 wiring check. Smoke output is kept separate
and cannot authorize real-data analysis. The full run writes
results/e2d_attribution/projected_attribution_report.json.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from e3_attribution_panel import (  # noqa: E402
    fast_subspace_angle,
    fast_subspace_null_values,
)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SIM = _load(REPO / "sim" / "simulator.py", "grd_sim_simulator_e2d")

D_LATENT = 10
D_OBS = 651
D_PROJECTED = 40
N_PROJECTIONS = 5
PROJECTION_SEEDS = list(range(7300, 7300 + N_PROJECTIONS))
FULL_SEEDS = list(range(20))
SMOKE_SEEDS = list(range(3))
FULL_BOOT = 500
SMOKE_BOOT = 100
ALPHA = 0.05
EDGE_PROB = 0.4
OBS_NOISE_FRAC = 0.1
IV_SCALE = 0.1

REGIMES = [
    dict(name="rpe1_like", n_control=2000, n_env=250),
    dict(name="small_n", n_control=100, n_env=94),
]

CRITERIA = dict(
    null_false_positive_rate_max=0.10,
    mechanism_false_attribution_rate_max=0.10,
    mixed_gain_power_min=0.80,
    projection_mixed_rate_range_max=0.25,
)

CASE_NAMES = (
    "independent_null",
    "mechanism_hard_iv",
    "mixed_gain_u05_15",
    "uniform_gain_0.7",
)


def fixed_projections():
    """Return the five predeclared D-by-q orthonormal projection matrices."""
    projections = []
    for seed in PROJECTION_SEEDS:
        rng = np.random.default_rng(seed)
        gaussian = rng.standard_normal((D_OBS, D_PROJECTED))
        Q, _ = np.linalg.qr(gaussian, mode="reduced")
        if not np.allclose(Q.T @ Q, np.eye(D_PROJECTED), atol=1e-12):
            raise RuntimeError(f"projection {seed} is not orthonormal")
        projections.append(Q)
    return projections


def _fresh_environment(ds, n_env, rng, gain):
    """Independent observational latents through a fixed feature-space gain."""
    latent = SIM.sample_latent(ds.B, ds.noise_var, n_env, rng)
    signal = SIM.mix_linear(latent, ds.A) * gain
    return SIM.add_obs_noise(signal, ds.sd_obs, rng)


def build_cases(seed, regime):
    """One shared control plus four independent, equally sized environments."""
    specs = [
        SIM.EnvSpec("control", None, ()),
        SIM.EnvSpec("mechanism", "hard", (0,), IV_SCALE),
    ]
    ds = SIM.simulate(
        D_LATENT,
        D_OBS,
        regime["n_control"],
        specs,
        seed,
        mixing="linear",
        edge_prob=EDGE_PROB,
        obs_noise_frac=OBS_NOISE_FRAC,
    )
    n_env = regime["n_env"]
    rng_null = np.random.default_rng(100_000 + 10_000 * seed + n_env)
    rng_mixed = np.random.default_rng(200_000 + 10_000 * seed + n_env)
    rng_uniform = np.random.default_rng(300_000 + 10_000 * seed + n_env)
    mixed_gain = rng_mixed.uniform(0.5, 1.5, D_OBS)

    cases = {
        "independent_null": _fresh_environment(
            ds, n_env, rng_null, np.ones(D_OBS)),
        "mechanism_hard_iv": ds.environments["mechanism"].X[:n_env],
        "mixed_gain_u05_15": _fresh_environment(
            ds, n_env, rng_mixed, mixed_gain),
        "uniform_gain_0.7": _fresh_environment(
            ds, n_env, rng_uniform, np.full(D_OBS, 0.7)),
    }
    return ds.environments["control"].X, cases


def finite_pvalue(statistic, null_values):
    """Upper-tail Monte Carlo p-value with the finite-simulation correction."""
    return float(
        (1 + np.count_nonzero(null_values >= statistic)) /
        (len(null_values) + 1)
    )


def evaluate_seed(seed, regime, regime_index, projections, B):
    X_control, cases = build_cases(seed, regime)
    case_records = {
        name: dict(angles_deg=[], projection_pvalues=[])
        for name in CASE_NAMES
    }
    null_metadata = []

    for projection_index, (projection_seed, Q) in enumerate(
            zip(PROJECTION_SEEDS, projections)):
        projected_control = X_control @ Q
        null_rng = np.random.default_rng(
            400_000 + regime_index * 100_000 + seed * 100 + projection_index)
        null_values, disjoint, backend = fast_subspace_null_values(
            projected_control,
            D_LATENT,
            regime["n_env"],
            B,
            null_rng,
        )
        null_metadata.append(dict(
            projection_seed=projection_seed,
            disjoint=bool(disjoint),
            backend=backend,
            null_median_deg=float(np.degrees(np.median(null_values))),
            null_q95_deg=float(np.degrees(np.quantile(null_values, 0.95))),
        ))
        for case_name in CASE_NAMES:
            angle = fast_subspace_angle(
                cases[case_name] @ Q, projected_control, D_LATENT)
            case_records[case_name]["angles_deg"].append(
                float(np.degrees(angle)))
            case_records[case_name]["projection_pvalues"].append(
                finite_pvalue(angle, null_values))

    for record in case_records.values():
        pvalues = record["projection_pvalues"]
        combined = min(1.0, N_PROJECTIONS * min(pvalues))
        record["combined_pvalue"] = float(combined)
        record["combined_reject"] = bool(combined <= ALPHA)
        record["projection_rejects_nominal"] = [
            bool(pvalue <= ALPHA) for pvalue in pvalues
        ]
    return dict(
        seed=seed,
        regime=regime["name"],
        null_metadata=null_metadata,
        cases=case_records,
    )


def _rate(flags):
    return float(np.mean(np.asarray(flags, dtype=float)))


def summarize_regime(rows):
    rates = {
        name: _rate([
            row["cases"][name]["combined_reject"] for row in rows
        ])
        for name in CASE_NAMES
    }
    projection_mixed_rates = []
    for projection_index in range(N_PROJECTIONS):
        projection_mixed_rates.append(_rate([
            row["cases"]["mixed_gain_u05_15"]
            ["projection_rejects_nominal"][projection_index]
            for row in rows
        ]))
    projection_range = float(
        max(projection_mixed_rates) - min(projection_mixed_rates))
    checks = {
        "null_false_positive_rate": bool(
            rates["independent_null"] <=
            CRITERIA["null_false_positive_rate_max"]),
        "mechanism_false_attribution_rate": bool(
            rates["mechanism_hard_iv"] <=
            CRITERIA["mechanism_false_attribution_rate_max"]),
        "mixed_gain_power": bool(
            rates["mixed_gain_u05_15"] >=
            CRITERIA["mixed_gain_power_min"]),
        "projection_stability": bool(
            projection_range <=
            CRITERIA["projection_mixed_rate_range_max"]),
    }
    return dict(
        n_seeds=len(rows),
        combined_rejection_rates=rates,
        projection_mixed_rejection_rates_nominal=dict(
            zip([str(seed) for seed in PROJECTION_SEEDS],
                projection_mixed_rates)),
        projection_mixed_rate_range=projection_range,
        criterion_checks=checks,
        passes_all=bool(all(checks.values())),
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
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true",
        help="run the non-decisional three-seed/B=100 wiring check")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = SMOKE_SEEDS if args.smoke else FULL_SEEDS
    B = SMOKE_BOOT if args.smoke else FULL_BOOT
    projections = fixed_projections()
    started = time.time()
    rows = []

    for regime_index, regime in enumerate(REGIMES):
        for seed in seeds:
            row = evaluate_seed(
                seed, regime, regime_index, projections, B)
            rows.append(row)
            print(
                f"[{regime['name']}] seed {seed} done "
                f"({time.time() - started:.1f}s)",
                flush=True,
            )

    summaries = {}
    for regime in REGIMES:
        regime_rows = [
            row for row in rows if row["regime"] == regime["name"]
        ]
        summaries[regime["name"]] = summarize_regime(regime_rows)

    all_pass = bool(all(summary["passes_all"]
                        for summary in summaries.values()))
    decision = "SMOKE_ONLY" if args.smoke else ("GO" if all_pass else "NO_GO")
    script_path = Path(__file__).resolve()
    report = dict(
        experiment="e2d_projected_attribution_power_gate",
        mode="smoke" if args.smoke else "full",
        decision=decision,
        decision_eligible=not args.smoke,
        decision_rule=(
            "GO only when every criterion passes in both predeclared regimes; "
            "otherwise NO_GO. Smoke runs cannot authorize real-data analysis."
        ),
        scientific_scope=(
            "Synthetic calibration of the projected subspace attributor only; "
            "no real data and no upstream detectability filter."
        ),
        alpha=ALPHA,
        B=B,
        seeds=seeds,
        d_latent=D_LATENT,
        d_observed=D_OBS,
        d_projected=D_PROJECTED,
        projection_seeds=PROJECTION_SEEDS,
        pvalue_combination="min(1, K * min_k(p_k))",
        regimes=REGIMES,
        edge_prob=EDGE_PROB,
        observation_noise_fraction=OBS_NOISE_FRAC,
        hard_intervention_variance=IV_SCALE,
        criteria=CRITERIA,
        summaries=summaries,
        rows=rows,
        provenance=dict(
            git_head=_git_head(),
            script_sha256=_sha256(script_path),
            numpy_version=np.__version__,
        ),
        wall_seconds=round(time.time() - started, 1),
    )

    output_dir = REPO / "results" / "e2d_attribution"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        "projected_attribution_smoke.json" if args.smoke
        else "projected_attribution_report.json"
    )
    output_path = output_dir / filename
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    print("\nREGIME       NULL  MECH  MIXED  UNIFORM  PROJ-RANGE  PASS", flush=True)
    for regime in REGIMES:
        summary = summaries[regime["name"]]
        rates = summary["combined_rejection_rates"]
        print(
            f"{regime['name']:12s} "
            f"{rates['independent_null']:.3f} "
            f"{rates['mechanism_hard_iv']:.3f} "
            f"{rates['mixed_gain_u05_15']:.3f} "
            f"{rates['uniform_gain_0.7']:.3f} "
            f"{summary['projection_mixed_rate_range']:.3f} "
            f"{summary['passes_all']}",
            flush=True,
        )
    print(f"\nDECISION: {decision}", flush=True)
    print(f"written {output_path} ({report['wall_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
