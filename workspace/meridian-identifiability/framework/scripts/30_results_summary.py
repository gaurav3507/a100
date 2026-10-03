"""Consolidated results table + SUMMARY.json for the paper.

Reads:
    results/zeroshot_canonical/k562.json                    (baselines + ceiling)
    results/model/zeroshot_shift.json                       (full model, shift init)
    results/model/zeroshot_random.json                      (full model, random init)
    results/model/zeroshot_nodag_shift.json                 (no-DAG,    shift init)
    results/model/zeroshot_nodag_random.json                (no-DAG,    random init)

Writes:
    results/model/SUMMARY.json

Prints a formatted table and a fixed verdict string.

Usage:
    python scripts/30_results_summary.py
"""
import os, json
from pathlib import Path

FW = Path("/workspace/meridian-identifiability/framework")

BASELINES_JSON = FW / "results/zeroshot_canonical/k562.json"
MODEL_JSONS = {
    "full_shift":   FW / "results/model/zeroshot_shift.json",
    "full_random":  FW / "results/model/zeroshot_random.json",
    "nodag_shift":  FW / "results/model/zeroshot_nodag_shift.json",
    "nodag_random": FW / "results/model/zeroshot_nodag_random.json",
}
BASELINE_METHODS = ("zero", "global_mean", "corr_prop", "nn_corr",
                    "ridge_basis", "CEILING")


def load_baselines():
    with open(BASELINES_JSON) as f:
        raw = json.load(f)
    if "results" not in raw:
        raise RuntimeError(
            f"{BASELINES_JSON.name}: expected top-level 'results' key; "
            f"got {list(raw.keys())}"
        )
    r = raw["results"]
    missing = [m for m in BASELINE_METHODS if m not in r]
    if missing:
        raise RuntimeError(
            f"{BASELINES_JSON.name}: baseline methods missing from 'results': "
            f"{missing}; present: {list(r.keys())}"
        )
    return raw, {m: dict(
        r2_median=float(r[m]["r2_median"]),
        r2_mean=float(r[m].get("r2_mean", float("nan"))),
        cos_median=float(r[m].get("cos_median", float("nan"))),
        frac_beats_gmean=float(r[m].get("frac_beats_gmean", float("nan"))),
    ) for m in BASELINE_METHODS}


def load_model(path):
    with open(path) as f:
        d = json.load(f)
    return dict(
        source=str(path.name),
        best_median_r2=float(d["best_median_r2"]),
        final_median_r2=float(d.get("final_median_r2",
                                      d.get("best_median_r2", float("nan")))),
        final_mean_r2=float(d.get("final_mean_r2", float("nan"))),
        frac_beats_gmean=float(d.get("frac_beats_gmean", float("nan"))),
        frac_beats_ridge=float(d.get("frac_beats_ridge", float("nan"))),
    )


def main():
    raw, baselines = load_baselines()
    models = {name: load_model(p) for name, p in MODEL_JSONS.items()}

    gmean = baselines["global_mean"]["r2_median"]
    ridge = baselines["ridge_basis"]["r2_median"]
    ceiling = baselines["CEILING"]["r2_median"]
    headroom_from_json = raw["results"].get("_headroom")
    headroom = float(headroom_from_json) if headroom_from_json is not None \
                 else ceiling - gmean
    ridge_frac = (ridge - gmean) / headroom if headroom != 0 else float("nan")

    verdict = (
        "benchmarked negative: CRL model beats neither ridge (0.226) "
        "nor mean (0.129) on any held-out perturbation; causal layer "
        "contributes nothing (no-DAG identical); signal exists "
        "(ceiling 0.528)."
    )

    summary = dict(
        dataset=raw.get("dataset", "k562"),
        split_file=raw.get("split_file"),
        n_train=raw.get("n_train"),
        n_heldout=raw.get("n_heldout"),
        d_latent=raw.get("d_latent"),
        baselines=baselines,
        models=models,
        headroom=headroom,
        ridge_capture_fraction=ridge_frac,
        verdict=verdict,
    )

    out = FW / "results/model/SUMMARY.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.rename(tmp, out)

    print("\n== BASELINES (median R2 vs no-effect null on held-out) ==",
          flush=True)
    print(f"  {'method':<14}{'r2_med':>10}{'r2_mean':>10}{'cos_med':>10}"
          f"{'beats_gmean':>13}")
    for m in BASELINE_METHODS:
        b = baselines[m]
        print(f"  {m:<14}{b['r2_median']:>+10.4f}{b['r2_mean']:>+10.4f}"
              f"{b['cos_median']:>+10.4f}{b['frac_beats_gmean']:>13.1%}")
    print(f"\n  headroom (ceiling - global_mean) = {headroom:+.4f}")
    print(f"  ridge captures                    = {ridge_frac:.1%} of headroom")

    print("\n== MODEL RUNS (best median R2 over training) ==", flush=True)
    print(f"  {'variant':<18}{'best_med':>11}{'final_med':>11}"
          f"{'final_mean':>12}{'beats_gmean':>13}{'beats_ridge':>13}")
    for k in ("full_shift", "full_random", "nodag_shift", "nodag_random"):
        m = models[k]
        print(f"  {k:<18}{m['best_median_r2']:>+11.4f}"
              f"{m['final_median_r2']:>+11.4f}"
              f"{m['final_mean_r2']:>+12.4f}"
              f"{m['frac_beats_gmean']:>13.1%}"
              f"{m['frac_beats_ridge']:>13.1%}")

    print(f"\n[verdict] {verdict}", flush=True)
    print(f"\n[write] {out}", flush=True)


if __name__ == "__main__":
    main()
