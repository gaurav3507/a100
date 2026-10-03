"""Is the 248-552 participation-ratio band a property of the data, or of the
normalisation? Four preprocessing arms at matched n_control = 2000.

A100. NO TEST, NO VERDICT. This computes the same descriptive estimators under
four normalisations and reports the spread. It expresses no assumption verdict
and Phase B remains not authorised.

WHY ONLY ONE CAP. n_control = 2000 is the matched-sample-size comparison: every
dataset contributes the same n, so a difference between arms cannot be a
sample-size artefact. Sweeping both caps would double the runtime and answer a
question the descriptives already answered.

THE FOUR ARMS, applied to the CONTROL-CELL matrix only:
    raw          exactly what the loader returns, untouched. This arm MUST
                 reproduce the existing cap-2000 numbers; the step-0 gate
                 below enforces it.
    standardise  per-gene centre and scale to unit variance
    log1p_std    log1p, then per-gene standardise
    rank_int     per-gene rank-inverse-normal transform

"RAW" IS PROBABLY NOT RAW, AND THAT CHANGES WHAT log1p_std MEANS.
None of the three loaders (03_screen.load, 40_screen_norman.load_norman,
41_screen_frangieh.load_expression) documents any normalisation; each passes
through whatever the upstream file holds. Evidence from the descriptives
fingerprints says the upstream files are already transformed: every control
matrix has a NON-INTEGRAL sum, and a float64 sum of ~1e7 integer counts would
be exact (integers are exact in float64 to 2^53). Their scale agrees --
per-entry means 0.11 to 0.82, RMS 0.48 to 1.96 -- which is log1p-normalised
territory, not counts.

So this script does NOT assume. It PROBES each control matrix at runtime and
classifies it on the fraction of NON-ZERO entries that are integral -- not the
fraction of all entries, because single-cell matrices are ~85% exact zeros and
a zero is an integer, so the naive fraction is high even for log-normalised
data. Only the non-zero entries separate counts from a transform.
and log1p_std is emitted as null-with-reason wherever the input is already
log-like, because a second log is not a preprocessing choice anyone would
make, and reporting it as if it were would mislead. The probe and the
classification are written into the artefact for every dataset.

Estimators are IMPORTED from 85_dataset_descriptives.py, never reimplemented,
so an arm differs from the descriptives only by its preprocessing.

Usage (A100, cb venv):
    nohup python -u causalbench/scripts/91_preprocessing_sweep.py \\
        --dataset k562 > logs/sweep_k562.log 2>&1 &
    python causalbench/scripts/91_preprocessing_sweep.py --selftest
"""
import argparse
import datetime
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# scripts/ sits at the repo root in this repository, so the root is
# one level up, not two. (It was two when these lived in
# causalbench/scripts/ in the source repo.)
REPO = HERE.parent
DESCRIPTIVES_DIR = REPO / "results/descriptives"

CAP = 2000                       # matched sample size, one cap only
ARMS = ("raw", "standardise", "log1p_std", "rank_int")
GATE_TOL = 1e-10                 # raw arm must reproduce to this
NUM_RANK_TOL = 1e-10             # same tolerance the figure uses
DATASETS = ("k562", "rpe1", "norman",
            "frangieh_coculture", "frangieh_control", "frangieh_ifng")


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    real = os.makedirs
    os.makedirs = lambda *a, **k: None
    try:
        spec.loader.exec_module(mod)
    finally:
        os.makedirs = real
    return mod


RIO = _load_module(HERE / "84_results_io.py", "_results_io")
DESC = _load_module(HERE / "85_dataset_descriptives.py", "_descriptives")


# ------------------------------------------------------------------- probing
def probe_matrix(X, rng, n_sample=200_000):
    """What did the loader actually hand us? Measured, not assumed."""
    X = np.asarray(X)
    flat = X.ravel()
    if flat.size > n_sample:                 # sample; these matrices are large
        idx = rng.choice(flat.size, size=n_sample, replace=False)
        s = np.asarray(flat[idx], dtype=np.float64)
    else:
        s = np.asarray(flat, dtype=np.float64)
    nz = s[s != 0.0]
    return dict(
        n_sampled=int(s.size),
        min=float(s.min()), max=float(s.max()), mean=float(s.mean()),
        frac_integer=float(np.mean(s == np.round(s))),
        # THE discriminator. Single-cell matrices are ~85% exact zeros, and a
        # zero is an integer, so frac_integer is high even for log-normalised
        # data. Only the NON-ZERO entries separate counts from a transform.
        frac_integer_nonzero=(float(np.mean(nz == np.round(nz)))
                              if nz.size else None),
        n_nonzero_sampled=int(nz.size),
        frac_zero=float(np.mean(s == 0.0)),
        has_negative=bool(s.min() < 0),
    )


def classify(pr):
    """Counts or already-transformed. Decides whether log1p_std is coherent."""
    if pr["has_negative"]:
        return ("already_transformed_signed",
                "matrix contains negative values, so it is centred or otherwise "
                "transformed; log1p is undefined on it")
    fin = pr.get("frac_integer_nonzero")
    if fin is None:
        return ("degenerate", "matrix has no non-zero entries")
    if fin > 0.99:
        return ("counts",
                f"{fin:.4f} of NON-ZERO entries are integral (max={pr['max']:.4g}): "
                f"raw counts, so log1p is the first log applied")
    return ("already_transformed",
            f"only {fin:.4f} of NON-ZERO entries are integral "
            f"(max={pr['max']:.4g}, frac_zero={pr['frac_zero']:.3f}): already "
            f"normalised upstream, most likely log1p. A further log1p would be "
            f"a DOUBLE log")


def arm_coherent(arm, kind):
    if arm in ("raw", "standardise", "rank_int"):
        return True, None
    if kind == "counts":
        return True, None
    if kind == "already_transformed_signed":
        return False, "log1p undefined: matrix has negative values"
    return False, ("input is already log-like, so log1p_std would be a double "
                   "log; emitted as null rather than reported as a "
                   "preprocessing choice")


# ------------------------------------------------------------------ the arms
def apply_arm(X, arm):
    """Return the transformed CONTROL matrix. float64 throughout."""
    X = np.asarray(X, dtype=np.float64)
    if arm == "raw":
        return X
    if arm == "standardise":
        mu, sd = X.mean(0), X.std(0)
        return (X - mu) / np.where(sd > 0, sd, 1.0)
    if arm == "log1p_std":
        L = np.log1p(X)
        mu, sd = L.mean(0), L.std(0)
        return (L - mu) / np.where(sd > 0, sd, 1.0)
    if arm == "rank_int":
        from scipy.special import ndtri
        n = X.shape[0]
        out = np.empty_like(X)
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j], kind="mergesort")
            ranks = np.empty(n, dtype=np.float64)
            ranks[order] = np.arange(1, n + 1, dtype=np.float64)
            out[:, j] = ndtri((ranks - 0.5) / n)     # Blom-style, offset 0.5
        return out
    raise ValueError(f"unknown arm {arm!r}")


# -------------------------------------------------------------- control grab
def control_matrix(dataset, args):
    """The EXACT control matrix 85_dataset_descriptives feeds to its estimators.

    Rather than re-deriving it (which would drift), 85's own loader is run with
    latent_dim_block swapped for a capture. That yields both the matrix and the
    rng in precisely the state the estimators would have seen, so the raw arm
    reproduces the published subsample bit for bit. The step-0 gate is what
    proves it did.
    """
    grabbed = {}

    def capture(Xc, rng, spectra=None, key_prefix=""):
        grabbed["X"] = np.asarray(Xc)
        grabbed["rng"] = rng
        return {}

    real = DESC.latent_dim_block
    DESC.latent_dim_block = capture
    try:
        base = dataset.split("_")[0]
        ns = argparse.Namespace(dataset=base, hvg=args.hvg, seed=args.seed,
                                arm=(dataset.split("_", 1)[1]
                                     if dataset.startswith("frangieh_") else None),
                                abide_npz=None, n_spec_cap=None,
                                overlap_check=None, selftest=False)
        rng = np.random.default_rng(args.seed)
        DESC.LOADERS[base](rng, ns, {})
    finally:
        DESC.latent_dim_block = real
    if "X" not in grabbed:
        sys.exit(f"[fatal] could not capture the control matrix for {dataset}")
    return grabbed["X"], grabbed["rng"]


# ------------------------------------------------------------- step-0 gate
def expected_raw_pr(dataset):
    """Read the published cap-2000 participation ratio FROM THE ARTEFACT."""
    for p in sorted(DESCRIPTIVES_DIR.glob("*.json")):
        try:
            doc = json.load(open(p))
        except ValueError:
            continue
        if (doc.get("meta") or {}).get("status") != "CURRENT":
            continue
        arm = ((doc.get("meta") or {}).get("config") or {}).get(
            "extra", {}).get("arm")
        label = f"{doc.get('dataset')}_{arm}" if arm else str(doc.get("dataset"))
        if label != dataset:
            continue
        for b in doc.get("blocks") or []:
            c = (b or {}).get("latent_dimension_from_controls")
            if isinstance(c, dict) and str(CAP) in c:
                return float(c[str(CAP)]["participation_ratio"]), p.name
    return None, None


# ------------------------------------------------------------------- selftest
def run_selftest():
    """Delegates to 85's pinned gate so the constants cannot drift apart."""
    return DESC.run_selftest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--hvg", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(run_selftest())
    if not a.dataset:
        ap.error("give --dataset or --selftest")

    rng_probe = np.random.default_rng(a.seed)
    print(f"[start] preprocessing sweep  dataset={a.dataset}  cap={CAP}  "
          f"arms={ARMS}", flush=True)

    Xc, rng = control_matrix(a.dataset, a)
    print(f"[data]  control matrix {Xc.shape} dtype={Xc.dtype}", flush=True)

    pr_probe = probe_matrix(Xc, rng_probe)
    kind, why = classify(pr_probe)
    print(f"[probe] {kind}: {why}", flush=True)
    print(f"[probe] min={pr_probe['min']:.4g} max={pr_probe['max']:.4g} "
          f"mean={pr_probe['mean']:.4g} frac_integer={pr_probe['frac_integer']:.4f} "
          f"frac_zero={pr_probe['frac_zero']:.4f}", flush=True)

    # ---- STEP-0 GATE: the raw arm must reproduce the published number.
    expected, src = expected_raw_pr(a.dataset)
    if expected is None:
        sys.exit(f"[fatal] no CURRENT descriptives artefact for {a.dataset}; "
                 f"cannot verify the raw arm. Refusing to sweep.")

    results, spectra = {}, {}
    for arm in ARMS:
        ok, reason = arm_coherent(arm, kind)
        if not ok:
            results[arm] = dict(status="null", reason=reason)
            print(f"[{arm:<11}] SKIPPED: {reason}", flush=True)
            continue
        Xa = apply_arm(Xc, arm)
        # a FRESH rng seeded identically per arm, so every arm sees the same
        # subsample of control cells and the arms differ only by preprocessing
        res, lam = DESC.spectrum_estimators(
            Xa, CAP, np.random.default_rng(a.seed), return_lam=True)
        if lam is None:
            results[arm] = dict(status="null", reason="degenerate: zero variance")
            continue
        lam = np.asarray(lam, dtype=np.float64)
        key = f"{a.dataset}__{arm}__{CAP}"
        spectra[key] = lam
        res.update(
            spectra_key=key,
            lam_len=int(lam.size), lam_sum=float(lam.sum()),
            lam_sumsq=float((lam ** 2).sum()),
            numerical_rank=int((lam > NUM_RANK_TOL * lam.max()).sum()),
            status="ok")
        results[arm] = res
        print(f"[{arm:<11}] PR={res['participation_ratio']:.4f}  "
              f"eff_rank={res['effective_rank_exp_spectral_entropy']:.1f}  "
              f"numrank={res['numerical_rank']}", flush=True)

        if arm == "raw":
            got = float(res["participation_ratio"])
            if abs(got - expected) > GATE_TOL:
                sys.exit(
                    f"\n[fatal] STEP-0 GATE FAILED for {a.dataset}\n"
                    f"        raw-arm participation ratio does not reproduce "
                    f"the published value.\n"
                    f"        expected {expected!r}  (from {src})\n"
                    f"        got      {got!r}\n"
                    f"        |delta| = {abs(got-expected):.3e} > tol "
                    f"{GATE_TOL:g}\n"
                    f"        A sweep whose control arm does not reproduce is "
                    f"measuring the pipeline, not the preprocessing. Refusing "
                    f"to continue.")
            print(f"[gate]  raw arm reproduces {expected!r} from {src}  PASS",
                  flush=True)

    prs = [r["participation_ratio"] for r in results.values()
           if r.get("status") == "ok"]
    payload = dict(
        dataset=a.dataset, cap=CAP, arms=list(ARMS),
        loader_probe=pr_probe, input_kind=kind, input_kind_reason=why,
        arms_result=results,
        pr_spread=(max(prs) / min(prs) if prs and min(prs) > 0 else None),
        expected_raw_participation_ratio=expected,
        expected_from_artefact=src,
        contains_no_test=True,
        disclaimer=("DESCRIPTIVE ONLY. No rank test was run and no assumption "
                    "verdict is expressed or implied."),
    )
    ts = DESC.utc_stamp()
    sidecar = f"{ts.replace(':', '-')}__{a.dataset}__sweep_spectra.npz"
    for arm, r in results.items():
        if r.get("status") == "ok":
            r["spectra_sidecar"] = sidecar
    payload["spectra_sidecar"] = sidecar if spectra else None
    payload["peak_rss_mb"] = DESC.peak_rss_mb()

    meta = RIO.make_meta(
        "descriptive", "preprocessing_sweep", ts,
        dict(alpha=None, B=None, n_e=CAP, d=None, d_latent=None, D=None,
             n_env=None, seeds=[a.seed], draws_per_point=None,
             dataset=a.dataset, arms=list(ARMS), cap=CAP, hvg=a.hvg,
             platform_tag=RIO.platform_tag(), timezone="UTC",
             timestamp_suffix="Z"),
        status="CURRENT",
        note=f"preprocessing sweep for {a.dataset} at matched n={CAP}")
    meta["migrated_from"] = None
    OUT = REPO / "results" / "preprocessing"
    OUT.mkdir(parents=True, exist_ok=True)
    path = RIO.write_results(
        payload, meta,
        path=OUT / f"{ts.replace(chr(58), chr(45))}__{a.dataset}.json")
    print(f"[write] {path}", flush=True)
    if spectra:
        sc = Path(path).parent / sidecar
        np.savez_compressed(sc, **spectra)
        print(f"[write] {sc}  ({len(spectra)} spectra)", flush=True)
    print(f"[rss]   peak {DESC.peak_rss_mb()} MB", flush=True)


if __name__ == "__main__":
    main()
