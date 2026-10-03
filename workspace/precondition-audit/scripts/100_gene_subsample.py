"""Does the control-covariance dimension survive cutting the gene count?

WHY. Section IV argues the observed dimension is not an artefact of feature
count. The evidence so far is cross-dataset: K562 has 1158 genes and a
participation ratio of 455, Frangieh has 23712 and 415, so the ratio plainly
does not track p. That is suggestive but confounded, because the datasets differ
in assay, cell line and depth as well as in p. This is the within-dataset
control: hold the dataset, the arm and the cells fixed, cut the genes, and see
what the dimension does.

THE QUESTION IS OPEN AND THIS SCRIPT DOES NOT TAKE A SIDE. If the participation
ratio stays in the hundreds as p falls from 23712 to 500, the feature-count
explanation is ruled out within a dataset. If it falls roughly in proportion to
p, the dimension is partly a feature-count effect and the paper has to say so.
Both outcomes are reported the same way, in the same table, with no verdict
field and no preferred rule. Two selection rules are run precisely so that
neither can be quietly chosen after the fact: random genes, and the
highest-variance genes a real pipeline would actually keep.

NOTHING IS REIMPLEMENTED. The control-cell selection is 94_control_pool_audit's
capture of 85's own loader, and the three estimators are 85's
spectrum_estimators. A divergent second copy of either is what produced the
cap-8000 discrepancy, so there is no second copy here.

CELLS ARE HELD FIXED ACROSS EVERY GENE CONFIGURATION. The rng state is
snapshotted before the cell draw and restored, so all 45 configurations for an
arm run on the same 2000 rows, and those rows are the ones behind Table II. Any
movement in the estimators is therefore attributable to genes and not to cell
resampling. The script verifies that claim rather than asserting it: the
baseline computed on the recovered index set must equal the baseline computed
through 85's own path, and a mismatch is fatal.

STEP 0 RUNS FIRST AND NO FLAG SKIPS IT. Before a single gene is dropped, the
full-gene participation ratio for each arm is compared against the committed
descriptives artefact for that arm. Expected values are read off disk, never
typed here. If the pipeline has drifted, every subsampling number below it would
be measuring the drift, so the run stops.

NO VERDICT ON THE DATA. This reports estimators. It runs no rank test, expresses
no assumption verdict, and Phase B remains not authorised.

Usage (A100). Long-running batch job; launch detached:

    nohup python -u scripts/100_gene_subsample.py > logs/gene_subsample.log 2>&1 &

    python scripts/100_gene_subsample.py --arm frangieh_control
    python scripts/100_gene_subsample.py --step0-only     # reproduction gate, stop

Env:
    PRECOND_DATA      raw data root; required.
    PRECOND_EXTERNAL  third-party checkouts. Resolution is
                      94_control_pool_audit.py's, imported, never reimplemented.
"""
import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_DIR = REPO / "results" / "subsample"
LOCKFILE = REPO / "logs" / "gene_subsample.lock"

AUDIT = HERE / "94_control_pool_audit.py"      # control selection + provenance
DESCRIPTIVES = HERE / "85_dataset_descriptives.py"   # the three estimators
RESULTS_IO = HERE / "84_results_io.py"         # versions, platform, BLAS
REQUIRED_SCRIPTS = (AUDIT, DESCRIPTIVES, RESULTS_IO)

# The three arms of one deposit. Held separate: the arm effect dominates and
# pooling would inflate everything downstream.
ARMS = ("frangieh_coculture", "frangieh_control", "frangieh_ifng")

# Matched control sample, the same one Table II uses. Read as a constant here
# because it defines the comparison, and asserted against the artefact the
# step-0 gate reads.
CAP = 2000

# Gene counts. 1158 and 5000 match K562 and Norman exactly, so the within-dataset
# result can be laid beside the cross-dataset comparison Section IV already makes.
# 2000 and 500 extend the range downwards. Chosen before any number was seen and
# not revised afterwards.
GENE_COUNTS = (5000, 2000, 1158, 500)

# Two rules, both run, neither preferred. Random says what an arbitrary slice of
# the transcriptome gives; highest-variance says what a real HVG reduction gives,
# and it is the one that would flatter a feature-count explanation if the
# dimension were concentrated in a few loud genes.
RULES = ("random", "highest_variance")

# A single random subset is noise, so random is repeated and summarised.
N_GENE_SEEDS = 10
SEED = 0

# The baseline must reproduce the committed artefact to this tolerance. It is a
# recompute of a published number on the same rows, not a new measurement.
GATE_TOL = 1e-10


def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    real = os.makedirs
    os.makedirs = lambda *a, **k: None
    try:
        spec.loader.exec_module(mod)
    finally:
        os.makedirs = real
    return mod


def utc_stamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H-%M-%SZ")


def estimators(DESC, X, rng):
    """85's own spectrum_estimators. X already has the rows and columns wanted.

    cap is set to the row count so no further subsampling happens inside: the
    cell selection was made once, upstream, and is held fixed.
    """
    return DESC.spectrum_estimators(X, X.shape[0], rng)


def three(est):
    """The three estimators Section IV reports, pulled out of 85's return."""
    return dict(
        participation_ratio=(None if est.get("participation_ratio") is None
                             else float(est["participation_ratio"])),
        effective_rank_exp_spectral_entropy=(
            None if est.get("effective_rank_exp_spectral_entropy") is None
            else float(est["effective_rank_exp_spectral_entropy"])),
        n_components_for_variance_95pct=(
            None if not est.get("n_components_for_variance")
            else int(est["n_components_for_variance"]["95pct"])),
        top_eigenvalue_share=(None if est.get("top_eigenvalue_share") is None
                              else float(est["top_eigenvalue_share"])),
        rank_bound=(None if est.get("rank_bound") is None
                    else int(est["rank_bound"])),
    )


# ===================================================================== step 0
def prepare_arm(AUD, DESC, arm, args):
    """Capture the control matrix, fix the cells, and reproduce the baseline.

    Returns everything downstream needs: the fixed 2000-row matrix at full gene
    width, the provenance block, and the baseline estimators.
    """
    ns = argparse.Namespace(hvg=args.hvg, seed=args.seed)
    Xc, rng, X_full, paths = AUD.control_matrix(DESC, arm, ns)
    n_all, p_all = Xc.shape
    if n_all < CAP:
        sys.exit(f"[fatal] {arm}: only {n_all} control cells, fewer than the "
                 f"matched sample {CAP}. The comparison this script makes does "
                 f"not exist for this arm. Refusing.")

    # Snapshot BEFORE the draw, so the same rows can be recovered for every gene
    # configuration and so the baseline goes through 85's own path untouched.
    state = rng.bit_generator.state
    baseline_est = DESC.spectrum_estimators(Xc, CAP, rng)

    # Recover the index set 85 drew, by repeating the draw from the same state.
    # This is the one line that mirrors 85 rather than calling it, and it exists
    # only to name the rows; the check below is what makes that safe.
    rng.bit_generator.state = state
    idx = rng.choice(n_all, size=CAP, replace=False)
    X = np.asarray(Xc[idx], dtype=np.float64)

    check = DESC.spectrum_estimators(X, CAP, rng)
    a, b = (baseline_est.get("participation_ratio"),
            check.get("participation_ratio"))
    if a is None or b is None or abs(float(a) - float(b)) > GATE_TOL:
        sys.exit(f"[fatal] {arm}: the recovered cell index set does not "
                 f"reproduce 85's own baseline.\n"
                 f"        through 85            : {a!r}\n"
                 f"        on the recovered rows : {b!r}\n"
                 f"        Every gene configuration below would then be run on "
                 f"different cells than\n"
                 f"        the baseline, and the comparison would be "
                 f"meaningless. Refusing.")

    prov = dict(
        resolved_input_paths=paths,
        input_path=(paths[0] if paths else None),
        n_cells=(int(X_full.shape[0]) if X_full is not None else None),
        n_genes_full=int(p_all),
        n_control_cells_available=int(n_all),
        n_control_cells_used=int(CAP),
        control_value_array=AUD.value_fingerprint(Xc, "control matrix"),
        matched_sample_value_array=AUD.value_fingerprint(
            X, f"control matrix at n={CAP}, full gene width"),
        cells_fixed_across_gene_configurations=True,
    )
    return X, rng, prov, three(baseline_est)


def step0_gate(AUD, DESC, arms, args):
    """Full-gene PR must reproduce the committed artefact. Runs before anything."""
    print("=" * 78, flush=True)
    print(f" STEP 0: reproduction gate, full gene width, n={CAP}", flush=True)
    print(" Expected values are READ from results/descriptives/, never typed "
          "here.", flush=True)
    print("=" * 78, flush=True)

    prepared, fatal = {}, []
    for arm in arms:
        exp, src, exp_cap = AUD.expected_pr(arm, CAP)
        if exp is None or exp_cap != CAP:
            fatal.append(
                f"{arm}: no CURRENT descriptives artefact carries a "
                f"participation ratio at cap {CAP}.\n"
                f"        There is nothing to reproduce against and no literal "
                f"to fall back on.")
            continue
        print(f"\n--- [{arm}] ---", flush=True)
        print(f"    expected {exp!r}", flush=True)
        print(f"    from     results/descriptives/{src}", flush=True)
        X, rng, prov, base = prepare_arm(AUD, DESC, arm, args)
        got = base["participation_ratio"]
        ok = got is not None and abs(got - exp) <= GATE_TOL
        print(f"    got      {got!r}", flush=True)
        print(f"    genes    {prov['n_genes_full']}   cells "
              f"{prov['n_control_cells_used']} of "
              f"{prov['n_control_cells_available']}", flush=True)
        print(f"    input    {prov['input_path']}", flush=True)
        print(f"    verdict  {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            fatal.append(
                f"{arm}: expected {exp!r}, got {got!r}, "
                f"|diff| = {abs((got or 0) - exp):.3e} > tol {GATE_TOL:g}\n"
                f"        read from results/descriptives/{src}")
            continue
        prepared[arm] = dict(X=X, rng=rng, provenance=prov, baseline=base,
                             expected=exp, expected_source=src)

    print("\n" + "=" * 78)
    print(f" {'arm':<24}{'expected':>14}{'got':>18}  verdict")
    print("-" * 78)
    for arm in arms:
        rec = prepared.get(arm)
        if rec is None:
            print(f" {arm:<24}{'-':>14}{'-':>18}  FAIL")
            continue
        print(f" {arm:<24}{rec['expected']:>14.4f}"
              f"{rec['baseline']['participation_ratio']:>18.10f}  PASS")
    print("-" * 78)
    print("=" * 78)

    if fatal:
        print("\n[fatal] STEP-0 REPRODUCTION GATE FAILED:", file=sys.stderr)
        for f in fatal:
            print(f"  - {f}", file=sys.stderr)
        print("        The pipeline has drifted from the committed artefacts, "
              "so every subsampling\n"
              "        number would be measuring the drift rather than the "
              "gene count. STOPPING\n"
              "        before a single gene is dropped.", file=sys.stderr)
        sys.exit(1)
    print("\n  [step 0] PASS\n", flush=True)
    return prepared


# ================================================================ subsampling
def gene_indices(X, n_genes, rule, seed):
    """Which columns to keep. Neither rule is tuned and both are always run."""
    p = X.shape[1]
    if n_genes > p:
        return None
    if rule == "random":
        return np.random.default_rng(seed).choice(p, size=n_genes,
                                                  replace=False)
    if rule == "highest_variance":
        # Variance on the fixed matched sample, so the ranking is a property of
        # the rows the estimators see and not of the whole pool.
        v = X.var(axis=0)
        return np.argsort(v, kind="stable")[::-1][:n_genes]
    raise ValueError(f"unknown rule {rule!r}")


def run_arm(AUD, DESC, arm, prep, args):
    X, rng = prep["X"], prep["rng"]
    p_all = X.shape[1]
    records = [dict(
        arm=arm, n_genes=int(p_all), rule="all_genes", seed=None,
        n_cells=int(CAP), is_baseline=True,
        estimators=prep["baseline"],
        expected_participation_ratio=prep["expected"],
        expected_source=prep["expected_source"],
        reproduces_expected=True,
        gene_fraction=1.0,
    )]
    print(f"\n--- [{arm}] baseline {p_all} genes: "
          f"PR={prep['baseline']['participation_ratio']:.4f}  "
          f"effr={prep['baseline']['effective_rank_exp_spectral_entropy']:.1f}  "
          f"var95={prep['baseline']['n_components_for_variance_95pct']}",
          flush=True)

    for n_genes in GENE_COUNTS:
        if n_genes > p_all:
            print(f"    [skip] {n_genes} genes exceeds the arm's {p_all}",
                  flush=True)
            continue
        for rule in RULES:
            seeds = (list(range(N_GENE_SEEDS)) if rule == "random" else [None])
            got = []
            for s in seeds:
                cols = gene_indices(X, n_genes, rule, s)
                est = three(estimators(DESC, np.ascontiguousarray(X[:, cols]),
                                       rng))
                records.append(dict(
                    arm=arm, n_genes=int(n_genes), rule=rule,
                    seed=(None if s is None else int(s)),
                    n_cells=int(CAP), is_baseline=False,
                    estimators=est,
                    gene_fraction=float(n_genes) / float(p_all),
                    participation_ratio_over_baseline=(
                        None if est["participation_ratio"] is None
                        else est["participation_ratio"]
                        / prep["baseline"]["participation_ratio"]),
                ))
                got.append(est["participation_ratio"])
            arr = np.asarray([g for g in got if g is not None],
                             dtype=np.float64)
            if arr.size == 0:
                print(f"    {n_genes:>6} {rule:<18} no estimate", flush=True)
                continue
            if rule == "random":
                print(f"    {n_genes:>6} {rule:<18} PR mean={arr.mean():8.3f}  "
                      f"sd={arr.std(ddof=1) if arr.size > 1 else 0.0:7.3f}  "
                      f"min={arr.min():8.3f}  max={arr.max():8.3f}  "
                      f"n={arr.size}", flush=True)
            else:
                print(f"    {n_genes:>6} {rule:<18} PR     ={arr[0]:8.3f}",
                      flush=True)
    return records


def summarise(records):
    """Mean and spread per (n_genes, rule). Reported; nothing is concluded."""
    out = []
    keys = sorted({(r["n_genes"], r["rule"]) for r in records
                   if not r["is_baseline"]})
    for n_genes, rule in keys:
        sel = [r for r in records
               if r["n_genes"] == n_genes and r["rule"] == rule]
        for field in ("participation_ratio",
                      "effective_rank_exp_spectral_entropy",
                      "n_components_for_variance_95pct"):
            v = np.asarray([r["estimators"][field] for r in sel
                            if r["estimators"][field] is not None],
                           dtype=np.float64)
            if v.size == 0:
                continue
            out.append(dict(
                n_genes=int(n_genes), rule=rule, estimator=field,
                n=int(v.size), mean=float(v.mean()),
                sd=(float(v.std(ddof=1)) if v.size > 1 else 0.0),
                min=float(v.min()), max=float(v.max()),
                median=float(np.median(v))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS, help="one arm only (default: all three)")
    ap.add_argument("--step0-only", action="store_true",
                    help="run the reproduction gate and stop")
    ap.add_argument("--hvg", type=int, default=None,
                    help="Frangieh gene cap passed to the loader; default unset")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", default=str(OUT_DIR))
    args = ap.parse_args()

    for q in REQUIRED_SCRIPTS:
        if not q.exists():
            sys.exit(f"[fatal] {q.name} is not in this repository (looked in "
                     f"{q.parent}). Not vendoring a copy and not "
                     f"reimplementing it: a divergent second copy of the "
                     f"estimator is the defect this script exists to avoid.")
    AUD = _load(AUDIT, "_audit94")
    DESC = _load(DESCRIPTIVES, "_descriptives")
    RIO = _load(RESULTS_IO, "_results_io")

    arms = [args.arm] if args.arm else list(ARMS)
    AUD.step0_gate(arms)                 # data presence and paths, loud on miss

    # THE REPRODUCTION GATE. Called here, before any subsampling, and reachable
    # by no flag that skips it: --step0-only stops AFTER it, never instead of it.
    prepared = step0_gate(AUD, DESC, arms, args)
    if args.step0_only:
        print("  --step0-only: gate passed, no gene was dropped.")
        return

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
            sys.exit(f"[fatal] another run is live (pid {other}, lock "
                     f"{LOCKFILE}). Two concurrent batches into one output "
                     f"directory is how the 11 Aug artefacts had to be "
                     f"discarded. Refusing.")
        print(f"[lock] stale lock from pid {other or 'unknown'}; reclaiming",
              flush=True)
        LOCKFILE.unlink()
    LOCKFILE.write_text(str(os.getpid()))

    try:
        out = Path(args.outdir)
        if not args.arm and out.exists():
            print(f"[clean] rm -rf {out}", flush=True)
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        usable = [g for g in GENE_COUNTS if g <= min(
            prepared[a]["X"].shape[1] for a in arms)]
        per_arm = 1 + len(usable) * (N_GENE_SEEDS + 1)
        written, failed, all_records = [], [], []

        for arm in arms:
            try:
                recs = run_arm(AUD, DESC, arm, prepared[arm], args)
            except Exception as e:                          # noqa: BLE001
                # One artefact per arm, written as it completes, so a later
                # failure cannot discard an arm that already finished.
                print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
                failed.append((arm, f"{type(e).__name__}: {e}"))
                continue
            payload = dict(
                arm=arm,
                matched_sample=CAP,
                gene_counts=list(usable),
                rules=list(RULES),
                n_gene_seeds=N_GENE_SEEDS,
                question=("Does the observed control-covariance dimension "
                          "survive reducing the gene count within one dataset? "
                          "Reported without a verdict: a participation ratio "
                          "roughly constant in p rules out a feature-count "
                          "explanation, one falling in proportion to p supports "
                          "it, and both are stated the same way."),
                provenance=prepared[arm]["provenance"],
                baseline=prepared[arm]["baseline"],
                expected_participation_ratio=prepared[arm]["expected"],
                expected_source=prepared[arm]["expected_source"],
                records=recs,
                summary=summarise(recs),
                n_configurations=len(recs),
                n_configurations_expected=per_arm,
                complete=bool(len(recs) == per_arm),
                versions=RIO.versions(),
                platform_tag=RIO.platform_tag(),
                repo_commit=RIO.git_commit(),
                audit_script_sha256=AUD.sha256(AUDIT),
                descriptives_script_sha256=AUD.sha256(DESCRIPTIVES),
                contains_no_test=True,
                disclaimer=("DESCRIPTIVE ONLY. No rank test was run and no "
                            "assumption verdict is expressed or implied."),
                peak_rss_mb=DESC.peak_rss_mb(),
            )
            path = out / f"{utc_stamp()}__{arm}__gene_subsample.json"
            path.write_text(json.dumps(payload, indent=2) + "\n")
            written.append(path)
            all_records.extend(recs)
            print(f"    [write] {path}  ({len(recs)}/{per_arm} configurations)",
                  flush=True)

        print("\n" + "=" * 100)
        print(f" {'arm':<22}{'genes':>7}{'rule':<18}{'PR mean':>10}{'sd':>9}"
              f"{'PR/base':>9}{'effr':>10}{'var95':>8}")
        print("-" * 100)
        for arm in arms:
            recs = [r for r in all_records if r["arm"] == arm]
            base = next((r for r in recs if r["is_baseline"]), None)
            if base:
                e = base["estimators"]
                print(f" {arm:<22}{base['n_genes']:>7}{'all_genes':<18}"
                      f"{e['participation_ratio']:>10.3f}{'-':>9}{1.0:>9.3f}"
                      f"{e['effective_rank_exp_spectral_entropy']:>10.1f}"
                      f"{e['n_components_for_variance_95pct']:>8}")
            for n_genes, rule in sorted({(r["n_genes"], r["rule"]) for r in recs
                                         if not r["is_baseline"]}):
                sel = [r for r in recs if r["n_genes"] == n_genes
                       and r["rule"] == rule]
                pr = np.asarray([r["estimators"]["participation_ratio"]
                                 for r in sel
                                 if r["estimators"]["participation_ratio"]
                                 is not None])
                er = np.asarray([r["estimators"]
                                 ["effective_rank_exp_spectral_entropy"]
                                 for r in sel])
                v9 = np.asarray([r["estimators"]
                                 ["n_components_for_variance_95pct"]
                                 for r in sel])
                ratio = np.asarray([r["participation_ratio_over_baseline"]
                                    for r in sel
                                    if r["participation_ratio_over_baseline"]
                                    is not None])
                print(f" {'':<22}{n_genes:>7}{rule:<18}{pr.mean():>10.3f}"
                      f"{(pr.std(ddof=1) if pr.size > 1 else 0.0):>9.3f}"
                      f"{ratio.mean():>9.3f}{er.mean():>10.1f}"
                      f"{v9.mean():>8.1f}")
        print("-" * 100)
        print(f" artefacts: {len(written)} of {len(arms)} arm(s); "
              f"{len(all_records)} configuration(s), "
              f"{per_arm} expected per arm")
        print(" No verdict is recorded. The table is the result.")
        print("=" * 100)

        want = len(arms) * per_arm
        if failed or len(all_records) != want or len(written) != len(arms):
            print(f"\n[fatal] SHORTFALL: {len(all_records)} configuration(s) "
                  f"for {want} requested, {len(written)} artefact(s) for "
                  f"{len(arms)} arm(s).", file=sys.stderr)
            for a, why in failed:
                print(f"  - {a}: {why}", file=sys.stderr)
            for q in written:
                print(f"  kept: {q}", file=sys.stderr)
            print("        The artefacts above ARE complete for their own arm "
                  "and are not discarded.\n"
                  "        Rerun only the missing arm with --arm.",
                  file=sys.stderr)
            sys.exit(1)
    finally:
        if LOCKFILE.exists():
            LOCKFILE.unlink()


if __name__ == "__main__":
    main()
