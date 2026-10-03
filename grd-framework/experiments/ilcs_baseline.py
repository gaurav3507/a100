"""iLCS ungated baseline on the E3 projected inputs, with a size-matched null.

iLCS (interventional Latent Causal Shift detector) is reimplemented from Chen et al.
2024 (NeurIPS, arXiv 2410.24059), Algorithm 1 and the empirical L variant of their
Section 5.1, because the original repository github.com/TianyuCodings/iLCS is 404.

Purpose: an ungated latent mechanism-shift detector, run on the SAME D=10 control-fit
projected space the E3 precision gate uses, so it is apples-to-apples with the gate.
It shows whether such a detector, WITHOUT a precondition (size-matched) gate, fires on
sampling noise (pure-control fakes), on systematic control heterogeneity (structured
splits), and on real perturbations. Two arms per environment:
  NAIVE       fixed alpha=0.2 (the paper default for d <= 10);
  CALIBRATED  alpha_env = 95th percentile of max_i L over B size-matched control-vs-
              control null draws (same Lesson-2 size-matching as pr.null_threshold).

Reuses, verbatim, the E3 panel's pr-loader and control-projection recipe and the
e3_rpe1_confound_check structured-split construction. Modifies no existing e3 script.

Cost note: the control-side ICA of the full control set Yobs is identical across every
environment and every null draw, so it is computed ONCE per dataset (control_rows) and
reused via ilcs_L_given_ctrl. The calibrated null is computed per SIZE BUCKET (6
log-spaced buckets over the powered-pert sizes), not per distinct cell count, so the
null costs <= 6 * B small-matrix fits per dataset rather than one null per environment.
These two changes only remove redundant ICA solves; the L statistic, psi-sort, the
Yobs-vs-n_env geometry, and the null seeding are unchanged.

Data paths are resolved through GRD_DATA_ROOT (legacy default /workspace/external).
Run --smoke only where the K562 h5ad is reachable. anndata is imported lazily inside
the loader so the iLCS functions can be exercised without it.
"""
import sys
import json
import argparse
import traceback
import importlib.util
from pathlib import Path

import numpy as np
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis

from data_paths import perturbseq_path

# --- panel importlib block, verbatim (loads the precision-gate readout module) ---
spec = importlib.util.spec_from_file_location("pr", "src/gate/precision_readout.py")
pr = importlib.util.module_from_spec(spec); spec.loader.exec_module(pr)

D = 10; NMIN = 200; SEED = 0   # SEED is overridden by --seed in main()

# The three dataset invocations, taken verbatim from experiments/e3_stability_perturbseq.py
# run(name, path, ctrl, single). ctrl is the control guide label ('' = control).
DATASETS = [
    ("K562", str(perturbseq_path("causalbench_k562.h5ad")), "", False),
    ("RPE1", str(perturbseq_path("causalbench_rpe1.h5ad")), "", False),
    ("Norman", str(perturbseq_path("Norman2019_raw.h5ad")), "", True),
]
GATE_DECISIONS = {
    "K562": "results/e3_decisions/K562.json",
    "RPE1": "results/e3_decisions/RPE1.json",
    "Norman": "results/e3_decisions/Norman.json",
}
OUT = Path("results/ilcs_baseline")

def seed_out(seed):
    """Per-seed output directory so multiple seeds never overwrite each other."""
    return OUT / f"seed{seed}"
NAIVE_ALPHA = 0.2
B_NULL = 20
N_BUCKETS = 6


def _log(name, msg):
    """Progress line, flushed immediately so the log shows where a run is at all times."""
    print(f"[{name}] {msg}", flush=True)


# ------------------------------------------------------------ iLCS (Chen 2024 Alg 1)
ICA_MAX_ITER = 500
ICA_TOL = 1e-3


def _ica_rows(Y, seed):
    """FastICA on one environment (already in D=10). Fit with the default 'parallel'
    algorithm at a looser tol; if it stalls (n_iter_ hits ICA_MAX_ITER, common on
    ill-conditioned real gene-expression PCs), retry the fit ONCE with 'deflation'.
    Returns the psi-sorted unmixing rows M (D x D), a unit-variance-ok flag, a converged
    flag (final fit's n_iter_ < ICA_MAX_ITER), and the algorithm that produced the fit.
    """
    Y = np.asarray(Y)
    algo = "parallel"
    ica = FastICA(n_components=D, algorithm=algo, whiten='unit-variance',
                  max_iter=ICA_MAX_ITER, tol=ICA_TOL, random_state=seed)
    S = ica.fit_transform(Y)
    if int(getattr(ica, "n_iter_", ICA_MAX_ITER)) >= ICA_MAX_ITER:
        algo = "deflation"
        ica = FastICA(n_components=D, algorithm=algo, whiten='unit-variance',
                      max_iter=ICA_MAX_ITER, tol=ICA_TOL, random_state=seed)
        S = ica.fit_transform(Y)
    M = ica.components_                       # D x D here (sources by projected dims)
    unit_ok = bool(np.allclose(S.var(0), 1.0, atol=0.1))
    converged = bool(int(getattr(ica, "n_iter_", ICA_MAX_ITER)) < ICA_MAX_ITER)
    psi = np.array([np.mean(np.abs(S[:, i]) <= 1.0) for i in range(S.shape[1])])
    order = np.argsort(psi)                   # ascending psi: least-Gaussian-ish first
    return M[order], unit_ok, converged, algo


def _L_from_rows(Mc, Me):
    """The iLCS empirical L statistic (Chen 2024 Sec 5.1) between two psi-sorted
    unmixing-row matrices. Unchanged from the original ilcs_L body.
    """
    num = np.abs(np.abs(Mc) - np.abs(Me)).sum(1)
    den = np.abs(Mc).sum(1) + np.abs(Me).sum(1) + 1e-12
    return num / den


def ilcs_L(Y_ctrl, Y_env, seed):
    """Length-D iLCS shift vector L between control and one environment (K=2), fitting
    ICA on BOTH arms. Retained for the standalone self-test; dataset work uses
    ilcs_L_given_ctrl with a precomputed control side.
    Returns (L, unit_var_ok, converged, used_deflation).
    """
    Mc, okc, cc, ac = _ica_rows(np.asarray(Y_ctrl), seed)
    Me, oke, ce, ae = _ica_rows(np.asarray(Y_env), seed)
    return _L_from_rows(Mc, Me), bool(okc and oke), bool(cc and ce), bool("deflation" in (ac, ae))


def ilcs_L_given_ctrl(control_rows, Y_env, seed):
    """iLCS L between a PRECOMPUTED control-row matrix and one environment. The control
    side (full Yobs) is identical across every environment and null draw, so it is fit
    once per dataset and passed in here. Only the environment arm is fit. This changes
    nothing in the statistic; it removes the redundant re-fit of the full control ICA.
    Returns (L, env_unit_ok, env_converged, used_deflation).
    """
    Mc, okc, cc, ac = control_rows
    Me, oke, ce, ae = _ica_rows(np.asarray(Y_env), seed)
    return _L_from_rows(Mc, Me), bool(okc and oke), bool(cc and ce), bool("deflation" in (ac, ae))


def ilcs_fires(L, alpha):
    return bool(np.any(np.asarray(L) > alpha))


# ------------------------------------------------------------ calibrated alpha per bucket
def calib_alpha_bucket(proj, Xc, control_rows, rep_size, B, rng):
    """95th percentile of max_i L over B size-matched null draws for one size bucket.
    Mirrors the real-test geometry: control side fixed at control_rows (from full Yobs),
    env side a control subsample of the bucket's representative size rep_size. Each draw
    uses a per-draw ICA seed SEED+b so the null carries ICA-init variation. Same
    size-matching principle as pr.null_threshold (Lesson 2).
    """
    N = len(Xc)
    m = rep_size if rep_size < N else N
    maxL = np.empty(B)
    for b in range(B):
        idx = rng.choice(N, m, replace=False)
        Lb, _, _, _ = ilcs_L_given_ctrl(control_rows, proj(Xc[idx]), SEED + b)
        maxL[b] = float(Lb.max())
    return float(np.percentile(maxL, 95))


def _bucket_index(n_cells, edges):
    """Bucket index for a cell count given the log-spaced bucket edges (len N_BUCKETS+1).
    np.digitize with the interior edges; clipped to [0, N_BUCKETS-1].
    """
    return int(np.clip(np.digitize(n_cells, edges[1:-1]), 0, N_BUCKETS - 1))


def _eval_env(Y_env, control_rows, proj, Xc, n_cells, rng, B, alpha_cache,
              edges, rep_sizes, env_id, kind):
    L, unit_ok, converged, used_deflation = ilcs_L_given_ctrl(control_rows, Y_env, SEED)
    naive = ilcs_fires(L, NAIVE_ALPHA)
    bi = _bucket_index(n_cells, edges)
    if bi not in alpha_cache:
        alpha_cache[bi] = calib_alpha_bucket(proj, Xc, control_rows, rep_sizes[bi], B, rng)
    alpha_env = alpha_cache[bi]
    calibrated = bool(float(L.max()) > alpha_env)
    return dict(kind=kind, id=env_id, n_cells=int(n_cells), bucket=bi,
                L=[round(float(x), 5) for x in L],
                L_max=round(float(L.max()), 5),
                naive_fires=naive, alpha_env=round(alpha_env, 5),
                calibrated_fires=calibrated, ica_unit_var_ok=unit_ok,
                ica_converged=converged, ica_used_deflation=used_deflation)


# ------------------------------------------------------------------ per dataset run
def run_dataset(name, path, ctrl, single, smoke, outdir):
    import anndata as ad                      # lazy: only needed for the real run
    rng = np.random.default_rng(SEED)
    _log(name, f"loading {path}")
    A = ad.read_h5ad(path)
    X = A.X.toarray().astype(np.float64) if hasattr(A.X, "toarray") else np.asarray(A.X, np.float64)
    g = A.obs['guide_ids'].astype(str).values
    del A
    _log(name, f"loaded X {X.shape}, {len(np.unique(g))} guide labels")
    if single:
        is_single = lambda s: s != ctrl and s != "" and "," not in s and "+" not in s and "_" not in s
        cand = sorted({s for s in np.unique(g) if is_single(s)})
    else:
        cand = [p for p in np.unique(g) if p != ctrl]

    Xc = X[g == ctrl]
    mu = Xc.mean(0); _, _, Vt = np.linalg.svd(Xc - mu, full_matrices=False); Bp = Vt[:D].T
    proj = lambda M: (M - mu) @ Bp
    Yobs = proj(Xc); Zc = Yobs
    perts = [p for p in cand if (g == p).sum() >= NMIN]
    sizes = [int((g == p).sum()) for p in perts]
    _log(name, f"control {len(Xc)} cells | powered perts {len(perts)} | sizes {min(sizes) if sizes else None}..{max(sizes) if sizes else None}")
    if not perts:
        raise RuntimeError(f"{name}: no powered perturbations at NMIN={NMIN}; cannot build size buckets")

    n_fake, n_rand, n_pert, B = (50, 20, len(perts), B_NULL)
    if smoke:
        n_fake, n_rand, n_pert, B = (5, 5, 20, 10)

    # Control-side ICA, computed ONCE and reused for every environment and null draw.
    _log(name, "fitting control ICA once")
    control_rows = _ica_rows(Yobs, SEED)
    _log(name, f"control ICA done (converged={control_rows[2]}, algo={control_rows[3]})")

    # Size buckets: N_BUCKETS log-spaced edges over the powered-pert sizes. One null per
    # bucket, evaluated at the bucket's geometric-mean representative size.
    smin, smax = int(min(sizes)), int(max(sizes))
    edges = np.unique(np.geomspace(smin, max(smax, smin + 1), N_BUCKETS + 1).astype(int))
    while len(edges) < N_BUCKETS + 1:                       # guard tiny/degenerate ranges
        edges = np.unique(np.append(edges, edges[-1] + 1))
    edges = edges[:N_BUCKETS + 1]
    rep_sizes = [int(round(np.sqrt(edges[i] * edges[i + 1]))) for i in range(N_BUCKETS)]

    # control-projection kurtosis (ICA needs at most one near-Gaussian component)
    kurt = kurtosis(Zc, axis=0, fisher=True)
    n_near_gaussian = int(np.sum(np.abs(kurt) < 0.5))

    alpha_cache = {}
    records = []

    # (i) size-matched pure-control fakes (size drawn from powered-pert sizes)
    _log(name, f"fakes x{n_fake}")
    for f in range(n_fake):
        n = int(rng.choice(sizes))
        sub = proj(Xc[rng.choice(len(Xc), n, replace=False)])
        records.append(_eval_env(sub, control_rows, proj, Xc, n, rng, B, alpha_cache,
                                 edges, rep_sizes, f, "fake"))

    # (ii) random control splits
    _log(name, f"random splits x{n_rand}")
    for r in range(n_rand):
        n = int(rng.choice(sizes))
        idx = rng.choice(len(Xc), n, replace=False)
        records.append(_eval_env(proj(Xc[idx]), control_rows, proj, Xc, n, rng, B,
                                 alpha_cache, edges, rep_sizes, r, "random"))

    # (iii) structured control splits along top control PCs (verbatim from confound check)
    pc = Zc - Zc.mean(0); msz = int(np.median(sizes)); struct = []
    for j in range(D):
        o = np.argsort(pc[:, j]); struct += [o[:msz], o[-msz:]]
    if smoke:
        struct = struct[:5]
    _log(name, f"structured splits x{len(struct)}")
    for si, sub_idx in enumerate(struct):
        records.append(_eval_env(proj(Xc[sub_idx]), control_rows, proj, Xc, len(sub_idx),
                                 rng, B, alpha_cache, edges, rep_sizes, si, "struct"))

    # (iv) powered real perturbations
    pert_list = perts[:n_pert] if smoke else perts
    _log(name, f"real perts x{len(pert_list)}")
    for p in pert_list:
        Yp = proj(X[g == p]); n = len(Yp)
        records.append(_eval_env(Yp, control_rows, proj, Xc, n, rng, B, alpha_cache,
                                 edges, rep_sizes, p, "pert"))

    def rate(kind, key):
        rows = [r for r in records if r["kind"] == kind]
        return round(float(np.mean([r[key] for r in rows])), 4) if rows else None

    evaluated_pert_ids = {
        str(r["id"]) for r in records if r["kind"] == "pert"
    }
    calib_pert_ids = {
        str(r["id"]) for r in records
        if r["kind"] == "pert" and r["calibrated_fires"]
    }
    jaccard, jreason = _jaccard_vs_gate(
        name, calib_pert_ids, evaluated_pert_ids)

    unit_warn = int(sum(1 for r in records if not r["ica_unit_var_ok"]))
    nonconv = int(sum(1 for r in records if not r["ica_converged"]))
    defl = int(sum(1 for r in records if r["ica_used_deflation"]))
    # non-convergence is an intended measurement. control_rows is fit once; the env fit
    # of each record is one solve, so the tracked denominator is 1 per record plus the
    # single control fit. Null-draw fits are not tracked.
    ctrl_converged = bool(control_rows[2])
    total_ica_fits = len(records) + 1
    nonconv_total = nonconv + (0 if ctrl_converged else 1)
    nonconv_frac = round(nonconv_total / total_ica_fits, 4) if total_ica_fits else 0.0
    per_dataset = dict(dataset=name, control_label=repr(ctrl), single_gene_only=single,
                       d_proj=D, nmin=NMIN, n_control=int(len(Xc)),
                       n_powered_perts=len(perts), naive_alpha=NAIVE_ALPHA, B_null=B,
                       n_buckets=N_BUCKETS, bucket_edges=[int(x) for x in edges],
                       bucket_rep_sizes=rep_sizes,
                       control_ica_converged=ctrl_converged,
                       kurtosis=[round(float(x), 4) for x in kurt],
                       n_dims_near_gaussian=n_near_gaussian,
                       ica_unit_var_warn=unit_warn, ica_nonconverged_count=nonconv_total,
                       ica_nonconverged_frac=nonconv_frac, total_ica_fits=total_ica_fits,
                       ica_nonconverged_frac_note="denominator = len(records) env fits + 1 control fit (null-draw fits not counted)",
                       ica_deflation_count=defl, records=records)
    summary = dict(
        dataset=name,
        naive_fake_rate=rate("fake", "naive_fires"), calib_fake_rate=rate("fake", "calibrated_fires"),
        naive_random_rate=rate("random", "naive_fires"), calib_random_rate=rate("random", "calibrated_fires"),
        naive_struct_rate=rate("struct", "naive_fires"), calib_struct_rate=rate("struct", "calibrated_fires"),
        naive_pert_rate=rate("pert", "naive_fires"), calib_pert_rate=rate("pert", "calibrated_fires"),
        jaccard_calib_vs_gate=jaccard, jaccard_note=jreason,
        n_dims_near_gaussian=n_near_gaussian, ica_unit_var_warn=unit_warn,
        ica_nonconverged_count=nonconv_total, ica_nonconverged_frac=nonconv_frac,
        total_ica_fits=total_ica_fits, ica_deflation_count=defl)
    per_dataset["summary"] = summary
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{name}.json").write_text(json.dumps(per_dataset, indent=2))
    _log(name, "written " + str(outdir / f"{name}.json"))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _jaccard_vs_gate(name, calib_ids, evaluated_ids):
    """Jaccard on the perturbations evaluated by this iLCS invocation."""
    path = Path(GATE_DECISIONS[name])
    if not path.exists():
        return None, f"gate decision JSON {path} not found"
    gate = json.loads(path.read_text())
    gate_ids = {
        str(row["environment"]) for row in gate["decisions"]
        if row["bh_detect"] and str(row["environment"]) in evaluated_ids
    }
    union = calib_ids | gate_ids
    jaccard = round(len(calib_ids & gate_ids) / len(union), 4) if union else 1.0
    note = (
        f"corrected-disjoint BH decisions; shared evaluated universe "
        f"n={len(evaluated_ids)}; gate source={path}"
    )
    return jaccard, note


def _summary_from_perdataset(pd):
    """Rebuild a dataset's summary from its per-dataset JSON. Uses the stored "summary"
    key when present; otherwise recomputes the rates from the records (this handles a
    per-dataset file written by an earlier version that lacked the key).
    """
    recs = pd["records"]
    def rate(kind, key):
        rows = [r for r in recs if r["kind"] == kind]
        return round(float(np.mean([r[key] for r in rows])), 4) if rows else None
    if "summary" in pd:
        summary = dict(pd["summary"])
    else:
        summary = dict(
            dataset=pd["dataset"],
            naive_fake_rate=rate("fake", "naive_fires"),
            calib_fake_rate=rate("fake", "calibrated_fires"),
            naive_random_rate=rate("random", "naive_fires"),
            calib_random_rate=rate("random", "calibrated_fires"),
            naive_struct_rate=rate("struct", "naive_fires"),
            calib_struct_rate=rate("struct", "calibrated_fires"),
            naive_pert_rate=rate("pert", "naive_fires"),
            calib_pert_rate=rate("pert", "calibrated_fires"),
            n_dims_near_gaussian=pd.get("n_dims_near_gaussian"),
            ica_unit_var_warn=pd.get("ica_unit_var_warn"),
            ica_nonconverged_count=pd.get("ica_nonconverged_count"),
            ica_nonconverged_frac=pd.get("ica_nonconverged_frac"),
            total_ica_fits=pd.get("total_ica_fits"),
            ica_deflation_count=pd.get("ica_deflation_count"),
        )
    evaluated = {str(r["id"]) for r in recs if r["kind"] == "pert"}
    calibrated = {
        str(r["id"]) for r in recs
        if r["kind"] == "pert" and r["calibrated_fires"]
    }
    if evaluated:
        jaccard, note = _jaccard_vs_gate(
            pd["dataset"], calibrated, evaluated)
        summary["jaccard_calib_vs_gate"] = jaccard
        summary["jaccard_note"] = note
    return summary


def write_summary(mode, outdir, seed):
    """Assemble summary.json from every per-dataset JSON currently on disk, plus any
    error files. Called at the end of every invocation, so a run of one dataset never
    discards the results of another.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    per, errors = {}, {}
    for name, _, _, _ in DATASETS:
        pj = outdir / f"{name}.json"
        ej = outdir / f"{name}.error.txt"
        if pj.exists():
            per[name] = _summary_from_perdataset(json.loads(pj.read_text()))
        if ej.exists():
            errors[name] = ej.read_text()[-2000:]
    (outdir / "summary.json").write_text(json.dumps(dict(
        mode=mode, seed=seed,
        datasets_complete=sorted(per.keys()),
        datasets_failed=sorted(errors.keys()),
        imported_from="experiments/e3_perturbseq_panel.py (pr-loader, proj recipe), "
                      "experiments/e3_rpe1_confound_check.py (structured splits), "
                      "experiments/e3_stability_perturbseq.py (dataset tuples)",
        ilcs="reimplemented from Chen et al. 2024 (arXiv 2410.24059) Algorithm 1; "
             "original repo github.com/TianyuCodings/iLCS is unavailable (404)",
        per_dataset=per, errors=errors), indent=2))
    return per, errors


def aggregate_seeds():
    """Read every results/ilcs_baseline/seed*/summary.json and write an aggregate with
    mean and SD of each rate across seeds, matching how the E3 panel reports stability.
    """
    seed_dirs = sorted(OUT.glob("seed*"))
    seeds = [int(p.name[4:]) for p in seed_dirs if (p / "summary.json").exists()]
    if not seeds:
        sys.exit("no per-seed summaries found under results/ilcs_baseline/seed*/")
    for path in seed_dirs:
        summary_path = path / "summary.json"
        if summary_path.exists():
            stored = json.loads(summary_path.read_text())
            write_summary(stored.get("mode", "full"), path, int(path.name[4:]))

    keys = ["naive_fake_rate","calib_fake_rate","naive_random_rate","calib_random_rate",
            "naive_struct_rate","calib_struct_rate","naive_pert_rate","calib_pert_rate",
            "jaccard_calib_vs_gate", "ica_nonconverged_frac"]
    names = [d[0] for d in DATASETS]
    agg = {}
    for name in names:
        per_key = {k: [] for k in keys}
        n_present = 0
        for p in seed_dirs:
            sj = p / "summary.json"
            if not sj.exists():
                continue
            s = json.loads(sj.read_text())["per_dataset"].get(name)
            if s is None:
                continue
            n_present += 1
            for k in keys:
                v = s.get(k)
                if v is not None:
                    per_key[k].append(float(v))
        agg[name] = {k: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4), len(v))
                     for k, v in per_key.items() if v}
        agg[name]["n_seeds"] = n_present
    out = dict(seeds=sorted(seeds), note="each entry is [mean, sd, n_seeds]", per_dataset=agg)
    (OUT / "aggregate.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


def main():
    global SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    help="K562 | RPE1 | Norman | all (default). Each writes its own JSON.")
    ap.add_argument("--smoke", action="store_true", help="K562 only, reduced counts")
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed; output goes to results/ilcs_baseline/seed<SEED>/")
    ap.add_argument("--aggregate", action="store_true",
                    help="read all seed*/ summaries and write aggregate.json (mean/SD across seeds)")
    args = ap.parse_args()

    if args.aggregate:
        aggregate_seeds()
        return

    SEED = args.seed
    mode = "smoke" if args.smoke else "full"
    outdir = seed_out(SEED)

    if args.smoke:
        todo = [DATASETS[0]]
    elif args.dataset == "all":
        todo = DATASETS
    else:
        todo = [d for d in DATASETS if d[0] == args.dataset]
        if not todo:
            sys.exit(f"unknown --dataset {args.dataset}; choose from {[d[0] for d in DATASETS]}")

    outdir.mkdir(parents=True, exist_ok=True)
    for (name, path, ctrl, single) in todo:
        err = outdir / f"{name}.error.txt"
        if err.exists():
            err.unlink()
        try:
            run_dataset(name, path, ctrl, single, args.smoke, outdir)
        except Exception:
            tb = traceback.format_exc()
            err.write_text(tb)
            _log(name, "FAILED, traceback written to " + str(err))
            print(tb, flush=True)
    per, errors = write_summary(mode, outdir, SEED)
    _log("done", f"seed={SEED} complete={sorted(per.keys())} failed={sorted(errors.keys())}")


if __name__ == "__main__":
    main()
