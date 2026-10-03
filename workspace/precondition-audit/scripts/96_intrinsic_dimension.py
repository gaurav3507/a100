"""Nonlinear intrinsic dimension of the control-cell matrices, matched to Table II.

WHY. The manuscript treats the observed control-covariance dimension as a lower
bound on latent dimension. That holds under linear noiseless mixing and fails
under nonlinear mixing: a one-dimensional latent embedded as (z, z^2) has a
rank-2 covariance and a manifold dimension of 1. So the covariance number can be
an order of magnitude too large without anything being wrong with it. This
measures the manifold dimension directly, at the same matched sample size, so
the paper can say whether the two agree in order of magnitude.

Precondition 3 is scoped and kept if intrinsic dimension lands near the
participation ratio, and reframed if it lands an order of magnitude lower.
NOTHING HERE IS TUNED TO FAVOUR EITHER OUTCOME, and the design that makes that
claim checkable is:

  * Every estimator that is run is reported, including ones that disagree.
    There is no preferred estimator and no post-hoc selection.
  * Neighbourhood parameters are fixed constants declared at the top of this
    file. MLE is reported across k in {5, 10, 20, 50} precisely because it is
    known to be k-sensitive, and reporting one k would hide that.
  * The correlation-dimension fit window is fixed by percentile of the pairwise
    distance distribution, stated up front, never chosen per dataset.
  * If the estimators disagree by more than a factor of two, that disagreement
    is the finding. It is recorded as `estimators_disagree` and stated in the
    summary rather than resolved by picking one.

STEP-0 SYNTHETIC GATE RUNS FIRST, ALWAYS. Five cases with known answers at the
shape of the real matrices; three are gated and two are calibration only. There is no flag that skips it: main() calls it
before the loaders are even imported, and a failure exits non-zero. Case 2 is
the whole point of the exercise. If the estimator returns the inflated
covariance rank on a nonlinear embedding of 10 latents, it cannot answer the
question this script exists to answer and the run stops there.

Cases 4 and 5 are calibration and are deliberately NOT gated on recovery. Case 3
showed the estimator's empirical ceiling at this (n, p) sits below the top of the
range the real participation ratios occupy, which means a real value near that
ceiling may be censored rather than measured. Cases 4 and 5 put known linear
dimensions inside that range and record how far short the estimate falls, so the
real numbers can be read against a measured bias instead of an assumed one.

NO VERDICT ON THE DATA. This reports dimension estimates. It runs no rank test,
expresses no assumption verdict, and Phase B remains not authorised.

Usage (A100). Long-running batch job; launch detached.

    nohup python -u scripts/96_intrinsic_dimension.py \
        > logs/intrinsic_dimension.log 2>&1 &

    python scripts/96_intrinsic_dimension.py --selftest      # synthetic gate only
    python scripts/96_intrinsic_dimension.py --dataset k562

Env:
    PRECOND_DATA      raw data root; required for the real-data pass.
    PRECOND_EXTERNAL  third-party checkouts (Norman AND Frangieh). Resolution is
                      94_control_pool_audit.py's, imported, never reimplemented here.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT_DIR = REPO / "results" / "intrinsic"
LOCKFILE = REPO / "logs" / "intrinsic_dimension.lock"

AUDIT = HERE / "94_control_pool_audit.py"          # control selection + provenance
SWEEP = HERE / "91_preprocessing_sweep.py"         # apply_arm, the two admissible arms
REQUIRED_SCRIPTS = (AUDIT, SWEEP,
                    HERE / "85_dataset_descriptives.py",
                    HERE / "81_ranktest_oracle.py")

# ---------------------------------------------------------------- parameters
# Fixed before any real data is touched. None of these is chosen per dataset.
N_GRID = (1000, 2000)              # Table II's matched size, and half of it
ARMS = ("raw", "standardise")      # the two admissible arms of Table III.
                                   # rank_int is deliberately NOT run: it is the
                                   # arm Table C note (d) shows concentrates the
                                   # spectrum into one direction on zero-inflated
                                   # input, and it is not a preprocessing choice
                                   # this paper endorses.
MLE_K = (5, 10, 20, 50)            # Levina-Bickel k, reported across all four
TWONN_DISCARD = 0.10               # Facco et al: drop the top 10% of mu ratios
CORRDIM_PCTL = (10.0, 30.0)        # correlation-dimension fit window, as
                                   # percentiles of the pairwise distance
                                   # distribution. Fixed, never per-dataset.
CORRDIM_NPTS = 20                  # log-spaced radii inside that window
BOOTSTRAP_N = 20                   # resamples for the spread; >= 20 as specified
BOOTSTRAP_FRAC = 0.80              # subsample fraction per resample
DISAGREE_FACTOR = 2.0              # max/min above this and the disagreement IS
                                   # the finding
SEED = 0

# ------------------------------------------------------- step-0 synthetic gate
# Known-answer cases at the shape of the real matrices. Expected values here are
# properties of constructions defined in this file, not of any real dataset.
SYN_N = 2000
SYN_P = 1158                       # k562's gene count, the narrowest real matrix
SYN_LATENT = 10
SYN_NOISE_SD = 0.01                # tiny, so the manifold is not blurred away
# Case 1 and 2 must land near SYN_LATENT. The band is wide because intrinsic
# dimension estimators are biased low on curved manifolds and this gate is
# checking fitness for purpose, not calibration.
SYN_REL_LO, SYN_REL_HI = 0.5, 2.5
# Case 2's failure mode is returning the COVARIANCE rank of the polynomial
# feature space instead of the manifold dimension. degree-2 features of d
# latents span d + d(d+1)/2 dimensions; anything at or above half of that is the
# estimator measuring the embedding, not the manifold.
SYN_POLY_DIM = SYN_LATENT + SYN_LATENT * (SYN_LATENT + 1) // 2      # 65
SYN_POLY_FAIL_AT = 0.5
# Case 3 has no ceiling to assert against: at n = 2000 in 1158 dimensions every
# neighbourhood estimator underestimates badly, by construction, because the
# sample can never fill the space. The honest assertion is that it separates
# from the structured cases by a wide margin, and that whatever it returns is
# RECORDED as this estimator's empirical ceiling at this (n, p).
SYN_CEILING_MULTIPLE = 3.0
# Cases 4 and 5 are CALIBRATION, not gates. Case 3 returned an empirical ceiling
# well below the top of the range the real participation ratios occupy, so a
# real value near that ceiling may be censored rather than measured. These two
# put known dimensions inside that range and measure where the censoring starts.
# Their recovery is NOT asserted: the whole point is that recovery is expected to
# fail, and by how much is the number being collected. Only "the estimator
# returned something" is fatal, because a missing calibration point is a missing
# calibration point.
SYN_CALIBRATION_LATENTS = (200, 450)


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


# ================================================================= estimators
def _knn_dists(X, k_max):
    """Distances to the k_max nearest neighbours, self excluded. Exact, no tree.

    Chunked so a 2000 x 1158 float64 matrix never materialises an n x n float64
    block bigger than the chunk. Exactness matters more than speed here: an
    approximate neighbour search would be one more knob nobody could audit.
    """
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    n = X.shape[0]
    k_max = int(min(k_max, n - 1))
    sq = (X * X).sum(1)
    out = np.empty((n, k_max), dtype=np.float64)
    step = max(1, int(2e7 // max(n, 1)))
    for a in range(0, n, step):
        b = min(n, a + step)
        d2 = sq[a:b, None] - 2.0 * (X[a:b] @ X.T) + sq[None, :]
        np.maximum(d2, 0.0, out=d2)
        for i in range(b - a):
            d2[i, a + i] = np.inf                      # exclude self
        part = np.partition(d2, k_max - 1, axis=1)[:, :k_max]
        part.sort(axis=1)
        out[a:b] = np.sqrt(part)
    return out


def twonn(X, discard=TWONN_DISCARD, dists=None):
    """TwoNN, Facco et al. 2017. ML on the ratio of the two nearest distances.

    mu_i = r2_i / r1_i is Pareto(1, d) under a locally uniform density, so
    -log(1 - F(mu)) is linear in log(mu) with slope d. Fitted through the
    origin on the lower (1 - discard) of the sorted ratios, which is the
    standard guard against the heavy right tail.
    """
    D = _knn_dists(X, 2) if dists is None else dists[:, :2]
    r1, r2 = D[:, 0], D[:, 1]
    ok = r1 > 0
    mu = np.sort(r2[ok] / r1[ok])
    n = mu.size
    keep = max(2, int(np.floor(n * (1.0 - discard))))
    mu = mu[:keep]
    mu = mu[mu > 1.0]
    if mu.size < 2:
        return None
    F = np.arange(1, mu.size + 1, dtype=np.float64) / (n + 1)
    x = np.log(mu)
    y = -np.log1p(-F)
    denom = float((x * x).sum())
    if denom <= 0:
        return None
    return float((x * y).sum() / denom)


def mle_levina_bickel(X, ks=MLE_K, dists=None):
    """Levina-Bickel MLE at each k, with the MacKay-Ghahramani correction.

    m_k(i)^-1 = mean_{j<k} log(T_k(i) / T_j(i)). Two aggregations are reported
    and neither is preferred: `avg` is the plain mean of the per-point m_k, and
    `inv_avg` inverts the mean of the inverses, which is the estimator MacKay
    and Ghahramani show is the consistent one. They differ, so both are shown.
    """
    D = _knn_dists(X, max(ks)) if dists is None else dists
    out = {}
    for k in ks:
        if k > D.shape[1] or k < 2:
            out[str(k)] = dict(avg=None, inv_avg=None,
                               reason=f"k={k} exceeds available neighbours")
            continue
        Tk = D[:, k - 1][:, None]
        Tj = D[:, :k - 1]
        ok = np.all(Tj > 0, axis=1) & (Tk[:, 0] > 0)
        if ok.sum() < 2:
            out[str(k)] = dict(avg=None, inv_avg=None,
                               reason="degenerate distances")
            continue
        inv_m = np.log(Tk[ok] / Tj[ok]).mean(axis=1)      # = 1 / m_k(i)
        good = inv_m > 0
        if good.sum() < 2:
            out[str(k)] = dict(avg=None, inv_avg=None,
                               reason="non-positive log ratios")
            continue
        m = 1.0 / inv_m[good]
        out[str(k)] = dict(avg=float(m.mean()),
                           inv_avg=float(1.0 / inv_m[good].mean()),
                           n_points_used=int(good.sum()))
    return out


def correlation_dimension(X, pctl=CORRDIM_PCTL, npts=CORRDIM_NPTS, rng=None):
    """Grassberger-Procaccia. Slope of log C(r) vs log r in a FIXED window.

    The window is the pctl[0]-pctl[1] percentile range of the pairwise distance
    distribution, identical for every dataset and arm. Choosing it per dataset
    is the classic way to make this estimator say whatever one wants, so it is
    not done. The fit's R^2 is reported so a bad linear region is visible.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    sq = (X * X).sum(1)
    lo_hi, counts, total = None, None, 0
    # one pass for the percentiles on a capped random pair sample, one for C(r)
    r = np.random.default_rng(SEED) if rng is None else rng
    m = min(n, 1500)
    idx = r.choice(n, size=m, replace=False) if m < n else np.arange(n)
    Y = X[idx]
    sqy = (Y * Y).sum(1)
    d2 = sqy[:, None] - 2.0 * (Y @ Y.T) + sqy[None, :]
    np.maximum(d2, 0.0, out=d2)
    iu = np.triu_indices(m, 1)
    dd = np.sqrt(d2[iu])
    dd = dd[dd > 0]
    if dd.size < npts:
        return dict(dimension=None, reason="too few positive distances")
    lo_hi = (float(np.percentile(dd, pctl[0])), float(np.percentile(dd, pctl[1])))
    if not (lo_hi[0] > 0 and lo_hi[1] > lo_hi[0]):
        return dict(dimension=None, reason="degenerate distance window")
    radii = np.logspace(np.log10(lo_hi[0]), np.log10(lo_hi[1]), npts)
    counts = np.zeros(npts, dtype=np.float64)
    step = max(1, int(2e7 // max(n, 1)))
    for a in range(0, n, step):
        b = min(n, a + step)
        blk = sq[a:b, None] - 2.0 * (X[a:b] @ X.T) + sq[None, :]
        np.maximum(blk, 0.0, out=blk)
        blk = np.sqrt(blk)
        for i in range(b - a):
            blk[i, a + i] = np.inf
        counts += (blk[:, :, None] < radii[None, None, :]).sum(axis=(0, 1))
        total += (b - a) * (n - 1)
    C = counts / max(total, 1)
    ok = C > 0
    if ok.sum() < 3:
        return dict(dimension=None, reason="C(r) zero across the window")
    x, y = np.log(radii[ok]), np.log(C[ok])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(dimension=float(coef[0]),
                r2=(None if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)),
                window=list(lo_hi), n_radii=int(ok.sum()),
                window_spec=(f"percentiles {pctl[0]}-{pctl[1]} of the pairwise "
                             f"distance distribution, fixed for every dataset"))


def all_estimators(X, rng=None):
    """Every estimator, always. Nothing is selected after seeing the answers."""
    D = _knn_dists(X, max(MLE_K))
    est = dict(
        twonn=twonn(X, dists=D),
        mle=mle_levina_bickel(X, dists=D),
        correlation_dimension=correlation_dimension(X, rng=rng),
    )
    est["point_estimates"] = point_estimates(est)
    vals = [v for v in est["point_estimates"].values() if v is not None and v > 0]
    if len(vals) >= 2:
        spread = max(vals) / min(vals)
        est["estimator_spread_max_over_min"] = float(spread)
        est["estimators_disagree"] = bool(spread > DISAGREE_FACTOR)
    else:
        est["estimator_spread_max_over_min"] = None
        est["estimators_disagree"] = None
    return est


def point_estimates(est):
    """Flatten to one number per estimator variant. Every variant appears."""
    out = {"twonn": est.get("twonn")}
    for k, rec in (est.get("mle") or {}).items():
        out[f"mle_k{k}_avg"] = rec.get("avg")
        out[f"mle_k{k}_inv_avg"] = rec.get("inv_avg")
    cd = est.get("correlation_dimension") or {}
    out["correlation_dimension"] = cd.get("dimension")
    return out


def bootstrap_spread(X, rng, n_boot=BOOTSTRAP_N, frac=BOOTSTRAP_FRAC):
    """Subsample spread per estimator variant. Reported as a range, not an SE.

    A percentile range says what the estimate did across resamples without
    asserting a sampling distribution these estimators do not have.
    """
    n = X.shape[0]
    m = max(20, int(round(frac * n)))
    acc = {}
    for _ in range(n_boot):
        idx = rng.choice(n, size=m, replace=False)
        pe = point_estimates(all_estimators(X[idx], rng=rng))
        for k, v in pe.items():
            acc.setdefault(k, []).append(v)
    out = {}
    for k, vals in acc.items():
        good = [v for v in vals if v is not None]
        if not good:
            out[k] = dict(n=0, median=None, p05=None, p95=None, min=None, max=None)
            continue
        a = np.asarray(good, dtype=np.float64)
        out[k] = dict(n=int(a.size), median=float(np.median(a)),
                      p05=float(np.percentile(a, 5)),
                      p95=float(np.percentile(a, 95)),
                      min=float(a.min()), max=float(a.max()))
    return dict(n_resamples=int(n_boot), subsample_size=int(m),
                subsample_fraction=float(frac), per_estimator=out)


# ========================================================== step-0 synthetic
def _syn_cases(rng):
    """Three known-answer constructions at the shape of the real matrices."""
    Z = rng.standard_normal((SYN_N, SYN_LATENT))

    A1 = rng.standard_normal((SYN_LATENT, SYN_P))
    X1 = Z @ A1 + SYN_NOISE_SD * rng.standard_normal((SYN_N, SYN_P))

    # degree-2 polynomial features: the linear terms and every product z_i z_j
    cols = [Z]
    for i in range(SYN_LATENT):
        cols.append(Z[:, i:i + 1] * Z[:, i:])
    P = np.hstack(cols)
    P = (P - P.mean(0)) / np.where(P.std(0) > 0, P.std(0), 1.0)
    A2 = rng.standard_normal((P.shape[1], SYN_P))
    X2 = P @ A2 + SYN_NOISE_SD * rng.standard_normal((SYN_N, SYN_P))

    X3 = rng.standard_normal((SYN_N, SYN_P))

    cases = [
        dict(case="1_linear_gaussian", expected=float(SYN_LATENT), X=X1,
             gated=True,
             description=(f"{SYN_LATENT} latents, random linear map into "
                          f"{SYN_P} columns")),
        dict(case="2_nonlinear_poly2", expected=float(SYN_LATENT), X=X2,
             gated=True,
             description=(f"the same {SYN_LATENT} latents through a degree-2 "
                          f"polynomial feature map ({P.shape[1]} features) then "
                          f"a random linear embedding into {SYN_P} columns; the "
                          f"covariance rank is {P.shape[1]} and the manifold "
                          f"dimension is {SYN_LATENT}")),
        dict(case="3_isotropic_noise", expected=float(SYN_P), X=X3,
             gated=True,
             description=(f"full-rank isotropic noise in {SYN_P} columns, no "
                          f"latent structure; no neighbourhood estimator can "
                          f"reach {SYN_P} at n={SYN_N}, so what is asserted is "
                          f"separation from cases 1 and 2 and what is recorded "
                          f"is this estimator's empirical ceiling")),
    ]

    # ---- calibration cases, same construction as case 1 at larger latent
    # dimension. Not gated on recovery; the ratio to truth IS the measurement.
    for i, dl in enumerate(SYN_CALIBRATION_LATENTS, start=4):
        Zc = rng.standard_normal((SYN_N, dl))
        Ac = rng.standard_normal((dl, SYN_P))
        Xc = Zc @ Ac + SYN_NOISE_SD * rng.standard_normal((SYN_N, SYN_P))
        cases.append(dict(
            case=f"{i}_linear_gaussian_d{dl}", expected=float(dl), X=Xc,
            gated=False,
            description=(f"{dl} latents, random linear map into {SYN_P} "
                         f"columns, the same construction as case 1. CALIBRATION "
                         f"ONLY: {dl} sits inside the range the real "
                         f"participation ratios occupy, and n={SYN_N} points "
                         f"cannot fill a {dl}-dimensional manifold, so recovery "
                         f"is not expected and is not asserted. The ratio to "
                         f"truth is what this case exists to record.")))
    return cases


def step0_synthetic_gate():
    """ALWAYS RUNS. No flag skips it. Exits non-zero on any failure."""
    print("=" * 78, flush=True)
    print(" STEP-0 SYNTHETIC GATE  (known answers, no real data touched)",
          flush=True)
    print(f"   shape n={SYN_N} x p={SYN_P}, latent={SYN_LATENT}, "
          f"noise sd={SYN_NOISE_SD}", flush=True)
    print("=" * 78, flush=True)
    rng = np.random.default_rng(SEED)
    cases = _syn_cases(rng)
    results, fatal = [], []

    for c in cases:
        tag = "GATED" if c.get("gated") else "CALIBRATION ONLY, not gated"
        print(f"\n--- [{c['case']}] expected ~ {c['expected']:g}  ({tag}) ---",
              flush=True)
        print(f"    {c['description']}", flush=True)
        est = all_estimators(c["X"], rng=rng)
        pe = est["point_estimates"]
        for name, v in pe.items():
            print(f"      {name:<24} {('None' if v is None else f'{v:.3f}')}",
                  flush=True)
        results.append(dict(case=c["case"], expected=c["expected"],
                            gated=bool(c.get("gated")),
                            description=c["description"], estimators=est))

    by_case = {r["case"]: r for r in results}

    def headline(case):
        """TwoNN is the gate's headline because it has no free parameter."""
        return by_case[case]["estimators"]["point_estimates"].get("twonn")

    d1, d2, d3 = (headline("1_linear_gaussian"),
                  headline("2_nonlinear_poly2"),
                  headline("3_isotropic_noise"))

    for r in results:
        if r.get("gated"):
            continue
        v = r["estimators"]["point_estimates"].get("twonn")
        if v is None:
            fatal.append(
                f"{r['case']}: TwoNN returned no estimate. Recovery is not "
                f"asserted for a calibration case, but a calibration point "
                f"that does not exist cannot calibrate anything.")

    for case, d in (("1_linear_gaussian", d1), ("2_nonlinear_poly2", d2)):
        if d is None:
            fatal.append(f"{case}: TwoNN returned no estimate.")
            continue
        lo, hi = SYN_REL_LO * SYN_LATENT, SYN_REL_HI * SYN_LATENT
        if not (lo <= d <= hi):
            fatal.append(
                f"{case}: TwoNN = {d:.3f}, outside the band "
                f"[{lo:g}, {hi:g}] around the known dimension {SYN_LATENT}.")

    if d2 is not None and d2 >= SYN_POLY_FAIL_AT * SYN_POLY_DIM:
        fatal.append(
            f"2_nonlinear_poly2: TwoNN = {d2:.3f}, at or above "
            f"{SYN_POLY_FAIL_AT:g} x the polynomial covariance rank "
            f"{SYN_POLY_DIM}. The estimator is measuring the embedding, not "
            f"the manifold. This is exactly the failure this script exists to "
            f"detect, so it cannot be used to answer the question. STOPPING.")

    if d3 is None:
        fatal.append("3_isotropic_noise: TwoNN returned no estimate.")
    elif d1 is not None and d3 < SYN_CEILING_MULTIPLE * d1:
        fatal.append(
            f"3_isotropic_noise: TwoNN = {d3:.3f}, less than "
            f"{SYN_CEILING_MULTIPLE:g} x the linear case ({d1:.3f}). The "
            f"estimator does not separate full-rank noise from a "
            f"{SYN_LATENT}-dimensional manifold at this (n, p).")

    print("\n" + "=" * 86)
    print(f" {'case':<28}{'expected':>10}{'TwoNN':>10}{'ratio':>9}"
          f"{'role':>16}  verdict")
    print("-" * 86)
    for r in results:
        case = r["case"]
        d = r["estimators"]["point_estimates"].get("twonn")
        exp = r["expected"]
        bad = [f for f in fatal if f.startswith(case)]
        ratio = None if (d is None or not exp) else d / exp
        print(f" {case:<28}{exp:>10.0f}"
              f"{('None' if d is None else f'{d:.2f}'):>10}"
              f"{('-' if ratio is None else f'{ratio:.3f}'):>9}"
              f"{('gate' if r['gated'] else 'calibration'):>16}  "
              f"{'FAIL' if bad else 'PASS'}")
    print("-" * 86)
    print(f" case 3 is recorded as the empirical ceiling at n={SYN_N}, "
          f"p={SYN_P}; it is not expected to reach {SYN_P}.")
    print(f" cases {', '.join(str(i) for i in range(4, 4 + len(SYN_CALIBRATION_LATENTS)))}"
          f" are calibration: recovery is NOT asserted and the ratio column is "
          f"the measurement.")
    print("=" * 86)

    # Per-estimator calibration on the three known answers. This is what makes
    # the real-data table readable: an estimator that returns half the true
    # dimension on case 1 is expected to return half of it on real data too,
    # and a reader can see that here instead of guessing.
    truth = {r["case"]: r["expected"] for r in results}
    order = [r["case"] for r in results]
    calib = {}
    for r in results:
        for name, v in r["estimators"]["point_estimates"].items():
            e = calib.setdefault(name, {"value": {}, "ratio_to_truth": {}})
            e["value"][r["case"]] = v
            t = truth[r["case"]]
            e["ratio_to_truth"][r["case"]] = (None if v is None or not t
                                              else float(v) / float(t))
    for name, e in calib.items():
        lin = e["value"].get("1_linear_gaussian")
        e["bias_vs_known_case1"] = (None if lin is None
                                    else float(lin) / SYN_LATENT)

    def _f(x, w=8, dp=2):
        return f"{'None':>{w}}" if x is None else f"{x:>{w}.{dp}f}"

    hdr = f" {'estimator':<24}"
    for c in order:
        hdr += f"{c.split('_')[0] + ' (' + format(truth[c], '.0f') + ')':>16}"
    print("\n CALIBRATION: value and value/truth for every estimator, "
          "every case")
    print(hdr)
    print("-" * (25 + 16 * len(order)))
    for name in sorted(calib):
        e = calib[name]
        row = f" {name:<24}"
        for c in order:
            row += (_f(e["value"].get(c), 8, 2)
                    + _f(e["ratio_to_truth"].get(c), 8, 3))
        print(row)
    print("-" * (25 + 16 * len(order)))
    print(" each cell is  value  ratio-to-truth. Case 3's truth is the ambient "
          "dimension, which")
    print(" no neighbourhood estimator can reach from "
          f"{SYN_N} points; read its ratio as a ceiling, not a bias.")

    gate = dict(shape=dict(n=SYN_N, p=SYN_P, latent=SYN_LATENT,
                           noise_sd=SYN_NOISE_SD),
                cases=results,
                calibration=calib,
                headline_estimator="twonn",
                band=dict(relative_lo=SYN_REL_LO, relative_hi=SYN_REL_HI),
                polynomial_covariance_rank=SYN_POLY_DIM,
                empirical_ceiling=d3,
                calibration_latents=list(SYN_CALIBRATION_LATENTS),
                calibration_note=(
                    f"Cases 4 and 5 put known linear dimensions "
                    f"{list(SYN_CALIBRATION_LATENTS)} inside the range the real "
                    f"participation ratios occupy. They are NOT gated on "
                    f"recovery: {SYN_N} points cannot fill a manifold of those "
                    f"dimensions, so under-recovery is expected and its size is "
                    f"the measurement. Read a real-data estimate near case 3's "
                    f"empirical ceiling as possibly censored rather than "
                    f"measured."),
                failures=fatal,
                passed=not fatal)
    if fatal:
        print("\n[fatal] STEP-0 SYNTHETIC GATE FAILED:", file=sys.stderr)
        for f in fatal:
            print(f"  - {f}", file=sys.stderr)
        print("        No real data was touched. Refusing to continue.",
              file=sys.stderr)
        sys.exit(1)
    print("\n  [gate] PASS\n", flush=True)
    return gate


# ======================================================== calibration inversion
# Anchors. The inversion is fitted on the two CALIBRATION cases only. Case 1
# (d = 10) is deliberately excluded: there the estimator is nearly unbiased
# (TwoNN ratio ~0.96) and the relation between estimate and truth is not the
# same one that holds where the estimator is censoring. Including it would bend
# the curve through a regime it does not describe.
INVERSION_ANCHOR_CASES = tuple(f"{i}_linear_gaussian_d{d}"
                               for i, d in enumerate(SYN_CALIBRATION_LATENTS,
                                                     start=4))
INVERSION_EXCLUDES = "1_linear_gaussian"


def fit_inversion(gate):
    """Per estimator, a power law d_true = A * est^B through the two anchors.

    Two anchors, two parameters, so the fit is exact through both and carries no
    residual. That is a statement about the construction, not about quality: it
    is an interpolation between d = 200 and d = 450 and nothing more.
    """
    by_case = {c["case"]: c for c in gate["cases"]}
    anchors = [by_case[c] for c in INVERSION_ANCHOR_CASES if c in by_case]
    out = {"anchor_cases": list(INVERSION_ANCHOR_CASES),
           "anchor_truths": [float(c["expected"]) for c in anchors],
           "excluded_case": INVERSION_EXCLUDES,
           "exclusion_reason": (
               "d = 10 is excluded from the fit. There the estimator is nearly "
               "unbiased and the estimate-to-truth relation is not the one that "
               "holds in the censoring regime the anchors sample. The inversion "
               "does NOT extend to it and must not be applied there."),
           "form": "d_true = A * estimate ** B, exact through the two anchors",
           "calibration_p": SYN_P,
           "calibration_n": SYN_N,
           "per_estimator": {}}
    if len(anchors) != 2:
        out["error"] = "need exactly two anchor cases; inversion not fitted"
        return out
    t1, t2 = (float(anchors[0]["expected"]), float(anchors[1]["expected"]))
    for name in anchors[0]["estimators"]["point_estimates"]:
        e1 = anchors[0]["estimators"]["point_estimates"].get(name)
        e2 = anchors[1]["estimators"]["point_estimates"].get(name)
        if e1 is None or e2 is None or e1 <= 0 or e2 <= 0 or e1 == e2:
            out["per_estimator"][name] = dict(
                A=None, B=None,
                reason="anchors missing, non-positive, or not distinct")
            continue
        B = float(np.log(t2 / t1) / np.log(e2 / e1))
        A = float(t1 / (e1 ** B))
        out["per_estimator"][name] = dict(
            A=A, B=B, anchor_estimates=[float(e1), float(e2)],
            anchor_truths=[t1, t2],
            valid_estimate_range=[float(min(e1, e2)), float(max(e1, e2))])
    return out


def invert(inv, name, estimate, p_condition):
    """Implied true dimension for one estimate, with its caveats attached."""
    rec = (inv.get("per_estimator") or {}).get(name) or {}
    A, B = rec.get("A"), rec.get("B")
    if A is None or B is None or estimate is None or estimate <= 0:
        return dict(implied_true_dimension=None,
                    reason=rec.get("reason", "no estimate"))
    lo, hi = rec["valid_estimate_range"]
    tol = 1e-9 * max(abs(lo), abs(hi), 1.0)      # an endpoint is inside
    outside = not (lo - tol <= estimate <= hi + tol)
    below = estimate < lo - tol
    return dict(
        implied_true_dimension=float(A * (estimate ** B)),
        raw_estimate=float(estimate),
        A=A, B=B,
        anchor_estimate_range=[lo, hi],
        outside_anchor_range=bool(outside),
        extrapolation=bool(outside or int(p_condition) != int(SYN_P)),
        extrapolation_reasons=[
            r for r in (
                ("estimate outside the anchored range "
                 f"[{lo:.2f}, {hi:.2f}]") if outside else None,
                ("estimate is BELOW the anchored range, heading towards the "
                 f"d={SYN_LATENT} regime where the estimator is nearly "
                 f"unbiased and this power law does not hold. The implied "
                 f"value there is not meaningful.") if below else None,
                (f"this condition has p={p_condition}, the calibration was "
                 f"fitted at p={SYN_P}") if int(p_condition) != int(SYN_P)
                else None,
            ) if r],
        note=("Fitted on the d=200 and d=450 anchors only; the fit does not "
              "extend to d=10, where the estimator is nearly unbiased."),
    )


# ================================================================= real data
def committed_pr(AUD, dataset, arm, n):
    """The committed participation ratio for this (dataset, arm, n), or None.

    raw          -> results/descriptives, latent_dimension_from_controls[n]
    standardise  -> results/preprocessing, arms_result.standardise
    Never a literal, and never a value from a different arm or a different n:
    an unmatched comparison is worse than no comparison.
    """
    if arm == "raw":
        val, src, cap = AUD.expected_pr(dataset, n)
        if val is None or cap != n:
            return None, None, "no committed raw PR at this n"
        return float(val), src, None
    for p in sorted((REPO / "results" / "preprocessing").glob("*.json")):
        try:
            doc = json.load(open(p))
        except ValueError:
            continue
        if (doc.get("meta") or {}).get("status") != "CURRENT":
            continue
        if doc.get("dataset") != dataset or int(doc.get("cap") or -1) != int(n):
            continue
        rec = (doc.get("arms_result") or {}).get("standardise") or {}
        if rec.get("status") == "ok" and rec.get("participation_ratio") is not None:
            return float(rec["participation_ratio"]), p.name, None
    return None, None, "no committed standardise PR at this n"


def run_condition(AUD, SW, DESC, dataset, args, gate, inv):
    a = argparse.Namespace(hvg=args.hvg, seed=args.seed)
    Xc, rng, X_full, paths = AUD.control_matrix(DESC, dataset, a)
    n_all, p = Xc.shape
    state = rng.bit_generator.state          # same draw as Table II, per config
    prov = dict(
        resolved_input_paths=paths,
        input_path=(paths[0] if paths else None),
        n_cells=(int(X_full.shape[0]) if X_full is not None else None),
        n_genes=int(p),
        n_control_cells=int(n_all),
        control_value_array=AUD.value_fingerprint(Xc, "control matrix"),
    )
    out = []
    for n in N_GRID:
        if n > n_all:
            out.append(dict(dataset=dataset, n_requested=int(n),
                            status="skipped",
                            reason=f"only {n_all} control cells available"))
            continue
        rng.bit_generator.state = state
        idx = rng.choice(n_all, size=int(n), replace=False)
        base = np.asarray(Xc[idx], dtype=np.float64)
        for arm in ARMS:
            X = SW.apply_arm(base, arm)
            print(f"    [{dataset}] n={n} arm={arm} ...", flush=True)
            est = all_estimators(X, rng=rng)
            boot = bootstrap_spread(X, rng)
            pr, pr_src, pr_why = committed_pr(AUD, dataset, arm, n)
            pe = est["point_estimates"]
            ratios = {k: (None if pr in (None, 0) or v is None else float(v) / pr)
                      for k, v in pe.items()}
            implied = {k: invert(inv, k, v, p) for k, v in pe.items()}
            out.append(dict(
                dataset=dataset, n_requested=int(n), n_used=int(n), arm=arm,
                status="ok",
                estimators=est,
                bootstrap=boot,
                participation_ratio=pr,
                participation_ratio_source=pr_src,
                participation_ratio_note=pr_why,
                intrinsic_over_participation=ratios,
                implied_true_dimension=implied,
                implied_over_participation={
                    k: (None if pr in (None, 0)
                        or implied[k]["implied_true_dimension"] is None
                        else implied[k]["implied_true_dimension"] / pr)
                    for k in pe},
                headline=dict(
                    twonn=pe.get("twonn"),
                    twonn_over_pr=ratios.get("twonn"),
                    estimator_spread=est.get("estimator_spread_max_over_min"),
                    estimators_disagree=est.get("estimators_disagree"),
                    twonn_implied=implied.get("twonn", {}).get(
                        "implied_true_dimension"),
                    twonn_implied_is_extrapolation=implied.get(
                        "twonn", {}).get("extrapolation"),
                ),
            ))
            hp = out[-1]["headline"]
            print(f"        twonn={hp['twonn']}  implied={hp['twonn_implied']}"
                  f"{' (extrap)' if hp['twonn_implied_is_extrapolation'] else ''}"
                  f"  PR={pr}  ratio={hp['twonn_over_pr']}  "
                  f"spread={hp['estimator_spread']}", flush=True)
    return prov, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="one condition only (default: all six)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the synthetic gate and stop")
    ap.add_argument("--hvg", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", default=str(OUT_DIR))
    args = ap.parse_args()

    # THE GATE RUNS FIRST, UNCONDITIONALLY. No flag reaches this line without
    # passing through it, and the loaders are not even imported until it has.
    gate = step0_synthetic_gate()
    if args.selftest:
        print("  --selftest: synthetic gate only, no real data touched.")
        return

    for q in REQUIRED_SCRIPTS:
        if not q.exists():
            sys.exit(f"[fatal] {q.name} is not in this repository (looked in "
                     f"{q.parent}). Not vendoring a copy and not "
                     f"reimplementing it.")
    AUD = _load(AUDIT, "_audit94")
    SW = _load(SWEEP, "_sweep91")
    todo = [args.dataset] if args.dataset else list(AUD.DATASETS)
    for d in todo:
        if d not in AUD.DATASETS:
            sys.exit(f"[fatal] unknown dataset {d!r}; expected one of "
                     f"{AUD.DATASETS}")
    resolution = AUD.step0_gate(todo)  # data presence, provenance, loud on miss
    if resolution is None:            # older 94 returned nothing
        resolution = []

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
        if not args.dataset and out.exists():
            print(f"[clean] rm -rf {out}", flush=True)
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

        DESC = _load(HERE / "85_dataset_descriptives.py", "_descriptives")
        inv = fit_inversion(gate)
        common = dict(
            step0_synthetic_gate=gate,
            calibration_inversion=inv,
            resolution_report=resolution,
            config=dict(n_grid=list(N_GRID), arms=list(ARMS),
                        mle_k=list(MLE_K), twonn_discard=TWONN_DISCARD,
                        corrdim_percentiles=list(CORRDIM_PCTL),
                        corrdim_n_radii=CORRDIM_NPTS,
                        bootstrap_n=BOOTSTRAP_N,
                        bootstrap_fraction=BOOTSTRAP_FRAC,
                        disagree_factor=DISAGREE_FACTOR, seed=args.seed,
                        rank_int_excluded=("deliberate: not an admissible "
                                           "preprocessing arm for these data")),
            contains_no_test=True,
            disclaimer=("DESCRIPTIVE ONLY. No rank test was run and no "
                        "assumption verdict is expressed or implied."),
            audit_script_sha256=AUD.sha256(AUDIT),
        )

        records, failed, written = [], [], []
        for ds in todo:
            print(f"\n--- [{ds}] ---", flush=True)
            try:
                prov, recs = run_condition(AUD, SW, DESC, ds, args, gate, inv)
            except Exception as e:                       # noqa: BLE001
                # ONE ARTEFACT PER CONDITION, WRITTEN AS IT COMPLETES. A batch
                # that wrote only at the end lost three finished conditions to a
                # Frangieh path failure and left results/intrinsic/ empty. A
                # later condition failing must never discard an earlier one.
                print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
                failed.append((ds, f"{type(e).__name__}: {e}"))
                continue
            ok_here = sum(1 for r in recs if r.get("status") == "ok")
            payload = dict(common)
            payload.update(
                dataset=ds,
                provenance=prov,
                records=recs,
                n_configurations=ok_here,
                n_configurations_expected=len(N_GRID) * len(ARMS),
                complete=bool(ok_here == len(N_GRID) * len(ARMS)),
                n_configurations_with_estimator_disagreement=sum(
                    1 for r in recs if r.get("status") == "ok"
                    and r["headline"].get("estimators_disagree")),
            )
            ts = DESC.utc_stamp()
            path = out / f"{ts.replace(':', '-')}__{ds}__intrinsic.json"
            path.write_text(json.dumps(payload, indent=2) + "\n")
            written.append(path)
            print(f"    [write] {path}  "
                  f"({ok_here}/{len(N_GRID) * len(ARMS)} configurations)",
                  flush=True)
            records.extend(recs)

        print("\n" + "=" * 118)
        print(f" {'dataset':<20}{'n':>6}{'arm':<14}{'TwoNN':>9}{'p05-p95':>16}"
              f"{'implied':>10}{'ext':>5}{'PR':>10}{'ratio':>8}"
              f"{'impl/PR':>9}{'disagree':>10}")
        print("-" * 118)
        for r in records:
            if r.get("status") != "ok":
                print(f" {r['dataset']:<20}{r['n_requested']:>6}"
                      f"{'-':<14}{r['status']:>9}")
                continue
            b = (r["bootstrap"]["per_estimator"].get("twonn") or {})
            rng_s = ("-" if b.get("p05") is None
                     else f"{b['p05']:.1f}-{b['p95']:.1f}")
            pr = r["participation_ratio"]
            ra = r["headline"]["twonn_over_pr"]
            im = r["headline"]["twonn_implied"]
            ip = r["implied_over_participation"].get("twonn")
            print(f" {r['dataset']:<20}{r['n_requested']:>6}{r['arm']:<14}"
                  f"{r['headline']['twonn']:>9.2f}{rng_s:>16}"
                  f"{('-' if im is None else f'{im:.1f}'):>10}"
                  f"{('yes' if r['headline']['twonn_implied_is_extrapolation'] else 'no'):>5}"
                  f"{('-' if pr is None else f'{pr:.1f}'):>10}"
                  f"{('-' if ra is None else f'{ra:.3f}'):>8}"
                  f"{('-' if ip is None else f'{ip:.3f}'):>9}"
                  f"{str(r['headline']['estimators_disagree']):>10}")
        print("-" * 118)
        got = sum(1 for r in records if r.get("status") == "ok")
        disagree = sum(1 for r in records if r.get("status") == "ok"
                       and r["headline"].get("estimators_disagree"))
        print(f" artefacts written: {len(written)} of {len(todo)} conditions")
        print(f" configurations with estimator spread > "
              f"{DISAGREE_FACTOR:g}x: {disagree} of {got}")
        print(f" 'implied' inverts the d={SYN_CALIBRATION_LATENTS[0]}/"
              f"{SYN_CALIBRATION_LATENTS[1]} calibration; 'ext' marks an "
              f"extrapolation outside the anchors or off p={SYN_P}.")
        print("=" * 118)

        # Assertion LAST, after every artefact is on disk, so a shortfall is
        # reported without discarding what did complete.
        on_disk = sorted(out.glob("*__intrinsic.json"))
        want = len(todo) * len(N_GRID) * len(ARMS)
        if failed or got != want or len(on_disk) != len(todo):
            print(f"\n[fatal] SHORTFALL: {got} configuration(s) for {want} "
                  f"requested ({len(todo)} conditions x {len(N_GRID)} sample "
                  f"sizes x {len(ARMS)} arms); {len(on_disk)} artefact(s) on "
                  f"disk for {len(todo)} condition(s).", file=sys.stderr)
            for d, why in failed:
                print(f"  - {d}: {why}", file=sys.stderr)
            for q in on_disk:
                print(f"  kept: {q}", file=sys.stderr)
            print("        The artefacts above ARE complete for their own "
                  "condition and are not discarded.", file=sys.stderr)
            print("        This run is INCOMPLETE as a sweep. Rerun only the "
                  "missing conditions with --dataset.", file=sys.stderr)
            sys.exit(1)
    finally:
        if LOCKFILE.exists():
            LOCKFILE.unlink()


if __name__ == "__main__":
    main()
