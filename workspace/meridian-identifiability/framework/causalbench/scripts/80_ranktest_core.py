"""Rank diagnostic for the multi-environment CRL assumption bundle.

WHAT IS BEING TESTED
--------------------
For each environment e, against the observational/control environment 0:

    H0(k):  rank( Sigma_e - Sigma_0 ) <= k,   with k = 2

Under the assumption bundle behind multi-environment CRL identifiability
theorems -- linear-Gaussian latent SCM, linear mixing, a SINGLE-NODE
intervention, and a mixing map shared across environments -- the observed
covariance difference is a rank-<=2 update, and rank 1 when the intervened
node is a source.

Sketch, latent space. Sigma = M D M^T with M = (I - B)^-1.
A hard intervention on node i zeroes row i of B, so by Sherman-Morrison
M_e = M + u v^T with u proportional to M e_i -- a rank-1 update to M -- and
D_e = D + delta e_i e_i^T. Expanding M_e D_e M_e^T - M D M^T leaves a column
space contained in span{M D v, u}, hence rank <= 2. For a source node v = 0
and only the delta term survives, hence rank 1. Linear mixing X = A Z maps
this to A (Sigma_Z,e - Sigma_Z,0) A^T, which can only lose rank.

THE DECISION -- reject_rank2_cf, LFC-calibrated (method="lfc", default)
------------------------------------------------------------------------
The decision fixes r_hat === r, the null rank being tested, and calibrates
the Chen-Fang bootstrap there. See _lfc_rank_test for the argument and the
citation. NO TUNING CONSTANT of any kind enters the decision path: no kappa,
no beta, nothing but (r, B, alpha).

Two rank ESTIMATORS are kept on disk, unused by default and selectable via
method= for the paper's comparison table only:
    method="cfa"  kappa-tuned, Chen-Fang eq (9)
    method="cft"  tuning-free sequential, Chen-Fang eq (C.1), with beta
Neither is the decision. Both are retired for the reasons below.

THREE STATISTICS HAVE ALREADY BEEN RETIRED HERE, all caught by the gates.

1. A count of exceedances over all d eigenvalues, rejecting when the count
   exceeded 2. That was d marginal level-alpha tests with no multiplicity
   control, so its null distribution was approximately Binomial(d, alpha)
   rather than a point mass at 0 and it fired on pure control data at up to
   0.09 at d=20. Gate 0 caught it.

2. The single comparison lam[2] > band[2] against a band resampled from
   control-vs-control splits. One test, so Gate 0 passed. But that band
   calibrates the rank-2 null at its EMPTIEST point, zero signal, while
   H0(2) is COMPOSITE and contains rank-2 signals of any magnitude. Gate 1
   caught it: with the population rank held at 1, the rejection rate ran
   0.083 / 0.106 / 0.156 / 0.217 / 0.300 / 0.322 as the intervened node's
   noise variance was scaled by 1.00 to 3.00. The test was reading
   intervention STRENGTH, not intervention RANK.

3. Rank ESTIMATION itself, in both its kappa (CF-A) and tuning-free
   sequential (CF-T) forms. Estimating r_hat picks ONE member of the
   composite null and calibrates there, so the level holds only if r_hat is
   right. CF-A over-selected (capped at 2 by construction) and was safe by
   accident; CF-T under-selected badly, returning r_hat=0 in 108 of 160 k=1
   soft runs where the truth was rank 1, and Gate 1a regressed from 0.013 to
   0.225. Rejection is monotone decreasing in r_hat (0.298 / 0.159 / 0.093),
   so the fix is not a better estimator but no estimator: pin r_hat to r, the
   least favourable configuration.

The lesson all three times: passing a gate at one point of a composite null
says nothing about the rest of it.

r_hat_stepdown is a descriptive rank readout, not the decision, and it still
rides on the retired zero-signal band, so it inherits that magnitude
sensitivity. Do not quote it as a rank estimate.

INTERPRETATION RULE -- this is the whole point of the diagnostic
----------------------------------------------------------------
* reject_rank2 True is INFORMATIVE. It falsifies the bundle: at least one of
  linear-Gaussian latents, linear mixing, single-node intervention, or a
  shared mixing map fails on this data.
* reject_rank2 False is NOT evidence that the bundle holds. Covariance is
  mean-centred, so a pure shift intervention moves the mean and leaves the
  covariance alone, producing no rejection trivially. Only REJECTION carries
  a claim.
* This is the SECOND-ORDER COMPLEMENT to the mean-shift screen in
  03_screen.py, not a replacement for it. The shift blind spot above is
  exactly what the mean-shift screen sees and this does not. Neither
  subsumes the other; they are run together.

DEVIATIONS FROM THE HANDOFF SPEC -- read before using any number from this
--------------------------------------------------------------------------
(1) fit_pca / project SIGNATURES. The handoff assumed
        P = fit_pca(X_basis, d)   -> (D, d)      and    Y = X @ P
    The functions in 03_screen.py are actually
        fit_pca(Xc, d)            -> (mu, W),  W is (D, d)
        project(X, mu, W, drop)   -> (X - mu) @ W_masked
    The NAMES match the handoff, so per the handoff's own stop-rule this is
    not a stop condition; only the call sites are adapted. PCA is imported,
    never reimplemented. The mu subtraction is irrelevant to this diagnostic
    -- np.cov centres each sample itself, so a constant offset cancels from
    Delta -- but it is kept so the arithmetic stays byte-identical to the
    screen.

(2) n_match. Resolved by amendment to
        n_match = min(n_e, n_p // 3)
    which supersedes the handoff's contradictory pair of
    n_match = min(n_e, floor(n_p/2)) and "raise if n_p < 3 * n_match" (those
    two could never both hold: whenever n_e was large the first gave
    floor(n_p/2), and n_p >= 3*floor(n_p/2) is false for every n_p >= 2).
    The n_match ARGUMENT is consequently inert -- it is not part of the min.
    It stays in the signature for compatibility and is echoed back as
    n_match_requested so a caller can see the request was not honoured.

(3) drop=. Exposed as a keyword-only argument, default empty, so Phase B can
    apply the screen's targeted-gene masking consistently to the environment,
    the reference AND the null draws. The handoff's positional signature is
    unchanged. With drop empty this is a no-op.

(4) null_band=. Additive, keyword-only, arithmetic-identical optimisation.
    The band depends only on (X_ref_pool, mu, W, n_match, alpha, B_null) and
    NOT on X_env, so a caller sweeping many environments against one fixed
    control pool may compute the band once and pass it back in. Gate 0 needs
    this: 200 splits x 3 d x 10 seeds x 2 scalings would otherwise recompute
    an identical band 12,000 times. Callers that reuse a band MUST report
    B_null, because the 200 decisions then share that band's Monte-Carlo
    error and are no longer independent.

Pure functions. No I/O, no CLI, no global seeding: every draw takes an
explicit rng.
"""
import importlib.util
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants
# See deviation (2) in the module docstring. The null needs two DISJOINT
# subsets of size n_match drawn from X_ref_pool; 3 leaves margin so that
# successive null draws are not forced to recycle the same cells.
REF_SPLIT_FACTOR = 3

_SCREEN_PY = Path(__file__).resolve().parent / "03_screen.py"


def _load_screen_module(path=_SCREEN_PY):
    """Import 03_screen.py for fit_pca/project. PCA is NEVER reimplemented.

    03_screen.py does os.makedirs("/workspace/.../results/screen") at module
    scope, which raises OSError on any machine that is not the A100. Only
    that side effect is neutralised, by swapping os.makedirs for a no-op for
    the duration of exec_module. No function body is touched, so fit_pca and
    project are byte-identical to the screen's.
    """
    spec = importlib.util.spec_from_file_location("_ranktest_screen03", str(path))
    mod = importlib.util.module_from_spec(spec)
    real_makedirs = os.makedirs
    os.makedirs = lambda *a, **k: None
    try:
        spec.loader.exec_module(mod)
    finally:
        os.makedirs = real_makedirs
    return mod


_SCREEN = _load_screen_module()
fit_pca = _SCREEN.fit_pca        # fit_pca(Xc, d) -> (mu, W),  W is (D, d)
project = _SCREEN.project        # project(X, mu, W, drop) -> (X - mu) @ W_masked


# ------------------------------------------------------------------ helpers
def _spectrum(Ya, Yb):
    """|eigenvalues| of cov(Ya) - cov(Yb), sorted descending. Ya, Yb are (n, d)."""
    if Ya.shape[0] != Yb.shape[0]:
        raise ValueError(
            f"numerator/denominator sample sizes differ: {Ya.shape[0]} vs "
            f"{Yb.shape[0]}; the covariance difference would be biased by the "
            f"n-dependent estimation noise alone"
        )
    Delta = np.cov(Ya, rowvar=False) - np.cov(Yb, rowvar=False)
    return np.sort(np.abs(np.linalg.eigvalsh(Delta)))[::-1]


def _two_disjoint(n_pool, n_match, rng):
    """Two index sets of size n_match into [0, n_pool), guaranteed disjoint."""
    take = rng.choice(n_pool, size=2 * n_match, replace=False)
    return take[:n_match], take[n_match:]


def _lfc_rank_test(Y_e, Y_0, r, B, alpha, rng):
    """H0: rank(Delta) <= r, calibrated at the LEAST FAVOURABLE CONFIGURATION.

    NO RANK ESTIMATION. r_hat is FIXED at r, the null rank being tested. There
    is no kappa, no beta, and no other tuning constant anywhere in this
    function: the only inputs are (r, B, alpha), all of which the caller
    states as part of the hypothesis and the resampling budget.

    WHY r_hat === r IS THE RIGHT FIX. H0: rank <= r is COMPOSITE, containing
    every true rank 0, 1, ..., r. Estimating r_hat picks one member of that
    null and calibrates there, so the level is only controlled if r_hat is
    right; both previous attempts got it wrong in opposite directions, and the
    kappa version was safe only by accident. Measured here, the rejection rate
    is MONOTONE DECREASING in the r_hat used for the projection:

        r_hat = 0 -> 0.298      r_hat = 1 -> 0.159      r_hat = 2 -> 0.093

    so the largest r_hat allowed under H0, namely r itself, is the LEAST
    FAVOURABLE CONFIGURATION: it yields the largest critical value and hence
    the smallest rejection probability. Calibrating there controls the level
    across the WHOLE composite null rather than at one estimated point of it,
    at the cost of conservatism when the true rank is strictly below r. This
    is the standard least-favourable-configuration construction for composite
    nulls (Lehmann & Romano, Testing Statistical Hypotheses, 3rd ed., ch. 3;
    the same device underlies least-favourable-distribution constructions in
    moment-inequality testing, e.g. Andrews & Soares 2010).

    Concretely this is Chen-Fang equation (11) with r_hat pinned to r, so
    P2, Q2 are the last (d - r) singular directions and the bootstrap
    statistic sums ALL (d - r) squared singular values of the projected
    fluctuation. The bootstrap for M is unchanged: nonparametric paired
    resampling of the cells that produced Delta, recentred.

    Returns (reject, stat, crit, r_used, None, trace) to match the CF-T
    signature; r_used is always r and the trace is empty because nothing was
    estimated.
    """
    n, d = Y_e.shape
    tau = np.sqrt(n)
    if r >= d:
        raise ValueError(f"need r < d for a non-trivial null, got r={r}, d={d}")

    Delta = np.cov(Y_e, rowvar=False) - np.cov(Y_0, rowvar=False)
    P, s, Qt = np.linalg.svd(Delta)
    Q = Qt.T

    # LFC: the null space is taken to be (d - r)-dimensional, no estimation.
    P2, Q2 = P[:, r:], Q[:, r:]

    boot = np.empty(B)
    for b in range(B):
        ie = rng.integers(0, n, n)
        i0 = rng.integers(0, n, n)
        Dstar = np.cov(Y_e[ie], rowvar=False) - np.cov(Y_0[i0], rowvar=False)
        M = tau * (Dstar - Delta)          # recentred fluctuation
        sv = np.linalg.svd(P2.T @ M @ Q2, compute_uv=False)
        boot[b] = float(np.sum(sv ** 2))   # ALL of them: lo = r - r_hat = 0
    crit = float(np.quantile(boot, 1.0 - alpha))

    stat = float(tau ** 2 * np.sum(s[r:] ** 2))
    return bool(stat > crit), stat, crit, int(r), None, []


def chen_fang_rank_test(Y_e, Y_0, r, B, alpha, rng, beta=None, method="lfc"):
    """Chen-Fang composite-null-calibrated test of H0: rank(Delta) <= r.

    Chen, Q. and Z. Fang (2019), "Improved inference on the rank of a matrix",
    Quantitative Economics 10(4), 1787-1824 (arXiv:1812.02337). This follows
    their implementation guide, Steps 1-5, with equations (9), (10) and (11).
    Their formulation is exactly H0: rank(Pi_0) <= r rather than
    rank(Pi_0) = r, which is the composite null we need: it does not assume
    away rank < r, and it is the assumption that over-rejects when violated.

    WHY THIS FIXES OUR FAILURE. Our own rule compared lam[2] against a band
    resampled from control-vs-control splits, i.e. it calibrated the rank-2
    null at its EMPTIEST point, zero signal. Chen-Fang instead projects the
    recentred bootstrap fluctuation M* onto the ESTIMATED null space
    (P2_hat, Q2_hat) before reading singular values, so the critical value
    depends on the data only through the null-space DIRECTIONS and not
    through the magnitude of the signal singular values. That is precisely
    the dependence that made our rejection rate climb from 0.083 to 0.322 as
    the intervention strengthened while the true rank stayed 1.

    THIS IS CF-T, THE TWO-STEP TUNING-FREE VARIANT, not the kappa-tuned CF-A.
    The paper introduces CF-T precisely because it is "unable to provide an
    optimal choice of kappa_n", and Appendix C formalises a sequential rank
    estimator that needs no tuning parameter at all:

      STEP 1  r_hat = min{ rr = 0..d-1 : tau^2 phi_rr(Delta) <= c_{1-beta}(rr) }
              and d if that set is empty.                            (eq C.1)
              At hypothesised rank rr the null space is (d-rr)-dimensional,
              so c_{1-beta}(rr) projects M* onto the last (d-rr) columns and
              sums ALL squared singular values of that block.
      STEP 2  If r_hat > r, reject H0 outright.
              Otherwise plug r_hat into (10)/(11) and reject if
              tau^2 phi_r(Delta) > c_{1-alpha+beta}, the adjusted quantile
              that buys back the beta spent selecting r_hat, leaving the
              overall level at alpha.

    WHY THE SWAP. The kappa-tuned version (eq 9) under-estimated r_hat at the
    rank-2 boundary in Gate 1: k=1 hard runs with r_hat=1 rejected at 0.182
    against 0.094 where r_hat=2 was correct, driving a 0.125 boundary
    over-rejection against a nominal 0.05.

    WHY THIS FIXES THE MAGNITUDE PROBLEM (unchanged from CF-A). The bootstrap
    projects the recentred fluctuation M* onto the ESTIMATED null space before
    reading singular values, so the critical value depends on the data through
    null-space DIRECTIONS and not through the magnitude of the signal singular
    values. That is the dependence which made our own retired rule climb from
    0.083 to 0.322 as the intervention strengthened while true rank stayed 1.

    IMPLEMENTATION NOTES:

    * beta. Required by CF-T, which asks for some beta < alpha. The paper
      does not prescribe a value; beta = alpha/10 is used here. THIS IS A
      CHOSEN CONSTANT, not one taken from the paper. It is a significance
      level rather than a bandwidth, so unlike kappa it does not need to be
      tied to the scale of Delta, but it is still a choice and any result
      should be checked for sensitivity to it.

    * Sequential test. The paper notes the KP test "may be utilized and is
      recommended as it is tuning parameter free and does not require
      additional simulations". KP would need a consistent estimate of the
      asymptotic variance of vec(Delta), i.e. fourth moments of the cells,
      which is a substantial new estimator to validate. Appendix C's own
      bootstrap-based sequential procedure is equally tuning-free and reuses
      machinery already validated by the acceptance sweep, so eq (C.1) is
      what is implemented. This is a deviation from "KP-based".

    * Bootstrap for M. Nonparametric paired bootstrap over the cells that
      produced Delta: resample n rows with replacement from Y_e and n from
      Y_0 independently, recompute Delta*, and recentre. Delta is a smooth
      function of two sample covariances of iid rows, so this is consistent
      for M. ONE sample of M* is drawn and reused across the sequential steps
      and the final test.

    The statistic is their phi_r, the SUM of the (d - r) smallest squared
    singular values, not lam[2] alone.

    Returns (reject, stat, crit, r_hat, beta, r_hat_trace).
    crit is nan when step 2 rejected outright on r_hat > r.
    """
    n, d = Y_e.shape
    tau = np.sqrt(n)
    if method not in ("lfc", "cft", "cfa"):
        raise ValueError(f"unknown method {method!r}; use lfc, cft or cfa")
    if method == "lfc":
        return _lfc_rank_test(Y_e, Y_0, r, B, alpha, rng)
    beta = alpha / 10.0 if beta is None else float(beta)
    if not (0.0 < beta < alpha):
        raise ValueError(f"need 0 < beta < alpha, got beta={beta}, alpha={alpha}")

    # Step 1: SVD of the observed covariance difference.
    Delta = np.cov(Y_e, rowvar=False) - np.cov(Y_0, rowvar=False)
    P, s, Qt = np.linalg.svd(Delta)
    Q = Qt.T

    def phi(rr):
        """tau^2 * sum_{j=rr+1}^{d} sigma_j^2(Delta), their phi_r."""
        return float(tau ** 2 * np.sum(s[rr:] ** 2))

    # ONE bootstrap sample of the recentred fluctuation M*, reused for the
    # sequential rank steps AND the final test. The paper draws M* once and
    # reads different functionals off it; reusing it here is both faithful
    # and much cheaper, since the two covariance recomputations dominate.
    Ms = []
    for _ in range(B):
        ie = rng.integers(0, n, n)
        i0 = rng.integers(0, n, n)
        Dstar = np.cov(Y_e[ie], rowvar=False) - np.cov(Y_0[i0], rowvar=False)
        Ms.append(tau * (Dstar - Delta))

    def crit_at(rr, level, lo):
        """(1-level) quantile of sum_{j=lo+1}^{d-rr} sigma_j^2(P2' M* Q2)."""
        P2, Q2 = P[:, rr:], Q[:, rr:]
        boot = np.empty(B)
        for b, M in enumerate(Ms):
            sv = np.linalg.svd(P2.T @ M @ Q2, compute_uv=False)
            boot[b] = float(np.sum(sv[lo:] ** 2))
        return float(np.quantile(boot, 1.0 - level))

    if method == "cfa":
        # RETIRED, kept selectable for the paper's comparison table only.
        # kappa-tuned rank estimator, eq (9), with the scale-equivariant
        # constant. Capped at r by construction, so it can only ever
        # under- or correctly-select, never exceed r.
        kappa = float(s[0]) * n ** -0.25
        r_hat = 0
        for j in range(min(r, d)):
            if s[j] >= kappa:
                r_hat = j + 1
            else:
                break
        crit = crit_at(r_hat, alpha, r - r_hat)
        return bool(phi(r) > crit), phi(r), crit, int(r_hat), float(kappa), []

    # ---- CF-T STEP 1: tuning-free sequential rank estimator, eq (C.1).
    #   r_hat = min{ rr = 0..d-1 : tau^2 phi_rr(Delta) <= c_{1-beta}(rr) },
    #   and d if the set is empty.
    # At the hypothesised rank rr the null space is (d-rr)-dimensional, so the
    # projection uses the last (d-rr) columns and ALL squared singular values
    # of that block (lo=0). No kappa anywhere. The loop breaks at the first
    # non-rejection, so it normally costs 1-3 extra small SVD passes.
    r_hat, r_hat_crits = d, []
    for rr in range(d):
        c = crit_at(rr, beta, 0)
        r_hat_crits.append((rr, phi(rr), c))
        if phi(rr) <= c:
            r_hat = rr
            break

    # ---- CF-T STEP 2.
    if r_hat > r:
        # The estimated rank already exceeds the hypothesised one, so H0 is
        # rejected outright without consulting the second-stage critical
        # value. This branch does not exist in the kappa-tuned CF-A test.
        return True, phi(r), float("nan"), int(r_hat), float(beta), r_hat_crits

    # Plug r_hat into (10)/(11) and test at the ADJUSTED quantile 1-alpha+beta,
    # which buys back the beta spent selecting r_hat so the overall level is
    # alpha.
    crit = crit_at(r_hat, alpha - beta, r - r_hat)
    stat = phi(r)
    return bool(stat > crit), stat, crit, int(r_hat), float(beta), r_hat_crits


def null_band_from_pool(Yp, n_match, B_null, alpha, rng):
    """(1 - alpha) quantile of each sorted null eigenvalue.

    Yp is the ALREADY-PROJECTED control reference pool, (n_p, d). Projection
    is linear, so projecting a subset of rows equals subsetting the projected
    matrix -- this is an exact restatement, not an approximation.

    Each of the B_null draws splits the pool into two disjoint halves of size
    n_match and takes the spectrum of their covariance difference. That is
    the same statistic as the real comparison, computed where the true rank
    is 0 by construction.
    """
    n_p, d = Yp.shape
    lam_null = np.empty((B_null, d))
    for b in range(B_null):
        ia, ib = _two_disjoint(n_p, n_match, rng)
        # LEAK GATE, per null draw: the two halves must not share a cell.
        assert not (set(ia.tolist()) & set(ib.tolist())), (
            f"null draw {b}: the two reference subsets share cells"
        )
        lam_null[b] = _spectrum(Yp[ia], Yp[ib])
    return np.quantile(lam_null, 1.0 - alpha, axis=0)


# ------------------------------------------------------------- entry point
def rank_diagnostic(X_env, X_basis, X_ref_pool, d, n_match, B_null, alpha, rng,
                    *, drop=(), basis_idx=None, ref_pool_idx=None,
                    null_band=None, method="lfc"):
    """Estimate rank(Sigma_e - Sigma_0) against a resampled null band.

    X_env      : (n_e, D) environment cells
    X_basis    : (n_a, D) control cells used ONLY to fit the projection
    X_ref_pool : (n_p, D) control cells used for the reference covariance and
                 the null. MUST be row-disjoint from X_basis.
    d          : projection dimension
    n_match    : REQUESTED matched sample size; clamped down to
                 min(n_match, n_e, n_p // REF_SPLIT_FACTOR)
    B_null     : null resampling draws (ignored when null_band is supplied)
    alpha      : per-eigenvalue upper-tail level
    rng        : np.random.Generator. Every draw uses it; nothing is seeded
                 globally anywhere in this module.

    Keyword-only, see deviations (3) and (4) in the module docstring:
    drop, basis_idx, ref_pool_idx, null_band.

    The projection is fitted on CONTROL CELLS ONLY and applied unchanged to
    the environment, the reference and every null draw. Projection can only
    reduce rank, never increase it, so a rejection in the projected space is
    a valid rejection in the full space. The converse does not hold: failing
    to reject in d dimensions says nothing about the other D - d.

    Returns a dict; see the keys assembled at the end of this function.
    """
    X_env = np.asarray(X_env)
    X_basis = np.asarray(X_basis)
    X_ref_pool = np.asarray(X_ref_pool)

    n_e = X_env.shape[0]
    n_p = X_ref_pool.shape[0]

    # ---- LEAK GATE: basis vs reference pool, on INDEX SETS, not on values.
    if basis_idx is not None and ref_pool_idx is not None:
        shared = set(np.asarray(basis_idx).tolist()) & set(np.asarray(ref_pool_idx).tolist())
        if shared:
            raise ValueError(
                f"leak: X_basis and X_ref_pool share {len(shared)} row indices "
                f"(e.g. {sorted(shared)[:5]}); the projection would be fitted "
                f"on the same cells that define the null"
            )

    # ---- sample-size matching. See deviation (2).
    # n_match = min(n_e, n_p // 3). The n_match ARGUMENT is not part of this
    # min and is therefore inert; it is kept in the signature for
    # compatibility and echoed back as n_match_requested so a caller can see
    # that its request was not honoured.
    n_match_eff = int(min(n_e, n_p // REF_SPLIT_FACTOR))
    if n_match_eff < 2:
        raise ValueError(
            f"n_match_eff={n_match_eff} < 2 (n_e={n_e}, n_p={n_p}, "
            f"requested n_match={n_match}); not enough cells to form a covariance"
        )
    if d < 3:
        raise ValueError(
            f"d={d} < 3: reject_rank2 tests the THIRD eigenvalue, which does "
            f"not exist below d=3"
        )
    if n_p < REF_SPLIT_FACTOR * n_match_eff:
        raise ValueError(
            f"reference pool too small: n_p={n_p} < {REF_SPLIT_FACTOR} * "
            f"n_match={n_match_eff}; refusing to silently reuse cells across "
            f"the null draws"
        )

    # ---- projection, fitted on the basis controls alone.
    mu, W = fit_pca(X_basis, d)

    # Project the pools ONCE. project() is linear in its rows, so subsetting
    # after projecting is identical to projecting a subset.
    Yp_all = project(X_ref_pool, mu, W, drop)
    Ye_all = project(X_env, mu, W, drop)

    # ---- matched draws for the observed comparison.
    env_take = rng.choice(n_e, size=n_match_eff, replace=False)
    ref_take = rng.choice(n_p, size=n_match_eff, replace=False)
    Y_e = Ye_all[env_take]
    Y_0 = Yp_all[ref_take]

    # Numerator and denominator must carry identical n.
    if Y_e.shape[0] != Y_0.shape[0]:
        raise AssertionError(
            f"matched-n violation: environment n={Y_e.shape[0]} but reference "
            f"n={Y_0.shape[0]}"
        )
    assert Y_e.shape[0] == n_match_eff and Y_0.shape[0] == n_match_eff

    lam = _spectrum(Y_e, Y_0)

    # ---- null band.
    band_reused = null_band is not None
    if band_reused:
        band = np.asarray(null_band, dtype=float)
        if band.shape != (d,):
            raise ValueError(f"null_band has shape {band.shape}, expected ({d},)")
    else:
        band = null_band_from_pool(Yp_all, n_match_eff, B_null, alpha, rng)

    # ---- THE DECISION: LFC-calibrated rank test. No rank estimation, and no
    # tuning constant of any kind in this path -- see _lfc_rank_test.
    reject_cf, cf_stat, cf_crit, cf_rhat, cf_aux, cf_trace = chen_fang_rank_test(
        Y_e, Y_0, 2, B_null, alpha, rng, method=method)

    exceed = lam > band

    # Rank estimate by step-down: walk the sorted spectrum from the top and
    # stop at the FIRST eigenvalue that fails to clear its band. Descriptive
    # only, and it inherits the zero-signal band's magnitude sensitivity.
    r_hat_stepdown = 0
    for j in range(d):
        if not exceed[j]:
            break
        r_hat_stepdown += 1

    return dict(
        reject_rank2_cf=reject_cf,
        cf_stat=cf_stat,
        cf_crit=cf_crit,
        cf_r_hat=cf_rhat,
        cf_method=method,
        cf_aux=cf_aux,
        cf_rejected_on_rhat=bool(cf_rhat > 2),
        r_hat_stepdown=int(r_hat_stepdown),
        lam=lam.tolist(),
        band=band.tolist(),
        exceed=exceed.tolist(),
        d=int(d),
        n_match=int(n_match_eff),
        n_match_requested=int(n_match),
        n_env=int(n_e),
        n_ref_pool=int(n_p),
        n_basis=int(X_basis.shape[0]),
        B_null=int(B_null),
        alpha=float(alpha),
        band_reused=bool(band_reused),
        ref_split_factor=int(REF_SPLIT_FACTOR),
        drop=sorted(int(x) for x in drop),
    )


def standardise(X, X_fit):
    """Column z-score of X using the mean/sd of X_fit (the CONTROL cells).

    Varsortability (Reisach et al., NeurIPS 2021): raw simulator output can
    carry the causal order in the marginal variances, and any result read off
    unstandardised data is provisional until it is reproduced here.

    Scaling is fitted on controls only, for the same reason the PCA basis is.
    Because it acts as Delta -> S^-1 Delta S^-1 with S diagonal and positive,
    it preserves rank EXACTLY, so the rank-<=2 theory is untouched by it.
    """
    mu = X_fit.mean(0)
    sd = X_fit.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X - mu) / sd
