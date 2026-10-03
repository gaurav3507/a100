"""P4 rank readout for the GATE module.

The Chen-Fang LFC (least-favourable-configuration) bootstrap rank test, ported
fresh from precondition-audit/scripts/80_ranktest_core.py (_lfc_rank_test).

Reference: Chen, Q. and Z. Fang (2019), "Improved inference on the rank of a
matrix", Quantitative Economics 10(4), 1787-1824 (arXiv:1812.02337), equation
(11) with r_hat pinned to r (the least-favourable configuration for the
composite null H0: rank <= r).

What this tests
  H0(r): rank(Delta) <= r, where Delta = cov(Y_e) - cov(Y_0).
  alpha = 0.05, B = 500 bootstrap draws.
  Statistic: tau^2 * sum of the trailing squared singular values of Delta,
             tau = sqrt(n). The pure function of the covariance difference is
             trailing_sq_singular_values(Delta, r) = sum(s[r:]**2); the test
             multiplies it by tau^2.
  Zero tuning constants: r_hat is FIXED at r, there is no kappa and no beta. The
  only inputs are (r, B, alpha), all stated by the caller as the hypothesis and
  the resampling budget.

Interpretation rule (carried over from the lineage): rejection is informative
(it falsifies rank <= r); a non-rejection is not positive evidence that the rank
is exactly r, because a covariance-only readout is blind to mean-shift changes.

Pure functions. No I/O, no global seeding: the bootstrap takes an explicit rng.
"""
import numpy as np


# ---------------------------------------------- pure covariance-difference API
def covariance_difference(Y_e, Y_0, matched_n=True):
    """Delta = cov(Y_e) - cov(Y_0). Both are (n, d) with matched n.

    The matched-n guard matters: an n mismatch would bias Delta by the
    n-dependent estimation noise alone. matched_n=False drops the guard for a
    caller that calibrates Delta against a null drawn at the same unequal
    sample-size geometry (the size-matched gate null, readout="covariance" in
    precision_readout.py); the default keeps the guard.
    """
    Y_e = np.asarray(Y_e)
    Y_0 = np.asarray(Y_0)
    if matched_n and Y_e.shape[0] != Y_0.shape[0]:
        raise ValueError(
            f"matched-n violation: Y_e has {Y_e.shape[0]} rows, Y_0 has "
            f"{Y_0.shape[0]}; the covariance difference would be biased")
    return np.cov(Y_e, rowvar=False) - np.cov(Y_0, rowvar=False)


def singular_values(Delta):
    """Singular values of Delta, sorted descending. Delta is symmetric here, so
    these equal the absolute eigenvalues.
    """
    return np.linalg.svd(np.asarray(Delta), compute_uv=False)


def trailing_sq_singular_values(Delta, r):
    """The rank statistic as a pure function of the covariance difference:
    sum of the squared singular values of Delta from index r onward.
    """
    s = singular_values(Delta)
    if r < 0 or r > s.shape[0]:
        raise ValueError(f"r={r} out of range for d={s.shape[0]}")
    return float(np.sum(s[r:] ** 2))


def spectral_rank(Delta, rtol=1e-2):
    """Deterministic rank readout: number of singular values of Delta above
    rtol times the largest one. Exact in the clean linear regime, where the
    covariance difference has a sharp gap between signal and noise directions.

    rtol is a standard numerical-rank tolerance, NOT a tuning constant of the
    statistical test (the LFC test below has none). In the clean regime the
    answer is insensitive to rtol across a wide range; gap_ratio() lets a caller
    confirm the gap is sharp.
    """
    s = singular_values(Delta)
    if s.shape[0] == 0 or s[0] <= 0:
        return 0
    return int(np.sum(s > rtol * s[0]))


def gap_ratio(Delta, r):
    """s[r-1] / s[r]: how sharply the spectrum drops after rank r. A large value
    means the numerical rank r is well separated from the noise floor.
    """
    s = singular_values(Delta)
    if r <= 0 or r >= s.shape[0]:
        return float("inf")
    denom = s[r]
    if denom <= 0:
        return float("inf")
    return float(s[r - 1] / denom)


# ------------------------------------------------------- LFC bootstrap test
def lfc_rank_test(Y_e, Y_0, r, B=500, alpha=0.05, rng=None):
    """H0(r): rank(Delta) <= r, calibrated at the least-favourable configuration.

    The recentred paired bootstrap needs the row samples (not just Delta), so
    this takes Y_e, Y_0 and forms Delta internally, exactly as the lineage does.
    r_hat is fixed at r: P2, Q2 are the last (d - r) singular directions and the
    bootstrap statistic sums ALL (d - r) squared singular values of the projected
    fluctuation. No tuning constant enters the decision.

    Returns a dict: reject, stat, crit, r, B, alpha, d, n.
    """
    if rng is None:
        raise ValueError("lfc_rank_test needs an explicit rng (no global seeding)")
    Y_e = np.asarray(Y_e)
    Y_0 = np.asarray(Y_0)
    n, d = Y_e.shape
    if Y_0.shape[0] != n:
        raise ValueError(
            f"matched-n violation: Y_e has {n} rows, Y_0 has {Y_0.shape[0]}")
    if r >= d:
        raise ValueError(f"need r < d for a non-trivial null, got r={r}, d={d}")
    tau = np.sqrt(n)

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
        M = tau * (Dstar - Delta)                # recentred fluctuation
        sv = np.linalg.svd(P2.T @ M @ Q2, compute_uv=False)
        boot[b] = float(np.sum(sv ** 2))         # all of them: lo = r - r_hat = 0
    crit = float(np.quantile(boot, 1.0 - alpha))
    stat = float(tau ** 2 * np.sum(s[r:] ** 2))

    return dict(reject=bool(stat > crit), stat=stat, crit=crit, r=int(r),
                B=int(B), alpha=float(alpha), d=int(d), n=int(n))


def estimate_rank_lfc(Y_e, Y_0, r_max=None, B=500, alpha=0.05, rng=None):
    """Sequential LFC rank estimate: the smallest r in 0..r_max for which H0(r)
    is NOT rejected. Tests larger r only after rejecting all smaller ones.

    Rejection at r < true rank is the informative, high-power direction. The
    stopping test at r = true rank is a nominal-alpha test, so a caller checking
    a known answer should read both k_hat and the per-r decisions.

    Returns a dict: k_hat, r_max, decisions (list of per-r test dicts),
    all_rejected (True if every r up to r_max rejected, i.e. rank > r_max).
    """
    if rng is None:
        raise ValueError("estimate_rank_lfc needs an explicit rng")
    d = np.asarray(Y_e).shape[1]
    if r_max is None:
        r_max = d - 1
    r_max = int(min(r_max, d - 1))

    decisions = []
    k_hat = None
    for r in range(r_max + 1):
        res = lfc_rank_test(Y_e, Y_0, r, B=B, alpha=alpha, rng=rng)
        decisions.append(res)
        if not res["reject"]:
            k_hat = r
            break
    all_rejected = k_hat is None
    if all_rejected:
        k_hat = r_max + 1                       # rank exceeds the tested range
    return dict(k_hat=int(k_hat), r_max=int(r_max), decisions=decisions,
                all_rejected=bool(all_rejected))
