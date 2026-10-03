"""Score whether a change between two environments loads on the diagonal
(measurement) or the off-diagonal (mechanism) of a VAR(1) coefficient matrix.

Given two fitted coefficient matrices with per-coefficient standard errors, we
form a z-score for each coefficient's change:

    z_ij = |A2_ij - A1_ij| / sqrt(SE1_ij**2 + SE2_ij**2)

count how many diagonal vs off-diagonal coefficients are rejected (z above a
threshold), and summarise where the change concentrates.

Diagonal-heavy  -> the setting changed how each channel is measured (its own
                   autocorrelation / noise scale). Reads as MEASUREMENT.
Off-diagonal    -> the setting changed how channels drive each other.
                   Reads as MECHANISM.

The two environments MUST have matched sample sizes: standard errors shrink with
sample size, so comparing rejection counts across environments with different n
would confound "more data" with "real change". This is enforced with an assert.

numpy only. Imports nothing outside ideas/axis/.
"""

import numpy as np

_SE_FLOOR = 1e-12  # guard against divide-by-zero for degenerate SEs


def dissociate(A1, se1, n1, A2, se2, n2, threshold=3.0):
    """Compare two VAR(1) coefficient matrices and locate the change.

    Parameters
    ----------
    A1, A2 : array (d, d)
        Coefficient matrices for environment 1 and environment 2.
    se1, se2 : array (d, d)
        Per-coefficient standard errors for A1 and A2 (e.g. from a bootstrap).
    n1, n2 : int
        Sample sizes used to fit each environment. Must be equal.
    threshold : float
        A coefficient counts as rejected (changed) when its z-score exceeds this.

    Returns
    -------
    ratio : float or None
        The off-diagonal share, offdiag_rate / (offdiag_rate + diag_rate), in
        [0, 1]. Near 0 -> change is on the diagonal (measurement). Near 1 ->
        change is off-diagonal (mechanism). None when there are no rejections
        at all (no signal).
    summary : dict
        Counts, rates, and diagnostics (JSON-serializable).
    """
    A1 = np.asarray(A1, dtype=float)
    A2 = np.asarray(A2, dtype=float)
    se1 = np.asarray(se1, dtype=float)
    se2 = np.asarray(se2, dtype=float)

    for name, M in (("A1", A1), ("A2", A2), ("se1", se1), ("se2", se2)):
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"{name} must be a square d x d matrix, got {M.shape}")
    if not (A1.shape == A2.shape == se1.shape == se2.shape):
        raise ValueError("A1, A2, se1, se2 must all share the same shape")

    # Rule: matched sample sizes across the two environments.
    assert int(n1) == int(n2), (
        f"sample sizes must match across environments: n1={n1}, n2={n2}. "
        "Subsample the larger environment before calling dissociate()."
    )

    d = A1.shape[0]
    se_diff = np.sqrt(se1 ** 2 + se2 ** 2)
    se_diff = np.maximum(se_diff, _SE_FLOOR)
    z = np.abs(A2 - A1) / se_diff

    diag_mask = np.eye(d, dtype=bool)
    offdiag_mask = ~diag_mask
    reject = z > threshold

    diag_reject = int(reject[diag_mask].sum())
    offdiag_reject = int(reject[offdiag_mask].sum())
    diag_total = d
    offdiag_total = d * (d - 1)

    diag_rate = diag_reject / diag_total if diag_total else 0.0
    offdiag_rate = offdiag_reject / offdiag_total if offdiag_total else 0.0

    denom = diag_rate + offdiag_rate
    ratio = (offdiag_rate / denom) if denom > 0 else None

    rate_ratio = (offdiag_rate / diag_rate) if diag_rate > 0 else None

    summary = {
        "d": d,
        "threshold": float(threshold),
        "n_matched": int(n1),
        "diag_reject": diag_reject,
        "offdiag_reject": offdiag_reject,
        "diag_total": diag_total,
        "offdiag_total": offdiag_total,
        "diag_rate": diag_rate,
        "offdiag_rate": offdiag_rate,
        "offdiag_share": ratio,
        "offdiag_to_diag_rate_ratio": rate_ratio,
        "max_z_diag": float(z[diag_mask].max()) if diag_total else None,
        "max_z_offdiag": float(z[offdiag_mask].max()) if offdiag_total else None,
    }
    return ratio, summary


if __name__ == "__main__":
    # Smoke test: build two coefficient matrices whose change is purely diagonal,
    # confirm it reads as measurement (low off-diagonal share).
    d = 4
    rng = np.random.default_rng(0)
    A1 = 0.3 * np.eye(d) + 0.05 * rng.standard_normal((d, d))
    A2 = A1.copy()
    A2[np.diag_indices(d)] += 0.4  # move only the diagonal
    se = np.full((d, d), 0.02)

    ratio, summary = dissociate(A1, se, 500, A2, se, 500, threshold=3.0)
    print("smoke test: dissociate (pure diagonal change)")
    print("  diag_reject   :", summary["diag_reject"])
    print("  offdiag_reject:", summary["offdiag_reject"])
    print("  offdiag_share :", ratio)
    assert summary["diag_reject"] > 0
    assert summary["offdiag_reject"] == 0
    assert ratio == 0.0

    # And confirm the matched-sample-size guard fires.
    try:
        dissociate(A1, se, 500, A2, se, 400)
    except AssertionError as e:
        print("  matched-n guard fired OK")
    else:
        raise SystemExit("matched-n guard did NOT fire")
    print("OK")
