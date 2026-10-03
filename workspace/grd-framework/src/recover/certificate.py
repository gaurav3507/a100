"""Finite-sample certificate for a precision-difference recovery direction.

The certificate is intentionally separate from the detectability gate. A large
precision signal does not imply that its leading eigenvector is identifiable:
the first and second eigenvalues must also be separated relative to sampling
uncertainty.

For iid Gaussian rows, the centered sample covariance has a Wishart law with
``n - 1`` degrees of freedom. Gaussian singular-value concentration gives a
data-dependent operator-norm bound for each sample precision matrix. Weyl's
inequality turns the empirical eigengap into a lower bound on the population
gap, and the population-gap Davis-Kahan variant of Yu, Wang and Samworth (2015)
then bounds the leading-eigenvector angle.

This certifies the sample direction against the leading eigenvector of the
population precision difference. Calling that population direction a causal
latent direction requires an additional model-specific alignment assumption.
"""

import numpy as np


def _as_data_matrix(Y, name):
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional data matrix")
    if not np.isfinite(Y).all():
        raise ValueError(f"{name} must contain only finite values")
    n, d = Y.shape
    if d < 2:
        raise ValueError(f"{name} needs at least two columns for an eigengap")
    if n <= d:
        raise ValueError(
            f"{name} needs n > d for an invertible sample covariance; "
            f"got n={n}, d={d}")
    return Y


def gaussian_precision_error_bound(Y, delta):
    """Bound ``||inv(cov(Y)) - inv(Sigma)||_op`` from the observed sample.

    Assumes the rows of Y are iid Gaussian with an unknown positive-definite
    covariance Sigma. The probability is at least ``1 - delta``. The formula
    uses the exact ``n - 1`` degrees of freedom of the centered Gaussian sample
    covariance and has no fitted constants.

    Returns the sample precision, its error radius, and concentration metadata.
    If the lower singular-value bound is nonpositive, ``error_bound`` is
    infinite and ``well_posed`` is false.
    """
    Y = _as_data_matrix(Y, "Y")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    n, d = Y.shape
    degrees_freedom = n - 1
    rho = (
        np.sqrt(d / degrees_freedom)
        + np.sqrt(2.0 * np.log(2.0 / delta) / degrees_freedom)
    )
    sample_precision = np.linalg.inv(np.cov(Y, rowvar=False))
    precision_norm = float(np.linalg.norm(sample_precision, ord=2))
    well_posed = bool(rho < 1.0)
    error_bound = (
        float((2.0 * rho + rho * rho) * precision_norm)
        if well_posed else float("inf")
    )
    return dict(
        sample_precision=sample_precision,
        error_bound=error_bound,
        rho=float(rho),
        precision_norm=precision_norm,
        n=int(n),
        d=int(d),
        degrees_freedom=int(degrees_freedom),
        delta=float(delta),
        well_posed=well_posed,
    )


def precision_direction_certificate(
        Y_env, Y_obs, delta=0.05, simultaneous_matrices=2):
    """Certify the leading precision-difference eigenvector under Gaussian data.

    ``simultaneous_matrices`` is the number K of distinct sample covariance
    matrices covered by the requested familywise probability. Each covariance
    receives failure probability ``delta / K``. For one environment and one
    control use K=2. For k environments sharing one control, call this function
    with K=k+1 for every environment; then all k direction bounds hold
    simultaneously with probability at least ``1 - delta`` by a union bound.

    The returned angle is relative to the leading eigenvector of the population
    precision difference, not automatically to a causal latent direction.
    ``nonvacuous`` is true exactly when the certified angle is below 90 degrees.
    """
    Y_env = _as_data_matrix(Y_env, "Y_env")
    Y_obs = _as_data_matrix(Y_obs, "Y_obs")
    if Y_env.shape[1] != Y_obs.shape[1]:
        raise ValueError("Y_env and Y_obs must have the same number of columns")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if (not isinstance(simultaneous_matrices, (int, np.integer))
            or simultaneous_matrices < 2):
        raise ValueError("simultaneous_matrices must be an integer >= 2")

    matrix_delta = float(delta / simultaneous_matrices)
    obs_bound = gaussian_precision_error_bound(Y_obs, matrix_delta)
    env_bound = gaussian_precision_error_bound(Y_env, matrix_delta)
    sample_difference = (
        env_bound["sample_precision"] - obs_bound["sample_precision"])
    eigenvalues, eigenvectors = np.linalg.eigh(sample_difference)
    direction = eigenvectors[:, -1]
    empirical_gap = float(eigenvalues[-1] - eigenvalues[-2])
    perturbation_bound = float(
        obs_bound["error_bound"] + env_bound["error_bound"])
    population_gap_lower = float(empirical_gap - 2.0 * perturbation_bound)

    gap_separated = bool(
        np.isfinite(perturbation_bound) and population_gap_lower > 0.0)
    if gap_separated:
        raw_sin_bound = float(
            2.0 * perturbation_bound / population_gap_lower)
        sin_bound = float(min(1.0, raw_sin_bound))
        angle_bound_deg = float(np.degrees(np.arcsin(sin_bound)))
    else:
        raw_sin_bound = float("inf")
        sin_bound = 1.0
        angle_bound_deg = 90.0

    nonvacuous = bool(gap_separated and raw_sin_bound < 1.0)
    if nonvacuous:
        status = "CERTIFIED"
    elif gap_separated:
        status = "VACUOUS_ANGLE"
    else:
        status = "GAP_NOT_SEPARATED"

    return dict(
        direction=direction,
        leading_eigenvalue=float(eigenvalues[-1]),
        second_eigenvalue=float(eigenvalues[-2]),
        empirical_gap=empirical_gap,
        perturbation_bound=perturbation_bound,
        population_gap_lower=population_gap_lower,
        raw_sin_bound=raw_sin_bound,
        sin_bound=sin_bound,
        angle_bound_deg=angle_bound_deg,
        gap_separated=gap_separated,
        nonvacuous=nonvacuous,
        status=status,
        confidence=float(1.0 - delta),
        familywise_delta=float(delta),
        simultaneous_matrices=int(simultaneous_matrices),
        per_matrix_delta=matrix_delta,
        obs_error_bound=float(obs_bound["error_bound"]),
        env_error_bound=float(env_bound["error_bound"]),
        obs_rho=float(obs_bound["rho"]),
        env_rho=float(env_bound["rho"]),
        assumption=(
            "iid Gaussian rows with positive-definite covariance; fixed or "
            "independently fitted projection"),
        target=(
            "leading eigenvector of the population precision difference"),
    )
