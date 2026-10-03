"""Oracle simulator: generate multi-environment VAR(1) data with a KNOWN shift.

Two modes, so the AXIS test can be checked against ground truth:

- "measurement": across environments, only per-channel properties change --- the
  diagonal autocorrelation of A and the per-channel noise scale. The off-diagonal
  (how channels drive each other) is held fixed. This is what a scanner / batch
  change is bet to look like.

- "mechanism": across environments, only the off-diagonal of A changes. The
  diagonal and the noise scale are held fixed. This is a genuine change in how
  the system works.

The base system is upper-triangular (diagonal + a superdiagonal of couplings),
so its eigenvalues equal its diagonal and the process stays stable for every
shift used here.

numpy only. Imports nothing outside ideas/axis/.
"""

import numpy as np

# Base system and per-mode step sizes. Kept modest so the process is always stable
# and even the smallest configured magnitude is clearly detectable at T ~ 2000.
BASE_DIAG = 0.25
BASE_SUPERDIAG = 0.10
DIAG_STEP = 0.5      # measurement: how far the diagonal moves per unit magnitude
NOISE_STEP = 0.5     # measurement: how far the noise scale moves per unit magnitude
OFFDIAG_STEP = 0.5   # mechanism:   how far the off-diagonal moves per unit magnitude

_MODES = ("measurement", "mechanism")


def _base_system(d, base_rng):
    """Upper-triangular base A0 (diag + superdiagonal) and per-channel noise std."""
    A0 = np.diag(np.full(d, BASE_DIAG)).astype(float)
    for i in range(d - 1):
        A0[i, i + 1] = BASE_SUPERDIAG
    noise0 = base_rng.uniform(0.25, 0.35, size=d)
    return A0, noise0


def _shift_amount(e, n_env, magnitude):
    """Shift for environment e: env 0 is the reference (0), last env is full."""
    if n_env <= 1:
        return 0.0
    return magnitude * (e / (n_env - 1))


def _env_params(mode, e, n_env, magnitude, A0, noise0):
    """Build (A_e, noise_e) for one environment under the chosen mode."""
    s = _shift_amount(e, n_env, magnitude)
    a0 = np.diag(A0).copy()

    if mode == "measurement":
        A_e = A0.copy()
        np.fill_diagonal(A_e, a0 + s * DIAG_STEP)   # diagonal moves
        noise_e = noise0 * (1.0 + s * NOISE_STEP)   # noise scale moves
        # off-diagonal untouched -> stays exactly equal to the base.
    elif mode == "mechanism":
        A_e = A0.copy()
        for i in range(A0.shape[0] - 1):
            A_e[i, i + 1] = BASE_SUPERDIAG + s * OFFDIAG_STEP  # off-diagonal moves
        noise_e = noise0.copy()                                # noise fixed
        # diagonal untouched -> stays exactly equal to the base.
    else:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")

    return A_e, noise_e


def _simulate_series(A, noise_std, T, rng, burnin=200):
    """Simulate T samples of x_t = A x_{t-1} + noise_std * eps after a burn-in."""
    d = A.shape[0]
    total = T + burnin
    x = np.zeros((total, d))
    for t in range(1, total):
        x[t] = A @ x[t - 1] + rng.normal(size=d) * noise_std
    return x[burnin:]


def simulate(mode, seed=0, d=6, T=2000, n_env=2, magnitude=0.6):
    """Generate n_env environments with a KNOWN shift of the given type.

    Returns
    -------
    dict with keys:
        mode      : the shift type.
        series    : list of n_env arrays, each (T, d).
        A         : list of n_env true coefficient matrices.
        noise_std : list of n_env per-channel noise std vectors.
        shifts    : list of the per-environment shift amounts.
        params    : provenance (mode, seed, d, T, n_env, magnitude).
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if n_env < 2:
        raise ValueError(f"need at least 2 environments to compare, got n_env={n_env}")

    ss = np.random.SeedSequence(seed)
    child = ss.spawn(1 + n_env)
    base_rng = np.random.default_rng(child[0])
    env_rngs = [np.random.default_rng(c) for c in child[1:]]

    A0, noise0 = _base_system(d, base_rng)

    series, A_list, noise_list, shifts = [], [], [], []
    for e in range(n_env):
        A_e, noise_e = _env_params(mode, e, n_env, magnitude, A0, noise0)
        X = _simulate_series(A_e, noise_e, T, env_rngs[e])
        series.append(X)
        A_list.append(A_e)
        noise_list.append(noise_e)
        shifts.append(_shift_amount(e, n_env, magnitude))

    return {
        "mode": mode,
        "series": series,
        "A": A_list,
        "noise_std": noise_list,
        "shifts": shifts,
        "params": {
            "mode": mode,
            "seed": int(seed),
            "d": int(d),
            "T": int(T),
            "n_env": int(n_env),
            "magnitude": float(magnitude),
        },
    }


if __name__ == "__main__":
    d = 3
    off = ~np.eye(d, dtype=bool)
    diag = np.eye(d, dtype=bool)

    # measurement: true off-diagonal identical across envs, diagonal differs.
    m = simulate("measurement", seed=0, d=d, T=300, n_env=2, magnitude=0.8)
    A0, A1 = m["A"][0], m["A"][1]
    print("smoke test: oracle_sim measurement")
    print("  max |off-diag change| :", float(np.abs(A1 - A0)[off].max()))
    print("  max |diag change|     :", float(np.abs(A1 - A0)[diag].max()))
    assert np.allclose(A1[off], A0[off]), "measurement must leave off-diagonal fixed"
    assert not np.allclose(A1[diag], A0[diag]), "measurement must move the diagonal"

    # mechanism: true diagonal identical across envs, off-diagonal differs.
    k = simulate("mechanism", seed=0, d=d, T=300, n_env=2, magnitude=0.8)
    B0, B1 = k["A"][0], k["A"][1]
    print("smoke test: oracle_sim mechanism")
    print("  max |diag change|     :", float(np.abs(B1 - B0)[diag].max()))
    print("  max |off-diag change| :", float(np.abs(B1 - B0)[off].max()))
    assert np.allclose(B1[diag], B0[diag]), "mechanism must leave the diagonal fixed"
    assert not np.allclose(B1[off], B0[off]), "mechanism must move the off-diagonal"

    for env in m["series"]:
        assert env.shape == (300, d)
    print("OK")
