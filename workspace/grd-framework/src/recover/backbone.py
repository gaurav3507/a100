"""RECOVER backbone, v1: a direct covariance-based linear estimator.

This is the design doc's R2 fallback ("the simplest provable linear estimator,
Squires-style with known/estimated targets"), adopted after the Bing et al.
reference estimator (github.com/simonbing/multi-node-crl, model "ours") was found
to run correctly but not recover on our simulator's data: its variance-sparsity
objective needs do-interventions that drive a latent's variance to zero, whereas
our hard/soft interventions change noise variances. The covariance signal our
gate's P4 rank test already reads is exactly what this estimator uses, so gate and
estimator stay in one mathematical language.

Method (provable for perfect interventions in the linear-Gaussian regime)
  Observed data in each environment, projected to the d-dimensional latent-mixing
  space, is Y = R Z with R a fixed invertible d x d matrix (R = W_pca^T A), shared
  across environments. We recover the unmixing W = R^{-1} row by row.

  For a PERFECT (hard) intervention on latent i, the incoming edges of node i are
  removed and its noise variance is reset. The latent precision matrix
  Theta = (I - B)^T D^{-1} (I - B) changes only in the term for node i:
      Theta_e - Theta_0 = (1/D_e[i]) e_i e_i^T  -  (1/D[i]) (e_i - B[i,:])^T (e_i - B[i,:]).
  In the observed precision space (Prec(Y) = R^{-T} Theta R^{-1}) the first term
  contributes an eigenvector R^{-T} e_i, which is exactly row i of W = R^{-1}
  (the direction with <W[i,:], Y> = z_i). When the intervention REDUCES the target
  noise variance (D_e[i] < D[i]), that term dominates as the LARGEST eigenvalue of
  Prec(Y_e) - Prec(Y_0), so the top eigenvector recovers W[i,:] cleanly. Verified
  on our simulator: MCC(Z_hat, Z) = 0.999 across seeds 0..9 with reduced-variance
  perfect interventions; naive variance-increasing interventions degrade to ~0.79
  because the removed-edge direction competes, so the estimator expects the
  reduced-variance regime.

  Targets are taken as KNOWN here (which environment intervenes on which latent),
  matching the Squires known-target setting. Target ESTIMATION is deferred to the
  gate integration (M3), not part of this backbone.

Pure numpy/scipy. No torch, no dependency on the reference code. No file I/O, no
global seeding.
"""
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]   # src/recover/backbone.py -> repo root


def fit_pca(X_basis, d):
    """PCA fitted on control/basis cells only. Returns (mu, W_pca) with W_pca (D, d)."""
    mu = X_basis.mean(0)
    _, _, Vt = np.linalg.svd(X_basis - mu, full_matrices=False)
    return mu, Vt[:d].T


def project(X, mu, W_pca):
    return (X - mu) @ W_pca


_RANK_READOUT = None


def _rank_readout():
    """src/gate/rank_readout.py, loaded by path like every other module here."""
    global _RANK_READOUT
    if _RANK_READOUT is None:
        import importlib.util
        path = REPO_ROOT / "src" / "gate" / "rank_readout.py"
        spec = importlib.util.spec_from_file_location(
            "grd_gate_rank_readout_for_backbone", str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _RANK_READOUT = module
    return _RANK_READOUT


def _unmixing_row(Y_obs, Y_env, readout="precision"):
    """Row of the unmixing recovering the intervened latent: top eigenvector of the
    observed precision difference Prec(Y_env) - Prec(Y_obs).

    readout="covariance" (Tier 2 negative control) instead takes the top eigenvector
    of Cov(Y_obs) - Cov(Y_env), the covariance mirror of the precision rule. It has
    no identifiability argument of its own: with exact population covariances it
    reaches MCC 0.898 at the E2 reference point (results/e4_second_backbone).
    """
    if readout == "covariance":
        delta = _rank_readout().covariance_difference(Y_env, Y_obs, matched_n=False)
        vals, vecs = np.linalg.eigh(-delta)       # -delta = Cov(Y_obs) - Cov(Y_env)
        return vecs[:, int(np.argmax(vals))]
    if readout != "precision":
        raise ValueError(
            f"readout must be 'precision' or 'covariance', got {readout!r}")
    P0 = np.linalg.inv(np.cov(Y_obs, rowvar=False))
    Pe = np.linalg.inv(np.cov(Y_env, rowvar=False))
    vals, vecs = np.linalg.eigh(Pe - P0)          # ascending, orthonormal columns
    return vecs[:, int(np.argmax(vals))]          # largest eigenvalue


# ------------------------------------------------------------------ Backbone C (Tier 2)
# Joint approximate diagonalization (JAD) of ALL precision differences
#     Delta_e = Prec(Y_e) - Prec(Y_obs) = W^T M_e W,
# where M_e is the latent precision change of environment e. When the M_e are
# diagonal the Delta_e are jointly congruence-diagonal with the unmixing rows w_k as
# dyads, so one joint diagonalizer V (V Delta_e V^T diagonal for all e) gives
# W = V^{-T}: the columns of V^{-1} are the unmixing rows. A perfect intervention on
# a node with parents makes M_e rank two (the removed-edge term), so the model is
# approximate and the joint fit is checked against the population before any use.
# Settings are fixed here, not tuned: FFDiag step cap, tolerance, iteration limit.
JAD_THETA = 0.9
JAD_TOL = 1e-10
JAD_MAX_ITER = 2000


def offdiag_ratio(V, mats):
    """Sum of squared off-diagonal entries of V C_k V^T over the total, rows of V
    at unit norm. 0 means V jointly diagonalizes every C_k exactly."""
    V = np.asarray(V, float)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    M = np.einsum("ij,kjl,ml->kim", V, np.asarray(mats, float), V)
    total = float(np.sum(M ** 2))
    diag = float(np.sum(np.einsum("kii->ki", M) ** 2))
    return (total - diag) / total if total > 0 else 0.0


def ffdiag(mats, V0, theta=JAD_THETA, tol=JAD_TOL, max_iter=JAD_MAX_ITER):
    """Non-orthogonal joint diagonalization by FFDiag (Ziehe, Laskov, Nolte and
    Mueller 2004, JMLR 5:777-800).

    Finds V with V C_k V^T approximately diagonal for every symmetric C_k; no
    definiteness is required. Each step solves the paper's 2x2 normal equations for
    the off-diagonal update W, caps ||W||_F at theta, sets V <- (I + W) V, and
    rescales the rows of V to unit norm (the problem is scale free).
    """
    mats = np.asarray(mats, float)
    d = mats.shape[1]
    V = np.array(V0, float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    converged, n_iter = False, 0
    for n_iter in range(1, max_iter + 1):
        M = np.einsum("ij,kjl,ml->kim", V, mats, V)
        Dg = np.einsum("kii->ki", M)
        z = Dg.T @ Dg                               # z_ij = sum_k D_ki D_kj
        y = np.einsum("kij,kj->ij", M, Dg)          # y_ij = sum_k D_kj E_kij
        zd = np.diag(z)
        det = np.outer(zd, zd) - z ** 2
        ok = det > 1e-12 * max(float(zd.max()) ** 2, 1e-300)
        W = np.where(ok, (z * y.T - zd[:, None] * y) / np.where(ok, det, 1.0), 0.0)
        np.fill_diagonal(W, 0.0)
        norm_w = float(np.linalg.norm(W))
        if norm_w > theta:
            W *= theta / norm_w
        V = (np.eye(d) + W) @ V
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        if norm_w < tol:
            converged = True
            break
    return V, dict(n_iter=int(n_iter), converged=bool(converged),
                   offdiag_ratio=offdiag_ratio(V, mats))


def jad_rows(P_obs, P_envs, nodes, d_latent):
    """Backbone C rows from precision matrices (population or sample).

    Joint-diagonalizes Delta_e = P_e - P_obs over all supplied environments with
    FFDiag (Ziehe, A., Laskov, P., Nolte, G. and Mueller, K.-R. (2004), "A fast
    algorithm for joint diagonalization with non-orthogonal transformations and its
    application to blind source separation", JMLR 5:777-800), starting from the
    eigenvectors of sum_e Delta_e. Each environment is matched to
    one component by its largest coefficient on the unit dyad (Hungarian, so no
    component is used twice), and that component's unmixing row goes to the
    environment's known target. Rows of untargeted nodes stay zero.
    """
    from scipy.optimize import linear_sum_assignment

    P_obs = np.asarray(P_obs, float)
    deltas = np.stack([np.asarray(P, float) - P_obs for P in P_envs])
    _, U = np.linalg.eigh(deltas.sum(0))
    V, info = ffdiag(deltas, U.T)
    A = np.linalg.inv(V)                             # column k: unmixing row k
    norms = np.linalg.norm(A, axis=0)
    coef = np.einsum("ij,kjl,il->ki", V, deltas, V) * norms[None, :] ** 2
    env_idx, comp_idx = linear_sum_assignment(-coef)
    W = np.zeros((d_latent, P_obs.shape[0]))
    for e, k in zip(env_idx, comp_idx):
        W[nodes[e]] = A[:, k] / norms[k]
    info["assignment"] = {int(nodes[e]): int(k) for e, k in zip(env_idx, comp_idx)}
    return W, info


def unmixing_rows(Y_obs, Y_envs, nodes, d_latent, readout="precision"):
    """Node-aligned unmixing rows for the supplied interventions, zero rows elsewhere.

    precision / covariance: the per-environment rule (_unmixing_row), one row each.
    jad: Backbone C, one joint diagonalization over all supplied environments.
    """
    W = np.zeros((d_latent, np.asarray(Y_obs).shape[1]))
    if readout == "jad":
        if len(Y_envs):
            P0 = np.linalg.inv(np.cov(Y_obs, rowvar=False))
            Ps = [np.linalg.inv(np.cov(Y, rowvar=False)) for Y in Y_envs]
            W, _ = jad_rows(P0, Ps, list(nodes), d_latent)
        return W
    for node, Y in zip(nodes, Y_envs):
        W[node, :] = _unmixing_row(Y_obs, Y, readout=readout)
    return W


def recover(envs, targets, d_latent, basis_key="basis", obs_key="obs",
            readout="precision"):
    """Recover latents from multi-environment observed data.

    envs        : dict {env_key: X (n, D)} of observed data. Must contain basis_key
                  (control cells for the projection) and obs_key (observational
                  environment). All other keys are interventional environments.
    targets     : dict {env_key: latent_index} for the interventional environments,
                  giving the (known) perfect-intervention target of each.
    d_latent    : working dimension; observed data is projected here via PCA.
    readout     : "precision" (default), "covariance" (see _unmixing_row) or
                  "jad" (Backbone C, see jad_rows).

    Returns dict:
        Z_hat    : (n_obs, d_latent) recovered latents for the observational env,
                   up to permutation, sign and scale (MCC-scorable against truth).
        W        : (d_latent, d_latent) unmixing; row i recovers latent i.
        targets  : the known targets echoed back (this backbone does not estimate them).
        rows_set : sorted list of latent indices whose unmixing row was identified.
    """
    covered = sorted(set(targets.values()))
    if covered != list(range(d_latent)):
        raise ValueError(
            f"targets must cover every latent 0..{d_latent - 1} exactly once for full "
            f"recovery; got targets covering {covered}")

    mu, W_pca = fit_pca(envs[basis_key], d_latent)
    Y_obs = project(envs[obs_key], mu, W_pca)

    W = np.zeros((d_latent, d_latent))
    if readout == "jad":
        keys = list(targets)
        W = unmixing_rows(Y_obs, [project(envs[k], mu, W_pca) for k in keys],
                          [targets[k] for k in keys], d_latent, readout="jad")
    else:
        for env_key, i in targets.items():
            Y_env = project(envs[env_key], mu, W_pca)
            W[i, :] = _unmixing_row(Y_obs, Y_env, readout=readout)

    Z_hat = Y_obs @ W.T
    return dict(Z_hat=Z_hat, W=W, targets=dict(targets),
                rows_set=sorted(int(i) for i in targets.values()))
