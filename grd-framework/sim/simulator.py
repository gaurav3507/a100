"""GRD simulator, v1 linear regime.

Ported fresh from the Paper-2 rank-test lineage (precondition-audit scripts
80_ranktest_core.py and 81_ranktest_oracle.py). No artefact migration (Lesson 20):
the logic is re-written here, nothing is imported from the old repo.

Design contract:
  Pure functions. No file I/O, no CLI, no global seeding, no side effects on
  import. Every random draw takes an explicit numpy Generator, so a run is fully
  determined by the seed the caller passes.

Simulator spec (v1):
  d_latent latent variables, D observed dimensions (default 200).
  Latent DAG with edge probability 0.4.
  Latent SCM weights U(0.5, 1.5) with random sign; noise variances U(0.5, 1.5).
  Linear mixing X = Z A^T by default, with A of shape (D, d_latent).
  A 2-layer leaky-ReLU mixing option exists behind mixing="mlp" but is DORMANT in
  v1: it is kept for a later (nonlinear) chapter and is not exercised by the E0
  oracle. Default is linear.
  Hard intervention: zero the incoming edges of each target node and set that
  node's noise variance to U(2.0, 4.0) (or a fixed iv_scale). Soft intervention:
  multiply the target noise variance by U(2.5, 4.0). Shift intervention: add a
  constant to the target node (moves the mean, leaves the covariance unchanged).
  Multi-environment: one observational environment plus one environment per
  intervention target SET; a target set may contain several nodes (multi-node),
  with the number of intervened nodes k configurable by the caller.

Rank fact used by the oracle (see 80_ranktest_core.py for the derivation):
  For a hard single-node intervention the observed covariance difference is a
  rank-<=2 update, and rank 1 when the intervened node is a source. Linear mixing
  and shared A preserve that rank. The constructed rank for any intervention set
  is read here directly off the exact population covariance difference, so the
  oracle never hand-derives it.
"""
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, List, Dict

import numpy as np


# --------------------------------------------------------------- latent SCM
def make_scm(d_latent: int, rng: np.random.Generator, edge_prob: float = 0.4):
    """Random DAG: strictly lower-triangular weights, then a random relabel.

    Returns (B, noise_var, is_source):
      B          (d_latent, d_latent) weighted adjacency, B[i, j] is the edge j->i.
      noise_var  (d_latent,) per-node noise variances, U(0.5, 1.5).
      is_source  (d_latent,) boolean, True where a node has no incoming edges.
    """
    W = rng.uniform(0.5, 1.5, (d_latent, d_latent)) * rng.choice([-1.0, 1.0], (d_latent, d_latent))
    B = np.where(np.tril(rng.random((d_latent, d_latent)) < edge_prob, -1), W, 0.0)
    perm = rng.permutation(d_latent)
    B = B[np.ix_(perm, perm)]          # relabelling a DAG leaves it a DAG
    noise_var = rng.uniform(0.5, 1.5, d_latent)
    is_source = (np.abs(B).sum(1) == 0)
    return B, noise_var, is_source


def _effective_scm(B, noise_var, kind, nodes, iv_scale=None, soft_scale=None):
    """Deterministic (Be, nv, shift) for an environment. No random draws.

    Used by population_latent_cov so the population covariance matches the data
    when the caller pins iv_scale / soft_scale. The rank of the covariance
    difference is invariant to the exact positive noise-variance values, so a
    representative value gives the correct constructed rank even when the data
    used an unpinned random draw.
    """
    d = len(noise_var)
    Be, nv, shift = B.copy(), noise_var.copy(), np.zeros(d)
    for i in nodes:
        if kind == "hard":
            Be[i, :] = 0.0
            nv[i] = 3.0 if iv_scale is None else float(iv_scale)
        elif kind == "soft":
            nv[i] = nv[i] * (3.25 if soft_scale is None else float(soft_scale))
        elif kind == "shift":
            pass  # a shift moves the mean only; covariance is unchanged
        else:
            raise ValueError(f"unknown intervention kind {kind!r}")
    return Be, nv, shift


def sample_latent(B, noise_var, n, rng, kind=None, nodes=(), rng_iv=None,
                  iv_scale=None):
    """Draw n latent samples Z of shape (n, d_latent).

    Z = (eps + shift) M^T with M = (I - Be)^-1. For kind=None this is the
    observational environment. Intervention draws use rng_iv when given (so the
    same target set can be reproduced independently of the sampling stream);
    otherwise they use rng. iv_scale=None keeps the U(2.0, 4.0) hard draw.
    """
    d = len(noise_var)
    Be, nv, shift = B.copy(), noise_var.copy(), np.zeros(d)
    riv = rng_iv if rng_iv is not None else rng
    for i in nodes:
        if kind == "hard":
            Be[i, :] = 0.0
            nv[i] = (float(riv.uniform(2.0, 4.0)) if iv_scale is None
                     else float(iv_scale))
        elif kind == "soft":
            nv[i] = nv[i] * float(riv.uniform(2.5, 4.0))
        elif kind == "shift":
            shift[i] = float(riv.uniform(2.0, 4.0))
        else:
            raise ValueError(f"unknown intervention kind {kind!r}")
    M = np.linalg.inv(np.eye(d) - Be)
    eps = rng.standard_normal((n, d)) * np.sqrt(nv)
    return (eps + shift) @ M.T


def population_latent_cov(B, noise_var, kind=None, nodes=(), iv_scale=None,
                          soft_scale=None):
    """Exact latent covariance M diag(nv) M^T for an environment. No sampling."""
    if kind is None or len(nodes) == 0:
        Be, nv = B.copy(), noise_var.copy()
    else:
        Be, nv, _ = _effective_scm(B, noise_var, kind, nodes, iv_scale, soft_scale)
    d = len(noise_var)
    M = np.linalg.inv(np.eye(d) - Be)
    return M @ np.diag(nv) @ M.T


def constructed_rank(B, noise_var, kind, nodes, iv_scale=None, soft_scale=None,
                     rtol=1e-9):
    """Numerical rank of the exact population covariance difference vs the
    observational environment. This is the ground-truth constructed rank.
    """
    d0 = population_latent_cov(B, noise_var)
    de = population_latent_cov(B, noise_var, kind, nodes, iv_scale, soft_scale)
    s = np.sort(np.abs(np.linalg.eigvalsh(de - d0)))[::-1]
    if s[0] <= 0:
        return 0
    return int(np.sum(s > rtol * s[0]))


# ------------------------------------------------------------------- mixing
def mix_linear(Z, A):
    """Linear mixing X = Z A^T. A is (D, d_latent)."""
    return Z @ A.T


def mix_mlp(Z, A1, A2, s, slope=0.1):
    """2-layer leaky-ReLU mixing, nonlinearity scale s. DORMANT in v1.

    h_nl = (1 - s) h + s leaky_relu(h), so s=0 is EXACTLY the linear map A2 @ A1
    and s=1 is the full leaky-ReLU network. Kept for a later chapter; the E0
    oracle does not exercise this path.
    """
    h = Z @ A1.T
    h = (1.0 - s) * h + s * np.where(h > 0, h, slope * h)
    return h @ A2.T


def add_obs_noise(X, sd, rng):
    """Isotropic observation noise. The SAME sd is used in every environment, so
    it contributes an identical term to each Sigma and cancels exactly from a
    covariance difference; only the PCA is regularised by it.
    """
    return X + sd * rng.standard_normal(X.shape)


# ----------------------------------------------------------- multi-env build
@dataclass
class EnvSpec:
    env_id: str
    kind: Optional[str] = None          # None (observational), "hard", "soft", "shift"
    nodes: Tuple[int, ...] = ()
    iv_scale: Optional[float] = None    # pin the hard noise variance for a known answer


@dataclass
class Environment:
    spec: EnvSpec
    X: np.ndarray                       # (n, D) observed
    Z: np.ndarray                       # (n, d_latent) true latents
    pop_cov: np.ndarray                 # (d_latent, d_latent) exact latent covariance


@dataclass
class Dataset:
    d_latent: int
    D: int
    n_per_env: int
    seed: int
    mixing: str
    edge_prob: float
    obs_noise_frac: float
    B: np.ndarray
    noise_var: np.ndarray
    is_source: np.ndarray
    A: np.ndarray
    sd_obs: float
    environments: Dict[str, Environment] = field(default_factory=dict)


def simulate(d_latent, D, n_per_env, env_specs: Sequence[EnvSpec], seed,
             mixing="linear", edge_prob=0.4, obs_noise_frac=0.1):
    """Build a multi-environment linear dataset.

    One SCM and one mixing map A are drawn per seed and SHARED across every
    environment (the shared-mixing assumption behind the identifiability
    theorems). Each environment draws its own independent samples. Observation
    noise uses one sd, fixed from the first environment's scale, so it cancels
    from every covariance difference.

    mixing="mlp" is refused here on purpose: v1 is linear only and the 2-layer
    path is dormant.
    """
    if mixing != "linear":
        raise NotImplementedError(
            "v1 is linear only; mix_mlp exists but is dormant and not wired into "
            "simulate(). Do not use it in M1.")
    rng = np.random.default_rng(seed)
    B, noise_var, is_source = make_scm(d_latent, rng, edge_prob=edge_prob)
    A = rng.standard_normal((D, d_latent))

    envs: Dict[str, Environment] = {}
    sd_obs = None
    for spec in env_specs:
        Z = sample_latent(B, noise_var, n_per_env, rng, kind=spec.kind,
                          nodes=spec.nodes, iv_scale=spec.iv_scale)
        Xsig = mix_linear(Z, A)
        if sd_obs is None:
            sd_obs = obs_noise_frac * float(np.mean(Xsig.std(0)))
        X = add_obs_noise(Xsig, sd_obs, rng)
        # Self-defending invariant: the linear regime must never produce
        # non-finite data. This catches a genuinely near-singular (I - Be) or an
        # overflow, as opposed to the benign matmul FPE flags that the macOS
        # Accelerate BLAS raises on finite inputs.
        if not (np.isfinite(Z).all() and np.isfinite(X).all()):
            raise FloatingPointError(
                f"non-finite samples in environment {spec.env_id!r} (seed {seed})")
        pop_cov = population_latent_cov(B, noise_var, spec.kind, spec.nodes,
                                        spec.iv_scale)
        envs[spec.env_id] = Environment(spec=spec, X=X, Z=Z, pop_cov=pop_cov)

    return Dataset(d_latent=d_latent, D=D, n_per_env=n_per_env, seed=int(seed),
                   mixing=mixing, edge_prob=edge_prob, obs_noise_frac=obs_noise_frac,
                   B=B, noise_var=noise_var, is_source=is_source, A=A,
                   sd_obs=float(sd_obs), environments=envs)
