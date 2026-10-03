"""DISCOVER module, v1: latent graph estimate with honest abstention.

Third pipeline stage. Takes the node-aligned latents the backbone recovered and
reports the causal graph among them, marking any edge it cannot resolve as
UNDECIDED rather than guessing.

Regime: PERFECT interventions with KNOWN targets (what RECOVER recovers on). The
recovered latents are permutation-resolved (node-aligned via known targets) and
sign/scale-ambiguous. In that regime the latent DAG is identified up to the
standard ambiguity, so most edges are decidable; the abstention rule still exists
and fires on any edge that is not.

Method (standard, not a new algorithm):
  1. Descendant / order identification from perfect interventions. Under a perfect
     intervention on node i (do, variance reduced), only i and its descendants
     change distribution, so |log(var_i(z_j) / var_obs(z_j))| large means j is a
     descendant of i. This is the classic interventional identification of the
     causal order (Eberhardt et al.; Hauser and Buhlmann 2012, interventional
     Markov equivalence / GIES). Order nodes by ancestor count.
  2. Edge coefficients by ordered least-squares regression of each node on its
     predecessors (classic linear-Gaussian SEM estimation). Standardized
     coefficients separate true edges from non-edges cleanly.
  3. Abstention (the honest part). An edge k -> j is:
       EDGE_DECIDED_PRESENT  coefficient clearly large AND the intervention data
                             confirms k is an ancestor of j (orientation checked);
       EDGE_DECIDED_ABSENT   coefficient clearly ~0 AND the pair is confidently
                             unrelated or oriented the other way;
       EDGE_UNDECIDED_ORDER  a coefficient is present but the order/orientation is
                             not confirmed by the interventions;
       EDGE_UNDECIDED_WEAK   the coefficient is borderline (cannot separate a weak
                             edge from noise).
  The orientation check makes the decided set safe: a misordered true edge fails the
  ancestor confirmation and is abstained rather than decided the wrong way, and a
  backward coefficient from an order error is never DECIDED_PRESENT. The per-edge
  status codes parallel the design doc's Module-3 codes; here they concern
  latent-recovery ambiguity.

Thresholds. Standardized-coefficient bands coef_hi=0.15 / coef_lo=0.08 bracket the
observed gap between true edges (|coef| above ~0.13) and non-edges (below ~0.09),
which are the conventional small-effect-size cutoffs for a standardized coefficient;
the band between them abstains, so the cutoffs are not knife-edge. The variance-log
bands var_hi=0.4 / var_lo=0.15 separate a real interventional variance change
(above ~1.5x) from none. Borderline cases abstain rather than misdecide.

Pure readout (numpy/scipy). Operates on already-recovered, node-aligned latents; no
file I/O, no global seeding.
"""
import numpy as np


DECIDED_PRESENT = "EDGE_DECIDED_PRESENT"
DECIDED_ABSENT = "EDGE_DECIDED_ABSENT"
UNDECIDED_ORDER = "EDGE_UNDECIDED_ORDER"
UNDECIDED_WEAK = "EDGE_UNDECIDED_WEAK"


def _zscore(A):
    A = A - A.mean(0)
    sd = A.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return A / sd


def discover(Z_obs, Z_int, var_hi=0.4, var_lo=0.15, coef_hi=0.15, coef_lo=0.08):
    """Estimate the latent graph with abstention.

    Z_obs : (n, d) node-aligned recovered latents, observational environment.
    Z_int : dict {node_index: (n, d)} node-aligned recovered latents under the
            perfect intervention on that node. Keys should cover 0..d-1.

    Returns dict:
        status       : {(k, j): code} for every ordered pair k != j (edge k -> j).
        B_hat        : (d, d) with B_hat[j, k] the decided-present coefficient of k -> j.
        order        : estimated topological order (ancestors first).
        n_decided_present / n_decided_absent / n_undecided : counts.
    """
    Z_obs = np.asarray(Z_obs)
    d = Z_obs.shape[1]
    if sorted(Z_int.keys()) != list(range(d)):
        raise ValueError(
            f"Z_int must provide a perfect-intervention environment for every latent "
            f"0..{d - 1}; got keys {sorted(Z_int.keys())}")

    v0 = Z_obs.var(0)
    logr = np.zeros((d, d))
    for i in range(d):
        ve = np.asarray(Z_int[i]).var(0)
        for j in range(d):
            if i != j:
                logr[i, j] = abs(np.log(ve[j] / v0[j]))
    reach_strong = logr > var_hi        # do(i) clearly changed var(j): j descendant of i
    reach_none = logr < var_lo          # do(i) left var(j) unchanged: j not a descendant

    order = list(np.argsort(reach_strong.sum(0), kind="stable"))

    Zs = _zscore(Z_obs)
    coef = {}
    for idx, j in enumerate(order):
        preds = order[:idx]
        if preds:
            beta, _, _, _ = np.linalg.lstsq(Zs[:, preds], Zs[:, j], rcond=None)
            for k, b in zip(preds, beta):
                coef[(int(k), int(j))] = abs(float(b))

    status = {}
    B_hat = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            c = coef.get((i, j), 0.0)
            if c > coef_hi and reach_strong[i, j] and not reach_strong[j, i]:
                status[(i, j)] = DECIDED_PRESENT
                B_hat[j, i] = c
            elif c < coef_lo and (reach_none[i, j] and reach_none[j, i]):
                status[(i, j)] = DECIDED_ABSENT
            elif c < coef_lo and reach_strong[j, i]:
                # j is clearly the ancestor, so there is no forward edge i -> j.
                status[(i, j)] = DECIDED_ABSENT
            elif c > coef_hi and not reach_strong[i, j]:
                # a coefficient is present but the intervention does not confirm i -> j.
                status[(i, j)] = UNDECIDED_ORDER
            else:
                status[(i, j)] = UNDECIDED_WEAK

    n_present = sum(1 for s in status.values() if s == DECIDED_PRESENT)
    n_absent = sum(1 for s in status.values() if s == DECIDED_ABSENT)
    n_undecided = sum(1 for s in status.values() if s.startswith("EDGE_UNDECIDED"))
    return dict(status={f"{k}->{j}": s for (k, j), s in status.items()},
                status_pairs=status, B_hat=B_hat, order=[int(x) for x in order],
                n_decided_present=int(n_present), n_decided_absent=int(n_absent),
                n_undecided=int(n_undecided))
