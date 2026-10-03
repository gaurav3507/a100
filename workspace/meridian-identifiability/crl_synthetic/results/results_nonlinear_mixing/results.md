# Nonlinear mixing: does anything survive when x = A f(z)?

- torch 2.4.0a0+f70bd71a48.nv24.06, numpy 1.26.4, device cuda
- MIXING_MODE = nonlinear, strengths [0.0, 0.25, 0.5, 1.0], d=10, N=50

Reuses run_closedform_ladder.py wholesale; the ONLY change is x = A @ f(z) with f a fixed invertible nonlinear map. Strength 0 = f identity = the linear file exactly (the control). The MSE floor is re-measured on the nonlinear data at every rung.

## rho1

| strength | cos mean | rank1 mean | CF-MCC | CSP-MCC | SHUF-MCC | MSE floor | flow rand | flow cf-init | obj true<pert |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 1.000 | 1.53e-08 | 1.000 | 0.914 | 0.796 | 0.527 | 0.787 | 1.000 | 1.00 |
| 0.25 | 1.000 | 7.68e-03 | 1.000 | 0.914 | 0.801 | 0.514 | 0.795 | 1.000 | 1.00 |
| 0.50 | 1.000 | 1.54e-02 | 1.000 | 0.913 | 0.803 | 0.517 | 0.801 | 1.000 | 1.00 |
| 1.00 | 0.999 | 3.06e-02 | 0.999 | 0.912 | 0.805 | 0.523 | 0.844 | 1.000 | 1.00 |

Per-seed CF-MCC (bimodality check):

- strength 0.00 (mean 1.000 ± 0.000, n=12): [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]
- strength 0.25 (mean 1.000 ± 0.000, n=12): [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]
- strength 0.50 (mean 1.000 ± 0.000, n=12): [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.999, 1.000, 1.000, 1.000, 1.000]
- strength 1.00 (mean 0.999 ± 0.001, n=12): [0.999, 0.998, 1.000, 1.000, 0.999, 1.000, 0.999, 0.998, 0.999, 0.999, 0.998, 0.999]

## independent_recursive

| strength | cos mean | rank1 mean | CF-MCC | CSP-MCC | SHUF-MCC | MSE floor | flow rand | flow cf-init | obj true<pert |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.829 | 6.92e-01 | 0.815 | 0.848 | 0.820 | 0.518 | 0.457 | 0.555 | 1.00 |
| 0.25 | 0.829 | 6.92e-01 | 0.817 | 0.848 | 0.821 | 0.534 | 0.438 | 0.521 | 1.00 |
| 0.50 | 0.829 | 6.92e-01 | 0.818 | 0.848 | 0.822 | 0.524 | 0.439 | 0.575 | 1.00 |
| 1.00 | 0.828 | 6.93e-01 | 0.819 | 0.848 | 0.822 | 0.527 | 0.455 | 0.541 | 1.00 |

Per-seed CF-MCC (bimodality check):

- strength 0.00 (mean 0.815 ± 0.218, n=12): [0.391, 0.953, 0.953, 0.944, 0.885, 0.915, 0.412, 0.958, 0.957, 0.942, 0.523, 0.949]
- strength 0.25 (mean 0.817 ± 0.216, n=12): [0.397, 0.953, 0.952, 0.944, 0.888, 0.915, 0.419, 0.959, 0.957, 0.942, 0.523, 0.949]
- strength 0.50 (mean 0.818 ± 0.214, n=12): [0.403, 0.953, 0.952, 0.944, 0.890, 0.915, 0.425, 0.959, 0.957, 0.942, 0.523, 0.949]
- strength 1.00 (mean 0.819 ± 0.212, n=12): [0.413, 0.952, 0.950, 0.943, 0.894, 0.915, 0.431, 0.958, 0.956, 0.941, 0.522, 0.948]

## IS THE PAIRING DOING ANY WORK?

CSP-MCC uses a between-condition covariance difference with NO pairing (the Common Spatial Subspace Decomposition idea, Neuroimage 1999). SHUF-MCC is the paired closed form run on time-shuffled env data — pairing destroyed, marginals and covariances preserved. If CSP ~ CF, a 1999 second-order technique already does the job; if SHUF ~ CF, the pairing itself contributes nothing at that rung.

| condition | strength | CF-MCC | CSP-MCC | SHUF-MCC | CF-CSP | CF-SHUF |
|---|---|---|---|---|---|---|
| rho1 | 0.00 | 1.000 | 0.914 | 0.796 | +0.086 | +0.204 |
| rho1 | 0.25 | 1.000 | 0.914 | 0.801 | +0.086 | +0.199 |
| rho1 | 0.50 | 1.000 | 0.913 | 0.803 | +0.086 | +0.197 |
| rho1 | 1.00 | 0.999 | 0.912 | 0.805 | +0.087 | +0.194 |
| independent_recursive | 0.00 | 0.815 | 0.848 | 0.820 | -0.033 | -0.005 |
| independent_recursive | 0.25 | 0.817 | 0.848 | 0.821 | -0.032 | -0.004 |
| independent_recursive | 0.50 | 0.818 | 0.848 | 0.822 | -0.031 | -0.004 |
| independent_recursive | 1.00 | 0.819 | 0.848 | 0.822 | -0.029 | -0.004 |

- **rho1, strength 1.0: CF 0.999 exceeds CSP 0.912 by 0.087** (CSP did not clearly collapse). Suggestive that pairing helps here, but not the clean beyond-second-order signature.
  - pairing-necessity (direct): CF 0.999 vs SHUF 0.805 (delta +0.194). Shuffling degrades the estimate -> the pairing carries real signal here.
- **independent_recursive, strength 1.0: the paired machinery adds NOTHING over a covariance-difference method (CSP 0.848 vs CF 0.819).** Under nonlinear mixing the estimator is behaving as a second-order method; the contribution of this line of work is therefore the identifiability ANALYSIS, not the estimator.
  - pairing-necessity (direct): CF 0.819 vs SHUF 0.822 (delta -0.004). Shuffling barely changes the estimate -> the pairing is NOT doing the work here.

## Controls and verdict

- **s=0 reproduction (rho1): OK** — closed-form MCC 1.000, |cos| 1.000, rank1 1.5e-08. Matches the linear file.
- **s=0 reproduction (independent_recursive): BROKEN** — closed-form MCC 0.815, |cos| 0.829, rank1 6.9e-01. Does NOT match the linear results — the refactor changed something.
- **(a) rho1: the closed form never fully collapses in the tested range** — unexpected; check whether the nonlinearity is strong enough (cos should fall well below 1 at strength 1).
- **(a) independent_recursive: the closed form never fully collapses in the tested range** — unexpected; check whether the nonlinearity is strong enough (cos should fall well below 1 at strength 1).
- **(b) rho1 at strength 1.0: flow CLEARS the MSE floor** — best flow (cfinit) MCC 1.000 ± 0.000 vs floor 0.523 ± 0.031 (pooled sd 0.031). Something survives the nonlinear CRL setting.
- **(b) independent_recursive at strength 1.0: flow does NOT clear the MSE floor** — best flow (cfinit) MCC 0.541 ± 0.132 vs floor 0.527 ± 0.028 (pooled sd 0.135). In the actual (nonlinear) CRL setting the paired objective does not beat a rotationally-blind baseline — this line of attack does not transfer.

## Caveats

1. **`obj true<pert` is a WEAK probe.** It checks whether the paired objective at the true latents beats a few random invertible perturbations; it does NOT optimize over the ambiguity group like the linear rotation search, and it is mixing-independent (a property of the objective and the latents). A pass is suggestive, a fail is a real disproof.
2. **|cos| is measured against the LINEAR (s=0) column A_eff[:,c]** — the object the closed form assumes exists. Under a nonlinear f there is no such column, so a falling |cos| is exactly the collapse being quantified, not a metric artifact.
3. **The MSE floor is re-measured per rung on the nonlinear data** and differs from the linear case; do not compare flow MCC to the linear floor.

## Figures

- `fig_nl_mcc.png`
- `fig_nl_estimator.png`
