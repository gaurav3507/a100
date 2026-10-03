# Mechanism-gate ceiling probe

- torch 2.4.0a0+f70bd71a48.nv24.06, numpy 1.24.4, scipy 1.13.1, sklearn 1.5.0
- device: cuda (NVIDIA A100-SXM4-80GB)
- d=10, N=50, max_lag=2, windows=256x64, seeds up to 12, gate MCC>0.4
- ceiling probe: no HRF, no observation noise
- identifying signal: sparsity of the cross-environment latent VAR coefficient difference (B_a - B_b), on unit-variance latents

## Results (mean ± std over seeds)

| condition | objective | data | n | MCC | subspace R² | recon r | mech Gini | top-k hit | on/off contrast |
|---|---|---|---|---|---|---|---|---|---|
| score_laplace | score | laplace | 1 | 0.526 ± 0.000 | 0.762 ± 0.000 | 0.968 ± 0.000 | 0.917 ± 0.000 | 0.00 ± 0.00 | 0.5 ± 0.0 |
| mse_laplace | mse | laplace | 1 | 0.542 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.608 ± 0.000 | 0.00 ± 0.00 | 4.1 ± 0.0 |

### MCC split by shift support

The mechanism term can only pin coordinates the shift touches. If the mechanism term is doing the work, `MCC touched` should be well above `MCC untouched`.

| condition | MCC (shift-touched coords) | MCC (untouched coords) |
|---|---|---|
| score_laplace | 0.634 ± 0.000 | 0.479 ± 0.000 |
| mse_laplace | 0.545 ± 0.000 | 0.541 ± 0.000 |

## Verdict

**GATE — score + Laplace: PASS** (MCC = 0.526 ± 0.000, threshold 0.4)

**WARNING — HARNESS SUSPECT: MSE + Laplace PASSED the gate (MCC = 0.542 ± 0.000 > 0.4). A pointwise linear MSE autoencoder is rotationally blind and must not identify latents. Do not trust the score result until this is explained.**

## Caveats — read before quoting these numbers

1. **The Gaussian arm is not a clean negative control under this design.** Mechanism-difference sparsity is a second-order signal: a Gaussian VAR with a sparse lagged shift is identifiable up to rotation with no non-Gaussianity at all. Under the previous marginal-score design, Gaussian data was expected to fail; here score+Gaussian passing is most likely the CORRECT result and not a bug. What it would show is that the objective's power comes from the mechanism term, not from heavy tails.
2. **The Laplace prior is a confound in the Laplace arm.** The loss contains both an ICA-style non-Gaussian prior and the mechanism-sparsity term, either of which can break rotation alone. This sweep cannot attribute the score+Laplace result to the mechanism term. The disambiguating runs are LAMBDA_MECH=0 (prior only) and LAMBDA_PRIOR=0 (mechanism only); neither is in this 2x2.
3. **Ceiling probe only.** No HRF, no noise, linear instantaneous mixing, correctly specified lag order. A pass here is a necessary condition, not evidence the objective survives realistic data.
4. `scale_gaming_check` in each seed JSON reports the spread of latent coordinate standard deviations. The penalty is computed on in-graph unit-variance latents, so scale cannot buy sparsity, but a large `latent_sd_ratio` still means the flow is doing something worth a look.
5. **The mechanism term cannot identify every coordinate.** With N_SHIFT=2 the shift touches only ~2-4 of the 10 coordinates. Any rotation acting only on the untouched coordinates leaves (B_a - B_b) equally sparse, so those coordinates are not pinned by the mechanism term at all. This caps the achievable MCC well below 1 and puts the 0.4 gate uncomfortably close to the arithmetic: roughly (n_touched * ~0.9 + n_untouched * ~0.2) / 10. Read the MCC-split table above before reading the gate — a pass driven by the untouched coordinates is the Laplace prior's doing, not the mechanism term's.

## Figures

- `fig_mechanism_score_laplace.png`
- `fig_mechanism_heatmap_score_laplace.png`
- `fig_mechanism_mse_laplace.png`
- `fig_mechanism_heatmap_mse_laplace.png`
