# Multi-env one-intervention-per-coordinate gate

- torch 2.4.0a0+f70bd71a48.nv24.06, numpy 1.24.4, scipy 1.13.1, sklearn 1.5.0
- device: cuda (NVIDIA A100-SXM4-80GB)
- d=10, N=50, max_lag=2, 10 intervention envs + 1 reference, windows=256x64, seeds up to 12
- LAMBDA_PAIR=1.0, LAMBDA_PRIOR=0.0 (prior off => result attributable to the pairing alone)
- ceiling probe: no HRF, no observation noise, exact counterfactual pairing
- signal: sum_c L1/L2 of the per-env paired difference; each term >= 1 by Cauchy-Schwarz, so the objective >= d = 10, uniquely met at the truth. No sparser-than-truth escape exists.

## THE KEY READOUT — permutation-free summed paired-difference sparsity

Summed L1/L2 in each model's OWN coordinates. TRUE = d = 10 (the global floor); TRAINED cannot go below it. **TRAINED ~ TRUE -> element-wise identified. TRAINED ~ RANDOM -> the objective did not act.**

| condition | summed L1/L2 TRUE | TRAINED | RANDOM | VAR R2 TRUE | VAR R2 TRAINED | VAR R2 RANDOM |
|---|---|---|---|---|---|---|
| multienv_laplace | 10.006 +/- 0.000 | 11.278 +/- 0.000 | 26.188 +/- 0.000 | 0.628 | 0.476 | 0.812 |
| mse_laplace | 10.006 +/- 0.000 | 26.108 +/- 0.000 | 26.188 +/- 0.000 | 0.628 | 0.820 | 0.812 |

## Recovery metrics (mean +/- std over seeds)

| condition | objective | data | n | MCC | subspace R2 | recon r | per-env localization hit |
|---|---|---|---|---|---|---|---|
| multienv_laplace | multienv | laplace | 1 | 0.530 +/- 0.000 | 0.521 +/- 0.000 | 0.993 +/- 0.000 | 0.40 +/- 0.00 |
| mse_laplace | mse | laplace | 1 | 0.554 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 0.50 +/- 0.00 |

## Verdict

**multienv + laplace - IDENTIFIED - TRAINED matches TRUE (element-wise recovery).** (TRUE 10.006, TRAINED 11.278, RANDOM 26.188)

  - reference VAR R2 dropped from 0.628 (TRUE) to 0.476 (TRAINED): the encoder degraded the dynamics.

**MCC vs the MSE floor (laplace): multienv 0.530 +/- 0.000, MSE 0.554 +/- 0.000, DELTA -0.023** (pooled sd 0.000)

  - Does NOT clear the MSE floor by a pooled sd. Not element-wise identification, whatever the absolute MCC.

## Caveats — read before quoting these numbers

1. **TRAINED cannot beat TRUE, by construction.** Each per-env term is L1/L2 of a nonnegative vector, hence >= 1, so the sum >= d and the truth is a global minimum. The degenerate out-sparsing that killed candidates 0 and 2 cannot happen. The only failure is TRAINED failing to reach TRUE, which the exact gradient is meant to prevent.
2. **The pairing is a free counterfactual.** env_c shares the reference's innovations AND history, so the difference is exactly 1-D on coordinate c and content is bit-identical elsewhere. Real data does not hand you paired counterfactuals; a pass here says nothing about obtaining them.
3. **The Gaussian arm is not a negative control.** The paired difference (B_env - B_ref) @ past is structural, independent of the innovation distribution, so both arms should behave alike. A difference implicates the flow, not identifiability.
4. **Weak interventions (small alpha) weaken a coordinate's signal.** If some env's alpha is small its paired difference is faint and that coordinate may not be pinned. The sanity block prints per-env alpha; oracle check 5b enforces 1-D-ness.
5. **Ceiling probe only.** No HRF, no noise, linear instantaneous mixing, correctly specified lag order. Necessary, not sufficient.

## Figures

- `fig_true_trained_random_multienv_laplace.png`
- `fig_profile_matrix_multienv_laplace.png`
- `fig_true_trained_random_mse_laplace.png`
- `fig_profile_matrix_mse_laplace.png`
