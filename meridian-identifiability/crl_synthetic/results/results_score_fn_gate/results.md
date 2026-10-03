# Score-function-difference gate — ceiling probe

- torch 2.4.0a0+f70bd71a48.nv24.06, numpy 1.24.4, scipy 1.13.1, sklearn 1.5.0
- device: cuda (NVIDIA A100-SXM4-80GB)
- d=10, N=50, max_lag=2, N_SHIFT=6, windows=256x64, seeds up to 12
- ceiling probe: no HRF, no observation noise
- identifying signal: group sparsity, over latent coordinates, of the cross-environment CONDITIONAL score difference grad_zt log p_a(z_t|past) - grad_zt log p_b(z_t|past), on unit-variance latents
- estimator: multi-level DSM, sigmas (0.05, 0.1, 0.2, 0.4), evaluated at sigma=0.05

## THE KEY READOUT — permutation-free score-difference sparsity

L1/L2 over the d per-coordinate norms of the score difference, each measured in that model's OWN coordinates. No MCC, no permutation. 1.0 = one coordinate carries everything; 3.16 = uniform across all 10.

**TRAINED ≈ TRUE → identified. TRAINED ≈ RANDOM → objective did not act. TRAINED < TRUE → DEGENERATE (out-sparsed reality; this is how the previous coefficient objective died).**

| condition | score L1/L2 TRUE | score L1/L2 TRAINED | score L1/L2 RANDOM | VAR R² TRUE | VAR R² TRAINED | VAR R² RANDOM |
|---|---|---|---|---|---|---|
| score_fn_laplace | 1.905 ± 0.000 | 2.973 ± 0.000 | 2.949 ± 0.000 | 0.645 | 0.727 | 0.871 |
| mse_laplace | 1.905 ± 0.000 | 2.890 ± 0.000 | 2.949 ± 0.000 | 0.645 | 0.863 | 0.871 |

### Score-estimator convergence (check BEFORE reading anything above)

DSM loss near 10.0 means the score net predicted nothing and every sparsity number above is vacuous. Well below means it converged. `|d|` is the raw magnitude of the score difference: if TRAINED `|d|` is far below TRUE `|d|`, the net simply never saw the environment difference.

| condition | DSM TRUE | DSM TRAINED | \|d\| TRUE | \|d\| TRAINED |
|---|---|---|---|---|
| score_fn_laplace | 7.357 | 3.033 | 27.926 | 9.025 |
| mse_laplace | 7.357 | 6.649 | 27.926 | 25.176 |

## Recovery metrics (mean ± std over seeds)

| condition | objective | data | n | MCC | subspace R² | recon r | score top-k hit | MCC touched | MCC untouched |
|---|---|---|---|---|---|---|---|---|---|
| score_fn_laplace | score_fn | laplace | 1 | 0.449 ± 0.000 | 0.635 ± 0.000 | 0.673 ± 0.000 | 0.67 ± 0.00 | 0.474 ± 0.000 | 0.411 ± 0.000 |
| mse_laplace | mse | laplace | 1 | 0.513 ± 0.000 | 1.000 ± 0.000 | 0.997 ± 0.000 | 0.67 ± 0.00 | 0.520 ± 0.000 | 0.504 ± 0.000 |

## Verdict

**score_fn + laplace — NOT IDENTIFIED — TRAINED sits at RANDOM. The objective did not act.** (TRUE 1.905, TRAINED 2.973, RANDOM 2.949)

**MCC vs the MSE floor (laplace): score_fn 0.449 ± 0.000, MSE 0.513 ± 0.000, DELTA -0.065** (pooled sd 0.000)

  - The score objective does not clear the rotationally-blind baseline by more than one pooled sd on this data. Whatever the absolute MCC is, that is not identification.

Legacy absolute gate, reported but NOT the verdict: score_fn+laplace MCC 0.449 ± 0.000 vs 0.4. See caveat 1 — at d=10 the MSE floor is already near 0.5, so this threshold cannot separate anything.

## Caveats — read before quoting these numbers

1. **The absolute MCC>0.4 gate is not informative at d=10.** A rotationally blind linear autoencoder scores around 0.5 here, purely from Hungarian matching on a 10x10 correlation matrix. The verdict above therefore uses the score-minus-MSE delta on identical data and the permutation-free TRUE/TRAINED/RANDOM comparison instead.
2. **The Gaussian arm is not a clean negative control.** The conditional score difference is sparse for Gaussian innovations too: with psi(eps) = -eps the difference at the shifted target is still exactly (B_b - B_a)·past, supported on the target coordinate alone. A Gaussian VAR with a sparse lagged shift is identifiable from this signal without any non-Gaussianity, so score_fn+gaussian passing is most likely CORRECT rather than a harness bug. What it would show is that the power comes from the mechanism/score structure, not from heavy tails.
3. **The Laplace prior is a confound.** The loss contains both an ICA-style non-Gaussian prior (LAMBDA_PRIOR) and the score-difference term (LAMBDA_SCORE), either of which can break rotation alone. This 2x2 cannot attribute the result to the score term. The disambiguating runs are LAMBDA_SCORE=0 (prior only) and LAMBDA_PRIOR=0 (score only); neither is in this sweep. If score_fn+gaussian and score_fn+laplace land in the same place, the prior is probably not the driver.
4. **The encoder gradient is an approximation.** The score net's dependence on the encoder is not differentiated through; only its evaluation points are. This is the standard bilevel approximation. Stage B and the inner loop keep the score estimate near its optimum, and the DSM losses are reported, but the gradient is not the exact total derivative and the optimum found may reflect that.
5. **The score-difference support is TARGET coordinates only.** DSM on z_t estimates grad_zt log p(z_t|past), whose environment difference is supported on the shifted TARGETS. Shifted SOURCES live in the score w.r.t. past, which is not estimated here. `score_topk_hit_rate` is scored against targets only; do not read it as full mechanism-support recovery.
6. **Coordinates the shift never touches are not pinned by the score term.** With N_SHIFT=6 the score-difference support is the shifted target set (printed per seed in the sanity block). Any rotation acting only on untouched coordinates leaves the penalty unchanged, so those coordinates are free under this term alone. Read the MCC touched/untouched split before the aggregate MCC.
7. **Ceiling probe only.** No HRF, no noise, linear instantaneous mixing, correctly specified lag order, and a score estimator fit on the same data it is evaluated on. A pass here is necessary, not sufficient.

## Figures

- `fig_score_diff_score_fn_laplace.png`
- `fig_true_trained_random_score_fn_laplace.png`
- `fig_score_diff_mse_laplace.png`
- `fig_true_trained_random_mse_laplace.png`
