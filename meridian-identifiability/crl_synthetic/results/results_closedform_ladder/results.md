# Closed-form mixing recovery under a pairing-degradation ladder

- torch 2.4.0a0+f70bd71a48.nv24.06, numpy 1.24.4, scipy 1.13.1, sklearn 1.5.0
- device: cuda (NVIDIA A100-SXM4-80GB)
- d=10, N=50, 10 intervention envs + 1 reference, windows=256x64

> **RHO=1 is synthetic-only** — it requires the same-innovation counterfactual pairing that real task-fMRI never provides. The scientific question is WHERE ON THE RHO LADDER recovery collapses, since real data sits near RHO=0.

## RHO ladder — the controlled interpolation

The additive RHO knob injects the innovation mismatch once and lets it decay; it is the clean interpolation between the verified RHO=1 ceiling and the independent end, but it does NOT let the mismatch compound through the VAR recursion, so it is **optimistic at low RHO**. Read it together with the independent_recursive rung below.

| RHO | cos mean | cos min | rank1 mean | rank1 max | CF MCC | MSE floor | cf-init MCC | rand-init MCC |
|---|---|---|---|---|---|---|---|---|
| 1.00 | 1.0000 | 1.0000 | 1.51e-08 | 2.43e-08 | 1.000 | 0.524 | 1.000 | 0.786 |
| 0.99 | 1.0000 | 1.0000 | 7.67e-02 | 1.36e-01 | 1.000 | 0.547 | 0.975 | 0.794 |
| 0.95 | 1.0000 | 0.9998 | 1.70e-01 | 2.99e-01 | 1.000 | 0.537 | 0.968 | 0.743 |
| 0.90 | 0.9998 | 0.9993 | 2.37e-01 | 4.13e-01 | 1.000 | 0.540 | 0.950 | 0.580 |
| 0.80 | 0.9994 | 0.9972 | 3.27e-01 | 5.58e-01 | 0.999 | 0.522 | 0.909 | 0.667 |
| 0.60 | 0.9975 | 0.9868 | 4.42e-01 | 7.28e-01 | 0.997 | 0.527 | 0.862 | 0.580 |
| 0.40 | 0.9800 | 0.8252 | 5.19e-01 | 8.21e-01 | 0.981 | 0.522 | 0.860 | 0.490 |
| 0.20 | 0.9655 | 0.6981 | 5.73e-01 | 8.58e-01 | 0.927 | 0.515 | 0.651 | 0.505 |
| 0.00 | 0.9605 | 0.6724 | 6.17e-01 | 8.87e-01 | 0.904 | 0.528 | 0.688 | 0.456 |

## INDEPENDENT_RECURSIVE — the transfer-relevant rung

Genuine two-independent-sessions construction: env_c re-simulated from scratch through its own shifted VAR with fresh innovations, so the mismatch COMPOUNDS through the recursion. This is the realistic task-fMRI analogue (same subject, different run) and the number to quote for transfer claims. No noise, no HRF, so it is directly comparable to the additive RHO=0 row above — the gap between them is exactly how optimistic the additive ladder is at the independent end.

| construction | cos mean | cos min | rank1 mean | rank1 max | CF MCC | MSE floor | cf-init MCC | rand-init MCC |
|---|---|---|---|---|---|---|---|---|
| independent_recursive | 0.7925 | 0.6311 | 6.93e-01 | 8.81e-01 | 0.766 | 0.542 | 0.627 | 0.470 |
| additive RHO=0 (optimistic) | 0.9605 | 0.6724 | 6.17e-01 | 8.87e-01 | 0.904 | 0.528 | 0.688 | 0.456 |

## Observation-noise ladder (RHO=1, HRF off)

| SNR | cos mean | cos min | rank1 mean | rank1 max | CF MCC | MSE floor |
|---|---|---|---|---|---|---|
| 20 | 1.0000 | 1.0000 | 2.18e-02 | 4.22e-02 | 0.999 | 0.535 |
| 10 | 1.0000 | 1.0000 | 4.35e-02 | 8.39e-02 | 0.997 | 0.484 |
| 5 | 1.0000 | 0.9999 | 8.64e-02 | 1.65e-01 | 0.989 | 0.553 |
| 2 | 0.9999 | 0.9995 | 2.08e-01 | 3.69e-01 | 0.941 | 0.526 |

## Other rungs (HRF, combined-realistic)

| rung | RHO | SNR | HRF | cos mean | rank1 mean | CF MCC | MSE floor |
|---|---|---|---|---|---|---|---|
| hrf | 1.00 | inf | True | 1.0000 | 1.52e-08 | 0.178 | 0.120 |
| realistic | 0.00 | 5 | True | 0.9654 | 5.46e-01 | 0.099 | 0.089 |

## Verdict

- **Additive ladder: closed-form MCC stays above the MSE floor (0.529) across the entire RHO sweep, including RHO=0.** Optimistic; read the recursive result before concluding anything about transfer.
- **INDEPENDENT_RECURSIVE (the transfer number): closed-form MCC 0.766 vs MSE floor 0.529** (clears the floor); mean |cos| 0.793, rank-1 ratio 0.693. cf-init trained 0.627, random-init trained 0.470. For comparison the additive RHO=0 closed-form MCC is 0.904 — the gap is how much the additive ladder overstates recovery at the independent end.
  - The closed form clears the floor even under genuine independent pairing. Surprising for transfer — scrutinize the rank-1 ratio (a high value means the recovered column is not a trustworthy mixing estimate).
- At RHO=1: closed-form MCC 1.000, random-init trained 0.786, closed-form-init trained 1.000.
  - Closed-form init clears the local minimum that random init fell into: **the Option B plateau was an initialization problem.**

## Caveats

1. **RHO=1 is not a real-data result.** It needs same-innovation counterfactual pairs. The transfer-relevant rows are the low-RHO end.
2. **The HRF is linear and per-coordinate**, so it does not rotate the mixing: it leaves |cos| and rank-1 intact at RHO=1 and only blurs the latent (lowering MCC via imperfect deconvolution). Do not read the HRF MCC drop as a mixing-recovery failure — the cos column shows mixing recovery is untouched.
3. **Ground truth is the NEURAL latent** (pre-HRF). Under the HRF toggle the recoverable quantity is the convolved latent, so MCC understates mixing recovery; |cos| is the clean mixing metric.
4. **SNR is amplitude** (noise std = signal std / SNR), independent per env.

## Figures

- `fig_rho_ladder_mcc.png`
- `fig_rho_ladder_rank1.png`
