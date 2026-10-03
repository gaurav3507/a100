# E5: split-control gate (Tier 2, Experiment B)

For each dataset and gate seed the control cells are split once, 50/50 at random
(`SeedSequence([5050, seed])`): Half A builds every environment's null, with the
unchanged size matching, and Half B is the fixed reference, never resampled, so the
observed statistic is `T_e = lambda_max(Omega_e - Omega_B)`. Everything else is
unchanged (precision readout, disjoint split inside Half A when `n_e <= n_A/2`, BH
within family at q=0.05, control-fit projection on all control cells), and no
recovery is run. Every family is screened under both the shared-reference design
(`corrected_disjoint`, which reproduces E3) and the split design (`split_control`)
on identical inputs and per-environment null streams.

Produced by `experiments/e5_split_control.py` (launcher:
`experiments/run_tier2_a100.sh`); the gate switch is `Y_null` in
`precision_readout.detect_with_pvalues`, and `experiments/tier2_screen.py
--split-control` screens one dataset and seed.

## JSON keys

`e5_real_panel.json` (layout of `results/e3_stability/<name>.json`, five seeds)
- `datasets.<name>.shared_vs_split.per_seed[]`: `shared` and `split` raw_count and bh_count,
  `jaccard_bh` (None when both BH sets are empty), plus the same for `negative_controls`.
- `datasets.<name>.shared_vs_split.summary`: seed means.
- `datasets.<name>.perturbation_stability`, `negative_control_stability`: per-design fraction
  mean, SD and per-seed values.
- `datasets.<name>.per_seed[].split`: half sizes and a SHA-256 of the Half A indices.
- `datasets.<name>.per_seed[].shared_reproduces_e3_stability`: whether the shared design
  reproduced the committed E3 raw and BH sets exactly.

`e5_rpe1_confound.json` (layout of `results/e3/e3_rpe1_confound_check.json`, seed 0)
- `perturbation_screen`, `random_control_screen`, `structured_control_screen`: both designs.
- `shared_vs_split.<perturbations|random_controls|structured_controls>`.
- `shared_reproduces_e3.checks`: reproduction of the committed E3 decisions per family.

`e5_poscontrol.json` and `e5_poscontrol_obs400.json` (layout of
`results/e3_poscontrol/poscontrol_final.json`; 200 and 400 observational cells)
- `n_obs`, `smoke_snr8`, `dose_response[]`: planted and null-draw screens under both designs.
- `shared_vs_split.<ratio>`: planted and null-draw counts under both designs.
- `split_geometry_note`: the geometry of that variant (below).
- `shared_reproduces_e3`: the 200-cell variant only; the 400-cell variant has a different
  observational draw.

## Positive-control geometry

With the E3 observational draw of 200 cells each half has 100, every 200-cell
environment is larger than Half A, and the split null falls back to 200 draws with
replacement from 100 rows while the observed statistic uses a 100-cell reference, so
that variant is not a like-for-like null. The 400-cell variant appends 200 more
observational latents from the same SCM (the first 200 unchanged) so each half has
200 cells, matching every environment; its shared design then uses the disjoint null
and is no longer the E3 construction. The signal scale is fixed on the original 200
observational cells in both, so the ratios are the same.
