# E4: second backbone (Tier 2, Experiment A)

Backbone C, the second backbone, recovers the unmixing by joint approximate
diagonalization (FFDiag) of all precision differences `Delta_e = Prec(Y_e) - Prec(Y_0)`,
with rows matched to the known targets, and its gate reads `||Delta_e||_F` against the
same size-matched null and BH. It must first pass a population check declared before
it was run (population MCC at the E2 reference point: mean >= 0.99 and every seed >=
0.98); only then does it run the four E2 arms and the E2c starvation series (10 seeds,
B=500) and, gate only, the K562 / RPE1 / Norman panel (five gate seeds). The
covariance readout (`lambda_max(Cov(Y_0) - Cov(Y_e))`, first run as "Backbone B") is
kept as a negative control through the same experiments; it was relabelled after the
preview showed its recovery rule reaches MCC 0.898 even with exact population
covariances, and that decomposition is its headline.

Produced by `experiments/e4_second_backbone.py` (`--backbone C` default, `--backbone
covneg` for the negative control); launcher `experiments/run_tier2_a100.sh`.

## Files

| file | contents |
|---|---|
| `e4_population_check.json` | Backbone C population MCC per seed, the declared rule, `passed` |
| `e4_attribution.json` | Backbone C on the five e2b attribution cases (schema of `results/e2b/attribution_report.json`); run on the Mac |
| `e4_calibration_report.json` | Backbone C, E2 arms (schema of `results/e2/e2_calibration_report.json`) |
| `e4_starvation_report.json` | Backbone C, E2c series (schema of `results/e2c/starvation_report.json`) |
| `e4_real_panel.json` | Backbone C gate on the real panel |
| `e4_negctrl_calibration_report.json` | negative control, E2 arms; `headline` first |
| `e4_negctrl_starvation_report.json` | negative control, E2c series |
| `e4_negctrl_real_panel.json` | negative control gate on the real panel |

## The Backbone C gate statistic is two-sided

`||Delta_e||_F` reacts to precision increases and decreases alike, whereas the
precision gate's `lambda_max(Prec(Y_e) - Prec(Y_0))` reacts only to a precision
increase (a variance reduction). The one-sidedness result for `lambda_max` (Prop 1:
variance inflation, such as a uniform gain above 1, is invisible to the screen) therefore
does not carry over to Backbone C: under `||.||_F` inflation is detectable.
`e4_attribution.json` measures this on the e2b cases (Mac, 10 seeds, B=500):
`uniform_gain_1.4` is detected on 10/10 seeds (0/10 under `lambda_max`), and the
unchanged subspace attributor then returns MECHANISM_SUPPORTED on all 10, because a
scalar gain preserves the signal subspace (the Prop 3 blind spot). The two-sided
statistic therefore turns the precision screen's safe miss into a false mechanism
attribution for variance inflation. The shrink case (`uniform_gain_0.7`) stays
UNATTRIBUTED under both statistics, through the lower-SNR side channel that
inflation does not open.

## JSON keys

Calibration reports
- `role`: "second backbone" or "negative control".
- `headline` (negative control): population versus sample MCC of the covariance rule.
- `population_check` (Backbone C): the check this run depended on.
- `arms.<arm>.levels[]`, `gate_behavior`, `crossover_*`, `silent_failure`: E2 classification, unchanged.
- `status`, `central_claim`: E2 meaning; a FAIL is a finding, not a script error.
- `precision_same_code`, `comparison`: the precision readout on the same code in the same job.
- `population_decomposition`: population versus sample MCC for the JAD, covariance and precision rules.
- `preregistered_expectation`: the three Tier 2 expectations and whether each held.

`e4_population_check.json`
- `per_seed[].population_mcc`, `offdiag_ratio_at_solution` versus `offdiag_ratio_at_truth`
  (solution below truth means the truth is not the exact joint-diagonal optimum, a model
  property, not an optimizer failure), `n_iter`, `converged`.

`e4_attribution.json`
- `summary.<case>`: detected rate and verdict rates in the e2b layout; `expected` holds the
  Frobenius expectation, `expected_precision_e2b` the original e2b one.
- `comparison_to_precision_e2b.per_case`: detected and UNATTRIBUTED rates under both
  statistics, and verdict agreement on environments both screens detect.
- `two_sided_note`, `rows_note`, `population_check`.

Starvation reports
- `aggregate.levels[].random_subset_control`: MCC mean and seed SD, gate cap, verdict counts.
- `aggregate.crossover_gate_first_restricts`, `crossover_random_subset_mcc_below_0p90`,
  `gate_restricts_at_or_before_mcc_crossover`; `precision_reference` for comparison.

Real-panel reports
- `datasets.<K562|RPE1|Norman>.n_environments`, `raw_count_per_seed`, `bh_count_per_seed`.
- `datasets.<name>.bh_selected_ids_per_seed`: BH-selected environment ids for Jaccard.
- `datasets.<name>.per_seed[].jaccard_bh_vs_precision`: against `results/e3_decisions` (seed 0)
  and `results/e3_stability` (same seed).
- `datasets.<name>.negative_control_*_count_per_seed`: the 50 random pure-control environments.
- `datasets.<name>.per_seed[].per_perturbation`: per-environment signal, threshold, p-value, decisions.
