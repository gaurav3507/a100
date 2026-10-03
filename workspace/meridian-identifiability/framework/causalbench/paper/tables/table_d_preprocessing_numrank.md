# table_d_preprocessing_numrank

```
generated  : 2026-08-11T11:12:17Z
generator  : 92_preprocessing_table.py @ 9f2b1d5
source dir : causalbench/results/ranktest/descriptive/preprocessing_sweep
artefacts  : 6 CURRENT, 0 skipped
  source   : 2026-08-11T09-36-58Z__k562.json   meta.git_commit=9f2b1d5
  source   : 2026-08-11T09-37-10Z__rpe1.json   meta.git_commit=9f2b1d5
  source   : 2026-08-11T09-39-00Z__norman.json   meta.git_commit=9f2b1d5
  source   : 2026-08-11T10-04-17Z__frangieh_coculture.json   meta.git_commit=9f2b1d5
  source   : 2026-08-11T10-23-05Z__frangieh_control.json   meta.git_commit=9f2b1d5
  source   : 2026-08-11T10-46-03Z__frangieh_ifng.json   meta.git_commit=9f2b1d5
```

| dataset | raw | standardise | log1p_std | rank_int |
|---|---|---|---|---|
| frangieh_coculture | 1999 | 1999 | n/a | 1999 |
| frangieh_control | 1999 | 1999 | n/a | 1999 |
| frangieh_ifng | 1999 | 1999 | n/a | 1999 |
| k562 | 1158 | 1158 | n/a | 1158 |
| norman | 1999 | 1999 | n/a | 1999 |
| rpe1 | 651 | 651 | n/a | 651 |

> Arm coherence is decided per dataset from a runtime probe of the control matrix, not assumed. k562: loader returns already_transformed; coherent arms raw, standardise, rank_int; log1p_std omitted (input is already log-like, so log1p_std would be a double log; emitted as null rather than reported as a preprocessing choice) | rpe1: loader returns already_transformed; coherent arms raw, standardise, rank_int; log1p_std omitted (input is already log-like, so log1p_std would be a double log; emitted as null rather than reported as a preprocessing choice) | norman: loader returns already_transformed; coherent arms raw, standardise, rank_int; log1p_std omitted (input is already log-like, so log1p_std would be a double log; emitted as null rather than reported as a preprocessing choice) | frangieh_coculture: loader returns already_transformed; coherent arms raw, standardise, rank_int; log1p_std omitted (input is already log-like, so log1p_std would be a double log; emitted as null rather than reported as a preprocessing choice) | frangieh_control: loader returns already_transformed; coherent arms raw, standardise, rank_int; log1p_std omitted (input is already log-like, so log1p_std would be a double log; emitted as null rather than reported as a preprocessing choice) | frangieh_ifng: loader returns already_transformed; coherent arms raw, standardise, rank_int; log1p_std omitted (input is already log-like, so log1p_std would be a double log; emitted as null rather than reported as a preprocessing choice)
