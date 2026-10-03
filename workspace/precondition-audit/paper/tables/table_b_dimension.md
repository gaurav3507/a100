# table_b_dimension

```
generated   : 2026-08-12T08:27:45Z
generator   : 88_make_descriptive_tables.py @ 11e9846
source dir  : results/descriptives
artefacts   : 8 CURRENT, 0 skipped
  source    : 2026-08-11T06-12-06Z__k562.json   meta.git_commit=8181090
  source    : 2026-08-11T06-12-20Z__rpe1.json   meta.git_commit=8181090
  source    : 2026-08-11T06-20-34Z__norman.json   meta.git_commit=8181090
  source    : 2026-08-11T06-53-37Z__frangieh_coculture.json   meta.git_commit=8181090
  source    : 2026-08-11T07-13-08Z__frangieh_control.json   meta.git_commit=8181090
  source    : 2026-08-11T07-47-30Z__frangieh_ifng.json   meta.git_commit=8181090
  source    : 2026-08-11T07-47-38Z__hcp.json   meta.git_commit=8181090
  source    : 2026-08-11T07-47-45Z__abide.json   meta.git_commit=8181090
```

| dataset | spec_cap | n_genes | n_control_available | n_control_used | p_over_n | n_lt_p_flag | participation_ratio | effective_rank_exp_spectral_entropy | n_comp_80pct | n_comp_90pct | n_comp_95pct | top_eigenvalue_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abide | n/a | n/a | n/a | n/a | n/a | n/a | n/a - no control condition | n/a | n/a | n/a | n/a | n/a |
| frangieh_coculture | 2000 | 23712 | 5707 | 2000 | 11.86 | RANK-DEFICIENT | 415.4 | 1472 | 1192 | 1514 | 1721 | 0.04114 |
| frangieh_coculture | 8000 | 23712 | 5707 | 5707 | 4.155 | RANK-DEFICIENT | 491.9 | 3243 | 2566 | 3469 | 4180 | 0.04022 |
| frangieh_control | 2000 | 23712 | 3770 | 2000 | 11.86 | RANK-DEFICIENT | 419.2 | 1471 | 1198 | 1519 | 1725 | 0.04051 |
| frangieh_control | 8000 | 23712 | 3770 | 3770 | 6.29 | RANK-DEFICIENT | 453.3 | 2421 | 1959 | 2581 | 3028 | 0.04104 |
| frangieh_ifng | 2000 | 23712 | 6144 | 2000 | 11.86 | RANK-DEFICIENT | 435.7 | 1484 | 1190 | 1511 | 1719 | 0.03983 |
| frangieh_ifng | 8000 | 23712 | 6144 | 6144 | 3.859 | RANK-DEFICIENT | 528 | 3427 | 2662 | 3616 | 4383 | 0.03878 |
| hcp | n/a | n/a | n/a | n/a | n/a | n/a | n/a - no control condition | n/a | n/a | n/a | n/a | n/a |
| k562 | 2000 | 1158 | 10691 | 2000 | 0.579 |  | 455.5 | 722.8 | 492 | 663 | 800 | 0.02073 |
| k562 | 8000 | 1158 | 10691 | 8000 | 0.145 |  | 552.5 | 885.8 | 646 | 822 | 942 | 0.02055 |
| norman | 2000 | 5000 | 8907 | 2000 | 2.5 | RANK-DEFICIENT | 224.1 | 538.6 | 389 | 551 | 700 | 0.03484 |
| norman | 8000 | 5000 | 8907 | 8000 | 0.625 |  | 248.1 | 655.7 | 480 | 658 | 826 | 0.03608 |
| rpe1 | 2000 | 651 | 11485 | 2000 | 0.326 |  | 234.5 | 436.4 | 314 | 418 | 496 | 0.04108 |
| rpe1 | 8000 | 651 | 11485 | 8000 | 0.081 |  | 261.6 | 490.4 | 368 | 469 | 538 | 0.04031 |

> Dimension estimates are sample-size-dependent lower bounds; compare only at matched n_control_used. Where n_control_used < n_genes the sample covariance is rank-deficient (rank bound n-1) and both effective rank and the variance-threshold counts are contaminated by sampling noise in the trailing eigenvalues: at the 8000 cap, effective rank is 0.56-0.64n and the 95pct count 0.71-0.80n across the three arms. Participation ratio is dominated by the leading eigenvalues and is the estimator comparable across datasets.
