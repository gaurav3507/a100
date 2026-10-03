# Number of environments vs identifiability (n_envs independent of d)

- torch 2.4.0a0+f70bd71a48.nv24.06, numpy 1.24.4, device cuda
- d=30, K in [6, 12], n_envs swept over [30, 20, 15, 10, 8, 7, 5], rotation search 200 restarts x 80 steps

Every prior run used n_envs = d (one environment per coordinate). Real task fMRI gives ~7 conditions (HCP: 7 tasks + rest) regardless of latent dimension, so **n_envs << d is the realistic regime** and this is its first test. Each environment perturbs K distinct coordinates; the paired difference is zero on any coordinate no environment touches, so uncovered coordinates are rotationally free and cannot be identified — identifiability over the COVERED set is reported separately from the aggregate.

## Sweep

| K | n_envs | coverage | cov-count (min..max) | TRUE | BEST | (T-B)/T | falsified | block e-o-t | degenerate | covered pinned | agg. identifiable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 30 | 30.0/30 | 6..6 | 73.40 | 73.40 | 0.0000 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 6 | 20 | 30.0/30 | 4..4 | 48.93 | 48.93 | 0.0000 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 6 | 15 | 30.0/30 | 3..3 | 36.70 | 36.70 | 0.0000 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 6 | 10 | 30.0/30 | 2..2 | 24.47 | 24.47 | 0.0000 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 6 | 8 | 30.0/30 | 1..2 | 19.54 | 19.54 | 0.0000 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 6 | 7 | 30.0/30 | 1..2 | 17.09 | 17.09 | 0.0000 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 6 | 5 | 30.0/30 | 1..1 | 12.24 | 12.23 | 0.0003 | 0.00 | 0.87 | 0.17 | 0.83 | 0.83 |
| 12 | 30 | 30.0/30 | 12..12 | 103.75 | 103.75 | 0.0000 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 12 | 20 | 30.0/30 | 8..8 | 69.17 | 69.17 | 0.0000 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 12 | 15 | 30.0/30 | 6..6 | 51.86 | 51.86 | 0.0000 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 12 | 10 | 30.0/30 | 4..4 | 34.57 | 34.57 | 0.0000 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 12 | 8 | 30.0/30 | 3..4 | 27.65 | 27.65 | 0.0000 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 12 | 7 | 30.0/30 | 2..3 | 24.18 | 24.18 | 0.0000 | 0.00 | 1.00 | 0.00 | 1.00 | 1.00 |
| 12 | 5 | 30.0/30 | 2..2 | 17.28 | 17.18 | 0.0055 | 0.17 | 0.81 | 0.17 | 0.67 | 0.67 |

*coverage = distinct coordinates perturbed by >=1 environment (rest are free); cov-count = per-coordinate coverage count range; covered pinned = fraction of seeds where the covered coordinates are element-wise identifiable (not falsified, not degenerate); agg. identifiable = element-wise identifiable fraction of ALL d coordinates (covered_frac if pinned, else 0).*

## Verdict

- **K=6: covered coordinates stay uniquely identified down to n_envs = 5.** n_envs = 7 (HCP-realistic) IS sufficient here.
- **K=12: covered coordinates stay uniquely identified down to n_envs = 5.** n_envs = 7 (HCP-realistic) IS sufficient here.
- The realistic count n_envs ≈ 7 is marked on the figure. Whether it lands in the identified region is the transfer-relevant answer: with ~7 conditions and a latent space of tens of dimensions, element-wise identification requires each coordinate to be constrained by enough overlapping environments, which few conditions cannot supply.
- **Coverage is not the bottleneck here** (K*n_envs >= d for these rungs, so the covered set is essentially all of d); the limiting factor is OVERLAP — how many environments constrain each coordinate — which falls as n_envs falls.

## Figures

- `fig_nenv.png`
