# iLCS baseline results

The authoritative multi-seed result is `aggregate.json`, built from complete runs
under `seed0/`, `seed1/`, and `seed2/`. Each aggregate entry is
`[mean, standard deviation, number of seeds]`.

The top-level dataset JSON files predate the multi-seed layout and are retained as
historical outputs. Use the seed directories and aggregate for manuscript numbers.

Key findings:

- Naive iLCS fires on 100 percent of pure-control resamples on K562, RPE1, and
  Norman.
- The size-matched calibration reduces pure-control firing to 0.1200, 0.0800,
  and 0.0667, respectively. This is a large correction, but it does not uniformly
  restore the nominal 0.05 level.
- Jaccard overlap with the corrected GRD BH gate is 0.0000, 0.0316, and 0.0000.
- ICA non-convergence is highest on RPE1: 0.2602, compared with 0.0840 on K562
  and 0.0798 on Norman. This supports the claim that iLCS is least well-posed on
  the confounded RPE1 input.

Run `python experiments/validate_ilcs_results.py` before citing the aggregate.
