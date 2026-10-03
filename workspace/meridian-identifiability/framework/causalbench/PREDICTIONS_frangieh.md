# Frangieh 2021 screen predictions — REGISTERED BEFORE RESULTS

Purpose: pre-register predictions for `41_screen_frangieh.py` so the
comparison against Systema is prospective rather than post-hoc. The git
timestamp on this commit is the evidence.

Rule: nothing in this file may be revised once results land. If a
prediction turns out to be wrong, record that in a follow-up file, do not
edit this one.

---

## Registered

- Date (UTC): TODO
- Git SHA of HEAD at registration: TODO

## Verified reference values

From `MERIDIAN_STATE.md` §3, primary metric `mean_ratio_pairs`, headline
configuration `filt0 / NMIN=200 / d=10` for CausalBench and Norman;
step-0 gate is the pairs-based gate on control pseudo-environments.

- K562: **4.27**
- Norman: **3.71**
- RPE1: **2.06**
- Step-0 gate range across the three: **0.93 – 1.02**

Workable threshold used by the screen: `mean_ratio_pairs > 2.0`.

Frangieh configuration under prediction: per-arm bases, `NMIN=100`,
`d=10`, primary metric `mean_ratio_pairs`.

## Predictions

### Predicted `mean_ratio_pairs` band per arm

- Co-culture: TODO (low – high)
- Control: TODO (low – high)
- IFNγ: TODO (low – high)

### Predicted rank of Frangieh relative to K562, Norman, RPE1

Using each arm's headline `mean_ratio_pairs` against K562 4.27, Norman
3.71, RPE1 2.06:

- Co-culture rank: TODO
- Control rank: TODO
- IFNγ rank: TODO

### Predicted correlation with Systema PearsonD

Systema reports 0.91 – 0.95 across their datasets. State the predicted
sign and rough magnitude for the correlation between `mean_ratio_pairs`
and Systema PearsonD across the four datasets (K562, Norman, RPE1,
Frangieh, per arm).

- Predicted sign: TODO (+ / − / near zero)
- Predicted magnitude: TODO
- Predicted position vs the Systema 0.91 – 0.95 range: TODO (above / inside / below)

### Predicted surviving environment count per arm at NMIN=100

`screen_run()` in `03_screen.py:71–72` aborts if fewer than 4
environments survive. State the predicted count per arm and whether the
abort is expected to trip.

- Co-culture surviving envs: TODO
- Control surviving envs: TODO
- IFNγ surviving envs: TODO
- Expected to trip the ≥4 envs abort? TODO (yes / no, per arm)

### FALSIFICATION

One line stating what result would falsify the detectability reading:

TODO

## Confounds acknowledged in advance

- Frangieh has the fewest perturbations of the four datasets. Systema
  reports 167; the actual surviving count per arm at NMIN=100 will be
  known only after the run.
- Frangieh is Perturb-CITE-seq, a different technology from Replogle
  CRISPRi (K562, RPE1) and Norman CRISPRa. Any metric difference between
  Frangieh and the other three confounds mechanism (activation vs
  interference vs Cas9-KRAB) with readout modality (CITE-seq vs
  standard Perturb-seq).
- Per-arm splitting is deliberate: pooling arms would put the
  condition effect into the between-environment numerator. This is a
  design choice, not a confound, but it means Frangieh's numbers are
  per-arm and cannot be reduced to a single "Frangieh score" without
  choosing an aggregation rule.
