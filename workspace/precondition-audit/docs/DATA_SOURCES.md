# Data sources

No data is committed. Every path resolves through `$PRECOND_DATA`, an explicit
argument, or an existence-checked candidate list; nothing is hardcoded.

| dataset | source | accession / URL |
|---|---|---|
| Replogle K562 | CausalBench release npz | `dataset_k562.npz` |
| Replogle RPE1 | CausalBench release npz | `dataset_rpe1.npz` |
| Norman 2019 | Norman2019_raw.h5ad | GEO GSE133344 |
| Frangieh (3 arms) | Perturb-CITE-seq, Single Cell Portal | SCP1064 |
| HCP | 7 tasks x 2 encodings, 176 frames | Human Connectome Project |
| ABIDE | preprocessed timeseries npz | ABIDE Preprocessed Initiative |

## What each loader does

- **Replogle** (`03_screen.load`) reads `expression_matrix` from the npz and
  casts to float64. No transformation is applied here.
- **Norman** (`40_screen_norman.load_norman`) reads `.X` from the h5ad,
  densifies, and maps `obs['guide_ids']` to the control / single / excluded
  convention. Loaded under an HVG cap of 5000 genes.
- **Frangieh** (`41_screen_frangieh`) skips row 2 (Single Cell Portal's TYPE
  convention row, which otherwise becomes a phantom cell), keeps `MOI == 1`
  only, parses the target by stripping the **trailing** guide index
  (`^(.*)_\d+$`, not a split on the first underscore, which would mangle
  `NO_SITE_1`), pools `NO_SITE_*` and `ONE_NON-GENE_SITE_*` as controls, and
  reads the dense CSV in chunks as float32. Arms are profiled separately and
  never pooled.
- **HCP / ABIDE** read timeseries arrays directly. Neither has a control
  condition, so neither contributes a control-covariance spectrum.

## Upstream normalisation is UNDOCUMENTED

**For all six transcriptomic matrices the upstream normalisation is not
recorded anywhere.** None of the three loaders applies or documents a
transformation; each passes through whatever the upstream file holds, and the
upstream deposits do not state it either.

What can be established from the committed artefacts, without the data:

| dataset | shape (control cells x genes) | dtype | per-entry mean | RMS | sum integral? |
|---|---|---|---|---|---|
| k562 | 10691 x 1158 | float64 | 0.8234 | 1.2599 | no |
| rpe1 | 11485 x 651 | float64 | 0.6142 | 1.0135 | no |
| norman | 8907 x 5000 | float64 | 0.1137 | 0.4750 | no |
| frangieh_coculture | 5707 x 23712 | float32 | 0.7223 | 1.9266 | no |
| frangieh_control | 3770 x 23712 | float32 | 0.7442 | 1.9618 | no |
| frangieh_ifng | 6144 x 23712 | float32 | 0.6495 | 1.8720 | no |

A float64 sum of order 1e7 integer counts is exact, since integers are exact in
float64 up to 2^53. Every one of these sums is non-integral, so **none of the
six is raw counts**. The per-entry scale is consistent with a log1p-style
transform.

**This is evidence, not proof, and log1p is not asserted as fact.** The
transform could be log1p on a different size factor, or something else with a
similar range. `scripts/91_preprocessing_sweep.py` therefore probes each matrix
at runtime — classifying on the fraction of **non-zero** entries that are
integral, because these matrices are mostly exact zeros and a zero is an
integer — and refuses the `log1p_std` arm wherever the input is already
log-like, rather than silently applying a second log.
