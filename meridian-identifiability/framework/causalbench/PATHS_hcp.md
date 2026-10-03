# HCP inputs for `causalbench/scripts/70_hcp_ceiling.py`

Sources traced from `hcp/scripts/mean_shift_v2.py` (the v1 `mean_shift.py` and
its LR/RL outputs are NOT used).

## Per-subject-run time series
- Directory: `/workspace/meridian-identifiability/hcp/ts/`
- Filename pattern: `{subject}_{task}_{encoding}.npy`
  - `{subject}` = HCP subject ID (string)
  - `{task}` ∈ `{WM, GAMBLING, MOTOR, LANGUAGE, SOCIAL, RELATIONAL, EMOTION}`
  - `{encoding}` ∈ `{LR, RL}`
- Each file: shape `(n_frames, n_regions)`, dtype float, `n_frames >= 176`
  (only the first 176 are used, matching v2)
- Split point for halves: frame 88
- Provenance: `hcp/scripts/mean_shift_v2.py:20-28`

## Subject inclusion
- Only subjects with all 7 tasks × 2 encodings = 14 files present
- Provenance: `hcp/scripts/mean_shift_v2.py:31-32`
- Expected complete-coverage count: 92 (matches `mean_shift_v2.json.n_subjects`)

## Scaling
- `subject_pooled`: per-subject-per-region z-score across all that subject's
  14 runs concatenated
- Provenance: `hcp/scripts/mean_shift_v2.py:37-44` (with `scaling="subject_pooled"`)
- Per-run z-score is NOT used (v2 line 15: "Per-run z-scoring is NOT used: it
  centers away the quantity measured")

## PCA basis
- Fit on all subject-scaled runs pooled, mean-centred
- Top `d=10` right singular vectors → `W` of shape `(n_regions, 10)`
- Provenance: `hcp/scripts/mean_shift_v2.py:46-48`

## What is NOT used
- `hcp/results/mean_shift_LR.json`, `mean_shift_RL.json` — deprecated v1 script
- `raw` scaling of `mean_shift_v2.json` — nuisance-uncorrected; the v2 docstring
  itself argues against it for signal recovery
