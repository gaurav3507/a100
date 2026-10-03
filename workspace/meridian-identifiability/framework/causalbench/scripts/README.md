# `causalbench/scripts/` — script index

Numbering is roughly chronological within a theme, not strictly by topic.
One exception is recorded explicitly below so the artifact appendix does not
mislead a reviewer.

## Numbering note (read this before citing a script by number)

**`71_specificity.py` is a GENOMICS script despite sitting in the 70-block,
which is otherwise HCP/fMRI.** It implements the specificity component of the
environment-validity screen and runs on K562 / Norman / RPE1 / Frangieh, not on
HCP.

It is deliberately **not renumbered**. Renaming would change its path and
break the SHA and commit references that `paper/EVIDENCE_PACK.md` points at.
The number carries no meaning; the docstring does.

## Index

| script | data | purpose |
|---|---|---|
| `01_download.py`, `01b/01c_get_token.py`, `01d_download.py` | — | data acquisition (SCP token handling, downloads) |
| `02_preprocess.py` | genomics | build the `dataset_*.npz` matrices |
| `03_screen.py` | genomics | **the screen kernel.** `fit_pca` / `project` / `coefs` / `offdiag`. Every later screen script imports these via `importlib` so the arithmetic stays byte-identical. **Do not modify.** |
| `04_spectrum.py` | genomics | effective dimensionality |
| `05_pc1_identity.py` | genomics | PC1 identity check |
| `40_screen_norman.py` | genomics | Norman 2019 screen |
| `40b_fix_step0_gate.py` | genomics | *(superseded)* patched the step-0 gate into an existing JSON. Later scripts compute the gate inline instead; do not repeat this pattern. |
| `41_screen_frangieh.py` | genomics | Frangieh 2021 (SCP1064) per-arm screen, seed sweep, target-drop toggle |
| `42_frangieh_poscontrol.py` | genomics | arm-as-environment positive control. **Control only, never a result.** |
| `43_check_drop_status.py` | genomics | is the target-column drop LIVE or a NO-OP per dataset |
| `48_nmin_ladder_all.py` | genomics | **NMIN ladder + slope diagnostic.** Native vs intersected env sets, step-0 gate at every rung, log-log slope, n-invariant `c_hat`. |
| `70_hcp_ceiling.py` | HCP | chance / cross-subject linear / ceiling-A / ceiling-B, permutation nulls, leak diagnostic |
| `71_specificity.py` | **genomics** | specificity component of the screen (see numbering note above) |

## Conventions these scripts follow

- **Primary metric is `mean_ratio_pairs`.** `mean_ratio_vs_ctrl` is degenerate
  under step-0 (`03_screen.py:63` sets `ref_pool = ctrl_rows[:0]`) and must not
  appear in any summary table.
- **The metric is not scale-free in NMIN.** See `paper/EVIDENCE_PACK.md` §2.1.
  Any cross-dataset comparison must state NMIN and should prefer the slope
  diagnostic from `48_nmin_ladder_all.py` over a level at a single rung.
- **Oracles gate real data.** Scripts that introduce a new estimator run a
  known-answer synthetic check first and exit non-zero on failure.
- **Sample sizes are asserted, not assumed.** Numerator and denominator must
  average the same number of cells per mean at every rung.
- **Results are written atomically** (`.tmp` + `os.rename`) so a partial file
  cannot masquerade as a finished one.
