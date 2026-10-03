# Evidence pack

Every number intended for the manuscript, traced to its source file and
independently re-verified.

**Verification status.** Run on the A100 with every result file present, plus
raw-data descriptors and third-party source:
**≈240 numbers traced, 141 independently recomputed, 0 stored-vs-recomputed
mismatches.** Every results file is internally consistent — each stored scalar
reproduces exactly from the per-item arrays in the same file. **All defects found
are between files, or are metric-choice problems.**

One correction to an earlier draft of this document is recorded at D12: the
claim that HCP's effective-dim value used a 1× threshold was wrong. It uses the
2× threshold like the others.

Verification scripts:

    python scripts/60_verify_evidence.py      # stored vs recomputed, all sections
    python scripts/62_crosscheck_claims.py    # prose/figure claims vs results files
    python scripts/61_collect_descriptors.py  # A100 only: raw-data descriptors

---

## 1. Screen — primary metric (`mean_ratio_pairs`)

Headline configuration: `filt0`, `NMIN=200`, `d=10`, `seed=0`.

### 1.1 Why `pairs` is primary

From `causalbench/scripts/03_screen.py`, both ratios share the denominator
`median(‖mean(Za) − mean(Zb)‖)` (split-half of the same environment). They differ
in the numerator:

- `mean_ratio_vs_ctrl` — environment against a draw from `ref_pool`.
- `mean_ratio_pairs` — environment against **a different environment**.

**Confirmed at `03_screen.py:63`:** under step-0, `ref_pool = ctrl_rows[:0]` is
empty, so the draw falls back to the pseudo-environment's own rows. Under step-0
`mean_ratio_vs_ctrl` therefore compares an environment against itself. Measured
step-0 `vs_ctrl` values are 0.672 (K562), 0.678 (RPE1), 0.722 (Norman) — a
construction artifact, not a calibrated null. Only `mean_ratio_pairs` returns
≈1.0. **`mean_ratio_pairs` is the primary metric.**

### 1.2 Headline values ✅ VERIFIED

| dataset | **`mean_ratio_pairs`** | `mean_ratio_vs_ctrl` | `n_envs` | `n_fit` | source |
|---|---:|---:|---:|---:|---|
| K562 | **4.270326807237259** | 3.0766671188533263 | 385 | 100 | `screen/k562_filt0_n200_d10.json` |
| RPE1 | **2.058949502375608** | 1.6709678829248977 | 146 | 100 | `screen/rpe1_filt0_n200_d10.json` |
| Norman | **3.7147385914720377** | 3.1895884263497645 | 101 | 100 | `screen/norman.json $.screen[nmin=200,d=10]` |
| HCP | *no such metric* — see §1.4 | 0.7483833300732464 (LR) | — | — | `hcp/results/mean_shift_LR.json` |

`n_envs` corroborated independently by the split files (`$.n_usable` = 385 / 146) ✅.

### 1.3 Step-0 gate ✅ VERIFIED

| dataset | `mean_ratio_pairs` (the gate) | `mean_ratio_vs_ctrl` (degenerate) | in [0.9, 1.1]? |
|---|---:|---:|---|
| K562 | 1.0178167396182576 | 0.6721562536093018 | ✅ |
| RPE1 | 0.9255876747619355 | 0.678180883144252 | ✅ |
| Norman | 1.0070856615811838 | 0.7216995131517371 | ✅ |
| HCP | **absent** | — | **no gate exists** |

Norman across all four of its configurations: 0.996, 1.007, 1.020, 1.029 ✅.

### 1.4 HCP — NOT COMPARABLE ✅ SOURCE COMPARISON COMPLETE

There are **two different HCP scripts** producing **three different ratios**.
This section supersedes any earlier statement about HCP.

**Line-by-line role mapping**, `03_screen.py` vs `hcp/scripts/mean_shift_v2.py`
(lines 56–63):

| role | `03_screen.py` | `mean_shift_v2.py` |
|---|---|---|
| denominator: same condition, split-half | `within_m` = ‖mean(Za) − mean(Zb)‖ | **`split`** = ‖A[s,t,e] − B[s,t,e]‖ (first 88 vs last 88 frames) |
| numerator: across condition | `pair_m` = ‖mean(Za) − mean(Zb)‖, two environments | **`task`** = ‖A[s,t1,e] − A[s,t2,e]‖, two tasks |
| nuisance null: same condition, different acquisition | **does not exist** | **`encode`** = ‖A[s,t,LR] − A[s,t,RL]‖ |

**The direct analogue of `mean_ratio_pairs` is `task_over_split`.** Measured
value at raw scaling, d=10: **17.306187786068094**
(`hcp/results/mean_shift_v2.json $.results.raw["10"].task_over_split`).

The number the project has been quoting for HCP is **not** that. Three candidates
exist and they disagree by more than an order of magnitude:

| quantity | file | d=10 value | is it the `pairs` analogue? |
|---|---|---:|---|
| **`task_over_split`** | `mean_shift_v2.json` raw | **17.306** | **yes — same numerator and denominator roles** |
| `task_over_encode` | `mean_shift_v2.json` raw | 0.6918 | no — denominator is a run-nuisance null |
| `mean_ratio` | `mean_shift_LR.json` (v1 script) | 0.7484 | no — `between/within`, the `vs_ctrl` form |

**Why the HCP pipeline deliberately does not use `task_over_split`.** Its own
docstring (lines 8–11) states: raw BOLD carries arbitrary per-run baseline and
gain, so task-vs-task differences include run nuisance; LR-vs-RL of the same task
contains that nuisance and no condition effect; if `task/encode ≈ 1` the apparent
condition effect is acquisition, not cognition. Measured `encode_over_split` =
25.0175362270292 — the acquisition nuisance is 25× the estimation noise, which
is why `split` is an inadequate null for fMRI.

**Comparability verdict: the two quantities are not comparable, in either
direction.**

- Computed the CausalBench way (`task_over_split`), HCP scores **17.3** and would
  be classified highly workable.
- Computed the HCP way, CausalBench has no `encode`-equivalent null. `03_screen.py`
  builds no "same perturbation, different batch" comparison — batch identity
  (`gem_group`) exists in the raw `.h5ad` but the screen never reads it.

So HCP's apparent failure and CausalBench's apparent success rest on
**structurally different denominators**, and neither dataset has a value of the
other's metric. Placing them on one axis is not supportable without either
(a) adding a batch-matched null to `03_screen.py`, or (b) reporting HCP's
`task_over_split` and accepting that it passes.

Other HCP facts:

- `mean_shift_v2.py` fits its PCA basis on **all runs pooled** (line 46–47), not
  on controls — HCP has no control condition, so there is no `ref_pool` and no
  step-0 gate can exist.
- Design: 7 tasks × 2 encodings × 92 (LR) / 93 (RL) subjects, 176 frames, split
  at 88.
- `max_dims = 6` throughout: the spectrum is built from 7 group task means,
  mean-centred (line 65), leaving rank 6.

---

## 2. Screen — sensitivity to configuration ✅ VERIFIED

All 16 configurations. `mean_ratio_pairs` is the primary metric; `vs_ctrl` shown
for the supplement.

| dataset | filt | NMIN | d | **pairs** | vs_ctrl | `n_envs` |
|---|---:|---:|---:|---:|---:|---:|
| K562 | 0 | 200 | 5 | 4.335 | 3.053 | 385 |
| K562 | 0 | 200 | 10 | **4.270** ← headline | 3.077 | 385 |
| K562 | 0 | 100 | 5 | 2.257 | 1.604 | 1158 |
| K562 | 0 | 100 | 10 | 2.228 | 1.550 | 1158 |
| K562 | 1 | 200 | 5 | 5.633 | 4.745 | 206 |
| K562 | 1 | 200 | 10 | 5.283 | 4.370 | 206 |
| K562 | 1 | 100 | 5 | 3.223 | 2.556 | 622 |
| K562 | 1 | 100 | 10 | 3.166 | 2.422 | 622 |
| RPE1 | 0 | 200 | 5 | 2.048 | 1.592 | 146 |
| RPE1 | 0 | 200 | 10 | **2.059** ← headline | 1.671 | 146 |
| RPE1 | 0 | 100 | 5 | 2.123 | 1.632 | 651 |
| RPE1 | 0 | 100 | 10 | 2.070 | 1.546 | 651 |
| RPE1 | 1 | 200 | 5 | 2.497 | 1.967 | 86 |
| RPE1 | 1 | 200 | 10 | 2.376 | 1.828 | 86 |
| RPE1 | 1 | 100 | 5 | 3.238 | 2.198 | 383 |
| RPE1 | 1 | 100 | 10 | 2.973 | 2.005 | 383 |

**K562 `pairs` range [2.228, 5.633], spread 3.405.**
**RPE1 `pairs` range [2.048, 3.238], spread 1.189.**

### 2.1 The metric is not scale-free in NMIN

Reading down the table, the ratio rises with NMIN and falls with the number of
environments admitted. K562 `filt0` goes 2.228 (n=100, 1158 envs) → 4.270
(n=200, 385 envs). This is a property of the construction: the denominator is a
split-half distance that shrinks as ≈1/√n while the between-environment
numerator does not. **The ratio is therefore only comparable across datasets at
matched NMIN.** Any cross-dataset claim must state NMIN.

### 2.2 D9 resolved: the README range is traceable, and it is the whole sweep

`README.md` claims K562 `2.0 – 5.6`. The measured `pairs` sweep is
[2.228, 5.633]. The upper bound matches (5.633 → 5.6); the lower bound is
rounded down from 2.228 to 2.0. **The README range is the min–max across all
eight K562 configurations**, i.e. it is a range over analysis choices, not a
confidence interval. It should be restated as **[2.23, 5.63] across
filt × NMIN × d**, or replaced by the single headline configuration.

---

## 3. Effective dimensionality

Definitions from `causalbench/scripts/04_spectrum.py`:

- `n_dims_above_2x = count(s_sig[i] / s_noi[i] > 2.0)` — a **count of
  components** above twice a control-derived null.
- `participation_ratio = (Σ s_sig)² / Σ s_sig²` — an **effective rank of the
  signal spectrum alone**, computed with no reference to the null.

These are different quantities. See D12.

### 3.1 All datasets ✅ VERIFIED

Every entry recomputed from `sing_signal` / `sing_noise`; all agree exactly.

| dataset | d | `dims_above_2x` | `dims_above_noise` | `participation_ratio` | `s2/s1` | `n_envs` |
|---|---:|---:|---:|---:|---:|---:|
| K562 | 10 | 9 / 10 | 10 / 10 | 5.050295711497646 | 0.6198524593548168 | 385 |
| K562 | 20 | 14 / 20 | 20 / 20 | 7.276305600242129 | 0.6335317163746099 | 385 |
| K562 | 50 | **15 / 50** | 50 / 50 | 11.929814856452163 | 0.6485644940315521 | 385 |
| RPE1 | 10 | 6 / 10 | 10 / 10 | 3.571238942175779 | 0.2388659303651289 | 146 |
| RPE1 | 20 | 9 / 20 | 20 / 20 | 5.614722346505135 | 0.2472676152100853 | 146 |
| RPE1 | 50 | 9 / 50 | 50 / 50 | 11.042364411736862 | 0.24993314940512823 | 146 |
| Norman | 10 | 10 / 10 | 10 / 10 | 4.446065832169267 | 0.6582844209194609 | 101 |
| Norman | 20 | 20 / 20 | 20 / 20 | 7.347425728612681 | 0.6679978502032126 | 101 |
| Norman | 50 | **39 / 50** | 50 / 50 | 13.226402291236367 | 0.6833521368943763 | 101 |
| HCP LR | 10 | **0** | 3 | 4.7356 | — | — |
| HCP LR | 20 | **0** | 3 | 4.9517 | — | — |
| HCP LR | 30 | **0** | — | 5.1423 | — | — |
| HCP LR | 50 | **0** | — | 5.1788 | — | — |
| HCP RL | 10 | **1** | 1 | 4.5126 | — | — |
| HCP RL | 20 | **0** | 0 | 4.8827 | — | — |
| HCP RL | 30 | **0** | — | 4.9968 | — | — |
| HCP RL | 50 | **0** | — | 5.0326 | — | — |

**HCP has `max_possible_dims = 6` and `len(sing_ratio) = 6` at every d**, capped
by the 7-task design. "dims above 2× out of d" is therefore not on the same
denominator as the CausalBench datasets, where the spectrum has length d.

### 3.2 Why Norman's 39 and 13.2 do not conflict

**39/50** is a threshold count over a *ratio of two spectra*: it grows as more of
the tail clears a fixed bar, and Norman's null spectrum decays fast.
**13.23** is the participation ratio of the *signal spectrum alone*: dominated by
the leading components, insensitive to a long shallow tail.

They diverge exactly when the signal spectrum has a long tail above the null
carrying little energy. Readable from the code; **no anomaly and no discrepancy.**
But the two must never be reported in one column — see D12.

---

## 4. Intervention subspace vs control PCA ✅ VERIFIED

Source: `results/cf_estimator/k562.json`, `nmin=200`.

| quantity | value |
|---|---:|
| environments | 385 |
| stable environments | 298 |
| **control-cell gate** `gate_cos_median` | **−0.006828483048328142** |
| `gate_cos_mean` | −0.009455955587287365 |
| split-half `stab_cos_median` | 0.6842780722525882 |
| `stab_cos_q10` | 0.06632494098453602 |
| `stab_cos_q90` | 0.8803670910740762 |
| `frac_cos_gt_0p3` | 0.7168831168831169 |
| `rank1_median` | 0.025447037879799767 |
| `shift_norm_median` | 2.5391828429016745 |

Held-out reconstruction R²; `gain` recomputed for every d, all agree to 1e-9 ✅.

| d | `shift_basis_r2` | `control_pca_r2` | `gain` |
|---:|---:|---:|---:|
| 5 | 0.7029684174159108 | 0.4846019163349933 | 0.21836650108091749 |
| 10 | 0.7642257409720432 | 0.592387419064074 | 0.17183832190796922 |
| 15 | 0.78473985776114 | 0.6193275985937475 | 0.16541225916739244 |
| 20 | 0.7992939864369504 | 0.630621176988112 | 0.16867280944883833 |
| 30 | 0.8165562129322685 | 0.6492430406894609 | 0.16731317224280762 |

The control-cell gate returning ≈0 is the intended known-answer result.

---

## 5. Zero-shot baselines, canonical split ✅ VERIFIED

Split provenance: `split_file = k562_zeroshot_split.json`, `n_train = 308`,
`n_heldout = 77`, `seed = 0`, `nmin = 200`, `d_latent = 15`. All confirmed
against the split file itself, including `len(train_perturbations) = 308`,
`len(heldout_perturbations) = 77`, `n_usable = 385 = 308 + 77`, and
`385 = count(cells_per_perturbation ≥ 200)` ✅.

**All 24 baseline statistics recomputed from the 462-row `$.per_perturbation`
array (6 methods × 77 genes); every one agrees exactly.** ✅

| method | median R² | mean R² | median cos | `frac_beats_gmean` |
|---|---:|---:|---:|---:|
| `zero` | 0.0 | 0.0 | 0.0 | 0.36363636363636365 |
| `global_mean` | **0.12902286583297684** | −0.04360618578319341 | 0.3777093107368002 | 0.0 |
| `corr_prop` | −0.18752402582055216 | −0.28580222935470173 | 0.13457430570772097 | 0.24675324675324675 |
| `nn_corr` | −0.24278568997495253 | −1.3799998480864606 | 0.22765124994501498 | 0.2727272727272727 |
| `ridge_basis` | **0.22558014602202647** | 0.033191204082618844 | 0.5010799951288413 | 0.7012987012987013 |
| `CEILING` | **0.5274920724871424** | 0.3168453803721219 | 0.7690887650072865 | 0.8311688311688312 |

`_headroom = 0.3984692066541655` ✅. Ridge capture = **24.232055721397924 %** ✅.

---

## 6. The model result ✅ VERIFIED

All four arms; `final_median_r2`, `final_mean_r2`, `frac_beats_gmean`,
`frac_beats_ridge` recomputed from each 77-entry `$.per_gene` array ✅.

| arm | best median R² | final median R² | final mean R² | per-gene R² range |
|---|---:|---:|---:|---|
| full / shift | 0.009694068092076957 | 0.012570545400028665 | −0.0518394092777882 | [−0.7075, 0.1128] |
| full / random | −0.005696563328918991 | −0.005466684895471641 | −0.006320124530840101 | [−0.0427, 0.0221] |
| no-DAG / shift | −0.04224525495841136 | −0.04793495530415348 | −0.10737029539395729 | [−0.7025, 0.0520] |
| no-DAG / random | 0.013044818906979505 | 0.013023438700739853 | 0.013743959567562424 | [−0.0304, 0.0704] |

`frac_beats_gmean` and `frac_beats_ridge` are **0.0 for all four arms**:
**0 of 77 held-out genes exceed 0.129, and 0 of 77 exceed 0.226** ✅.

`results/model/SUMMARY.json` cross-checked field by field: **28 fields, all
agree** ✅.

### 6.1 DAG condition number ✅ VERIFIED

`results/model/dag_stability.csv`, 223 rows, epochs 0–14, shift-init.

| quantity | value |
|---|---:|
| `cond2` min / max / final | 1.000471 / **1.812932** / 1.569744 |
| `inv_norm_2` min / max | 1.000219 / 1.357119 |
| `G_frobenius` min / max | 1.20544 / 1.587544 |

Zero NaNs. The condition number never approaches 10³.

### 6.2 Training trajectories ✅ VERIFIED — D7 resolved

`results/model/trajectories/*.json`, 11 evaluation points per arm.

| arm | **epoch-10 R²** | trajectory min | `best_median_r2` = max(trajectory) |
|---|---:|---:|---|
| full / shift | **−1.087** | −1.087 | 0.0097 ✅ |
| full / random | −0.0611 | −0.2056 | −0.0057 ✅ |
| no-DAG / shift | **−0.9972** | −0.9972 | −0.0422 ✅ |
| no-DAG / random | +0.012 | −0.0997 | 0.0130 ✅ |

Both earlier records were correct, merely rounded: the handoff's **−1.09** is
−1.087, and the later note's **−0.997** is −0.9972. The epoch-10 excursion is
the trajectory minimum for both shift-init arms, and it occurs **with and
without the DAG**.

---

## 7. Dataset descriptors

### 7.1 From the split files ✅ VERIFIED

| quantity | K562 | RPE1 |
|---|---:|---:|
| genes in `cells_per_perturbation` | 1158 | 651 |
| environments clearing NMIN=200 | **385** | **146** |
| `n_train` / `n_heldout` | 308 / 77 | 117 / 29 |
| cells/env median (all recorded) | 165.0 | 146.0 |
| cells/env min / max | 101.0 / 1996.0 | 101.0 / 3580.0 |
| seed | 0 | 0 |

The median is over **all** recorded genes including those below NMIN. The median
over the 385 / 146 that clear NMIN is a different number, stored nowhere.

### 7.2 Raw-data descriptors ✅ VERIFIED

From `scripts/61_collect_descriptors.py` (`paper/descriptors.json`).

| quantity | K562 | RPE1 | Norman 2019 |
|---|---:|---:|---:|
| cells, raw | 310385 | 247914 | 108497 |
| cells excluded | 70237 | 108089 | — |
| **cells analysed** | **240148** | **139825** | 108497 |
| **features** | **1158** | **651** | **5000** |
| **control cells** | **10691** | **11485** | **8907** |
| perturbed cells | 229457 | 128340 | 57831 single + 41759 double |
| **unique perturbations** | **1158** | **651** | **105** (single) |
| clearing NMIN=200 | **385** | **146** | **101** |
| clearing NMIN=100 | 1158 | 651 | 105 |
| cells/pert min | 101 | 101 | 113 |
| cells/pert p10 | 111.0 | 107.0 | — |
| cells/pert median | 165.0 | 146.0 | 495.0 |
| cells/pert p90 | 326.0 | 277.0 | — |
| cells/pert max | 1996 | 3580 | 1960 |
| value min / max | 0.0 / 6.644 | 0.0 / 6.700 | 0 – 8.5 |
| fraction exactly zero | 0.4109 | 0.5038 | 0.914 |
| integer-valued? | **No** | **No** | No |
| **targets among features** | **1158 / 1158** | **651 / 651** | **0 / 105** |

Preprocessing: none of the three matrices is integer-valued and all have
bounded positive ranges — **all three are log-normalised**, including Norman
despite its `_raw` filename.

**The target-column asymmetry is total, not partial.** Every CausalBench target
is one of the observed features (1158/1158, 651/651); no Norman target is
(0/105). The "zero the targeted gene's row" step in `project()` therefore fires
for every CausalBench environment and never for Norman
(`norman.json $.target_column_drop_is_noop = true` ✅).

`causalbench_k562.h5ad` — the file `cb_data.py` actually reads — has 240148
cells, 1158 features, 10691 empty-string control cells, and 1159 unique
`guide_ids` (1158 genes + the empty string) ✅ consistent with the `.npz`.

**Batch identity is available for Norman but not for CausalBench.** Norman's
`obs` carries `gemgroup` and `batch`; `causalbench_k562.h5ad` carries only
`guide_ids`. No batch variable is used by `03_screen.py` for any dataset.

CausalBench is CRISPR interference; Norman is CRISPR activation.

HCP design (from `mean_shift_v2.py` and its JSON): 7 tasks
(WM, GAMBLING, MOTOR, LANGUAGE, SOCIAL, RELATIONAL, EMOTION) × 2 encodings ×
92 (LR) / 93 (RL) subjects, 176 frames per run.

---

## 8. Third-party provenance ✅ VERIFIED

| repository | commit |
|---|---|
| discrepancy_vae (Zhang et al., NeurIPS 2023) | `4451fdbc9d0aa3a1dee4e7d1b743a434e98fa58a` |
| causalbench_repo (Chevalley et al., NeurIPS 2023) | `1a2143cffdc85f835b41ce8d52034be1bf903e71` |

`COMMITS.txt` mirrors both and agrees ✅.

`vendor_analysis/my_changes.patch`: **4 hunks, touching only `src/dataset.py`** —
dataset-path strings. No model or loss change.

`model/cb_train.py`: `from train import loss_function` = **True**; defines its own
`loss_function` = **False**. Same for `scripts/27_no_causal_ablation.py` ✅.

Our hyperparameters (argparse defaults): `zdim=15`, `epochs=100`, `batch=128`,
`lr=1e-3`, `mxAlpha=10.0`, `mxBeta=2.0`, `mxTemp=5.0`, `MMD_sigma=1000.0`,
`kernel_num=10`, `lmbda=1e-3`.

Their signatures, read from their tree ✅:

    src/train.py  loss_function(y_hat, y, x_recon, x, mu, var, G,
                                MMD_sigma, kernel_num, matched_IO=False)
    src/model.py  CMVAE.dag(z, bc, csz, bc2, csz2, num_interv=1)
                  torch.inverse(torch.eye(self.z_dim) ...)

Both match how we call them ✅. `cb_train.py` and `27_no_causal_ablation.py` each
call `loss_function(...)` exactly once and neither redefines it ✅. Our
`CMVAE_CB_NoDAG.dag` override reproduces their signature exactly ✅.

⚠️ **Their argparse defaults could not be extracted** — `scripts/61` found no
`add_argument(... default=...)` matches in `src/train.py`. Our defaults therefore
remain unconfirmed against theirs. Whether `MMD_sigma=1000.0` / `kernel_num=10`
match the values Zhang et al. used is **still open**; those two were taken from
their `MMD_loss` call site, not from a defaults block.

---

# DISCREPANCIES AND OPEN ITEMS

Ordered by severity. Machine-readable copy: `paper/crosscheck_findings.json`.

### D16 — The Frangieh screen is POST-HOC, not pre-registered (CRITICAL)

`causalbench/PREDICTIONS_frangieh.md` was committed to establish the Frangieh
test as prospective. It does not do so, on two independent grounds.

**1. The lockfile contains no predictions.** It holds **16 `TODO` placeholders**
and was never edited after registration (single commit `8722c84`, no follow-ups).
Every prediction field is empty:

```
- Co-culture: TODO (low – high)
- Control: TODO (low – high)
- IFNγ: TODO (low – high)
- Predicted sign: TODO (+ / − / near zero)
- Co-culture surviving envs: TODO
```

A registration containing no registered prediction carries no prospective
weight, regardless of its timestamp.

**2. Results for two of three arms existed before registration.** Normalised to
UTC:

| event | commit | UTC |
|---|---|---|
| Frangieh script added | `9230b16` | 2026-07-24 17:28Z |
| IFNγ label fix committed | `4fab4f4` | 2026-07-25 **08:09Z** |
| `PREDICTIONS_frangieh.md` registered | `8722c84` | 2026-07-25 **10:58Z** |
| `frangieh.json` produced (mtime) | — | 2026-07-25 **11:36Z** |

The IFNγ fix at 08:09Z was made *because* an earlier run had silently dropped
the IFNγ arm (76913 cells retained instead of 126966, an ASCII `IFNg` vs Greek
`IFNγ` U+03B3 label mismatch). That run necessarily produced `mean_ratio_pairs`
for Co-culture and Control. Those two values were therefore visible **2h49m
before** the lockfile was committed. Registration preceded the *final* results
file by 38 minutes, but not the *first* results.

**Consequence.** The Frangieh result must be written as a post-hoc observation.
No prospective-validity claim can be attached to it. Proposed limitations text:

> The Frangieh screen was not pre-registered. A predictions file was committed
> before the final results were written, but it contained no filled-in
> predictions, and an earlier run of the same screen had already produced values
> for two of the three arms. We report the Frangieh result as an observation and
> do not claim prospective validity for it.

**Recovery.** Adamson (GSE90546) is genuinely unrun. A lockfile with predictions
actually filled in, committed before that run, would be prospective. This is the
one remaining opportunity to make a prospective claim in this paper and it
should not be spent carelessly.

### D13 — RPE1 clears the workability threshold on the primary metric (CRITICAL)

RPE1 is claimed as one of the two **correctly-flagged negatives** underpinning
the screen's predictive validity. On the primary metric it is not flagged:

| metric | RPE1 range across all 8 configs | vs 2.0 threshold |
|---|---|---|
| **`mean_ratio_pairs` (primary)** | **[2.048, 3.238]** | **above at every configuration** |
| `mean_ratio_vs_ctrl` (secondary) | [1.546, 2.198] | below at 6 of 8 |

Under `mean_ratio_pairs`, RPE1 would be classified **workable** at every single
configuration, including the headline one (2.059). The "RPE1 correctly flagged as
too weak" claim holds **only** under `vs_ctrl` — the metric that §1.1 establishes
is not calibrated.

This directly affects the predictive-validity count. If the primary metric is
`pairs`, the screen has one negative (HCP), not two, and RPE1 becomes a
candidate **false positive** — a dataset the screen passes that was
independently found unusable (negative zero-shot noise ceiling). Resolving this
is prerequisite to any statement about how many datasets the screen has
correctly classified.

### D14 — HCP's true `pairs` analogue is 17.3, not 0.75 (CRITICAL)

The quantity structurally equivalent to `mean_ratio_pairs` is
`mean_shift_v2.json` `task_over_split` = **17.306187786068094** (raw, d=10). Every
HCP number the project has quoted — 1.00 in the README, 0.7484 from
`mean_shift_LR.json`, 0.6918 from `task_over_encode` — uses a **different
denominator**.

HCP's pipeline uses `encode` (same task, different acquisition run) as its null
rather than `split`, because `encode_over_split` = **25.0175362270292**: run-level
acquisition nuisance is 25× estimation noise in raw BOLD. That is a defensible
choice **for fMRI**, and it is why `task_over_encode` ≈ 0.69 rather than 17.3.

But `03_screen.py` has no `encode`-equivalent. It never compares the same
perturbation across batches; `gem_group` is present in the raw `.h5ad` and unused.
So the CausalBench ratio contains no acquisition-nuisance correction at all.

**The HCP-vs-CausalBench comparison is therefore between a nuisance-corrected
ratio and an uncorrected one.** Under matched construction HCP passes (17.3); under
its own construction it fails (0.69). Which is reported determines whether HCP is
the screen's negative control. See §1.4.

### D12 — The README "effective dim" column mixes two metrics (HIGH) — CORRECTED

**An earlier draft of this document claimed HCP's `3` was `n_dims_above_noise`.
That was wrong.** With `mean_shift_v2.json` now read, `3` is
`$.results.raw["10"].dims_above_2x` — the **same 2× threshold** as the other
datasets. The HCP entry is consistent; the Norman entry is not.

| README cell | value | what it actually is | that dataset's `dims_above_2x` |
|---|---:|---|---:|
| CausalBench `~15` | 15 | `n_dims_above_2x` at d=50 ✅ | 15 |
| Norman `13.2` | 13.226 | **`participation_ratio`** at d=50 | **39** |
| HCP `3` | 3 | `dims_above_2x`, v2 raw, d=10 ✅ | 3 |

So the column mixes **two** metrics, not three: Norman is the odd one out. Under
`dims_above_2x` throughout the column reads 15 / 39 / 3; under
`participation_ratio` it reads 11.93 / 13.23 / 4.74. The two orderings differ.

### D15 — HCP's two scripts disagree on `dims_above_2x` (MEDIUM)

| file | script | `dims_above_2x` at d=10 |
|---|---|---:|
| `mean_shift_v2.json` (raw) | `mean_shift_v2.py` | **3** |
| `mean_shift_LR.json` | `mean_shift.py` (v1) | **0** |
| `mean_shift_RL.json` | `mean_shift.py` (v1) | 1 |

Both are internally consistent ✅ (each recomputes from its own `sing_ratio`), but
they build different null matrices: v2 uses the LR−RL difference of group task
means (lines 66–67, scaled by 1/√2), v1 uses something else. **The paper must
state which HCP script it reports.**

### D4 — Model and baselines are scored against different targets (HIGH)

| file | target |
|---|---|
| `scripts/25_zeroshot_baselines.py` | **half B** of the held-out cells (`A, B = shift(cells[:h]), shift(cells[h:])`; all methods scored against `B`) |
| `model/cb_train.py::zeroshot_eval` | the **full-data** mean shift (`truth = X[iv==g].mean(0) - ctrl_mu`) |

The R² *formula* is identical ✅ (`1 − RSS / Σtrue²`), but the model's target is
the less noisy of the two. Yet `cb_train.py` hard-codes **0.129** and **0.226**
— produced by the half-B evaluation — as its comparison thresholds. Affects
every "model vs ridge" statement.

### D3 — `frac_beats_gmean` denotes two different quantities (HIGH)

| file | definition |
|---|---|
| `scripts/25_zeroshot_baselines.py` | per-gene: `r2 > gm_by_g[gene]` |
| `model/cb_train.py` | constant: `np.mean(zs > 0.129)` |

Both recompute exactly ✅ and both are internally correct, but ridge's **0.701**
and the models' **0.000** answer different questions and cannot share a column.

### D1 — Norman headline uses the secondary metric (HIGH)

`README.md` and `causalbench/results/screen/norman_verdict.txt` both state
**3.19** = `mean_ratio_vs_ctrl`. The primary metric gives **3.7147385914720377**.

Note also that `norman.json` contains a key literally named
`summary.primary_mean_ratio_vs_ctrl` — the file asserts the wrong metric is
primary.

### D2 — Figure 2 mixes two metrics on one axis (HIGH)

`scripts/50_figures.py::fig2_mean_shift_ratio` takes bar heights from
`mean_ratio_vs_ctrl` while its null line and gate band come from
`mean_ratio_pairs`. **Still unfixed**; the script is deliberately left
uncommitted for this reason.

### D5 — HCP has no step-0 gate and cannot have one (HIGH)

`mean_shift_v2.py` fits its PCA basis on all runs pooled (lines 46–47). HCP has
no control condition, so there is no `ref_pool`, no control pseudo-environments,
and **no step-0 gate can be constructed**. The gate that calibrates the primary
metric to 1.0 for K562, RPE1 and Norman does not exist for HCP, so HCP's ratio
is uncalibrated in the sense the other three are calibrated.

HCP's spectrum also has only 6 entries at every d (7 group task means,
mean-centred → rank 6), so "dims above 2× out of d" has a different denominator
than the CausalBench datasets, where the spectrum has length d.

### D6 — README HCP value 1.00 is not traceable to any file (MEDIUM)

`README.md` states **1.00**. No HCP results file contains it. The three
candidates are 17.306 (`task_over_split`), 0.6918 (`task_over_encode`) and
0.7484 (v1 `mean_ratio`). 1.00 appears to be a rounding of ~0.75 up to the null,
or a placeholder. See D14.

### D9 — README K562 range is a sweep over analysis choices (MEDIUM) — RESOLVED

`2.0 – 5.6` is the min–max of `mean_ratio_pairs` across all eight K562
configurations, measured as **[2.228, 5.633]**. Upper bound matches; lower bound
rounded down from 2.228. It is a range over `filt × NMIN × d`, not a confidence
interval, and must be labelled as such.

### D8 — README CEILING rounds the wrong way (LOW)

`README.md` gives **+0.528**; stored value is **0.5274920724871424**, which
rounds to 0.527. The same 0.528 appears in `norman_verdict.txt` and in the FIG 6
baseline line — propagated to three places.

### D7 — Epoch-10 value (RESOLVED)

Now traceable: `full_shift` **−1.087**, `nodag_shift` **−0.9972**. Both earlier
records (−1.09, −0.997) were correct. Closed.

### D10 — "CausalBench effective dim ~15" (RESOLVED into D12)

Confirmed as `n_dims_above_2x` at d=50 for K562 = 15 ✅. The problem is not this
number but the column it sits in — see D12.

### Open items

1. **§8** — Zhang et al.'s hyperparameter defaults. `scripts/61` found no
   `add_argument(... default=...)` block in their `src/train.py`, so our
   `MMD_sigma=1000.0` and `kernel_num=10` remain unconfirmed against theirs.
   Resolving this needs a manual read of their training entry point.
2. **D14 / §1.4** — decide which HCP quantity the paper reports. This is a
   methodological choice, not a lookup: `task_over_split` (matched construction,
   HCP passes at 17.3) versus `task_over_encode` (nuisance-corrected, HCP fails
   at 0.69). Both are defensible; they cannot both be reported as "the HCP
   mean-shift ratio".
3. **D13** — decide whether RPE1 is a negative under the primary metric. Same
   character: a metric choice with a classification consequence.

Items 2 and 3 are the only two places where the evidence does not determine the
answer. Everything else in this document is now a traced, recomputed value.

### Where `mean_ratio_vs_ctrl` is still in use

| location | use |
|---|---|
| `scripts/50_figures.py` fig2 | bar heights, printout, reference table (D2) |
| `README.md` | Norman 3.19, K562 range, HCP row |
| `causalbench/results/screen/norman_verdict.txt` | the saved verdict sentence |
| `causalbench/results/screen/norman.json` | key named `summary.primary_mean_ratio_vs_ctrl` |
