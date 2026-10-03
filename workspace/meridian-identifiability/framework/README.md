# meridian-causalbench

An environment-validity screen for causal representation learning (CRL) applied
to single-cell perturbation datasets, together with a benchmarked-negative
evaluation of a state-of-the-art CRL model on CausalBench K562.

Companion code for:

> Goyal, G. *Environment-validity screen for causal representation learning: a
> benchmarked-negative case study on CausalBench.* IEEE Transactions on
> Computational Biology and Bioinformatics, under review.

## Overview

Causal representation learning models promise to recover mechanistic latent
variables from interventional data. Whether a particular dataset supplies
enough mechanism-shift signal to support such recovery is rarely tested in
advance. This repository contains:

1. **An environment-validity screen** that measures, before any model is
   trained, whether a dataset has (a) a mean-shift signal above a
   control-derived null and (b) enough effective latent dimensions above 2×
   noise. Validated on three datasets: CausalBench K562 (positive),
   Norman 2019 (positive), HCP fMRI (negative).
2. **A benchmarked-negative evaluation** on CausalBench K562 (Replogle 2022
   Perturb-seq CRISPRi) of `CMVAE_CB`, a subclass of the discrepancy-VAE
   (Zhang et al., NeurIPS 2023), trained for zero-shot intervention
   prediction. Best median R² across 77 held-out perturbations is +0.010,
   below the linear-ridge baseline of 0.226 and the mean baseline of 0.129,
   while the split-half ceiling is 0.528. Removing the causal DAG layer
   changes nothing, reproducing the authors' own Table 3 (causal layer
   contribution ≈ 0) on this task.

## Headline numbers

Environment-validity screen (`causalbench/results/screen/`):

| dataset          | mean-shift ratio (n=200, d=10) | effective dim (participation ratio) | step-0 pairs gate |
|------------------|-----------------------------:|------------------------------------:|------------------:|
| CausalBench K562 | 2.0 – 5.6                    | ~15                                 | 0.96 – 1.05       |
| Norman 2019      | 3.19                         | 13.2                                | 0.996 – 1.029     |
| HCP fMRI         | 1.00                         | 3 (group only)                      | passed            |

CausalBench K562 zero-shot baselines (77 held-out perturbations,
median R² vs no-effect null, target column zeroed;
`results/zeroshot_canonical/k562.json`):

| method       | median R² |
|--------------|----------:|
| zero         |    +0.000 |
| global_mean  |    +0.129 |
| corr_prop    |    −0.188 |
| nn_corr      |    −0.243 |
| ridge_basis  |    +0.226 |
| CEILING (split-half of held-out) | +0.528 |

Headroom = 0.399 R² units; ridge captures 24.2%.

CMVAE_CB and no-DAG-ablation results (best median R² over 100 training
epochs; `results/model/SUMMARY.json`):

| variant                       | best median R² | beats gmean | beats ridge |
|-------------------------------|---------------:|------------:|------------:|
| full model, shift-init        |        +0.0097 |          0% |          0% |
| full model, random-init       |        −0.0057 |          0% |          0% |
| no-DAG ablation, shift-init   |        −0.0422 |          0% |          0% |
| no-DAG ablation, random-init  |        +0.0130 |          0% |          0% |

The DAG condition number stays in [1.0, 1.8] across training
(`results/model/dag_stability.csv`), ruling out matrix-inverse blow-up as the
failure mode.

## Requirements

Two isolated Python 3.10 environments are used:

- **`dvae`**: PyTorch 2.x + the discrepancy-VAE runtime. Used by the CRL
  training pipeline (`model/cb_train.py`, `scripts/27_no_causal_ablation.py`,
  `scripts/29_dag_condnum_log.py`).
- **`cb`**: NumPy, SciPy, scikit-learn, anndata, scanpy. Used by the
  environment-validity screen and the baseline / diagnostic scripts.

Installation is not scripted here; the two environments are constructed with
the dependencies of `discrepancy_vae` and `causalbench` respectively (see
their `pyproject.toml` / `requirements.txt` in the upstream repositories at
the pinned commits).

## Data

Datasets are not distributed with this repository (`*.npz`, `*.h5ad`, `*.pt`
are gitignored). They are all publicly available:

- **CausalBench** (Replogle 2022 Perturb-seq CRISPRi K562 and RPE1):
  obtained via the CausalBench repository. See
  `vendor_analysis/CAUSALBENCH_COMMIT.txt` for the pinned commit.
- **Norman 2019** (Perturb-seq CRISPR activation, K562): the CPA-preprocessed
  h5ad shipped with the discrepancy-VAE repository. See
  `vendor_analysis/DVAE_COMMIT.txt` for the pinned commit.

Reproduction assumes these paths on the analysis machine:

    causalbench/data/dataset_k562.npz
    causalbench/data/dataset_rpe1.npz
    external/discrepancy_vae/datasets/causalbench_k562.h5ad
    external/discrepancy_vae/datasets/Norman2019_raw.h5ad

## The canonical split

`results/splits/k562_zeroshot_split.json` holds out 77 of 385 usable K562
perturbations at the whole-gene level. It is the **single source of truth**
for every zero-shot number here. It must not be regenerated: a different
random seed or a different construction order produces a different held-out
set, which silently invalidates every comparison against the baselines. All
scripts read this file; they never rebuild it.

## Reproducing the headline numbers

Activate the appropriate environment before each command.

**Baselines (ridge, ceiling, global mean, correlation methods):**

    source /workspace/venvs/cb/bin/activate
    python scripts/25_zeroshot_baselines.py

Writes `results/zeroshot_canonical/k562.json` and `rpe1.json`.

**Full CMVAE_CB training:**

    source /workspace/venvs/dvae/bin/activate
    python model/cb_train.py --init shift  --epochs 100
    python model/cb_train.py --init random --epochs 100

Writes `results/model/zeroshot_shift.json` and `zeroshot_random.json`.

**No-DAG ablation:**

    python scripts/27_no_causal_ablation.py --init shift  --epochs 100
    python scripts/27_no_causal_ablation.py --init random --epochs 100
    python scripts/28_export_nodag_results.py

Writes `results/model/zeroshot_nodag_shift.json` and `_random.json`.

**DAG numerical-stability diagnostic:**

    python scripts/29_dag_condnum_log.py --epochs 15 --log_every 50

Writes `results/model/dag_stability.csv`.

**Consolidated summary table:**

    python scripts/30_results_summary.py

Writes `results/model/SUMMARY.json`.

**Environment-validity screen on CausalBench K562 and RPE1:**

    source /workspace/venvs/cb/bin/activate
    python causalbench/scripts/03_screen.py
    python causalbench/scripts/04_spectrum.py

Writes per-configuration JSONs under `causalbench/results/screen/` and
`causalbench/results/spectrum/`.

**Environment-validity screen on Norman 2019 (positive control):**

    python causalbench/scripts/40_screen_norman.py

Writes `causalbench/results/screen/norman.json` and
`causalbench/results/spectrum/norman.json`.

## Repository structure

    model/
      cb_data.py                    CausalBench loader, perturbation-level zero-shot split
      cb_init.py                    shift-basis + nonlinear-decoder initialization fit
      cb_model.py                   CMVAE_CB, subclass of discrepancy-VAE's CMVAE
      cb_train.py                   training loop; --init {shift,random}
    scripts/
      10_cf_estimator.py            closed-form shift-basis test on real data
      12_eval_calibration.py        eval network scoring at matched edge budget
      13_shift_baseline.py          DE baseline for CausalBench's statistical metric
      14_biological_eval.py         CORUM / STRING / ChIP-seq precision at budget
      15_zeroshot.py                (superseded; see 25_zeroshot_baselines.py)
      20_to_anndata.py              build causalbench_k562.h5ad + canonical split
      25_zeroshot_baselines.py      reference baselines (0.129, 0.226, 0.528)
      27_no_causal_ablation.py      no-DAG ablation training
      28_export_nodag_results.py    export ablation results as canonical JSON
      29_dag_condnum_log.py         DAG condition-number log
      30_results_summary.py         SUMMARY.json
      run_simu_repro.py             discrepancy-VAE simulation reproduction
      run_simu_seeds.py             discrepancy-VAE simulation seed sweep
    causalbench/
      scripts/
        03_screen.py                mean-shift ratio + step-0 gate
        04_spectrum.py              effective dimensionality (dims above 2x noise)
        40_screen_norman.py         same screen on Norman 2019
        40b_fix_step0_gate.py       one-shot fix: step0_pass uses mean_ratio_pairs
      results/
        screen/                     screen output per dataset
        spectrum/                   spectrum output per dataset
    results/
      splits/                       canonical held-out perturbation splits
      zeroshot_canonical/           baselines (source of truth)
      model/                        CMVAE_CB results, no-DAG ablation, SUMMARY.json
      zeroshot/, bio_eval/, eval_calibration/, cf_estimator/
                                    intermediate evaluations
    vendor_analysis/
      DVAE_COMMIT.txt               pinned discrepancy-VAE commit
      CAUSALBENCH_COMMIT.txt        pinned CausalBench commit
      my_changes.patch              path-only edits applied to discrepancy-VAE
      scripts/, results/            vendor scripts and their outputs
    COMMITS.txt                     upstream commit hashes (mirror of vendor_analysis)

## Third-party code

- **discrepancy-VAE** (Zhang et al., NeurIPS 2023): `CMVAE_CB` is a subclass
  of their `CMVAE`; our training loop wraps their `loss_function`. The source
  is not modified beyond dataset-path fixes, captured in
  `vendor_analysis/my_changes.patch`. Pinned commit in
  `vendor_analysis/DVAE_COMMIT.txt` and `COMMITS.txt`.
- **CausalBench** (Chevalley et al., NeurIPS 2023): data source and the origin
  of the reference benchmark this work evaluates against. Pinned commit in
  `vendor_analysis/CAUSALBENCH_COMMIT.txt` and `COMMITS.txt`.

Neither third-party repository is redistributed here; both must be cloned
separately at the pinned commits.

## Citation

If you use this code or the environment-validity screen, please cite the
accompanying manuscript (see `CITATION.cff`). Details will be updated on
acceptance.

## License

MIT. See `LICENSE`.
