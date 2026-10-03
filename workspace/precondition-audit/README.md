# Benchmark Datasets for Multi-Environment Causal Representation Learning Do Not Meet Their Identifiability Preconditions

Gaurav Goyal, Shailendra Tiwari, Manju Khurana.
Submitted to *IEEE Journal of Biomedical and Health Informatics*.

This repository holds the analysis code, the figure-generation scripts, and
every result artefact behind the paper.

## What the audit does, and what it found

Identifiability theorems for multi-environment causal representation learning
assume three things about the data: enough distinct environments, enough samples
within each environment, and a latent dimension small enough to sit inside the
identifiable regime. The paper measures those three preconditions on six
Perturb-seq control matrices and two fMRI datasets, and reports where they are
met and where they are not.

They are mostly not met. Environment counts collapse under any realistic
per-environment sample-size floor: of 1159 K562 environments, 27 survive a
500-cell floor and one survives 8000. The observed dimension of the control
covariance sits between 224 and 456 by participation ratio at a matched sample
of 2000 control cells, well above the dimensions identifiability results
contemplate. That dimension is not an artefact of feature count, and it is not
stable under preprocessing: the rank-inverse-normal transform moves it by a
factor of 7.5 on K562 and up to 604 on Frangieh IFN-gamma, because on
zero-inflated matrices it concentrates the spectrum into a single direction. The
two fMRI datasets show the opposite failure from the Perturb-seq ones, with
ample depth and almost no breadth.

The audit is descriptive. No rank test is run on real data anywhere in this
repository, and no assumption verdict is expressed. The rank test used to
establish the detection floors is gated on simulated data only; the gates showed
it loses level control under nonlinear mixing, so it was never applied to a real
dataset.

## Reproducing

```bash
git clone https://github.com/gaurav3507/precondition-audit && cd precondition-audit
python3.11 -m venv .venv && . .venv/bin/activate && pip install -r requirements-mac.txt
bash run_all.sh
```

`run_all.sh` rebuilds every table, every figure under `paper/figures/`, and
`results/INDEX.md` from committed artefacts. It needs no raw data. It prints a
PASS/FAIL table with artefact counts before and after each stage and exits
non-zero if any stage produced nothing.

Add `--with-data` only on a machine that holds the raw matrices, after setting
`$PRECOND_DATA` and, for Norman and Frangieh, `$PRECOND_EXTERNAL`. Every loader
resolves through an explicit argument, then those two variables, then a
repo-local `data/` directory; a missing file is fatal and names every path
tried.

## Repository map

| directory | contents |
|---|---|
| `scripts/` | Analysis code. Loaders and profilers (`03`, `40`, `41`, `85`), the rank-test core and simulator (`80`, `81`, `83`), the artefact schema gate (`84`), table, figure and index builders (`87`–`92`), and the A100 audit, regeneration and control jobs (`93`, `94`, `96`, `99`, `100`). |
| `results/descriptives/` | Per-dataset profiles for all eight datasets: environment attrition, cells per environment, and the control-covariance dimension estimates. Each carries a compressed `__spectra.npz` sidecar holding the eigenvalue arrays. |
| `results/preprocessing/` | The four-arm preprocessing sweep at a matched sample of 2000 control cells, with its own eigenvalue sidecars. |
| `results/intrinsic/` | Nonlinear intrinsic-dimension estimates, one artefact per condition, each carrying the synthetic calibration the estimates are read against. |
| `results/gates/` | Simulator and oracle gate artefacts, organised `<statistic>/<gate>/<timestamp>.json`. Four statistics are present, of which `lfc` is the final one; `diy_retired`, `cfa_kappa` and `cft` are kept so the record of what was tried and discarded is complete. |
| `results/INDEX.md` | Generated index of every artefact, with status and provenance. Regenerate with `python scripts/89_make_index.py`; do not hand-edit. |
| `paper/tables/` | Generated tables in Markdown and CSV. |
| `paper/figures/` | Generated eigenvalue-spectrum panels. |
| `manuscript/` | The LaTeX source and its figures. |
| `docs/` | Reproduction notes, data sources, and known limitations. |

## Datasets

No data is redistributed here. None of it is ours to redistribute. Each dataset
must be obtained from its original source.

| dataset | modality | source | how to obtain | DOI |
|---|---|---|---|---|
| Replogle K562 | Perturb-seq (CRISPRi) | Replogle et al. 2022, *Cell* | CausalBench release archive | 10.1016/j.cell.2022.05.013 |
| Replogle RPE1 | Perturb-seq (CRISPRi) | Replogle et al. 2022, *Cell* | CausalBench release archive | 10.1016/j.cell.2022.05.013 |
| Norman 2019 | Perturb-seq (CRISPRa) | Norman et al. 2019, *Science* | GEO accession GSE133344 | 10.1126/science.aax4438 |
| Frangieh co-culture | Perturb-CITE-seq | Frangieh et al. 2021, *Nat. Genet.* | Single Cell Portal SCP1064 | 10.1038/s41588-021-00779-1 |
| Frangieh control | Perturb-CITE-seq | Frangieh et al. 2021, *Nat. Genet.* | Single Cell Portal SCP1064 | 10.1038/s41588-021-00779-1 |
| Frangieh IFN-gamma | Perturb-CITE-seq | Frangieh et al. 2021, *Nat. Genet.* | Single Cell Portal SCP1064 | 10.1038/s41588-021-00779-1 |
| HCP | Task fMRI | Human Connectome Project | Requires registration and acceptance of the HCP data use terms before download. Not obtainable anonymously. | 10.1016/j.neuroimage.2013.05.041 |
| ABIDE | Resting-state fMRI | ABIDE Preprocessed Initiative | Public download | 10.1038/mp.2013.78 |

The three Frangieh rows are one deposit read as three arms. Arms are profiled
separately and never pooled: the arm effect dominates and would inflate every
downstream quantity.

`docs/DATA_SOURCES.md` records what each loader does, and states plainly that
the upstream normalisation of all six transcriptomic matrices is undocumented in
the original deposits.

## Artefact schema

Every result file carries a `meta` block. `scripts/84_results_io.py` refuses to
write one that is missing a mandatory field, so an artefact whose provenance is
incomplete does not reach disk. A worked example, from
`results/descriptives/2026-08-11T06-12-06Z__k562.json`:

```json
{
  "meta": {
    "statistic": "descriptive",
    "gate": "descriptives",
    "git_commit": "8181090",
    "timestamp": "2026-08-11T06:12:06Z",
    "status": "CURRENT",
    "superseded_by": null,
    "migrated_from": null,
    "config": {
      "alpha": null, "B": null, "n_e": null,
      "d": null, "d_latent": null, "D": null, "n_env": null,
      "seeds": [0], "draws_per_point": null,
      "extra": {
        "dataset": "k562",
        "spec_caps": [2000, 8000],
        "platform_tag": "Linux-x86_64-py3.10.12"
      }
    },
    "versions": {
      "python": "3.10.12", "numpy": "2.2.6", "scipy": "1.15.3",
      "pandas": "2.3.3", "sklearn": "1.7.2",
      "scanpy": "1.11.5", "anndata": "0.11.4",
      "platform": {
        "system": "Linux", "machine": "x86_64",
        "python_implementation": "CPython",
        "numpy_blas": "scipy-openblas"
      }
    }
  }
}
```

`status` is one of `CURRENT`, `SUPERSEDED` or
`HISTORICAL-FOR-COMPARISON-TABLE`. Quote only `CURRENT` rows. `git_commit`
records the commit that produced the artefact, which is why this repository's
history is never rewritten: those hashes have to keep resolving.

Coverage of that schema is uneven, and the gap is recorded rather than papered
over. All 42 result files carry `meta.versions`. The 14 real-data artefacts,
under `results/descriptives/` and `results/preprocessing/`, additionally carry
the platform block and the BLAS backend. The 28 simulator gate artefacts do not:
they were written before the platform block was added to the schema, and they
are not regenerated, because regenerating them would change the `git_commit`
they record. Seven artefacts carry an input-data fingerprint; the rest are
simulator output, for which there is no input file to fingerprint.

## Two environments, and why both are pinned

The real-data artefacts were produced on an A100 node under CPython 3.10.12 with
scipy-openblas. The simulator gates, tables and figures were produced on an
Apple Silicon Mac under CPython 3.11.15 with Accelerate. numpy differs (2.2.6
against 2.4.6) and scipy differs (1.15.3 against 1.17.1).

That difference is load-bearing rather than incidental. Every quantity in the
dimension tables comes from an SVD, and SVD results depend on the BLAS backend
at the last digits. `requirements-a100.txt` and `requirements-mac.txt` are kept
separate for that reason. `python scripts/85_dataset_descriptives.py --selftest`
checks cross-machine agreement against constants pinned from a fixed synthetic
matrix, reporting a data-stream difference separately from a BLAS difference in
the SVD, so a version change is caught rather than absorbed.

## Known gap in the record

The two power curves in Section III-D, the soft and hard detection floors of
8000 and 125 cells, were produced by driver scripts that were not committed at
the time of the run. The artefacts themselves are committed, under
`results/gates/lfc/power_soft/` and `results/gates/lfc/power_hard/`, with their
full configuration, and the simulator that generated them is
`scripts/81_ranktest_oracle.py`, which is committed and unchanged. What is
missing is the loop that swept `n_e` and tabulated the result.

`scripts/93_regen_power_curves.py` closes that gap going forward. It imports the
same simulator and calls its `gate1` directly rather than reimplementing
anything, regenerates both curves, and emits a point-by-point comparison against
the committed ones. That comparison is reported and deliberately not gated: at
ten seeds per point a difference of a tenth or two is sampling noise, and the
RNG stream position of the original uncommitted run cannot be recovered.

Read `docs/KNOWN_LIMITATIONS.md` before quoting any number.

## Licence and citation

Code is MIT licensed; see `LICENSE`. Citation metadata is in `CITATION.cff`. The
datasets are not covered by that licence and remain under the terms of their
original deposits.
