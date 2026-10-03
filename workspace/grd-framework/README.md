# Gate, Recover, Discover (GRD)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Status: under review](https://img.shields.io/badge/status-under%20review-orange.svg)](#citation)

Reference implementation and committed result artifacts for

> **Gate, Recover, Discover: Testing Identifiability Preconditions Before Causal Representation Learning.**
> Gaurav Goyal, Shailendra Tiwari, and Manju. Department of Computer Science and Engineering, Thapar Institute of Engineering and Technology, Patiala, India. Under review, 2026.

GRD reframes latent causal recovery as a **selective decision problem**. Modern causal
representation learning (CRL) results identify latent variables and their graph only under
explicit conditions on environments, power, mechanism shifts, and measurement stability, yet
an estimator returns *something* for every input. GRD adds a data-side decision layer:

1. **Gate** — test estimator-coupled preconditions with a size-matched precision-difference
   null and BH-FDR within each declared family; return a certified environment set `C` and a
   recoverable resolution `d_rec = min(|C|, d*)`.
2. **Recover** — run the identifiable backbone **only** on `C`; withhold uncertified latents
   rather than numerically completing them.
3. **Discover** — report a partial graph with explicit *decided* and *undecided* edge states.

The verdict is the first output: `PROCEED`, `PROCEED-CAPPED`, or `ABSTAIN`. Abstention and
unresolved attribution are treated as scientific results, not failures.

---

## Headline results

Every number below is read directly from the committed JSON artifacts in `results/`.

**Synthetic calibration** (ten seeds, `d = 5` latents). Under environment starvation the gate
first restricts at `m = 4` while recovery MCC is still `0.9908`, crossing the `0.90` bar only
at `m = 2`: the gate is conservative, not fit to the error curve.

| Precondition violated | Gate first restricts at | Ungated MCC there | Paper table |
|---|---|---:|:--:|
| Environments (`m`)        | `m = 4`       | 0.9908 | 2 |
| Power (`n_e`)             | `n_e = 40`    | 0.9899 | 6 |
| Signal (`scale`)          | `scale = 0.7` | 0.9092 | 7 |
| Measurement contamination | not caught (declared failure class) | — | 8 |

**Real Perturb-seq screens** (corrected disjoint null, BH-FDR at `q = 0.05`). GRD **declines to
certify** the large majority of environments; 584 of 632 powered environments fail the primary
screen.

| Dataset | Envs. | Raw reject | BH reject | BH top-2 control-PC alignment |
|---|---:|---:|---:|---:|
| K562 (CRISPRi)   | 385 | 73 | **0**  | –      |
| RPE1 (CRISPRi)   | 146 | 77 | **48** | 0.7288 |
| Norman (CRISPRa) | 101 | 11 | **0**  | –      |

RPE1 has the highest certification rate, but all 20 structured control splits also fire and the
detected shifts concentrate in the leading control PCs (median 0.7288), so the detections track
systematic control heterogeneity. Attribution (mechanism vs measurement) is reported as
**unresolved**; no real-data latent graph is claimed.

**Ungated baseline (iLCS).** The nominal rule fires on 100% of pure-control resamples across all
three datasets. A size-matched null reduces this to 12.0% / 8.0% / 6.7% (K562 / RPE1 / Norman)
but does not eliminate structured-control firing.

---

## Repository layout

```
src/           Gate / Recover / Discover modules and the precision-difference readout
sim/           multi-environment linear-SCM simulator
experiments/   one runnable script per experiment (E0-E3, baselines, certificate)
results/       committed JSON artifacts (source of every number in the paper)
paper/         figure-generation script and the six manuscript figures
notes/         track sheet and the Davis-Kahan certificate note
docs/          framework design document
```

## Reproduction

Synthetic experiments need only the dependencies in `pyproject.toml` (NumPy, SciPy,
scikit-learn; Python >= 3.10). Real-data paths are configured through the `GRD_DATA_ROOT`
environment variable; `DATA.md` documents the expected filenames. The source datasets
(Replogle Perturb-seq via CausalBench, Norman, HCP, ABIDE) are **not** redistributed here.

```bash
pip install -e .

# print the full E3 regeneration plan without running it
python experiments/reproduce_e3.py

# execute E3 with an external-data SHA-256 manifest
python experiments/reproduce_e3.py --execute

# add the three-seed iLCS baseline
python experiments/reproduce_e3.py --execute --include-ilcs

# regenerate the manuscript figures from the committed JSONs
python paper/make_all_figures.py --figure all
```

Every stochastic script takes an explicit seed. Compact gate decisions in `results/e3_decisions/`
are linked to their full E3 source JSON by SHA-256, and `results/repro/data_manifest.json`
records external-data hashes.

## Artifact map

| Section of the paper | Artifacts |
|---|---|
| Clean and capped pipelines (Table 1)                 | `results/e1_full`, `results/e1_capped` |
| Violation sweeps and attribution (Fig 2, Tables 6-8, 10-11) | `results/e2`, `e2b`, `e2c`, `e2d_attribution` |
| Finite-sample direction certificate (Fig 7, Table 17) | `results/e2e_certificate` |
| Corrected real-data screens (Figs 3-4, Tables 3-4, 12-13) | `results/e3`, `e3_stability`, `e3_attribution` |
| Positive control (Fig 6, Table 15)                   | `results/e3_poscontrol` |
| iLCS baseline (Fig 5, Tables 5, 14)                  | `results/ilcs_baseline/aggregate.json` |

The iLCS detector in `experiments/ilcs_baseline.py` is reimplemented from Algorithm 1 of
Chen et al. 2024 (arXiv 2410.24059) because the original repository is unavailable; it runs on
the same `D = 10` control-fit projected inputs as the gate, with a size-matched null.

## Scope and limitations

The implemented backbone assumes linear mixing, Gaussian latent noise, and known atomic
reduced-variance hard-intervention targets. The gate is re-derived per backbone; this is one
exact regime, not a universal certificate. The real pipeline runs in a control-fit
10-dimensional projection, so linear exactness holds only there. The direction certificate is
valid but vacuous below `n ~ 16,000`. See Section 9 of the paper.

## Citation

If you use this code or the committed artifacts, please cite the paper. A machine-readable entry
is in [`CITATION.cff`](CITATION.cff) (rendered as a "Cite this repository" button on GitHub).

```bibtex
@unpublished{goyal2026grd,
  author = {Goyal, Gaurav and Tiwari, Shailendra and Khurana, Manju},
  title  = {Gate, Recover, Discover: Testing Identifiability Preconditions
            Before Causal Representation Learning},
  year   = {2026},
  note   = {Under review. \url{https://github.com/gaurav3507/grd-framework}}
}
```

## License

MIT, see [`LICENSE`](LICENSE). The source datasets retain their own licenses and are not
redistributed here.
