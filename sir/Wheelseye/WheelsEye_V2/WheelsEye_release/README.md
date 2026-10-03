# WheelsEye

**Label-Free Per-Driver Calibration for Subject-Independent Multimodal Drowsiness Detection**

Code, audited fold protocols, analysis scripts and the manuscript source for the
paper submitted to *IEEE Transactions on Intelligent Transportation Systems*.

The paper shows that subject-mixed cross-validation of multimodal drowsiness
detectors measures memorization (a model scores 98.7% on labels randomly permuted
per four-minute block), and that a label-free per-driver calibration, the mean of
each driver's first eight minutes of alert driving subtracted once, raises
leave-one-subject-out accuracy on UL-DD from 47.5% to 75.9%. The effect replicates
on MePhy and on RLDD (59 subjects, own-device video, +15.6 pp with enrollment
windows excluded from scoring).

## Repository layout

```
analysis/
  calibration/    per-driver calibration module and the exact-string patches
                  that add --n-enroll, --cal-mean-only and --num-workers to the
                  experiment runner
  experiments/    calibration variants, enrollment curve, modality lattice,
                  leakage controls, MePhy replication, multi-seed runs,
                  enrollment-exclusion runs, efficiency benchmark
  stats/          paired statistics (Wilcoxon, bootstrap CI, Cohen's d_z,
                  Cliff's delta), result collectors, ordinal-error analysis
  controls/       reviewer controls: majority baselines, time-on-task,
                  within-session discrimination, wrong-state enrollment,
                  warning-level metrics, feature audit
  rldd/           RLDD replication (LOSO over 59 subjects, enrollment excluded)
paper/            main.tex (IEEEtran)
results/          canonical numbers reported in the paper (paper_numbers.json)
                  and a manifest of the result files each table is built from
tools/            verify_repo.py: structure check and paper-number regression test
docs/             defect register and protocol notes
```

The V2 experiment runner (`scripts/run_experiment.py`), feature pipeline
(`common/`) and fold builders live in this repository's existing tree; the
`analysis/` package sits beside them and is run from the repository root.

## Data

None of the datasets is redistributed here.

| Dataset | Source | Used for |
|---|---|---|
| UL-DD | Zenodo, via Bodaghi et al., *Sci. Data* 2026 | all main results |
| MePhy | Zendehbad et al., *Sci. Rep.* 2025 | physiology replication |
| UTA-RLDD | Ghoddoosian et al., CVPRW 2019 (UTA page / Kaggle mirror) | own-device video replication |

Place the UL-DD feature table at `data/processed/uldd_features.parquet`
(built by `scripts/build_feature_table.py`), then `scripts/build_v2_folds.py`
persists the three fold regimes once so that every model consumes byte-identical
splits.

## Environment

Python 3.10, PyTorch 2.6 (CUDA 12.4), LightGBM, scikit-learn, SciPy, pandas,
pyarrow. `pip install -r requirements.txt`. All scripts are Python 3.10-grammar
clean; the analysis scripts are CPU-only except the neural runs in
`run_multiseed.py`.

## Reproducing the paper

`REPRODUCE.md` maps every table and figure to the command that produces it and
the result file it reads. The short version:

```bash
# 1. apply the runner patches once (each aborts untouched on mismatch)
python analysis/calibration/patch_enroll_exact.py
python analysis/calibration/patch_meanonly_exact.py
python analysis/calibration/patch_dataloader_v1.py

# 2. headline configuration (Table II / IV, LightGBM row)
python scripts/run_experiment.py --tier 3 --model lightgbm --calibrate --n-enroll 96 --cal-mean-only

# 3. calibration variants, enrollment curve, lattice, MePhy (Tables V-VIII, X)
python analysis/experiments/exp_meanonly.py
python analysis/experiments/exp_modality_ablation.py --cal-mode mean
python analysis/experiments/exp_mephy_calibration.py

# 4. reviewer controls (Tables XII-XIII)
python analysis/controls/make_controls_csv.py
python analysis/controls/revision_controls.py --table v2_windows.csv --out revision_results \
    --k 96 --threads 32 --features-file v2_feature_names.txt --drift-file v2_drift_names.txt

# 5. verify the numbers in results/ against the paper
python tools/verify_repo.py --results data/processed/results
```

Runtimes on one A100 node: LightGBM configurations run in about a minute each;
the modality lattice (64 subsets) in about 45 minutes; the reviewer controls in
about 45 minutes; the four neural architectures at three seeds each in about
eight hours with `--num-workers 8`.

## Verification

`tools/verify_repo.py` has two modes. With no arguments it checks that every
file the paper depends on is present and parses. With `--results DIR` it opens
each result JSON, recomputes the aggregate from per-fold values and compares it
with the number printed in the paper (`results/paper_numbers.json`), reporting
any deviation above 0.05 percentage points. Continuous integration runs the
structure check on every push.

## Evaluation conventions stated in the paper

* Enrollment windows are the first *k* windows of the driver's alert session;
  calibration statistics use features only, never labels.
* Only the 122 sensor features are calibrated; the 13 drift-context features are
  already relative to the current drive and are left as-is.
* Every calibrated result can be computed with the held-out driver's enrollment
  windows scored or excluded; the paper reports both on UL-DD and uses the
  excluded figure throughout on RLDD.
* Paired tests are by held-out driver; LightGBM is deterministic under the
  headline configuration, so its dispersion is reported across folds.

## Citation

See `CITATION.cff`. Please also cite the UL-DD, MePhy and RLDD dataset papers
when using the corresponding results.

## License

MIT (see `LICENSE`). Dataset licenses are those of their respective providers.
