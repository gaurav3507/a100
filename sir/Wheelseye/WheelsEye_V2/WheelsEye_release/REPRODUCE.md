# Reproducing every table and figure

All commands run from the repository root with the venv active. Result files
land in `data/processed/results/`. Where two pipelines produced a table, both
are named; the paper states the agreement between them (Sec. III-C).

| Paper item | Command | Result file(s) read |
|---|---|---|
| Table I (base-paper reproduction) | `python scripts/reproduce_paper_protocol.py` | `repro_svm.json`, `repro_rf.json` |
| Table II / Fig. 2 (protocol ladder) | `run_experiment.py --tier {1,2,3} --model lightgbm [--calibrate --n-enroll 96 --cal-mean-only]` | `tier{1,2,3}_lightgbm_v2*.json` |
| Table III / Fig. 3 (leakage controls) | `python analysis/experiments/exp_leakage_controls.py` | `leak_R{1,2}_{true,permuted}_labels.json`, `leak_R{1,2}_subject_reid.json` |
| Table IV / Fig. 4 (architectures) | `python analysis/experiments/run_multiseed.py` | `tier3_{i2m2,stacked,gnn_v2_nognn,transformer}_v2_cal_enroll96_mean_seed{0,1,2}.json` |
| Table V (calibration variants) | `python analysis/experiments/exp_meanonly.py` + `exp_calibration_baselines.py` | `tier3_cal2_*.json`, `tier3_calib_*.json` |
| Table VI / Fig. 5 (enrollment curve) | `python analysis/experiments/exp_meanonly.py` | `tier3_cal2_mean_enroll{12..460}.json`, `tier3_cal2_mean_whole.json` |
| Table VII / lattice | `python analysis/experiments/exp_modality_ablation.py --cal-mode mean` then `python analysis/stats/collect_ablation.py --suffix _mean` | `tier3_ablation_*_enroll96_mean.json` |
| Table VIII (per-subject) | `python analysis/stats/paired_tests.py --ref ...` per-fold values | `tier3_lightgbm_v2_seed0.json`, `tier3_cal2_mean_enroll96.json` |
| Table IX / Fig. 7 (per-class, ordinal) | `python analysis/stats/collect_v2.py`; `python analysis/experiments/exp_ordinal_errors.py "tier3_*"` | confusion matrices in the above |
| Table X (MePhy) | `python analysis/experiments/exp_mephy_calibration.py` | `mephy_loso_*.json` |
| Table XI / Fig. 8 (RLDD) | `python analysis/rldd/rldd_calibration.py --both` | `features_v2/rldd_calibration_results.json` |
| Fig. 9 (enrollment-scoring inflation) | `python analysis/experiments/uldd_enroll_excluded.py --k 96` + RLDD `--both` run | `uldd_enrollment_excluded.json`, RLDD results |
| Table XII (time-on-task, session, wrong-state) | `python analysis/controls/revision_controls.py --table v2_windows.csv --out revision_results --features-file v2_feature_names.txt --drift-file v2_drift_names.txt` | `revision_results/results.json` |
| Table XIII (warning-level) | same run | same file, key `warning_level` |
| Table XIV (paired statistics) | `python analysis/stats/paired_tests.py --ref <file> --all data/processed/results` | any pair of result JSONs |
| Table XV (efficiency) | `python analysis/experiments/bench_efficiency.py` | `efficiency_lightgbm.json` |
| Table XVI (V1 leakage audits) | v1 repository (`track_a_train.py` with/without leak fixes) | v1 `track_a_results/*.json` |

## Producing the controls CSV

`revision_controls.py` expects a CSV with `subject`, `session` in
{alert, drowsy}, `t` in seconds and `label` in {Low, Medium, High}.
`make_controls_csv.py` converts the V2 parquet:

```bash
python analysis/controls/make_controls_csv.py            # writes v2_windows.csv
python - <<'PY'
from common.dataset import V2_FEATURES
from common.context_features import CONTEXT_FEATURES
open("v2_feature_names.txt","w").write("\n".join(list(V2_FEATURES)+list(CONTEXT_FEATURES))+"\n")
open("v2_drift_names.txt","w").write("\n".join(CONTEXT_FEATURES)+"\n")
PY
```

Always pass `--features-file` and `--drift-file`; without them the script uses
every numeric column in the CSV, which includes timestamp metadata that the V2
feature list excludes.

## Checking the numbers

```bash
python tools/verify_repo.py                               # structure only
python tools/verify_repo.py --results data/processed/results
```

The second form recomputes each aggregate from its per-fold values and compares
with `results/paper_numbers.json`. Tolerance is 0.05 percentage points.
