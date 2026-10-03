#!/usr/bin/env bash
# Decisive follow-up queue: mean-only calibration under V2's convention. ~1 h CPU. Unbuffered output.
set -u; cd "$(dirname "$0")"; export PYTHONUNBUFFERED=1
run(){ echo; echo "=================== $1 ==================="; shift; "$@" || echo "!!! step failed -- continuing"; }
run "A. mean-only: reconciliation, enrollment curve, transductive baseline, supervised bound" python exp_meanonly.py
run "B. paired tests around mean-only 8-min"  python paired_tests.py --ref data/processed/results/tier3_cal2_mean_enroll96.json data/processed/results/tier3_cal2_meanstd_enroll96.json data/processed/results/tier3_calib_none.json data/processed/results/tier3_cal2_meanstd_whole.json data/processed/results/tier3_cal2_mean_whole.json data/processed/results/tier3_i2m2_v2_cal_smooth3_seed0.json data/processed/results/tier3_i2m2_v2_seed0.json data/processed/results/tier3_cal2_supervised_mean_enroll96.json
run "C. per-class recall + ordinal errors for mean-only" bash -c 'python collect_v2.py | grep -E "cal2_|calib_none"; python exp_ordinal_errors.py "tier3_cal2_*"'
run "D. modality lattice under mean-only (~45 min)" python exp_modality_ablation.py --cal-mode mean
run "E. lattice summary + sensor-level paired tests" bash -c 'python collect_ablation.py --suffix _mean; python paired_tests.py --ref data/processed/results/tier3_ablation_all_context_enroll96_mean.json data/processed/results/tier3_ablation_cardiac_autonomic_enroll96_mean.json data/processed/results/tier3_ablation_face_geom_fau_posture_enroll96_mean.json data/processed/results/tier3_ablation_all_enroll96_mean.json'
run "F. MePhy under mean-only" python exp_mephy_calibration.py
run "G. MePhy paired" python paired_tests.py --ref data/processed/results/mephy_loso_rest_enroll24_mean.json data/processed/results/mephy_loso_none.json data/processed/results/mephy_loso_rest_enroll24.json
echo; echo "ALL DONE $(date)"
