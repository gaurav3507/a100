#!/usr/bin/env bash
# Tier-1 experiment queue (WheelsEye_V2 root, venv active). ~15 min total; CPU only.
set -u; cd "$(dirname "$0")"
run(){ echo; echo "=================== $1 ==================="; shift; "$@" || echo "!!! step failed -- continuing"; }
run "1. modality ablation under calibration"   python exp_modality_ablation.py
run "2. calibration baselines"                 python exp_calibration_baselines.py
run "4. leakage controls (permuted labels, subject re-id)" python exp_leakage_controls.py
run "3. MePhy replication"                     python exp_mephy_calibration.py
run "8. ordinal error analysis"                python exp_ordinal_errors.py "tier3_*"
run "7. paired tests: ablation vs bio-only"    python paired_tests.py --ref data/processed/results/tier3_ablation_all_context_enroll96.json data/processed/results/tier3_ablation_bio_enroll96.json data/processed/results/tier3_ablation_all_enroll96.json
run "7. paired tests: calibration variants"    python paired_tests.py --ref data/processed/results/tier3_calib_enroll96_meanstd.json --all data/processed/results
run "7. paired tests: MePhy"                   python paired_tests.py --ref data/processed/results/mephy_loso_rest_enroll24.json data/processed/results/mephy_loso_none.json data/processed/results/mephy_loso_rest_all.json
echo; echo "ALL DONE $(date)"
