# Defect register

Every evaluation defect found during the project, when, how it was quantified, and the fix.
Kept public because the paper's argument is that such defects are common and rarely reported.

| # | Defect | Effect measured | Fix | Where in paper |
|---|---|---|---|---|
| 1 | Best-epoch checkpoint selected on the held-out subject | 56.02 -> 47.39 % (+8.6 pp inflation) | inner 2-subject validation per fold | Table XVI |
| 2 | Normalization statistics from the full table incl. held-out subject | GNN ablation delta -16.91 -> -0.83 pp | per-fold train-only statistics | Table XVI |
| 3 | Shared training sets across folds (fold-independent RNG) | identical loss curves across folds | seed x 1000 + fold | Sec. III-B |
| 4 | Awake-only subjects silently dropped from training | smaller pools | 18-subject pools, 16 eval folds | Sec. III-A |
| 5 | Near-duplicate leakage from 50 % window overlap under random splits | 98.68 % on permuted labels | measured, not fixed (property of R1) | Table III |
| 6 | Whole-session calibration standardizes the scored windows | z-score rises only when baseline covers scored session | enrollment-length sweep; enrollment excluded from scoring | Sec. IV-C, V-B |
| 7 | Enrollment windows scored after calibration | +2.0 to +3.7 pp on RLDD, ~0 on UL-DD | report both; exclude on RLDD | Fig. 9 |
| 8 | Drift-context gain reproducible by elapsed time | sensor+elapsed 75.59 vs sensor+drift 75.70 | reported as temporal; RLDD clock-free replication | Sec. IV-G |
| 9 | Adapter passed timestamp columns as features (analysis-side, caught before publication) | 10 pp on an inflated feature set | features pinned to the V2 list via --features-file | REPRODUCE.md |
| 10 | Sequence-model DataLoader single-process | 48 min per fold | --num-workers 8 (7x) | patch_dataloader_v1.py |
