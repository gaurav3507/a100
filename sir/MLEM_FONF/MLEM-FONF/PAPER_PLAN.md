# Paper plan (locked 2026-09-07)

**Title.** A-FONF: Adaptive Fractional-Order Notch Filtering for Robust and
Self-Tuning Low-Dose CT Reconstruction

**Target venue.** IEEE JBHI — LOCKED 2026-09-09. The TMI decision run
returned REVIEW: at dose-matched realistic geometry a per-dose-tuned TV
baseline leads A-FONF by 0.12-0.67 dB, so the sparse-view advantage is
reported as regime-specific rather than universal.

## Primary contributions
C1. **Fully automatic reconstruction.** Poisson-discrepancy selection of the
    fractional order (<= 0.15 dB of oracle over 7 content x dose configs) +
    robust two-gate notch rule with acquisition-band guard (263/263
    specificity and sensitivity on held-out clinical DICOM) + damped
    multiplicative step; frozen as one algorithm, zero run-time tuning.
C2. **Stability theory that predicts real-data behaviour.** Symmetric,
    nonexpansive spectral operator and monotone damped iteration; predicted
    the observed failure taxonomy (0/200 catastrophic reconstructions at
    photon starvation vs 3-16/200 for nonlinear spatial priors) and the
    fixed-budget divergence of unregularized MLEM (-14 dB vs +20 dB, fan).
C3. **Invariance to distribution shift (HEADLINE).** Zero re-tuning across
    datasets: 26.57 -> 26.73 dB (26.19 -> 26.19 at 1500), while Mayo-trained
    RED-CNN / LPD / DPS lose 7.0 / 3.7 / 2.7 dB and the ranking inverts;
    degradation orders by each architecture's physics content.
C4. **Statistically rigorous released benchmark.** 10 methods x 4 doses x
    2 datasets x 2 geometries x 10 realizations, paired two-level Wilcoxon
    (seed-pairs and slice-level), > 20k rows, code + logs + audit trail.

## Scope decisions
- **CHO / task-based evaluation: INCLUDED** (3-day plan: implement + oracle
  test; overnight ensemble; analysis + d' figure).
- **TomoBank ring correction: INCLUDED**, primary tomo_00072 (soft tissue,
  severe rings), replication tomo_00076, stress tomo_00071 (broad rings).
- **Transformer denoiser: OMITTED** with stated reason (offline cluster;
  Eulig 2024 shows newer nets do not consistently beat RED-CNN).
- **Honest-scope box (our words, not a reviewer's):** supervised methods win
  in-distribution (+1.6 dB); photon-starvation regime belongs to TV/PnP/LPD;
  selection stage costs ~90 s CPU once per acquisition.

## Gate outcome (decision run, closed)
Executed: 512^2, 736 bins, 576 views, dose-matched incidents (41/82/326),
5 slices x 3 seeds, TV tuned per dose on the validation patient.
Result: TV 27.65/27.74/29.09 vs A-FONF 26.98/27.62/28.78 dB. Verdict
REVIEW -> JBHI. Reported honestly, together with the tuning evidence: the
TV weight had to move 0.0065 -> 0.002 across dose (and 0.05 -> 0.0065
across geometry, a 4-5 dB penalty if left at its sparse-view value) while
A-FONF selected alpha automatically at every point.

## Figures
F1 dose curves (PSNR/SSIM x 3 settings) | F2 distribution-shift slope chart
F3 robustness violins | F4 stability trajectories | F5 qualitative grid
(ROI zoom + difference maps) | F6 CHO d' vs dose | F7 TomoBank rings
