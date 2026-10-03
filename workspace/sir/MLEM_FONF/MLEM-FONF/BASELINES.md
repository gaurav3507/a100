# Baseline lineup (LOCKED after evidence audit, 2026-09-05)

Mixture principle: classical + training-free-learned + supervised + generative.
Every pick carries a reviewer-facing justification; every pick has a
pre-decided fallback. Changes to this list require a logged decision.

## Tier A — Analytic & classical iterative (main tables)
1. FBP                — analytic reference; mandatory in any CT paper. [ADDED by audit]
2. MLEM               — statistical baseline; SPL continuity.
3. MLEM+TV            — piecewise-smooth prior in the same loop; SPL continuity.
4. MLEM+AD            — diffusion prior in the same loop; SPL continuity.
5. PnP-ADMM(TV)       — training-free PnP splitting baseline; SPL continuity.
   (MLEM+FuzzyAD -> SUPPLEMENT only: SPL continuity, per-case-kappa fragility.)

## Tier B — Training-free learned-adjacent (thesis-critical) [ADDED by audit]
6. DIP(+TV)           — Deep Image Prior; the training-free *neural* comparator
   (Baguer et al., Inverse Problems 2020; the LoDoPaB DIP baseline). Blunts the
   sharpest reviewer attack on our thesis: "training-free NNs exist — compare."

## Tier C — Supervised learned
7. RED-CNN            — Mayo-standard CNN post-processing (Chen et al., TMI
   2017). Justified as the CNN representative by Eulig et al., Med. Phys. 2024:
   the benchmarking study finds RED-CNN outperforming many NEWER networks —
   citable armor against "why not the latest CNN".
8. Learned Primal-Dual — canonical unrolled/physics-informed method
   (Adler & Oektem, TMI 2018).

## Tier D — Generative
9. Diffusion posterior sampling (DPS-class), score model trained on Mayo
   abdomen only — serves the benchmark AND the OOD/hallucination studies.

## Optional gate (pre-decided, no debate later)
10. One transformer/SSM denoiser (UFormer / CTformer class): include IFF
    public code + weights run within 1 week AND GPU budget remains after
    Tiers C-D training; otherwise cite Eulig 2024 ("newer does not
    consistently beat older") and DenoMamba comparisons, and omit.

## Deliberate exclusions (state in paper)
- GAN denoisers (WGAN-VGG/DU-GAN): adversarial-training instability; the
  generative axis is covered by diffusion, which is the current standard.
- SART/OS variants: subsumed by MLEM as the iterative-statistical anchor.

## Fallbacks
- RED-CNN / LPD weights unavailable -> train from scratch on the locked Mayo
  split (protocols in experiments/train_*.py; RED-CNN ~2-4 h, LPD ~6-12 h).
- DPS public score unavailable -> train IDDPM-style score on Mayo abdomen
  (~1-2 days GPU); strengthens the OOD narrative.
- DIP: per-image optimization (no training); hyperparameters from Baguer 2020.
