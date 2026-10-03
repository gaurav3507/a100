# GRD figure plan

Every quantitative figure must be generated from committed JSON under
`results/`. Vector PDF is the submission format; PNG copies are local previews
and remain ignored by Git.

## Main figures

1. **GRD and the real-data decision surface.** Framework schematic, corrected
   raw detectability versus descriptive control-PC alignment, and primary
   BH-FDR-certified fractions. This immediately separates detection from
   trustworthiness and from the final actionable set.
2. **Synthetic precondition violations.** Non-tautological starvation plus
   power and weak-signal sweeps, showing gate restriction relative to ungated
   recovery degradation. Measurement contamination is shown as the declared
   failure class, not folded into the successful calibration panels.
3. **Why RPE1 is not a success case.** Perturbations versus random and
   structured control splits, PC-energy alignment, and the real-data
   attribution non-result.
4. **Closest-prior-method stress test.** Naive and size-matched iLCS firing on
   pure controls, structured controls, and perturbations, together with ICA
   non-convergence.

## Supplementary figures

- Corrected-null geometry and positive-control sensitivity.
- Genuine capped end-to-end pipeline.
- Synthetic attribution operating characteristics and projected-rescue NO_GO.
- Davis-Kahan certificate feasibility and its finite-sample vacuity.

## Figure 1 draft caption

**Gate-Recover-Discover tests whether recovery is supportable before reporting a
latent graph.** (a) The gate evaluates power, detectable mechanism shift, and
confound diagnostics; recovery is restricted to certified environments, and
discovery reports unsupported edges as undecided. (b) Across the real-data
panels, raw detectability and alignment with leading control PCs are distinct
quantities. RPE1 has the largest raw detectable fraction among the Perturb-seq
datasets but also the strongest alignment with control heterogeneity; all 20
structured control splits fire. Alignment is a descriptive diagnostic, not a
formal confound threshold. (c) The primary BH-FDR decision retains 48/146 RPE1
perturbations, none in K562 or Norman, 5/6 HCP task environments, and 13/13
ABIDE sites. The fMRI environments are dataset/group shifts and should not be
interpreted as perturbation-mechanism certification.
