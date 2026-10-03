# E3 real-data gate screens

Gate detection outputs for the E3 real datasets (Perturb-seq K562 / RPE1 / Norman,
fMRI HCP / ABIDE), plus stability and positive-control artefacts under the sibling
`results/e3_stability/` and `results/e3_poscontrol/` directories.

## Retained pre-fix artefact (do not read as a current number)

`e3_k562_gate_VOID_sample_size_artifact.json` is a **deliberately retained** artefact
from **before** the sample-size-matched null correction (Lesson 2). It is the K562
screen run with the old, non-size-matched null, which inflated detection to about
**99 percent** of powered perturbations. After the size-matched-null fix the same
screen detects about **13 percent**.

It is kept as **evidence for the measurement-integrity result** (the whole point of
the precondition gate: a mismatched null manufactures detections out of sampling
noise), NOT as a live result. Do not cite the numbers in the VOID file as a current
detection rate. The `VOID` in the filename marks it as void.

## Live results

The current corrected-disjoint, BH-FDR gate screens are:

    results/e3/e3_K562_CRISPRi.json
    results/e3/e3_rpe1_gate_fixed.json
    results/e3/e3_Norman_CRISPRa_singlegene.json
    results/e3/e3_fMRI_HCP_task.json
    results/e3/e3_fMRI_ABIDE_site.json

These files include per-environment raw and BH decisions under both the historical
two-bootstrap and corrected-disjoint nulls. The primary decision is the
corrected-disjoint BH result at `q=0.05`.

`e3_k562_gate_fixed.json` predates the disjoint-null/BH rerun and is retained only
as historical evidence. Do not use it as the current K562 result.

Compact corrected-BH decision exports for K562, RPE1, and Norman are under
`results/e3_decisions/`. Their source-result checksums are embedded in each file.
