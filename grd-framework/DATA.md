# Data

The synthetic experiments (E0, E1, E2, and the E3 null-size regression check) need no
external data: they generate everything from `sim/simulator.py` and run on CPU.

The E3 real-data gate screens and the iLCS baseline read external single-cell and
neuroimaging datasets that are **not** included in this repository (they are large and
have their own licences).

## Expected location

Perturb-seq scripts resolve external data under a root given by the environment
variable

    GRD_DATA_ROOT      (default: /workspace/external)

## Expected files

Perturb-seq (CausalBench / Norman), under `discrepancy_vae/datasets/`:

    causalbench_k562.h5ad        (K562 CRISPRi)
    causalbench_rpe1.h5ad         (RPE1 CRISPRi)
    Norman2019_raw.h5ad          (Norman CRISPRa, single-gene)

Each carries per-cell perturbation labels in `obs['guide_ids']`, with the control
cells labelled by the empty string `''`. fMRI (HCP task, ABIDE site) inputs are read
by the corresponding `experiments/e3_fmri_*` and `experiments/e3_stability_hcp.py`
scripts.

When `GRD_DATA_ROOT` is explicitly set, the fMRI defaults are:

    $GRD_DATA_ROOT/meridian-identifiability/hcp/ts/*.npy
    $GRD_DATA_ROOT/ranktest-diagnostics/data/abide_harmonized.npz

They can also be overridden independently:

    GRD_HCP_TS_ROOT    directory containing the HCP task `.npy` files
    GRD_ABIDE_NPZ      path to `abide_harmonized.npz`

If no environment variables are set, the original A100 locations remain the
defaults: `/workspace/meridian-identifiability/hcp/ts` and
`/workspace/ranktest-diagnostics/data/abide_harmonized.npz`.

So, for the default root, the K562 file is expected at:

    /workspace/external/discrepancy_vae/datasets/causalbench_k562.h5ad

## Path portability

All E3 and iLCS real-data scripts use `experiments/data_paths.py`; no source edit is
needed to move the datasets. Existing committed results retain their original
provenance and are not rewritten merely because path resolution was made portable.
