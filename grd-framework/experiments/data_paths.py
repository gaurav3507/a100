"""Portable paths for GRD real-data experiments.

Perturb-seq data uses ``GRD_DATA_ROOT`` and keeps ``/workspace/external`` as
the legacy default. The historical fMRI inputs lived outside that tree, so they
also accept explicit overrides. When ``GRD_DATA_ROOT`` is set, their default
locations move under that root as documented in DATA.md.
"""

import os
from pathlib import Path


LEGACY_DATA_ROOT = Path("/workspace/external")
LEGACY_HCP_TS_ROOT = Path("/workspace/meridian-identifiability/hcp/ts")
LEGACY_ABIDE_NPZ = Path(
    "/workspace/ranktest-diagnostics/data/abide_harmonized.npz")


def data_root():
    return Path(os.environ.get("GRD_DATA_ROOT", LEGACY_DATA_ROOT)).expanduser()


def perturbseq_path(filename):
    return data_root() / "discrepancy_vae" / "datasets" / filename


def hcp_ts_root():
    explicit = os.environ.get("GRD_HCP_TS_ROOT")
    if explicit:
        return Path(explicit).expanduser()
    if "GRD_DATA_ROOT" in os.environ:
        return data_root() / "meridian-identifiability" / "hcp" / "ts"
    return LEGACY_HCP_TS_ROOT


def abide_npz_path():
    explicit = os.environ.get("GRD_ABIDE_NPZ")
    if explicit:
        return Path(explicit).expanduser()
    if "GRD_DATA_ROOT" in os.environ:
        return (data_root() / "ranktest-diagnostics" / "data" /
                "abide_harmonized.npz")
    return LEGACY_ABIDE_NPZ


def real_data_inputs():
    """Named paths used by the real-data regeneration and manifest scripts."""
    return {
        "K562": perturbseq_path("causalbench_k562.h5ad"),
        "RPE1": perturbseq_path("causalbench_rpe1.h5ad"),
        "Norman": perturbseq_path("Norman2019_raw.h5ad"),
        "ABIDE": abide_npz_path(),
        "HCP": hcp_ts_root(),
    }
