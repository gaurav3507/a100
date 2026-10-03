import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from mlem_fonf import (build_system_matrix, poisson_sinogram, mlem, mlem_fonf,
                       mlem_spatial, pnp_admm_tv, all_metrics, PHANTOMS)

INCIDENT = 60          # photon level of the low-dose regime (calibrated)
N, BINS, ANG = 256, 64, 90

def setup(phantom="shepp_logan", seed=0):
    A = build_system_matrix(N, BINS, ANG)
    ref = PHANTOMS[phantom](N)
    g = poisson_sinogram(A, ref, INCIDENT, np.random.default_rng(seed))
    return A, ref, g

METHODS = {
    "MLEM":         lambda g, A, it, cb=None: mlem(g, A, N, it, callback=cb),
    "MLEM+TV":      lambda g, A, it, cb=None: mlem_spatial(g, A, N, "tv", it, callback=cb),
    "MLEM+AD":      lambda g, A, it, cb=None: mlem_spatial(g, A, N, "ad", it, callback=cb),
    "MLEM+FuzzyAD": lambda g, A, it, cb=None: mlem_spatial(g, A, N, "fuzzyad", it, callback=cb),
    "PnP-ADMM(TV)": lambda g, A, it, cb=None: pnp_admm_tv(g, A, N, it, callback=cb),
    "MLEM+FONF":    lambda g, A, it, cb=None: mlem_fonf(g, A, N, it, alpha=0.5, lam_dt=0.95, callback=cb),
}
