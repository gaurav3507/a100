import numpy as np

from crlcd_audit.representation_metrics import rank_aware_mcc, standard_mcc


def test_pearson_mcc_identity():
    z = np.random.default_rng(3).normal(size=(1000, 3, 5))
    assert np.isclose(standard_mcc(z, z), 1.0)


def test_rank_mcc_monotone_transform():
    z = np.random.default_rng(4).normal(size=(1000, 3, 5))
    transformed = z + 2 * z**3
    assert np.isclose(rank_aware_mcc(z, transformed), 1.0)

