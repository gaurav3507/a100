import numpy as np

from crlcd_audit.gcarl_adapter import generate_sim1, load_sim1_config
from crlcd_audit.graph_truth import inter_group_adjacency, intra_group_adjacency


def test_sim1_true_z_shape_and_grouping():
    data = generate_sim1(0)
    assert data["z"].shape == (65536, 3, 10)
    assert np.isfinite(data["z"]).all()


def test_inter_group_truth_extraction():
    lam1 = np.zeros((2, 2, 2, 1))
    lam1[0, 1, 0, 0] = 0.7
    out = inter_group_adjacency(lam1, 2)
    assert out[0, 3] == 0.7
    assert np.count_nonzero(out) == 1


def test_intra_group_truth_extraction():
    lamin1 = np.zeros((2, 2, 2))
    lamin1[0, 1, 1] = 0.8
    out = intra_group_adjacency(lamin1)
    assert out[2, 3] == 0.8
    assert np.count_nonzero(out) == 1

