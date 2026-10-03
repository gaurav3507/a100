import numpy as np

from crlcd_audit.graph_metrics import directed_metrics, orientation_metrics, skeleton_metrics
from crlcd_audit.method_adapters.discovery import castle_graph_adapter, lingam_graph_adapter, pc_graph_adapter
from crlcd_audit.method_adapters.notears_cov import notears_linear_covariance


def test_lingam_direction_adapter_is_not_transposed():
    raw_effect_by_cause = np.array([[0, 0, 0], [2, 0, 0], [0, -1, 0]])
    expected = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    assert np.array_equal(lingam_graph_adapter(raw_effect_by_cause, 0.1), expected)


def test_castle_direction_adapter_for_all_three_included_castle_methods():
    raw = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    for method in ["NOTEARS-linear", "NOTEARS-MLP", "GOLEM"]:
        assert np.array_equal(castle_graph_adapter(raw), raw), method


def test_cached_covariance_notears_known_direction():
    rng = np.random.default_rng(7)
    x0 = rng.normal(size=1000)
    x1 = 2.0 * x0 + rng.normal(scale=0.2, size=1000)
    estimate = notears_linear_covariance(np.c_[x0, x1], lambda1=0.01, max_iter=50, w_threshold=0.1)
    assert estimate[0, 1] != 0
    assert estimate[1, 0] == 0


def test_pc_skeleton_and_orientation_extraction():
    # causal-learn: tail at row endpoint, arrow at column endpoint.
    raw = np.array([[0, -1, 0], [1, 0, -1], [0, -1, 0]])
    directed, skeleton, _ = pc_graph_adapter(raw)
    assert directed[0, 1] == 1
    assert directed[1, 0] == 0
    assert skeleton[1, 2] and skeleton[2, 1]
    assert directed[1, 2] == directed[2, 1] == 0


def test_known_answer_scoring_and_node_indexing():
    truth = np.zeros((4, 4))
    truth[0, 3] = 1
    mask = np.ones((4, 4), dtype=bool)
    np.fill_diagonal(mask, False)
    assert directed_metrics(truth, truth, mask)["f1"] == 1
    skeleton = (truth != 0) | (truth.T != 0)
    assert skeleton_metrics(truth, skeleton, mask)["f1"] == 1
    assert orientation_metrics(truth, truth, skeleton, mask)["f1"] == 1
