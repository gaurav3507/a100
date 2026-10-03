import numpy as np

from crlcd_audit.graph_truth import remap_truth_for_permutation
from crlcd_audit.transforms import additive_measurement_noise, component_scaling, cubic_warp, within_group_entanglement, within_group_permutation


def test_within_group_permutation_never_crosses_groups():
    z = np.broadcast_to(np.arange(3)[None, :, None], (4, 3, 5)).copy()
    result = within_group_permutation(z, 8)
    for group in range(3):
        assert np.all(result.values[:, group, :] == group)
        assert sorted(result.permutations[group]) == list(range(5))


def test_truth_remapping_under_permutation():
    truth = np.zeros((4, 4))
    truth[0, 3] = 1
    permutations = [[1, 0], [1, 0]]
    remapped = remap_truth_for_permutation(truth, permutations, 2)
    assert remapped[1, 2] == 1
    assert np.sum(remapped) == 1


def test_cubic_is_strictly_monotone_for_all_levels():
    points = np.linspace(-20, 20, 10001)
    for lam in [0, 0.05, 0.1, 0.25, 0.5, 1, 2]:
        values = cubic_warp(points[:, None, None], lam).values[:, 0, 0]
        assert np.all(np.diff(values) > 0)


def test_entanglement_pairs_stay_within_groups():
    z = np.ones((20, 3, 5))
    metadata = within_group_entanglement(z, 0.5, 9).metadata
    assert {pair["group"] for pair in metadata["pairs"]} == {0, 1, 2}
    assert all(pair["target"] != pair["source"] for pair in metadata["pairs"])


def test_noise_uses_variable_relative_scale():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(10000, 1, 2)) * np.array([1.0, 10.0])
    result = additive_measurement_noise(z, 0.5, 2)
    added_sd = np.std(result.values - z, axis=0)[0]
    ratio = added_sd[1] / added_sd[0]
    assert 9.0 < ratio < 11.0


def test_scaling_rule_varies_and_is_deterministic():
    z = np.ones((3, 2, 5))
    a = component_scaling(z, 4.0, 10)
    b = component_scaling(z, 4.0, 10)
    assert np.array_equal(a.values, b.values)
    assert set(np.unique(a.values)) == {0.25, 4.0}


def test_lambda_zero_cubic_matches_oracle_exactly():
    z = np.random.default_rng(1).normal(size=(20, 3, 4))
    assert np.array_equal(cubic_warp(z, 0).values, z)

