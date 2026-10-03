from copy import deepcopy

import numpy as np

from crlcd_audit import manifest as manifest_module
from crlcd_audit.manifest import build_manifest, validate_results
from crlcd_audit.method_adapters.discovery import load_method_config
from crlcd_audit.provenance import canonical_array_fingerprint, write_json_atomic


def test_manifest_cell_uniqueness(tmp_path):
    rows = build_manifest(["PC-Pearson", "DirectLiNGAM"], tmp_path / "manifest.csv", seeds=[0, 1])
    ids = [row["cell_id"] for row in rows]
    assert len(rows) == 2 * 22 * 2
    assert len(ids) == len(set(ids))
    method_seeds = {(row["scm_seed"], row["method"]): row["method_seed"] for row in rows}
    assert method_seeds[(0, "DirectLiNGAM")] == 200002
    assert method_seeds[(1, "DirectLiNGAM")] == 201002


def test_result_count_assertion(tmp_path, monkeypatch):
    monkeypatch.setattr(manifest_module, "ROOT", tmp_path)
    cells = tmp_path / "results" / "cells"
    cells.mkdir(parents=True)
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("cell_id,expected_output_path\na,results/cells/a.json\n", encoding="utf-8")
    write_json_atomic(cells / "a.json", {"cell_id": "a", "method": "x", "standard_mcc": 1, "rank_aware_mcc": 1, "inter_group_f1": 1, "normalized_degradation": 0})
    assert validate_results(manifest_path)["completed"] == 1


def test_dataset_fingerprint_is_deterministic():
    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    assert canonical_array_fingerprint(a, {"x": 1}) == canonical_array_fingerprint(a.copy(), {"x": 1})


def test_frozen_method_configuration_does_not_mutate():
    before = load_method_config()
    local = deepcopy(before["methods"]["NOTEARS-MLP"])
    local["hidden_layers"][0] = 999
    after = load_method_config()
    assert after == before


def test_scientific_notears_numeric_configuration_is_coercible():
    config = load_method_config()["methods"]
    for method in ["NOTEARS-linear", "NOTEARS-MLP"]:
        assert float(config[method]["h_tol"]) > 0
        assert float(config[method]["rho_max"]) > 0
