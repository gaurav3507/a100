import json

import pytest

import scripts.write_prediction_table as prediction_module
from crlcd_audit.method_adapters import included_method_names


def _oracle_payload(seed_count=5):
    methods = {}
    for method in included_method_names():
        methods[method] = {
            "eligible": True,
            "individual_primary_f1": [1.0] * seed_count,
        }
    return {"scm_seeds": [0, 1, 2, 3, 4], "methods": methods}


def _write_oracle_files(root):
    oracle_dir = root / "results" / "oracle"
    oracle_dir.mkdir(parents=True)
    for method in included_method_names():
        token = method.lower().replace(" ", "-")
        for seed in range(5):
            (oracle_dir / f"seed_{seed}_{token}.json").write_text("{}\n", encoding="utf-8")


def test_prediction_table_requires_all_oracle_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_module, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "oracle_summary.json").write_text(json.dumps(_oracle_payload()), encoding="utf-8")
    with pytest.raises(SystemExit, match="Missing oracle artifact"):
        prediction_module.main()


def test_prediction_table_is_written_only_after_complete_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_module, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "oracle_summary.json").write_text(json.dumps(_oracle_payload()), encoding="utf-8")
    _write_oracle_files(tmp_path)
    prediction_module.main()
    text = (tmp_path / "prediction_table.md").read_text(encoding="utf-8")
    assert "PC-Pearson" in text
    assert "NOTEARS-MLP" in text


def test_prediction_table_refuses_existing_transformed_results(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_module, "ROOT", tmp_path)
    cells = tmp_path / "results" / "cells"
    cells.mkdir(parents=True)
    (cells / "already_seen.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="after transformed-grid results exist"):
        prediction_module.main()
