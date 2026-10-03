from __future__ import annotations

import importlib.metadata
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parents[3]


def load_method_config() -> dict:
    return yaml.safe_load((ROOT / "config" / "methods.yaml").read_text(encoding="utf-8"))


def included_method_names() -> list[str]:
    return list(load_method_config()["methods"])


def method_type(name: str) -> str:
    return load_method_config()["methods"][name]["type"]


def versions() -> dict[str, str]:
    result = {name: importlib.metadata.version(name) for name in ["causal-learn", "lingam", "gcastle", "torch"]}
    result["notears_commit"] = (ROOT / "vendor" / "NOTEARS_COMMIT.txt").read_text(encoding="utf-8").strip()
    return result


def standardize(x: np.ndarray) -> np.ndarray:
    std = np.std(x, axis=0, ddof=0)
    std[std == 0] = 1.0
    return (x - np.mean(x, axis=0)) / std


def nonparanormal(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    ranks = np.column_stack([rankdata(x[:, j], method="average") for j in range(x.shape[1])])
    return norm.ppf(np.clip((ranks - 0.5) / n, 1e-8, 1 - 1e-8))


def pc_graph_adapter(graph: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Convert causal-learn endpoints to parent-row directed and skeleton matrices."""
    g = np.asarray(graph)
    d = g.shape[0]
    directed = np.zeros((d, d), dtype=int)
    skeleton = np.zeros((d, d), dtype=bool)
    endpoint_counts: dict[str, int] = {}
    for i in range(d):
        for j in range(i + 1, d):
            a, b = int(g[i, j]), int(g[j, i])
            if a != 0 or b != 0:
                skeleton[i, j] = skeleton[j, i] = True
            endpoint_counts[f"{a},{b}"] = endpoint_counts.get(f"{a},{b}", 0) + 1
            if a == -1 and b == 1:
                directed[i, j] = 1
            elif a == 1 and b == -1:
                directed[j, i] = 1
    return directed, skeleton, endpoint_counts


def lingam_graph_adapter(adjacency: np.ndarray, threshold: float) -> np.ndarray:
    """lingam stores effect by cause, so transpose to parent by child."""
    return (np.abs(np.asarray(adjacency).T) >= threshold).astype(int)


def castle_graph_adapter(causal_matrix: np.ndarray) -> np.ndarray:
    """gCastle causal_matrix is already parent-row, child-column."""
    return (np.asarray(causal_matrix) != 0).astype(int)


def fit_method(name: str, z: np.ndarray, method_seed: int) -> dict:
    cfg_all = load_method_config()
    cfg = deepcopy(cfg_all["methods"][name])
    x = z.reshape(z.shape[0], -1)
    if cfg_all["common"]["standardize_per_dataset"]:
        x = standardize(x)
    if name == "PC-nonparanormal-preproc":
        x = nonparanormal(x)
    if cfg["type"] == "pc":
        from causallearn.search.ConstraintBased.PC import pc
        cg = pc(x, alpha=cfg["alpha"], indep_test="fisherz", stable=cfg["stable"], uc_rule=cfg["uc_rule"], uc_priority=cfg["uc_priority"], show_progress=False)
        directed, skeleton, endpoints = pc_graph_adapter(cg.G.graph)
        return {"directed": directed, "skeleton": skeleton, "raw_endpoints": endpoints}
    if name == "DirectLiNGAM":
        import lingam
        model = lingam.DirectLiNGAM(random_state=method_seed)
        model.fit(x)
        directed = lingam_graph_adapter(model.adjacency_matrix_, cfg["weight_threshold"])
        return {"directed": directed, "skeleton": None, "raw_endpoints": None}
    if name == "NOTEARS-linear":
        from .notears_cov import notears_linear_covariance
        weight = notears_linear_covariance(x, lambda1=float(cfg["lambda1"]), max_iter=int(cfg["max_iter"]), h_tol=float(cfg["h_tol"]), rho_max=float(cfg["rho_max"]), w_threshold=float(cfg["weight_threshold"]))
        return {"directed": castle_graph_adapter(weight), "skeleton": None, "raw_endpoints": None}
    os.environ.setdefault("CASTLE_BACKEND", "pytorch")
    from castle.algorithms import GOLEM, NotearsNonlinear
    if name == "NOTEARS-MLP":
        import torch
        torch.manual_seed(method_seed)
        device = "gpu" if cfg["device_type"] == "auto" and torch.cuda.is_available() else "cpu"
        model = NotearsNonlinear(lambda1=float(cfg["lambda1"]), lambda2=float(cfg["lambda2"]), max_iter=int(cfg["max_iter"]), h_tol=float(cfg["h_tol"]), rho_max=float(cfg["rho_max"]), w_threshold=float(cfg["weight_threshold"]), hidden_layers=tuple(cfg["hidden_layers"]), device_type=device)
    elif name == "GOLEM":
        import torch
        device = "gpu" if cfg["device_type"] == "auto" and torch.cuda.is_available() else "cpu"
        model = GOLEM(lambda_1=cfg["lambda_1"], lambda_2=cfg["lambda_2"], equal_variances=cfg["equal_variances"], learning_rate=cfg["learning_rate"], num_iter=cfg["num_iter"], checkpoint_iter=cfg["checkpoint_iter"], seed=method_seed, graph_thres=cfg["graph_threshold"], device_type=device)
    else:
        raise ValueError(name)
    model.learn(x)
    return {"directed": castle_graph_adapter(model.causal_matrix), "skeleton": None, "raw_endpoints": None}
