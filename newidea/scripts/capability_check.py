#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rng = np.random.default_rng(7)
    n = 300
    x0 = rng.laplace(size=n)
    x1 = 1.7 * x0 + rng.laplace(scale=0.3, size=n)
    x2 = -1.2 * x1 + rng.laplace(scale=0.3, size=n)
    x = np.c_[x0, x1, x2]
    rows = []

    def record(method: str, implementation: str, version: str, fn) -> None:
        row = {"method": method, "implementation": implementation, "version": version, "import": "PASS", "smoke_fit": "FAIL", "graph_returned": "FAIL", "included": "NO", "reason": None}
        try:
            graph = fn()
            row["smoke_fit"] = "PASS"
            if np.asarray(graph).shape == (3, 3):
                row["graph_returned"] = "PASS"
                row["included"] = "YES"
            else:
                row["reason"] = f"Unexpected graph shape {np.asarray(graph).shape}"
        except Exception as exc:
            row["reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    from causallearn.search.ConstraintBased.PC import pc
    cl_version = importlib.metadata.version("causal-learn")
    record("PC-Pearson", "causal-learn", cl_version, lambda: pc(x, alpha=0.01, indep_test="fisherz", show_progress=False).G.graph)
    ranks = np.column_stack([rankdata(x[:, j], method="average") for j in range(3)])
    npn = norm.ppf(np.clip((ranks - 0.5) / n, 1e-8, 1 - 1e-8))
    record("PC-nonparanormal-preproc", "causal-learn", cl_version, lambda: pc(npn, alpha=0.01, indep_test="fisherz", show_progress=False).G.graph)

    import lingam
    record("DirectLiNGAM", "lingam", importlib.metadata.version("lingam"), lambda: _lingam(lingam, x))

    os.environ.setdefault("CASTLE_BACKEND", "pytorch")
    from castle.algorithms import GOLEM, NotearsNonlinear
    gc_version = importlib.metadata.version("gcastle")
    from crlcd_audit.method_adapters.notears_cov import notears_linear_covariance
    notears_commit = (ROOT / "vendor" / "NOTEARS_COMMIT.txt").read_text(encoding="utf-8").strip()
    record("NOTEARS-linear", "xunzheng/notears plus exact cached covariance objective", notears_commit, lambda: notears_linear_covariance(x, lambda1=0.01, max_iter=20, w_threshold=0.1))
    record("NOTEARS-MLP", "gCastle", gc_version, lambda: _learn(NotearsNonlinear(max_iter=2, hidden_layers=(5, 1), w_threshold=0.1, device_type="cpu"), x))
    record("GOLEM", "gCastle", gc_version, lambda: _learn(GOLEM(num_iter=50, checkpoint_iter=25, graph_thres=0.1, device_type="cpu", seed=7), x))
    rows.append({"method": "CAM", "implementation": "R CAM", "version": None, "import": "FAIL", "smoke_fit": "FAIL", "graph_returned": "FAIL", "included": "NO", "reason": "R runtime unavailable"})
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "capability_methods.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


def _lingam(module, x):
    model = module.DirectLiNGAM(random_state=7)
    model.fit(x)
    return model.adjacency_matrix_


def _learn(model, x):
    model.learn(x)
    return model.causal_matrix


if __name__ == "__main__":
    main()
