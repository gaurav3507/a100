from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import yaml

from .graph_truth import inter_group_adjacency, intra_group_adjacency

ROOT = Path(__file__).resolve().parents[2]
GCARL_ROOT = ROOT / "vendor" / "GCaRL"


def load_sim1_config(path: Path | None = None) -> dict:
    with open(path or ROOT / "config" / "sim1.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def generate_sim1(seed: int, config: dict | None = None) -> dict:
    """Call pinned G-CaRL generate_dataset without training any model."""
    cfg = dict(config or load_sim1_config())
    sys.dont_write_bytecode = True
    if str(GCARL_ROOT) not in sys.path:
        sys.path.insert(0, str(GCARL_ROOT))
    from subfunc.generate_dataset import generate_dataset

    kwargs = {k: v for k, v in cfg.items() if k != "scm_seeds"}
    kwargs["random_seed"] = seed
    with contextlib.redirect_stdout(io.StringIO()):
        x, s, lam1, lam2, lamin1, lamin2 = generate_dataset(**kwargs)
    expected = (cfg["num_data"], cfg["num_group"], cfg["num_dim"])
    if s.shape != expected:
        raise RuntimeError(f"Pinned G-CaRL returned latent shape {s.shape}, expected {expected}")
    return {
        "x": x,
        "z": s,
        "lam1": lam1,
        "lam2": lam2,
        "lamin1": lamin1,
        "lamin2": lamin2,
        "inter_truth": inter_group_adjacency(lam1, cfg["num_group"]),
        "intra_truth": intra_group_adjacency(lamin1),
        "config": cfg,
    }
