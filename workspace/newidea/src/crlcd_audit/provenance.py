from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def canonical_array_fingerprint(array: np.ndarray, config: dict | None = None) -> dict:
    canonical = np.ascontiguousarray(array)
    config_json = json.dumps(config or {}, sort_keys=True, separators=(",", ":"))
    return {
        "shape": list(canonical.shape),
        "dtype": canonical.dtype.str,
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
        "scm_config_hash": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
    }


def git_commit(path: Path = ROOT) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def gcarl_commit() -> str:
    return (ROOT / "vendor" / "GCaRL_COMMIT.txt").read_text(encoding="utf-8").strip()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)

