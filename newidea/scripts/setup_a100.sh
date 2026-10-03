#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_DIR/.uv-cache}"
command -v uv >/dev/null
command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
git submodule update --init --recursive
test "$(git -C vendor/GCaRL rev-parse HEAD)" = "$(tr -d '[:space:]' < vendor/GCaRL_COMMIT.txt)"
test "$(git -C vendor/notears rev-parse HEAD)" = "$(tr -d '[:space:]' < vendor/NOTEARS_COMMIT.txt)"
uv sync --frozen --python 3.11
export PYTHONPATH="$PROJECT_DIR/src"
.venv/bin/python scripts/capability_check.py
.venv/bin/python -m pytest -q
.venv/bin/python - <<'PY'
import torch

assert torch.cuda.is_available(), "PyTorch cannot see CUDA"
device = torch.device("cuda:0")
a = torch.randn((1024, 1024), device=device)
b = a @ a.T
torch.cuda.synchronize()
assert torch.isfinite(b).all()
properties = torch.cuda.get_device_properties(device)
print(f"torch={torch.__version__}")
print(f"torch_cuda_runtime={torch.version.cuda}")
print(f"gpu={properties.name}")
print(f"compute_capability={properties.major}.{properties.minor}")
print(f"gpu_memory_bytes={properties.total_memory}")
PY
