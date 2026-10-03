#!/usr/bin/env bash
# CAFE Phase 0 - reproducible environment build for the C-MET baseline.
#
# Verified against the official C-MET repo (ChanHyeok-Choi/C-MET, branch main)
# and the EDTalk repo it depends on. Torch is NOT pinned in C-MET's
# requirements.txt, so this script installs it deliberately. The single pin
# choice below (torch 2.1.2 + torchvision 0.16.2) is deliberate: it supports
# both Ampere (A4000) and Hopper (H100), and torchvision 0.16 still ships
# transforms.functional_tensor, which basicsr imports (removed in tv 0.17).
#
# Usage:
#   bash setup_cmet_env.sh a4000     # Ampere: CUDA 11.8 wheel
#   bash setup_cmet_env.sh h100      # Hopper: CUDA 12.1 wheel
#
# This script is intended to run on the target machine (GPU node). It does
# not download datasets or checkpoints; see PHASE0_SETUP.md for those.

set -euo pipefail

TARGET="${1:-}"
ENV_NAME="${ENV_NAME:-C_MET}"
REPO_URL="https://github.com/ChanHyeok-Choi/C-MET"

case "$TARGET" in
  a4000) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
  h100)  TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
  *) echo "usage: bash setup_cmet_env.sh [a4000|h100]"; exit 1 ;;
esac

echo "[phase0] target=$TARGET  env=$ENV_NAME  torch_index=$TORCH_INDEX"

# 1. Clone the baseline at a PINNED commit. Record the commit before anything.
if [ ! -d C-MET ]; then
  git clone "$REPO_URL"
fi
cd C-MET
PIN_COMMIT="$(git rev-parse HEAD)"
echo "[phase0] C-MET pinned at $PIN_COMMIT"
git rev-parse HEAD > ../CMET_PINNED_COMMIT.txt

# 2. Conda env. C-MET's README uses python 3.9 (EDTalk's own repo uses 3.8;
#    C-MET re-uses only EDTalk's encoder/decoder under this newer env).
if ! conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  conda create -y -n "$ENV_NAME" python=3.9
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 3. System-level media deps (verified from README): ffmpeg 4.4 + pkg-config.
conda install -y -c conda-forge "ffmpeg=4.4" pkg-config

# 4. Torch FIRST, so pip does not pull an arbitrary CPU/older build via
#    audiocraft/funasr transitive deps.
pip install "torch==2.1.2" "torchvision==0.16.2" "torchaudio==2.1.2" \
    --index-url "$TORCH_INDEX"

# 5. C-MET's own requirements (torch already satisfied above).
pip install -r requirements.txt

# 6. Known-failure guard: basicsr imports torchvision.transforms.functional_tensor.
#    With tv 0.16 it exists, so this should pass. If a later tv sneaks in,
#    this line patches the import in place rather than failing silently.
python - <<'PY'
import importlib, sys
try:
    import torchvision.transforms.functional_tensor  # noqa
    print("[phase0] basicsr/torchvision import path OK")
except Exception as e:
    print("[phase0] patching basicsr degradations import:", e)
    import basicsr, os, re
    deg = os.path.join(os.path.dirname(basicsr.__file__), "data", "degradations.py")
    src = open(deg).read()
    src = src.replace(
        "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
        "from torchvision.transforms.functional import rgb_to_grayscale")
    open(deg, "w").write(src)
    print("[phase0] patched", deg)
PY

# 7. Import smoke test: fail loudly now, not mid-training.
python - <<'PY'
import importlib
mods = ["torch", "torchvision", "transformers", "funasr",
        "basicsr", "gfpgan", "timm", "face_alignment", "moviepy",
        "librosa", "numba", "numpy", "cv2"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append((m, repr(e)))
import torch
print("[phase0] torch", torch.__version__, "cuda_build", torch.version.cuda,
      "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[phase0] device", torch.cuda.get_device_name(0),
          torch.cuda.get_device_capability(0))
if bad:
    print("[phase0] IMPORT FAILURES:")
    for m, e in bad:
        print("   ", m, e)
    raise SystemExit(1)
print("[phase0] all key imports OK")
PY

echo "[phase0] environment build complete for target=$TARGET"
echo "[phase0] NEXT: place checkpoints, then run freeze_provenance.py to close Gate 0"
