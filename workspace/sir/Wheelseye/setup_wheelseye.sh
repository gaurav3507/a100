#!/usr/bin/env bash
# =====================================================================
# WheelsEye setup for a Linux GPU server (DGX + JupyterLab)
#
# Read this before running it. It is written in stages -- you can run
# the whole thing, or copy one stage at a time into the terminal.
#
# Usage:
#   bash setup_wheelseye.sh
#
# Assumes you are running it from INSIDE the folder that should contain
# the repo (e.g. the "Wheelseye" folder visible in your file browser).
# =====================================================================
set -euo pipefail

# --- Adjust these two if needed -------------------------------------
# Match the CUDA build to the node's driver. Your existing cldrive
# kernel is cu121, so the driver supports at least 12.1. cu124 is a
# safe modern choice; drop to cu121 if torch complains at runtime.
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

# Where the datasets will live. Keep them OUTSIDE the git repo.
DATA_DIR="$(pwd)/datasets"
# ---------------------------------------------------------------------

REPO_DIR="$(pwd)/WheelsEye"

echo "############ STAGE 0: environment check ############"
echo "Working directory: $(pwd)"
echo
echo "--- Python ---"
python3 -V
echo
echo "--- GPUs (check whether anyone else is using them) ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
           --format=csv || echo "nvidia-smi unavailable"
echo
echo "--- Disk space (datasets are large; make sure there is room) ---"
df -h .
echo
read -r -p "Continue? [y/N] " ok
[[ "$ok" == "y" || "$ok" == "Y" ]] || exit 0


echo "############ STAGE 1: get the code ############"
if [[ -d "$REPO_DIR/.git" ]]; then
    echo "Repo already present, pulling latest."
    git -C "$REPO_DIR" pull --ff-only
else
    git clone https://github.com/Parthgogia/WheelsEye.git "$REPO_DIR"
fi
cd "$REPO_DIR"


echo "############ STAGE 2: isolated environment ############"
# A dedicated venv. This does NOT touch the cldrive conda env.
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

echo "--- Installing torch from $TORCH_INDEX ---"
pip install torch --index-url "$TORCH_INDEX"

echo "--- Verifying torch sees the GPUs ---"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
PY

echo "--- Installing the rest of requirements.txt ---"
# If a pin fails to resolve on this Python, relax it here rather than
# editing the tracked file:
#   grep -v '^torch-geometric' requirements.txt > /tmp/req.txt
pip install -r requirements.txt


echo "############ STAGE 3: make dataset paths configurable ############"
# The repo hardcodes Windows paths. Patch them to read env vars, so the
# change survives future `git pull`s instead of being clobbered.
python - <<'PY'
import re
from pathlib import Path

targets = {
    Path("common/paths.py"): ["ULDD_ROOT", "MEPHY_ROOT"],
    Path("mephy_repro/paths.py"): ["MEPHY_ROOT"],
}

for path, names in targets.items():
    text = path.read_text(encoding="utf-8")
    if "os.environ" in text:
        print(f"{path}: already patched, skipping")
        continue
    for name in names:
        # Rewrite:  NAME = Path(r"E:\...")
        # As:       NAME = Path(os.environ.get("NAME", r"E:\..."))
        text = re.sub(
            rf'^{name}\s*=\s*Path\((r"[^"]*")\)',
            rf'{name} = Path(os.environ.get("{name}", \1))',
            text,
            flags=re.MULTILINE,
        )
    if "\nimport os" not in text:
        text = text.replace("from pathlib import Path",
                            "import os\nfrom pathlib import Path", 1)
    path.write_text(text, encoding="utf-8")
    print(f"{path}: patched -> {', '.join(names)} now read from the environment")
PY

mkdir -p "$DATA_DIR"
cat > env.sh <<EOF
# source this before running anything in the pipeline
export ULDD_ROOT="$DATA_DIR/ul-dd"
export MEPHY_ROOT="$DATA_DIR/mephy/MePhy Dataset/MePhy Dataset"
EOF
echo "Wrote env.sh pointing at $DATA_DIR"


echo "############ STAGE 4: Jupyter kernel ############"
pip install ipykernel
python -m ipykernel install --user \
    --name wheelseye \
    --display-name "WheelsEye (torch)"
echo "Refresh the JupyterLab browser tab to see the new kernel."


echo "############ STAGE 5: verify with the test suite ############"
# These need no dataset at all -- they run on hand-computed synthetic
# inputs. If all 13 pass, the environment is sound.
python -m pytest tests/ -v


cat <<'EOF'

#####################################################################
Setup complete. The environment works; the datasets are still missing.

NEXT:

  1. Copy UL-DD (and MePhy, if you want the sanity check) into the
     datasets/ folder created above. From your Windows machine:

       scp -r E:\capstone\ul-dd root@172.16.224.132:<path>/datasets/

     Do NOT use the JupyterLab upload button for multi-GB archives.

  2. Every time you open a new terminal:

       cd <repo>
       source .venv/bin/activate
       source env.sh

  3. Then run the pipeline in order:

       python scripts/verify_uldd.py
       python scripts/build_feature_table.py
       python track_b_light/train.py --model mlp

  4. For anything longer than a few minutes, use tmux so the job
     survives your browser disconnecting:

       tmux new -s wheels
       # ... start training ...
       # Ctrl-b then d to detach; `tmux attach -t wheels` to return

  5. On a shared node, pin yourself to one GPU so you do not collide
     with other users:

       CUDA_VISIBLE_DEVICES=0 python track_a_heavy/train.py --epochs 15
#####################################################################
EOF
