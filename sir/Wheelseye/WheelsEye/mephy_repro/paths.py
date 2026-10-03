"""Path resolution and coverage tables for the MePhy dataset (tasks.md 1.2),
grounded against the dataset's own ReadMe.pdf (E:\\capstone\\mephy\\MePhy
Dataset\\ReadMe.pdf):

  - All 60 users (user0..user59) have ECG.
  - Only user0..user29 (30 users) have EDA and EMG.
  - Only user0..user18 (19 users) have EyeBlinking.

FatigueNet's architecture treats ECG/EDA/EMG/blink as four GNN nodes, so a
faithful reproduction needs all four modalities per subject -> that's the
19-user (user0..user18) subset, matching how the original paper would have
had to constrain its training set too.
"""
from pathlib import Path

MEPHY_ROOT = Path("/workspace/sir/Wheelseye/datasets/mephy/MePhy Dataset/MePhy Dataset")

CONDITIONS = ("rest", "cognitive-fatigue", "physical-fatigue", "combo-fatigue")
CONDITION_LABEL = {c: i for i, c in enumerate(CONDITIONS)}  # 0=rest .. 3=combo
LABEL_NAMES = CONDITIONS

MODALITIES = ("ECG", "EDA", "EMG", "EyeBlinking")

USERS_ALL = tuple(f"user{i}" for i in range(60))
USERS_WITH_EDA_EMG = tuple(f"user{i}" for i in range(30))
USERS_WITH_BLINK = tuple(f"user{i}" for i in range(19))
FULL_COVERAGE_USERS = USERS_WITH_BLINK  # all 4 modalities present


def file_path(user: str, modality: str, condition: str) -> Path:
    return MEPHY_ROOT / user / f"{modality}_Collection" / f"{user}_{condition}.txt"
