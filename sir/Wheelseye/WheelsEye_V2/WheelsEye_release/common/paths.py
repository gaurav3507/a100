"""Path resolution for the on-disk UL-DD and MePhy dataset layouts.

UL-DD's CSV_Files/ and Extracted_Features/ archives were extracted with an
extra pair of self-named wrapper folders (an artifact of how the zip was
built), so the real per-subject data sits three levels down. Video_Data/
does not have that wrapper. All other modules should go through this file
rather than hardcoding these paths.
"""
from pathlib import Path

# v2: dataset moved to this Linux machine (was E:\capstone\ul-dd on the old
# Windows workstation). Verified 2026-08-18: same triple-nested wrapper
# layout, 35/35 files per modality (34/35 Telemetry, matching the manifest).
ULDD_ROOT = Path("/workspace/sir/Wheelseye/datasets/ul-dd")
# MePhy has NOT been re-verified on this machine -- this is still the old
# Windows path. mephy_repro/ is frozen as a completed, citable study
# (88.79% vs the paper's 90.2%), so nothing in the v2 pipeline needs it.
MEPHY_ROOT = Path(r"E:\capstone\mephy\MePhy Dataset\MePhy Dataset")

CSV_ROOT = ULDD_ROOT / "CSV_Files" / "CSV_Files" / "CSV_Files"
FEATURES_ROOT = ULDD_ROOT / "Extracted_Features" / "Extracted_Features" / "Extracted_Features"
VIDEO_ROOT = ULDD_ROOT / "Video_Data"
INFO_XLSX = ULDD_ROOT / "Info.xlsx"
LABELS_CSV = ULDD_ROOT / "Labels.csv"

SUBJECTS = tuple("ABCDEFGHIJKLMNOPQRS")  # 19 subjects
SESSIONS = ("A", "D")  # Awake, Drowsy
SESSION_NAMES = {"A": "Alert", "D": "Drowsy"}
NO_DROWSY_SUBJECTS = frozenset({"C", "F", "L"})  # no Drowsy session recorded at all

CSV_MODALITIES = (
    "ACC", "BVP", "EDA", "HR", "IBI", "LGP", "O2M", "RGP", "TEMP",
    "Labels", "Telemetry",
)
FEATURE_MODALITIES = ("FL", "PL", "FAU")
VIDEO_MODALITIES = ("IR", "L3D", "R3D", "Pose")

# UL-DD Info.xlsx "Publicity" sheet column -> (kind, modality code(s))
# kind selects which resolver (csv_path / feature_path / video_path) applies.
PUBLICITY_COLUMN_MAP = {
    "IR Video": ("video", ("IR",)),
    "3D depth Video": ("video", ("L3D", "R3D")),
    "Pose Video": ("video", ("Pose",)),
    "ACC": ("csv", ("ACC",)),
    "BVP": ("csv", ("BVP",)),
    "EDA": ("csv", ("EDA",)),
    "HR": ("csv", ("HR",)),
    "IBI": ("csv", ("IBI",)),
    "Left GP": ("csv", ("LGP",)),
    "Right GP": ("csv", ("RGP",)),
    "SPO2": ("csv", ("O2M",)),
    "PR": ("csv", ("O2M",)),
    "Motion": ("csv", ("O2M",)),
    "TEMP": ("csv", ("TEMP",)),
    "Telemetry": ("csv", ("Telemetry",)),
    "PL": ("feature", ("PL",)),
    "FL": ("feature", ("FL",)),
    "FAU": ("feature", ("FAU",)),
}


def csv_path(subject: str, session: str, modality: str) -> Path:
    return CSV_ROOT / subject / session / f"{subject}_{modality}_{session}.csv"


def feature_path(subject: str, session: str, modality: str) -> Path:
    return FEATURES_ROOT / subject / session / f"{subject}_{modality}_{session}.csv"


def video_path(subject: str, session: str, modality: str) -> Path:
    return VIDEO_ROOT / subject / session / f"{subject}_{modality}_{session}.mp4"


_RESOLVERS = {"csv": csv_path, "feature": feature_path, "video": video_path}


def resolve(kind: str, subject: str, session: str, modality: str) -> Path:
    return _RESOLVERS[kind](subject, session, modality)
