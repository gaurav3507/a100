"""Task 1.1 - verify the downloaded UL-DD files against the dataset's own
per-subject/session/modality availability manifest (Info.xlsx, "Publicity"
sheet: 'a' = available, 'r' = removed/not recorded).

Flags (does not silently skip):
  - expected-available files that are missing or empty on disk
  - expected-removed files that unexpectedly exist and are non-empty
  - the O2M.csv triple (SPO2/PR/Motion columns) disagreeing on availability

Usage: .venv/Scripts/python.exe scripts/verify_uldd.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from common.paths import INFO_XLSX, PUBLICITY_COLUMN_MAP, resolve


def load_manifest() -> pd.DataFrame:
    df = pd.read_excel(INFO_XLSX, sheet_name="Publicity")
    df["Participants"] = df["Participants"].ffill()
    return df


def main() -> int:
    manifest = load_manifest()
    problems = []
    checked = 0

    for _, row in manifest.iterrows():
        subject, session = row["Participants"], row["Session"]

        o2m_flags = set()
        for column, (kind, modalities) in PUBLICITY_COLUMN_MAP.items():
            flag = str(row[column]).strip().lower()
            if flag not in ("a", "r"):
                problems.append(
                    f"{subject}/{session}: unrecognized manifest flag "
                    f"{flag!r} for column {column!r}"
                )
                continue
            if column in ("SPO2", "PR", "Motion"):
                o2m_flags.add(flag)

            for modality in modalities:
                path = resolve(kind, subject, session, modality)
                checked += 1
                exists = path.exists() and path.stat().st_size > 0
                if flag == "a" and not exists:
                    problems.append(
                        f"{subject}/{session}: MISSING expected file "
                        f"({column} -> {modality}) at {path}"
                    )
                elif flag == "r" and exists:
                    problems.append(
                        f"{subject}/{session}: UNEXPECTED file present "
                        f"({column} -> {modality} marked removed) at {path}"
                    )

        if len(o2m_flags) > 1:
            problems.append(
                f"{subject}/{session}: SPO2/PR/Motion flags disagree "
                f"({o2m_flags}) despite sharing one O2M.csv file"
            )

    print(f"Checked {checked} (subject, session, modality) entries "
          f"across {len(manifest)} subject-sessions.")
    if problems:
        print(f"\n{len(problems)} problem(s) found:\n")
        for p in problems:
            print(" -", p)
        return 1

    print("All files match the Info.xlsx availability manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
