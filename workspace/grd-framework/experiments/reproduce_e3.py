"""Print or execute the complete E3 and optional iLCS regeneration sequence.

Examples:

    python experiments/reproduce_e3.py
    python experiments/reproduce_e3.py --execute
    python experiments/reproduce_e3.py --execute --include-ilcs

The default only prints commands. ``--execute`` first verifies all resolved input
paths, writes their SHA-256 manifest, and then runs each result-producing script
in dependency order.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from data_paths import real_data_inputs


REPO = Path(__file__).resolve().parents[1]


def commands(include_ilcs):
    py = sys.executable
    inputs = real_data_inputs()
    k562 = str(inputs["K562"])
    rpe1 = str(inputs["RPE1"])
    norman = str(inputs["Norman"])
    steps = [
        [py, "experiments/write_data_manifest.py"],
        [py, "experiments/e3_perturbseq_panel.py", "K562_CRISPRi",
         k562, "EMPTY", "0"],
        [py, "experiments/e3_perturbseq_panel.py", "causalbench_rpe1",
         rpe1, "EMPTY", "0", "rpe1_gate_fixed"],
        [py, "experiments/e3_perturbseq_panel.py",
         "Norman_CRISPRa_singlegene", norman, "EMPTY", "1"],
        [py, "experiments/e3_fmri_connectivity.py", "hcp"],
        [py, "experiments/e3_fmri_connectivity.py", "abide"],
        [py, "experiments/e3_rpe1_confound_check.py"],
        [py, "experiments/e3_stability_perturbseq.py", "k562"],
        [py, "experiments/e3_stability_perturbseq.py", "rpe1"],
        [py, "experiments/e3_stability_perturbseq.py", "norman"],
        [py, "experiments/e3_stability_hcp.py"],
        [py, "experiments/e3_poscontrol_faithful.py"],
        [py, "experiments/e3_attribution_panel.py"],
        [py, "experiments/export_gate_decisions.py"],
    ]
    if include_ilcs:
        steps.extend([
            [py, "experiments/ilcs_baseline.py", "--dataset", "all",
             "--seed", str(seed)]
            for seed in range(3)
        ])
        steps.append([py, "experiments/ilcs_baseline.py", "--aggregate"])
    return steps


def validate_inputs():
    missing = []
    for name, path in real_data_inputs().items():
        if not path.exists():
            missing.append(f"{name}: {path}")
    if missing:
        raise SystemExit("missing external inputs:\n  " + "\n  ".join(missing))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--include-ilcs", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    steps = commands(args.include_ilcs)
    for index, command in enumerate(steps, 1):
        print(f"[{index:02d}/{len(steps):02d}] " + " ".join(command), flush=True)
    if not args.execute:
        print("print-only mode; pass --execute to run", flush=True)
        return
    validate_inputs()
    for index, command in enumerate(steps, 1):
        print(f"RUN [{index:02d}/{len(steps):02d}]", flush=True)
        subprocess.run(command, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
