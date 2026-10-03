"""Export compact per-perturbation corrected-BH decisions from committed E3 runs.

The full E3 JSON files retain every diagnostic. These compact, deterministic
exports provide a stable comparison contract for baselines such as iLCS without
requiring those baselines to know the full panel-result schema.
"""

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "results" / "e3_decisions"
SOURCES = {
    "K562": REPO / "results" / "e3" / "e3_K562_CRISPRi.json",
    "RPE1": REPO / "results" / "e3" / "e3_rpe1_gate_fixed.json",
    "Norman": (
        REPO / "results" / "e3" / "e3_Norman_CRISPRa_singlegene.json"),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_dataset(name, source):
    report = json.loads(source.read_text())
    decisions = []
    for row in report["per_perturbation"]:
        corrected = row["corrected_disjoint"]
        decisions.append(dict(
            environment=str(row["environment"]),
            n_samples=int(row["n_samples"]),
            signal=float(row["signal"]),
            pvalue=float(corrected["pvalue"]),
            raw_detect=bool(corrected["raw_detect"]),
            bh_detect=bool(corrected["bh_detect"]),
            disjoint_applied=bool(corrected["disjoint_applied"]),
        ))
    detected = [row["environment"] for row in decisions if row["bh_detect"]]
    output = dict(
        dataset=name,
        source_result=str(source.relative_to(REPO)),
        source_sha256=sha256(source),
        primary_decision=report["primary_decision"],
        alpha=float(report["alpha"]),
        q=float(report["q"]),
        B=int(report["B"]),
        n_perturbations=len(decisions),
        n_bh_detected=len(detected),
        detected_ids=detected,
        decisions=decisions,
    )
    path = OUTPUT / f"{name}.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    return path, output


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        path, report = export_dataset(name, source)
        print(
            f"{name}: {report['n_bh_detected']}/"
            f"{report['n_perturbations']} -> {path}"
        )


if __name__ == "__main__":
    main()
