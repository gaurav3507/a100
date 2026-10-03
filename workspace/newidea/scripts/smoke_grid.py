#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from crlcd_audit.graph_metrics import directed_metrics
from crlcd_audit.graph_truth import inter_group_mask
from crlcd_audit.method_adapters.discovery import castle_graph_adapter, lingam_graph_adapter, pc_graph_adapter
from crlcd_audit.provenance import canonical_array_fingerprint
from crlcd_audit.representation_metrics import rank_aware_mcc, standard_mcc
from crlcd_audit.transforms import additive_measurement_noise, component_scaling, cubic_warp, within_group_entanglement, within_group_permutation

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rng = np.random.default_rng(19)
    z = rng.normal(size=(200, 3, 4))
    transforms = [
        within_group_permutation(z, 1),
        component_scaling(z, 2.0, 2),
        cubic_warp(z, 0.1),
        within_group_entanglement(z, 0.1, 3),
        additive_measurement_noise(z, 0.1, 4),
    ]
    rows = [{"shape": list(item.values.shape), "standard_mcc": standard_mcc(z, item.values), "rank_aware_mcc": rank_aware_mcc(z, item.values), "fingerprint": canonical_array_fingerprint(item.values), "metadata": item.metadata} for item in transforms]
    path = ROOT / ".capability" / "tiny_grid.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} transformed smoke artifacts to {path}")


if __name__ == "__main__":
    main()

