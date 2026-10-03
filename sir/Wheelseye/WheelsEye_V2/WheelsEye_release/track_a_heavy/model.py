"""Track A (tasks.md Task 3.1): the shared GNN + Transformer + MGAF
architecture (common/fatiguenet_model.py) configured for our own three
feature "nodes" -- vision, bio, grip (common/dataset.py's
CORE_FEATURE_GROUPS) -- not the FatigueNet paper's four (ECG/EDA/EMG/blink)
and not UL-DD's own four available modalities either. This is a deliberate
departure from "reproduce the paper's node set on a new dataset": the node
set is chosen to match *our own hardware* --

  vision -> camera feed (Jetson + IR/RGB camera)
  bio    -> physiological sensor (ESP32-S3 PPG+GSR breakout board)
  grip   -> pressure sensor (ESP32-S3 left/right FSR grip sensors)

UL-DD also ships driving-simulator Telemetry (heading/speed/rpm/gear),
which the paper-reproduction instinct would fold in as a fourth node the
same way MePhy's `mephy_repro/` uses its four native signals -- but no
telemetry sensor exists anywhere in the deployed ESP32-S3/Jetson hardware
(tasks.md Phases 5-6), so a Track A that depends on it would be measuring
an accuracy ceiling Track B could never approach even in principle, which
defeats the point of the two-track comparison. `build_track_a_model(...,
include_telemetry=True)` still exists for an explicit, separately-reported
"what if we also had simulator telemetry" ceiling experiment (same spirit
as tasks.md 3.6's deep-vision node: Track-A-only, never a claim about what
Track B can do) -- it is not the default and must never feed Track B.

Same shared architecture code as mephy_repro/, adapted only via the
modality dimensions and number of output classes (3, not 4: UL-DD's
Low/Medium/High KSS binning per common/labels.py).

Sizing matches tasks.md 3.1's spec (compute is not a training constraint,
per tasks.md Section 0): GNN 4 layers / 256 hidden dim (sweepable to 512),
Transformer 4 layers / 4-8 heads / 64-128 dim per head, 12-24 window
temporal context.
"""
from __future__ import annotations

from common.dataset import feature_groups
from common.fatiguenet_model import FatigueNetModel
from common.labels import CLASS_NAMES

MODALITY_DIMS = {name: len(cols) for name, cols in feature_groups().items()}
NUM_CLASSES = len(CLASS_NAMES)


def build_track_a_model(
    hidden_dim: int = 256,
    gnn_layers: int = 4,
    gnn_hops: int = 3,
    gnn_heads: int = 4,
    transformer_layers: int = 4,
    transformer_heads: int = 4,
    transformer_head_dim: int = 64,
    dropout: float = 0.4,
    context_len: int = 12,
    include_telemetry: bool = False,
) -> FatigueNetModel:
    modality_dims = {name: len(cols) for name, cols in feature_groups(include_telemetry).items()}
    return FatigueNetModel(
        modality_dims,
        num_classes=NUM_CLASSES,
        hidden_dim=hidden_dim,
        gnn_layers=gnn_layers,
        gnn_hops=gnn_hops,
        gnn_heads=gnn_heads,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        transformer_head_dim=transformer_head_dim,
        dropout=dropout,
        max_context_len=context_len,
    )
