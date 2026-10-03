"""GNN-v2 (plan §7): the capstone proposal's promised GNN + Transformer
core, rebuilt for a fair trial -- not retired, demoted to a hypothesis the
clean ablation gets to judge.

What changed from the v1 FatigueNet clone, and why (each backed by a
measured number from the defect-free v1 ablation):

  - MGAF fusion REMOVED (use_mgaf=False -> plain mean fusion): removing it
    IMPROVED accuracy by +0.81 pp. Without MGAF there is also no decoder,
    so the reconstruction loss disappears with it (lambda2 pinned to 0;
    removing recon alone was worth +2.78 pp).
  - Transformer KEPT: the one component that earned its place (-4.30 pp
    when removed).
  - GNN KEPT but given a real graph: v1 fed it a trivial 3-node graph where
    the vision node was near-noise under LOSO (35.8%) -- a starved input,
    not a fair trial. v2 uses the 6 fine-grained nodes over the enriched
    features (face_geom / fau / posture / cardiac / autonomic /
    grip_motion), where cross-node correlation structure can actually
    exist. v1's GNN measured neutral (-0.83 pp), never harmful.
  - CAPACITY SHRUNK (hidden 96, 2 GNN layers / 2 hops, 2 transformer
    layers): 16k windows overfit the v1 sizing from epoch ~6; the fair
    trial should not be an overfitting trial.

Verdict rule (plan Phase 3): GNN-v2 runs head-to-head with I2M2 /
Transformer-fusion / LightGBM under Tier 1, plus its own no-GNN ablation.
It stays if it earns its place; if it still contributes <=1 pp it is
retired with evidence, and either outcome goes in the report.
"""
from __future__ import annotations

from common.dataset import V2_FEATURE_GROUPS
from common.fatiguenet_model import FatigueNetModel
from common.labels import CLASS_NAMES

NUM_CLASSES = len(CLASS_NAMES)

# Reconstruction loss is removed by construction (no MGAF -> no decoder);
# keep the constant so training code states the intent explicitly.
GNN_V2_LAMBDA2 = 0.0


def build_gnn_v2(
    hidden_dim: int = 96,
    gnn_layers: int = 2,
    gnn_hops: int = 2,
    gnn_heads: int = 4,
    transformer_layers: int = 2,
    transformer_heads: int = 4,
    transformer_head_dim: int = 32,
    dropout: float = 0.3,
    context_len: int = 12,
    use_gnn: bool = True,
    groups: dict[str, list[str]] | None = None,
) -> FatigueNetModel:
    """`use_gnn=False` is the fair-trial ablation arm: identical everything,
    graph block skipped -- the delta between the two IS the GNN's verdict."""
    node_groups = groups if groups is not None else V2_FEATURE_GROUPS
    modality_dims = {name: len(cols) for name, cols in node_groups.items()}
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
        use_gnn=use_gnn,
        use_transformer=True,
        use_mgaf=False,  # measured harmful in v1 (+0.81 pp when removed)
    )
