"""I2M2 -- Inter- & Intra-Modality Modeling (Madaan et al., NeurIPS 2024),
the current best published result on UL-DD (88.03% under the dataset
authors' stratified 5-fold protocol) and the plan's headline multimodal
architecture (plan §7).

Structure, per the paper's product-of-experts strategy as described in the
UL-DD paper's validation section: one expert model per modality
(intra-modality patterns) plus one joint model over the concatenated
features (inter-modality dependencies); the final prediction multiplies
their probability outputs. In log space that is a SUM of per-expert
log-softmax vectors -- so a modality whose expert is uncertain (flat
distribution) contributes ~nothing, while a confident expert moves the
product. That is precisely the property v1's naive concatenation lacked:
a weak modality could not be ignored. Here it silences itself.

Loss: cross-entropy on the combined product prediction PLUS cross-entropy
on every individual expert and the joint model -- each expert must stand on
its own, which keeps experts calibrated and doubles as a free per-modality
ablation readout at eval time.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _mlp(in_dim: int, hidden: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden // 2, num_classes),
    )


class I2M2Model(nn.Module):
    """Input: {modality: (batch, dim_m)} single-window node tensors
    (ULDDWindowDataset's "nodes"). Output dict:
      expert_logits: {modality: (batch, C)}   intra-modality experts
      joint_logits:  (batch, C)               inter-modality model
      logits:        (batch, C)               product-of-experts combination
                     (sum of log-softmaxes; unnormalized log-probs, valid
                     input to argmax and F.cross_entropy alike)
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        num_classes: int = 3,
        expert_hidden: int = 64,
        joint_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.modalities = list(modality_dims.keys())
        self.experts = nn.ModuleDict(
            {m: _mlp(d, expert_hidden, num_classes, dropout) for m, d in modality_dims.items()}
        )
        self.joint = _mlp(sum(modality_dims.values()), joint_hidden, num_classes, dropout)

    def forward(self, nodes: dict[str, Tensor]) -> dict:
        expert_logits = {m: self.experts[m](nodes[m]) for m in self.modalities}
        joint_logits = self.joint(torch.cat([nodes[m] for m in self.modalities], dim=-1))
        combined = F.log_softmax(joint_logits, dim=-1)
        for m in self.modalities:
            combined = combined + F.log_softmax(expert_logits[m], dim=-1)
        return {"expert_logits": expert_logits, "joint_logits": joint_logits, "logits": combined}


def i2m2_loss(out: dict, labels: Tensor) -> tuple[Tensor, dict[str, float]]:
    parts = {"combined": F.cross_entropy(out["logits"], labels),
             "joint": F.cross_entropy(out["joint_logits"], labels)}
    for m, logits in out["expert_logits"].items():
        parts[f"expert_{m}"] = F.cross_entropy(logits, labels)
    total = sum(parts.values())
    return total, {k: v.item() for k, v in parts.items()} | {"total": total.item()}
