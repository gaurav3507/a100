"""Transformer early-fusion (plan §7 contender): a single Transformer
encoder over the temporal sequence of flat (all-modality) feature vectors --
the second-best published architecture on UL-DD (86.39% in the dataset
paper's Table 8, described there as "leveraging self-attention and
positional encoding to learn long-range dependencies across the fused
feature sequence").

Reuses common/fatiguenet_model.py's TemporalTransformer (learned positional
embedding + standard encoder) so the temporal block is the same code the
v1 ablation already credited (-4.30 pp when removed); the only new pieces
are a linear input embedding and the classifier head.

Input: ULDDSequenceDataset's "sequence" tensor, (batch, T, n_features).
"""
from __future__ import annotations

from torch import Tensor, nn

from common.fatiguenet_model import TemporalTransformer


class TransformerFusion(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        head_dim: int = 32,
        dropout: float = 0.3,
        max_context_len: int = 24,
    ):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.temporal = TemporalTransformer(
            hidden_dim, num_layers, num_heads, head_dim, dropout, max_len=max_context_len
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, T, input_dim); classify the last (current) timestep
        h = self.temporal(self.embed(x))
        return self.classifier(self.dropout(h[:, -1, :]))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
