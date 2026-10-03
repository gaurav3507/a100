"""Track B (tasks.md Task 4.1): the actual deployment candidate, must run
on Jetson Orin Nano 8GB. Three options implemented and compared on the
same features/splits/labels as Track A (common/dataset.py), so the
comparison in Phase 7 is apples-to-apples:

  1. Small MLP (recommended default): 2-3 hidden layers, <50K params,
     single-window classification on the flat 40-dim feature vector.
  2. LightGBM: no GPU needed at all for inference, trivial to deploy.
  3. Small GRU (only if 1/2 underperform): temporal trend over the last
     6-12 windows, single layer, hidden size 32-64, <100K params.

All three consume common/dataset.py's ULDDWindowDataset (MLP/LightGBM,
single-window) or ULDDSequenceDataset (GRU, short temporal context) --
same feature table, same LOSO folds as Track A.

Input is `common.dataset.CORE_FEATURES`: vision + bio + grip only, never
UL-DD's Telemetry columns. Unlike Track A (which can optionally opt into
telemetry for a ceiling-only ablation, see track_a_heavy/model.py), Track B
is the actual deployment candidate and must only ever see features the
real ESP32-S3/Jetson sensor set can produce at inference time -- there is
no vehicle-telemetry sensor in that hardware.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from common.dataset import CORE_FEATURES

INPUT_DIM = len(CORE_FEATURES)  # 37: vision(17) + bio(15) + grip(5)


class MLPClassifier(nn.Module):
    """Option 1: small MLP, 64 -> 32 -> 16 -> num_classes by default."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dims: tuple[int, ...] = (64, 32, 16),
                 num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GRUClassifier(nn.Module):
    """Option 3: single-layer GRU over the last context_len windows,
    hidden size 32-64, <100K params -- only used if MLP/LightGBM
    underperform (tasks.md 4.1)."""

    def __init__(self, input_dim: int = INPUT_DIM, hidden_size: int = 48,
                 num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, context_len, input_dim)
        _, h_n = self.gru(x)  # h_n: (1, batch, hidden_size)
        return self.head(self.dropout(h_n.squeeze(0)))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_lightgbm_params(num_classes: int = 3, num_leaves: int = 31, max_depth: int = -1,
                           n_estimators: int = 200, learning_rate: float = 0.05) -> dict:
    """Option 2: LightGBM hyperparameters. Trained directly with
    lightgbm.LGBMClassifier in track_b_light/train.py -- no torch module
    needed, hence returning a plain params dict rather than an nn.Module."""
    return dict(
        objective="multiclass", num_class=num_classes, num_leaves=num_leaves,
        max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate,
        class_weight="balanced",
    )
