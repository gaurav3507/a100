"""Shared GNN + Transformer + MetaNet/MGAF architecture, adapted from the
FatigueNet paper (Zendehbad, Ghasemi & Samsami Khodadad, Sci Rep 2025,
doi:10.1038/s41598-025-00640-z). Used by both mephy_repro/ (Task 1.2/3.1's
reproduction sanity check, on MePhy's native ECG/EDA/EMG/blink nodes) and
track_a_heavy/ (Task 3.1, on our own vision/bio/grip nodes -- our actual
camera + physiological + pressure sensor set, see track_a_heavy/model.py
for why that's not the same three-or-four modalities as either the paper
or UL-DD's own full modality list) -- tasks.md is explicit that this must
be the *same code* in both places, since the whole point of the MePhy
reproduction is validating the reimplementation independent of which
dataset/modalities it's pointed at.

Every design decision below that isn't unambiguous in the paper is
documented at the point it's made -- the paper's Methods section is loose
in a few places (e.g. Eq. 4's "correlation between modality feature
vectors" doesn't say over what axis; Fig. 3/5/7 show 5 graph nodes for a
4-modality model). tasks.md already resolves the biggest one: replace the
paper's MSVM (RBF kernel, C=300) classifier head with a plain softmax
linear layer so the whole model trains end-to-end with backprop.

Paper equation references, for traceability:
  Eq. 3   message passing (superseded here by Eq. 5's multi-hop filter)
  Eq. 4   dynamic correlation-based adjacency        -> dynamic_adjacency()
  Eq. 5,7 multi-hop graph filter + ReLU               -> GraphConvLayer
  Eq. 6   per-node L2 normalization                   -> GraphConvLayer
  Eq. 8   attention-enhanced neighbor aggregation      -> NodeAttention
  Eq. 9   transformer "edge weight" adaptation         -> not implemented
          literally (see TemporalTransformer docstring); a standard
          Transformer encoder is used instead, matching tasks.md 3.1's
          "4 layers, 4-8 heads" simplification.
  Eq. 10  MetaNet per-modality relevance weights       -> MetaNet
  Eq. 11  stage-wise weighted aggregation              -> MGAF.forward
  Eq. 12  attention-refined per-modality embeddings    -> MGAF.forward
  Eq. 14  cross-entropy loss                           -> fatiguenet_loss()
  Eq. 15  + L2 weight reg + reconstruction consistency -> fatiguenet_loss()
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def dynamic_adjacency(node_embeddings: Tensor) -> Tensor:
    """Eq. 4: A_ij = corr(Xi, Xj) / sum_{k!=i} |corr(Xi, Xk)|.

    tasks.md 3.1 specifies this is computed "from pairwise correlation
    between modality feature vectors within a batch" -- read literally,
    each node's embedding is pooled to one scalar per sample (mean over
    the hidden dim), then Pearson-correlated *across the batch dimension*
    between every pair of nodes, giving one (M, M) adjacency shared by the
    whole batch (a genuinely "dynamic, data-driven" graph that changes
    batch to batch, matching the paper's framing, without requiring a
    same-shape raw signal per node to correlate directly).

    Needs >= 2 samples in the batch dimension to define a correlation;
    falls back to a uniform (non-self) adjacency for batch size 1 (e.g.
    single-sample inference).
    """
    batch = node_embeddings.shape[0]
    num_nodes = node_embeddings.shape[1]
    if batch < 2:
        A = torch.ones(num_nodes, num_nodes, device=node_embeddings.device)
        A.fill_diagonal_(0)
        return A / (num_nodes - 1)

    pooled = node_embeddings.mean(dim=-1)  # (batch, M)
    centered = pooled - pooled.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, keepdim=True) + 1e-8
    normed = centered / std
    corr = (normed.T @ normed) / batch  # (M, M) Pearson correlation
    corr = corr - torch.diag(torch.diagonal(corr))  # zero the diagonal (sum_{k!=i})
    abs_row_sum = corr.abs().sum(dim=1, keepdim=True) + 1e-8
    return corr / abs_row_sum


class GraphConvLayer(nn.Module):
    """Eq. 5 (multi-hop polynomial graph filter) + Eq. 7 (ReLU) + Eq. 6
    (L2-normalize each node's output). One learnable linear map per hop
    order k=1..num_hops, applied to the k-hop-propagated features and
    summed -- this is the multi-scale short/long-range filtering the paper
    describes, without needing the full (channel_in, channel_out, hop)
    coefficient tensor of Eq. 5 written out explicitly."""

    def __init__(self, hidden_dim: int, num_hops: int = 3):
        super().__init__()
        self.num_hops = num_hops
        self.hop_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hops)])
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        # x: (batch, M, H), adjacency: (M, M)
        num_nodes = adjacency.shape[0]
        a_power = torch.eye(num_nodes, device=adjacency.device, dtype=adjacency.dtype)
        out = self.bias
        for hop_linear in self.hop_linears:
            a_power = a_power @ adjacency  # A^k
            propagated = torch.einsum("ij,bjh->bih", a_power, x)
            out = out + hop_linear(propagated)
        out = F.relu(out)
        return F.normalize(out, p=2, dim=-1, eps=1e-8)


class NodeAttention(nn.Module):
    """Eq. 8: attention-enhanced aggregation across nodes (Q/K/V self
    attention over the small modality-node set, applied once after the
    stack of graph-conv layers)."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.mha(x, x, x)
        return F.relu(out)


class FatigueGNN(nn.Module):
    """GNN block: `num_layers` GraphConvLayers, each with a freshly
    recomputed dynamic adjacency (Eq. 4) from the current node embeddings,
    followed by one NodeAttention (Eq. 8) refinement pass."""

    def __init__(self, hidden_dim: int = 256, num_layers: int = 4, num_hops: int = 3, num_heads: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([GraphConvLayer(hidden_dim, num_hops) for _ in range(num_layers)])
        self.attn = NodeAttention(hidden_dim, num_heads)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            adjacency = dynamic_adjacency(x)
            x = layer(x, adjacency)
        return self.attn(x)


class TemporalTransformer(nn.Module):
    """Standard Transformer encoder over a temporal sequence of per-node
    embeddings (tasks.md 3.1: 4 layers, 4-8 heads, 64-128 dim/head). The
    paper's Eq. 9 describes a nonstandard "edge weight" hybrid of learned
    attention and adjacency priors specific to their framing of the
    Transformer as another graph-like operator; that's not literally
    implemented here in favor of a standard encoder, per tasks.md's own
    simplification of this block."""

    def __init__(self, hidden_dim: int = 256, num_layers: int = 4, num_heads: int = 4,
                 head_dim: int = 64, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=head_dim * num_heads * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim) * 0.02)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        return self.encoder(x + self.pos_embed[:, :seq_len, :])


class MetaNet(nn.Module):
    """Eq. 10: alpha_m = MetaNet(Fm, Theta), a lightweight MLP producing a
    per-modality relevance logit, softmax-normalized across modalities so
    the weights in Eq. 11's weighted sum sum to 1."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, modality_stack: Tensor) -> Tensor:
        # modality_stack: (batch, M, H) -> (batch, M)
        logits = self.mlp(modality_stack).squeeze(-1)
        return torch.softmax(logits, dim=-1)


class MGAF(nn.Module):
    """MetaNet + Meta-Gated Adaptive Fusion: Eq. 11 (stage-wise weighted
    aggregation) followed by Eq. 12 (attention-refined residual), plus a
    small decoder used only to produce the reconstruction targets for the
    Eq. 15 consistency loss (tasks.md 3.1 explicitly asks to include this
    loss term rather than skip it, and ablate it separately in Task 3.4)."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.metanet = MetaNet(hidden_dim)
        self.refine_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, modality_stack: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # modality_stack: (batch, M, H)
        alpha = self.metanet(modality_stack)  # (batch, M), Eq. 10
        fusion = (alpha.unsqueeze(-1) * modality_stack).sum(dim=1)  # (batch, H), Eq. 11

        query = fusion.unsqueeze(1)  # (batch, 1, H)
        refined, _ = self.refine_attn(query, modality_stack, modality_stack)  # Eq. 12
        fusion_refined = fusion + refined.squeeze(1)  # residual connection

        reconstructed = self.decoder(fusion_refined).unsqueeze(1).expand_as(modality_stack)
        recon_loss = F.mse_loss(reconstructed, modality_stack)  # Eq. 15's ||Fk - Fhat_k||^2 term

        return fusion_refined, alpha, recon_loss


class FatigueNetModel(nn.Module):
    """Full pipeline: per-modality input embedding -> GNN (per timestep,
    across modality nodes) -> Transformer (per modality, across time) ->
    MetaNet+MGAF fusion -> softmax classifier head (tasks.md 3.1: no MSVM).

    Input: a dict {modality_name: (batch, T, raw_feature_dim)} of raw
    per-window feature sequences, T = temporal context length (the last
    timestep is the "current" window being classified).

    `use_gnn`/`use_transformer`/`use_mgaf` back the four ablation variants
    tasks.md 3.4 asks for (full model, no-GNN, no-Transformer, no-MGAF; the
    fifth variant, no-reconstruction-loss, needs no model change -- just
    set lambda2=0 in fatiguenet_loss):
      - no-GNN: skip straight from per-modality embeddings to the temporal
        stage (tasks.md: "concatenation instead" -- here, no cross-modal
        mixing at all before the Transformer, which is the GNN's entire
        job, so simply not running it *is* the concatenation-equivalent
        baseline).
      - no-Transformer: skip temporal attention, use the last window's
        embedding directly (tasks.md: "use last-window features only").
      - no-MGAF: replace MetaNet-weighted fusion + attention refinement
        with a plain unweighted average across modalities (tasks.md:
        "simple average fusion instead"), and recon_loss is reported as 0
        since there's no MGAF decoder to reconstruct from.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        num_classes: int,
        hidden_dim: int = 256,
        gnn_layers: int = 4,
        gnn_hops: int = 3,
        gnn_heads: int = 4,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        transformer_head_dim: int = 64,
        mgaf_heads: int = 4,
        dropout: float = 0.4,
        max_context_len: int = 64,
        use_gnn: bool = True,
        use_transformer: bool = True,
        use_mgaf: bool = True,
    ):
        super().__init__()
        self.modalities = list(modality_dims.keys())
        self.hidden_dim = hidden_dim
        self.use_gnn = use_gnn
        self.use_transformer = use_transformer
        self.use_mgaf = use_mgaf

        self.embed = nn.ModuleDict(
            {m: nn.Sequential(nn.Linear(d, hidden_dim), nn.ReLU()) for m, d in modality_dims.items()}
        )
        self.gnn = FatigueGNN(hidden_dim, gnn_layers, gnn_hops, gnn_heads) if use_gnn else None
        self.temporal = (
            TemporalTransformer(
                hidden_dim, transformer_layers, transformer_heads, transformer_head_dim,
                dropout, max_len=max_context_len,
            )
            if use_transformer else None
        )
        self.mgaf = MGAF(hidden_dim, mgaf_heads) if use_mgaf else None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, node_sequences: dict[str, Tensor]) -> dict[str, Tensor]:
        batch, seq_len, _ = node_sequences[self.modalities[0]].shape
        num_modalities = len(self.modalities)

        embedded = torch.stack(
            [self.embed[m](node_sequences[m]) for m in self.modalities], dim=2
        )  # (batch, T, M, H)

        if self.use_gnn:
            flat = embedded.reshape(batch * seq_len, num_modalities, self.hidden_dim)
            embedded = self.gnn(flat).reshape(batch, seq_len, num_modalities, self.hidden_dim)

        if self.use_transformer:
            per_modality_seq = embedded.permute(0, 2, 1, 3).reshape(
                batch * num_modalities, seq_len, self.hidden_dim
            )
            temporal_out = self.temporal(per_modality_seq)
            last_step = temporal_out[:, -1, :]
        else:
            last_step = embedded[:, -1, :, :].reshape(batch * num_modalities, self.hidden_dim)

        modality_stack = last_step.reshape(batch, num_modalities, self.hidden_dim)

        if self.use_mgaf:
            fused, alpha, recon_loss = self.mgaf(modality_stack)
        else:
            fused = modality_stack.mean(dim=1)  # tasks.md 3.4: "simple average fusion instead"
            alpha = torch.full(
                (batch, num_modalities), 1.0 / num_modalities, device=fused.device, dtype=fused.dtype
            )
            recon_loss = torch.zeros((), device=fused.device, dtype=fused.dtype)

        logits = self.classifier(self.dropout(fused))
        return {"logits": logits, "alpha": alpha, "recon_loss": recon_loss}


def fatiguenet_loss(
    logits: Tensor, labels: Tensor, recon_loss: Tensor, classifier_weight: Tensor,
    lambda1: float = 0.01, lambda2: float = 0.05,
) -> tuple[Tensor, dict[str, float]]:
    """Eq. 14 + 15: cross-entropy + L2 weight regularization (applied to
    the classifier head's weights) + reconstruction consistency."""
    ce = F.cross_entropy(logits, labels)
    l2 = classifier_weight.pow(2).sum()
    total = ce + lambda1 * l2 + lambda2 * recon_loss
    return total, {"ce": ce.item(), "l2": l2.item(), "recon": recon_loss.item(), "total": total.item()}
