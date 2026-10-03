"""Torch port of the certified sparse projector (for learned baselines).

Gradients are supplied MANUALLY via the exact transpose (custom
autograd.Function), so no reliance on torch's sparse-CSR backward support:
d/dx [A x] = A^T (exact adjoint, same certified matrix). fp32 is enforced
and CUDA autocast disabled around the sparse ops (AMP-safe).
"""
from __future__ import annotations

import numpy as np


def to_torch_projector(A, device="cuda"):
    import torch
    Ac = A.tocsr()
    At = A.T.tocsr()

    def _csr(m):
        return torch.sparse_csr_tensor(
            torch.from_numpy(m.indptr.astype(np.int64)),
            torch.from_numpy(m.indices.astype(np.int64)),
            torch.from_numpy(m.data.astype(np.float32)),
            size=m.shape, device=device)

    Af, Ab = _csr(Ac), _csr(At)

    class _MM(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, M_fwd, M_grad):
            ctx.M_grad = M_grad
            with torch.autocast(device_type="cuda", enabled=False):
                return torch.sparse.mm(M_fwd, x.float().T).T

        @staticmethod
        def backward(ctx, gy):
            with torch.autocast(device_type="cuda", enabled=False):
                return torch.sparse.mm(ctx.M_grad, gy.float().T).T, None, None

    def forward(x):                    # (B, n*n) -> (B, m); grad via A^T
        return _MM.apply(x, Af, Ab)

    def backward(y):                   # (B, m) -> (B, n*n); grad via A
        return _MM.apply(y, Ab, Af)

    return forward, backward, A.shape
