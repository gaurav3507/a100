"""Track B ONNX export (tasks.md Task 4.4): torch.onnx.export for the MLP
(or GRU) variant, then verify the ONNX Runtime output matches the PyTorch
output numerically before proceeding -- silently shipping a broken export
would only surface as a mysterious accuracy drop on the Jetson.

The exported model is trained on *all* subjects (no held-out fold) --
LOSO is for evaluating generalization (track_b_light/train.py), but the
actual deployment artifact should use every available subject's data.

Usage:
  .venv/Scripts/python.exe track_b_light/export_onnx.py --model mlp --epochs 30
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader

from common.dataset import ULDDSequenceDataset, ULDDWindowDataset
from track_b_light.model import GRUClassifier, INPUT_DIM, MLPClassifier

NUM_CLASSES = 3
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "track_b_export"


def train_deploy_mlp(epochs: int, lr: float, dropout: float, batch_size: int) -> MLPClassifier:
    dataset = ULDDWindowDataset(folds=list(range(16)))  # exclude fold=-1 (no-drowsy subjects)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = MLPClassifier(input_dim=INPUT_DIM, num_classes=NUM_CLASSES, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss, n = 0.0, 0
        for batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch["flat"]), batch["label"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/n:.4f}")
    return model.eval()


def train_deploy_gru(epochs: int, lr: float, dropout: float, batch_size: int, context_len: int, hidden: int) -> GRUClassifier:
    dataset = ULDDSequenceDataset(context_len=context_len, folds=list(range(16)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = GRUClassifier(input_dim=INPUT_DIM, hidden_size=hidden, num_classes=NUM_CLASSES, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss, n = 0.0, 0
        for batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch["sequence"]), batch["label"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/n:.4f}")
    return model.eval()


def export_and_verify(model: torch.nn.Module, dummy_input: torch.Tensor, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False: torch 2.13's default dynamo-based exporter prints a
    # unicode checkmark that crashes on Windows' cp1252 console codepage
    # (UnicodeEncodeError), and is overkill for a model this simple anyway.
    # The traditional TorchScript-based exporter is more mature here.
    torch.onnx.export(
        model, dummy_input, str(out_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    with torch.no_grad():
        torch_out = model(dummy_input).numpy()

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"input": dummy_input.numpy()})[0]

    max_abs_diff = np.abs(torch_out - onnx_out).max()
    print(f"Exported to {out_path}")
    print(f"Max abs diff between PyTorch and ONNX Runtime outputs: {max_abs_diff:.2e}")
    if max_abs_diff > 1e-4:
        raise RuntimeError(
            f"ONNX output diverges from PyTorch by {max_abs_diff:.2e} (> 1e-4 tolerance) -- export is not numerically equivalent"
        )
    print("Verified: ONNX output matches PyTorch output within tolerance.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "gru"], default="mlp")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-len", type=int, default=8)
    parser.add_argument("--gru-hidden", type=int, default=48)
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.model == "mlp":
        model = train_deploy_mlp(args.epochs, args.lr, args.dropout, args.batch_size)
        dummy = torch.randn(4, INPUT_DIM)
    else:
        model = train_deploy_gru(
            args.epochs, args.lr, args.dropout, args.batch_size, args.context_len, args.gru_hidden
        )
        dummy = torch.randn(4, args.context_len, INPUT_DIM)

    out_path = OUT_DIR / f"track_b_{args.model}.onnx"
    export_and_verify(model, dummy, out_path)

    n_params = sum(p.numel() for p in model.parameters())
    size_kb = out_path.stat().st_size / 1024
    print(f"Model params: {n_params:,}  ONNX file size: {size_kb:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
