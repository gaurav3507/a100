"""Export Mayo training slices to a single npz (for DGX transfer).

  python training/export_train_npz.py --root /media/ant-pc/HDD2/datasets \
      --out mayo_train.npz --limit 1200
"""
import argparse, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from mlem_fonf import data as D

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="mayo_train.npz")
    ap.add_argument("--limit", type=int, default=1200)
    a = ap.parse_args()
    X = np.stack([ref for _, _, ref in
                  D.mayo_slices(a.root, "train", "full", every=2,
                                limit=a.limit)]).astype(np.float32)
    np.savez_compressed(a.out, slices=X)
    print(f"saved {X.shape} -> {a.out} "
          f"({pathlib.Path(a.out).stat().st_size/1e6:.0f} MB)")
