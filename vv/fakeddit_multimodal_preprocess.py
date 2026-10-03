"""
fakeddit_multimodal_preprocess.py

Loads and preprocesses the Fakeddit dataset for multimodal (text + image)
fake news detection research, following the paper's own evaluation protocol:
    "results in the paper are based on multimodal samples only
     (samples that have both text and image)."

WHAT THIS DOES
--------------
1. Loads the multimodal_only_samples tsv split(s) (train / validation / test).
2. Cleans up the dataframe (drops Unnamed:* columns, uses clean_title as text).
3. Verifies that every referenced image actually exists on disk and is a
   readable, non-corrupt image file.
4. Drops any row whose image is missing/corrupt -- it does NOT silently
   fall back to a text-only sample. This is deliberate: for multimodal
   benchmarking, a "sample with both text and image" that quietly loses
   its image during preprocessing will corrupt your reported multimodal
   numbers without any error being raised.
5. Writes out a cleaned, image-verified manifest (csv) + prints a summary
   report of how many rows were dropped and why.
6. Optionally builds a PyTorch Dataset (see FakedditMultimodalDataset) that
   returns (image_tensor, text, label) tuples, ready for a DataLoader.

EXPECTED DIRECTORY LAYOUT (adjust paths in main() as needed)
--------------------------------------------------------------
fakeddit_root/
    multimodal_only_samples/
        multimodal_train.tsv
        multimodal_validate.tsv
        multimodal_test_public.tsv
    images/
        <submission_id>.jpg   (or whatever extension the downloader saved)

USAGE
-----
$ python fakeddit_multimodal_preprocess.py \
    --data_dir /path/to/fakeddit_root/multimodal_only_samples \
    --image_dir /path/to/fakeddit_root/images \
    --split train \
    --out_csv ./fakeddit_train_verified.csv
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

try:
    from PIL import Image
except ImportError:
    Image = None  # Pillow check happens at runtime with a clear error message


# ----------------------------------------------------------------------
# Label columns in Fakeddit (v2.0). Pick the granularity you need:
#   2_way_label   -> real (0) vs fake (1)
#   3_way_label   -> real / fake-with-true-text / fake-with-false-text
#   6_way_label   -> fine-grained (true, satire/parody, misleading content,
#                    manipulated content, false connection, imposter content)
# ----------------------------------------------------------------------
LABEL_COLUMNS = ["2_way_label", "3_way_label", "6_way_label"]

SPLIT_FILENAMES = {
    "train": "multimodal_train.tsv",
    "validate": "multimodal_validate.tsv",
    "test": "multimodal_test_public.tsv",
}

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def load_split(data_dir: Path, split: str) -> pd.DataFrame:
    """Load one multimodal_only_samples tsv split and do basic cleanup."""
    fname = SPLIT_FILENAMES.get(split)
    if fname is None:
        raise ValueError(f"Unknown split '{split}'. Choose from {list(SPLIT_FILENAMES)}")

    path = data_dir / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Make sure --data_dir points at the "
            f"'multimodal_only_samples' folder from the v2.0 download."
        )

    df = pd.read_csv(path, sep="\t")

    # Drop stray Unnamed:* index columns, as the dataset README instructs
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    if "clean_title" not in df.columns:
        raise KeyError(
            "Expected a 'clean_title' column (filtered text field) but "
            f"didn't find one. Columns present: {list(df.columns)}"
        )

    if "id" not in df.columns and "submission_id" not in df.columns:
        raise KeyError(
            "Expected an 'id' or 'submission_id' column to match images. "
            f"Columns present: {list(df.columns)}"
        )

    return df


def find_image_path(image_dir: Path, submission_id: str) -> Path | None:
    """Return the path to this submission's image file if it exists, else None."""
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{submission_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def is_readable_image(path: Path) -> bool:
    """Open + verify the image isn't truncated/corrupt (common with scraped images)."""
    if Image is None:
        raise RuntimeError(
            "Pillow is required for image verification. Install with: "
            "pip install Pillow"
        )
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def verify_and_filter(df: pd.DataFrame, image_dir: Path) -> tuple[pd.DataFrame, dict]:
    """
    Keep only rows where the image file exists on disk AND opens cleanly.
    Returns the filtered dataframe plus a summary dict of drop reasons.
    """
    id_col = "id" if "id" in df.columns else "submission_id"

    image_paths = []
    drop_reasons = {"missing_file": 0, "corrupt_image": 0}
    keep_mask = []

    for sid in df[id_col].astype(str):
        img_path = find_image_path(image_dir, sid)
        if img_path is None:
            drop_reasons["missing_file"] += 1
            keep_mask.append(False)
            image_paths.append(None)
            continue

        if not is_readable_image(img_path):
            drop_reasons["corrupt_image"] += 1
            keep_mask.append(False)
            image_paths.append(None)
            continue

        keep_mask.append(True)
        image_paths.append(str(img_path))

    df = df.copy()
    df["image_path"] = image_paths
    df["_image_ok"] = keep_mask

    n_total = len(df)
    df_clean = df[df["_image_ok"]].drop(columns=["_image_ok"]).reset_index(drop=True)
    n_kept = len(df_clean)

    summary = {
        "total_rows": n_total,
        "kept_rows": n_kept,
        "dropped_rows": n_total - n_kept,
        **drop_reasons,
    }
    return df_clean, summary


def print_summary(summary: dict, split: str) -> None:
    print(f"\n=== Fakeddit [{split}] multimodal verification summary ===")
    print(f"Total rows in tsv:      {summary['total_rows']}")
    print(f"Rows with valid image:  {summary['kept_rows']}")
    print(f"Rows dropped:           {summary['dropped_rows']}")
    print(f"  - missing image file: {summary['missing_file']}")
    print(f"  - corrupt/unreadable: {summary['corrupt_image']}")
    if summary["dropped_rows"] > 0:
        pct = 100 * summary["dropped_rows"] / summary["total_rows"]
        print(
            f"\nNOTE: {pct:.1f}% of nominally-multimodal rows had no usable "
            f"image and were EXCLUDED, not text-only-backfilled. Re-run "
            f"image_downloader.py if this fraction looks too high before "
            f"trusting downstream multimodal results."
        )
    print()


# ----------------------------------------------------------------------
# Optional PyTorch Dataset wrapper
# ----------------------------------------------------------------------
def build_torch_dataset(df: pd.DataFrame, label_col: str = "2_way_label",
                         image_transform=None):
    """
    Returns a torch.utils.data.Dataset yielding (image_tensor, text, label).
    Only imports torch/torchvision if actually called.
    """
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms

    if label_col not in df.columns:
        raise KeyError(f"'{label_col}' not in dataframe columns: {list(df.columns)}")

    if image_transform is None:
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

    class FakedditMultimodalDataset(Dataset):
        def __init__(self, dataframe, transform):
            self.df = dataframe.reset_index(drop=True)
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = Image.open(row["image_path"]).convert("RGB")
            img = self.transform(img)
            text = row["clean_title"]
            label = int(row[label_col])
            return img, text, label

    return FakedditMultimodalDataset(df, image_transform)


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True,
                         help="Path to the 'multimodal_only_samples' folder")
    parser.add_argument("--image_dir", type=Path, required=True,
                         help="Path to the downloaded images folder")
    parser.add_argument("--split", choices=list(SPLIT_FILENAMES), default="train")
    parser.add_argument("--out_csv", type=Path, default=None,
                         help="Where to write the verified manifest csv")
    args = parser.parse_args()

    df = load_split(args.data_dir, args.split)
    df_clean, summary = verify_and_filter(df, args.image_dir)
    print_summary(summary, args.split)

    out_csv = args.out_csv or Path(f"fakeddit_{args.split}_verified.csv")
    df_clean.to_csv(out_csv, index=False)
    print(f"Wrote {len(df_clean)} verified multimodal rows -> {out_csv}")


if __name__ == "__main__":
    main()
