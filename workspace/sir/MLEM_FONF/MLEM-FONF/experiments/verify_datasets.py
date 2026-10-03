"""Inventory + smoke-load the local datasets (run on the machine holding the data).

Usage (on ant-pc):
    python experiments/verify_datasets.py --root /media/ant-pc/HDD2/datasets

Prints a machine-readable inventory, loads a few slices from each dataset,
and writes a preview grid to results/dataset_preview.png. Paste the console
output back for loader lock-in; no patient data leaves your machine.
"""
import argparse, json, pathlib, sys, traceback
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--preview", type=int, default=4)
    args = ap.parse_args()
    from mlem_fonf.version import banner
    banner("verify_datasets")
    from mlem_fonf import data as D

    print("=" * 60)
    print("MAYO / AAPM LDCT")
    try:
        series = D.find_mayo_series(args.root)
        inv = {p: {d: len(fs) for d, fs in dd.items()} for p, dd in sorted(series.items())}
        print(json.dumps(inv, indent=1))
        missing = [p for s in D.MAYO_SPLITS.values() for p in s if p not in series]
        print("patients missing vs expected split:", missing or "none")
        print("thickness breakdown (1mm/3mm/other):")
        print(json.dumps(D.thickness_report(series), indent=1))
    except Exception:
        traceback.print_exc()

    print("=" * 60)
    print("LoDoPaB-CT")
    try:
        lod = D.find_lodopab(args.root)
        print("unique files per part:", json.dumps({k: len(v) for k, v in lod.items()}))
        dups = D.lodopab_duplicates(args.root)
        if dups:
            b0 = sorted(dups)[0]
            print(f"DUPLICATE copies detected for {len(dups)} files; e.g. {b0}:")
            for p in dups[b0]:
                print("   ", p)
        else:
            print("no duplicate copies")
    except Exception:
        traceback.print_exc()

    # smoke-load + preview
    tiles, titles = [], []
    try:
        for pat, i, img in D.mayo_slices(args.root, "test", "full", limit=args.preview):
            tiles.append(img); titles.append(f"Mayo {pat} #{i}  [{img.min():.2f},{img.max():.2f}]")
    except Exception:
        traceback.print_exc()
    try:
        for fi, si, img in D.lodopab_slices(args.root, "test", limit=args.preview):
            tiles.append(img); titles.append(f"LoDoPaB f{fi}s{si}  [{img.min():.2f},{img.max():.2f}]")
    except Exception:
        traceback.print_exc()

    if tiles:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        n = len(tiles)
        fig, ax = plt.subplots(1, n, figsize=(2.6 * n, 2.9))
        ax = np.atleast_1d(ax)
        for a, im, t in zip(ax, tiles, titles):
            a.imshow(im, cmap="gray", vmin=0, vmax=1); a.set_title(t, fontsize=7); a.axis("off")
        out = pathlib.Path(__file__).resolve().parents[1] / "results" / "dataset_preview.png"
        fig.tight_layout(); fig.savefig(out, dpi=140)
        print("preview:", out, f"({n} tiles)")
    print("=" * 60)
    print("Paste everything above back for loader lock-in.")
