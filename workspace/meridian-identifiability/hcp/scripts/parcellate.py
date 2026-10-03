import nibabel as nib, numpy as np

ATLAS = "/workspace/hcp/atlas/Schaefer2018_200Parcels_17Networks_order.dlabel.nii"
N_PARCELS = 200
HEMI_V = 32492   # fsLR 32k vertices per hemisphere

def load_atlas():
    lab = nib.load(ATLAS).get_fdata().ravel().astype(int)
    assert lab.size == 2 * HEMI_V, lab.size
    return lab

def parcellate(dtseries_path, atlas_labels):
    img = nib.load(dtseries_path)
    data = img.get_fdata(dtype=np.float32)
    ax = img.header.get_axis(1)
    T = data.shape[0]
    sums = np.zeros((T, N_PARCELS), dtype=np.float64)
    counts = np.zeros(N_PARCELS, dtype=np.int64)
    for name, sl, bm in ax.iter_structures():
        if name == "CIFTI_STRUCTURE_CORTEX_LEFT":
            labs = atlas_labels[bm.vertex]
        elif name == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            labs = atlas_labels[HEMI_V + bm.vertex]
        else:
            continue
        block = data[:, sl]
        for p in range(1, N_PARCELS + 1):
            m = labs == p
            if m.any():
                sums[:, p - 1] += block[:, m].sum(1)
                counts[p - 1] += m.sum()
    assert (counts > 0).all(), f"empty parcels: {np.where(counts == 0)[0] + 1}"
    return (sums / counts).astype(np.float32), counts

if __name__ == "__main__":
    import sys
    atlas = load_atlas()
    ts, counts = parcellate(sys.argv[1], atlas)
    print("time-series shape (T, parcels):", ts.shape)
    print("vertices per parcel: min", counts.min(), "max", counts.max(), "total", counts.sum())
    print("mean abs value:", np.abs(ts).mean())
    lag1 = np.mean([np.corrcoef(ts[:-1, i], ts[1:, i])[0, 1] for i in range(N_PARCELS)])
    print("mean lag-1 autocorrelation:", round(float(lag1), 3))
    print("any NaN:", bool(np.isnan(ts).any()), " any all-zero parcel:", bool((ts.std(0) == 0).any()))
