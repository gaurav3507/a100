"""Dataset access for the JBHI campaign: Mayo/AAPM LDCT and LoDoPaB-CT.

Preprocessing decisions are LOCKED here (documented once, used everywhere):

* Mayo DICOM slices are converted to linear attenuation
      mu = mu_water * (1 + HU / 1000),   mu_water = 0.0192 /mm,
  clipped at mu >= 0 and normalized by mu(+1000 HU) so soft tissue sits
  mid-range in [0, 1]; slices are center-cropped/resized to 256 x 256.
* Patient-level splits (2016 challenge set): train = L067, L096, L143, L192,
  L286, L291, L310, L333 · val = L109 · test = L506. Training splits matter
  only for the learned baselines; FONF is training-free.
* Slice thinning: every 4th axial slice per series (reduces inter-slice
  correlation in statistics).
* LoDoPaB ground-truth images (362 x 362) are resized to 256 x 256 and are
  already in [0, 1].

Nothing here redistributes data: loaders read from a user-supplied local root
(e.g., /media/ant-pc/HDD2/datasets); the Mayo DUA stays intact.
"""
from __future__ import annotations

import os
import re
import glob
import pathlib

import numpy as np

MU_WATER = 0.0192
MAYO_SPLITS = {
    "train": ["L067", "L096", "L143", "L192", "L286", "L291", "L310", "L333"],
    "val": ["L109"],
    "test": ["L506"],
}
_DOSE_KEYS = {
    "full": ("full", "fd_", "_fd", "full dose", "full_dose"),
    "quarter": ("quarter", "qd_", "_qd", "low dose", "low_dose", "sim "),
}


def _to_256(img: np.ndarray, size: int = 256) -> np.ndarray:
    from skimage.transform import resize
    if img.shape != (size, size):
        img = resize(img, (size, size), anti_aliasing=True, preserve_range=True)
    return img.astype(np.float64)


def hu_to_unit(hu: np.ndarray) -> np.ndarray:
    """HU -> normalized linear attenuation in [0, ~1]."""
    mu = MU_WATER * (1.0 + hu / 1000.0)
    return np.clip(mu, 0.0, None) / (MU_WATER * 2.0)


# ---------------------------------------------------------------- Mayo LDCT
def find_mayo_series(root: str):
    """Discover Mayo LDCT DICOM series under `root`.

    Returns {patient: {dose: [sorted file list]}} for dose in {full, quarter}.
    Handles both the 2016 AAPM layout (L067/full_1mm/*.IMA) and TCIA exports
    (nested dirs whose path mentions Full/Low Dose)."""
    hits: dict = {}
    for f in glob.iglob(os.path.join(root, "**", "*"), recursive=True):
        if not f.lower().endswith((".ima", ".dcm")):
            continue
        m = re.search(r"(L\d{3})", f)
        if not m:
            continue
        pat = m.group(1)
        low = f.lower()
        dose = None
        for d, keys in _DOSE_KEYS.items():
            if any(k in low for k in keys):
                dose = d
                break
        if dose is None:
            dose = "unknown"
        hits.setdefault(pat, {}).setdefault(dose, []).append(f)
    for pat in hits:
        for dose in hits[pat]:
            hits[pat][dose].sort()
    return hits


def load_mayo_slice(path: str, size: int = 256) -> np.ndarray:
    """One DICOM slice -> normalized size x size attenuation image."""
    import pydicom
    ds = pydicom.dcmread(path)
    hu = ds.pixel_array.astype(np.float64) * float(
        getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    return _to_256(hu_to_unit(hu), size)


def thickness_report(series: dict) -> dict:
    """Per patient/dose: how many paths mention 1mm / 3mm / neither."""
    rep = {}
    for pat, dd in series.items():
        rep[pat] = {}
        for dose, files in dd.items():
            rep[pat][dose] = {
                "1mm": sum("1mm" in f.lower() for f in files),
                "3mm": sum("3mm" in f.lower() for f in files),
                "other": sum(("1mm" not in f.lower()) and ("3mm" not in f.lower())
                             for f in files),
            }
    return rep


def mayo_slices(root: str, split: str = "test", dose: str = "full",
                every: int = 4, limit: int | None = None,
                thickness: str = "1mm", size: int = 256):
    """Yield (patient, index, image) for the requested split.

    `thickness` filters file paths by substring ("1mm" default, the paper's
    working series); if no path matches, all files are used (with a warning)
    so odd layouts still load."""
    series = find_mayo_series(root)
    count = 0
    for pat in MAYO_SPLITS[split]:
        files = series.get(pat, {}).get(dose, [])
        if thickness:
            sel = [f for f in files if thickness.lower() in f.lower()]
            if sel:
                files = sel
            elif files:
                import warnings
                warnings.warn(f"{pat}/{dose}: no '{thickness}' paths; using all "
                              f"{len(files)} files")
        for i, f in enumerate(files[::every]):
            yield pat, i, load_mayo_slice(f, size)
            count += 1
            if limit and count >= limit:
                return


# ---------------------------------------------------------------- LoDoPaB-CT
def find_lodopab(root: str, dedup: bool = True):
    """Locate LoDoPaB ground-truth HDF5 files: {part: [files]}.

    With `dedup`, one file per unique basename is kept (first path wins);
    duplicate locations are retrievable via lodopab_duplicates()."""
    out: dict = {}
    for f in sorted(glob.iglob(os.path.join(root, "**", "ground_truth_*.hdf5"),
                               recursive=True)):
        m = re.search(r"ground_truth_(train|validation|test)_(\d+)\.hdf5",
                      os.path.basename(f))
        if m:
            out.setdefault(m.group(1), []).append(f)
    if dedup:
        for part in out:
            seen, uniq = set(), []
            for f in out[part]:
                b = os.path.basename(f)
                if b not in seen:
                    seen.add(b)
                    uniq.append(f)
            out[part] = uniq
    for part in out:
        out[part].sort()
    return out


def lodopab_duplicates(root: str) -> dict:
    """{basename: [paths]} for ground-truth files present more than once."""
    locs: dict = {}
    for f in sorted(glob.iglob(os.path.join(root, "**", "ground_truth_*.hdf5"),
                               recursive=True)):
        locs.setdefault(os.path.basename(f), []).append(f)
    return {b: ps for b, ps in locs.items() if len(ps) > 1}


def lodopab_slices(root: str, part: str = "test", every: int = 8,
                   limit: int | None = None):
    """Yield (file_idx, sample_idx, image) from LoDoPaB ground truth."""
    import h5py
    files = find_lodopab(root).get(part, [])
    count = 0
    for fi, f in enumerate(files):
        with h5py.File(f, "r") as h:
            data = h["data"]
            for si in range(0, data.shape[0], every):
                img = np.asarray(data[si], dtype=np.float64)
                yield fi, si, _to_256(np.clip(img, 0.0, None) / max(img.max(), 1e-9))
                count += 1
                if limit and count >= limit:
                    return
