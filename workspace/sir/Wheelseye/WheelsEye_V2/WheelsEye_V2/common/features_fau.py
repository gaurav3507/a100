"""Facial Action Unit features from UL-DD's pre-extracted FAU CSVs (plan §4
"Vision -- FAU": the single biggest untouched signal in v1 -- 30 AU intensity
channels shipped with the dataset and never opened by the old pipeline).

The UL-DD paper (Table 2 + "Facial Features") derives these 30 AUs from the
facial landmarks and explicitly flags the upper-face/yawning units as the
drowsiness-relevant ones; its own validation puts FAU alone at 52-59%
accuracy. Column names below are copied verbatim from the dataset CSV header
(verified against A_FAU_A.csv, 2026-08-18) and asserted against the loaded
file at extraction time -- a silent column mismatch would poison every
downstream model, so it fails loudly instead.

Features per window: mean + std of every AU intensity (the same
window-summary treatment every other modality gets), plus max intensity for
the six drowsiness-critical AUs (lid droop, slit, eyes closed, squint,
blink, jaw drop) -- max catches a single long eyelid closure that a mean
over 600 frames would wash out. Event-*rate* features (blinks/min from AU45
etc.) are deliberately deferred: they need an intensity threshold calibrated
against real data distributions, and picking one blind would be guesswork;
blink rate already exists from EAR in features_vision.py.
"""
from __future__ import annotations

import numpy as np

from common.windowing import Signal

# Verbatim from the FAU CSV header, in file order (30 AUs; the CSV's first
# column "Frame" is stripped by common/loaders.load_fau).
FAU_COLUMNS = (
    "inner_brow_raiser", "outer_brow_raiser", "brow_lowerer",
    "upper_lid_raiser", "cheek_raiser_feature", "lid_tightener",
    "nose_wrinkler", "upper_lip_raiser", "nasolabial_furrow_deepener",
    "lip_corner_puller", "cheek_puffer", "dimpler", "lip_corner_depressor",
    "lower_lip_depressor", "chin_raiser", "lip_puckerer", "lip_stretcher",
    "lip_funneler", "lip_tightener", "lip_pressor", "lips_part", "jaw_drop",
    "mouth_stretch", "lip_suck", "lid_droop", "slit", "eye_closed", "squint",
    "blink", "wink",
)

# The units the UL-DD paper singles out as drowsiness-relevant (upper-face +
# yawning): AU41 lid_droop, AU42 slit, AU43 eye_closed, AU44 squint,
# AU45 blink, AU26 jaw_drop.
KEY_AUS = ("lid_droop", "slit", "eye_closed", "squint", "blink", "jaw_drop")

FAU_FEATURES = (
    [f"fau_{c}_mean" for c in FAU_COLUMNS]
    + [f"fau_{c}_std" for c in FAU_COLUMNS]
    + [f"fau_{c}_max" for c in KEY_AUS]
)


def extract_fau_features(fau: Signal) -> dict[str, float]:
    """One dict of scalar features for a single window's FAU slice. An empty
    slice (face not detected for the whole window) yields NaN for every
    feature -- handled downstream as a missing value, same as vision."""
    if len(fau.times) > 0 and list(fau.columns) != list(FAU_COLUMNS):
        raise ValueError(
            "FAU columns don't match the expected UL-DD layout -- refusing to "
            f"extract by position. Got: {fau.columns}"
        )

    out: dict[str, float] = {}
    empty = len(fau.times) == 0
    for j, col in enumerate(FAU_COLUMNS):
        if empty:
            out[f"fau_{col}_mean"] = float("nan")
            out[f"fau_{col}_std"] = float("nan")
        else:
            v = fau.values[:, j]
            out[f"fau_{col}_mean"] = float(np.mean(v))
            out[f"fau_{col}_std"] = float(np.std(v))
    for col in KEY_AUS:
        if empty:
            out[f"fau_{col}_max"] = float("nan")
        else:
            v = fau.values[:, FAU_COLUMNS.index(col)]
            out[f"fau_{col}_max"] = float(np.max(v))
    return out
