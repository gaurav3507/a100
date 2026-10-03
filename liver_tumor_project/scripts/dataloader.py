"""
Universal Data Loader for LiTS
Used by all 10 baselines + novel method
"""

import os
import json
import numpy as np
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandAdjustContrastd,
    EnsureTyped,
)
from monai.data import CacheDataset, Dataset, DataLoader, list_data_collate
from monai.utils import set_determinism


class DataConfig:
    DATA_ROOT = "/workspace/liver_tumor_project/data/preprocessed"
    SPLIT_FILE = "/workspace/liver_tumor_project/data/splits/fold_splits.json"
    
    # Training patch size
    PATCH_SIZE = (128, 128, 128)
    
    # Tumor-focused sampling
    POS_NEG_RATIO = 2.0
    NUM_SAMPLES = 4
    
    # Batch sizes
    TRAIN_BATCH_SIZE = 2
    VAL_BATCH_SIZE = 1
    TEST_BATCH_SIZE = 1
    
    NUM_WORKERS = 8
    CACHE_RATE = 1.0
    SEED = 42


def build_data_dicts(cfg=DataConfig):
    with open(cfg.SPLIT_FILE, "r") as f:
        splits = json.load(f)
    
    def make_dicts(volume_names, subset):
        dicts = []
        for vol_name in volume_names:
            idx = vol_name.replace("volume-", "").replace(".nii", "")
            if subset == "test":
                img_path = f"{cfg.DATA_ROOT}/imagesTs/volume-{idx}.nii.gz"
                lbl_path = f"{cfg.DATA_ROOT}/labelsTs/segmentation-{idx}.nii.gz"
            else:
                img_path = f"{cfg.DATA_ROOT}/imagesTr/volume-{idx}.nii.gz"
                lbl_path = f"{cfg.DATA_ROOT}/labelsTr/segmentation-{idx}.nii.gz"
            
            if os.path.exists(img_path) and os.path.exists(lbl_path):
                dicts.append({
                    "image": img_path,
                    "label": lbl_path,
                    "name": vol_name,
                })
        return dicts
    
    return (
        make_dicts(splits["train"]["volumes"], "train"),
        make_dicts(splits["val"]["volumes"], "train"),
        make_dicts(splits["test"]["volumes"], "test"),
    )


def get_train_transforms(cfg=DataConfig):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=cfg.PATCH_SIZE,
            pos=cfg.POS_NEG_RATIO,
            neg=1.0,
            num_samples=cfg.NUM_SAMPLES,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5),
        RandScaleIntensityd(keys=["image"], factors=0.10, prob=0.5),
        RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.3),
        RandGaussianNoised(keys=["image"], mean=0.0, std=0.01, prob=0.15),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_val_transforms(cfg=DataConfig):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ])


def get_dataloaders(cfg=DataConfig, verbose=True):
    set_determinism(seed=cfg.SEED)
    train_dicts, val_dicts, test_dicts = build_data_dicts(cfg)
    
    if verbose:
        print(f"Dataset sizes: train={len(train_dicts)}, val={len(val_dicts)}, test={len(test_dicts)}")
    
    train_ds = CacheDataset(
        data=train_dicts,
        transform=get_train_transforms(cfg),
        cache_rate=cfg.CACHE_RATE,
        num_workers=cfg.NUM_WORKERS,
        progress=verbose,
    )
    val_ds = CacheDataset(
        data=val_dicts,
        transform=get_val_transforms(cfg),
        cache_rate=cfg.CACHE_RATE,
        num_workers=cfg.NUM_WORKERS,
        progress=verbose,
    )
    test_ds = Dataset(data=test_dicts, transform=get_val_transforms(cfg))
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.VAL_BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.TEST_BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
