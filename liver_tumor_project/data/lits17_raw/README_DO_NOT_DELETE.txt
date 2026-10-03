
================================================================================
⚠️  DO NOT DELETE THIS FOLDER ⚠️
================================================================================

Raw LiTS17 dataset with HU values.

Downloaded: 2026-08-14 15:08:07
Size: ~50 GB
Source: kaggle.com/datasets/javariatahir/litstrain-val

STATUS: READ-ONLY

Contents:
- LiTS(train_test)/train_CT/    (111 CT volumes, raw HU)
- LiTS(train_test)/train_mask/  (111 segmentation masks)
- LiTS(train_test)/test_CT/     (20 CT volumes, raw HU)
- LiTS(train_test)/test_mask/   (20 segmentation masks)

WHY IT MATTERS:
- Used for vessel extraction (novel VasTA-Net method)
- Source of truth for all preprocessing
- Cannot easily re-download (Kaggle CLI single-file bug)
- Paper reproducibility requires original HU values

TO MODIFY (only if absolutely needed):
   chmod -R u+w /workspace/liver_tumor_project/data/lits17_raw
================================================================================
