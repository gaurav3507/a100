"""Stacked late fusion -- Track B's v2 deployment candidate (plan §7):
one small LightGBM expert per modality group, a logistic-regression
meta-learner over the concatenated expert probabilities, and modality
dropout during meta training so the stack stays calibrated when a sensor
drops out at runtime (a real product scenario: wristband off, face not
detected -- and a selling point the panel narrative uses).

Why this shape for deployment: LightGBM experts are scale-invariant,
NaN-tolerant at inference, CPU-cheap on the Jetson; the meta-learner is a
single matrix multiply. No torch at inference time at all.

Leakage discipline: the meta-learner is trained on OUT-OF-FOLD expert
probabilities (inner CV inside the training set) -- training it on the
experts' fitted-on-same-rows probabilities would teach it to trust
overconfident experts (a mini version of v1 defect #1). Modality dropout is
implemented as augmentation: each OOF row is replicated with one modality's
probability block replaced by the uniform distribution -- exactly what
predict-time masking produces for a missing sensor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class StackedFusionClassifier:
    """sklearn-style fit/predict over a feature DataFrame.

    groups: {modality_name: [feature columns]} -- e.g. dataset.V2_FEATURE_GROUPS
    (optionally with a "context" group appended for the tabular stack).
    """

    def __init__(
        self,
        groups: dict[str, list[str]],
        num_classes: int = 3,
        n_inner_folds: int = 3,
        lgbm_params: dict | None = None,
        seed: int = 0,
    ):
        self.groups = {k: list(v) for k, v in groups.items()}
        self.num_classes = num_classes
        self.n_inner_folds = n_inner_folds
        self.seed = seed
        self.lgbm_params = lgbm_params or dict(
            objective="multiclass", num_class=num_classes, num_leaves=31,
            n_estimators=200, learning_rate=0.05, class_weight="balanced",
        )
        self.experts_: dict[str, object] = {}
        self.meta_ = None
        self._medians: dict[str, pd.Series] = {}

    def _expert(self):
        import lightgbm as lgb

        return lgb.LGBMClassifier(**self.lgbm_params, random_state=self.seed, verbosity=-1)

    def _group_matrix(self, df: pd.DataFrame, name: str, fit: bool = False) -> np.ndarray:
        cols = self.groups[name]
        x = df[cols]
        if fit:
            self._medians[name] = x.median()
        return x.fillna(self._medians[name]).to_numpy()

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "StackedFusionClassifier":
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold

        y = np.asarray(y)
        uniform = np.full(self.num_classes, 1.0 / self.num_classes)

        # 1) out-of-fold expert probabilities for the meta-learner
        oof = {name: np.full((len(df), self.num_classes), np.nan) for name in self.groups}
        skf = StratifiedKFold(n_splits=self.n_inner_folds, shuffle=True, random_state=self.seed)
        for tr_idx, va_idx in skf.split(np.zeros(len(df)), y):
            for name in self.groups:
                x = self._group_matrix(df.iloc[tr_idx], name, fit=True)
                expert = self._expert()
                expert.fit(x, y[tr_idx])
                oof[name][va_idx] = expert.predict_proba(
                    self._group_matrix(df.iloc[va_idx], name)
                )

        # 2) meta training set: clean rows + one masked copy per modality
        blocks = [oof[name] for name in self.groups]
        meta_x = [np.concatenate(blocks, axis=1)]
        meta_y = [y]
        for j, name in enumerate(self.groups):
            masked = [b.copy() for b in blocks]
            masked[j] = np.tile(uniform, (len(df), 1))
            meta_x.append(np.concatenate(masked, axis=1))
            meta_y.append(y)
        self.meta_ = LogisticRegression(max_iter=1000, C=1.0)
        self.meta_.fit(np.vstack(meta_x), np.concatenate(meta_y))

        # 3) refit each expert on the full training set for inference
        for name in self.groups:
            x = self._group_matrix(df, name, fit=True)
            expert = self._expert()
            expert.fit(x, y)
            self.experts_[name] = expert
        return self

    def _expert_probs(self, df: pd.DataFrame, missing: set[str] | None = None) -> np.ndarray:
        uniform = np.full(self.num_classes, 1.0 / self.num_classes)
        blocks = []
        for name in self.groups:
            if missing and name in missing:
                blocks.append(np.tile(uniform, (len(df), 1)))
            else:
                blocks.append(self.experts_[name].predict_proba(self._group_matrix(df, name)))
        return np.concatenate(blocks, axis=1)

    def predict_proba(self, df: pd.DataFrame, missing: set[str] | None = None) -> np.ndarray:
        """`missing` names modality groups to mask with the uniform prior --
        the sensor-dropout robustness evaluation path."""
        return self.meta_.predict_proba(self._expert_probs(df, missing))

    def predict(self, df: pd.DataFrame, missing: set[str] | None = None) -> np.ndarray:
        return np.argmax(self.predict_proba(df, missing), axis=1)
