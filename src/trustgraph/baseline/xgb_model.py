"""
xgb_model.py — XGBoost Fraud Risk Classifier
============================================

Drop-in replacement for BaselineModel implementing the XGBoost configuration
for IEEE-CIS Fraud Detection:
  - n_estimators = 2000 (controlled by early stopping rounds = 100)
  - max_depth = 12
  - learning_rate = 0.02
  - subsample = 0.8
  - colsample_bytree = 0.4
  - missing = -1.0
  - eval_metric = "auc"
  - tree_method = "hist"

Provides:
  - fit(X_train, y_train, X_val, y_val): Trains with early stopping on validation AUC.
  - predict_risk(X): Produces continuous fraud risk probability A_t in [0.0, 1.0].
  - predict_proba(X): Standard sklearn (N, 2) array.
  - predict_label(X, threshold): Binary prediction.
  - save(path) / load(path): Persistent serialization.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)

DEFAULT_XGB_PARAMS = {
    "n_estimators": 2000,
    "max_depth": 12,
    "learning_rate": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.4,
    "missing": -1.0,
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}
DEFAULT_KAGGLE_XGB_PARAMS = DEFAULT_XGB_PARAMS


class XGBRiskModel:
    """
    XGBoost binary classifier for transaction fraud detection.

    Produces:
        A_t = P(isFraud = 1 | X_t) — continuous, in [0.0, 1.0].
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        early_stopping_rounds: int = 100,
    ) -> None:
        self.params = dict(DEFAULT_XGB_PARAMS)
        if params is not None:
            self.params.update(params)
        self.early_stopping_rounds = early_stopping_rounds
        self.model: Optional[xgb.XGBClassifier] = None
        self.best_iteration: int = 0
        self.feature_names: List[str] = []

    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        verbose: int = 50,
    ) -> "XGBRiskModel":
        """
        Train the XGBoost classifier with early stopping on validation AUC.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
        elif hasattr(X_train, "shape"):
            self.feature_names = [f"f{i}" for i in range(X_train.shape[1])]

        logger.info(
            "Training XGBoost: %d train rows (%d features), early_stopping=%d",
            len(y_train), len(self.feature_names), self.early_stopping_rounds
        )
        logger.info("Validation: %d rows", len(y_val))
        logger.info("XGBoost params: %s", self.params)

        self.model = xgb.XGBClassifier(
            early_stopping_rounds=self.early_stopping_rounds,
            **self.params,
        )

        start_t = time.perf_counter()
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=verbose,
        )
        duration = time.perf_counter() - start_t

        self.best_iteration = int(getattr(self.model, "best_iteration", self.params["n_estimators"]))
        logger.info(
            "XGBoost training finished in %.2f s. Best iteration: %d",
            duration, self.best_iteration
        )
        return self

    def predict_risk(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Primary continuous risk prediction: A_t in [0.0, 1.0].
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted or loaded.")
        probs = self.model.predict_proba(X)
        return np.asarray(probs[:, 1], dtype=np.float32)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Standard sklearn predict_proba returning (N, 2) float array."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted or loaded.")
        return self.model.predict_proba(X)

    def predict(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """Binary predictions using threshold."""
        return self.predict_label(X, threshold=threshold)

    def predict_label(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        """Binary classification at decision boundary."""
        risk = self.predict_risk(X)
        return (risk >= threshold).astype(int)

    def save(self, path: Union[str, Path]) -> None:
        """Persist model and metadata to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "params": self.params,
            "best_iteration": self.best_iteration,
            "early_stopping_rounds": self.early_stopping_rounds,
            "feature_names": self.feature_names,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved XGBRiskModel -> %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "XGBRiskModel":
        """Load trained model and metadata from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            payload = pickle.load(f)

        instance = cls(
            params=payload["params"],
            early_stopping_rounds=payload.get("early_stopping_rounds", 100),
        )
        instance.model = payload["model"]
        instance.best_iteration = payload.get("best_iteration", 0)
        instance.feature_names = payload.get("feature_names", [])
        logger.info("Loaded XGBRiskModel <- %s (%d features)", path, len(instance.feature_names))
        return instance


# Backward compatibility alias
KaggleXGBModel = XGBRiskModel
