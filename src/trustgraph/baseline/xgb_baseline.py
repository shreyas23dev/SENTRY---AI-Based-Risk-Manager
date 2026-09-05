"""
xgb_baseline.py — Unified Wrapper for XGBoost Fraud Detection Baseline
======================================================================

Provides a drop-in adapter maintaining the existing TRUSTGRAPH baseline interface:
  - predict_risk(X) -> A_t in [0.0, 1.0]
  - predict_proba(X) -> np.ndarray of shape (N, 2)
  - predict(X, threshold) -> binary predictions
  - score_dataframe(df) -> pd.DataFrame with [TransactionID, risk_score]
  - load(dir_path) / save(dir_path)

Encapsulates:
  1. ModelFeaturePipeline (feature engineering)
  2. XGBRiskModel (XGBoost classifier)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from trustgraph.baseline.model_features import ModelFeaturePipeline
from trustgraph.baseline.xgb_model import XGBRiskModel

logger = logging.getLogger(__name__)


class XGBBaselineWrapper:
    """
    Complete baseline engine encapsulating feature engineering and XGBoost inference.

    From Phase 1 onward, A_t is defined as:
        A_t = P(isFraud = 1 | Features_t)
    """

    def __init__(
        self,
        pipeline: ModelFeaturePipeline,
        model: XGBRiskModel,
        default_threshold: float = 0.12,
    ) -> None:
        self.pipeline = pipeline
        self.model = model
        self.default_threshold = default_threshold
        self.feature_cols: List[str] = list(pipeline.feature_cols)

    @classmethod
    def load(cls, artifact_dir: Union[str, Path]) -> "XGBBaselineWrapper":
        """
        Load the fitted pipeline, trained XGBoost model, and metadata.
        """
        artifact_dir = Path(artifact_dir)
        pipeline_path = artifact_dir / "feature_pipeline.pkl"
        model_path = artifact_dir / "xgb_model.pkl"
        metadata_path = artifact_dir / "metadata.json"

        if not pipeline_path.exists():
            raise FileNotFoundError(f"Feature pipeline not found at {pipeline_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"XGBoost model not found at {model_path}")

        pipe = ModelFeaturePipeline.load(pipeline_path)
        xgb = XGBRiskModel.load(model_path)

        th = 0.12
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    meta = json.load(f)
                    th = float(meta.get("validation_selected_threshold", 0.12))
            except Exception:
                pass

        logger.info("XGBBaselineWrapper loaded successfully (threshold=%.4f).", th)
        return cls(pipeline=pipe, model=xgb, default_threshold=th)

    def save(self, artifact_dir: Union[str, Path]) -> None:
        """Persist wrapper components to directory."""
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline.save(artifact_dir / "feature_pipeline.pkl")
        self.model.save(artifact_dir / "xgb_model.pkl")

    def predict_risk(self, X: Union[pd.DataFrame, Dict[str, Any], np.ndarray]) -> np.ndarray:
        """
        Produce A_t in [0.0, 1.0].

        Accepts:
          - raw transaction pd.DataFrame (runs full pipeline)
          - raw single transaction Dict (converts to DataFrame and runs pipeline)
          - pre-computed feature matrix pd.DataFrame or np.ndarray
        """
        if isinstance(X, dict):
            df_single = pd.DataFrame([X])
            X_feat = self.pipeline.transform(df_single)
            return self.model.predict_risk(X_feat)

        if isinstance(X, pd.DataFrame):
            # Check if input already has the expected feature columns
            if set(self.feature_cols).issubset(set(X.columns)):
                X_feat = X[self.feature_cols]
            else:
                # Raw transaction DataFrame
                X_feat = self.pipeline.transform(X)
            return self.model.predict_risk(X_feat)

        # Pre-computed numpy array
        return self.model.predict_risk(X)

    def predict_proba(self, X: Union[pd.DataFrame, Dict[str, Any], np.ndarray]) -> np.ndarray:
        """Standard sklearn predict_proba returning (N, 2) array."""
        risk = self.predict_risk(X)
        return np.column_stack([1.0 - risk, risk])

    def predict(
        self,
        X: Union[pd.DataFrame, Dict[str, Any], np.ndarray],
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Binary predictions using threshold (defaults to validation-selected threshold)."""
        th = self.default_threshold if threshold is None else threshold
        risk = self.predict_risk(X)
        return (risk >= th).astype(int)

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a DataFrame and return transaction_id, risk_score (A_t).
        """
        risk_scores = self.predict_risk(df)
        out = pd.DataFrame(index=df.index)
        if "TransactionID" in df.columns:
            out["transaction_id"] = df["TransactionID"]
        out["risk_score"] = risk_scores
        out["A_t"] = risk_scores
        return out


# Backward compatibility alias
KaggleBaselineWrapper = XGBBaselineWrapper
