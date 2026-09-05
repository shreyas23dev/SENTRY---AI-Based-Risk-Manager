"""
model.py — TRUSTGRAPH Phase 1 Baseline Model
=============================================

Wraps LightGBM binary classifier with a clean interface:

    fit(X_train, y_train, X_val, y_val)
        Train with early stopping on validation AUC.

    predict_risk(X) → A_t ∈ [0, 1]
        Continuous fraud-risk probability for each transaction.
        This is the primary output consumed by future TRUSTGRAPH components.

    predict_label(X, threshold) → binary np.ndarray
        Binary prediction using the frozen validation threshold.

    save(path) / load(path)
        Persist the trained model.

Class imbalance:
    scale_pos_weight = negative_count / positive_count ≈ 27.58
    This is set in LGBM_PARAMS and adjusts the loss to weight
    fraud cases more heavily. No oversampling or undersampling is used
    on validation/test data.

Phase-1 only:
    This model is purely point-wise. It receives the feature vector
    of a single transaction and outputs A_t with no memory of prior
    transactions. Future TRUSTGRAPH phases will augment this output.
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from trustgraph.baseline import config as cfg

logger = logging.getLogger(__name__)


class BaselineModel:
    """
    LightGBM binary classifier for IEEE-CIS fraud detection.

    Produces:
        A_t = P(isFraud = 1 | X_t)   — continuous, in [0, 1]
    """

    def __init__(self, params: Optional[Dict] = None) -> None:
        self.params = dict(cfg.LGBM_PARAMS) if params is None else dict(params)
        self.model: Optional[lgb.LGBMClassifier] = None
        self.best_iteration: int = 0
        self.cat_cols: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cat_cols: Optional[List[str]] = None,
    ) -> "BaselineModel":
        """
        Train the LightGBM classifier.

        Early stopping is applied on validation AUC.
        scale_pos_weight handles class imbalance.

        Parameters
        ----------
        X_train, y_train : training features and labels
        X_val, y_val     : validation features and labels (for early stopping)
        cat_cols         : list of column names to treat as categorical
        """
        self.cat_cols = cat_cols or []

        # Compute actual scale_pos_weight from training labels
        n_negative = int((y_train == 0).sum())
        n_positive = int((y_train == 1).sum())
        actual_spw = n_negative / max(n_positive, 1)
        self.params["scale_pos_weight"] = actual_spw

        logger.info(
            "Training LightGBM: %d train rows (%d fraud / %d legit), SPW=%.2f",
            len(y_train), n_positive, n_negative, actual_spw
        )
        logger.info(
            "Validation: %d rows (%d fraud / %d legit)",
            len(y_val), int(y_val.sum()), int((y_val == 0).sum())
        )
        logger.info("LightGBM params: %s", self.params)

        # Build the classifier
        n_estimators = self.params.pop("n_estimators", 3000)
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            **self.params,
        )
        self.params["n_estimators"] = n_estimators  # restore

        # Identify categorical feature indices
        cat_feature_arg = "auto"
        if self.cat_cols:
            # Only use cat cols that are actually in X_train
            active_cats = [c for c in self.cat_cols if c in X_train.columns]
            cat_feature_arg = active_cats if active_cats else "auto"
            logger.info("Categorical features for LightGBM: %d columns", len(active_cats))

        t0 = time.perf_counter()
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=cfg.EARLY_STOPPING_ROUNDS,
                    first_metric_only=True,
                    verbose=True,
                ),
                lgb.log_evaluation(period=50),
            ],
            categorical_feature=cat_feature_arg,
        )
        elapsed = time.perf_counter() - t0
        self.best_iteration = self.model.best_iteration_
        logger.info(
            "Training complete in %.1fs. Best iteration: %d",
            elapsed, self.best_iteration
        )
        return self

    def predict_risk(self, X) -> np.ndarray:
        """
        Produce continuous fraud-risk scores A_t ∈ [0, 1].

        This is the primary output of the baseline model.
        Future TRUSTGRAPH components consume these scores.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray — preprocessed feature matrix.
            When a numpy array is passed (fast single-row path), column ordering
            must match the training feature order exactly.

        Returns
        -------
        A_t : np.ndarray, shape (n_samples,), dtype float64
              Each value is P(isFraud=1 | X_t).
        """
        import warnings
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        # Suppress sklearn's feature-name warning when a numpy array is passed.
        # The array column order is guaranteed by FastPreprocessor to match
        # the training feature order exactly — the warning is purely cosmetic.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            proba = self.model.predict_proba(X, num_iteration=self.best_iteration)
        A_t = proba[:, 1]
        # Verify output contract
        assert A_t.ndim == 1, "A_t must be 1-D"
        assert len(A_t) == len(X), "A_t length must match input rows"
        assert np.all(A_t >= 0.0) and np.all(A_t <= 1.0), "A_t values outside [0, 1]!"
        return A_t

    def predict_label(self, X: pd.DataFrame, threshold: float) -> np.ndarray:
        """
        Apply the frozen threshold to convert A_t to binary predictions.

        Parameters
        ----------
        X         : preprocessed feature matrix
        threshold : decision threshold (selected on validation, frozen)

        Returns
        -------
        labels : np.ndarray, shape (n_samples,), dtype int, values ∈ {0, 1}
        """
        A_t = self.predict_risk(X)
        return (A_t >= threshold).astype(int)

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances as a sorted DataFrame."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        names = self.model.feature_name_
        gains = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": names, "importance": gains})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: Path) -> None:
        """Persist the model using pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BaselineModel saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "BaselineModel":
        """Load a previously saved model."""
        path = Path(path)
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("BaselineModel loaded ← %s", path)
        return model

    def get_config(self) -> Dict:
        """Return a JSON-serialisable model configuration summary."""
        return {
            "model_type":      "LightGBM binary classifier",
            "best_iteration":  self.best_iteration,
            "params":          {
                k: (float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in self.params.items()
            },
            "cat_cols":        self.cat_cols,
            "early_stopping_rounds": cfg.EARLY_STOPPING_ROUNDS,
        }
