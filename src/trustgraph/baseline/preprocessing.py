"""
preprocessing.py — TRUSTGRAPH Phase 1 Baseline Preprocessing
=============================================================

Responsibilities:
  - Encode categorical features: fit on TRAIN ONLY
  - Map unseen categories in val/test to NaN (safe, not crash)
  - Mark categorical columns as pandas Categorical for LightGBM
  - Save and load preprocessing artifacts

Strategy:
  - LightGBM handles NaN natively via optimal split search
  - No imputation of missing values (preserves signal)
  - Categorical features encoded as integer codes ≥ 0; NaN for unknown/missing
  - fit() uses train partition only; transform() is applied to all partitions

Leakage prevention:
  - fit() MUST be called only on TRAIN data
  - No statistics from val/test leak into the encoder
"""

import gc
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trustgraph.baseline import config as cfg

logger = logging.getLogger(__name__)


class CategoricalEncoder:
    """
    Ordinal encoder for categorical string features.

    Fitted on training data only. At transform time, unknown categories
    and NaN values are both mapped to NaN (float), which LightGBM
    handles natively.

    Note: LightGBM requires categorical feature codes to be non-negative
    integers. NaN is passed as-is and LightGBM uses its own NaN bin.
    """

    def __init__(self) -> None:
        # Maps column_name → {category_str: int_code}
        self.mappings: Dict[str, Dict[str, int]] = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame, categorical_cols: List[str]) -> "CategoricalEncoder":
        """
        Learn category → integer mappings from the training dataframe.
        Only called on TRAIN data.
        """
        self.mappings = {}
        for col in categorical_cols:
            if col not in df.columns:
                logger.warning("Column %s not found in dataframe during fit — skipping.", col)
                continue
            # Collect non-null unique values
            unique_vals = sorted(df[col].dropna().astype(str).unique())
            self.mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}
            logger.debug("  %s: %d categories", col, len(unique_vals))

        self.fitted = True
        logger.info("CategoricalEncoder fitted on %d columns.", len(self.mappings))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted mappings to a dataframe partition.

        Unknown categories → NaN (float).
        NaN values         → NaN (float, unchanged).
        Known categories   → non-negative integer code (float column).

        Returns a copy of df with categorical columns replaced by
        their integer-coded float versions.
        """
        if not self.fitted:
            raise RuntimeError("CategoricalEncoder must be fitted before transform().")

        df = df.copy()
        for col, mapping in self.mappings.items():
            if col not in df.columns:
                # Column absent in this partition (e.g. test set)
                continue
            # Convert to string (for consistent lookup), map, keep NaN for unknowns
            df[col] = df[col].astype(object).map(
                lambda v: mapping.get(str(v), np.nan) if pd.notna(v) else np.nan
            )
            # Ensure float so NaN is representable
            df[col] = df[col].astype("float32")

        return df

    def fit_transform(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
    ) -> pd.DataFrame:
        self.fit(df, categorical_cols)
        return self.transform(df)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("CategoricalEncoder saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "CategoricalEncoder":
        path = Path(path)
        with open(path, "rb") as f:
            encoder = pickle.load(f)
        logger.info("CategoricalEncoder loaded ← %s", path)
        return encoder


# ---------------------------------------------------------------------------
# Column-level helpers
# ---------------------------------------------------------------------------

def get_final_feature_list(df: pd.DataFrame) -> List[str]:
    """
    Intersect config.ALL_FEATURES with columns actually present in df.
    Ensures the feature list is deterministic and safe for both train and test.

    Returns the ordered list of feature column names.
    """
    present = [c for c in cfg.ALL_FEATURES if c in df.columns]
    absent  = [c for c in cfg.ALL_FEATURES if c not in df.columns]
    if absent:
        logger.warning("The following configured features are absent from df: %s", absent)
    # Safety checks
    assert "isFraud"       not in present, "LEAKAGE: isFraud in feature list!"
    assert "TransactionID" not in present, "IDENTIFIER LEAK: TransactionID in feature list!"
    logger.info("Final feature list: %d features (%d absent from df)", len(present), len(absent))
    return present


def mark_categorical_columns(
    df: pd.DataFrame,
    categorical_cols: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Mark encoded integer columns as pandas Categorical dtype.
    LightGBM will use native categorical treatment when dtype is 'category'.

    Returns df with updated dtypes, and the list of actual categorical columns
    present in df.
    """
    active_cats = []
    for col in categorical_cols:
        if col not in df.columns:
            continue
        if df[col].dtype in ("float32", "float64", "object"):
            # Convert to nullable integer first, then category
            # (to preserve NaN as NaN in the category dtype)
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            try:
                df[col] = df[col].astype("float32")
                active_cats.append(col)
            except (ValueError, TypeError) as exc:
                logger.warning("Could not mark %s as categorical: %s", col, exc)

    logger.info("Marked %d columns as float32 (categorical for LightGBM).", len(active_cats))
    return df, active_cats


# ---------------------------------------------------------------------------
# Public pipeline API
# ---------------------------------------------------------------------------

class BaselinePreprocessor:
    """
    End-to-end preprocessing pipeline for the Phase-1 baseline.

    Workflow:
        preprocessor = BaselinePreprocessor()
        X_train = preprocessor.fit_transform(train_df)
        X_val   = preprocessor.transform(val_df)
        X_test  = preprocessor.transform(test_df)

    Leakage guarantee:
        fit_transform() MUST be called on TRAIN data only.
        transform() is safe to call on any partition.
    """

    def __init__(self) -> None:
        self.encoder       = CategoricalEncoder()
        self.feature_cols: List[str] = []
        self.cat_cols:     List[str] = []
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit on training data and return transformed feature matrix.

        Parameters
        ----------
        df : pd.DataFrame — TRAIN partition (must contain isFraud)

        Returns
        -------
        X : pd.DataFrame — feature matrix (no isFraud, no TransactionID)
        """
        logger.info("BaselinePreprocessor.fit_transform() called on %d rows.", len(df))

        # Determine feature list from config ∩ df.columns
        self.feature_cols = get_final_feature_list(df)

        # Identify which categorical features are actually present
        self.cat_cols = [c for c in cfg.ALL_CATEGORICAL_FEATURES if c in self.feature_cols]
        logger.info("Categorical features: %d", len(self.cat_cols))

        # Select only feature columns
        X = df[self.feature_cols].copy()

        # Fit encoder on train categorical columns (ONLY called here)
        X = self.encoder.fit_transform(X, self.cat_cols)

        # Mark categoricals as float32 (LightGBM will detect by name from categorical_feature)
        X, self.cat_cols = mark_categorical_columns(X, self.cat_cols)

        # Safety: ensure no object columns remain
        for col in X.select_dtypes(include=["object"]).columns:
            logger.warning("Converting unexpected object column %s to float32", col)
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float32")

        self.fitted = True
        logger.info("fit_transform complete. X shape: %s", X.shape)
        return X

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted preprocessing to a val/test partition.

        Parameters
        ----------
        df : pd.DataFrame — val or test partition

        Returns
        -------
        X : pd.DataFrame — feature matrix aligned to training feature columns
        """
        if not self.fitted:
            raise RuntimeError("BaselinePreprocessor must be fit before transform().")

        logger.info("BaselinePreprocessor.transform() called on %d rows.", len(df))

        # Select feature columns that exist in this df
        available = [c for c in self.feature_cols if c in df.columns]
        X = df[available].copy()

        # Apply encoder (unseen categories → NaN; safe)
        X = self.encoder.transform(X)

        # Re-order to match training column order
        X = X.reindex(columns=self.feature_cols)

        # Safety: ensure all columns are float32/int32, not object
        for col in X.select_dtypes(include=["object"]).columns:
            logger.warning("Converting unexpected object column %s in transform to float32", col)
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float32")

        logger.info("transform complete. X shape: %s", X.shape)
        return X

    def save(self, directory: Path) -> None:
        """Save all preprocessing artifacts to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save encoder
        self.encoder.save(directory / "label_encoders.pkl")

        # Save feature list and cat list
        meta = {
            "feature_cols": self.feature_cols,
            "cat_cols":     self.cat_cols,
        }
        with open(directory / "preprocessor_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("BaselinePreprocessor saved → %s", directory)

    @classmethod
    def load(cls, directory: Path) -> "BaselinePreprocessor":
        """Load preprocessing artifacts from directory."""
        directory = Path(directory)
        preprocessor = cls()
        preprocessor.encoder = CategoricalEncoder.load(directory / "label_encoders.pkl")
        with open(directory / "preprocessor_meta.json") as f:
            meta = json.load(f)
        preprocessor.feature_cols = meta["feature_cols"]
        preprocessor.cat_cols     = meta["cat_cols"]
        preprocessor.fitted = True
        logger.info("BaselinePreprocessor loaded ← %s", directory)
        return preprocessor
