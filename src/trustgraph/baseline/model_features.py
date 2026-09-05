"""
model_features.py — IEEE-CIS Fraud Detection Feature Pipeline
=============================================================

Feature engineering pipeline:
  - 120 redundant V-columns filtered out by correlation analysis
  - D-columns normalization: D_i - TransactionDT / 86400 for i in {1..15} \\ {1,2,3,5,9}
  - Categorical factorizations with sorted categories
  - Numeric positive shift and -1 missing value imputation
  - Transaction amount cents feature: TransactionAmt - floor(TransactionAmt)
  - Frequency encodings (FE) on key categorical and interaction identifiers
  - Combined interaction columns (card1+addr1, card1+addr1+P_emaildomain)
  - UID reconstruction: card1_addr1 + '_' + floor(day - D1)
  - Group aggregations on UID (mean, std, nunique)
  - Outlier flag: outsider15 = (|D1 - D15| > 3)
  - Time-consistency feature selection (dropping inconsistent features)

Supports:
  - fit(df_train): Causal training-partition-only fitting (zero test leakage).
  - transform(df): Vectorized transform applying stored mappings and statistics.
  - fit_transform(df_train): Convenient single-pass training transform.
  - save(path) / load(path): Persistent serialization.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constant Definitions
# ---------------------------------------------------------------------------

STR_COLUMNS = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_30",
    "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
    "DeviceType", "DeviceInfo",
]
KAGGLE_STR_COLUMNS = STR_COLUMNS

# 120 V-columns kept after correlation analysis (219 dropped as redundant)
V_COLUMNS = [
    1, 3, 4, 6, 8, 11,
    13, 14, 17, 20, 23, 26, 27, 30,
    36, 37, 40, 41, 44, 47, 48,
    54, 56, 59, 62, 65, 67, 68, 70,
    76, 78, 80, 82, 86, 88, 89, 91,
    107, 108, 111, 115, 117, 120, 121, 123,
    124, 127, 129, 130, 136,
    138, 139, 142, 147, 156, 162,
    165, 160, 166,
    178, 176, 173, 182,
    187, 203, 205, 207, 215,
    169, 171, 175, 180, 185, 188, 198, 210, 209,
    218, 223, 224, 226, 228, 229, 235,
    240, 258, 257, 253, 252, 260, 261,
    264, 266, 267, 274, 277,
    220, 221, 234, 238, 250, 271,
    294, 284, 285, 286, 291, 297,
    303, 305, 307, 309, 310, 320,
    281, 283, 289, 296, 301, 314,
]
KAGGLE_V_COLUMNS = V_COLUMNS

# Baseline 53 transaction columns
BASE_COLS = [
    "TransactionID", "TransactionDT", "TransactionAmt",
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11",
    "C12", "C13", "C14", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",
    "D9", "D10", "D11", "D12", "D13", "D14", "D15", "M1", "M2", "M3", "M4",
    "M5", "M6", "M7", "M8", "M9",
]
KAGGLE_BASE_COLS = BASE_COLS

# Features removed during feature selection (time inconsistency / high-NaN / temporary)
DROP_FEATURES = [
    "TransactionDT",
    "D6", "D7", "D8", "D9", "D12", "D13", "D14",
    "oof", "DT_M", "day", "uid",
    # Failed time consistency test:
    "C3", "M5", "id_08", "id_33",
    "card4", "id_07", "id_14", "id_21", "id_30", "id_32", "id_34",
    "id_22", "id_23", "id_24", "id_25", "id_26", "id_27",
]
KAGGLE_DROP_FEATURES = DROP_FEATURES


class ModelFeaturePipeline:
    """
    Stateful feature engineering pipeline for XGBoost risk model.
    """

    def __init__(self, offline_only: bool = False) -> None:
        self.offline_only = offline_only
        self.cat_mappings: Dict[str, Dict[str, int]] = {}
        self.numeric_mins: Dict[str, float] = {}
        self.freq_encodings: Dict[str, Dict[Any, float]] = {}
        self.ag_mean_std_mappings: Dict[str, Dict[str, float]] = {}
        self.ag_nunique_mappings: Dict[str, Dict[str, float]] = {}
        self.feature_cols: List[str] = []
        self.is_fitted: bool = False

    # ---------------------------------------------------------------------------
    # Step 1: Input preparation and D-column normalization
    # ---------------------------------------------------------------------------

    def _prepare_base_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to required base columns, normalize identity column names, and normalize D."""
        out_dict: Dict[str, Any] = {}
        df_cols_set = set(df.columns)

        # Normalize dash to underscore in identity columns if present
        rename_map = {c: c.replace("-", "_") for c in df_cols_set if "-" in c}
        if rename_map:
            df = df.rename(columns=rename_map)
            df_cols_set = set(df.columns)

        # Copy base transaction columns (default to NaN if omitted in sparse inference)
        for c in BASE_COLS:
            if c in df_cols_set:
                out_dict[c] = df[c]
            else:
                out_dict[c] = np.nan

        # Copy selected V columns
        for v_num in V_COLUMNS:
            col_name = f"V{v_num}"
            if col_name in df_cols_set:
                out_dict[col_name] = df[col_name].astype(np.float32)
            else:
                out_dict[col_name] = np.nan

        # Copy identity columns id_01 to id_38
        for i in range(1, 39):
            col_name = f"id_{i:02d}"
            if col_name in df_cols_set:
                out_dict[col_name] = df[col_name]
            else:
                out_dict[col_name] = np.nan

        # Copy DeviceType and DeviceInfo
        for c in ["DeviceType", "DeviceInfo"]:
            if c in df_cols_set:
                out_dict[c] = df[c]
            else:
                out_dict[c] = np.nan

        # Target if present
        if "isFraud" in df_cols_set:
            out_dict["isFraud"] = df["isFraud"]

        out = pd.DataFrame(out_dict, index=df.index)

        # Normalize D columns: D_i = D_i - TransactionDT / 86400 (skip 1, 2, 3, 5, 9)
        dt_val = out["TransactionDT"].fillna(0.0)
        dt_day = (dt_val / np.float32(86400.0)).astype(np.float32)
        for i in range(1, 16):
            if i in [1, 2, 3, 5, 9]:
                continue
            col_name = f"D{i}"
            if col_name in out.columns:
                out[col_name] = (out[col_name] - dt_day).astype(np.float32)

        return out

    # ---------------------------------------------------------------------------
    # Step 2: Categorical factorizations & numeric shift
    # ---------------------------------------------------------------------------

    def _apply_categoricals_and_numerics(
        self, df: pd.DataFrame, is_train: bool
    ) -> pd.DataFrame:
        for f in list(df.columns):
            if f in ["isFraud", "TransactionDT", "TransactionID"]:
                continue

            # Categorical factorize with sort=True
            if f in STR_COLUMNS or df[f].dtype == "object" or str(df[f].dtype) == "category":
                if is_train:
                    series_str = df[f].dropna().astype(str)
                    uniques = sorted(series_str.unique().tolist())
                    mapping = {val: idx for idx, val in enumerate(uniques)}
                    self.cat_mappings[f] = mapping

                mapping = self.cat_mappings.get(f, {})
                mapped = df[f].astype(str).map(mapping)
                df[f] = mapped.fillna(-1).astype(np.int16)

            elif f != "TransactionAmt":
                # Numeric: shift positive, set NaN to -1
                if is_train:
                    mn = float(df[f].min(skipna=True)) if not df[f].isna().all() else 0.0
                    self.numeric_mins[f] = mn

                mn = self.numeric_mins.get(f, 0.0)
                df[f] = (df[f] - np.float32(mn)).fillna(-1.0).astype(np.float32)

        return df

    # ---------------------------------------------------------------------------
    # Step 3: Base feature engineering (cents, combinations, UID)
    # ---------------------------------------------------------------------------

    def _engineer_base_and_uid_features(
        self, df: pd.DataFrame, is_train: bool
    ) -> pd.DataFrame:
        new_cols: Dict[str, Any] = {}

        # cents
        new_cols["cents"] = (df["TransactionAmt"] - np.floor(df["TransactionAmt"])).astype(np.float32)

        # Combined interaction columns
        card1_addr1_series = df["card1"].astype(str) + "_" + df["addr1"].astype(str)
        card1_addr1_pem_series = card1_addr1_series + "_" + df["P_emaildomain"].astype(str)

        # Factorize combined interactions
        if is_train:
            u_ca = sorted(card1_addr1_series.dropna().unique().tolist())
            self.cat_mappings["card1_addr1"] = {val: idx for idx, val in enumerate(u_ca)}
            u_cap = sorted(card1_addr1_pem_series.dropna().unique().tolist())
            self.cat_mappings["card1_addr1_P_emaildomain"] = {val: idx for idx, val in enumerate(u_cap)}

        new_cols["card1_addr1"] = card1_addr1_series.map(self.cat_mappings.get("card1_addr1", {})).fillna(-1).astype(np.int32)
        new_cols["card1_addr1_P_emaildomain"] = card1_addr1_pem_series.map(self.cat_mappings.get("card1_addr1_P_emaildomain", {})).fillna(-1).astype(np.int32)

        # DT_M (month index)
        dt_series = pd.to_datetime(df["TransactionDT"], unit="s", origin="2017-11-30")
        new_cols["DT_M"] = (dt_series.dt.year - 2017) * 12 + dt_series.dt.month

        # day and magic UID
        new_cols["day"] = (df["TransactionDT"] / np.float32(86400.0)).astype(np.float32)
        new_cols["uid"] = card1_addr1_series + "_" + np.floor(new_cols["day"] - df["D1"]).astype(str)

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    # ---------------------------------------------------------------------------
    # Step 4: Frequency encodings (encode_FE)
    # ---------------------------------------------------------------------------

    def _apply_frequency_encodings(
        self, df: pd.DataFrame, is_train: bool
    ) -> pd.DataFrame:
        fe_cols = [
            "addr1", "card1", "card2", "card3", "P_emaildomain",
            "card1_addr1", "card1_addr1_P_emaildomain", "uid"
        ]
        new_cols: Dict[str, Any] = {}
        for col in fe_cols:
            nm = col + "_FE"
            if is_train:
                vc = df[col].value_counts(dropna=True, normalize=True).to_dict()
                vc[-1] = -1.0
                self.freq_encodings[col] = vc

            vc_map = self.freq_encodings.get(col, {})
            new_cols[nm] = df[col].map(vc_map).fillna(-1.0).astype(np.float32)

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    # ---------------------------------------------------------------------------
    # Step 5: Group aggregations (encode_AG & encode_AG2)
    # ---------------------------------------------------------------------------

    def _apply_group_aggregations(
        self, df: pd.DataFrame, is_train: bool
    ) -> pd.DataFrame:
        new_cols: Dict[str, Any] = {}

        # Group aggregations mean and std
        ag_specs = [
            # 1. Base card / interaction aggregations
            (["TransactionAmt", "D9", "D11"], ["card1", "card1_addr1", "card1_addr1_P_emaildomain"], ["mean", "std"]),
            # 2. Magic UID aggregations
            (["TransactionAmt", "D4", "D9", "D10", "D15"], ["uid"], ["mean", "std"]),
            ([f"C{x}" for x in range(1, 15) if x != 3], ["uid"], ["mean"]),
            ([f"M{x}" for x in range(1, 10)], ["uid"], ["mean"]),
            (["C14"], ["uid"], ["std"]),
        ]

        for main_cols, uids, aggs in ag_specs:
            for main_col in main_cols:
                if main_col not in df.columns:
                    continue
                for col in uids:
                    if col not in df.columns:
                        continue
                    for agg_type in aggs:
                        new_col_name = f"{main_col}_{col}_{agg_type}"
                        if is_train:
                            sub_df = df[[col, main_col]].copy()
                            sub_df.loc[sub_df[main_col] == -1, main_col] = np.nan
                            grouped = sub_df.groupby(col)[main_col].agg(agg_type)
                            mapping = grouped.to_dict()
                            self.ag_mean_std_mappings[new_col_name] = mapping

                        mapping = self.ag_mean_std_mappings.get(new_col_name, {})
                        new_cols[new_col_name] = df[col].map(mapping).fillna(-1.0).astype(np.float32)

        # Group aggregations nunique (encode_AG2)
        nunique_specs = [
            (["P_emaildomain", "dist1", "DT_M", "id_02", "cents"], ["uid"]),
            (["C13", "V314"], ["uid"]),
            (["V127", "V136", "V309", "V307", "V320"], ["uid"]),
        ]

        for main_cols, uids in nunique_specs:
            for main_col in main_cols:
                if main_col not in df.columns:
                    continue
                for col in uids:
                    if col not in df.columns:
                        continue
                    new_col_name = f"{col}_{main_col}_ct"
                    if is_train:
                        grouped = df.groupby(col)[main_col].nunique()
                        mapping = grouped.to_dict()
                        self.ag_nunique_mappings[new_col_name] = mapping

                    mapping = self.ag_nunique_mappings.get(new_col_name, {})
                    new_cols[new_col_name] = df[col].map(mapping).fillna(-1.0).astype(np.float32)

        # outsider15
        if "D1" in df.columns and "D15" in df.columns:
            new_cols["outsider15"] = (np.abs(df["D1"] - df["D15"]) > 3.0).astype(np.int8)

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    # ---------------------------------------------------------------------------
    # Step 6: Feature selection and cleanup
    # ---------------------------------------------------------------------------

    def _select_and_filter_features(
        self, df: pd.DataFrame, is_train: bool
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        y: Optional[pd.Series] = None
        if "isFraud" in df.columns:
            y = df["isFraud"].copy()

        cols_to_drop = set(DROP_FEATURES)
        cols_to_drop.update(["isFraud", "TransactionID"])

        remaining_cols = [c for c in df.columns if c not in cols_to_drop]

        if is_train:
            self.feature_cols = sorted(remaining_cols)
            logger.info("ModelFeaturePipeline fitted with %d features.", len(self.feature_cols))

        # Reindex to ensure identical column ordering
        X = df.reindex(columns=self.feature_cols, fill_value=-1.0)
        return X, y

    # ---------------------------------------------------------------------------
    # Public Interface
    # ---------------------------------------------------------------------------

    def fit(self, df_train: pd.DataFrame) -> "ModelFeaturePipeline":
        """
        Fit all mappings, encoders, and group statistics on the training partition.
        Zero test leakage.
        """
        logger.info("Fitting ModelFeaturePipeline on %d training rows...", len(df_train))
        df = self._prepare_base_dataframe(df_train)
        df = self._apply_categoricals_and_numerics(df, is_train=True)
        df = self._engineer_base_and_uid_features(df, is_train=True)
        df = self._apply_frequency_encodings(df, is_train=True)
        df = self._apply_group_aggregations(df, is_train=True)
        self._select_and_filter_features(df, is_train=True)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform any partition (train, val, test, or single row) using fitted mappings.
        """
        if not self.is_fitted:
            raise RuntimeError("ModelFeaturePipeline must be fitted before calling transform().")

        df_prep = self._prepare_base_dataframe(df)
        df_prep = self._apply_categoricals_and_numerics(df_prep, is_train=False)
        df_prep = self._engineer_base_and_uid_features(df_prep, is_train=False)
        df_prep = self._apply_frequency_encodings(df_prep, is_train=False)
        df_prep = self._apply_group_aggregations(df_prep, is_train=False)
        X, _ = self._select_and_filter_features(df_prep, is_train=False)
        return X

    def fit_transform(self, df_train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Convenience single-pass fit and transform."""
        self.fit(df_train)
        df_prep = self._prepare_base_dataframe(df_train)
        y = df_prep["isFraud"].copy() if "isFraud" in df_prep.columns else None
        df_prep = self._apply_categoricals_and_numerics(df_prep, is_train=False)
        df_prep = self._engineer_base_and_uid_features(df_prep, is_train=False)
        df_prep = self._apply_frequency_encodings(df_prep, is_train=False)
        df_prep = self._apply_group_aggregations(df_prep, is_train=False)
        X, _ = self._select_and_filter_features(df_prep, is_train=False)
        return X, y

    def save(self, path: Path) -> None:
        """Serialize pipeline state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved ModelFeaturePipeline -> %s", path)

    @classmethod
    def load(cls, path: Path) -> "ModelFeaturePipeline":
        """Deserialize pipeline state from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Loaded ModelFeaturePipeline <- %s (%d features)", path, len(obj.feature_cols))
        return obj


# Backward compatibility alias
KaggleFeaturePipeline = ModelFeaturePipeline
