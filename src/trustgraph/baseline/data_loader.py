"""
data_loader.py — TRUSTGRAPH Phase 1 Baseline Data Loading
===========================================================

Responsibilities:
  - Load IEEE-CIS transaction and identity CSV files (READ ONLY)
  - Normalise identity column names (test uses id-XX, train uses id_XX)
  - Left-join identity onto transactions (preserves all transactions)
  - Report join statistics
  - Sort by TransactionDT (chronological order)
  - Apply deterministic chronological 70 / 15 / 15 train/val/test split
  - Verify no partition overlap

The raw CSV files are NEVER modified.
"""

import gc
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from trustgraph.baseline import config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reduce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns to save RAM.
    float64 → float32 where the value range permits.
    int64   → int32   for columns that fit.
    This is applied ONLY after all joins; it does not affect CSV reading.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def _normalise_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename id-XX → id_XX (dash → underscore) in the identity table.
    The test_identity.csv uses dashes; train_identity.csv uses underscores.
    After normalisation both tables share the same column names.
    """
    rename_map = {c: c.replace("-", "_") for c in df.columns if "-" in c}
    if rename_map:
        logger.info("Normalising %d identity column names (dash → underscore)", len(rename_map))
        df = df.rename(columns=rename_map)
    return df


def _load_and_join(
    transaction_path: Path,
    identity_path: Path,
    has_target: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load a transaction CSV and an identity CSV and left-join them on TransactionID.

    Returns
    -------
    merged_df : pd.DataFrame
        The merged dataset, sorted by TransactionDT.
    join_stats : dict
        Row counts and join statistics.
    """
    logger.info("Loading transactions from: %s", transaction_path)
    txn = pd.read_csv(transaction_path, low_memory=False)
    logger.info("  → %d rows, %d columns", *txn.shape)

    logger.info("Loading identity from: %s", identity_path)
    idt = pd.read_csv(identity_path, low_memory=False)
    idt = _normalise_identity_columns(idt)
    logger.info("  → %d rows, %d columns", *idt.shape)

    # Verify no duplicates on join key
    txn_dup = txn["TransactionID"].duplicated().sum()
    idt_dup = idt["TransactionID"].duplicated().sum()
    if txn_dup > 0:
        logger.warning("Found %d duplicate TransactionIDs in transaction file!", txn_dup)
    if idt_dup > 0:
        logger.warning("Found %d duplicate TransactionIDs in identity file!", idt_dup)

    # Join statistics before merge
    matched = txn["TransactionID"].isin(idt["TransactionID"]).sum()
    unmatched = len(txn) - matched
    join_stats = {
        "transaction_rows":      len(txn),
        "identity_rows":         len(idt),
        "matched_identity_rows": int(matched),
        "unmatched_rows":        int(unmatched),
        "identity_coverage_pct": round(100.0 * matched / len(txn), 2),
    }
    logger.info(
        "Join stats: %d/%d transactions have identity data (%.1f%% unmatched → kept as NaN)",
        matched, len(txn), 100.0 * unmatched / len(txn)
    )

    # Left-join: EVERY transaction is preserved; identity columns are NaN where absent
    merged = txn.merge(idt, on="TransactionID", how="left")
    del txn, idt
    gc.collect()

    logger.info("After merge: %d rows, %d columns", *merged.shape)

    # Sort chronologically
    merged = merged.sort_values("TransactionDT").reset_index(drop=True)
    logger.info("Sorted by TransactionDT: [%d, %d]", merged["TransactionDT"].min(), merged["TransactionDT"].max())

    # Dtype reduction for memory efficiency
    merged = _reduce_dtypes(merged)
    gc.collect()

    return merged, join_stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_train_data() -> Tuple[pd.DataFrame, Dict]:
    """
    Load and join the training transaction and identity tables.

    Returns
    -------
    df : pd.DataFrame
        Full merged training dataset (590,540 rows), sorted by TransactionDT.
    join_stats : dict
        Join statistics for logging and reproducibility.
    """
    return _load_and_join(
        cfg.TRAIN_TRANSACTION_CSV,
        cfg.TRAIN_IDENTITY_CSV,
        has_target=True,
    )


def load_test_data() -> Tuple[pd.DataFrame, Dict]:
    """
    Load and join the test transaction and identity tables.
    These have no isFraud column. Used only for final inference output.

    Returns
    -------
    df : pd.DataFrame
        Full merged test dataset (506,691 rows), sorted by TransactionDT.
    join_stats : dict
        Join statistics.
    """
    return _load_and_join(
        cfg.TEST_TRANSACTION_CSV,
        cfg.TEST_IDENTITY_CSV,
        has_target=False,
    )


def chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Apply the deterministic chronological 70/15/15 split.

    Boundaries (seconds from reference epoch):
      TRAIN:      TransactionDT <= TRAIN_DT_BOUNDARY
      VALIDATION: TRAIN_DT_BOUNDARY < TransactionDT <= VAL_DT_BOUNDARY
      TEST:       TransactionDT > VAL_DT_BOUNDARY

    The split is NOT random — it respects temporal ordering.
    No transaction in TRAIN appears after any in VALIDATION or TEST.

    Returns
    -------
    train, val, test : pd.DataFrame
    split_meta : dict — split boundaries and row counts for reproducibility
    """
    train_mask = df["TransactionDT"] <= cfg.TRAIN_DT_BOUNDARY
    val_mask   = (df["TransactionDT"] > cfg.TRAIN_DT_BOUNDARY) & \
                 (df["TransactionDT"] <= cfg.VAL_DT_BOUNDARY)
    test_mask  = df["TransactionDT"] > cfg.VAL_DT_BOUNDARY

    train = df[train_mask].reset_index(drop=True)
    val   = df[val_mask].reset_index(drop=True)
    test  = df[test_mask].reset_index(drop=True)

    # Verify no partition overlap
    _verify_no_overlap(train, val, test)

    split_meta = {
        "train_dt_boundary":    cfg.TRAIN_DT_BOUNDARY,
        "val_dt_boundary":      cfg.VAL_DT_BOUNDARY,
        "train_rows":           len(train),
        "val_rows":             len(val),
        "test_rows":            len(test),
        "train_dt_min":         int(train["TransactionDT"].min()),
        "train_dt_max":         int(train["TransactionDT"].max()),
        "val_dt_min":           int(val["TransactionDT"].min()),
        "val_dt_max":           int(val["TransactionDT"].max()),
        "test_dt_min":          int(test["TransactionDT"].min()),
        "test_dt_max":          int(test["TransactionDT"].max()),
        "train_fraud_rate":     round(float(train["isFraud"].mean()), 6),
        "val_fraud_rate":       round(float(val["isFraud"].mean()), 6),
        "test_fraud_rate":      round(float(test["isFraud"].mean()), 6),
        "train_fraud_count":    int(train["isFraud"].sum()),
        "val_fraud_count":      int(val["isFraud"].sum()),
        "test_fraud_count":     int(test["isFraud"].sum()),
    }

    logger.info(
        "Split: TRAIN=%d (DT %d–%d, fraud=%.2f%%) | VAL=%d (DT %d–%d, fraud=%.2f%%) | TEST=%d (DT %d–%d, fraud=%.2f%%)",
        len(train), split_meta["train_dt_min"], split_meta["train_dt_max"], 100*split_meta["train_fraud_rate"],
        len(val),   split_meta["val_dt_min"],   split_meta["val_dt_max"],   100*split_meta["val_fraud_rate"],
        len(test),  split_meta["test_dt_min"],  split_meta["test_dt_max"],  100*split_meta["test_fraud_rate"],
    )

    return train, val, test, split_meta


def _verify_no_overlap(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Assert chronological ordering between all three partitions.
    Raises AssertionError if any overlap is found.
    """
    train_max_dt = train["TransactionDT"].max()
    val_min_dt   = val["TransactionDT"].min()
    val_max_dt   = val["TransactionDT"].max()
    test_min_dt  = test["TransactionDT"].min()

    assert train_max_dt <= val_min_dt, (
        f"PARTITION OVERLAP: max(train DT)={train_max_dt} > min(val DT)={val_min_dt}"
    )
    assert val_max_dt <= test_min_dt, (
        f"PARTITION OVERLAP: max(val DT)={val_max_dt} > min(test DT)={test_min_dt}"
    )

    train_ids = set(train["TransactionID"])
    val_ids   = set(val["TransactionID"])
    test_ids  = set(test["TransactionID"])

    assert len(train_ids & val_ids) == 0, "TransactionID overlap between TRAIN and VAL!"
    assert len(val_ids & test_ids) == 0,  "TransactionID overlap between VAL and TEST!"
    assert len(train_ids & test_ids) == 0, "TransactionID overlap between TRAIN and TEST!"

    logger.info("Partition overlap check PASSED: no DT overlap, no TransactionID overlap.")


def get_feature_and_target(
    df: pd.DataFrame,
    feature_cols: list,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract the feature matrix X and target series y from a partition.

    Programmatic safety checks:
      - isFraud must NOT appear in feature_cols
      - TransactionID must NOT appear in feature_cols

    Parameters
    ----------
    df : pd.DataFrame — a partition (train / val / test)
    feature_cols : list — the ordered list of model feature columns

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    # Safety assertions (will also be tested in test suite)
    assert "isFraud" not in feature_cols, (
        "LEAKAGE: isFraud found in feature_cols!"
    )
    assert "TransactionID" not in feature_cols, (
        "IDENTIFIER LEAKAGE: TransactionID found in feature_cols!"
    )

    # Only use columns that actually exist in this dataframe
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning("The following feature_cols are missing from df and will be skipped: %s", missing)

    X = df[available].copy()
    y = df["isFraud"].copy()

    return X, y
