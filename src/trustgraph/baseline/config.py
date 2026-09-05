"""
config.py — TRUSTGRAPH Phase 1 Baseline Configuration
=======================================================

Single source of truth for all constants, paths, seeds, split
boundaries, and model hyperparameters used in the Phase-1 baseline.

Dataset:  IEEE-CIS Fraud Detection
Source:   IEEE DataPort
DOI:      10.21227/y5e7-wp63
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Project root (two levels up from this file: src/trustgraph/baseline/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Raw data location — READ ONLY, never modify these files
DATA_DIR = PROJECT_ROOT / "ieee-fraud-detection"

TRAIN_TRANSACTION_CSV = DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY_CSV    = DATA_DIR / "train_identity.csv"
TEST_TRANSACTION_CSV  = DATA_DIR / "test_transaction.csv"
TEST_IDENTITY_CSV     = DATA_DIR / "test_identity.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

# Artifact locations
ARTIFACTS_DIR      = PROJECT_ROOT / "artifacts" / "baseline"
MODEL_DIR          = ARTIFACTS_DIR / "model"
PREPROCESSING_DIR  = ARTIFACTS_DIR / "preprocessing"
PLOTS_DIR          = ARTIFACTS_DIR / "plots"
RESULTS_DIR        = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Chronological Split Boundaries
# ---------------------------------------------------------------------------
# Derived from inspection of TransactionDT distribution on train_transaction.csv
# (590,540 rows sorted by TransactionDT; approx 70/15/15 partition)
#
#   TRAIN:      TransactionDT <= TRAIN_DT_BOUNDARY   →  ~413,379 rows  (3.52% fraud)
#   VALIDATION: TRAIN_DT_BOUNDARY < DT <= VAL_DT_BOUNDARY →  ~88,581 rows  (3.43% fraud)
#   TEST:       TransactionDT > VAL_DT_BOUNDARY       →  ~88,580 rows  (3.48% fraud)
#
# TransactionDT is seconds since a reference epoch (not wall-clock time).
# Day 1 = DT 86400.  Full range: 86400 – 15,811,131  (≈ 182 days).

TRAIN_DT_BOUNDARY = 10_438_003   # ≈ day 120.8   (70th percentile)
VAL_DT_BOUNDARY   = 13_151_880   # ≈ day 152.2   (85th percentile)

# ---------------------------------------------------------------------------
# Feature Configuration
# ---------------------------------------------------------------------------
# These lists define exactly which columns appear in the model feature matrix.
# NEVER include: isFraud (target), TransactionID (identifier only).
# See README.md §Feature Selection for justification of each decision.

# Identifiers – used for joins/output only, NOT model features
ID_COLUMNS = ["TransactionID"]

# Target
TARGET_COLUMN = "isFraud"

# Numeric transaction features
NUMERIC_TRANSACTION_FEATURES = [
    "TransactionDT",    # Current transaction's temporal position (point-in-time feature)
    "TransactionAmt",   # Transaction amount
    "card1",            # Card attribute 1 (masked)
    "card2",            # Card attribute 2 (masked)
    "card3",            # Card attribute 3 (masked)
    "card5",            # Card attribute 5 (masked)
    "addr1",            # Address code 1 (billing)
    "addr2",            # Address code 2 (billing)
    "dist1",            # Distance 1 (59.7% missing — kept, LightGBM handles NaN)
    "dist2",            # Distance 2 (93.6% missing — kept, LightGBM handles NaN)
    # C-features: Vesta count features, no missing values
    "C1",  "C2",  "C3",  "C4",  "C5",  "C6",  "C7",
    "C8",  "C9",  "C10", "C11", "C12", "C13", "C14",
    # D-features: timedelta features, variable missingness
    "D1",  "D2",  "D3",  "D4",  "D5",  "D6",  "D7",
    "D8",  "D9",  "D10", "D11", "D12", "D13", "D14", "D15",
]

# V-features: Vesta engineered features (V1–V339), variable missingness
# All included — LightGBM handles NaN natively via optimal split search
VESTA_FEATURES = [f"V{i}" for i in range(1, 340)]

# Categorical transaction features (string-valued, require encoding)
CATEGORICAL_TRANSACTION_FEATURES = [
    "ProductCD",        # Product category (5 unique values)
    "card4",            # Card network: visa/mastercard/etc (4 unique)
    "card6",            # Card type: debit/credit (4 unique)
    "P_emaildomain",    # Purchaser email domain (59 unique)
    "R_emaildomain",    # Recipient email domain (60 unique)
    "M1",  "M2",  "M3", "M4",  "M5",               # Match flags (T/F)
    "M6",  "M7",  "M8", "M9",                       # Match flags (T/F)
]

# Identity numeric features (NaN for transactions with no identity record)
NUMERIC_IDENTITY_FEATURES = [
    "id_01", "id_02", "id_03", "id_04", "id_05",
    "id_06", "id_07", "id_08", "id_09", "id_10",
    "id_11", "id_13", "id_14", "id_17", "id_18",
    "id_19", "id_20", "id_21", "id_22",
    "id_24", "id_25", "id_26", "id_32",
]

# Identity categorical features (string-valued)
CATEGORICAL_IDENTITY_FEATURES = [
    "id_12",            # Found/NotFound (2 values)
    "id_15",            # New/Found/Unknown (3 values)
    "id_16",            # Found/NotFound (2 values)
    "id_23",            # IP_PROXY (string)
    "id_27",            # Found/NotFound (2 values)
    "id_28",            # Found/New (2 values)
    "id_29",            # Found/New/NotFound (3 values)
    "id_30",            # OS info (high cardinality string)
    "id_31",            # Browser info (high cardinality string)
    "id_33",            # Screen resolution (high cardinality string)
    "id_34",            # match_status (low cardinality)
    "id_35",            # T/F match
    "id_36",            # T/F match
    "id_37",            # T/F match
    "id_38",            # T/F match
    "DeviceType",       # desktop / mobile
    "DeviceInfo",       # Device model (high cardinality)
]

# All categorical columns in one list
ALL_CATEGORICAL_FEATURES = (
    CATEGORICAL_TRANSACTION_FEATURES + CATEGORICAL_IDENTITY_FEATURES
)

# All numeric columns in one list
ALL_NUMERIC_FEATURES = (
    NUMERIC_TRANSACTION_FEATURES
    + VESTA_FEATURES
    + NUMERIC_IDENTITY_FEATURES
)

# Final feature list (what the model sees)
ALL_FEATURES = ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# LightGBM Hyperparameters
# ---------------------------------------------------------------------------
# scale_pos_weight = negative_count / positive_count
# From inspection: 569877 / 20663 ≈ 27.58
# This is computed dynamically in training but we set a nominal value here.
FRAUD_POSITIVE_COUNT  = 20_663
FRAUD_NEGATIVE_COUNT  = 569_877
SCALE_POS_WEIGHT      = FRAUD_NEGATIVE_COUNT / FRAUD_POSITIVE_COUNT  # ≈ 27.58

LGBM_PARAMS = {
    "objective":          "binary",
    "metric":             "auc",
    "boosting_type":      "gbdt",
    "n_estimators":       3000,          # high ceiling; early stopping controls actual count
    "learning_rate":      0.05,
    "num_leaves":         256,
    "max_depth":          -1,            # unlimited depth controlled by num_leaves
    "min_child_samples":  20,
    "subsample":          0.8,
    "subsample_freq":     1,
    "colsample_bytree":   0.8,
    "reg_alpha":          0.1,
    "reg_lambda":         1.0,
    "scale_pos_weight":   SCALE_POS_WEIGHT,
    "random_state":       RANDOM_SEED,
    "n_jobs":             -1,
    "verbose":            -1,
}

# Early stopping: stop if validation AUC doesn't improve for this many rounds
EARLY_STOPPING_ROUNDS = 100

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
# Threshold is selected on validation set to maximise F1.
# Default value here is overwritten by the trained threshold saved to disk.
DEFAULT_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Dataset Provenance
# ---------------------------------------------------------------------------
DATASET_PROVENANCE = {
    "name":   "IEEE-CIS Fraud Detection",
    "source": "IEEE DataPort",
    "doi":    "10.21227/y5e7-wp63",
    "files": [
        "train_transaction.csv",
        "train_identity.csv",
        "test_transaction.csv",
        "test_identity.csv",
        "sample_submission.csv",
    ],
}
