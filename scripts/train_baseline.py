"""
train_baseline.py — TRUSTGRAPH Phase 1 End-to-End Training Script
==================================================================

Usage:
    python scripts/train_baseline.py

This script orchestrates the complete Phase-1 baseline training pipeline:

    1. Load and join IEEE-CIS train_transaction + train_identity
    2. Apply chronological 70/15/15 TRAIN/VALIDATION/TEST split
    3. Fit preprocessing on TRAIN only
    4. Train LightGBM binary classifier (scale_pos_weight, early stopping)
    5. Select binary threshold on VALIDATION (maximise F1)
    6. Freeze model, threshold, and all artifacts
    7. Log validation metrics

LEAKAGE PREVENTION:
    - Preprocessing fitted on TRAIN only
    - Threshold selected on VALIDATION only
    - TEST partition untouched until evaluate_baseline.py is run

DO NOT modify this script to evaluate on TEST here.
Run evaluate_baseline.py for the final test-set evaluation.
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgraph.baseline import config as cfg
from trustgraph.baseline.data_loader import (
    chronological_split,
    get_feature_and_target,
    load_train_data,
)
from trustgraph.baseline.evaluate import (
    compute_metrics,
    select_threshold_max_f1,
)
from trustgraph.baseline.model import BaselineModel
from trustgraph.baseline.preprocessing import BaselinePreprocessor

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_baseline")


def main() -> None:
    t_start = time.perf_counter()
    logger.info("=" * 70)
    logger.info("TRUSTGRAPH Phase 1 — Baseline Training")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading and joining train data...")
    df, join_stats = load_train_data()
    logger.info("Join statistics: %s", join_stats)

    # ------------------------------------------------------------------
    # 2. Chronological split
    # ------------------------------------------------------------------
    logger.info("Step 2: Applying chronological 70/15/15 split...")
    train_df, val_df, test_df, split_meta = chronological_split(df)

    # Hold out TEST — do not use it here
    logger.info("TEST partition held out (%d rows). Not used in training/tuning.", len(test_df))
    del df  # free memory
    gc.collect()

    # ------------------------------------------------------------------
    # 3. Preprocessing — fit on TRAIN only
    # ------------------------------------------------------------------
    logger.info("Step 3: Fitting preprocessing on TRAIN partition...")
    preprocessor = BaselinePreprocessor()
    X_train = preprocessor.fit_transform(train_df)
    y_train = train_df["isFraud"].values

    logger.info("Step 3b: Transforming VALIDATION partition (no refitting)...")
    X_val = preprocessor.transform(val_df)
    y_val = val_df["isFraud"].values

    # Safety assertions
    feature_cols = preprocessor.feature_cols
    assert "isFraud"       not in feature_cols, "LEAKAGE: isFraud in features!"
    assert "TransactionID" not in feature_cols, "IDENTIFIER LEAK: TransactionID in features!"
    logger.info("Leakage check PASSED: isFraud and TransactionID not in feature list.")
    logger.info("Feature count: %d", len(feature_cols))

    # Free raw dataframes (features are now in X_train / X_val)
    train_df_ids = train_df[["TransactionID", "TransactionDT", "isFraud"]].copy()
    val_df_ids   = val_df[["TransactionID", "TransactionDT", "isFraud"]].copy()
    test_df_ids  = test_df[["TransactionID", "TransactionDT", "isFraud"]].copy()
    del train_df, val_df
    gc.collect()

    # ------------------------------------------------------------------
    # 4. Train LightGBM
    # ------------------------------------------------------------------
    logger.info("Step 4: Training LightGBM classifier...")
    model = BaselineModel()
    model.fit(
        X_train, y_train,
        X_val,   y_val,
        cat_cols=preprocessor.cat_cols,
    )
    del X_train, y_train  # free training data
    gc.collect()

    # ------------------------------------------------------------------
    # 5. Validation metrics + threshold selection
    # ------------------------------------------------------------------
    logger.info("Step 5: Evaluating on VALIDATION partition...")
    A_val = model.predict_risk(X_val)

    # Select threshold that maximises F1 on validation
    threshold, val_f1 = select_threshold_max_f1(y_val, A_val, beta=1.0)
    logger.info("Frozen threshold: %.6f  (val F1 = %.4f)", threshold, val_f1)

    val_metrics = compute_metrics(y_val, A_val, threshold, partition_name="validation")
    logger.info("Validation metrics: %s", val_metrics)

    # ------------------------------------------------------------------
    # 6. Save all artifacts
    # ------------------------------------------------------------------
    logger.info("Step 6: Saving artifacts...")

    cfg.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    cfg.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = cfg.MODEL_DIR / "lgbm_model.pkl"
    model.save(model_path)

    # Preprocessing
    preprocessor.save(cfg.PREPROCESSING_DIR)

    # Feature list
    feature_list_path = cfg.ARTIFACTS_DIR / "feature_list.json"
    with open(feature_list_path, "w") as f:
        json.dump({
            "feature_cols":            feature_cols,
            "n_features":              len(feature_cols),
            "categorical_features":    preprocessor.cat_cols,
            "n_categorical":           len(preprocessor.cat_cols),
            "excluded_features": {
                "isFraud":         "TARGET — never in feature matrix",
                "TransactionID":   "Identifier only — no predictive use as numerical feature",
            },
        }, f, indent=2)

    # Model config
    config_path = cfg.ARTIFACTS_DIR / "config.json"
    model_config = model.get_config()
    model_config["lgbm_version"] = __import__("lightgbm").__version__
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)

    # Threshold
    threshold_path = cfg.ARTIFACTS_DIR / "threshold.json"
    with open(threshold_path, "w") as f:
        json.dump({
            "threshold":            float(threshold),
            "selected_on":          "validation",
            "selection_criterion":  "maximise F1",
            "validation_f1":        float(val_f1),
        }, f, indent=2)

    # Split metadata
    split_path = cfg.ARTIFACTS_DIR / "split.json"
    with open(split_path, "w") as f:
        json.dump({**split_meta, **join_stats}, f, indent=2)

    # Validation metrics
    val_metrics_path = cfg.ARTIFACTS_DIR / "val_metrics.json"
    with open(val_metrics_path, "w") as f:
        json.dump(val_metrics, f, indent=2)

    # Feature importance
    fi = model.feature_importance()
    fi.to_csv(cfg.ARTIFACTS_DIR / "feature_importance.csv", index=False)

    # Reproducibility record
    import platform
    import sklearn
    import lightgbm
    repro_path = cfg.ARTIFACTS_DIR / "reproducibility.json"
    with open(repro_path, "w") as f:
        json.dump({
            "dataset":          cfg.DATASET_PROVENANCE,
            "random_seed":      cfg.RANDOM_SEED,
            "train_dt_boundary": cfg.TRAIN_DT_BOUNDARY,
            "val_dt_boundary":   cfg.VAL_DT_BOUNDARY,
            "python_version":   platform.python_version(),
            "lightgbm_version": lightgbm.__version__,
            "sklearn_version":  sklearn.__version__,
            "numpy_version":    np.__version__,
            "pandas_version":   pd.__version__,
            "split_meta":       split_meta,
            "join_stats":       join_stats,
        }, f, indent=2)

    elapsed = time.perf_counter() - t_start
    logger.info("=" * 70)
    logger.info("Training complete in %.1f s", elapsed)
    logger.info("Artifacts saved to: %s", cfg.ARTIFACTS_DIR)
    logger.info("")
    logger.info("VALIDATION METRICS (frozen model):")
    for k, v in val_metrics.items():
        logger.info("  %s: %s", k, v)
    logger.info("")
    logger.info("Next step: run  python scripts/evaluate_baseline.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
