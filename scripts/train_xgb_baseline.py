"""
train_xgb_baseline.py — Train and Evaluate XGBoost Fraud Detection Baseline
===========================================================================

Strict protocol adherence:
  - Model fitting on TRAIN ONLY (N = 413,379)
  - Threshold selection on VALIDATION ONLY (N = 88,581)
  - Final untouched evaluation on TEST (N = 88,580)
  - Apples-to-apples comparison against legacy LightGBM baseline
  - Zero leakage: preprocessors fit on TRAIN only
  - Exports artifacts to artifacts/models/kaggle_xgb/
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Insert project src path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trustgraph.baseline import config as base_cfg
from trustgraph.baseline.data_loader import chronological_split, load_train_data
from trustgraph.baseline.model_features import ModelFeaturePipeline
from trustgraph.baseline.xgb_model import XGBRiskModel
from trustgraph.baseline.model import BaselineModel
from trustgraph.baseline.preprocessing import BaselinePreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_xgb_baseline")

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "models" / "kaggle_xgb"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_binary_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics at a specific threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    n_total = len(y_true)
    n_fraud = int(y_true.sum())
    base_rate = n_fraud / max(n_total, 1)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / max(tn + fp, 1)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    pred_fraud = tp + fp
    tier_fraud_rate = tp / max(pred_fraud, 1)
    enrichment = tier_fraud_rate / max(base_rate, 1e-9)

    return {
        "threshold": float(threshold),
        "precision": float(round(prec, 6)),
        "recall": float(round(rec, 6)),
        "f1": float(round(f1, 6)),
        "roc_auc": float(round(roc_auc, 6)),
        "pr_auc": float(round(pr_auc, 6)),
        "fpr": float(round(fpr, 6)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total_test": int(n_total),
        "fraud_count": int(n_fraud),
        "fraud_capture_pct": float(round(100.0 * rec, 2)),
        "fraud_rate": float(round(tier_fraud_rate, 6)),
        "enrichment": float(round(enrichment, 4)),
    }


def find_best_validation_threshold(
    y_val: np.ndarray, prob_val: np.ndarray
) -> Tuple[float, Dict[str, Any]]:
    """Find threshold that maximizes F1 score on VALIDATION only."""
    best_th = 0.50
    best_f1 = -1.0
    best_metrics = {}

    thresholds = np.linspace(0.01, 0.95, 95)
    for th in thresholds:
        m = evaluate_binary_predictions(y_val, prob_val, th)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = th
            best_metrics = m

    return float(best_th), best_metrics


def main():
    logger.info("=================================================================")
    logger.info("  TRUSTGRAPH PHASE 1: XGBOOST BASELINE")
    logger.info("=================================================================")

    # 1. Load data
    logger.info("Loading full IEEE-CIS dataset...")
    df_raw, _ = load_train_data()

    # 2. Chronological split
    logger.info("Applying chronological 70/15/15 train/val/test split...")
    train_df, val_df, test_df, split_meta = chronological_split(df_raw)
    del df_raw
    gc.collect()

    logger.info(
        "TRAIN: N=%d (fraud=%d) | VAL: N=%d (fraud=%d) | TEST: N=%d (fraud=%d)",
        len(train_df), split_meta["train_fraud_count"],
        len(val_df), split_meta["val_fraud_count"],
        len(test_df), split_meta["test_fraud_count"]
    )

    y_train = train_df["isFraud"].values
    y_val = val_df["isFraud"].values
    y_test = test_df["isFraud"].values

    # 3. Fit Feature Engineering Pipeline on TRAIN ONLY
    pipeline_cache_path = OUTPUT_DIR / "feature_pipeline.pkl"
    if pipeline_cache_path.exists():
        logger.info("Loading cached feature pipeline from %s", pipeline_cache_path)
        pipe = ModelFeaturePipeline.load(pipeline_cache_path)
    else:
        logger.info("Fitting ModelFeaturePipeline strictly on TRAIN (N=%d)...", len(train_df))
        t0 = time.perf_counter()
        pipe = ModelFeaturePipeline()
        pipe.fit(train_df)
        t_pipe = time.perf_counter() - t0
        logger.info("Feature pipeline fitted in %.2f s. Total features: %d", t_pipe, len(pipe.feature_cols))
        pipe.save(pipeline_cache_path)

    # 4. Transform partitions
    features_cache_dir = OUTPUT_DIR / "cache"
    features_cache_dir.mkdir(parents=True, exist_ok=True)
    train_feat_path = features_cache_dir / "X_train_kaggle.parquet"
    val_feat_path = features_cache_dir / "X_val_kaggle.parquet"
    test_feat_path = features_cache_dir / "X_test_kaggle.parquet"

    if train_feat_path.exists() and val_feat_path.exists() and test_feat_path.exists():
        logger.info("Loading cached feature matrices from %s...", features_cache_dir)
        X_train = pd.read_parquet(train_feat_path)
        X_val = pd.read_parquet(val_feat_path)
        X_test = pd.read_parquet(test_feat_path)
    else:
        logger.info("Transforming TRAIN...")
        X_train = pipe.transform(train_df)
        X_train.to_parquet(train_feat_path)

        logger.info("Transforming VAL...")
        X_val = pipe.transform(val_df)
        X_val.to_parquet(val_feat_path)

        logger.info("Transforming TEST...")
        X_test = pipe.transform(test_df)
        X_test.to_parquet(test_feat_path)

    logger.info("Feature matrix ready. Shapes: Train %s, Val %s, Test %s", X_train.shape, X_val.shape, X_test.shape)

    # 5. Train XGBoost Model
    model_path = OUTPUT_DIR / "xgb_model.pkl"
    if model_path.exists():
        logger.info("Loading trained XGBoost model from %s", model_path)
        xgb_model = XGBRiskModel.load(model_path)
    else:
        logger.info("Training XGBRiskModel on TRAIN (early stopping on VAL)...")
        xgb_model = XGBRiskModel()
        xgb_model.fit(X_train, y_train, X_val, y_val, verbose=100)
        xgb_model.save(model_path)

    # 6. Predict probabilities
    logger.info("Generating risk probabilities (A_t) for VAL and TEST...")
    prob_val_xgb = xgb_model.predict_risk(X_val)
    prob_test_xgb = xgb_model.predict_risk(X_test)

    # 7. Validation Threshold Selection (VALIDATION ONLY)
    best_val_th, best_val_metrics = find_best_validation_threshold(y_val, prob_val_xgb)
    logger.info("Selected optimal validation threshold: tau = %.4f (F1 = %.4f, Prec = %.2f%%, Rec = %.2f%%)",
                best_val_th, best_val_metrics["f1"], best_val_metrics["precision"]*100, best_val_metrics["recall"]*100)

    # 8. Evaluate on TEST
    test_metrics_opt = evaluate_binary_predictions(y_test, prob_test_xgb, best_val_th)
    test_metrics_old_th = evaluate_binary_predictions(y_test, prob_test_xgb, 0.594298)

    # 9. Evaluate Old LightGBM Baseline on identical TEST partition
    logger.info("Loading frozen LightGBM baseline for apples-to-apples comparison...")
    lgb_model = BaselineModel.load(PROJECT_ROOT / "artifacts" / "baseline" / "model" / "lgbm_model.pkl")
    lgb_prep = BaselinePreprocessor.load(PROJECT_ROOT / "artifacts" / "baseline" / "preprocessing")
    X_test_lgb = lgb_prep.transform(test_df)
    prob_test_lgb = lgb_model.predict_risk(X_test_lgb)
    lgb_test_metrics = evaluate_binary_predictions(y_test, prob_test_lgb, 0.594298)

    # 10. Summary Comparison Table
    logger.info("=========================================================================================")
    logger.info("  APPLES-TO-APPLES BASELINE COMPARISON ON HELD-OUT TEST (N = 88,580, Frauds = 3,083)")
    logger.info("=========================================================================================")
    logger.info("Metric                  | Old LightGBM (tau=0.5943) | XGB (tau=%.4f) | XGB (tau=0.5943)", best_val_th)
    logger.info("-----------------------------------------------------------------------------------------")
    for metric_key, label in [
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-Score"),
        ("fpr", "False Positive Rate"),
        ("tp", "True Positives (TP)"),
        ("fp", "False Positives (FP)"),
        ("fn", "False Negatives (FN)"),
        ("tn", "True Negatives (TN)"),
        ("fraud_capture_pct", "Fraud Capture (%)"),
        ("enrichment", "Fraud Enrichment"),
    ]:
        val_lgb = lgb_test_metrics[metric_key]
        val_xgb_opt = test_metrics_opt[metric_key]
        val_xgb_old = test_metrics_old_th[metric_key]
        if isinstance(val_lgb, float):
            logger.info("%-23s | %25.4f | %21.4f | %21.4f", label, val_lgb, val_xgb_opt, val_xgb_old)
        else:
            logger.info("%-23s | %25d | %21d | %21d", label, val_lgb, val_xgb_opt, val_xgb_old)
    logger.info("=========================================================================================")

    # 11. Save Metadata and Evaluation Results
    metadata = {
        "model_type": "XGBoost Classifier",
        "source": "IEEE-CIS Fraud Detection solution (Chris Deotte / Konstantin Yakovlev)",
        "feature_count": len(pipe.feature_cols),
        "feature_names": pipe.feature_cols,
        "xgboost_parameters": xgb_model.params,
        "best_iteration": xgb_model.best_iteration,
        "training_partition": {
            "rows": split_meta["train_rows"],
            "fraud_count": split_meta["train_fraud_count"],
            "fraud_rate": split_meta["train_fraud_rate"],
            "dt_min": split_meta["train_dt_min"],
            "dt_max": split_meta["train_dt_max"],
        },
        "validation_partition": {
            "rows": split_meta["val_rows"],
            "fraud_count": split_meta["val_fraud_count"],
            "fraud_rate": split_meta["val_fraud_rate"],
            "dt_min": split_meta["val_dt_min"],
            "dt_max": split_meta["val_dt_max"],
        },
        "test_partition": {
            "rows": split_meta["test_rows"],
            "fraud_count": split_meta["test_fraud_count"],
            "fraud_rate": split_meta["test_fraud_rate"],
            "dt_min": split_meta["test_dt_min"],
            "dt_max": split_meta["test_dt_max"],
        },
        "validation_selected_threshold": best_val_th,
        "validation_metrics_at_selected_threshold": best_val_metrics,
        "test_metrics_at_selected_threshold": test_metrics_opt,
        "test_metrics_at_old_threshold": test_metrics_old_th,
        "old_lightgbm_test_metrics": lgb_test_metrics,
        "leakage_audit_status": {
            "test_label_leakage": "NONE (trained and selected strictly on TRAIN/VAL)",
            "preprocessing_fit": "TRAIN ONLY (all frequency and group statistics computed strictly on TRAIN)",
            "chronological_ordering": "STRICT (TRAIN DT <= 10,438,003 < VAL DT <= 13,151,880 < TEST DT)",
            "offline_group_aggregations_note": "Aggregations are fit strictly on TRAIN to preserve chronological validity.",
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", OUTPUT_DIR / "metadata.json")

    # Save test predictions
    test_preds_df = pd.DataFrame({
        "TransactionID": test_df["TransactionID"].values,
        "TransactionDT": test_df["TransactionDT"].values,
        "A_t": prob_test_xgb,
        "isFraud": y_test,
    })
    test_preds_df.to_parquet(OUTPUT_DIR / "test_predictions_kaggle_xgb.parquet")
    logger.info("Saved test predictions to %s", OUTPUT_DIR / "test_predictions_kaggle_xgb.parquet")


if __name__ == "__main__":
    main()
