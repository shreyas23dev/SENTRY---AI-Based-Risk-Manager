"""
evaluate_knowledge_graph.py — Evaluate Point-in-Time Payment Knowledge Graph
=============================================================================

Chronological protocol:
  - Stream TRAIN (N = 413,379) to build initial graph state and historical fraud registry.
  - Stream VAL (N = 88,581) to advance graph state and tune validation threshold for G_t.
  - Stream held-out TEST (N = 88,580) with is_train=False (zero test label leakage).
  - Compute standalone G_t metrics, coverage, cold-start rate, and incremental comparison with A_t.
  - Export results to artifacts/graph/graph_evaluation.json.
"""

from __future__ import annotations

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
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trustgraph.baseline.data_loader import chronological_split, load_train_data
from trustgraph.graph import GraphPipelineBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_knowledge_graph")

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "graph"
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

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.5
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = base_rate

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
        "total": int(n_total),
        "fraud_count": int(n_fraud),
        "fraud_capture_pct": float(round(100.0 * rec, 2)),
        "fraud_rate": float(round(tier_fraud_rate, 6)),
        "enrichment": float(round(enrichment, 4)),
    }


def find_best_validation_threshold(
    y_val: np.ndarray, prob_val: np.ndarray
) -> Tuple[float, Dict[str, Any]]:
    """Find threshold that maximizes F1 score on VALIDATION only."""
    best_th = 0.10
    best_f1 = -1.0
    best_metrics = {}

    thresholds = np.linspace(0.01, 0.90, 90)
    for th in thresholds:
        m = evaluate_binary_predictions(y_val, prob_val, th)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_th = th
            best_metrics = m

    return float(best_th), best_metrics


def main():
    logger.info("=================================================================")
    logger.info("  TRUSTGRAPH PHASE 2: PAYMENT KNOWLEDGE GRAPH EVALUATION")
    logger.info("=================================================================")

    # 1. Load dataset
    logger.info("Loading full IEEE-CIS dataset...")
    df_raw, _ = load_train_data()

    # 2. Chronological split
    logger.info("Applying chronological 70/15/15 train/val/test split...")
    train_df, val_df, test_df, split_meta = chronological_split(df_raw)
    del df_raw
    gc.collect()

    logger.info(
        "TRAIN: N=%d | VAL: N=%d | TEST: N=%d",
        len(train_df), len(val_df), len(test_df)
    )

    y_train = train_df["isFraud"].values
    y_val = val_df["isFraud"].values
    y_test = test_df["isFraud"].values

    builder = GraphPipelineBuilder()

    # 3. Stream TRAIN through graph (ingests nodes, edges, and historical fraud labels)
    logger.info("Streaming TRAIN (N = %d) through PaymentKnowledgeGraph...", len(train_df))
    t0 = time.perf_counter()
    _ = builder.process_dataframe_stream(train_df, is_train=True, log_interval=100_000)
    t_train = time.perf_counter() - t0
    logger.info("TRAIN streaming completed in %.2f s.", t_train)

    del train_df
    gc.collect()

    # 4. Stream VALIDATION through graph
    logger.info("Streaming VALIDATION (N = %d) through PaymentKnowledgeGraph...", len(val_df))
    t0 = time.perf_counter()
    val_graph_df = builder.process_dataframe_stream(val_df, is_train=True, log_interval=40_000)
    t_val = time.perf_counter() - t0
    logger.info("VAL streaming completed in %.2f s.", t_val)

    g_val = val_graph_df["graph_risk"].values
    best_val_th, val_metrics = find_best_validation_threshold(y_val, g_val)
    logger.info(
        "Optimal VALIDATION threshold for G_t: tau_G = %.4f (F1 = %.4f, Prec = %.2f%%, Rec = %.2f%%)",
        best_val_th, val_metrics["f1"], val_metrics["precision"] * 100, val_metrics["recall"] * 100
    )

    del val_df
    gc.collect()

    # 5. Stream held-out TEST through graph (is_train=False: ZERO test labels added)
    logger.info("Streaming held-out TEST (N = %d) through PaymentKnowledgeGraph (ZERO TEST LABELS)...", len(test_df))
    t0 = time.perf_counter()
    test_graph_df = builder.process_dataframe_stream(test_df, is_train=False, log_interval=40_000)
    t_test = time.perf_counter() - t0
    logger.info("TEST streaming completed in %.2f s.", t_test)

    # 6. Compute Coverage & Cold-Start Rates on TEST
    g_test = test_graph_df["graph_risk"].values
    has_context_mask = test_graph_df["has_graph_context"].values == 1
    coverage_pct = float(round(100.0 * has_context_mask.mean(), 2))
    cold_start_pct = float(round(100.0 - coverage_pct, 2))

    logger.info("TEST Coverage (transactions with prior graph context): %.2f%%", coverage_pct)
    logger.info("TEST Cold-Start Rate (transactions with zero prior graph context): %.2f%%", cold_start_pct)

    # 7. Evaluate G_t on TEST
    test_metrics_g = evaluate_binary_predictions(y_test, g_test, best_val_th)

    # 8. Load Phase 1 A_t predictions for comparison
    a_test_path = PROJECT_ROOT / "artifacts" / "models" / "kaggle_xgb" / "test_predictions_kaggle_xgb.parquet"
    if a_test_path.exists():
        a_df = pd.read_parquet(a_test_path)
        a_test = a_df["A_t"].values
        test_metrics_a = evaluate_binary_predictions(y_test, a_test, 0.1200)

        # Analytical combination A_t + G_t (0.7 * A_t + 0.3 * G_t)
        combined_prob = 0.70 * a_test + 0.30 * g_test
        test_metrics_combined = evaluate_binary_predictions(y_test, combined_prob, 0.1200)
    else:
        test_metrics_a = {}
        test_metrics_combined = {}

    # 9. Comparison Output
    logger.info("=========================================================================================")
    logger.info("  PHASE 2 EVALUATION ON HELD-OUT TEST (N = 88,580, Frauds = 3,083)")
    logger.info("=========================================================================================")
    logger.info("Metric                  | A_t Only (XGBoost)        | G_t Only (Graph)          | A_t + G_t (Combined)")
    logger.info("-----------------------------------------------------------------------------------------")
    for k, label in [
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
        v_a = test_metrics_a.get(k, 0.0)
        v_g = test_metrics_g.get(k, 0.0)
        v_c = test_metrics_combined.get(k, 0.0)
        if isinstance(v_a, float):
            logger.info("%-23s | %25.4f | %25.4f | %21.4f", label, v_a, v_g, v_c)
        else:
            logger.info("%-23s | %25d | %25d | %21d", label, v_a, v_g, v_c)
    logger.info("=========================================================================================")

    # 10. Extract Sample Evidence for high-risk test transaction
    high_risk_idx = np.where((g_test >= best_val_th) & (y_test == 1))[0]
    sample_evidence = None
    if len(high_risk_idx) > 0:
        sample_txn_id = int(test_df.iloc[high_risk_idx[0]]["TransactionID"])
        ev = builder.get_transaction_evidence(sample_txn_id)
        if ev:
            sample_evidence = ev.to_dict()
            logger.info("Sample high-risk fraud evidence extracted for Txn %d:", sample_txn_id)
            logger.info(json.dumps(sample_evidence, indent=2))

    # 11. Save artifacts
    results = {
        "evaluation_partition": "Held-out TEST (N = 88,580)",
        "coverage_percentage": coverage_pct,
        "cold_start_percentage": cold_start_pct,
        "validation_selected_threshold_g": best_val_th,
        "validation_metrics_g": val_metrics,
        "test_metrics_g_only": test_metrics_g,
        "test_metrics_a_only": test_metrics_a,
        "test_metrics_combined_analytical": test_metrics_combined,
        "sample_evidence": sample_evidence,
        "graph_state_summary": builder.graph.get_state_summary(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    with open(OUTPUT_DIR / "graph_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved graph evaluation to %s", OUTPUT_DIR / "graph_evaluation.json")

    # Save test graph features parquet
    test_graph_df.to_parquet(OUTPUT_DIR / "test_graph_features.parquet")
    logger.info("Saved test graph features to %s", OUTPUT_DIR / "test_graph_features.parquet")


if __name__ == "__main__":
    main()
