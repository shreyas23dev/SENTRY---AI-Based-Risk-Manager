"""
evaluate_baseline.py — TRUSTGRAPH Phase 1 Final Test-Set Evaluation
====================================================================

Usage:
    python scripts/evaluate_baseline.py

This script performs the FINAL evaluation on the held-out TEST partition.

    1. Load the frozen model, preprocessor, threshold, and split metadata
    2. Reconstruct the TEST partition from the training data
    3. Apply preprocessing (no refitting)
    4. Score with predict_risk() → A_t
    5. Compute all metrics with the FROZEN threshold
    6. Generate all four evaluation plots
    7. Measure inference latency
    8. Save test_predictions.csv and metrics.json

IMPORTANT: The model, preprocessor, and threshold are all FROZEN from
train_baseline.py. Nothing is re-fitted or re-selected here.
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgraph.baseline import config as cfg
from trustgraph.baseline.data_loader import chronological_split, load_train_data
from trustgraph.baseline.evaluate import (
    compute_metrics,
    generate_all_plots,
    measure_latency,
)
from trustgraph.baseline.model import BaselineModel
from trustgraph.baseline.preprocessing import BaselinePreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_baseline")


def main() -> None:
    t_start = time.perf_counter()
    logger.info("=" * 70)
    logger.info("TRUSTGRAPH Phase 1 — Final Test-Set Evaluation")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load frozen artifacts
    # ------------------------------------------------------------------
    logger.info("Loading frozen model...")
    model_path = cfg.MODEL_DIR / "lgbm_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train_baseline.py first.")
    model = BaselineModel.load(model_path)

    logger.info("Loading frozen preprocessor...")
    preprocessor = BaselinePreprocessor.load(cfg.PREPROCESSING_DIR)

    logger.info("Loading frozen threshold...")
    with open(cfg.ARTIFACTS_DIR / "threshold.json") as f:
        threshold_data = json.load(f)
    threshold = float(threshold_data["threshold"])
    logger.info("Frozen threshold: %.6f", threshold)

    # ------------------------------------------------------------------
    # 2. Reconstruct TEST partition
    # ------------------------------------------------------------------
    logger.info("Loading train data to extract TEST partition...")
    df, join_stats = load_train_data()
    _, _, test_df, split_meta = chronological_split(df)
    del df
    gc.collect()

    logger.info(
        "TEST partition: %d rows, fraud rate=%.3f%%",
        len(test_df), 100 * test_df["isFraud"].mean()
    )

    # Keep identifiers for output
    test_ids = test_df[["TransactionID", "TransactionDT", "isFraud"]].copy()
    y_test   = test_df["isFraud"].values

    # ------------------------------------------------------------------
    # 3. Preprocess TEST (no refitting)
    # ------------------------------------------------------------------
    logger.info("Applying frozen preprocessor to TEST partition...")
    X_test = preprocessor.transform(test_df)
    del test_df
    gc.collect()

    # ------------------------------------------------------------------
    # 4. Score: A_t
    # ------------------------------------------------------------------
    logger.info("Generating A_t scores for TEST partition...")
    A_test = model.predict_risk(X_test)

    # Verify A_t contract
    assert A_test.ndim == 1
    assert len(A_test) == len(y_test)
    assert np.all(A_test >= 0.0) and np.all(A_test <= 1.0), "A_t out of [0,1] range!"
    logger.info("A_t contract verified: shape=%s, range=[%.4f, %.4f]", A_test.shape, A_test.min(), A_test.max())

    # ------------------------------------------------------------------
    # 5. Compute metrics
    # ------------------------------------------------------------------
    logger.info("Computing TEST metrics with frozen threshold=%.6f...", threshold)
    test_metrics = compute_metrics(y_test, A_test, threshold, partition_name="test")

    # ------------------------------------------------------------------
    # 6. Generate plots
    # ------------------------------------------------------------------
    logger.info("Generating evaluation plots...")
    plot_paths = generate_all_plots(
        y_test, A_test, threshold,
        plots_dir=cfg.PLOTS_DIR,
        partition="test",
    )

    # ------------------------------------------------------------------
    # 7. Inference latency
    # ------------------------------------------------------------------
    logger.info("Measuring inference latency...")
    latency = measure_latency(model, X_test, n_warmup=2, n_repeats=3)
    logger.info("Latency: %s", latency)

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    # Final metrics JSON
    test_metrics_path = cfg.ARTIFACTS_DIR / "metrics.json"
    full_metrics = {
        "test_metrics":    test_metrics,
        "latency":         latency,
        "threshold":       threshold_data,
        "split":           split_meta,
    }
    with open(test_metrics_path, "w") as f:
        json.dump(full_metrics, f, indent=2)
    logger.info("Metrics saved → %s", test_metrics_path)

    # Test predictions CSV
    baseline_preds = (A_test >= threshold).astype(int)
    test_results = test_ids.copy()
    test_results["A_t"]                = A_test
    test_results["baseline_prediction"] = baseline_preds

    results_path = cfg.RESULTS_DIR / "test_predictions.csv"
    test_results.to_csv(results_path, index=False)
    logger.info("Test predictions saved → %s  (%d rows)", results_path, len(test_results))

    elapsed = time.perf_counter() - t_start
    logger.info("=" * 70)
    logger.info("Evaluation complete in %.1f s", elapsed)
    logger.info("")
    logger.info("FINAL TEST METRICS (frozen threshold = %.6f):", threshold)
    logger.info("  ROC-AUC:           %.6f", test_metrics["roc_auc"])
    logger.info("  PR-AUC:            %.6f", test_metrics["pr_auc"])
    logger.info("  Precision:         %.6f", test_metrics["precision"])
    logger.info("  Recall:            %.6f", test_metrics["recall"])
    logger.info("  F1:                %.6f", test_metrics["f1"])
    logger.info("  FPR:               %.6f", test_metrics["fpr"])
    logger.info("  FNR:               %.6f", test_metrics["fnr"])
    logger.info("  Total transactions: %d", test_metrics["total_transactions"])
    logger.info("  Fraudulent:         %d", test_metrics["fraudulent"])
    logger.info("  Legitimate:         %d", test_metrics["legitimate"])
    logger.info("  Fraud prevalence:   %.4f%%", 100 * test_metrics["fraud_prevalence"])
    logger.info("")
    logger.info("Plots:")
    for name, path in plot_paths.items():
        logger.info("  %s → %s", name, path)
    logger.info("")
    logger.info("Artifacts: %s", cfg.ARTIFACTS_DIR)
    logger.info("Results:   %s", results_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
