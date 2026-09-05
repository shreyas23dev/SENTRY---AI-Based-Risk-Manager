"""
evaluate_temporal.py — Phase 2 Final Evaluation on Held-Out Test Partition
===========================================================================

Usage:
    python scripts/evaluate_temporal.py

Evaluates:
  1. Global Stream Temporal Memory (Single continuous interleaved stream)
  2. Entity-Scoped Temporal Memory (Entity key: card_email = card1 + P_emaildomain)

Comparative evaluations against B0 (Frozen Baseline):
  - False Negative recovery
  - False Positive increase
  - Recall, Precision, F1, PR-AUC, ROC-AUC
  - Slow-burn sequence evaluation
  - Temporal throughput and latency

LEAKAGE / FREEZE SAFETY:
  - Consumes existing frozen A_t from results/test_predictions.csv directly.
  - Phase 1 baseline artifacts and model are NOT modified.
"""

import json
import logging
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgraph.baseline import config as base_cfg
from trustgraph.temporal import config as temp_cfg
from trustgraph.temporal.engine import TemporalRiskEngine, EntityTemporalRiskTracker
from trustgraph.temporal.evaluator import (
    evaluate_temporal_comparison,
    make_temporal_prediction,
    plot_temporal_sequence,
    plot_slow_burn_demonstration,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_temporal")


def main():
    t_start = time.perf_counter()
    logger.info("=" * 70)
    logger.info("TRUSTGRAPH Phase 2 — Final Temporal Risk Memory Evaluation")
    logger.info("=" * 70)

    # 1. Load frozen Phase 1 test predictions
    test_pred_path = temp_cfg.TEST_PREDICTIONS_CSV
    if not test_pred_path.exists():
        raise FileNotFoundError(f"Missing {test_pred_path}. Run Phase 1 baseline first.")

    logger.info("Loading frozen Phase 1 predictions from: %s", test_pred_path)
    df_test = pd.read_csv(test_pred_path)
    logger.info("Loaded %d rows. Columns: %s", len(df_test), list(df_test.columns))

    # Join card1 and P_emaildomain for entity tracking
    raw_txn = pd.read_csv(base_cfg.TRAIN_TRANSACTION_CSV, usecols=["TransactionID", "card1", "P_emaildomain"])
    df_test = df_test.merge(raw_txn, on="TransactionID", how="left")
    del raw_txn

    # Ensure chronological sorting by TransactionDT
    if not df_test["TransactionDT"].is_monotonic_increasing:
        logger.info("Sorting test predictions chronologically by TransactionDT...")
        df_test = df_test.sort_values("TransactionDT").reset_index(drop=True)

    y_test   = df_test["isFraud"].values
    A_test   = df_test["A_t"].values
    base_pred = df_test["baseline_prediction"].values

    # Load frozen baseline threshold
    with open(base_cfg.ARTIFACTS_DIR / "threshold.json") as f:
        base_thr_data = json.load(f)
    base_thr = float(base_thr_data["threshold"])

    # ------------------------------------------------------------------
    # MODE 1: Global Stream Evaluation
    # ------------------------------------------------------------------
    with open(temp_cfg.TEMPORAL_ARTIFACTS_DIR / "parameters.json") as f:
        global_params_data = json.load(f)
    g_params = global_params_data["selected_parameters"]
    g_beta     = float(g_params["beta"])
    g_gamma    = float(g_params["gamma"])
    g_lambda   = float(g_params["lambda"])
    g_delta    = float(g_params["delta"])
    g_temp_thr = float(g_params["temporal_threshold"])

    logger.info("\n--- Evaluating Global Stream Temporal Memory ---")
    g_engine = TemporalRiskEngine(beta=g_beta, gamma=g_gamma, lambda_=g_lambda, delta=g_delta)
    t_g_start = time.perf_counter()
    E_global, P_global = g_engine.process_stream(A_test)
    t_g_elapsed = time.perf_counter() - t_g_start
    g_throughput = len(A_test) / t_g_elapsed
    g_latency_ms = (t_g_elapsed / len(A_test)) * 1000

    metrics_global = evaluate_temporal_comparison(
        y_test, A_test, E_global, P_global,
        baseline_threshold=base_thr,
        temporal_threshold=g_temp_thr,
        partition_name="test_global",
    )

    # ------------------------------------------------------------------
    # MODE 2: Entity-Scoped Evaluation (card_email = card1 + P_emaildomain)
    # ------------------------------------------------------------------
    entity_params_path = temp_cfg.TEMPORAL_ARTIFACTS_DIR / "entity_parameters.json"
    if entity_params_path.exists():
        with open(entity_params_path) as f:
            ent_params_data = json.load(f)
        e_params = ent_params_data["selected_parameters"]
    else:
        e_params = {"entity_key": "card_email", "beta": 0.5, "gamma": 0.5, "lambda": 0.05, "delta": 0.05, "temporal_threshold": 0.6}

    e_beta     = float(e_params["beta"])
    e_gamma    = float(e_params["gamma"])
    e_lambda   = float(e_params["lambda"])
    e_delta    = float(e_params["delta"])
    e_temp_thr = float(e_params["temporal_threshold"])

    logger.info("\n--- Evaluating Entity-Scoped Temporal Memory (Key: card_email) ---")
    df_test["card_email"] = df_test["card1"].astype(str) + "_" + df_test["P_emaildomain"].fillna("missing").astype(str)
    
    ent_tracker = EntityTemporalRiskTracker(beta=e_beta, gamma=e_gamma, lambda_=e_lambda, delta=e_delta)
    t_e_start = time.perf_counter()
    E_entity, P_entity = ent_tracker.process_dataframe(df_test, entity_col="card_email", score_col="A_t")
    t_e_elapsed = time.perf_counter() - t_e_start
    e_throughput = len(A_test) / t_e_elapsed
    e_latency_ms = (t_e_elapsed / len(A_test)) * 1000

    metrics_entity = evaluate_temporal_comparison(
        y_test, A_test, E_entity, P_entity,
        baseline_threshold=base_thr,
        temporal_threshold=e_temp_thr,
        partition_name="test_entity_card_email",
    )

    # Logging comparisons
    b0 = metrics_global["B0_baseline"]
    b1_g = metrics_global["B1_temporal"]
    d_g = metrics_global["comparative_delta"]

    b1_e = metrics_entity["B1_temporal"]
    d_e = metrics_entity["comparative_delta"]

    logger.info("=" * 75)
    logger.info("COMPARATIVE TEST METRICS SUMMARY:")
    logger.info("Metric                  B0 (Baseline)    B1 (Global Stream)    B1 (Entity Stream)")
    logger.info("-" * 75)
    logger.info("ROC-AUC                 %10.6f       %10.6f            %10.6f", b0["roc_auc"], b1_g["roc_auc"], b1_e["roc_auc"])
    logger.info("PR-AUC                  %10.6f       %10.6f            %10.6f", b0["pr_auc"], b1_g["pr_auc"], b1_e["pr_auc"])
    logger.info("Precision               %10.6f       %10.6f            %10.6f", b0["precision"], b1_g["precision"], b1_e["precision"])
    logger.info("Recall                  %10.6f       %10.6f            %10.6f", b0["recall"], b1_g["recall"], b1_e["recall"])
    logger.info("F1 Score                %10.6f       %10.6f            %10.6f", b0["f1"], b1_g["f1"], b1_e["f1"])
    logger.info("FPR                     %10.6f       %10.6f            %10.6f", b0["fpr"], b1_g["fpr"], b1_e["fpr"])
    logger.info("-" * 75)
    logger.info("Frauds Recovered        %10s       %10d            %10d", "-", d_g["additional_frauds_recovered"], d_e["additional_frauds_recovered"])
    logger.info("Additional False Pos    %10s       %10d            %10d", "-", d_g["additional_false_positives"], d_e["additional_false_positives"])
    logger.info("=" * 75)

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------
    logger.info("\nGenerating Phase 2 visualizations...")
    temp_cfg.TEMPORAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Plot 1: Representative sequence
    fraud_indices = np.where(y_test == 1)[0]
    sample_start = int(fraud_indices[20]) - 10
    sample_end   = sample_start + 60
    sample_slice = slice(sample_start, sample_end)

    plot_temporal_sequence(
        A_test[sample_slice], E_global[sample_slice], P_global[sample_slice], y_test[sample_slice],
        base_thr, g_temp_thr,
        temp_cfg.TEMPORAL_PLOTS_DIR / "temporal_sequence_trace.png",
        title="Representative Test Sequence (A_t, E_t, P_t Dynamics)"
    )

    # Plot 2: Controlled Slow-Burn demonstration
    slow_burn_results = plot_slow_burn_demonstration(
        gamma=g_gamma, lambda_=g_lambda, delta=g_delta, beta=g_beta,
        baseline_thr=base_thr, temporal_thr=g_temp_thr,
        save_path=temp_cfg.TEMPORAL_PLOTS_DIR / "slow_burn_demonstration.png",
    )

    # Plot 3: PR Curve Comparison
    prec_b0, rec_b0, _ = precision_recall_curve(y_test, A_test)
    prec_b1_g, rec_b1_g, _ = precision_recall_curve(y_test, np.maximum(A_test, P_global))
    prec_b1_e, rec_b1_e, _ = precision_recall_curve(y_test, np.maximum(A_test, P_entity))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec_b0, prec_b0, lw=2, color="#1f77b4", label=f"B0 Baseline (AP={b0['pr_auc']:.4f})")
    ax.plot(rec_b1_g, prec_b1_g, lw=1.5, color="#ff7f0e", linestyle="--", label=f"B1 Global Stream (AP={b1_g['pr_auc']:.4f})")
    ax.plot(rec_b1_e, prec_b1_e, lw=2, color="#2ca02c", linestyle="-.", label=f"B1 Entity Stream (AP={b1_e['pr_auc']:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    fig.savefig(temp_cfg.TEMPORAL_PLOTS_DIR / "pr_curve_comparison.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Save Results & Artifacts
    # ------------------------------------------------------------------
    # Output predictions CSV
    df_results = pd.DataFrame({
        "TransactionID":       df_test["TransactionID"],
        "TransactionDT":       df_test["TransactionDT"],
        "isFraud":             df_test["isFraud"],
        "A_t":                 df_test["A_t"],
        "E_t":                 E_global,
        "P_t":                 P_global,
        "E_t_entity":          E_entity,
        "P_t_entity":          P_entity,
        "baseline_prediction": base_pred,
        "temporal_prediction": make_temporal_prediction(A_test, P_global, base_thr, g_temp_thr),
        "entity_temporal_prediction": make_temporal_prediction(A_test, P_entity, base_thr, e_temp_thr),
    })
    df_results.to_csv(temp_cfg.TEMPORAL_PREDICTIONS_CSV, index=False)
    logger.info("Saved temporal predictions → %s (%d rows)", temp_cfg.TEMPORAL_PREDICTIONS_CSV, len(df_results))

    # Metrics JSON
    full_metrics = {
        "global_stream": {
            "metrics": metrics_global,
            "parameters": g_params,
            "timing": {
                "total_seconds": round(t_g_elapsed, 4),
                "throughput_txn_per_s": round(g_throughput, 1),
                "latency_ms_per_txn": round(g_latency_ms, 6),
            }
        },
        "entity_stream_card_email": {
            "metrics": metrics_entity,
            "parameters": e_params,
            "timing": {
                "total_seconds": round(t_e_elapsed, 4),
                "throughput_txn_per_s": round(e_throughput, 1),
                "latency_ms_per_txn": round(e_latency_ms, 6),
            }
        },
        "slow_burn_results": slow_burn_results,
    }
    with open(temp_cfg.TEMPORAL_ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)

    logger.info("Artifacts saved to: %s", temp_cfg.TEMPORAL_ARTIFACTS_DIR)
    logger.info("Phase 2 evaluation completed in %.2f s", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
