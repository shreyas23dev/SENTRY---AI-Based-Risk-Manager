"""
benchmark_inference.py — Inference Performance Benchmark Breakdown
=====================================================================

Measures inference latency and throughput across multiple granularities:
  1. LightGBM model-only batch inference (88,580 transactions)
  2. Preprocessing-only batch transformation (88,580 transactions)
  3. Preprocessing + Model batch inference (88,580 transactions)
  4. Single-transaction online latency (1 transaction at a time)

Does NOT modify any frozen Phase 1 artifacts.
"""

import gc
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgraph.baseline import config as cfg
from trustgraph.baseline.data_loader import chronological_split, load_train_data
from trustgraph.baseline.model import BaselineModel
from trustgraph.baseline.preprocessing import BaselinePreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


def main():
    logger.info("=" * 70)
    logger.info("TRUSTGRAPH — Inference Performance Detailed Benchmark")
    logger.info("=" * 70)

    # Load frozen artifacts
    model = BaselineModel.load(cfg.MODEL_DIR / "lgbm_model.pkl")
    preprocessor = BaselinePreprocessor.load(cfg.PREPROCESSING_DIR)

    # Reconstruct test dataframe
    df, _ = load_train_data()
    _, _, test_df, _ = chronological_split(df)
    del df
    gc.collect()

    n_txns = len(test_df)
    logger.info("Test dataset size: %d transactions", n_txns)

    # -------------------------------------------------------------
    # 1. Preprocessed batch ready
    # -------------------------------------------------------------
    X_test = preprocessor.transform(test_df)

    # Benchmark 1: Model-only batch inference
    logger.info("\n--- Benchmark 1: Model-Only Batch Inference ---")
    for _ in range(2):  # Warmup
        _ = model.predict_risk(X_test)

    model_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = model.predict_risk(X_test)
        model_times.append(time.perf_counter() - t0)

    b1_mean = float(np.mean(model_times))
    b1_per_txn_ms = (b1_mean / n_txns) * 1000
    b1_throughput = n_txns / b1_mean
    logger.info("Mean total: %.4f s (std: %.4f s)", b1_mean, float(np.std(model_times)))
    logger.info("Per-transaction: %.6f ms", b1_per_txn_ms)
    logger.info("Throughput: %.1f txn/s", b1_throughput)

    # Benchmark 2: Preprocessing-only batch
    logger.info("\n--- Benchmark 2: Preprocessing-Only Batch ---")
    for _ in range(2):  # Warmup
        _ = preprocessor.transform(test_df)

    prep_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = preprocessor.transform(test_df)
        prep_times.append(time.perf_counter() - t0)

    b2_mean = float(np.mean(prep_times))
    b2_per_txn_ms = (b2_mean / n_txns) * 1000
    b2_throughput = n_txns / b2_mean
    logger.info("Mean total: %.4f s (std: %.4f s)", b2_mean, float(np.std(prep_times)))
    logger.info("Per-transaction: %.6f ms", b2_per_txn_ms)
    logger.info("Throughput: %.1f txn/s", b2_throughput)

    # Benchmark 3: End-to-end Preprocessing + Model Batch
    logger.info("\n--- Benchmark 3: Preprocessing + Model Batch ---")
    b3_mean = b1_mean + b2_mean
    b3_per_txn_ms = (b3_mean / n_txns) * 1000
    b3_throughput = n_txns / b3_mean
    logger.info("Mean total: %.4f s", b3_mean)
    logger.info("Per-transaction: %.6f ms", b3_per_txn_ms)
    logger.info("Throughput: %.1f txn/s", b3_throughput)

    # Benchmark 4: Single-transaction Online Latency (N=1000 samples)
    logger.info("\n--- Benchmark 4: Single-Transaction Online Latency (N=1000) ---")
    sample_rows = [test_df.iloc[[i]] for i in range(1000)]
    
    # Warmup
    for r in sample_rows[:50]:
        x_r = preprocessor.transform(r)
        _ = model.predict_risk(x_r)

    single_model_times = []
    single_e2e_times = []

    for r in sample_rows:
        t0 = time.perf_counter()
        x_r = preprocessor.transform(r)
        t_prep_done = time.perf_counter()
        _ = model.predict_risk(x_r)
        t_all_done = time.perf_counter()

        single_model_times.append((t_all_done - t_prep_done) * 1000)
        single_e2e_times.append((t_all_done - t0) * 1000)

    logger.info("Single-txn Model-only Latency (p50): %.4f ms", float(np.median(single_model_times)))
    logger.info("Single-txn Model-only Latency (p99): %.4f ms", float(np.percentile(single_model_times, 99)))
    logger.info("Single-txn End-to-End Latency (p50): %.4f ms", float(np.median(single_e2e_times)))
    logger.info("Single-txn End-to-End Latency (p99): %.4f ms", float(np.percentile(single_e2e_times, 99)))

    results = {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "threads": "n_jobs=-1 (all cores)",
        },
        "benchmark_1_model_only_batch": {
            "n_transactions": n_txns,
            "mean_total_s": round(b1_mean, 4),
            "latency_ms_per_txn": round(b1_per_txn_ms, 6),
            "throughput_txn_per_s": round(b1_throughput, 1),
            "scope": "LightGBM predict_proba on in-memory preprocessed DataFrame",
        },
        "benchmark_2_preprocessing_only_batch": {
            "n_transactions": n_txns,
            "mean_total_s": round(b2_mean, 4),
            "latency_ms_per_txn": round(b2_per_txn_ms, 6),
            "throughput_txn_per_s": round(b2_throughput, 1),
            "scope": "Categorical mapping, column reindexing, dtype casting",
        },
        "benchmark_3_preprocessing_plus_model_batch": {
            "n_transactions": n_txns,
            "mean_total_s": round(b3_mean, 4),
            "latency_ms_per_txn": round(b3_per_txn_ms, 6),
            "throughput_txn_per_s": round(b3_throughput, 1),
            "scope": "Batch Preprocessing + Model Inference (in-memory DataFrame)",
        },
        "benchmark_4_single_transaction_online": {
            "samples_evaluated": 1000,
            "model_only_latency_ms": {
                "mean": round(float(np.mean(single_model_times)), 4),
                "p50": round(float(np.median(single_model_times)), 4),
                "p95": round(float(np.percentile(single_model_times, 95)), 4),
                "p99": round(float(np.percentile(single_model_times, 99)), 4),
            },
            "end_to_end_latency_ms": {
                "mean": round(float(np.mean(single_e2e_times)), 4),
                "p50": round(float(np.median(single_e2e_times)), 4),
                "p95": round(float(np.percentile(single_e2e_times, 95)), 4),
                "p99": round(float(np.percentile(single_e2e_times, 99)), 4),
            },
            "scope": "Per-transaction online evaluation (1 row at a time)",
        }
    }

    out_path = cfg.ARTIFACTS_DIR / "inference_benchmark_breakdown.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nBenchmark breakdown saved → %s", out_path)


if __name__ == "__main__":
    main()
