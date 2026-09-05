"""
benchmark_relational.py — TRUSTGRAPH Phase 3 Graph Layer Performance Benchmark
================================================================================

Measures:
  - Batch throughput: transactions/sec for process_partition (TRAIN + VAL + TEST)
  - Online latency: per-transaction score() + update() latency distribution
    (p50, p95, p99, max in ms)

Uses frozen parameters from artifacts/relational/parameters.json.
Does NOT re-evaluate fraud metrics. Pure timing only.
"""

import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd

from trustgraph.baseline.data_loader import load_train_data, chronological_split
from trustgraph.temporal.entity_tracker import resolve_entity_key
from trustgraph.relational.config import RELATIONAL_DIR, ENTITY_KEY_TYPE
from trustgraph.relational.graph_engine import (
    GraphParameters, LightweightRelationalGraph,
    process_partition, build_attr_dict,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_frozen_params() -> dict:
    p = RELATIONAL_DIR / "parameters.json"
    if not p.exists():
        raise FileNotFoundError(f"Frozen parameters not found: {p}")
    with open(p) as f:
        return json.load(f)


def make_engine(frozen_params: dict) -> LightweightRelationalGraph:
    params = GraphParameters(
        k_attr_max=frozen_params["k_attr_max"],
        window_sec=frozen_params["window_sec"],
        d_ref=frozen_params["d_ref"],
        v_ref=frozen_params["v_ref"],
        w_D=frozen_params["w_D"],
        w_V=frozen_params["w_V"],
        relational_attrs=tuple(frozen_params["relational_attrs"]),
    )
    return LightweightRelationalGraph(params)


if __name__ == "__main__":
    frozen_params = load_frozen_params()
    logger.info("Loaded frozen params: %s", frozen_params)

    logger.info("Loading dataset...")
    df, _ = load_train_data()
    train_df, val_df, test_df, _ = chronological_split(df)
    del df

    for part, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        part["entity_proxy"] = resolve_entity_key(part, key_type=ENTITY_KEY_TYPE)
        if "A_t" not in part.columns:
            part["A_t"] = 0.0

    # -----------------------------------------------------------------------
    # Batch throughput benchmark: process all three partitions in sequence
    # -----------------------------------------------------------------------
    engine = make_engine(frozen_params)
    engine.fit_attribute_frequency_ceiling(train_df)

    logger.info("Batch benchmark — TRAIN (%d rows)...", len(train_df))
    t0 = time.perf_counter()
    process_partition(train_df, engine)
    train_elapsed = time.perf_counter() - t0
    train_tps = len(train_df) / train_elapsed

    logger.info("Batch benchmark — VAL (%d rows)...", len(val_df))
    t0 = time.perf_counter()
    process_partition(val_df, engine)
    val_elapsed = time.perf_counter() - t0
    val_tps = len(val_df) / val_elapsed

    logger.info("Batch benchmark — TEST (%d rows)...", len(test_df))
    t0 = time.perf_counter()
    process_partition(test_df, engine)
    test_elapsed = time.perf_counter() - t0
    test_tps = len(test_df) / test_elapsed

    total_rows = len(train_df) + len(val_df) + len(test_df)
    total_elapsed = train_elapsed + val_elapsed + test_elapsed
    overall_tps = total_rows / total_elapsed

    logger.info("Batch throughput:")
    logger.info("  TRAIN: %.1f txn/s  (%.3f s for %d rows)", train_tps, train_elapsed, len(train_df))
    logger.info("  VAL:   %.1f txn/s  (%.3f s for %d rows)", val_tps,   val_elapsed,   len(val_df))
    logger.info("  TEST:  %.1f txn/s  (%.3f s for %d rows)", test_tps,  test_elapsed,  len(test_df))
    logger.info("  TOTAL: %.1f txn/s  (%.3f s for %d rows)", overall_tps, total_elapsed, total_rows)

    # -----------------------------------------------------------------------
    # Online latency benchmark: per-transaction score() + update() timing
    # on a sample of 2000 transactions from VAL
    # -----------------------------------------------------------------------
    SAMPLE = 2000
    sample_df = val_df.sample(n=min(SAMPLE, len(val_df)), random_state=42).reset_index(drop=True)

    # Fresh engine with train history only (representative deployment state)
    engine2 = make_engine(frozen_params)
    engine2.fit_attribute_frequency_ceiling(train_df)
    process_partition(train_df, engine2)

    attrs = list(frozen_params["relational_attrs"])
    latencies_ms = []

    for _, row in sample_df.iterrows():
        entity_id  = str(row["entity_proxy"])
        timestamp  = float(row["TransactionDT"])
        txn_id     = int(row["TransactionID"])
        attr_dict  = build_attr_dict(row, attrs)

        t_start = time.perf_counter()
        rec = engine2.score(entity_id, timestamp, txn_id, attr_dict)
        engine2.update(entity_id, timestamp, attr_dict)
        latencies_ms.append((time.perf_counter() - t_start) * 1000.0)

    lat = np.array(latencies_ms)
    logger.info("\nOnline latency (score + update, n=%d sample from VAL):", len(lat))
    logger.info("  Mean:  %.4f ms", lat.mean())
    logger.info("  p50:   %.4f ms", np.percentile(lat, 50))
    logger.info("  p95:   %.4f ms", np.percentile(lat, 95))
    logger.info("  p99:   %.4f ms", np.percentile(lat, 99))
    logger.info("  Max:   %.4f ms", lat.max())

    # Save benchmark results
    benchmark = {
        "batch": {
            "train": {"rows": len(train_df), "elapsed_sec": round(train_elapsed, 4), "txn_per_sec": round(train_tps, 1)},
            "val":   {"rows": len(val_df),   "elapsed_sec": round(val_elapsed,   4), "txn_per_sec": round(val_tps, 1)},
            "test":  {"rows": len(test_df),  "elapsed_sec": round(test_elapsed,  4), "txn_per_sec": round(test_tps, 1)},
            "total": {"rows": total_rows,    "elapsed_sec": round(total_elapsed, 4), "txn_per_sec": round(overall_tps, 1)},
        },
        "online_latency_ms": {
            "n_sample": int(len(lat)),
            "mean":  round(float(lat.mean()), 4),
            "p50":   round(float(np.percentile(lat, 50)), 4),
            "p95":   round(float(np.percentile(lat, 95)), 4),
            "p99":   round(float(np.percentile(lat, 99)), 4),
            "max":   round(float(lat.max()), 4),
        },
    }
    bench_path = RELATIONAL_DIR / "benchmark.json"
    with open(bench_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    logger.info("\nBenchmark saved to %s", bench_path)
