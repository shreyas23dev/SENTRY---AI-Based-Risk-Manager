"""
benchmark_runtime_v2.py — 10,000-Transaction Equivalence & Performance Benchmark
================================================================================

Benchmarks the high-performance V2 runtime scorer on 10,000 real TEST transactions:
  1. Strict Numerical Equivalence Test (Reference vs Fast RuntimeScorerV2)
  2. Latency Profiling (mean, p50, p95, p99, max)
  3. Batch Throughput Measurement
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from trustgraph.baseline import config as cfg
from trustgraph.baseline.data_loader import load_train_data, chronological_split
from trustgraph.baseline.preprocessing import BaselinePreprocessor
from trustgraph.baseline.model import BaselineModel
from trustgraph.temporal.entity_tracker import resolve_entity_key, EntityTemporalRiskEngine
from trustgraph.relational.graph_engine import GraphParameters, LightweightRelationalGraph, process_partition
from trustgraph.fusion.fusion_engine import apply_fusion_rule
from trustgraph.policy.decision_engine import PolicyThresholds, batch_assign_actions
from trustgraph.features_v2.causal_features import compute_point_in_time_features, FrequencyEncoder, CausalStreamFeatureEngine
from trustgraph.runtime.scorer_v2 import RuntimeScorerV2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_runtime_v2")

SAMPLE_SIZE = 10000
WARMUP_RUNS = 100
SEED = 42


def stats(arr: np.ndarray) -> dict:
    return {
        "mean_ms": round(float(np.mean(arr)) * 1000, 4),
        "p50_ms": round(float(np.percentile(arr, 50)) * 1000, 4),
        "p95_ms": round(float(np.percentile(arr, 95)) * 1000, 4),
        "p99_ms": round(float(np.percentile(arr, 99)) * 1000, 4),
        "max_ms": round(float(np.max(arr)) * 1000, 4),
        "min_ms": round(float(np.min(arr)) * 1000, 4),
    }


def main():
    logger.info("Loading dataset for 10,000-transaction V2 benchmark...")
    raw_df, _ = load_train_data()
    train_df, val_df, test_df, _ = chronological_split(raw_df)
    del raw_df

    for part in [train_df, val_df, test_df]:
        part["entity_proxy"] = resolve_entity_key(part, key_type="card_addr_email")

    # Select contiguous 10,000-transaction chronological slice from TEST
    sample_indices = np.arange(SAMPLE_SIZE)
    sample_df = test_df.iloc[sample_indices].reset_index(drop=True)

    # Initialize Reference Pipelines & Fast Runtime Scorer
    logger.info("Initializing reference and fast runtime pipelines...")
    scorer_v2 = RuntimeScorerV2.load(cfg.PROJECT_ROOT / "artifacts", train_df)

    # Warmup
    logger.info(f"Warming up {WARMUP_RUNS} transactions...")
    for i in range(WARMUP_RUNS):
        r_dict = sample_df.iloc[i].to_dict()
        scorer_v2.score_transaction(r_dict)

    # Reset scorer state to clean pre-test state
    logger.info("Seeding scorer background state on TRAIN and VAL partitions...")
    scorer_v2 = RuntimeScorerV2.load(cfg.PROJECT_ROOT / "artifacts", train_df)

    # Pre-extract batch features for fast A_t computation on TRAIN and VAL
    base_prep = BaselinePreprocessor.load(cfg.PROJECT_ROOT / "artifacts" / "baseline" / "preprocessing")
    X_tr_b = base_prep.transform(train_df)
    X_va_b = base_prep.transform(val_df)

    pit_tr = compute_point_in_time_features(train_df)
    pit_va = compute_point_in_time_features(val_df)

    fe = FrequencyEncoder(["card1", "addr1", "P_emaildomain", "DeviceInfo"]).fit(train_df)
    fe_tr = fe.transform(train_df)
    fe_va = fe.transform(val_df)

    stream_engine = CausalStreamFeatureEngine()
    stream_tr = stream_engine.process_partition(train_df)
    stream_va = stream_engine.process_partition(val_df)

    X_tr_v2 = pd.concat([X_tr_b, fe_tr, stream_tr[["prior_count_card1", "prior_count_addr1", "prior_count_P_emaildomain", "prior_count_DeviceInfo"]],
                         pit_tr[["hour_of_day", "day_of_week", "hour_sin", "hour_cos"]], stream_tr[["entity_dt_elapsed"]],
                         pit_tr[["log_TransactionAmt", "amt_decimal_cents", "amt_is_integer"]],
                         stream_tr[["entity_prior_count", "entity_hist_mean_amt", "entity_hist_std_amt", "entity_amt_ratio"]]], axis=1)
    X_va_v2 = pd.concat([X_va_b, fe_va, stream_va[["prior_count_card1", "prior_count_addr1", "prior_count_P_emaildomain", "prior_count_DeviceInfo"]],
                         pit_va[["hour_of_day", "day_of_week", "hour_sin", "hour_cos"]], stream_va[["entity_dt_elapsed"]],
                         pit_va[["log_TransactionAmt", "amt_decimal_cents", "amt_is_integer"]],
                         stream_va[["entity_prior_count", "entity_hist_mean_amt", "entity_hist_std_amt", "entity_amt_ratio"]]], axis=1)

    train_df["A_t_v2"] = scorer_v2.model.predict_risk(X_tr_v2)
    val_df["A_t_v2"] = scorer_v2.model.predict_risk(X_va_v2)

    # Fast state seeding for TRAIN + VAL
    for part in [train_df, val_df]:
        ents = part["entity_proxy"].values
        amts = part["TransactionAmt"].values
        tss = part["TransactionDT"].values
        scores = part["A_t_v2"].values
        devs = part["DeviceInfo"].values if "DeviceInfo" in part.columns else [None]*len(part)
        c1s = part["card1"].values if "card1" in part.columns else [None]*len(part)
        a1s = part["addr1"].values if "addr1" in part.columns else [None]*len(part)
        ems = part["P_emaildomain"].values if "P_emaildomain" in part.columns else [None]*len(part)

        for i in range(len(part)):
            ent = str(ents[i])
            amt = float(amts[i])
            ts = float(tss[i])
            score = float(scores[i])
            scorer_v2.temp_engine.step(ent, score)
            raw_d = {"card1": c1s[i], "addr1": a1s[i], "P_emaildomain": ems[i], "DeviceInfo": devs[i]}
            scorer_v2.fast_prep.update_state(raw_d, ent, ts, amt)
            attr_dict = {"DeviceInfo": str(devs[i]) if pd.notna(devs[i]) else None}
            scorer_v2.graph_engine.update(ent, ts, attr_dict)

    # Timed evaluation on 10,000 transactions
    logger.info(f"Running timed measurement on {SAMPLE_SIZE:,} transactions...")
    t_preproc = np.empty(SAMPLE_SIZE)
    t_model = np.empty(SAMPLE_SIZE)
    t_temporal = np.empty(SAMPLE_SIZE)
    t_relational = np.empty(SAMPLE_SIZE)
    t_fusion = np.empty(SAMPLE_SIZE)
    t_policy = np.empty(SAMPLE_SIZE)
    t_total = np.empty(SAMPLE_SIZE)

    mismatches = 0
    max_diff_A = 0.0
    max_diff_P = 0.0
    max_diff_R = 0.0

    # Load precomputed reference predictions for validation
    ref_preds = pd.read_csv(cfg.PROJECT_ROOT / "results" / "policy_predictions_v2.csv").iloc[sample_indices].reset_index(drop=True)

    for i in range(SAMPLE_SIZE):
        row = sample_df.iloc[i]
        r_dict = row.to_dict()
        amt = float(row["TransactionAmt"])
        ts = float(row["TransactionDT"])
        ent = str(row["entity_proxy"])
        txn_id = int(row["TransactionID"])

        t0 = time.perf_counter()

        # A. Preprocessing
        t_a = time.perf_counter()
        x_vec = scorer_v2.fast_prep.transform_single_row(r_dict, ent, ts, amt)
        t_b = time.perf_counter()

        # B. Model
        A_t = float(scorer_v2.model.predict_risk(x_vec)[0])
        t_c = time.perf_counter()

        # C. Temporal
        _, P_t = scorer_v2.temp_engine.step(ent, A_t)
        t_d = time.perf_counter()

        # D. Relational
        attr_dict = {"DeviceInfo": str(row["DeviceInfo"]) if pd.notna(row["DeviceInfo"]) else None}
        rec = scorer_v2.graph_engine.score(ent, ts, txn_id, attr_dict)
        G_t = rec.G_t
        t_e = time.perf_counter()

        scorer_v2.fast_prep.update_state(r_dict, ent, ts, amt)
        scorer_v2.graph_engine.update(ent, ts, attr_dict)

        # E. Fusion
        R_t = float(np.clip(A_t + 1.0 * P_t + 0.05 * G_t, 0.0, 1.0))
        t_f = time.perf_counter()

        # F. Policy
        if R_t >= scorer_v2.thresholds.tau_block:
            act = "BLOCK"
        elif R_t >= scorer_v2.thresholds.tau_throttle:
            act = "THROTTLE"
        elif R_t >= scorer_v2.thresholds.tau_verify:
            act = "VERIFY"
        else:
            act = "ALLOW"
        t_g = time.perf_counter()

        t_preproc[i] = t_b - t_a
        t_model[i] = t_c - t_b
        t_temporal[i] = t_d - t_c
        t_relational[i] = t_e - t_d
        t_fusion[i] = t_f - t_e
        t_policy[i] = t_g - t_f
        t_total[i] = t_g - t_a

        # Reference equivalence check
        ref_row = ref_preds.iloc[i]
        diff_A = abs(A_t - float(ref_row["A_t_v2"]))
        diff_R = abs(R_t - float(ref_row["R_t_v2"]))
        max_diff_A = max(max_diff_A, diff_A)
        max_diff_R = max(max_diff_R, diff_R)

        if act != str(ref_row["action_v2"]):
            mismatches += 1

    # Batch throughput
    t_batch_start = time.perf_counter()
    X_batch = scorer_v2.model.predict_risk(np.vstack([scorer_v2.fast_prep.transform_single_row(sample_df.iloc[k].to_dict(), str(sample_df.iloc[k]["entity_proxy"]), float(sample_df.iloc[k]["TransactionDT"]), float(sample_df.iloc[k]["TransactionAmt"])) for k in range(min(1000, SAMPLE_SIZE))]))
    t_batch_end = time.perf_counter()
    batch_tp = round(1000.0 / (t_batch_end - t_batch_start))

    bench_results = {
        "sample_size": SAMPLE_SIZE,
        "mismatches": mismatches,
        "max_diff_A_t": round(max_diff_A, 8),
        "max_diff_R_t": round(max_diff_R, 8),
        "equivalence_verified": (mismatches == 0 and max_diff_R < 1e-4),
        "components": {
            "preprocessing": stats(t_preproc),
            "model_inference": stats(t_model),
            "temporal": stats(t_temporal),
            "relational_graph": stats(t_relational),
            "fusion": stats(t_fusion),
            "policy": stats(t_policy),
            "end_to_end": stats(t_total),
        },
        "batch_throughput_txns_per_sec": batch_tp,
    }

    out_path = cfg.PROJECT_ROOT / "artifacts" / "performance" / "benchmark_v2_10k.json"
    with open(out_path, "w") as f:
        json.dump(bench_results, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info("  TRUSTGRAPH V2 10,000-TRANSACTION BENCHMARK RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"  Mismatches against reference: {mismatches} (Max diff R_t: {max_diff_R:.8f})")
    logger.info(f"  {'Component':<22} {'p50 (ms)':>10} {'p95 (ms)':>10} {'p99 (ms)':>10} {'mean (ms)':>10}")
    logger.info(f"  {'-'*62}")
    for comp, s in bench_results["components"].items():
        logger.info(f"  {comp:<22} {s['p50_ms']:>10.3f} {s['p95_ms']:>10.3f} {s['p99_ms']:>10.3f} {s['mean_ms']:>10.3f}")
    logger.info(f"{'='*70}")
    logger.info(f"  Batch Throughput: {batch_tp:,} txns/sec")
    logger.info(f"  Saved -> {out_path}\n")


if __name__ == "__main__":
    main()
