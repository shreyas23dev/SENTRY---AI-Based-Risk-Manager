"""
benchmark_runtime.py — TRUSTGRAPH Phase 5A Reproducible Runtime Benchmark
==========================================================================

Measures component-level and end-to-end single-transaction latency on a
deterministic 5,000-transaction sample from the frozen TEST partition.

Produces:
    artifacts/performance/before.json  (first run, before optimization)
    artifacts/performance/after.json   (second run, after optimization)
    artifacts/performance/comparison.json (delta table)

Usage:
    python scripts/benchmark_runtime.py --phase before
    python scripts/benchmark_runtime.py --phase after
    python scripts/benchmark_runtime.py --phase compare
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from trustgraph.baseline.data_loader import load_train_data, chronological_split
from trustgraph.baseline.model import BaselineModel
from trustgraph.baseline.preprocessing import BaselinePreprocessor
from trustgraph.temporal.entity_tracker import resolve_entity_key, EntityTemporalRiskEngine
from trustgraph.relational.graph_engine import (
    GraphParameters, LightweightRelationalGraph, process_partition,
)
from trustgraph.fusion.config import BASELINE_THRESHOLD, ENTITY_KEY_TYPE, PROJECT_ROOT
from trustgraph.fusion.fusion_engine import apply_fusion_rule
from trustgraph.policy.config import PolicyAction
from trustgraph.policy.decision_engine import PolicyThresholds, batch_assign_actions

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmark_runtime")

PERF_DIR = PROJECT_ROOT / "artifacts" / "performance"
PERF_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SIZE = 5000
WARMUP_RUNS = 50
SEED = 42
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
BASELINE_MODEL_PATH = ARTIFACTS_ROOT / "baseline" / "model" / "lgbm_model.pkl"
BASELINE_PREP_PATH  = ARTIFACTS_ROOT / "baseline" / "preprocessing"
FROZEN_THRESHOLDS_PATH = ARTIFACTS_ROOT / "policy" / "thresholds.json"


def load_frozen_thresholds() -> PolicyThresholds:
    with open(FROZEN_THRESHOLDS_PATH) as f:
        data = json.load(f)
    return PolicyThresholds(tau_verify=data["tau_verify"],
                            tau_throttle=data["tau_throttle"],
                            tau_block=data["tau_block"])


def build_pipeline_state():
    """Load all frozen artifacts ONCE — not counted in per-transaction timings."""
    print("[benchmark] Loading frozen artifacts (not included in per-transaction timing)...")

    raw_df, _ = load_train_data()
    train_df, val_df, test_df, _ = chronological_split(raw_df)
    del raw_df

    # Entity proxy
    for part in [train_df, val_df, test_df]:
        part["entity_proxy"] = resolve_entity_key(part, key_type=ENTITY_KEY_TYPE)

    # Model + preprocessor
    model = BaselineModel.load(BASELINE_MODEL_PATH)
    preprocessor = BaselinePreprocessor.load(BASELINE_PREP_PATH)

    # Temporal engine — process TRAIN and VAL to reach TEST state
    temp_engine = EntityTemporalRiskEngine(beta=0.3, gamma=0.5, lambda_=0.05, delta=0.05)
    for part in [train_df, val_df]:
        X_part = preprocessor.transform(part)
        part["A_t"] = model.predict_risk(X_part)
        ents = part["entity_proxy"].values
        scores = part["A_t"].values
        for i in range(len(part)):
            temp_engine.step(str(ents[i]), float(scores[i]))

    # Graph engine — process TRAIN and VAL
    rel_params = GraphParameters(k_attr_max=25, window_sec=86400.0, d_ref=3.0, v_ref=10.0,
                                 w_D=0.6, w_V=0.4, relational_attrs=("DeviceInfo",))
    graph_engine = LightweightRelationalGraph(rel_params)
    graph_engine.fit_attribute_frequency_ceiling(train_df)
    process_partition(train_df, graph_engine)
    process_partition(val_df, graph_engine)

    # Compute TEST A_t for the full sample pool
    X_test = preprocessor.transform(test_df)
    test_df["A_t"] = model.predict_risk(X_test)

    thresholds = load_frozen_thresholds()

    print(f"[benchmark] Pipeline state ready. TEST size: {len(test_df):,}")
    return model, preprocessor, temp_engine, graph_engine, thresholds, test_df, train_df


def stats(arr: np.ndarray) -> dict:
    return {
        "mean_ms": round(float(np.mean(arr)) * 1000, 4),
        "p50_ms": round(float(np.percentile(arr, 50)) * 1000, 4),
        "p95_ms": round(float(np.percentile(arr, 95)) * 1000, 4),
        "p99_ms": round(float(np.percentile(arr, 99)) * 1000, 4),
        "max_ms": round(float(np.max(arr)) * 1000, 4),
        "min_ms": round(float(np.min(arr)) * 1000, 4),
    }


def run_benchmark(phase: str):
    model, preprocessor, temp_engine, graph_engine, thresholds, test_df, train_df = build_pipeline_state()

    # Phase 5A: wrap the frozen preprocessor with the fast single-row path
    from trustgraph.runtime.fast_preprocessor import FastPreprocessor
    fast_prep = FastPreprocessor(preprocessor)

    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(test_df), size=min(SAMPLE_SIZE, len(test_df)), replace=False)
    sample_df = test_df.iloc[sample_idx].reset_index(drop=True)
    attrs = ["DeviceInfo"]

    # Pre-extract rows as plain dicts (outside timing)
    # The fast path consumes plain dicts — no DataFrame needed at score time
    row_dicts = [sample_df.iloc[i].to_dict() for i in range(len(sample_df))]
    rows = [sample_df.iloc[i] for i in range(len(sample_df))]

    # Warm-up (not measured)
    print(f"[benchmark] Warming up {WARMUP_RUNS} transactions...")
    for i in range(min(WARMUP_RUNS, len(rows))):
        row = rows[i]
        entity_id = str(row["entity_proxy"])
        timestamp = float(row["TransactionDT"])
        txn_id = int(row["TransactionID"])
        A_t = float(row["A_t"])
        attr_dict = {a: (str(row[a]) if pd.notna(row[a]) else None) for a in attrs}

        # Use fast single-row path in warm-up
        x_fast = fast_prep.transform_single_row(row_dicts[i])
        _ = model.predict_risk(x_fast)
        _, P_t = temp_engine.step(entity_id, A_t)
        rec = graph_engine.score(entity_id, timestamp, txn_id, attr_dict)
        G_t = rec.G_t
        R_t = float(np.clip(A_t + 1.0 * P_t + 0.05 * G_t, 0.0, 1.0))
        graph_engine.update(entity_id, timestamp, attr_dict)

    # Reset engine state and rebuild to test state for actual measurement
    print(f"[benchmark] Running {len(rows)} timed measurements...")

    t_preproc = np.empty(len(rows))
    t_model = np.empty(len(rows))
    t_temporal = np.empty(len(rows))
    t_relational = np.empty(len(rows))
    t_fusion = np.empty(len(rows))
    t_policy = np.empty(len(rows))
    t_total = np.empty(len(rows))

    # Re-initialize engines to pre-test state (required for correct causal order)
    temp_engine2 = EntityTemporalRiskEngine(beta=0.3, gamma=0.5, lambda_=0.05, delta=0.05)
    for part in [train_df]:
        X_part = preprocessor.transform(part)
        part_A = model.predict_risk(X_part)
        ents = part["entity_proxy"].values
        for ii in range(len(part)):
            temp_engine2.step(str(ents[ii]), float(part_A[ii]))

    graph_engine2 = LightweightRelationalGraph(
        GraphParameters(k_attr_max=25, window_sec=86400.0, d_ref=3.0, v_ref=10.0,
                        w_D=0.6, w_V=0.4, relational_attrs=("DeviceInfo",)))
    graph_engine2.fit_attribute_frequency_ceiling(train_df)
    process_partition(train_df, graph_engine2)

    for i, row in enumerate(rows):
        entity_id = str(row["entity_proxy"])
        timestamp = float(row["TransactionDT"])
        txn_id = int(row["TransactionID"])
        attr_dict = {a: (str(row[a]) if pd.notna(row[a]) else None) for a in attrs}
        raw_dict = row_dicts[i]

        # A. Preprocessing — FAST SINGLE-ROW PATH (no DataFrame)
        t_a = time.perf_counter()
        X_row = fast_prep.transform_single_row(raw_dict)
        t_b = time.perf_counter()

        # B. Model inference
        A_t = float(model.predict_risk(X_row)[0])
        t_c = time.perf_counter()

        # C. Temporal
        _, P_t = temp_engine2.step(entity_id, A_t)
        t_d = time.perf_counter()

        # D. Relational
        rec = graph_engine2.score(entity_id, timestamp, txn_id, attr_dict)
        G_t = rec.G_t
        t_e = time.perf_counter()
        graph_engine2.update(entity_id, timestamp, attr_dict)

        # E. Fusion
        R_t = float(np.clip(A_t + 1.0 * P_t + 0.05 * G_t, 0.0, 1.0))
        t_f = time.perf_counter()

        # F. Policy
        action = PolicyAction.ALLOW.value
        if R_t >= thresholds.tau_block:
            action = PolicyAction.BLOCK.value
        elif R_t >= thresholds.tau_throttle:
            action = PolicyAction.THROTTLE.value
        elif R_t >= thresholds.tau_verify:
            action = PolicyAction.VERIFY.value
        t_g = time.perf_counter()

        t_preproc[i]   = t_b - t_a
        t_model[i]     = t_c - t_b
        t_temporal[i]  = t_d - t_c
        t_relational[i]= t_e - t_d
        t_fusion[i]    = t_f - t_e
        t_policy[i]    = t_g - t_f
        t_total[i]     = t_g - t_a

    # Batch throughput
    print("[benchmark] Measuring batch throughput...")
    t_batch_start = time.perf_counter()
    X_batch = preprocessor.transform(sample_df)
    A_batch = model.predict_risk(X_batch)
    t_batch_end = time.perf_counter()
    batch_elapsed = t_batch_end - t_batch_start
    batch_throughput = round(len(sample_df) / batch_elapsed)

    result = {
        "phase": phase,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_size": len(rows),
        "warmup_runs": WARMUP_RUNS,
        "seed": SEED,
        "components": {
            "preprocessing":   stats(t_preproc),
            "model_inference": stats(t_model),
            "temporal":        stats(t_temporal),
            "relational_graph":stats(t_relational),
            "fusion":          stats(t_fusion),
            "policy":          stats(t_policy),
            "end_to_end":      stats(t_total),
        },
        "batch_throughput_txns_per_sec": batch_throughput,
        "notes": {
            "timing_scope": "Single-row dict preprocessing (FastPreprocessor) + model + temporal + graph + fusion + policy",
            "not_included": "Dataset loading, chronological splitting, graph construction from TRAIN",
            "preprocessing_path": "FastPreprocessor.transform_single_row(dict) — zero DataFrame, numpy only",
        }
    }

    out_path = PERF_DIR / f"{phase}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[benchmark] Saved -> {out_path}")

    # Print summary table
    print(f"\n{'='*65}")
    print(f"  TRUSTGRAPH Runtime Benchmark — Phase: {phase.upper()}")
    print(f"  Sample: {len(rows):,} transactions | Warmup: {WARMUP_RUNS}")
    print(f"{'='*65}")
    print(f"  {'Component':<22} {'p50 (ms)':>10} {'p95 (ms)':>10} {'p99 (ms)':>10} {'mean (ms)':>10}")
    print(f"  {'-'*62}")
    for comp, s in result["components"].items():
        print(f"  {comp:<22} {s['p50_ms']:>10.3f} {s['p95_ms']:>10.3f} {s['p99_ms']:>10.3f} {s['mean_ms']:>10.3f}")
    print(f"{'='*65}")
    print(f"  Batch throughput (preproc+model): {batch_throughput:,} txns/sec")
    print(f"{'='*65}\n")

    return result


def run_comparison():
    before_path = PERF_DIR / "before.json"
    after_path = PERF_DIR / "after.json"

    if not before_path.exists() or not after_path.exists():
        print("[benchmark] Both before.json and after.json must exist to compare.")
        return

    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    comparison = {"components": {}}
    for comp in before["components"]:
        b = before["components"][comp]
        a = after["components"].get(comp, {})
        comparison["components"][comp] = {}
        for metric in ["mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"]:
            b_val = b.get(metric, 0)
            a_val = a.get(metric, 0)
            pct = round((b_val - a_val) / b_val * 100, 1) if b_val != 0 else 0.0
            comparison["components"][comp][metric] = {
                "before": b_val,
                "after": a_val,
                "improvement_pct": pct,
                "direction": "improved" if pct > 0 else ("regressed" if pct < 0 else "unchanged"),
            }

    b_tp = before.get("batch_throughput_txns_per_sec", 0)
    a_tp = after.get("batch_throughput_txns_per_sec", 0)
    comparison["batch_throughput"] = {
        "before": b_tp,
        "after": a_tp,
        "improvement_pct": round((a_tp - b_tp) / b_tp * 100, 1) if b_tp else 0.0,
    }

    out_path = PERF_DIR / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"[benchmark] Saved -> {out_path}")

    print(f"\n{'='*70}")
    print("  TRUSTGRAPH Before/After Comparison")
    print(f"{'='*70}")
    print(f"  {'Component':<22} {'Before p50':>12} {'After p50':>10} {'Improvement':>12}")
    print(f"  {'-'*56}")
    for comp, metrics in comparison["components"].items():
        b50 = metrics["p50_ms"]["before"]
        a50 = metrics["p50_ms"]["after"]
        imp = metrics["p50_ms"]["improvement_pct"]
        print(f"  {comp:<22} {b50:>12.3f}ms {a50:>10.3f}ms {imp:>+11.1f}%")
    print(f"{'='*70}")
    b_tp = comparison["batch_throughput"]["before"]
    a_tp = comparison["batch_throughput"]["after"]
    tp_imp = comparison["batch_throughput"]["improvement_pct"]
    print(f"  Batch throughput: {b_tp:,} -> {a_tp:,} txns/sec ({tp_imp:+.1f}%)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["before", "after", "compare"], default="before")
    args = parser.parse_args()

    if args.phase == "compare":
        run_comparison()
    else:
        run_benchmark(args.phase)
