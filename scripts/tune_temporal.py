"""
tune_temporal.py — Fast Validation-Only Parameter Tuning for Temporal Risk Memory
==================================================================================

Usage:
    python scripts/tune_temporal.py

Tuning methodology:
  1. Load frozen baseline model & preprocessor
  2. Extract VALIDATION partition (88,581 rows) and generate A_t^{val}
  3. Load frozen baseline threshold (0.594298)
  4. Perform controlled grid search on VALIDATION ONLY for:
       β  (EMA smoothing)
       γ  (Suspicion threshold)
       λ  (Upward accumulation step)
       δ  (Downward decay step)
       τ_temporal (Temporal decision threshold)
  5. Select parameters maximizing Validation F1 score on B1 = Baseline + Temporal
  6. Perform sensitivity analysis
  7. Save frozen temporal parameters to artifacts/temporal/parameters.json

LEAKAGE SAFETY:
  - Final TEST set is NEVER accessed during tuning.
"""

import gc
import json
import logging
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgraph.baseline import config as base_cfg
from trustgraph.baseline.data_loader import chronological_split, load_train_data
from trustgraph.baseline.model import BaselineModel
from trustgraph.baseline.preprocessing import BaselinePreprocessor
from trustgraph.temporal import config as temp_cfg
from trustgraph.temporal.engine import TemporalRiskEngine
from trustgraph.temporal.evaluator import evaluate_temporal_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tune_temporal")


def fast_stream_eval(A_val: np.ndarray, beta: float, gamma: float, lambda_: float, delta: float):
    """Fast stream execution loop in compiled/vectorized steps."""
    N = len(A_val)
    E_arr = np.empty(N, dtype=np.float64)
    P_arr = np.empty(N, dtype=np.float64)
    
    e_val = 0.0
    p_val = 0.0
    one_minus_beta = 1.0 - beta

    for i in range(N):
        a = float(A_val[i])
        e_val = beta * a + one_minus_beta * e_val
        E_arr[i] = e_val
        P_arr[i] = p_val
        
        # Next state
        if e_val > gamma:
            p_val = min(1.0, p_val + lambda_)
        else:
            p_val = max(0.0, p_val - delta)
            
    return E_arr, P_arr


def main():
    t_start = time.perf_counter()
    logger.info("=" * 70)
    logger.info("TRUSTGRAPH Phase 2 — Validation-Only Parameter Tuning")
    logger.info("=" * 70)

    # 1. Load frozen baseline artifacts
    logger.info("Loading frozen baseline model and preprocessor...")
    model = BaselineModel.load(base_cfg.MODEL_DIR / "lgbm_model.pkl")
    preprocessor = BaselinePreprocessor.load(base_cfg.PREPROCESSING_DIR)

    with open(base_cfg.ARTIFACTS_DIR / "threshold.json") as f:
        thr_data = json.load(f)
    base_thr = float(thr_data["threshold"])
    logger.info("Frozen baseline threshold: %.6f", base_thr)

    # 2. Reconstruct VALIDATION partition
    logger.info("Loading training data to extract VALIDATION partition...")
    df, join_stats = load_train_data()
    _, val_df, _, split_meta = chronological_split(df)
    del df
    gc.collect()

    logger.info("VALIDATION partition: %d rows (fraud rate: %.3f%%)", len(val_df), 100 * val_df["isFraud"].mean())
    y_val = (val_df["isFraud"].values == 1)
    n_fraud = int(np.sum(y_val))
    n_legit = int(len(y_val) - n_fraud)

    # Preprocess & score validation partition
    X_val = preprocessor.transform(val_df)
    A_val = model.predict_risk(X_val)
    del X_val, val_df
    gc.collect()

    # Precompute baseline predictions on validation
    base_flags = (A_val >= base_thr)
    base_tp = np.sum(base_flags & y_val)
    base_fp = np.sum(base_flags & (~y_val))
    base_fn = n_fraud - base_tp
    base_f1 = (2 * base_tp) / (2 * base_tp + base_fp + base_fn)
    logger.info("Baseline Validation F1: %.6f (TP=%d, FP=%d, FN=%d)", base_f1, base_tp, base_fp, base_fn)

    # 3. Validation Grid Search
    logger.info("Beginning fast grid search over candidate parameters on VALIDATION partition...")
    betas      = temp_cfg.CANDIDATE_BETAS
    gammas     = temp_cfg.CANDIDATE_GAMMAS
    lambdas    = temp_cfg.CANDIDATE_LAMBDAS
    deltas     = temp_cfg.CANDIDATE_DELTAS
    thresholds = temp_cfg.CANDIDATE_THRESHOLDS

    total_combos = len(betas) * len(gammas) * len(lambdas) * len(deltas)
    logger.info("Evaluating %d (beta, gamma, lambda, delta) combinations across %d thresholds...",
                total_combos, len(thresholds))

    best_f1 = -1.0
    best_params = {}
    search_records = []

    for beta, gamma, lambda_, delta in product(betas, gammas, lambdas, deltas):
        E_val, P_val = fast_stream_eval(A_val, beta, gamma, lambda_, delta)

        for thr_t in thresholds:
            temp_flags = (P_val >= thr_t)
            combined_flags = base_flags | temp_flags

            tp = int(np.sum(combined_flags & y_val))
            fp = int(np.sum(combined_flags & (~y_val)))
            fn = int(n_fraud - tp)

            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall    = float(tp / n_fraud) if n_fraud > 0 else 0.0
            f1        = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
            fpr       = float(fp / n_legit) if n_legit > 0 else 0.0

            fn_recovered = int(tp - base_tp)
            extra_fp     = int(fp - base_fp)

            search_records.append({
                "beta": beta, "gamma": gamma, "lambda": lambda_, "delta": delta,
                "temporal_threshold": thr_t,
                "f1": f1, "recall": recall, "precision": precision, "fpr": fpr,
                "frauds_recovered": fn_recovered, "extra_fp": extra_fp,
            })

            # Selection criterion: maximize F1 on validation partition
            if f1 > best_f1:
                best_f1 = f1
                best_params = {
                    "beta": beta,
                    "gamma": gamma,
                    "lambda": lambda_,
                    "delta": delta,
                    "temporal_threshold": thr_t,
                }

    # Run full comprehensive evaluation on the winning parameter set
    engine = TemporalRiskEngine(
        beta=best_params["beta"],
        gamma=best_params["gamma"],
        lambda_=best_params["lambda"],
        delta=best_params["delta"],
    )
    E_best, P_best = engine.process_stream(A_val)
    best_eval = evaluate_temporal_comparison(
        val_df_y := np.array(y_val, dtype=int),
        A_val, E_best, P_best,
        baseline_threshold=base_thr,
        temporal_threshold=best_params["temporal_threshold"],
        partition_name="validation",
    )

    logger.info("=" * 70)
    logger.info("TUNING RESULTS (Validation Set):")
    logger.info("Best Validation F1: %.6f (Baseline Val F1: %.6f)", best_f1, base_f1)
    logger.info("Selected Parameters: %s", best_params)
    logger.info("Val Frauds Recovered: %d (%.2f%% of baseline FNs)",
                best_eval["comparative_delta"]["additional_frauds_recovered"],
                best_eval["comparative_delta"]["pct_baseline_fn_recovered"])
    logger.info("Val Extra FPs: %d (FPR: %.4f -> %.4f)",
                best_eval["comparative_delta"]["additional_false_positives"],
                best_eval["B0_baseline"]["fpr"], best_eval["B1_temporal"]["fpr"])

    # 4. Parameter Sensitivity Summary
    df_search = pd.DataFrame(search_records)
    sensitivity = {
        "by_beta": df_search.groupby("beta")["f1"].agg(["mean", "max", "min"]).to_dict(),
        "by_gamma": df_search.groupby("gamma")["f1"].agg(["mean", "max", "min"]).to_dict(),
        "by_lambda": df_search.groupby("lambda")["f1"].agg(["mean", "max", "min"]).to_dict(),
        "by_delta": df_search.groupby("delta")["f1"].agg(["mean", "max", "min"]).to_dict(),
        "by_temporal_threshold": df_search.groupby("temporal_threshold")["f1"].agg(["mean", "max", "min"]).to_dict(),
    }

    # 5. Save Artifacts
    temp_cfg.TEMPORAL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_cfg.TEMPORAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    params_path = temp_cfg.TEMPORAL_ARTIFACTS_DIR / "parameters.json"
    with open(params_path, "w") as f:
        json.dump({
            "selected_parameters": best_params,
            "validation_metrics": best_eval,
            "selection_criterion": "maximise Validation F1 on B1 = Baseline + Temporal",
            "tuning_duration_seconds": round(time.perf_counter() - t_start, 2),
        }, f, indent=2)

    sens_path = temp_cfg.TEMPORAL_ARTIFACTS_DIR / "parameter_sensitivity.json"
    with open(sens_path, "w") as f:
        json.dump(sensitivity, f, indent=2)

    logger.info("Parameters and sensitivity saved to: %s", temp_cfg.TEMPORAL_ARTIFACTS_DIR)
    logger.info("Tuning completed in %.2f s", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
