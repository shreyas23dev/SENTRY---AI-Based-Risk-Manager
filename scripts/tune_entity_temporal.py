"""
tune_entity_temporal.py — Validation-Only Parameter Tuning for Entity-Scoped Temporal Memory
=============================================================================================

Usage:
    python scripts/tune_entity_temporal.py

Tuning Methodology:
  1. Load frozen baseline model & preprocessor
  2. Extract VALIDATION partition (88,581 rows) and generate A_t^{val}
  3. Load frozen baseline threshold (0.594298)
  4. Perform controlled grid search on VALIDATION ONLY across candidate entity keys:
       - 'card1'
       - 'card_composite' (card1-6)
       - 'card_email' (card1 + P_emaildomain)
       - 'card_addr' (card1 + addr1)
     and parameter grid:
       β ∈ {0.3, 0.4, 0.5, 0.6}
       γ ∈ {0.3, 0.4, 0.5}
       λ ∈ {0.02, 0.05, 0.10}
       δ ∈ {0.02, 0.05, 0.10}
       τ_temporal ∈ {0.4, 0.5, 0.6, 0.7}
  5. Select the optimal entity key and parameter set maximizing Validation F1.
  6. Save parameters to artifacts/temporal_entity/parameters.json and ablation table.

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
from trustgraph.temporal.entity_tracker import resolve_entity_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tune_entity_temporal")

OUT_DIR = base_cfg.PROJECT_ROOT / "artifacts" / "temporal_entity"


def fast_entity_stream(entities: np.ndarray, A_val: np.ndarray, beta: float, gamma: float, lambda_: float, delta: float):
    """Fast entity-scoped stream execution loop."""
    N = len(A_val)
    E_arr = np.empty(N, dtype=np.float64)
    P_arr = np.empty(N, dtype=np.float64)

    states = {}  # entity -> [E_state, P_state]
    one_minus_beta = 1.0 - beta

    for i in range(N):
        ent = entities[i]
        a = float(A_val[i])

        if ent in states:
            e_prev, p_prev = states[ent]
        else:
            e_prev, p_prev = 0.0, 0.0

        e_curr = beta * a + one_minus_beta * e_prev
        E_arr[i] = e_curr
        P_arr[i] = p_prev

        if e_curr > gamma:
            p_next = min(1.0, p_prev + lambda_)
        else:
            p_next = max(0.0, p_prev - delta)

        states[ent] = (e_curr, p_next)

    return E_arr, P_arr


def main():
    t_start = time.perf_counter()
    logger.info("=" * 70)
    logger.info("TRUSTGRAPH Phase 2.1 — Entity-Scoped Temporal Tuning (Validation Only)")
    logger.info("=" * 70)

    # 1. Load frozen baseline artifacts
    logger.info("Loading frozen baseline model and preprocessor...")
    model = BaselineModel.load(base_cfg.MODEL_DIR / "lgbm_model.pkl")
    preprocessor = BaselinePreprocessor.load(base_cfg.PREPROCESSING_DIR)

    with open(base_cfg.ARTIFACTS_DIR / "threshold.json") as f:
        thr_data = json.load(f)
    base_thr = float(thr_data["threshold"])
    logger.info("Frozen baseline threshold: %.6f", base_thr)

    # 2. Extract VALIDATION partition
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
    del X_val
    gc.collect()

    base_flags = (A_val >= base_thr)
    base_tp = np.sum(base_flags & y_val)
    base_fp = np.sum(base_flags & (~y_val))
    base_fn = n_fraud - base_tp
    base_f1 = float((2 * base_tp) / (2 * base_tp + base_fp + base_fn))
    base_prec = float(base_tp / (base_tp + base_fp))
    base_rec  = float(base_tp / n_fraud)
    base_fpr  = float(base_fp / n_legit)

    logger.info("Baseline Validation: F1=%.6f, Recall=%.6f, Prec=%.6f, FPR=%.6f (TP=%d, FP=%d, FN=%d)",
                base_f1, base_rec, base_prec, base_fpr, base_tp, base_fp, base_fn)

    # 3. Grid Search across Candidate Entity Keys
    candidate_keys = ["card_email", "card_composite", "card1", "card_addr"]
    betas      = [0.3, 0.4, 0.5, 0.6]
    gammas     = [0.3, 0.4, 0.5]
    lambdas    = [0.02, 0.05, 0.10]
    deltas     = [0.02, 0.05, 0.10]
    thresholds = [0.4, 0.5, 0.6, 0.7]

    key_ablation_results = {}
    overall_best_f1 = -1.0
    overall_best_config = {}

    for key_name in candidate_keys:
        logger.info("\nEvaluating Candidate Key: %s...", key_name)
        ent_series = resolve_entity_key(val_df, key_type=key_name)
        ent_array  = ent_series.astype(str).values
        n_unique   = len(np.unique(ent_array))
        unresolved_count = int(np.sum(ent_series.str.startswith("unresolved_")))

        logger.info("  Unique entities: %d, Unresolved rows: %d (%.2f%%)",
                    n_unique, unresolved_count, 100 * unresolved_count / len(val_df))

        best_key_f1 = -1.0
        best_key_params = {}

        for beta, gamma, lambda_, delta in product(betas, gammas, lambdas, deltas):
            E_arr, P_arr = fast_entity_stream(ent_array, A_val, beta, gamma, lambda_, delta)

            for thr_t in thresholds:
                temp_flags = (P_arr >= thr_t)
                comb_flags = base_flags | temp_flags

                tp = int(np.sum(comb_flags & y_val))
                fp = int(np.sum(comb_flags & (~y_val)))
                fn = int(n_fraud - tp)

                f1 = float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
                prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                rec  = float(tp / n_fraud) if n_fraud > 0 else 0.0
                fpr  = float(fp / n_legit) if n_legit > 0 else 0.0

                if f1 > best_key_f1:
                    best_key_f1 = f1
                    best_key_params = {
                        "entity_key": key_name,
                        "beta": beta, "gamma": gamma, "lambda": lambda_, "delta": delta,
                        "temporal_threshold": thr_t,
                        "f1": f1, "recall": rec, "precision": prec, "fpr": fpr,
                        "frauds_recovered": int(tp - base_tp),
                        "extra_fp": int(fp - base_fp),
                    }

        logger.info("  Key %s Best Val F1: %.6f (Recov=%d, ExtraFP=%d, Prec=%.4f, Rec=%.4f, FPR=%.4f)",
                    key_name, best_key_f1, best_key_params["frauds_recovered"], best_key_params["extra_fp"],
                    best_key_params["precision"], best_key_params["recall"], best_key_params["fpr"])

        key_ablation_results[key_name] = {
            "unique_entities_val": n_unique,
            "unresolved_count_val": unresolved_count,
            "unresolved_pct_val": round(100 * unresolved_count / len(val_df), 2),
            "best_f1": best_key_f1,
            "best_params": best_key_params,
        }

        if best_key_f1 > overall_best_f1:
            overall_best_f1 = best_key_f1
            overall_best_config = best_key_params

    logger.info("=" * 70)
    logger.info("OVERALL BEST CONFIGURATION (Validation Set):")
    logger.info("  Entity Key:         %s", overall_best_config["entity_key"])
    logger.info("  Parameters:         beta=%.2f, gamma=%.2f, lambda=%.2f, delta=%.2f, tau_temp=%.2f",
                overall_best_config["beta"], overall_best_config["gamma"],
                overall_best_config["lambda"], overall_best_config["delta"],
                overall_best_config["temporal_threshold"])
    logger.info("  Validation F1:      %.6f (Baseline: %.6f, Delta: %+.6f)",
                overall_best_f1, base_f1, overall_best_f1 - base_f1)
    logger.info("  Frauds Recovered:   %d (%.2f%% of baseline FNs)",
                overall_best_config["frauds_recovered"],
                100 * overall_best_config["frauds_recovered"] / base_fn)
    logger.info("  Extra False Pos:    %d (FPR: %.6f -> %.6f)",
                overall_best_config["extra_fp"], base_fpr, overall_best_config["fpr"])

    # 4. Save Artifacts
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "parameters.json", "w") as f:
        json.dump({
            "selected_parameters": overall_best_config,
            "baseline_validation": {
                "f1": base_f1, "recall": base_rec, "precision": base_prec, "fpr": base_fpr,
                "tp": int(base_tp), "fp": int(base_fp), "fn": int(base_fn),
            },
            "selection_criterion": "maximise Validation F1 on B1_entity = Baseline + Entity Temporal",
            "tuning_duration_seconds": round(time.perf_counter() - t_start, 2),
        }, f, indent=2)

    with open(OUT_DIR / "entity_key_ablation.json", "w") as f:
        json.dump(key_ablation_results, f, indent=2)

    logger.info("Parameters and key ablation saved → %s", OUT_DIR)


if __name__ == "__main__":
    main()
