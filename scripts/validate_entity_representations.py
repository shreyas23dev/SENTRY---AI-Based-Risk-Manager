"""
validate_entity_representations.py — Validation-Only Robustness Check of Entity Grouping Keys
==============================================================================================

Evaluates 5 candidate dataset-derived pseudonymous grouping keys on VALIDATION (88,581 rows):
  1. card1
  2. card1 + P_emaildomain (card_email)
  3. card1 + addr1 (card_addr)
  4. card1 + card2 + card3 + card4 + card5 + card6 (card_composite)
  5. card1 + addr1 + P_emaildomain (card_addr_email)

Strict Protocol:
  - Evaluation executed strictly on VALIDATION partition.
  - TEST partition is NEVER accessed.
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
logger = logging.getLogger("validate_entity_representations")

OUT_DIR = base_cfg.PROJECT_ROOT / "artifacts" / "temporal_entity"


def fast_entity_stream(entities: np.ndarray, A_val: np.ndarray, beta: float, gamma: float, lambda_: float, delta: float):
    N = len(A_val)
    E_arr = np.empty(N, dtype=np.float64)
    P_arr = np.empty(N, dtype=np.float64)

    states = {}
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
    logger.info("=" * 80)
    logger.info("TRUSTGRAPH — Validation-Only Robustness Check of 5 Entity Grouping Keys")
    logger.info("=" * 80)

    # 1. Load frozen baseline model & threshold
    logger.info("Loading frozen baseline model and preprocessor...")
    model = BaselineModel.load(base_cfg.MODEL_DIR / "lgbm_model.pkl")
    preprocessor = BaselinePreprocessor.load(base_cfg.PREPROCESSING_DIR)

    with open(base_cfg.ARTIFACTS_DIR / "threshold.json") as f:
        thr_data = json.load(f)
    base_thr = float(thr_data["threshold"])
    logger.info("Frozen baseline decision threshold (tau_base): %.6f", base_thr)

    # 2. Extract VALIDATION partition strictly
    logger.info("Loading training data to extract VALIDATION partition...")
    df, join_stats = load_train_data()
    _, val_df, _, _ = chronological_split(df)
    del df
    gc.collect()

    logger.info("VALIDATION Partition Size: %d rows (Fraud Rate: %.3f%%)", len(val_df), 100 * val_df["isFraud"].mean())
    y_val = (val_df["isFraud"].values == 1)
    n_fraud = int(np.sum(y_val))
    n_legit = int(len(y_val) - n_fraud)

    # Preprocess and score validation partition
    X_val = preprocessor.transform(val_df)
    A_val = model.predict_risk(X_val)
    del X_val
    gc.collect()

    base_flags = (A_val >= base_thr)
    base_tp = int(np.sum(base_flags & y_val))
    base_fp = int(np.sum(base_flags & (~y_val)))
    base_fn = int(n_fraud - base_tp)
    base_f1 = float((2 * base_tp) / (2 * base_tp + base_fp + base_fn))
    base_prec = float(base_tp / (base_tp + base_fp))
    base_rec  = float(base_tp / n_fraud)
    base_fpr  = float(base_fp / n_legit)

    logger.info("Frozen Baseline on Validation: F1=%.6f, Prec=%.6f, Rec=%.6f, FPR=%.6f (TP=%d, FP=%d, FN=%d)",
                base_f1, base_prec, base_rec, base_fpr, base_tp, base_fp, base_fn)

    # 3. Evaluate the 5 Candidate Grouping Keys
    candidate_keys = [
        ("card1", "card1 (Single Attribute Key)"),
        ("card_email", "card1 + P_emaildomain (Composite Key)"),
        ("card_addr", "card1 + addr1 (Composite Key)"),
        ("card_composite", "card1..card6 (Multi-attribute Tuple Key)"),
        ("card_addr_email", "card1 + addr1 + P_emaildomain (3-Field Composite Key)"),
    ]

    betas      = [0.3, 0.4, 0.5, 0.6]
    gammas     = [0.3, 0.4, 0.5]
    lambdas    = [0.02, 0.05, 0.10]
    deltas     = [0.02, 0.05, 0.10]
    thresholds = [0.4, 0.5, 0.6, 0.7]

    results_table = {}

    for key_id, key_label in candidate_keys:
        logger.info("\nEvaluating Candidate [%s]: %s...", key_id, key_label)
        ent_series = resolve_entity_key(val_df, key_type=key_id)
        unres_mask = ent_series.str.startswith("unresolved_")
        resolved_s = ent_series[~unres_mask]

        n_unres = int(unres_mask.sum())
        n_res   = int(len(resolved_s))
        pct_res = round(100.0 * n_res / len(val_df), 2)

        unique_total = int(ent_series.nunique())
        unique_resolved = int(resolved_s.nunique()) if n_res > 0 else 0

        counts = resolved_s.value_counts().values if n_res > 0 else np.array([0])
        density_mean   = round(float(np.mean(counts)), 2) if len(counts) > 0 else 0.0
        density_median = round(float(np.median(counts)), 1) if len(counts) > 0 else 0.0
        density_p95    = round(float(np.percentile(counts, 95)), 1) if len(counts) > 0 else 0.0
        density_max    = int(np.max(counts)) if len(counts) > 0 else 0

        ent_array = ent_series.astype(str).values

        best_f1 = -1.0
        best_params = {}

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

                if f1 > best_f1:
                    best_f1 = f1
                    best_params = {
                        "key_id": key_id,
                        "key_label": key_label,
                        "beta": beta, "gamma": gamma, "lambda": lambda_, "delta": delta,
                        "temporal_threshold": thr_t,
                        "f1": f1, "precision": prec, "recall": rec, "fpr": fpr,
                        "frauds_recovered": int(tp - base_tp),
                        "extra_fp": int(fp - base_fp),
                    }

        logger.info("  Validation Result for %s:", key_id)
        logger.info("    Resolved Coverage: %d/%d (%.2f%%) | Unique Entities: %d (Resolved: %d)",
                    n_res, len(val_df), pct_res, unique_total, unique_resolved)
        logger.info("    Density: Mean=%.2f, Median=%.1f, p95=%.1f, Max=%d",
                    density_mean, density_median, density_p95, density_max)
        logger.info("    Optimal Params: beta=%.2f, gamma=%.2f, lambda=%.2f, delta=%.2f, tau_temp=%.2f",
                    best_params["beta"], best_params["gamma"], best_params["lambda"], best_params["delta"], best_params["temporal_threshold"])
        logger.info("    Validation F1: %.6f (Prec=%.6f, Rec=%.6f, FPR=%.6f)",
                    best_f1, best_params["precision"], best_params["recall"], best_params["fpr"])
        logger.info("    Frauds Recovered: %d | Extra FP: %d",
                    best_params["frauds_recovered"], best_params["extra_fp"])

        results_table[key_id] = {
            "key_label": key_label,
            "resolved_coverage_pct": pct_res,
            "resolved_txns": n_res,
            "unresolved_txns": n_unres,
            "unique_entities_total": unique_total,
            "unique_entities_resolved": unique_resolved,
            "density_mean": density_mean,
            "density_median": density_median,
            "density_p95": density_p95,
            "density_max": density_max,
            "validation_f1": best_f1,
            "validation_precision": best_params["precision"],
            "validation_recall": best_params["recall"],
            "validation_fpr": best_params["fpr"],
            "frauds_recovered": best_params["frauds_recovered"],
            "extra_false_positives": best_params["extra_fp"],
            "best_parameters": {
                "beta": best_params["beta"],
                "gamma": best_params["gamma"],
                "lambda": best_params["lambda"],
                "delta": best_params["delta"],
                "temporal_threshold": best_params["temporal_threshold"],
            },
        }

    # 4. Save validation robustness report
    report_data = {
        "baseline_validation_metrics": {
            "f1": base_f1,
            "precision": base_prec,
            "recall": base_rec,
            "fpr": base_fpr,
            "tp": base_tp, "fp": base_fp, "fn": base_fn,
            "decision_threshold": base_thr,
        },
        "candidate_key_evaluations": results_table,
        "evaluation_scope": "VALIDATION ONLY (88,581 chronological transactions)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(OUT_DIR / "validation_entity_robustness.json", "w") as f:
        json.dump(report_data, f, indent=2)

    logger.info("Validation entity robustness analysis saved → %s", OUT_DIR / "validation_entity_robustness.json")
    logger.info("Completed in %.2f s", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
