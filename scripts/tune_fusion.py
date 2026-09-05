"""
tune_fusion.py — TRUSTGRAPH Phase 3.1 Validation-Only Fusion Parameter Tuning
=============================================================================

Protocol:
  1. Reconstruct TRAIN and VALIDATION partitions.
  2. Compute exact frozen A_t (LightGBM), P_t (Entity Temporal), G_t (Relational Graph).
  3. Evaluate candidate fusion rules F1, F2, F3, F4 across parameter grids on VALIDATION ONLY.
  4. Verify non-suppression (R_t >= A_t) and zero-context invariance.
  5. Select winning candidate rule based on Validation F1 and FPR control.
  6. Save validation results and frozen parameters.

TEST partition is NOT accessed or loaded anywhere in this script.
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

from trustgraph.baseline import config as base_cfg
from trustgraph.baseline.data_loader import load_train_data, chronological_split
from trustgraph.baseline.model import BaselineModel
from trustgraph.baseline.preprocessing import BaselinePreprocessor
from trustgraph.temporal.entity_tracker import resolve_entity_key, EntityTemporalRiskEngine
from trustgraph.relational.graph_engine import (
    GraphParameters,
    LightweightRelationalGraph,
    process_partition,
)
from trustgraph.fusion.config import (
    FUSION_DIR,
    BASELINE_THRESHOLD,
    TEMPORAL_BETA, TEMPORAL_GAMMA, TEMPORAL_LAMBDA, TEMPORAL_DELTA, TEMPORAL_THRESHOLD,
    RELATIONAL_K_MAX, RELATIONAL_WINDOW, RELATIONAL_D_REF, RELATIONAL_V_REF,
    RELATIONAL_WD, RELATIONAL_WV, RELATIONAL_THRESHOLD, ENTITY_KEY_TYPE,
    ALPHA_GRID, BETA_GRID, CP_GRID, CG_GRID, TAU_COMB_GRID,
)
from trustgraph.fusion.fusion_engine import apply_fusion_rule, verify_fusion_invariance
from trustgraph.fusion.evaluator import compute_system_metrics, compute_coverage_aware_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tune_fusion")

FUSION_DIR.mkdir(parents=True, exist_ok=True)


def prepare_validation_data():
    """Load train data, compute A_t, P_t, G_t for TRAIN and VAL."""
    logger.info("Loading dataset...")
    df, _ = load_train_data()
    train_df, val_df, _, split_meta = chronological_split(df)
    del df

    # Resolve entity proxy
    logger.info("Resolving entity proxies (%s)...", ENTITY_KEY_TYPE)
    train_df["entity_proxy"] = resolve_entity_key(train_df, key_type=ENTITY_KEY_TYPE)
    val_df["entity_proxy"]   = resolve_entity_key(val_df,   key_type=ENTITY_KEY_TYPE)

    # 1. Point-wise LightGBM Model Inference: A_t
    logger.info("Loading frozen LightGBM model and preprocessor...")
    model = BaselineModel.load(base_cfg.MODEL_DIR / "lgbm_model.pkl")
    preprocessor = BaselinePreprocessor.load(base_cfg.PREPROCESSING_DIR)

    logger.info("Generating A_t scores for TRAIN and VAL...")
    X_train = preprocessor.transform(train_df)
    A_train = model.predict_risk(X_train)
    train_df["A_t"] = A_train
    del X_train

    X_val = preprocessor.transform(val_df)
    A_val = model.predict_risk(X_val)
    val_df["A_t"] = A_val
    del X_val

    # 2. Entity Temporal Risk: P_t
    logger.info("Generating P_t scores across TRAIN -> VAL...")
    temp_engine = EntityTemporalRiskEngine(
        beta=TEMPORAL_BETA, gamma=TEMPORAL_GAMMA,
        lambda_=TEMPORAL_LAMBDA, delta=TEMPORAL_DELTA,
    )
    # Process TRAIN
    train_ents = train_df["entity_proxy"].values
    for i in range(len(train_df)):
        temp_engine.step(str(train_ents[i]), float(A_train[i]))

    # Process VAL
    val_ents = val_df["entity_proxy"].values
    val_P = np.zeros(len(val_df), dtype=float)
    for i in range(len(val_df)):
        _, p_val = temp_engine.step(str(val_ents[i]), float(A_val[i]))
        val_P[i] = p_val
    val_df["P_t"] = val_P

    # 3. Persistent Relational Graph: G_t
    logger.info("Generating G_t scores across TRAIN -> VAL...")
    rel_params = GraphParameters(
        k_attr_max=RELATIONAL_K_MAX,
        window_sec=RELATIONAL_WINDOW,
        d_ref=RELATIONAL_D_REF,
        v_ref=RELATIONAL_V_REF,
        w_D=RELATIONAL_WD,
        w_V=RELATIONAL_WV,
        relational_attrs=("DeviceInfo",),
    )
    graph_engine = LightweightRelationalGraph(rel_params)
    graph_engine.fit_attribute_frequency_ceiling(train_df)
    # Build graph on TRAIN
    process_partition(train_df, graph_engine)
    # Score VAL from persistent graph
    val_records = process_partition(val_df, graph_engine)
    val_G = np.array([r.G_t for r in val_records], dtype=float)
    val_df["G_t"] = val_G

    logger.info("Validation feature preparation complete: N=%d rows, fraud_rate=%.4f%%",
                len(val_df), 100 * val_df["isFraud"].mean())
    return val_df, split_meta


def run_grid_search(val_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate candidate fusion rules F1, F2, F3, F4 on VALIDATION ONLY."""
    y_val = val_df["isFraud"].values.astype(int)
    A_t = val_df["A_t"].values.astype(float)
    P_t = val_df["P_t"].values.astype(float)
    G_t = val_df["G_t"].values.astype(float)

    # Reference Baselines on Validation
    b0_pred = (A_t >= BASELINE_THRESHOLD).astype(int)
    b1_pred = ((A_t >= BASELINE_THRESHOLD) | (P_t >= TEMPORAL_THRESHOLD)).astype(int)
    b2_pred = ((A_t >= BASELINE_THRESHOLD) | (G_t >= RELATIONAL_THRESHOLD)).astype(int)

    b0_metrics = compute_system_metrics(y_val, b0_pred, A_t)
    b1_metrics = compute_system_metrics(
        y_val, b1_pred,
        base_tp=b0_metrics["tp"], base_fp=b0_metrics["fp"],
        base_recall=b0_metrics["recall"], base_fpr=b0_metrics["fpr"],
    )
    b2_metrics = compute_system_metrics(
        y_val, b2_pred,
        base_tp=b0_metrics["tp"], base_fp=b0_metrics["fp"],
        base_recall=b0_metrics["recall"], base_fpr=b0_metrics["fpr"],
    )

    logger.info("Baseline Reference (Validation):")
    logger.info("  B0: F1=%.6f  Prec=%.6f  Rec=%.6f  FPR=%.6f",
                b0_metrics["f1"], b0_metrics["precision"], b0_metrics["recall"], b0_metrics["fpr"])
    logger.info("  B1: F1=%.6f  Prec=%.6f  Rec=%.6f  FPR=%.6f  dFrauds=%+d  dFP=%+d",
                b1_metrics["f1"], b1_metrics["precision"], b1_metrics["recall"], b1_metrics["fpr"],
                b1_metrics["additional_frauds_recovered"], b1_metrics["additional_false_positives"])
    logger.info("  B2: F1=%.6f  Prec=%.6f  Rec=%.6f  FPR=%.6f  dFrauds=%+d  dFP=%+d",
                b2_metrics["f1"], b2_metrics["precision"], b2_metrics["recall"], b2_metrics["fpr"],
                b2_metrics["additional_frauds_recovered"], b2_metrics["additional_false_positives"])

    all_evaluations: List[Dict[str, Any]] = []
    best_per_rule: Dict[str, Dict[str, Any]] = {}

    # 1. Candidate F1: clip(A_t + α*P_t + β*G_t, 0, 1)
    logger.info("Evaluating Candidate F1...")
    best_f1_score = -1.0
    for a in ALPHA_GRID:
        for b in BETA_GRID:
            R = apply_fusion_rule("F1", A_t, P_t, G_t, {"alpha": a, "beta": b})
            passed, diag = verify_fusion_invariance(A_t, P_t, G_t, R)
            if not passed:
                continue
            for tau in TAU_COMB_GRID:
                pred = (R >= tau).astype(int)
                m = compute_system_metrics(
                    y_val, pred, R,
                    base_tp=b0_metrics["tp"], base_fp=b0_metrics["fp"],
                    base_recall=b0_metrics["recall"], base_fpr=b0_metrics["fpr"],
                )
                entry = {
                    "rule": "F1", "formula": "clip(A_t + alpha*P_t + beta*G_t, 0, 1)",
                    "params": {"alpha": a, "beta": b, "tau_comb": tau},
                    "metrics": m, "invariance_diagnostics": diag,
                }
                all_evaluations.append(entry)
                if m["f1"] > best_f1_score:
                    best_f1_score = m["f1"]
                    best_per_rule["F1"] = entry

    # 2. Candidate F2: clip(A_t + β*G_t*(1-A_t), 0, 1)
    logger.info("Evaluating Candidate F2...")
    best_f2_score = -1.0
    for b in BETA_GRID:
        R = apply_fusion_rule("F2", A_t, P_t, G_t, {"beta": b})
        passed, diag = verify_fusion_invariance(A_t, P_t, G_t, R)
        if not passed:
            continue
        for tau in TAU_COMB_GRID:
            pred = (R >= tau).astype(int)
            m = compute_system_metrics(
                y_val, pred, R,
                base_tp=b0_metrics["tp"], base_fp=b0_metrics["fp"],
                base_recall=b0_metrics["recall"], base_fpr=b0_metrics["fpr"],
            )
            entry = {
                "rule": "F2", "formula": "clip(A_t + beta*G_t*(1-A_t), 0, 1)",
                "params": {"beta": b, "tau_comb": tau},
                "metrics": m, "invariance_diagnostics": diag,
            }
            all_evaluations.append(entry)
            if m["f1"] > best_f2_score:
                best_f2_score = m["f1"]
                best_per_rule["F2"] = entry

    # 3. Candidate F3: clip(A_t + α*P_t*(1-A_t) + β*G_t*(1-A_t), 0, 1)
    logger.info("Evaluating Candidate F3...")
    best_f3_score = -1.0
    for a in ALPHA_GRID:
        for b in BETA_GRID:
            R = apply_fusion_rule("F3", A_t, P_t, G_t, {"alpha": a, "beta": b})
            passed, diag = verify_fusion_invariance(A_t, P_t, G_t, R)
            if not passed:
                continue
            for tau in TAU_COMB_GRID:
                pred = (R >= tau).astype(int)
                m = compute_system_metrics(
                    y_val, pred, R,
                    base_tp=b0_metrics["tp"], base_fp=b0_metrics["fp"],
                    base_recall=b0_metrics["recall"], base_fpr=b0_metrics["fpr"],
                )
                entry = {
                    "rule": "F3", "formula": "clip(A_t + alpha*P_t*(1-A_t) + beta*G_t*(1-A_t), 0, 1)",
                    "params": {"alpha": a, "beta": b, "tau_comb": tau},
                    "metrics": m, "invariance_diagnostics": diag,
                }
                all_evaluations.append(entry)
                if m["f1"] > best_f3_score:
                    best_f3_score = m["f1"]
                    best_per_rule["F3"] = entry

    # 4. Candidate F4: clip(max(A_t, cP*P_t, cG*G_t), 0, 1)
    logger.info("Evaluating Candidate F4...")
    best_f4_score = -1.0
    for cp in CP_GRID:
        for cg in CG_GRID:
            R = apply_fusion_rule("F4", A_t, P_t, G_t, {"cP": cp, "cG": cg})
            passed, diag = verify_fusion_invariance(A_t, P_t, G_t, R)
            if not passed:
                continue
            for tau in TAU_COMB_GRID:
                pred = (R >= tau).astype(int)
                m = compute_system_metrics(
                    y_val, pred, R,
                    base_tp=b0_metrics["tp"], base_fp=b0_metrics["fp"],
                    base_recall=b0_metrics["recall"], base_fpr=b0_metrics["fpr"],
                )
                entry = {
                    "rule": "F4", "formula": "clip(max(A_t, cP*P_t, cG*G_t), 0, 1)",
                    "params": {"cP": cp, "cG": cg, "tau_comb": tau},
                    "metrics": m, "invariance_diagnostics": diag,
                }
                all_evaluations.append(entry)
                if m["f1"] > best_f4_score:
                    best_f4_score = m["f1"]
                    best_per_rule["F4"] = entry

    # Summary of best configuration per rule on Validation
    logger.info("\n=== BEST CONFIGURATION PER CANDIDATE RULE (VALIDATION ONLY) ===")
    overall_best = None
    overall_best_f1 = -1.0
    for r_name in ["F1", "F2", "F3", "F4"]:
        b = best_per_rule[r_name]
        m = b["metrics"]
        logger.info("  %-3s  F1=%.6f  Prec=%.6f  Rec=%.6f  FPR=%.6f  dFrauds=%+d  dFP=%+d  params=%s",
                    r_name, m["f1"], m["precision"], m["recall"], m["fpr"],
                    m["additional_frauds_recovered"], m["additional_false_positives"],
                    b["params"])
        if m["f1"] > overall_best_f1:
            overall_best_f1 = m["f1"]
            overall_best = b

    logger.info("\nWINNING FUSION RULE SELECTED ON VALIDATION: %s (F1=%.6f)",
                overall_best["rule"], overall_best_f1)

    return {
        "reference_baselines": {
            "B0": b0_metrics, "B1": b1_metrics, "B2": b2_metrics,
        },
        "best_per_rule": best_per_rule,
        "selected_rule": overall_best,
        "all_evaluations": all_evaluations,
    }


def main():
    t0 = time.time()
    val_df, split_meta = prepare_validation_data()
    tuning_results = run_grid_search(val_df)
    elapsed = time.time() - t0

    selected = tuning_results["selected_rule"]

    # Compute coverage-aware validation metrics for the winning rule
    R_val_best = apply_fusion_rule(
        selected["rule"],
        val_df["A_t"].values,
        val_df["P_t"].values,
        val_df["G_t"].values,
        {k: v for k, v in selected["params"].items() if k != "tau_comb"},
    )
    b0_pred = (val_df["A_t"].values >= BASELINE_THRESHOLD).astype(int)
    b3_pred = (R_val_best >= selected["params"]["tau_comb"]).astype(int)
    cov_val = compute_coverage_aware_metrics(
        val_df["isFraud"].values,
        val_df["A_t"].values,
        val_df["P_t"].values,
        val_df["G_t"].values,
        R_val_best,
        b0_pred,
        b3_pred,
    )

    # 1. Save selected_rule.json
    selected_path = FUSION_DIR / "selected_rule.json"
    with open(selected_path, "w") as f:
        json.dump(selected, f, indent=2)
    logger.info("Selected rule saved -> %s", selected_path)

    # 2. Save parameters.json
    frozen_params = {
        "rule_name": selected["rule"],
        "formula": selected["formula"],
        "params": selected["params"],
        "upstream_frozen": {
            "tau_base": BASELINE_THRESHOLD,
            "tau_temp": TEMPORAL_THRESHOLD,
            "tau_rel": RELATIONAL_THRESHOLD,
            "entity_key": ENTITY_KEY_TYPE,
            "temporal_params": {
                "beta": TEMPORAL_BETA, "gamma": TEMPORAL_GAMMA,
                "lambda": TEMPORAL_LAMBDA, "delta": TEMPORAL_DELTA,
            },
            "relational_params": {
                "k_attr_max": RELATIONAL_K_MAX, "window_sec": RELATIONAL_WINDOW,
                "d_ref": RELATIONAL_D_REF, "v_ref": RELATIONAL_V_REF,
                "w_D": RELATIONAL_WD, "w_V": RELATIONAL_WV,
            },
        },
    }
    params_path = FUSION_DIR / "parameters.json"
    with open(params_path, "w") as f:
        json.dump(frozen_params, f, indent=2)
    logger.info("Frozen parameters saved -> %s", params_path)

    # 3. Save validation_results.json
    val_out = {
        "split_meta": split_meta,
        "reference_baselines": tuning_results["reference_baselines"],
        "best_per_rule": tuning_results["best_per_rule"],
        "selected_rule": selected,
        "coverage_aware_validation": cov_val,
        "tuning_elapsed_sec": round(elapsed, 1),
    }
    val_results_path = FUSION_DIR / "validation_results.json"
    with open(val_results_path, "w") as f:
        json.dump(val_out, f, indent=2)
    logger.info("Validation results saved -> %s", val_results_path)


if __name__ == "__main__":
    main()
