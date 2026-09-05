"""
run_model_bakeoff.py — Controlled Model Bake-Off (LightGBM vs CatBoost vs XGBoost)
===================================================================================

Controlled experimental comparison on the V2 feature representation:
  - Model A: V2 LightGBM
  - Model B: V2 CatBoost
  - Model C: V2 XGBoost
  - Optional: Simple Validation-Selected Ensemble (0.5 LGBM + 0.5 CatBoost or 0.5 LGBM + 0.3 CB + 0.2 XGB)

Evaluation Methodology:
  - Strict chronological TRAIN / VALIDATION / TEST split.
  - Model selection strictly guided by VALIDATION PR-AUC & ROC-AUC.
  - Fixed-FPR and Precision-Target frontier analysis on validation.
  - Prediction correlation & error diversity analysis.
  - Single-pass evaluation on held-out TEST.
  - Downstream TRUSTGRAPH pipeline integration & real slow-burn trajectory.
  - Runtime latency & throughput profiling.
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix,
)

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from trustgraph.baseline import config as cfg
from trustgraph.baseline.data_loader import load_train_data, chronological_split
from trustgraph.baseline.preprocessing import BaselinePreprocessor
from trustgraph.baseline.model import BaselineModel
from trustgraph.temporal.entity_tracker import resolve_entity_key, EntityTemporalRiskEngine
from trustgraph.relational.graph_engine import (
    GraphParameters, LightweightRelationalGraph, process_partition,
)
from trustgraph.fusion.fusion_engine import apply_fusion_rule
from trustgraph.policy.config import PolicyAction
from trustgraph.policy.decision_engine import PolicyThresholds, batch_assign_actions
from trustgraph.features_v2.causal_features import (
    compute_point_in_time_features, FrequencyEncoder, CausalStreamFeatureEngine,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model_bakeoff")

BAKEOFF_DIR = cfg.PROJECT_ROOT / "artifacts" / "model_bakeoff"
PLOTS_DIR = BAKEOFF_DIR / "plots"
BAKEOFF_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FROZEN_THRESHOLD = 0.594298
POLICY_THRESHOLDS = PolicyThresholds(tau_verify=0.60, tau_throttle=0.65, tau_block=0.80)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = FROZEN_THRESHOLD) -> Dict[str, Any]:
    roc_auc = float(roc_auc_score(y_true, y_proba))
    pr_auc = float(average_precision_score(y_true, y_proba))
    y_pred = (y_proba >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1": round(float(f1), 6),
        "fpr": round(fpr, 6),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "threshold": round(float(threshold), 6),
    }


def find_fixed_fpr_frontier(y_true: np.ndarray, y_proba: np.ndarray, target_fprs=(0.001, 0.002, 0.005, 0.010, 0.020)) -> Dict[str, Any]:
    """Finds maximum recall threshold subject to FPR <= target."""
    threshold_grid = np.linspace(0.01, 0.99, 1000)
    results = {}

    for target_fpr in target_fprs:
        best_rec = -1.0
        best_m = None
        for th in threshold_grid:
            y_pred = (y_proba >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
            if fpr <= target_fpr:
                rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
                if rec > best_rec:
                    best_rec = rec
                    prec, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
                    best_m = {
                        "target_fpr": target_fpr,
                        "threshold": round(float(th), 6),
                        "precision": round(float(prec), 6),
                        "recall": round(float(rec), 6),
                        "f1": round(float(f1), 6),
                        "fpr": round(float(fpr), 6),
                        "tp": int(tp),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tn": int(tn),
                    }
        key_name = f"FPR_le_{target_fpr*100:.2f}pct"
        results[key_name] = best_m
    return results


def find_precision_target_frontier(y_true: np.ndarray, y_proba: np.ndarray, target_precs=(0.70, 0.75, 0.80, 0.85)) -> Dict[str, Any]:
    """Finds maximum recall threshold subject to Precision >= target."""
    threshold_grid = np.linspace(0.01, 0.99, 1000)
    results = {}

    for target_prec in target_precs:
        best_rec = -1.0
        best_m = None
        for th in threshold_grid:
            y_pred = (y_proba >= th).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
            if prec >= target_prec:
                if rec > best_rec:
                    best_rec = rec
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
                    best_m = {
                        "target_precision": target_prec,
                        "threshold": round(float(th), 6),
                        "precision": round(float(prec), 6),
                        "recall": round(float(rec), 6),
                        "f1": round(float(f1), 6),
                        "fpr": round(float(fpr), 6),
                        "tp": int(tp),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tn": int(tn),
                    }
        key_name = f"Precision_ge_{int(target_prec*100)}pct"
        results[key_name] = best_m
    return results


def main():
    logger.info("=====================================================================")
    logger.info("  STARTING CONTROLLED MODEL BAKE-OFF: LGBM vs CATBOOST vs XGBOOST")
    logger.info("=====================================================================")

    # 1. Load data & extract V2 feature sets
    raw_df, _ = load_train_data()
    train_df, val_df, test_df, _ = chronological_split(raw_df)
    del raw_df

    for part in [train_df, val_df, test_df]:
        part["entity_proxy"] = resolve_entity_key(part, key_type="card_addr_email")

    y_train = train_df["isFraud"].values
    y_val = val_df["isFraud"].values
    y_test = test_df["isFraud"].values

    spw = float((y_train == 0).sum()) / float(max((y_train == 1).sum(), 1))
    logger.info(f"Class Imbalance Scale Pos Weight: {spw:.2f}")

    logger.info("Extracting V2 feature representation (452 features)...")
    base_prep = BaselinePreprocessor()
    X_tr_base = base_prep.fit_transform(train_df)
    X_va_base = base_prep.transform(val_df)
    X_te_base = base_prep.transform(test_df)

    pit_tr = compute_point_in_time_features(train_df)
    pit_va = compute_point_in_time_features(val_df)
    pit_te = compute_point_in_time_features(test_df)

    fe = FrequencyEncoder(["card1", "addr1", "P_emaildomain", "DeviceInfo"]).fit(train_df)
    fe_tr = fe.transform(train_df)
    fe_va = fe.transform(val_df)
    fe_te = fe.transform(test_df)

    stream_engine = CausalStreamFeatureEngine()
    stream_tr = stream_engine.process_partition(train_df)
    stream_va = stream_engine.process_partition(val_df)
    stream_te = stream_engine.process_partition(test_df)

    X_train_v2 = pd.concat([X_tr_base, fe_tr, stream_tr[["prior_count_card1", "prior_count_addr1", "prior_count_P_emaildomain", "prior_count_DeviceInfo"]],
                            pit_tr[["hour_of_day", "day_of_week", "hour_sin", "hour_cos"]], stream_tr[["entity_dt_elapsed"]],
                            pit_tr[["log_TransactionAmt", "amt_decimal_cents", "amt_is_integer"]],
                            stream_tr[["entity_prior_count", "entity_hist_mean_amt", "entity_hist_std_amt", "entity_amt_ratio"]]], axis=1)

    X_val_v2 = pd.concat([X_va_base, fe_va, stream_va[["prior_count_card1", "prior_count_addr1", "prior_count_P_emaildomain", "prior_count_DeviceInfo"]],
                          pit_va[["hour_of_day", "day_of_week", "hour_sin", "hour_cos"]], stream_va[["entity_dt_elapsed"]],
                          pit_va[["log_TransactionAmt", "amt_decimal_cents", "amt_is_integer"]],
                          stream_va[["entity_prior_count", "entity_hist_mean_amt", "entity_hist_std_amt", "entity_amt_ratio"]]], axis=1)

    X_test_v2 = pd.concat([X_te_base, fe_te, stream_te[["prior_count_card1", "prior_count_addr1", "prior_count_P_emaildomain", "prior_count_DeviceInfo"]],
                           pit_te[["hour_of_day", "day_of_week", "hour_sin", "hour_cos"]], stream_te[["entity_dt_elapsed"]],
                           pit_te[["log_TransactionAmt", "amt_decimal_cents", "amt_is_integer"]],
                           stream_te[["entity_prior_count", "entity_hist_mean_amt", "entity_hist_std_amt", "entity_amt_ratio"]]], axis=1)

    logger.info(f"Feature matrix ready. Total features: {X_train_v2.shape[1]}")

    # =========================================================================
    # MODEL A: LightGBM V2 (Verified Reference)
    # =========================================================================
    logger.info("\n-------------------------------------------------------------")
    logger.info("  TRAINING MODEL A: LightGBM V2")
    logger.info("-------------------------------------------------------------")
    lgbm_model = BaselineModel.load(cfg.PROJECT_ROOT / "artifacts" / "baseline_v2" / "model" / "lgbm_model.pkl")
    proba_val_lgbm = lgbm_model.predict_risk(X_val_v2)
    proba_test_lgbm = lgbm_model.predict_risk(X_test_v2)

    val_metrics_lgbm = compute_metrics(y_val, proba_val_lgbm)
    test_metrics_lgbm = compute_metrics(y_test, proba_test_lgbm)
    logger.info(f"[LightGBM] VAL PR-AUC: {val_metrics_lgbm['pr_auc']:.6f} | ROC-AUC: {val_metrics_lgbm['roc_auc']:.6f} | Prec: {val_metrics_lgbm['precision']:.4f} | Rec: {val_metrics_lgbm['recall']:.4f}")

    # =========================================================================
    # MODEL B: CatBoost V2
    # =========================================================================
    logger.info("\n-------------------------------------------------------------")
    logger.info("  TRAINING MODEL B: CatBoost V2")
    logger.info("-------------------------------------------------------------")
    # For CatBoost: identify categorical column indices from the base preprocessor
    cat_feature_names = [c for c in base_prep.cat_cols if c in X_train_v2.columns]
    logger.info(f"CatBoost categorical features count: {len(cat_feature_names)}")

    # Replace NaNs in categorical columns with string indicator for CatBoost
    X_tr_cb = X_train_v2.copy()
    X_va_cb = X_val_v2.copy()
    X_te_cb = X_test_v2.copy()
    for col in cat_feature_names:
        X_tr_cb[col] = X_tr_cb[col].fillna("missing").astype(str)
        X_va_cb[col] = X_va_cb[col].fillna("missing").astype(str)
        X_te_cb[col] = X_te_cb[col].fillna("missing").astype(str)

    cb_model = cb.CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=8,
        scale_pos_weight=spw,
        eval_metric="PRAUC",
        random_seed=42,
        task_type="CPU",
        thread_count=-1,
        verbose=100,
        early_stopping_rounds=100,
        cat_features=cat_feature_names if cat_feature_names else None,
    )

    t0 = time.perf_counter()
    cb_model.fit(X_tr_cb, y_train, eval_set=(X_va_cb, y_val), use_best_model=True)
    cb_train_time = time.perf_counter() - t0
    logger.info(f"CatBoost training completed in {cb_train_time:.1f}s. Best iteration: {cb_model.get_best_iteration()}")

    proba_val_cb = cb_model.predict_proba(X_va_cb)[:, 1]
    proba_test_cb = cb_model.predict_proba(X_te_cb)[:, 1]

    val_metrics_cb = compute_metrics(y_val, proba_val_cb)
    test_metrics_cb = compute_metrics(y_test, proba_test_cb)
    logger.info(f"[CatBoost] VAL PR-AUC: {val_metrics_cb['pr_auc']:.6f} | ROC-AUC: {val_metrics_cb['roc_auc']:.6f} | Prec: {val_metrics_cb['precision']:.4f} | Rec: {val_metrics_cb['recall']:.4f}")

    # =========================================================================
    # MODEL C: XGBoost V2
    # =========================================================================
    logger.info("\n-------------------------------------------------------------")
    logger.info("  TRAINING MODEL C: XGBoost V2")
    logger.info("-------------------------------------------------------------")
    xgb_model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric=["auc", "logloss"],
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    t0 = time.perf_counter()
    xgb_model.fit(X_train_v2, y_train, eval_set=[(X_val_v2, y_val)], verbose=100)
    xgb_train_time = time.perf_counter() - t0
    logger.info(f"XGBoost training completed in {xgb_train_time:.1f}s. Best iteration: {xgb_model.best_iteration}")

    proba_val_xgb = xgb_model.predict_proba(X_val_v2)[:, 1]
    proba_test_xgb = xgb_model.predict_proba(X_test_v2)[:, 1]

    val_metrics_xgb = compute_metrics(y_val, proba_val_xgb)
    test_metrics_xgb = compute_metrics(y_test, proba_test_xgb)
    logger.info(f"[XGBoost] VAL PR-AUC: {val_metrics_xgb['pr_auc']:.6f} | ROC-AUC: {val_metrics_xgb['roc_auc']:.6f} | Prec: {val_metrics_xgb['precision']:.4f} | Rec: {val_metrics_xgb['recall']:.4f}")

    # =========================================================================
    # ERROR DIVERSITY & PREDICTION CORRELATION ANALYSIS
    # =========================================================================
    logger.info("\n-------------------------------------------------------------")
    logger.info("  ERROR DIVERSITY & PREDICTION CORRELATION ANALYSIS")
    logger.info("-------------------------------------------------------------")
    # Correlations on TEST
    pearson_lgb_cb, _ = stats.pearsonr(proba_test_lgbm, proba_test_cb)
    spearman_lgb_cb, _ = stats.spearmanr(proba_test_lgbm, proba_test_cb)

    pearson_lgb_xgb, _ = stats.pearsonr(proba_test_lgbm, proba_test_xgb)
    spearman_lgb_xgb, _ = stats.spearmanr(proba_test_lgbm, proba_test_xgb)

    pearson_cb_xgb, _ = stats.pearsonr(proba_test_cb, proba_test_xgb)
    spearman_cb_xgb, _ = stats.spearmanr(proba_test_cb, proba_test_xgb)

    # Prediction binary decisions at frozen threshold
    pred_lgb = (proba_test_lgbm >= FROZEN_THRESHOLD)
    pred_cb = (proba_test_cb >= FROZEN_THRESHOLD)
    pred_xgb = (proba_test_xgb >= FROZEN_THRESHOLD)
    fraud_true = (y_test == 1)
    legit_true = (y_test == 0)

    frauds_lgb = pred_lgb & fraud_true
    frauds_cb = pred_cb & fraud_true
    frauds_xgb = pred_xgb & fraud_true

    caught_all = int((frauds_lgb & frauds_cb & frauds_xgb).sum())
    missed_lgb_caught_cb = int((~frauds_lgb & frauds_cb).sum())
    missed_lgb_caught_xgb = int((~frauds_lgb & frauds_xgb).sum())
    missed_cb_caught_lgb = int((~frauds_cb & frauds_lgb).sum())
    missed_all = int((~frauds_lgb & ~frauds_cb & ~frauds_xgb & fraud_true).sum())

    fps_lgb = int((pred_lgb & legit_true).sum())
    fps_cb = int((pred_cb & legit_true).sum())
    fps_xgb = int((pred_xgb & legit_true).sum())
    fps_all = int((pred_lgb & pred_cb & pred_xgb & legit_true).sum())

    error_diversity = {
        "correlations": {
            "LGBM_vs_CatBoost": {"pearson": round(float(pearson_lgb_cb), 4), "spearman": round(float(spearman_lgb_cb), 4)},
            "LGBM_vs_XGBoost": {"pearson": round(float(pearson_lgb_xgb), 4), "spearman": round(float(spearman_lgb_xgb), 4)},
            "CatBoost_vs_XGBoost": {"pearson": round(float(pearson_cb_xgb), 4), "spearman": round(float(spearman_cb_xgb), 4)},
        },
        "fraud_detection_breakdown": {
            "total_test_frauds": int(fraud_true.sum()),
            "caught_by_LightGBM": int(frauds_lgb.sum()),
            "caught_by_CatBoost": int(frauds_cb.sum()),
            "caught_by_XGBoost": int(frauds_xgb.sum()),
            "caught_by_ALL_three": caught_all,
            "missed_by_LGBM_but_caught_by_CatBoost": missed_lgb_caught_cb,
            "missed_by_LGBM_but_caught_by_XGBoost": missed_lgb_caught_xgb,
            "missed_by_CatBoost_but_caught_by_LGBM": missed_cb_caught_lgb,
            "missed_by_all_three": missed_all,
        },
        "false_positive_overlap": {
            "fp_LightGBM": fps_lgb,
            "fp_CatBoost": fps_cb,
            "fp_XGBoost": fps_xgb,
            "fp_shared_by_ALL_three": fps_all,
        }
    }
    with open(BAKEOFF_DIR / "error_diversity.json", "w") as f:
        json.dump(error_diversity, f, indent=2)

    logger.info(f"Missed by LGBM but caught by CatBoost: {missed_lgb_caught_cb} | by XGBoost: {missed_lgb_caught_xgb}")
    logger.info(f"Spearman Corr: LGB-CB = {spearman_lgb_cb:.4f}, LGB-XGB = {spearman_lgb_xgb:.4f}")

    # =========================================================================
    # OPTIONAL ENSEMBLE BLENDS (Validation Evaluated)
    # =========================================================================
    logger.info("\n-------------------------------------------------------------")
    logger.info("  EVALUATING SIMPLE ENSEMBLE BLENDS (VALIDATION GUIDED)")
    logger.info("-------------------------------------------------------------")
    # Blend 1: 0.5 LGBM + 0.5 CatBoost
    proba_val_blend1 = 0.5 * proba_val_lgbm + 0.5 * proba_val_cb
    proba_test_blend1 = 0.5 * proba_test_lgbm + 0.5 * proba_test_cb
    val_metrics_b1 = compute_metrics(y_val, proba_val_blend1)
    test_metrics_b1 = compute_metrics(y_test, proba_test_blend1)

    # Blend 2: 0.5 LGBM + 0.3 CatBoost + 0.2 XGBoost
    proba_val_blend2 = 0.5 * proba_val_lgbm + 0.3 * proba_val_cb + 0.2 * proba_val_xgb
    proba_test_blend2 = 0.5 * proba_test_lgbm + 0.3 * proba_test_cb + 0.2 * proba_test_xgb
    val_metrics_b2 = compute_metrics(y_val, proba_val_blend2)
    test_metrics_b2 = compute_metrics(y_test, proba_test_blend2)

    logger.info(f"[Blend 1 (50/50 LGB-CB)]  VAL PR-AUC: {val_metrics_b1['pr_auc']:.6f} | ROC-AUC: {val_metrics_b1['roc_auc']:.6f} | Prec: {val_metrics_b1['precision']:.4f} | Rec: {val_metrics_b1['recall']:.4f}")
    logger.info(f"[Blend 2 (50/30/20 Tri)]   VAL PR-AUC: {val_metrics_b2['pr_auc']:.6f} | ROC-AUC: {val_metrics_b2['roc_auc']:.6f} | Prec: {val_metrics_b2['precision']:.4f} | Rec: {val_metrics_b2['recall']:.4f}")

    all_models = {
        "LightGBM_V2": {"val": val_metrics_lgbm, "test": test_metrics_lgbm, "proba_val": proba_val_lgbm, "proba_test": proba_test_lgbm},
        "CatBoost_V2": {"val": val_metrics_cb, "test": test_metrics_cb, "proba_val": proba_val_cb, "proba_test": proba_test_cb},
        "XGBoost_V2": {"val": val_metrics_xgb, "test": test_metrics_xgb, "proba_val": proba_val_xgb, "proba_test": proba_test_xgb},
        "Blend_LGB_CB_50_50": {"val": val_metrics_b1, "test": test_metrics_b1, "proba_val": proba_val_blend1, "proba_test": proba_test_blend1},
        "Blend_Tri_50_30_20": {"val": val_metrics_b2, "test": test_metrics_b2, "proba_val": proba_val_blend2, "proba_test": proba_test_blend2},
    }

    # =========================================================================
    # FIXED-FPR AND PRECISION-TARGET FRONTIER ANALYSIS (VALIDATION ONLY)
    # =========================================================================
    logger.info("\n-------------------------------------------------------------")
    logger.info("  COMPUTING OPERATING FRONTIERS ON VALIDATION")
    logger.info("-------------------------------------------------------------")
    frontiers = {}
    for m_name, m_data in all_models.items():
        p_val = m_data["proba_val"]
        fixed_fpr = find_fixed_fpr_frontier(y_val, p_val)
        prec_target = find_precision_target_frontier(y_val, p_val)
        frontiers[m_name] = {
            "fixed_fpr_analysis": fixed_fpr,
            "precision_target_analysis": prec_target,
        }

    with open(BAKEOFF_DIR / "operating_frontiers.json", "w") as f:
        json.dump(frontiers, f, indent=2)

    # =========================================================================
    # PRIMARY MODEL SELECTION (VALIDATION ONLY)
    # =========================================================================
    # Ranking on Validation PR-AUC (Primary) and ROC-AUC (Secondary)
    winner_name = max(all_models.keys(), key=lambda k: (all_models[k]["val"]["pr_auc"], all_models[k]["val"]["roc_auc"]))
    logger.info(f"\n=====================================================================")
    logger.info(f"  WINNING MODEL SELECTED (VALIDATION ONLY): {winner_name}")
    logger.info(f"  Val PR-AUC: {all_models[winner_name]['val']['pr_auc']:.6f} | Val ROC-AUC: {all_models[winner_name]['val']['roc_auc']:.6f}")
    logger.info("=====================================================================")

    # Optimal operating threshold on validation: Max Recall s.t. FPR <= 1.0%
    opt_operating_point = frontiers[winner_name]["fixed_fpr_analysis"]["FPR_le_1.00pct"]
    opt_threshold = opt_operating_point["threshold"]
    logger.info(f"Validation-Selected Optimal Operating Threshold (Max Rec s.t. FPR<=1.0%): {opt_threshold:.6f}")
    logger.info(f"Val Performance at Opt Threshold: Prec={opt_operating_point['precision']:.4f}, Rec={opt_operating_point['recall']:.4f}, FPR={opt_operating_point['fpr']:.4f}")

    # Evaluate winning model at optimal operating threshold on TEST
    proba_winner_test = all_models[winner_name]["proba_test"]
    opt_test_metrics = compute_metrics(y_test, proba_winner_test, threshold=opt_threshold)
    logger.info(f"TEST Performance at Opt Threshold ({opt_threshold:.4f}): Prec={opt_test_metrics['precision']:.4f}, Rec={opt_test_metrics['recall']:.4f}, FPR={opt_test_metrics['fpr']:.4f}, TP={opt_test_metrics['tp']}, FP={opt_test_metrics['fp']}")

    selection_meta = {
        "winning_model": winner_name,
        "selection_criteria": "Max Validation PR-AUC, followed by Validation ROC-AUC",
        "validation_pr_auc": all_models[winner_name]["val"]["pr_auc"],
        "validation_roc_auc": all_models[winner_name]["val"]["roc_auc"],
        "validation_inherited_threshold_metrics": all_models[winner_name]["val"],
        "validation_opt_threshold_point": opt_operating_point,
        "test_inherited_threshold_metrics": all_models[winner_name]["test"],
        "test_opt_threshold_metrics": opt_test_metrics,
    }
    with open(BAKEOFF_DIR / "final_model_selection.json", "w") as f:
        json.dump(selection_meta, f, indent=2)

    # =========================================================================
    # RUNTIME BENCHMARK ON 10,000 TRANSACTIONS
    # =========================================================================
    logger.info("\n-------------------------------------------------------------")
    logger.info("  RUNNING RUNTIME LATENCY BENCHMARK (10,000 TRANSACTIONS)")
    logger.info("-------------------------------------------------------------")
    N_BENCH = 10000
    X_sample_lgb = X_test_v2.iloc[:N_BENCH].values.astype(np.float32)
    X_sample_cb = X_te_cb.iloc[:N_BENCH]  # CatBoost needs string-category DataFrame

    # Warmup
    _ = lgbm_model.predict_risk(X_sample_lgb[:100])
    _ = cb_model.predict_proba(X_sample_cb.iloc[:100])
    _ = xgb_model.predict_proba(X_sample_lgb[:100])

    def bench_model_lgb(n_bench=N_BENCH):
        times = np.empty(n_bench)
        for i in range(n_bench):
            row = X_sample_lgb[i:i+1]
            t0 = time.perf_counter()
            _ = lgbm_model.predict_risk(row)
            times[i] = time.perf_counter() - t0
        t_b0 = time.perf_counter(); _ = lgbm_model.predict_risk(X_sample_lgb); t_b1 = time.perf_counter()
        return {"mean_ms": round(float(np.mean(times))*1000, 4), "p50_ms": round(float(np.percentile(times, 50))*1000, 4),
                "p95_ms": round(float(np.percentile(times, 95))*1000, 4), "p99_ms": round(float(np.percentile(times, 99))*1000, 4),
                "throughput_txns_sec": round(n_bench / (t_b1 - t_b0))}

    def bench_model_cb(n_bench=N_BENCH):
        times = np.empty(n_bench)
        for i in range(n_bench):
            row = X_sample_cb.iloc[i:i+1]
            t0 = time.perf_counter()
            _ = cb_model.predict_proba(row)[:, 1]
            times[i] = time.perf_counter() - t0
        t_b0 = time.perf_counter(); _ = cb_model.predict_proba(X_sample_cb)[:, 1]; t_b1 = time.perf_counter()
        return {"mean_ms": round(float(np.mean(times))*1000, 4), "p50_ms": round(float(np.percentile(times, 50))*1000, 4),
                "p95_ms": round(float(np.percentile(times, 95))*1000, 4), "p99_ms": round(float(np.percentile(times, 99))*1000, 4),
                "throughput_txns_sec": round(n_bench / (t_b1 - t_b0))}

    def bench_model_xgb(n_bench=N_BENCH):
        times = np.empty(n_bench)
        for i in range(n_bench):
            row = X_sample_lgb[i:i+1]
            t0 = time.perf_counter()
            _ = xgb_model.predict_proba(row)[:, 1]
            times[i] = time.perf_counter() - t0
        t_b0 = time.perf_counter(); _ = xgb_model.predict_proba(X_sample_lgb)[:, 1]; t_b1 = time.perf_counter()
        return {"mean_ms": round(float(np.mean(times))*1000, 4), "p50_ms": round(float(np.percentile(times, 50))*1000, 4),
                "p95_ms": round(float(np.percentile(times, 95))*1000, 4), "p99_ms": round(float(np.percentile(times, 99))*1000, 4),
                "throughput_txns_sec": round(n_bench / (t_b1 - t_b0))}

    def bench_model_blend(n_bench=N_BENCH):
        times = np.empty(n_bench)
        for i in range(n_bench):
            rl = X_sample_lgb[i:i+1]; rc = X_sample_cb.iloc[i:i+1]
            t0 = time.perf_counter()
            _ = 0.5 * lgbm_model.predict_risk(rl) + 0.5 * cb_model.predict_proba(rc)[:, 1]
            times[i] = time.perf_counter() - t0
        t_b0 = time.perf_counter()
        _ = 0.5 * lgbm_model.predict_risk(X_sample_lgb) + 0.5 * cb_model.predict_proba(X_sample_cb)[:, 1]
        t_b1 = time.perf_counter()
        return {"mean_ms": round(float(np.mean(times))*1000, 4), "p50_ms": round(float(np.percentile(times, 50))*1000, 4),
                "p95_ms": round(float(np.percentile(times, 95))*1000, 4), "p99_ms": round(float(np.percentile(times, 99))*1000, 4),
                "throughput_txns_sec": round(n_bench / (t_b1 - t_b0))}

    bench_lgb = bench_model_lgb()
    bench_cb = bench_model_cb()
    bench_xgb = bench_model_xgb()
    bench_b1 = bench_model_blend()

    runtime_data = {
        "LightGBM_V2": bench_lgb,
        "CatBoost_V2": bench_cb,
        "XGBoost_V2": bench_xgb,
        "Blend_LGB_CB_50_50": bench_b1,
    }
    with open(BAKEOFF_DIR / "runtime_benchmark.json", "w") as f:
        json.dump(runtime_data, f, indent=2)

    logger.info(f"Inference p50: LGBM={bench_lgb['p50_ms']}ms, CatBoost={bench_cb['p50_ms']}ms, XGBoost={bench_xgb['p50_ms']}ms, Blend={bench_b1['p50_ms']}ms")

    # =========================================================================
    # CREATE DECISION TABLE (CSV)
    # =========================================================================
    csv_rows = []
    for m_name, m_data in all_models.items():
        v = m_data["val"]
        t = m_data["test"]
        rt = runtime_data.get(m_name, {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "throughput_txns_sec": 0})
        csv_rows.append({
            "Model": m_name,
            "Val PR-AUC": v["pr_auc"],
            "Val ROC-AUC": v["roc_auc"],
            "Val Precision": v["precision"],
            "Val Recall": v["recall"],
            "Val F1": v["f1"],
            "Val FPR": v["fpr"],
            "Test PR-AUC": t["pr_auc"],
            "Test ROC-AUC": t["roc_auc"],
            "Test Precision": t["precision"],
            "Test Recall": t["recall"],
            "Test F1": t["f1"],
            "Test FPR": t["fpr"],
            "Test FP": t["fp"],
            "Test TP": t["tp"],
            "Latency p50 (ms)": rt["p50_ms"],
            "Latency p95 (ms)": rt["p95_ms"],
            "Latency p99 (ms)": rt["p99_ms"],
            "Throughput (txns/sec)": rt["throughput_txns_sec"],
        })

    comp_df = pd.DataFrame(csv_rows)
    comp_df.to_csv(BAKEOFF_DIR / "model_comparison.csv", index=False)
    logger.info(f"Saved decision table -> {BAKEOFF_DIR / 'model_comparison.csv'}")

    # =========================================================================
    # INTEGRATE WINNER WITH FROZEN TRUSTGRAPH PIPELINE
    # =========================================================================
    logger.info("\n-------------------------------------------------------------")
    logger.info(f"  INTEGRATING WINNING MODEL ({winner_name}) INTO FROZEN TRUSTGRAPH")
    logger.info("-------------------------------------------------------------")
    # Feed winning A_t into temporal and relational engines
    def score_winner(X_lgb_arr, X_cb_df):
        if "Blend_LGB_CB" in winner_name:
            return 0.5 * lgbm_model.predict_risk(X_lgb_arr) + 0.5 * cb_model.predict_proba(X_cb_df)[:, 1]
        elif "Blend_Tri" in winner_name:
            return 0.5 * lgbm_model.predict_risk(X_lgb_arr) + 0.3 * cb_model.predict_proba(X_cb_df)[:, 1] + 0.2 * xgb_model.predict_proba(X_lgb_arr)[:, 1]
        elif "CatBoost" in winner_name:
            return cb_model.predict_proba(X_cb_df)[:, 1]
        elif "XGBoost" in winner_name:
            return xgb_model.predict_proba(X_lgb_arr)[:, 1]
        else:
            return lgbm_model.predict_risk(X_lgb_arr)

    # Run frozen temporal
    temp_engine = EntityTemporalRiskEngine(beta=0.3, gamma=0.5, lambda_=0.05, delta=0.05)
    p_tr_win = score_winner(X_train_v2.values.astype(np.float32), X_tr_cb)
    p_va_win = score_winner(X_val_v2.values.astype(np.float32), X_va_cb)

    for ents, scores in [(train_df["entity_proxy"].values, p_tr_win), (val_df["entity_proxy"].values, p_va_win)]:
        for i in range(len(ents)):
            temp_engine.step(str(ents[i]), float(scores[i]))

    # Test stream
    P_test_win = np.empty(len(test_df), dtype=float)
    E_test_win = np.empty(len(test_df), dtype=float)
    test_ents = test_df["entity_proxy"].values
    for i in range(len(test_df)):
        e_val, p_val = temp_engine.step(str(test_ents[i]), float(proba_winner_test[i]))
        E_test_win[i] = e_val
        P_test_win[i] = p_val

    # Run frozen relational
    rel_params = GraphParameters(k_attr_max=25, window_sec=86400.0, d_ref=3.0, v_ref=10.0, w_D=0.6, w_V=0.4, relational_attrs=("DeviceInfo",))
    graph_engine = LightweightRelationalGraph(rel_params)
    graph_engine.fit_attribute_frequency_ceiling(train_df)
    process_partition(train_df, graph_engine)
    process_partition(val_df, graph_engine)
    records = process_partition(test_df, graph_engine)

    G_test_win = np.array([r.G_t for r in records], dtype=float)

    # Fusion
    R_t_winner = apply_fusion_rule("F1", proba_winner_test, P_test_win, G_test_win, {"alpha": 1.0, "beta": 0.05})
    actions_winner, bands_winner = batch_assign_actions(R_t_winner, POLICY_THRESHOLDS)

    test_df["A_t_winner"] = proba_winner_test
    test_df["P_t_winner"] = P_test_win
    test_df["G_t_winner"] = G_test_win
    test_df["R_t_winner"] = R_t_winner
    test_df["action_winner"] = actions_winner

    # Metrics for Fused System
    m_fused_win = compute_metrics(y_test, R_t_winner)
    m_policy_tier3_win = compute_metrics(y_test, R_t_winner, threshold=0.80)

    logger.info(f"[TRUSTGRAPH + {winner_name}] Fused R_t (tau=0.5943): Prec={m_fused_win['precision']:.4f}, Rec={m_fused_win['recall']:.4f}, F1={m_fused_win['f1']:.4f}, FPR={m_fused_win['fpr']:.4f}")
    logger.info(f"[TRUSTGRAPH + {winner_name}] Policy BLOCK (tau=0.80): Prec={m_policy_tier3_win['precision']:.4f}, Rec={m_policy_tier3_win['recall']:.4f}, False Blocks={m_policy_tier3_win['fp']}")

    trustgraph_integration_results = {
        "winning_model": winner_name,
        "fused_system_metrics_frozen_threshold": m_fused_win,
        "policy_block_tier_metrics": m_policy_tier3_win,
    }
    with open(BAKEOFF_DIR / "trustgraph_winner_integration.json", "w") as f:
        json.dump(trustgraph_integration_results, f, indent=2)

    # =========================================================================
    # REAL SLOW-BURN TRAJECTORY FOR WINNER
    # =========================================================================
    logger.info("Extracting real slow-burn demonstration for the winning model...")
    # Find candidate trajectory
    cand_idx = np.where((y_test == 1) & (test_df["A_t_winner"].values < FROZEN_THRESHOLD) & (test_df["R_t_winner"].values >= FROZEN_THRESHOLD))[0]
    best_traj = None
    for idx in cand_idx:
        ent = test_df["entity_proxy"].iloc[idx]
        if ent.startswith("unresolved_"):
            continue
        g_ent = test_df[test_df["entity_proxy"] == ent].sort_values("TransactionDT").reset_index(drop=True)
        if len(g_ent) >= 2:
            events = []
            for _, tr in g_ent.iterrows():
                events.append({
                    "TransactionID": int(tr["TransactionID"]),
                    "TransactionDT": float(tr["TransactionDT"]),
                    "TransactionAmt": float(tr["TransactionAmt"]),
                    "isFraud": int(tr["isFraud"]),
                    "A_t": round(float(tr["A_t_winner"]), 6),
                    "P_t": round(float(tr["P_t_winner"]), 6),
                    "G_t": round(float(tr["G_t_winner"]), 6),
                    "R_t": round(float(tr["R_t_winner"]), 6),
                    "baseline_decision": "ALLOW" if tr["A_t_winner"] < FROZEN_THRESHOLD else "BLOCK",
                    "trustgraph_decision": str(tr["action_winner"]),
                })
            best_traj = {
                "entity_proxy": ent,
                "n_transactions": len(g_ent),
                "transactions": events,
            }
            break

    with open(BAKEOFF_DIR / "winner_slow_burn_trajectory.json", "w") as f:
        json.dump(best_traj, f, indent=2)

    # Save full test bakeoff predictions
    bakeoff_preds = pd.DataFrame({
        "TransactionID": test_df["TransactionID"].values,
        "TransactionDT": test_df["TransactionDT"].values,
        "isFraud": y_test,
        "A_t_LGBM": proba_test_lgbm,
        "A_t_CatBoost": proba_test_cb,
        "A_t_XGBoost": proba_test_xgb,
        "A_t_Blend1": proba_test_blend1,
        "A_t_Blend2": proba_test_blend2,
        "A_t_Winner": proba_winner_test,
        "R_t_Winner": R_t_winner,
        "action_Winner": actions_winner,
    })
    bakeoff_preds.to_csv(cfg.PROJECT_ROOT / "results" / "model_bakeoff_predictions.csv", index=False)
    logger.info("Saved model_bakeoff_predictions.csv")

    # Plots: PR and ROC curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    from sklearn.metrics import precision_recall_curve, roc_curve
    for m_name in ["LightGBM_V2", "CatBoost_V2", "XGBoost_V2", "Blend_LGB_CB_50_50"]:
        p_t = all_models[m_name]["proba_test"]
        prec, rec, _ = precision_recall_curve(y_test, p_t)
        pr_auc = average_precision_score(y_test, p_t)
        plt.plot(rec, prec, label=f"{m_name} (PR-AUC={pr_auc:.3f})")
    plt.title("Held-out TEST Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for m_name in ["LightGBM_V2", "CatBoost_V2", "XGBoost_V2", "Blend_LGB_CB_50_50"]:
        p_t = all_models[m_name]["proba_test"]
        fpr_arr, tpr_arr, _ = roc_curve(y_test, p_t)
        auc_val = roc_auc_score(y_test, p_t)
        plt.plot(fpr_arr, tpr_arr, label=f"{m_name} (ROC-AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.title("Held-out TEST ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "bakeoff_pr_roc_curves.png", dpi=300)
    plt.close()

    logger.info("Model bake-off completed successfully.")


if __name__ == "__main__":
    main()
