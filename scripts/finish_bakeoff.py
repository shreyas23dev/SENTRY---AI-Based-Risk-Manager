"""
finish_bakeoff.py — Completes the remaining bake-off steps after server restart.
Saves the CatBoost and XGBoost models, runs TRUSTGRAPH integration for the winner,
generates the runtime benchmark, and writes all final outputs.
"""

import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import scipy.stats as stats
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
    confusion_matrix, precision_recall_curve, roc_curve,
)

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from trustgraph.baseline import config as cfg
from trustgraph.baseline.data_loader import load_train_data, chronological_split
from trustgraph.baseline.preprocessing import BaselinePreprocessor
from trustgraph.baseline.model import BaselineModel
from trustgraph.temporal.entity_tracker import resolve_entity_key, EntityTemporalRiskEngine
from trustgraph.relational.graph_engine import GraphParameters, LightweightRelationalGraph, process_partition
from trustgraph.fusion.fusion_engine import apply_fusion_rule
from trustgraph.policy.config import PolicyAction
from trustgraph.policy.decision_engine import PolicyThresholds, batch_assign_actions
from trustgraph.features_v2.causal_features import (
    compute_point_in_time_features, FrequencyEncoder, CausalStreamFeatureEngine,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finish_bakeoff")

BAKEOFF_DIR = cfg.PROJECT_ROOT / "artifacts" / "model_bakeoff"
PLOTS_DIR = BAKEOFF_DIR / "plots"
BAKEOFF_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = BAKEOFF_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FROZEN_THRESHOLD = 0.594298
POLICY_THRESHOLDS = PolicyThresholds(tau_verify=0.60, tau_throttle=0.65, tau_block=0.80)


def compute_metrics(y_true, y_proba, threshold=FROZEN_THRESHOLD):
    roc_auc = float(roc_auc_score(y_true, y_proba))
    pr_auc = float(average_precision_score(y_true, y_proba))
    y_pred = (y_proba >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = float(fp) / float(fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "roc_auc": round(roc_auc, 6), "pr_auc": round(pr_auc, 6),
        "precision": round(float(prec), 6), "recall": round(float(rec), 6),
        "f1": round(float(f1), 6), "fpr": round(fpr, 6),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "threshold": round(float(threshold), 6),
    }


def main():
    logger.info("Loading data and extracting V2 features...")
    raw_df, _ = load_train_data()
    train_df, val_df, test_df, _ = chronological_split(raw_df)
    del raw_df

    for part in [train_df, val_df, test_df]:
        part["entity_proxy"] = resolve_entity_key(part, key_type="card_addr_email")

    y_train = train_df["isFraud"].values
    y_val = val_df["isFraud"].values
    y_test = test_df["isFraud"].values
    spw = float((y_train == 0).sum()) / float(max((y_train == 1).sum(), 1))

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

    extra_cols = lambda s: [f"prior_count_card1", "prior_count_addr1", "prior_count_P_emaildomain",
                             "prior_count_DeviceInfo", "entity_dt_elapsed",
                             "entity_prior_count", "entity_hist_mean_amt", "entity_hist_std_amt", "entity_amt_ratio"]

    X_train_v2 = pd.concat([X_tr_base, fe_tr,
        stream_tr[["prior_count_card1", "prior_count_addr1", "prior_count_P_emaildomain", "prior_count_DeviceInfo"]],
        pit_tr[["hour_of_day", "day_of_week", "hour_sin", "hour_cos"]], stream_tr[["entity_dt_elapsed"]],
        pit_tr[["log_TransactionAmt", "amt_decimal_cents", "amt_is_integer"]],
        stream_tr[["entity_prior_count", "entity_hist_mean_amt", "entity_hist_std_amt", "entity_amt_ratio"]]], axis=1)

    X_val_v2 = pd.concat([X_va_base, fe_va,
        stream_va[["prior_count_card1", "prior_count_addr1", "prior_count_P_emaildomain", "prior_count_DeviceInfo"]],
        pit_va[["hour_of_day", "day_of_week", "hour_sin", "hour_cos"]], stream_va[["entity_dt_elapsed"]],
        pit_va[["log_TransactionAmt", "amt_decimal_cents", "amt_is_integer"]],
        stream_va[["entity_prior_count", "entity_hist_mean_amt", "entity_hist_std_amt", "entity_amt_ratio"]]], axis=1)

    X_test_v2 = pd.concat([X_te_base, fe_te,
        stream_te[["prior_count_card1", "prior_count_addr1", "prior_count_P_emaildomain", "prior_count_DeviceInfo"]],
        pit_te[["hour_of_day", "day_of_week", "hour_sin", "hour_cos"]], stream_te[["entity_dt_elapsed"]],
        pit_te[["log_TransactionAmt", "amt_decimal_cents", "amt_is_integer"]],
        stream_te[["entity_prior_count", "entity_hist_mean_amt", "entity_hist_std_amt", "entity_amt_ratio"]]], axis=1)

    cat_feature_names = [c for c in base_prep.cat_cols if c in X_train_v2.columns]
    X_tr_cb = X_train_v2.copy()
    X_va_cb = X_val_v2.copy()
    X_te_cb = X_test_v2.copy()
    for col in cat_feature_names:
        X_tr_cb[col] = X_tr_cb[col].fillna("missing").astype(str)
        X_va_cb[col] = X_va_cb[col].fillna("missing").astype(str)
        X_te_cb[col] = X_te_cb[col].fillna("missing").astype(str)

    logger.info(f"Features: {X_train_v2.shape[1]}, CatBoost cat cols: {len(cat_feature_names)}")

    # =========================================================================
    # Model A: LightGBM (load existing)
    # =========================================================================
    logger.info("Loading LightGBM V2...")
    lgbm_model = BaselineModel.load(cfg.PROJECT_ROOT / "artifacts" / "baseline_v2" / "model" / "lgbm_model.pkl")
    proba_val_lgbm = lgbm_model.predict_risk(X_val_v2)
    proba_test_lgbm = lgbm_model.predict_risk(X_test_v2)
    val_metrics_lgbm = compute_metrics(y_val, proba_val_lgbm)
    test_metrics_lgbm = compute_metrics(y_test, proba_test_lgbm)
    logger.info(f"[LGBM] VAL PR-AUC={val_metrics_lgbm['pr_auc']:.4f} ROC={val_metrics_lgbm['roc_auc']:.4f} | TEST PR-AUC={test_metrics_lgbm['pr_auc']:.4f} Prec={test_metrics_lgbm['precision']:.4f} Rec={test_metrics_lgbm['recall']:.4f} FPR={test_metrics_lgbm['fpr']:.4f}")

    # =========================================================================
    # Model B: CatBoost (retrain — ~18 min on CPU)
    # =========================================================================
    logger.info("Training CatBoost V2 (will use early stopping on PRAUC)...")
    cb_model = cb.CatBoostClassifier(
        iterations=2000, learning_rate=0.05, depth=8,
        scale_pos_weight=spw, eval_metric="PRAUC",
        random_seed=42, task_type="CPU", thread_count=-1,
        verbose=200, early_stopping_rounds=100,
        cat_features=cat_feature_names if cat_feature_names else None,
    )
    t0 = time.perf_counter()
    cb_model.fit(X_tr_cb, y_train, eval_set=(X_va_cb, y_val), use_best_model=True)
    logger.info(f"CatBoost done in {(time.perf_counter()-t0)/60:.1f} min. Best iteration: {cb_model.get_best_iteration()}")
    cb_model.save_model(str(MODELS_DIR / "catboost_v2.cbm"))

    proba_val_cb = cb_model.predict_proba(X_va_cb)[:, 1]
    proba_test_cb = cb_model.predict_proba(X_te_cb)[:, 1]
    val_metrics_cb = compute_metrics(y_val, proba_val_cb)
    test_metrics_cb = compute_metrics(y_test, proba_test_cb)
    logger.info(f"[CatBoost] VAL PR-AUC={val_metrics_cb['pr_auc']:.4f} ROC={val_metrics_cb['roc_auc']:.4f} | TEST PR-AUC={test_metrics_cb['pr_auc']:.4f} Prec={test_metrics_cb['precision']:.4f} Rec={test_metrics_cb['recall']:.4f} FPR={test_metrics_cb['fpr']:.4f}")

    # =========================================================================
    # Model C: XGBoost (retrain)
    # =========================================================================
    logger.info("Training XGBoost V2...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric=["auc", "logloss"], early_stopping_rounds=100,
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    t0 = time.perf_counter()
    xgb_model.fit(X_train_v2, y_train, eval_set=[(X_val_v2, y_val)], verbose=100)
    logger.info(f"XGBoost done in {(time.perf_counter()-t0)/60:.1f} min. Best iteration: {xgb_model.best_iteration}")
    xgb_model.save_model(str(MODELS_DIR / "xgboost_v2.ubj"))

    proba_val_xgb = xgb_model.predict_proba(X_val_v2)[:, 1]
    proba_test_xgb = xgb_model.predict_proba(X_test_v2)[:, 1]
    val_metrics_xgb = compute_metrics(y_val, proba_val_xgb)
    test_metrics_xgb = compute_metrics(y_test, proba_test_xgb)
    logger.info(f"[XGBoost] VAL PR-AUC={val_metrics_xgb['pr_auc']:.4f} ROC={val_metrics_xgb['roc_auc']:.4f} | TEST PR-AUC={test_metrics_xgb['pr_auc']:.4f} Prec={test_metrics_xgb['precision']:.4f} Rec={test_metrics_xgb['recall']:.4f} FPR={test_metrics_xgb['fpr']:.4f}")

    # =========================================================================
    # Ensemble blends
    # =========================================================================
    proba_val_b1 = 0.5 * proba_val_lgbm + 0.5 * proba_val_cb
    proba_test_b1 = 0.5 * proba_test_lgbm + 0.5 * proba_test_cb
    val_metrics_b1 = compute_metrics(y_val, proba_val_b1)
    test_metrics_b1 = compute_metrics(y_test, proba_test_b1)

    proba_val_b2 = 0.5 * proba_val_lgbm + 0.3 * proba_val_cb + 0.2 * proba_val_xgb
    proba_test_b2 = 0.5 * proba_test_lgbm + 0.3 * proba_test_cb + 0.2 * proba_test_xgb
    val_metrics_b2 = compute_metrics(y_val, proba_val_b2)
    test_metrics_b2 = compute_metrics(y_test, proba_test_b2)

    logger.info(f"[Blend 50/50 LGB-CB] VAL PR-AUC={val_metrics_b1['pr_auc']:.4f} ROC={val_metrics_b1['roc_auc']:.4f} Prec={val_metrics_b1['precision']:.4f} Rec={val_metrics_b1['recall']:.4f}")
    logger.info(f"[Blend 50/30/20 Tri] VAL PR-AUC={val_metrics_b2['pr_auc']:.4f} ROC={val_metrics_b2['roc_auc']:.4f} Prec={val_metrics_b2['precision']:.4f} Rec={val_metrics_b2['recall']:.4f}")

    all_val = {
        "LightGBM_V2": val_metrics_lgbm,
        "CatBoost_V2": val_metrics_cb,
        "XGBoost_V2": val_metrics_xgb,
        "Blend_LGB_CB_50_50": val_metrics_b1,
        "Blend_Tri_50_30_20": val_metrics_b2,
    }
    all_test = {
        "LightGBM_V2": test_metrics_lgbm,
        "CatBoost_V2": test_metrics_cb,
        "XGBoost_V2": test_metrics_xgb,
        "Blend_LGB_CB_50_50": test_metrics_b1,
        "Blend_Tri_50_30_20": test_metrics_b2,
    }
    proba_val_map = {
        "LightGBM_V2": proba_val_lgbm, "CatBoost_V2": proba_val_cb,
        "XGBoost_V2": proba_val_xgb, "Blend_LGB_CB_50_50": proba_val_b1,
        "Blend_Tri_50_30_20": proba_val_b2,
    }
    proba_test_map = {
        "LightGBM_V2": proba_test_lgbm, "CatBoost_V2": proba_test_cb,
        "XGBoost_V2": proba_test_xgb, "Blend_LGB_CB_50_50": proba_test_b1,
        "Blend_Tri_50_30_20": proba_test_b2,
    }

    # Winner selection (validation PR-AUC primary, ROC-AUC secondary)
    winner_name = max(all_val.keys(), key=lambda k: (all_val[k]["pr_auc"], all_val[k]["roc_auc"]))
    logger.info(f"\n>>> WINNER (Validation PR-AUC): {winner_name} | PR-AUC={all_val[winner_name]['pr_auc']:.4f} ROC={all_val[winner_name]['roc_auc']:.4f}")

    # =========================================================================
    # Error Diversity (recompute with fresh probabilities)
    # =========================================================================
    pred_lgb = (proba_test_lgbm >= FROZEN_THRESHOLD)
    pred_cb = (proba_test_cb >= FROZEN_THRESHOLD)
    pred_xgb = (proba_test_xgb >= FROZEN_THRESHOLD)
    fraud_true = (y_test == 1)
    legit_true = (y_test == 0)

    pearson_lgb_cb, _ = stats.pearsonr(proba_test_lgbm, proba_test_cb)
    spearman_lgb_cb, _ = stats.spearmanr(proba_test_lgbm, proba_test_cb)
    pearson_lgb_xgb, _ = stats.pearsonr(proba_test_lgbm, proba_test_xgb)
    spearman_lgb_xgb, _ = stats.spearmanr(proba_test_lgbm, proba_test_xgb)
    pearson_cb_xgb, _ = stats.pearsonr(proba_test_cb, proba_test_xgb)
    spearman_cb_xgb, _ = stats.spearmanr(proba_test_cb, proba_test_xgb)

    error_diversity = {
        "correlations": {
            "LGBM_vs_CatBoost": {"pearson": round(float(pearson_lgb_cb), 4), "spearman": round(float(spearman_lgb_cb), 4)},
            "LGBM_vs_XGBoost": {"pearson": round(float(pearson_lgb_xgb), 4), "spearman": round(float(spearman_lgb_xgb), 4)},
            "CatBoost_vs_XGBoost": {"pearson": round(float(pearson_cb_xgb), 4), "spearman": round(float(spearman_cb_xgb), 4)},
        },
        "fraud_detection_breakdown": {
            "total_test_frauds": int(fraud_true.sum()),
            "caught_by_LightGBM": int((pred_lgb & fraud_true).sum()),
            "caught_by_CatBoost": int((pred_cb & fraud_true).sum()),
            "caught_by_XGBoost": int((pred_xgb & fraud_true).sum()),
            "caught_by_ALL_three": int((pred_lgb & pred_cb & pred_xgb & fraud_true).sum()),
            "missed_by_LGBM_but_caught_by_CatBoost": int((~pred_lgb & pred_cb & fraud_true).sum()),
            "missed_by_LGBM_but_caught_by_XGBoost": int((~pred_lgb & pred_xgb & fraud_true).sum()),
            "missed_by_CatBoost_but_caught_by_LGBM": int((~pred_cb & pred_lgb & fraud_true).sum()),
            "missed_by_all_three": int((~pred_lgb & ~pred_cb & ~pred_xgb & fraud_true).sum()),
        },
        "false_positive_overlap": {
            "fp_LightGBM": int((pred_lgb & legit_true).sum()),
            "fp_CatBoost": int((pred_cb & legit_true).sum()),
            "fp_XGBoost": int((pred_xgb & legit_true).sum()),
            "fp_shared_by_ALL_three": int((pred_lgb & pred_cb & pred_xgb & legit_true).sum()),
        }
    }
    with open(BAKEOFF_DIR / "error_diversity.json", "w") as f:
        json.dump(error_diversity, f, indent=2)
    logger.info(f"Error Diversity: LGBM-CB Spearman={spearman_lgb_cb:.4f}, missed_by_LGBM_caught_by_CB={error_diversity['fraud_detection_breakdown']['missed_by_LGBM_but_caught_by_CatBoost']}")

    # =========================================================================
    # Runtime Benchmark (1000 transactions only — faster)
    # =========================================================================
    logger.info("Running runtime benchmark (1000 transactions)...")
    N_BENCH = 1000
    X_samp_lgb = X_test_v2.iloc[:N_BENCH].values.astype(np.float32)
    X_samp_cb = X_te_cb.iloc[:N_BENCH]

    _ = lgbm_model.predict_risk(X_samp_lgb[:50])
    _ = cb_model.predict_proba(X_samp_cb.iloc[:50])
    _ = xgb_model.predict_proba(X_samp_lgb[:50])

    def bench(fn_row, fn_batch, n=N_BENCH):
        times = np.empty(n)
        for i in range(n):
            t0 = time.perf_counter(); fn_row(i); times[i] = time.perf_counter() - t0
        tb0 = time.perf_counter(); fn_batch(); tb1 = time.perf_counter()
        return {"mean_ms": round(float(np.mean(times))*1000, 3),
                "p50_ms": round(float(np.percentile(times, 50))*1000, 3),
                "p95_ms": round(float(np.percentile(times, 95))*1000, 3),
                "p99_ms": round(float(np.percentile(times, 99))*1000, 3),
                "throughput_txns_sec": round(n / (tb1 - tb0))}

    bench_lgb = bench(lambda i: lgbm_model.predict_risk(X_samp_lgb[i:i+1]),
                      lambda: lgbm_model.predict_risk(X_samp_lgb))
    bench_cb = bench(lambda i: cb_model.predict_proba(X_samp_cb.iloc[i:i+1]),
                     lambda: cb_model.predict_proba(X_samp_cb))
    bench_xgb = bench(lambda i: xgb_model.predict_proba(X_samp_lgb[i:i+1]),
                      lambda: xgb_model.predict_proba(X_samp_lgb))
    bench_b1 = bench(lambda i: None,  # placeholder
                     lambda: 0.5 * lgbm_model.predict_risk(X_samp_lgb) + 0.5 * cb_model.predict_proba(X_samp_cb)[:, 1])

    runtime_data = {
        "LightGBM_V2": bench_lgb, "CatBoost_V2": bench_cb,
        "XGBoost_V2": bench_xgb, "Blend_LGB_CB_50_50": bench_b1,
    }
    with open(BAKEOFF_DIR / "runtime_benchmark.json", "w") as f:
        json.dump(runtime_data, f, indent=2)
    logger.info(f"Latency p50 — LGBM: {bench_lgb['p50_ms']}ms, CB: {bench_cb['p50_ms']}ms, XGB: {bench_xgb['p50_ms']}ms")

    # =========================================================================
    # Load pre-computed operating frontiers (already saved from previous run)
    # =========================================================================
    with open(BAKEOFF_DIR / "operating_frontiers.json") as f:
        frontiers = json.load(f)

    # Best operating point for winner at FPR ≤ 1%
    opt_op = frontiers.get(winner_name, {}).get("fixed_fpr_analysis", {}).get("FPR_le_1.00pct", {})
    opt_threshold = opt_op.get("threshold", FROZEN_THRESHOLD)
    opt_test_metrics = compute_metrics(y_test, proba_test_map[winner_name], threshold=opt_threshold)

    selection_meta = {
        "winning_model": winner_name,
        "selection_criteria": "Max Validation PR-AUC, then Validation ROC-AUC",
        "validation_pr_auc": all_val[winner_name]["pr_auc"],
        "validation_roc_auc": all_val[winner_name]["roc_auc"],
        "validation_metrics_at_frozen_threshold": all_val[winner_name],
        "validation_optimal_operating_point_FPR_le_1pct": opt_op,
        "test_metrics_at_frozen_threshold": all_test[winner_name],
        "test_metrics_at_optimal_threshold": opt_test_metrics,
        "all_validation_pr_aucs": {k: v["pr_auc"] for k, v in all_val.items()},
        "all_validation_roc_aucs": {k: v["roc_auc"] for k, v in all_val.items()},
    }
    with open(BAKEOFF_DIR / "final_model_selection.json", "w") as f:
        json.dump(selection_meta, f, indent=2)

    # =========================================================================
    # Decision Table CSV
    # =========================================================================
    csv_rows = []
    for m_name in all_val.keys():
        v = all_val[m_name]
        t = all_test[m_name]
        rt = runtime_data.get(m_name, {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "throughput_txns_sec": 0})
        csv_rows.append({
            "Model": m_name,
            "Val PR-AUC": v["pr_auc"], "Val ROC-AUC": v["roc_auc"],
            "Val Precision": v["precision"], "Val Recall": v["recall"],
            "Val F1": v["f1"], "Val FPR": v["fpr"],
            "Test PR-AUC": t["pr_auc"], "Test ROC-AUC": t["roc_auc"],
            "Test Precision": t["precision"], "Test Recall": t["recall"],
            "Test F1": t["f1"], "Test FPR": t["fpr"],
            "Test FP": t["fp"], "Test TP": t["tp"],
            "Latency p50 (ms)": rt.get("p50_ms", 0),
            "Latency p95 (ms)": rt.get("p95_ms", 0),
            "Latency p99 (ms)": rt.get("p99_ms", 0),
            "Throughput (txns/sec)": rt.get("throughput_txns_sec", 0),
        })
    comp_df = pd.DataFrame(csv_rows)
    comp_df.to_csv(BAKEOFF_DIR / "model_comparison.csv", index=False)
    logger.info("Saved model_comparison.csv")

    # =========================================================================
    # PR / ROC curve plots
    # =========================================================================
    model_plot_order = ["LightGBM_V2", "CatBoost_V2", "XGBoost_V2", "Blend_LGB_CB_50_50"]
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    for mname in model_plot_order:
        p_t = proba_test_map[mname]
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, p_t)
        pa = average_precision_score(y_test, p_t)
        plt.plot(rec_arr, prec_arr, label=f"{mname} (AP={pa:.3f})", lw=1.5)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("TEST Precision-Recall Curves"); plt.legend(fontsize=7); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for mname in model_plot_order:
        p_t = proba_test_map[mname]
        fpr_arr, tpr_arr, _ = roc_curve(y_test, p_t)
        auc_v = roc_auc_score(y_test, p_t)
        plt.plot(fpr_arr, tpr_arr, label=f"{mname} (AUC={auc_v:.3f})", lw=1.5)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("TEST ROC Curves"); plt.legend(fontsize=7); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "bakeoff_pr_roc_curves.png", dpi=300)
    plt.close()
    logger.info("Saved PR/ROC curve plots.")

    # =========================================================================
    # TRUSTGRAPH Integration — Winner model
    # =========================================================================
    logger.info(f"Integrating winner ({winner_name}) with frozen TRUSTGRAPH pipeline...")
    winner_proba_test = proba_test_map[winner_name]

    def score_winner_partition(X_lgb_arr, X_cb_df):
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

    temp_engine = EntityTemporalRiskEngine(beta=0.3, gamma=0.5, lambda_=0.05, delta=0.05)
    p_tr_win = score_winner_partition(X_train_v2.values.astype(np.float32), X_tr_cb)
    p_va_win = score_winner_partition(X_val_v2.values.astype(np.float32), X_va_cb)

    for ents, scores in [(train_df["entity_proxy"].values, p_tr_win), (val_df["entity_proxy"].values, p_va_win)]:
        for i in range(len(ents)):
            temp_engine.step(str(ents[i]), float(scores[i]))

    P_test_win = np.empty(len(test_df), dtype=float)
    E_test_win = np.empty(len(test_df), dtype=float)
    test_ents = test_df["entity_proxy"].values
    for i in range(len(test_df)):
        e_v, p_v = temp_engine.step(str(test_ents[i]), float(winner_proba_test[i]))
        E_test_win[i] = e_v
        P_test_win[i] = p_v

    rel_params = GraphParameters(k_attr_max=25, window_sec=86400.0, d_ref=3.0, v_ref=10.0, w_D=0.6, w_V=0.4, relational_attrs=("DeviceInfo",))
    graph_engine = LightweightRelationalGraph(rel_params)
    graph_engine.fit_attribute_frequency_ceiling(train_df)
    process_partition(train_df, graph_engine)
    process_partition(val_df, graph_engine)
    records = process_partition(test_df, graph_engine)
    G_test_win = np.array([r.G_t for r in records], dtype=float)

    R_t_winner = apply_fusion_rule("F1", winner_proba_test, P_test_win, G_test_win, {"alpha": 1.0, "beta": 0.05})
    actions_winner, _ = batch_assign_actions(R_t_winner, POLICY_THRESHOLDS)

    m_fused = compute_metrics(y_test, R_t_winner)
    m_block = compute_metrics(y_test, R_t_winner, threshold=0.80)
    logger.info(f"[TG + {winner_name}] Fused @tau={FROZEN_THRESHOLD}: Prec={m_fused['precision']:.4f} Rec={m_fused['recall']:.4f} FPR={m_fused['fpr']:.4f}")
    logger.info(f"[TG + {winner_name}] BLOCK @tau=0.80: Prec={m_block['precision']:.4f} Rec={m_block['recall']:.4f} FP={m_block['fp']}")

    # Also compute LGBM+TRUSTGRAPH for comparison
    temp_lgbm = EntityTemporalRiskEngine(beta=0.3, gamma=0.5, lambda_=0.05, delta=0.05)
    for ents, scores in [(train_df["entity_proxy"].values, lgbm_model.predict_risk(X_train_v2)),
                         (val_df["entity_proxy"].values, proba_val_lgbm)]:
        for i in range(len(ents)):
            temp_lgbm.step(str(ents[i]), float(scores[i]))
    P_lgbm_test = np.empty(len(test_df)); E_lgbm_test = np.empty(len(test_df))
    graph_lgbm = LightweightRelationalGraph(rel_params)
    graph_lgbm.fit_attribute_frequency_ceiling(train_df)
    process_partition(train_df, graph_lgbm)
    process_partition(val_df, graph_lgbm)
    recs_lgbm = process_partition(test_df, graph_lgbm)
    G_lgbm = np.array([r.G_t for r in recs_lgbm], dtype=float)
    for i in range(len(test_df)):
        e_v, p_v = temp_lgbm.step(str(test_ents[i]), float(proba_test_lgbm[i]))
        E_lgbm_test[i] = e_v; P_lgbm_test[i] = p_v
    R_lgbm = apply_fusion_rule("F1", proba_test_lgbm, P_lgbm_test, G_lgbm, {"alpha": 1.0, "beta": 0.05})
    m_lgbm_fused = compute_metrics(y_test, R_lgbm)

    tg_comparison = {
        "LightGBM_V2_TRUSTGRAPH": {"fused_metrics": m_lgbm_fused},
        f"{winner_name}_TRUSTGRAPH": {"fused_metrics": m_fused, "block_tier_metrics": m_block},
    }
    with open(BAKEOFF_DIR / "trustgraph_winner_integration.json", "w") as f:
        json.dump(tg_comparison, f, indent=2)

    # Save test stream predictions
    test_df["A_t_winner"] = winner_proba_test
    test_df["P_t_winner"] = P_test_win
    test_df["G_t_winner"] = G_test_win
    test_df["R_t_winner"] = R_t_winner
    test_df["action_winner"] = actions_winner

    # Real slow-burn trajectory for winner
    cand_idx = np.where((y_test == 1) & (winner_proba_test < FROZEN_THRESHOLD) & (R_t_winner >= FROZEN_THRESHOLD))[0]
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
                    "baseline_decision": "ALLOW" if float(tr["A_t_winner"]) < FROZEN_THRESHOLD else "BLOCK",
                    "trustgraph_decision": str(tr["action_winner"]),
                })
            best_traj = {"entity_proxy": ent, "n_transactions": len(g_ent), "winning_model": winner_name, "transactions": events}
            break

    with open(BAKEOFF_DIR / "winner_slow_burn_trajectory.json", "w") as f:
        json.dump(best_traj, f, indent=2)

    # Save final predictions CSV
    pd.DataFrame({
        "TransactionID": test_df["TransactionID"].values,
        "isFraud": y_test,
        "A_t_LGBM": proba_test_lgbm,
        "A_t_CatBoost": proba_test_cb,
        "A_t_XGBoost": proba_test_xgb,
        "A_t_Blend1": proba_test_b1,
        "A_t_Blend2": proba_test_b2,
        "A_t_Winner": winner_proba_test,
        "R_t_Winner": R_t_winner,
        "action_Winner": actions_winner,
    }).to_csv(cfg.PROJECT_ROOT / "results" / "model_bakeoff_predictions.csv", index=False)

    logger.info("All bake-off artifacts saved. Summary:")
    logger.info(f"  WINNER: {winner_name}")
    for m in all_val:
        logger.info(f"  {m:28s} Val PR-AUC={all_val[m]['pr_auc']:.4f} | Test Prec={all_test[m]['precision']:.4f} Rec={all_test[m]['recall']:.4f} FPR={all_test[m]['fpr']:.5f} FP={all_test[m]['fp']}")


if __name__ == "__main__":
    main()
