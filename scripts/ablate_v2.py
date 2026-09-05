"""
ablate_v2.py — Baseline V2 Causal Feature Engineering & Ablation Suite
======================================================================

Runs controlled ablation experiments on the VALIDATION partition:
  - V2-A: Baseline + Frequency Features
  - V2-B: Baseline + Time Features
  - V2-C: Baseline + Amount Features
  - V2-D: Baseline + Historical Aggregation Features
  - V2-E: Combined Best Features (Selected on VALIDATION ONLY)

Then freezes the winning configuration and evaluates once on TEST.
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
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix

from trustgraph.baseline import config as cfg
from trustgraph.baseline.data_loader import load_train_data, chronological_split
from trustgraph.baseline.preprocessing import BaselinePreprocessor
from trustgraph.baseline.model import BaselineModel
from trustgraph.temporal.entity_tracker import resolve_entity_key
from trustgraph.features_v2.causal_features import (
    compute_point_in_time_features,
    FrequencyEncoder,
    CausalStreamFeatureEngine,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ablate_v2")

ARTIFACTS_V2_DIR = cfg.PROJECT_ROOT / "artifacts" / "baseline_v2"
MODEL_V2_DIR = ARTIFACTS_V2_DIR / "model"
PREPROC_V2_DIR = ARTIFACTS_V2_DIR / "preprocessing"
METRICS_V2_DIR = ARTIFACTS_V2_DIR / "metrics"

for d in [ARTIFACTS_V2_DIR, MODEL_V2_DIR, PREPROC_V2_DIR, METRICS_V2_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FROZEN_THRESHOLD = 0.594298


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
        "threshold": threshold,
    }


def main():
    logger.info("Loading raw dataset for Baseline V2 ablation...")
    raw_df, _ = load_train_data()
    train_df, val_df, test_df, split_meta = chronological_split(raw_df)
    del raw_df

    logger.info("Resolving entity proxies on TRAIN, VAL, TEST...")
    for part in [train_df, val_df, test_df]:
        part["entity_proxy"] = resolve_entity_key(part, key_type="card_addr_email")

    y_train = train_df["isFraud"].values
    y_val = val_df["isFraud"].values
    y_test = test_df["isFraud"].values

    # 1. Base preprocessor (frozen tabular features)
    logger.info("Running base preprocessing...")
    base_prep = BaselinePreprocessor()
    X_train_base = base_prep.fit_transform(train_df)
    X_val_base = base_prep.transform(val_df)
    X_test_base = base_prep.transform(test_df)

    # 2. Extract feature sets
    logger.info("Extracting causal feature sets...")
    # PIT (Time + Amount)
    pit_train = compute_point_in_time_features(train_df)
    pit_val = compute_point_in_time_features(val_df)
    pit_test = compute_point_in_time_features(test_df)

    # Frequency encoding
    freq_cols = ["card1", "addr1", "P_emaildomain", "DeviceInfo"]
    fe = FrequencyEncoder(freq_cols).fit(train_df)
    fe_train = fe.transform(train_df)
    fe_val = fe.transform(val_df)
    fe_test = fe.transform(test_df)

    # Causal stream engine (sequential processing across train -> val -> test)
    logger.info("Computing causal stream running features across splits...")
    stream_engine = CausalStreamFeatureEngine()
    stream_train = stream_engine.process_partition(train_df)
    stream_val = stream_engine.process_partition(val_df)
    stream_test = stream_engine.process_partition(test_df)

    # Feature groups definitions
    feat_groups = {
        "A_freq": pd.concat([fe_train, stream_train[["prior_count_card1", "prior_count_addr1", "prior_count_P_emaildomain", "prior_count_DeviceInfo"]]], axis=1),
        "B_time": pit_train[["hour_of_day", "day_of_week", "hour_sin", "hour_cos"]].join(stream_train[["entity_dt_elapsed"]]),
        "C_amount": pit_train[["log_TransactionAmt", "amt_decimal_cents", "amt_is_integer"]],
        "D_hist_agg": stream_train[["entity_prior_count", "entity_hist_mean_amt", "entity_hist_std_amt", "entity_amt_ratio"]],
    }
    feat_groups_val = {
        "A_freq": pd.concat([fe_val, stream_val[["prior_count_card1", "prior_count_addr1", "prior_count_P_emaildomain", "prior_count_DeviceInfo"]]], axis=1),
        "B_time": pit_val[["hour_of_day", "day_of_week", "hour_sin", "hour_cos"]].join(stream_val[["entity_dt_elapsed"]]),
        "C_amount": pit_val[["log_TransactionAmt", "amt_decimal_cents", "amt_is_integer"]],
        "D_hist_agg": stream_val[["entity_prior_count", "entity_hist_mean_amt", "entity_hist_std_amt", "entity_amt_ratio"]],
    }
    feat_groups_test = {
        "A_freq": pd.concat([fe_test, stream_test[["prior_count_card1", "prior_count_addr1", "prior_count_P_emaildomain", "prior_count_DeviceInfo"]]], axis=1),
        "B_time": pit_test[["hour_of_day", "day_of_week", "hour_sin", "hour_cos"]].join(stream_test[["entity_dt_elapsed"]]),
        "C_amount": pit_test[["log_TransactionAmt", "amt_decimal_cents", "amt_is_integer"]],
        "D_hist_agg": stream_test[["entity_prior_count", "entity_hist_mean_amt", "entity_hist_std_amt", "entity_amt_ratio"]],
    }

    # Define ablations
    ablations = {
        "V0_Base": ([], "Frozen Baseline Tabular"),
        "V2_A_Frequency": (["A_freq"], "Baseline + Frequency & Prior Counts"),
        "V2_B_Time": (["B_time"], "Baseline + Cyclical & Elapsed Time"),
        "V2_C_Amount": (["C_amount"], "Baseline + Amount Decimals & Log"),
        "V2_D_HistAgg": (["D_hist_agg"], "Baseline + Causal Entity Historical Aggregations"),
        "V2_E_Combined": (["A_freq", "B_time", "C_amount", "D_hist_agg"], "Baseline + All Causal Engineered Features"),
    }

    ablation_results = {}
    models = {}

    for name, (groups, desc) in ablations.items():
        logger.info(f"\n================ Running Ablation: {name} ({desc}) ================")
        if not groups:
            X_tr = X_train_base
            X_va = X_val_base
        else:
            add_tr = [feat_groups[g] for g in groups]
            add_va = [feat_groups_val[g] for g in groups]
            X_tr = pd.concat([X_train_base] + add_tr, axis=1)
            X_va = pd.concat([X_val_base] + add_va, axis=1)

        model = BaselineModel()
        model.fit(X_tr, train_df["isFraud"], X_va, val_df["isFraud"], cat_cols=base_prep.cat_cols)
        proba_val = model.predict_risk(X_va)
        val_metrics = compute_metrics(y_val, proba_val)
        logger.info(f"[{name}] VAL ROC-AUC: {val_metrics['roc_auc']:.6f} | PR-AUC: {val_metrics['pr_auc']:.6f} | F1: {val_metrics['f1']:.6f} | Prec: {val_metrics['precision']:.6f} | Rec: {val_metrics['recall']:.6f}")

        ablation_results[name] = {
            "description": desc,
            "feature_groups": groups,
            "n_features": X_tr.shape[1],
            "val_metrics": val_metrics,
        }
        models[name] = model

    # Select Best Model on VALIDATION ONLY
    # Primary ranking criteria: PR-AUC and ROC-AUC on validation
    best_name = max(ablation_results.keys(), key=lambda k: (ablation_results[k]["val_metrics"]["pr_auc"], ablation_results[k]["val_metrics"]["roc_auc"]))
    logger.info(f"\n>>> WINNING CONFIGURATION (VALIDATION ONLY): {best_name} <<<")
    logger.info(f"Val PR-AUC: {ablation_results[best_name]['val_metrics']['pr_auc']:.6f}, Val ROC-AUC: {ablation_results[best_name]['val_metrics']['roc_auc']:.6f}")

    # Evaluate Best Configuration once on TEST
    best_groups = ablations[best_name][0]
    best_model = models[best_name]
    if not best_groups:
        X_te = X_test_base
    else:
        add_te = [feat_groups_test[g] for g in best_groups]
        X_te = pd.concat([X_test_base] + add_te, axis=1)

    proba_test = best_model.predict_risk(X_te)
    test_metrics = compute_metrics(y_test, proba_test)
    ablation_results[best_name]["test_metrics"] = test_metrics

    logger.info(f"\n================ Final Held-out TEST Evaluation for {best_name} ================")
    logger.info(f"TEST ROC-AUC:   {test_metrics['roc_auc']:.6f}")
    logger.info(f"TEST PR-AUC:    {test_metrics['pr_auc']:.6f}")
    logger.info(f"TEST F1:        {test_metrics['f1']:.6f}")
    logger.info(f"TEST Precision: {test_metrics['precision']:.6f}")
    logger.info(f"TEST Recall:    {test_metrics['recall']:.6f}")
    logger.info(f"TEST FPR:       {test_metrics['fpr']:.6f}")

    # Save artifacts
    with open(ARTIFACTS_V2_DIR / "ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    with open(METRICS_V2_DIR / "v2_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Save winning model
    best_model.save(MODEL_V2_DIR / "lgbm_model.pkl")

    # Save test predictions
    test_pred_df = pd.DataFrame({
        "TransactionID": test_df["TransactionID"].values,
        "TransactionDT": test_df["TransactionDT"].values,
        "isFraud": y_test,
        "A_t_v2": proba_test,
        "pred_label_v2": (proba_test >= FROZEN_THRESHOLD).astype(int),
    })
    test_pred_df.to_csv(cfg.PROJECT_ROOT / "results" / "test_predictions_v2.csv", index=False)

    logger.info("Baseline V2 ablation and model freezing completed successfully.")


if __name__ == "__main__":
    main()
