"""
evaluate_trustgraph_v2.py — Run Frozen TRUSTGRAPH Pipeline on Baseline V2
========================================================================

Applies the completely frozen downstream TRUSTGRAPH components:
  - Entity-scoped temporal risk engine (P_t)
  - Lightweight relational graph (G_t)
  - Conditional fusion rule (R_t = clip(A_t + P_t + 0.05 G_t, 0, 1))
  - Progressive risk decision policy (ALLOW / VERIFY / THROTTLE / BLOCK)

Compares:
  - System 1: Baseline Point-wise (B0 vs V2)
  - System 2: Baseline + Entity Temporal (B0+P_t vs V2+P_t)
  - System 3: Fused TRUSTGRAPH (B0+P_t+G_t vs V2+P_t+G_t)
  - System 4: Progressive Policy (Tiers 1, 2, 3)

Also extracts genuine slow-burn fraud trajectories from the TEST stream.
"""

import sys
import json
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
from trustgraph.temporal.entity_tracker import resolve_entity_key, EntityTemporalRiskEngine
from trustgraph.relational.graph_engine import (
    GraphParameters, LightweightRelationalGraph, process_partition,
)
from trustgraph.fusion.fusion_engine import apply_fusion_rule
from trustgraph.policy.config import PolicyAction
from trustgraph.policy.decision_engine import PolicyThresholds, batch_assign_actions
from trustgraph.features_v2.causal_features import (
    compute_point_in_time_features,
    FrequencyEncoder,
    CausalStreamFeatureEngine,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_trustgraph_v2")

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
        "threshold": threshold,
    }


def main():
    logger.info("Loading raw dataset for downstream TRUSTGRAPH V2 evaluation...")
    raw_df, _ = load_train_data()
    train_df, val_df, test_df, split_meta = chronological_split(raw_df)
    del raw_df

    for part in [train_df, val_df, test_df]:
        part["entity_proxy"] = resolve_entity_key(part, key_type="card_addr_email")

    y_test = test_df["isFraud"].values

    # Load frozen V2 model & extract full feature matrix
    logger.info("Loading trained Baseline V2 model...")
    v2_model = BaselineModel.load(cfg.PROJECT_ROOT / "artifacts" / "baseline_v2" / "model" / "lgbm_model.pkl")

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

    # Full V2 feature sets
    X_tr_v2 = pd.concat([X_tr_base, fe_tr, stream_tr[["prior_count_card1", "prior_count_addr1", "prior_count_P_emaildomain", "prior_count_DeviceInfo"]],
                         pit_tr[["hour_of_day", "day_of_week", "hour_sin", "hour_cos"]], stream_tr[["entity_dt_elapsed"]],
                         pit_tr[["log_TransactionAmt", "amt_decimal_cents", "amt_is_integer"]],
                         stream_tr[["entity_prior_count", "entity_hist_mean_amt", "entity_hist_std_amt", "entity_amt_ratio"]]], axis=1)

    X_va_v2 = pd.concat([X_va_base, fe_va, stream_va[["prior_count_card1", "prior_count_addr1", "prior_count_P_emaildomain", "prior_count_DeviceInfo"]],
                         pit_va[["hour_of_day", "day_of_week", "hour_sin", "hour_cos"]], stream_va[["entity_dt_elapsed"]],
                         pit_va[["log_TransactionAmt", "amt_decimal_cents", "amt_is_integer"]],
                         stream_va[["entity_prior_count", "entity_hist_mean_amt", "entity_hist_std_amt", "entity_amt_ratio"]]], axis=1)

    X_te_v2 = pd.concat([X_te_base, fe_te, stream_te[["prior_count_card1", "prior_count_addr1", "prior_count_P_emaildomain", "prior_count_DeviceInfo"]],
                         pit_te[["hour_of_day", "day_of_week", "hour_sin", "hour_cos"]], stream_te[["entity_dt_elapsed"]],
                         pit_te[["log_TransactionAmt", "amt_decimal_cents", "amt_is_integer"]],
                         stream_te[["entity_prior_count", "entity_hist_mean_amt", "entity_hist_std_amt", "entity_amt_ratio"]]], axis=1)

    train_df["A_t_v2"] = v2_model.predict_risk(X_tr_v2)
    val_df["A_t_v2"] = v2_model.predict_risk(X_va_v2)
    test_df["A_t_v2"] = v2_model.predict_risk(X_te_v2)

    # 1. Run Frozen Temporal Engine across splits
    logger.info("Running frozen temporal engine on V2 risk scores...")
    temp_engine = EntityTemporalRiskEngine(beta=0.3, gamma=0.5, lambda_=0.05, delta=0.05)
    
    # Process train & val to seed temporal state
    for part in [train_df, val_df]:
        ents = part["entity_proxy"].values
        scores = part["A_t_v2"].values
        for i in range(len(part)):
            temp_engine.step(str(ents[i]), float(scores[i]))

    # Test stream processing
    test_ents = test_df["entity_proxy"].values
    test_scores = test_df["A_t_v2"].values
    P_test = np.empty(len(test_df), dtype=float)
    E_test = np.empty(len(test_df), dtype=float)
    for i in range(len(test_df)):
        e_val, p_val = temp_engine.step(str(test_ents[i]), float(test_scores[i]))
        E_test[i] = e_val
        P_test[i] = p_val
    test_df["P_t_v2"] = P_test
    test_df["E_t_v2"] = E_test

    # 2. Run Frozen Relational Graph Engine across splits
    logger.info("Running frozen relational graph on V2...")
    rel_params = GraphParameters(k_attr_max=25, window_sec=86400.0, d_ref=3.0, v_ref=10.0,
                                 w_D=0.6, w_V=0.4, relational_attrs=("DeviceInfo",))
    graph_engine = LightweightRelationalGraph(rel_params)
    graph_engine.fit_attribute_frequency_ceiling(train_df)
    process_partition(train_df, graph_engine)
    process_partition(val_df, graph_engine)
    test_records = process_partition(test_df, graph_engine)

    test_df["G_t"] = np.array([r.G_t for r in test_records], dtype=float)
    test_df["D_t"] = np.array([r.D_t for r in test_records], dtype=float)
    test_df["V_t"] = np.array([r.V_t for r in test_records], dtype=float)

    # 3. Apply Frozen Conditional Fusion Rule
    logger.info("Applying frozen conditional risk fusion rule...")
    R_t_v2 = apply_fusion_rule("F1", test_df["A_t_v2"].values, test_df["P_t_v2"].values, test_df["G_t"].values, {"alpha": 1.0, "beta": 0.05})
    test_df["R_t_v2"] = R_t_v2

    # 4. Apply Frozen Progressive Policy
    logger.info("Applying frozen progressive policy...")
    actions_v2, bands_v2 = batch_assign_actions(R_t_v2, POLICY_THRESHOLDS)
    test_df["action_v2"] = actions_v2
    test_df["risk_band_v2"] = bands_v2

    # Metrics computation
    # System 1: Baseline V2 Point-wise
    m_base_v2 = compute_metrics(y_test, test_df["A_t_v2"].values)
    # System 2: V2 + Temporal (A_t + P_t)
    m_temp_v2 = compute_metrics(y_test, np.clip(test_df["A_t_v2"].values + test_df["P_t_v2"].values, 0.0, 1.0))
    # System 3: Fused TRUSTGRAPH V2 (R_t)
    m_fused_v2 = compute_metrics(y_test, R_t_v2)

    # Policy Tiers
    # Tier 1: Any intervention (VERIFY + THROTTLE + BLOCK) -> R_t >= 0.60
    m_tier1 = compute_metrics(y_test, R_t_v2, threshold=0.60)
    # Tier 2: Strong intervention (THROTTLE + BLOCK) -> R_t >= 0.65
    m_tier2 = compute_metrics(y_test, R_t_v2, threshold=0.65)
    # Tier 3: Direct Block -> R_t >= 0.80
    m_tier3 = compute_metrics(y_test, R_t_v2, threshold=0.80)

    # Load frozen B0 results for direct 4-system side-by-side comparison
    with open(cfg.PROJECT_ROOT / "artifacts" / "system_audit" / "master_metrics.json") as f:
        b0_master = json.load(f)

    b0_systems = b0_master["system_performances"]

    comparison = {
        "System_1_Baseline": {
            "B0_Original": b0_systems["B0_baseline"],
            "V2_Enhanced": m_base_v2,
            "delta_precision": round(m_base_v2["precision"] - b0_systems["B0_baseline"]["precision"], 6),
            "delta_recall": round(m_base_v2["recall"] - b0_systems["B0_baseline"]["recall"], 6),
            "delta_f1": round(m_base_v2["f1"] - b0_systems["B0_baseline"]["f1"], 6),
            "delta_fpr": round(m_base_v2["fpr"] - b0_systems["B0_baseline"]["fpr"], 6),
            "delta_pr_auc": round(m_base_v2["pr_auc"] - 0.534008, 6),
            "delta_roc_auc": round(m_base_v2["roc_auc"] - 0.901943, 6),
        },
        "System_2_Temporal": {
            "B0_Original": b0_systems["B1_entity_temporal"],
            "V2_Enhanced": m_temp_v2,
            "delta_precision": round(m_temp_v2["precision"] - b0_systems["B1_entity_temporal"]["precision"], 6),
            "delta_recall": round(m_temp_v2["recall"] - b0_systems["B1_entity_temporal"]["recall"], 6),
            "delta_f1": round(m_temp_v2["f1"] - b0_systems["B1_entity_temporal"]["f1"], 6),
            "delta_fpr": round(m_temp_v2["fpr"] - b0_systems["B1_entity_temporal"]["fpr"], 6),
        },
        "System_3_Fused_TRUSTGRAPH": {
            "B0_Original": b0_systems["B3_fused"],
            "V2_Enhanced": m_fused_v2,
            "delta_precision": round(m_fused_v2["precision"] - b0_systems["B3_fused"]["precision"], 6),
            "delta_recall": round(m_fused_v2["recall"] - b0_systems["B3_fused"]["recall"], 6),
            "delta_f1": round(m_fused_v2["f1"] - b0_systems["B3_fused"]["f1"], 6),
            "delta_fpr": round(m_fused_v2["fpr"] - b0_systems["B3_fused"]["fpr"], 6),
        },
        "System_4_Progressive_Policy": {
            "Tier_1_Any_Intervention": m_tier1,
            "Tier_2_Strong_Intervention": m_tier2,
            "Tier_3_Direct_Block": m_tier3,
        }
    }

    out_comp_path = cfg.PROJECT_ROOT / "artifacts" / "baseline_v2" / "trustgraph_v2_comparison.json"
    with open(out_comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Saved 4-system comparison -> {out_comp_path}")

    # Search for real slow-burn fraud trajectories in TEST
    logger.info("Searching for real slow-burn trajectories on TEST stream...")
    # Find entities with multiple test transactions where the final fraud is sub-threshold on A_t but caught on R_t
    candidate_indices = np.where((test_df["isFraud"].values == 1) &
                                 (test_df["A_t_v2"].values < FROZEN_THRESHOLD) &
                                 (test_df["R_t_v2"].values >= FROZEN_THRESHOLD))[0]

    slow_burn_cases = []
    for idx in candidate_indices:
        ent = test_df["entity_proxy"].iloc[idx]
        if ent.startswith("unresolved_"):
            continue
        ent_txns = test_df[test_df["entity_proxy"] == ent]
        if len(ent_txns) >= 2:
            case_records = []
            for _, r in ent_txns.iterrows():
                case_records.append({
                    "TransactionID": int(r["TransactionID"]),
                    "TransactionDT": float(r["TransactionDT"]),
                    "TransactionAmt": float(r["TransactionAmt"]),
                    "isFraud": int(r["isFraud"]),
                    "A_t": round(float(r["A_t_v2"]), 6),
                    "E_t": round(float(r["E_t_v2"]), 6),
                    "P_t": round(float(r["P_t_v2"]), 6),
                    "G_t": round(float(r["G_t"]), 6),
                    "R_t": round(float(r["R_t_v2"]), 6),
                    "baseline_decision": "ALLOW" if r["A_t_v2"] < FROZEN_THRESHOLD else "BLOCK",
                    "trustgraph_decision": r["action_v2"],
                })
            slow_burn_cases.append({
                "entity_proxy": ent,
                "n_transactions": len(ent_txns),
                "transactions": case_records,
            })

    out_sb_path = cfg.PROJECT_ROOT / "artifacts" / "baseline_v2" / "real_slow_burn_cases.json"
    with open(out_sb_path, "w") as f:
        json.dump(slow_burn_cases, f, indent=2)
    logger.info(f"Found {len(slow_burn_cases)} real slow-burn cases. Saved -> {out_sb_path}")

    # Save full predictions csv
    test_df.to_csv(cfg.PROJECT_ROOT / "results" / "policy_predictions_v2.csv", index=False)
    logger.info("Saved policy_predictions_v2.csv")


if __name__ == "__main__":
    main()
