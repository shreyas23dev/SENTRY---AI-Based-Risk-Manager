"""
tune_policy.py — TRUSTGRAPH Phase 4 Validation-Only Policy Threshold Tuning
===========================================================================

Evaluates candidate progressive policy threshold combinations on VALIDATION ONLY.
Selects and freezes the optimal threshold triplet: (tau_verify, tau_throttle, tau_block).
Saves:
  - artifacts/policy/thresholds.json
  - artifacts/policy/validation_results.json
  - artifacts/policy/policy_rules.json
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

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
    BASELINE_THRESHOLD, ENTITY_KEY_TYPE,
)
from trustgraph.fusion.fusion_engine import apply_fusion_rule
from trustgraph.policy.config import (
    POLICY_DIR, CANDIDATE_TAU_VERIFY, CANDIDATE_TAU_THROTTLE, CANDIDATE_TAU_BLOCK,
    PolicyAction,
)
from trustgraph.policy.decision_engine import PolicyThresholds, batch_assign_actions, verify_policy_invariants
from trustgraph.policy.evaluator import evaluate_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tune_policy")

POLICY_DIR.mkdir(parents=True, exist_ok=True)


def prepare_validation_scores() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct exact frozen A_t, P_t, G_t, R_t scores on VALIDATION ONLY."""
    logger.info("Loading dataset for validation scoring...")
    raw_df, _ = load_train_data()
    train_df, val_df, _, split_meta = chronological_split(raw_df)
    del raw_df

    # 1. Resolve entity proxy
    for part in [train_df, val_df]:
        part["entity_proxy"] = resolve_entity_key(part, key_type=ENTITY_KEY_TYPE)

    # 2. Point-wise LightGBM Model Inference
    logger.info("Computing A_t via frozen LightGBM on TRAIN and VAL...")
    model = BaselineModel.load(Path(__file__).resolve().parents[1] / "artifacts" / "baseline" / "model" / "lgbm_model.pkl")
    preprocessor = BaselinePreprocessor.load(Path(__file__).resolve().parents[1] / "artifacts" / "baseline" / "preprocessing")

    train_df["A_t"] = model.predict_risk(preprocessor.transform(train_df))
    val_df["A_t"]   = model.predict_risk(preprocessor.transform(val_df))

    # 3. Entity Temporal Risk Engine
    logger.info("Computing P_t across TRAIN -> VAL...")
    temp_engine = EntityTemporalRiskEngine(beta=0.3, gamma=0.5, lambda_=0.05, delta=0.05)
    for part in [train_df, val_df]:
        ents = part["entity_proxy"].values
        scores = part["A_t"].values
        P_arr = np.zeros(len(part), dtype=float)
        for i in range(len(part)):
            _, p_val = temp_engine.step(str(ents[i]), float(scores[i]))
            P_arr[i] = p_val
        part["P_t"] = P_arr

    # 4. Relational Graph Engine
    logger.info("Computing G_t across TRAIN -> VAL...")
    rel_params = GraphParameters(
        k_attr_max=25, window_sec=86400.0, d_ref=3.0, v_ref=10.0,
        w_D=0.6, w_V=0.4, relational_attrs=("DeviceInfo",)
    )
    graph_engine = LightweightRelationalGraph(rel_params)
    graph_engine.fit_attribute_frequency_ceiling(train_df)
    process_partition(train_df, graph_engine)
    val_records = process_partition(val_df, graph_engine)
    val_df["G_t"] = np.array([r.G_t for r in val_records], dtype=float)

    # 5. Frozen Conditional Fusion Rule F1
    logger.info("Computing fused score R_t on VAL...")
    R_t = apply_fusion_rule("F1", val_df["A_t"].values, val_df["P_t"].values, val_df["G_t"].values, {"alpha": 1.0, "beta": 0.05})
    val_df["R_t"] = R_t

    y_val = val_df["isFraud"].values.astype(int)
    A_val = val_df["A_t"].values.astype(float)
    P_val = val_df["P_t"].values.astype(float)
    G_val = val_df["G_t"].values.astype(float)
    R_val = val_df["R_t"].values.astype(float)

    logger.info("Validation data prepared: N=%d, frauds=%d (%.4f%%)",
                len(val_df), int(y_val.sum()), 100 * y_val.mean())
    return y_val, A_val, P_val, G_val, R_val


def run_validation_grid_search(y_val: np.ndarray, R_val: np.ndarray) -> Dict[str, Any]:
    """Evaluate candidate threshold triplets on validation data."""
    logger.info("Running validation grid search over candidate thresholds...")

    results = []
    best_candidate = None
    best_score = -1.0

    for tau_v in CANDIDATE_TAU_VERIFY:
        for tau_t in CANDIDATE_TAU_THROTTLE:
            for tau_b in CANDIDATE_TAU_BLOCK:
                if not (tau_v < tau_t < tau_b):
                    continue

                thresholds = PolicyThresholds(tau_verify=tau_v, tau_throttle=tau_t, tau_block=tau_b)
                actions, bands = batch_assign_actions(R_val, thresholds)
                passed, diag = verify_policy_invariants(R_val, actions, thresholds)
                assert passed, f"Invariant check failed: {diag}"

                eval_res = evaluate_policy(y_val, actions, thresholds)

                # Objective criteria on validation:
                # 1. High fraud capture across all interventions (VERIFY + THROTTLE + BLOCK)
                # 2. Strong precision in BLOCK tier
                # 3. High fraud rate in THROTTLE and BLOCK tiers
                # 4. Low false alarm impact in ALLOW tier
                tier1 = eval_res["operational_tiers"]["tier_1_any_intervention_verify_plus"]
                tier2 = eval_res["operational_tiers"]["tier_2_strong_intervention_throttle_plus"]
                tier3 = eval_res["operational_tiers"]["tier_3_critical_direct_block"]

                block_prec = tier3["precision"]
                throttle_prec = tier2["precision"]
                intervened_f1 = tier1["f1"]
                intervened_recall = tier1["recall"]

                # Composite validation score:
                # Prioritizes balanced multi-tier fraud concentration
                # F1 of intervention tier + precision of block tier + fraction of severe frauds in block
                composite_score = (
                    0.40 * intervened_f1 +
                    0.30 * block_prec +
                    0.30 * (tier3["true_positives"] / max(1, tier1["true_positives"]))
                )

                entry = {
                    "tau_verify": tau_v,
                    "tau_throttle": tau_t,
                    "tau_block": tau_b,
                    "composite_score": round(composite_score, 6),
                    "intervened_recall": tier1["recall"],
                    "intervened_precision": tier1["precision"],
                    "intervened_f1": tier1["f1"],
                    "intervened_fpr": tier1["fpr"],
                    "block_precision": tier3["precision"],
                    "block_recall": tier3["recall"],
                    "block_frauds": tier3["true_positives"],
                    "block_false_positives": tier3["false_positives"],
                    "actions_distribution": {
                        act: {
                            "txns": eval_res["actions"][act]["transaction_count"],
                            "frauds": eval_res["actions"][act]["fraud_count"],
                            "fraud_rate": eval_res["actions"][act]["fraud_rate"],
                            "legit": eval_res["actions"][act]["legitimate_count"],
                        }
                        for act in [a.value for a in PolicyAction]
                    },
                }
                results.append(entry)

                if composite_score > best_score:
                    best_score = composite_score
                    best_candidate = entry

    logger.info("Validation search complete: evaluated %d valid threshold triplets.", len(results))
    logger.info("Selected optimal policy: tau_verify=%.2f, tau_throttle=%.2f, tau_block=%.2f (score=%.4f)",
                best_candidate["tau_verify"], best_candidate["tau_throttle"],
                best_candidate["tau_block"], best_candidate["composite_score"])
    return {
        "all_evaluations": sorted(results, key=lambda x: x["composite_score"], reverse=True),
        "selected_policy": best_candidate,
    }


def main():
    y_val, A_val, P_val, G_val, R_val = prepare_validation_scores()
    search_res = run_validation_grid_search(y_val, R_val)
    selected = search_res["selected_policy"]

    # 1. Save artifacts/policy/thresholds.json
    frozen_thresholds = {
        "tau_verify": selected["tau_verify"],
        "tau_throttle": selected["tau_throttle"],
        "tau_block": selected["tau_block"],
        "selected_on": "VALIDATION ONLY (88,581 chronological transactions)",
        "selection_objective": "Balanced progressive fraud concentration & precision tiering",
        "composite_validation_score": selected["composite_score"],
    }
    with open(POLICY_DIR / "thresholds.json", "w") as f:
        json.dump(frozen_thresholds, f, indent=2)
    logger.info("Saved -> %s", POLICY_DIR / "thresholds.json")

    # 2. Save artifacts/policy/validation_results.json
    with open(POLICY_DIR / "validation_results.json", "w") as f:
        json.dump(search_res, f, indent=2)
    logger.info("Saved -> %s", POLICY_DIR / "validation_results.json")

    # 3. Save artifacts/policy/policy_rules.json
    policy_rules = {
        "policy_name": "TRUSTGRAPH Progressive Risk Decision Policy",
        "version": "1.0",
        "input_signals": ["A_t (Point-wise)", "P_t (Temporal)", "G_t (Relational)", "R_t (Fused Risk)"],
        "fused_risk_formula": "R_t = clip(A_t + 1.0 * P_t + 0.05 * G_t, 0.0, 1.0)",
        "decision_tiers": [
            {
                "action": "ALLOW",
                "risk_band": "LOW",
                "condition": f"R_t < {selected['tau_verify']:.2f}",
                "operational_directive": "Transaction proceeds normally without friction.",
                "intervention_severity": 0,
            },
            {
                "action": "VERIFY",
                "risk_band": "MODERATE",
                "condition": f"{selected['tau_verify']:.2f} <= R_t < {selected['tau_throttle']:.2f}",
                "operational_directive": "Step-up authentication required (e.g. 3D-Secure / OTP / biometric challenge).",
                "intervention_severity": 1,
            },
            {
                "action": "THROTTLE",
                "risk_band": "HIGH",
                "condition": f"{selected['tau_throttle']:.2f} <= R_t < {selected['tau_block']:.2f}",
                "operational_directive": "Operational restriction applied (delayed settlement, reduced velocity ceiling, manual queue).",
                "intervention_severity": 2,
            },
            {
                "action": "BLOCK",
                "risk_band": "VERY_HIGH",
                "condition": f"R_t >= {selected['tau_block']:.2f}",
                "operational_directive": "Hard transaction decline / immediate rejection.",
                "intervention_severity": 3,
            },
        ],
    }
    with open(POLICY_DIR / "policy_rules.json", "w") as f:
        json.dump(policy_rules, f, indent=2)
    logger.info("Saved -> %s", POLICY_DIR / "policy_rules.json")


if __name__ == "__main__":
    main()
