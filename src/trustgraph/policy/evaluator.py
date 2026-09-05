"""
evaluator.py — TRUSTGRAPH Progressive Risk Decision Policy Evaluator
===================================================================

Evaluation metrics, risk-band analysis, and policy comparison tools
for the progressive risk decision policy.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from trustgraph.policy.config import PolicyAction, RiskBand
from trustgraph.policy.decision_engine import PolicyThresholds, batch_assign_actions


def evaluate_policy(
    y_true: np.ndarray,
    actions: np.ndarray,
    thresholds: PolicyThresholds,
) -> Dict[str, Any]:
    """
    Compute comprehensive action-stratified evaluation metrics.
    
    Parameters
    ----------
    y_true : binary ground truth array (0 or 1)
    actions : array of action strings (ALLOW, VERIFY, THROTTLE, BLOCK)
    thresholds : PolicyThresholds
    
    Returns
    -------
    Dictionary of action distributions, fraud concentration, and intervention metrics.
    """
    y = np.asarray(y_true, dtype=int)
    n_total = len(y)
    total_frauds = int(np.sum(y == 1))
    total_legit = n_total - total_frauds
    base_fraud_rate = total_frauds / n_total if n_total > 0 else 0.0

    action_stats: Dict[str, Any] = {}
    for act in PolicyAction:
        mask = (actions == act.value)
        n_act = int(np.sum(mask))
        n_fraud_act = int(np.sum((y == 1) & mask))
        n_legit_act = n_act - n_fraud_act
        fraud_rate_act = n_fraud_act / n_act if n_act > 0 else 0.0
        fraud_capture_pct = (n_fraud_act / total_frauds * 100.0) if total_frauds > 0 else 0.0
        legit_share_pct = (n_legit_act / total_legit * 100.0) if total_legit > 0 else 0.0

        action_stats[act.value] = {
            "action": act.value,
            "transaction_count": n_act,
            "pct_of_total_traffic": round(100.0 * n_act / n_total, 2) if n_total > 0 else 0.0,
            "fraud_count": n_fraud_act,
            "legitimate_count": n_legit_act,
            "fraud_rate": round(fraud_rate_act, 6),
            "fraud_capture_pct_of_total": round(fraud_capture_pct, 2),
            "legitimate_share_pct": round(legit_share_pct, 2),
            "enrichment_over_base_rate": round(fraud_rate_act / base_fraud_rate, 2) if base_fraud_rate > 0 else 0.0,
        }

    # Aggregate Operational Intervention Tiers:
    # Tier 1: Total Intervened (VERIFY + THROTTLE + BLOCK)
    intervened_mask = (actions != PolicyAction.ALLOW.value)
    tp_int = int(np.sum((y == 1) & intervened_mask))
    fp_int = int(np.sum((y == 0) & intervened_mask))
    prec_int = tp_int / (tp_int + fp_int) if (tp_int + fp_int) > 0 else 0.0
    rec_int = tp_int / total_frauds if total_frauds > 0 else 0.0
    f1_int = (2 * prec_int * rec_int) / (prec_int + rec_int) if (prec_int + rec_int) > 0 else 0.0
    fpr_int = fp_int / total_legit if total_legit > 0 else 0.0

    # Tier 2: Strong Interventions (THROTTLE + BLOCK)
    strong_mask = (actions == PolicyAction.THROTTLE.value) | (actions == PolicyAction.BLOCK.value)
    tp_str = int(np.sum((y == 1) & strong_mask))
    fp_str = int(np.sum((y == 0) & strong_mask))
    prec_str = tp_str / (tp_str + fp_str) if (tp_str + fp_str) > 0 else 0.0
    rec_str = tp_str / total_frauds if total_frauds > 0 else 0.0
    f1_str = (2 * prec_str * rec_str) / (prec_str + rec_str) if (prec_str + rec_str) > 0 else 0.0
    fpr_str = fp_str / total_legit if total_legit > 0 else 0.0

    # Tier 3: Critical Direct Block (BLOCK only)
    block_mask = (actions == PolicyAction.BLOCK.value)
    tp_blk = int(np.sum((y == 1) & block_mask))
    fp_blk = int(np.sum((y == 0) & block_mask))
    prec_blk = tp_blk / (tp_blk + fp_blk) if (tp_blk + fp_blk) > 0 else 0.0
    rec_blk = tp_blk / total_frauds if total_frauds > 0 else 0.0
    f1_blk = (2 * prec_blk * rec_blk) / (prec_blk + rec_blk) if (prec_blk + rec_blk) > 0 else 0.0
    fpr_blk = fp_blk / total_legit if total_legit > 0 else 0.0

    return {
        "thresholds": thresholds.to_dict(),
        "total_transactions": n_total,
        "total_frauds": total_frauds,
        "total_legitimate": total_legit,
        "base_fraud_rate": round(base_fraud_rate, 6),
        "actions": action_stats,
        "operational_tiers": {
            "tier_1_any_intervention_verify_plus": {
                "description": "Any intervention (VERIFY, THROTTLE, or BLOCK)",
                "total_transactions": int(np.sum(intervened_mask)),
                "true_positives": tp_int,
                "false_positives": fp_int,
                "precision": round(prec_int, 6),
                "recall": round(rec_int, 6),
                "f1": round(f1_int, 6),
                "fpr": round(fpr_int, 6),
                "intervention_rate_on_legitimate": round(100.0 * fp_int / total_legit, 2),
            },
            "tier_2_strong_intervention_throttle_plus": {
                "description": "High-severity intervention (THROTTLE or BLOCK)",
                "total_transactions": int(np.sum(strong_mask)),
                "true_positives": tp_str,
                "false_positives": fp_str,
                "precision": round(prec_str, 6),
                "recall": round(rec_str, 6),
                "f1": round(f1_str, 6),
                "fpr": round(fpr_str, 6),
                "intervention_rate_on_legitimate": round(100.0 * fp_str / total_legit, 2),
            },
            "tier_3_critical_direct_block": {
                "description": "Direct transaction rejection (BLOCK only)",
                "total_transactions": int(np.sum(block_mask)),
                "true_positives": tp_blk,
                "false_positives": fp_blk,
                "precision": round(prec_blk, 6),
                "recall": round(rec_blk, 6),
                "f1": round(f1_blk, 6),
                "fpr": round(fpr_blk, 6),
                "rejection_rate_on_legitimate": round(100.0 * fp_blk / total_legit, 4),
            },
        },
    }


def compare_baseline_and_progressive_policy(
    y_true: np.ndarray,
    A_t: np.ndarray,
    R_t: np.ndarray,
    tau_base: float,
    thresholds: PolicyThresholds,
) -> Dict[str, Any]:
    """
    Compare Policy A (Binary Baseline Threshold) vs Policy B (Progressive TRUSTGRAPH Policy).
    """
    y = np.asarray(y_true, dtype=int)
    n_total = len(y)
    total_frauds = int(np.sum(y == 1))
    total_legit = n_total - total_frauds

    # Policy A: Binary Baseline (A_t >= tau_base => BLOCK, else ALLOW)
    b0_pos = (A_t >= tau_base)
    tp_b0 = int(np.sum((y == 1) & b0_pos))
    fp_b0 = int(np.sum((y == 0) & b0_pos))
    prec_b0 = tp_b0 / (tp_b0 + fp_b0) if (tp_b0 + fp_b0) > 0 else 0.0
    rec_b0 = tp_b0 / total_frauds if total_frauds > 0 else 0.0
    f1_b0 = 2 * prec_b0 * rec_b0 / (prec_b0 + rec_b0) if (prec_b0 + rec_b0) > 0 else 0.0
    fpr_b0 = fp_b0 / total_legit if total_legit > 0 else 0.0

    # Policy B: Progressive Policy
    actions, bands = batch_assign_actions(R_t, thresholds)
    policy_b_eval = evaluate_policy(y, actions, thresholds)

    return {
        "policy_A_binary_baseline": {
            "name": "Policy A: Binary Point-wise LightGBM Baseline",
            "threshold": tau_base,
            "actions": {
                "ALLOW": {"txns": int(np.sum(~b0_pos)), "frauds": int(np.sum((y == 1) & (~b0_pos))), "fp": 0},
                "BLOCK": {"txns": int(np.sum(b0_pos)), "frauds": tp_b0, "fp": fp_b0},
            },
            "metrics": {
                "precision": round(prec_b0, 6),
                "recall": round(rec_b0, 6),
                "f1": round(f1_b0, 6),
                "fpr": round(fpr_b0, 6),
                "tp": tp_b0,
                "fp": fp_b0,
            },
        },
        "policy_B_progressive_trustgraph": {
            "name": "Policy B: Progressive TRUSTGRAPH Risk Policy",
            "thresholds": thresholds.to_dict(),
            "evaluation": policy_b_eval,
        },
    }
