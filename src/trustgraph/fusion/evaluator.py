"""
evaluator.py — TRUSTGRAPH Phase 3.1 Fusion Evaluator & Coverage Diagnostics
===========================================================================

Provides system metric computation, PR-AUC / ROC-AUC, 4-way comparison,
and fine-grained coverage-aware evaluation (G_t = 0 vs G_t > 0, P_t = 0 vs P_t > 0).
"""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_system_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    base_tp: Optional[int] = None,
    base_fp: Optional[int] = None,
    base_recall: Optional[float] = None,
    base_fpr: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute complete classification and ranking metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall    = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr       = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr       = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1        = float((2 * precision * recall / (precision + recall))
                      if (precision + recall) > 0 else 0.0)

    pr_auc = None
    roc_auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass

    metrics = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 6),
        "recall":    round(recall, 6),
        "f1":        round(f1, 6),
        "fpr":       round(fpr, 6),
        "fnr":       round(fnr, 6),
        "pr_auc":    round(pr_auc, 6) if pr_auc is not None else None,
        "roc_auc":   round(roc_auc, 6) if roc_auc is not None else None,
    }

    if base_tp is not None:
        metrics["additional_frauds_recovered"] = tp - base_tp
    if base_fp is not None:
        metrics["additional_false_positives"] = fp - base_fp
    if base_recall is not None:
        metrics["recall_gain_over_b0"] = round(recall - base_recall, 6)
    if base_fpr is not None:
        metrics["fpr_change_over_b0"] = round(fpr - base_fpr, 6)

    return metrics


def compute_coverage_aware_metrics(
    y_true: np.ndarray,
    A_t: np.ndarray,
    P_t: np.ndarray,
    G_t: np.ndarray,
    R_t: np.ndarray,
    baseline_preds: np.ndarray,
    combined_preds: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute fine-grained metrics across context availability slices:
      - overall
      - G_t == 0 (missing relational evidence)
      - G_t > 0  (active relational evidence)
      - P_t == 0 (missing temporal evidence)
      - P_t > 0  (active temporal evidence)
      - P_t == 0 and G_t == 0 (uncontextualized)
      - P_t > 0 or G_t > 0    (any contextual evidence)
    """
    slices = {
        "overall": np.ones(len(y_true), dtype=bool),
        "relational_zero_Gt": (G_t == 0.0),
        "relational_active_Gt": (G_t > 0.0),
        "temporal_zero_Pt": (P_t == 0.0),
        "temporal_active_Pt": (P_t > 0.0),
        "uncontextualized_zero_both": (P_t == 0.0) & (G_t == 0.0),
        "contextualized_any_active": (P_t > 0.0) | (G_t > 0.0),
    }

    results = {}
    for name, mask in slices.items():
        sub_y = y_true[mask]
        sub_b0 = baseline_preds[mask]
        sub_b3 = combined_preds[mask]
        sub_R  = R_t[mask]
        sub_A  = A_t[mask]

        n_sub = int(np.sum(mask))
        n_fraud = int(np.sum(sub_y == 1))
        fraud_rate = float(n_fraud / n_sub) if n_sub > 0 else 0.0

        if n_sub == 0:
            continue

        b0_m = compute_system_metrics(sub_y, sub_b0, sub_A)
        b3_m = compute_system_metrics(
            sub_y, sub_b3, sub_R,
            base_tp=b0_m["tp"],
            base_fp=b0_m["fp"],
            base_recall=b0_m["recall"],
            base_fpr=b0_m["fpr"],
        )

        results[name] = {
            "transaction_count": n_sub,
            "pct_of_total": round(100.0 * n_sub / len(y_true), 2),
            "fraud_count": n_fraud,
            "fraud_prevalence": round(fraud_rate, 6),
            "baseline_B0": b0_m,
            "combined_B3": b3_m,
            "frauds_recovered": b3_m["additional_frauds_recovered"],
            "extra_false_positives": b3_m["additional_false_positives"],
            "delta_recall": b3_m.get("recall_gain_over_b0", 0.0),
            "delta_fpr": b3_m.get("fpr_change_over_b0", 0.0),
            "mean_A_t": float(np.mean(sub_A)),
            "mean_R_t": float(np.mean(sub_R)),
            "mean_boost_Rt_minus_At": float(np.mean(sub_R - sub_A)),
        }

    return results
