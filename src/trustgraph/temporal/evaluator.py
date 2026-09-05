"""
evaluator.py — Evaluation and Comparison for Phase 2 Temporal Risk Memory
==========================================================================

Responsibilities:
  - Temporal decision rule and risk fusion
  - Comparison of Baseline (B0) vs Baseline + Temporal (B1)
  - Baseline False Negative (FN) recovery metrics
  - Detection delay calculation on fraud sequences
  - Sequence-level visualization and PR curves
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "font.size":        11,
}


def make_temporal_prediction(
    A_t: np.ndarray,
    P_t: np.ndarray,
    baseline_threshold: float,
    temporal_threshold: float,
    E_t: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Produce binary predictions under B1 = Baseline + Temporal Memory.

    A transaction is flagged as fraud if:
      - Point-wise baseline risk exceeds baseline threshold: A_t >= baseline_threshold
      OR
      - Persistent-risk accumulator exceeds temporal threshold: P_t >= temporal_threshold
    """
    baseline_flags = (A_t >= baseline_threshold)
    temporal_flags = (P_t >= temporal_threshold)
    combined = baseline_flags | temporal_flags
    return combined.astype(int)


def evaluate_temporal_comparison(
    y_true: np.ndarray,
    A_t: np.ndarray,
    E_t: np.ndarray,
    P_t: np.ndarray,
    baseline_threshold: float,
    temporal_threshold: float,
    partition_name: str = "test",
) -> Dict:
    """
    Comprehensive comparative evaluation of B0 (Baseline) vs B1 (Baseline + Temporal).
    """
    N = len(y_true)
    n_fraud = int(y_true.sum())
    n_legit = int(N - n_fraud)

    # 1. Baseline Predictions & Metrics (B0)
    baseline_pred = (A_t >= baseline_threshold).astype(int)
    cm_base = confusion_matrix(y_true, baseline_pred)
    tn_b, fp_b, fn_b, tp_b = cm_base.ravel()

    base_roc_auc = float(roc_auc_score(y_true, A_t))
    base_pr_auc  = float(average_precision_score(y_true, A_t))
    base_prec    = float(precision_score(y_true, baseline_pred, zero_division=0))
    base_rec     = float(recall_score(y_true, baseline_pred, zero_division=0))
    base_f1      = float(f1_score(y_true, baseline_pred, zero_division=0))
    base_fpr     = float(fp_b / (fp_b + tn_b)) if (fp_b + tn_b) > 0 else 0.0
    base_fnr     = float(fn_b / (fn_b + tp_b)) if (fn_b + tp_b) > 0 else 0.0

    # 2. Temporal Predictions & Metrics (B1)
    temporal_pred = make_temporal_prediction(A_t, P_t, baseline_threshold, temporal_threshold, E_t=E_t)
    cm_temp = confusion_matrix(y_true, temporal_pred)
    tn_t, fp_t, fn_t, tp_t = cm_temp.ravel()

    # Combined continuous score for ROC/PR: max(A_t, P_t)
    combined_score = np.maximum(A_t, P_t)
    temp_roc_auc = float(roc_auc_score(y_true, combined_score))
    temp_pr_auc  = float(average_precision_score(y_true, combined_score))
    temp_prec    = float(precision_score(y_true, temporal_pred, zero_division=0))
    temp_rec     = float(recall_score(y_true, temporal_pred, zero_division=0))
    temp_f1      = float(f1_score(y_true, temporal_pred, zero_division=0))
    temp_fpr     = float(fp_t / (fp_t + tn_t)) if (fp_t + tn_t) > 0 else 0.0
    temp_fnr     = float(fn_t / (fn_t + tp_t)) if (fn_t + tp_t) > 0 else 0.0

    # 3. Recovery of Baseline False Negatives
    baseline_fn_mask = (y_true == 1) & (baseline_pred == 0)
    recovered_mask   = baseline_fn_mask & (temporal_pred == 1)
    n_baseline_fn    = int(baseline_fn_mask.sum())
    n_recovered      = int(recovered_mask.sum())
    pct_recovered    = float(100.0 * n_recovered / n_baseline_fn) if n_baseline_fn > 0 else 0.0

    # 4. False Positive Trade-off
    additional_fp = int(fp_t - fp_b)

    metrics = {
        "partition":             partition_name,
        "total_transactions":     N,
        "fraud_count":           n_fraud,
        "legit_count":           n_legit,
        "fraud_prevalence":      round(float(n_fraud / N), 6),
        "baseline_threshold":    float(baseline_threshold),
        "temporal_threshold":    float(temporal_threshold),
        "B0_baseline": {
            "roc_auc":   round(base_roc_auc, 6),
            "pr_auc":    round(base_pr_auc, 6),
            "precision": round(base_prec, 6),
            "recall":    round(base_rec, 6),
            "f1":        round(base_f1, 6),
            "fpr":       round(base_fpr, 6),
            "fnr":       round(base_fnr, 6),
            "tp": int(tp_b), "fp": int(fp_b), "tn": int(tn_b), "fn": int(fn_b),
        },
        "B1_temporal": {
            "roc_auc":   round(temp_roc_auc, 6),
            "pr_auc":    round(temp_pr_auc, 6),
            "precision": round(temp_prec, 6),
            "recall":    round(temp_rec, 6),
            "f1":        round(temp_f1, 6),
            "fpr":       round(temp_fpr, 6),
            "fnr":       round(temp_fnr, 6),
            "tp": int(tp_t), "fp": int(fp_t), "tn": int(tn_t), "fn": int(fn_t),
        },
        "comparative_delta": {
            "additional_frauds_recovered":     n_recovered,
            "baseline_false_negatives":        n_baseline_fn,
            "pct_baseline_fn_recovered":       round(pct_recovered, 2),
            "additional_false_positives":       additional_fp,
            "recall_gain":                     round(temp_rec - base_rec, 6),
            "f1_gain":                         round(temp_f1 - base_f1, 6),
            "fpr_change":                      round(temp_fpr - base_fpr, 6),
            "precision_change":                round(temp_prec - base_prec, 6),
        },
    }

    return metrics


def plot_temporal_sequence(
    A_t: np.ndarray,
    E_t: np.ndarray,
    P_t: np.ndarray,
    y_true: np.ndarray,
    baseline_thr: float,
    temporal_thr: float,
    save_path: Path,
    title: str = "Temporal Risk Dynamics (A_t, E_t, P_t)",
) -> None:
    """
    Plot A_t, E_t, and P_t trajectories over a representative sequence.
    """
    steps = np.arange(len(A_t))
    with plt.style.context(PLOT_STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        # Top panel: A_t and E_t
        ax1.plot(steps, A_t, "o-", color="#1f77b4", label=r"Instantaneous Risk $A_t$", alpha=0.7, markersize=4)
        ax1.plot(steps, E_t, "-", color="#ff7f0e", lw=2, label=r"EMA Evidence $E_t$")
        ax1.axhline(baseline_thr, color="red", linestyle="--", alpha=0.8, label=f"Baseline Threshold ({baseline_thr:.3f})")
        ax1.set_ylabel("Risk / Evidence")
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_title(title)
        ax1.legend(loc="upper left")

        # Bottom panel: P_t accumulator & fraud ground truth
        ax2.plot(steps, P_t, "s-", color="#d62728", lw=2, label=r"Persistent Risk $P_t$", markersize=4)
        ax2.axhline(temporal_thr, color="black", linestyle=":", lw=1.5, label=f"Temporal Threshold ({temporal_thr:.3f})")
        
        # Shade actual fraud steps
        fraud_steps = steps[y_true == 1]
        for fs in fraud_steps:
            ax2.axvline(fs, color="red", alpha=0.2, lw=4)
        if len(fraud_steps) > 0:
            ax2.plot([], [], color="red", alpha=0.4, lw=4, label="True Fraud Event")

        ax2.set_xlabel("Transaction Step (Chronological)")
        ax2.set_ylabel(r"Persistent Accumulator $P_t$")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(loc="upper left")

        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)


def plot_slow_burn_demonstration(
    gamma: float,
    lambda_: float,
    delta: float,
    beta: float,
    baseline_thr: float,
    temporal_thr: float,
    save_path: Path,
) -> Dict:
    """
    Controlled Slow-Burn attack demonstration where every single A_t < baseline_thr,
    yet sustained low-level activity causes P_t to cross temporal_thr.
    """
    # 10 baseline legit steps, 15 sub-threshold fraud steps (A_t=0.40 < baseline_thr=0.594), then 15 recovery steps
    steps_legit1 = [0.05] * 10
    steps_attack = [0.40] * 15   # Sub-threshold attack (A_t < 0.594)
    steps_recovery = [0.02] * 15
    A_stream = np.array(steps_legit1 + steps_attack + steps_recovery)
    y_stream = np.array([0]*10 + [1]*15 + [0]*15)

    from trustgraph.temporal.engine import TemporalRiskEngine
    engine = TemporalRiskEngine(beta=beta, gamma=gamma, lambda_=lambda_, delta=delta)
    E_stream, P_stream = engine.process_stream(A_stream)

    # Calculate detection delay
    attack_start = 10
    detected_steps = np.where((P_stream[attack_start:attack_start+15] >= temporal_thr))[0]
    if len(detected_steps) > 0:
        delay = int(detected_steps[0])
    else:
        delay = -1

    plot_temporal_sequence(
        A_stream, E_stream, P_stream, y_stream,
        baseline_thr, temporal_thr, save_path,
        title=f"Controlled Slow-Burn Sequence (Sub-Threshold A_t=0.40 < {baseline_thr:.3f})"
    )

    return {
        "attack_magnitude_At": 0.40,
        "baseline_threshold": baseline_thr,
        "temporal_threshold": temporal_thr,
        "point_wise_detected": False,
        "temporal_detected": (delay >= 0),
        "detection_delay_steps": delay,
    }
