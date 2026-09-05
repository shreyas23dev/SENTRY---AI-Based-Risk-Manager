"""
evaluate.py — TRUSTGRAPH Phase 1 Baseline Evaluation
=====================================================

Responsibilities:
  - Select binary decision threshold on validation set (maximises F1)
  - Compute all required metrics: ROC-AUC, PR-AUC, Precision, Recall,
    F1, FPR, FNR
  - Generate evaluation plots:
      1. Precision-Recall curve
      2. ROC curve
      3. Confusion matrix
      4. A_t distribution (fraud vs legitimate)
  - Measure inference latency

Leakage guarantee:
  - Threshold selection uses VALIDATION only
  - TEST metrics computed with the frozen threshold
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threshold selection (validation only)
# ---------------------------------------------------------------------------

def select_threshold_max_f1(
    y_true: np.ndarray,
    A_t: np.ndarray,
    beta: float = 1.0,
) -> Tuple[float, float]:
    """
    Select the classification threshold that maximises F-beta score
    on the validation set.

    Parameters
    ----------
    y_true : ground-truth binary labels (0 / 1)
    A_t    : continuous risk scores in [0, 1]
    beta   : F-beta parameter (default 1.0 = F1)

    Returns
    -------
    best_threshold : float
    best_f_score   : float
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, A_t)

    # Compute F-beta at each threshold point
    # precision_recall_curve returns n+1 points; thresholds has n points
    beta_sq = beta ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        f_scores = np.where(
            (beta_sq * precisions[:-1] + recalls[:-1]) > 0,
            (1 + beta_sq) * precisions[:-1] * recalls[:-1] / (beta_sq * precisions[:-1] + recalls[:-1]),
            0.0,
        )

    best_idx = int(np.argmax(f_scores))
    best_threshold = float(thresholds[best_idx])
    best_f_score   = float(f_scores[best_idx])

    logger.info(
        "Threshold selection (max F%.0f): threshold=%.4f, F%.0f=%.4f",
        beta, best_threshold, beta, best_f_score
    )
    return best_threshold, best_f_score


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    A_t: np.ndarray,
    threshold: float,
    partition_name: str = "test",
) -> Dict:
    """
    Compute all required evaluation metrics.

    Parameters
    ----------
    y_true         : ground-truth binary labels
    A_t            : continuous risk scores
    threshold      : frozen decision threshold
    partition_name : label for logging

    Returns
    -------
    metrics : dict with all required metrics
    """
    y_pred = (A_t >= threshold).astype(int)

    roc_auc  = float(roc_auc_score(y_true, A_t))
    pr_auc   = float(average_precision_score(y_true, A_t))
    prec     = float(precision_score(y_true, y_pred, zero_division=0))
    rec      = float(recall_score(y_true, y_pred, zero_division=0))
    f1       = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    n_total   = int(len(y_true))
    n_fraud   = int(y_true.sum())
    n_legit   = int(n_total - n_fraud)
    prevalence = float(n_fraud / n_total)

    metrics = {
        "partition":         partition_name,
        "threshold":         float(threshold),
        "total_transactions": n_total,
        "fraudulent":        n_fraud,
        "legitimate":        n_legit,
        "fraud_prevalence":  round(prevalence, 6),
        "roc_auc":           round(roc_auc, 6),
        "pr_auc":            round(pr_auc, 6),
        "precision":         round(prec, 6),
        "recall":            round(rec, 6),
        "f1":                round(f1, 6),
        "fpr":               round(fpr, 6),
        "fnr":               round(fnr, 6),
        "tp":                int(tp),
        "fp":                int(fp),
        "tn":                int(tn),
        "fn":                int(fn),
    }

    logger.info(
        "[%s] ROC-AUC=%.4f  PR-AUC=%.4f  F1=%.4f  Prec=%.4f  Rec=%.4f  FPR=%.4f  FNR=%.4f",
        partition_name.upper(),
        roc_auc, pr_auc, f1, prec, rec, fpr, fnr
    )

    return metrics


# ---------------------------------------------------------------------------
# Inference latency
# ---------------------------------------------------------------------------

def measure_latency(
    model,
    X: pd.DataFrame,
    n_warmup: int = 3,
    n_repeats: int = 5,
) -> Dict:
    """
    Measure model inference latency over n_repeats runs.

    Parameters
    ----------
    model    : BaselineModel
    X        : feature matrix to score
    n_warmup : warm-up runs (not counted)
    n_repeats: timed runs

    Returns
    -------
    latency_info : dict with timing statistics
    """
    # Warm-up
    for _ in range(n_warmup):
        _ = model.predict_risk(X)

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        _ = model.predict_risk(X)
        times.append(time.perf_counter() - t0)

    total_rows = len(X)
    avg_total  = float(np.mean(times))
    avg_per_txn = avg_total / total_rows

    latency_info = {
        "n_transactions":       total_rows,
        "n_repeats":            n_repeats,
        "mean_total_s":         round(avg_total, 4),
        "std_total_s":          round(float(np.std(times)), 4),
        "min_total_s":          round(float(np.min(times)), 4),
        "max_total_s":          round(float(np.max(times)), 4),
        "mean_per_transaction_ms": round(avg_per_txn * 1000, 6),
        "throughput_txn_per_s": round(total_rows / avg_total, 1),
        "note": "Model-only inference (LightGBM predict_proba). Excludes preprocessing.",
    }

    logger.info(
        "Latency: mean=%.3fs over %d transactions (%.4f ms/txn, %.1f txn/s)",
        avg_total, total_rows, avg_per_txn * 1000, total_rows / avg_total
    )
    return latency_info


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "font.size":        11,
}


def plot_roc_curve(
    y_true: np.ndarray,
    A_t: np.ndarray,
    save_path: Path,
    title: str = "ROC Curve — TEST partition",
) -> None:
    """Precision-agnostic ROC curve plot."""
    fpr_vals, tpr_vals, _ = roc_curve(y_true, A_t)
    roc_auc = auc(fpr_vals, tpr_vals)

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr_vals, tpr_vals, lw=2, color="#1f77b4",
                label=f"LightGBM baseline (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    logger.info("ROC curve saved → %s", save_path)


def plot_pr_curve(
    y_true: np.ndarray,
    A_t: np.ndarray,
    save_path: Path,
    title: str = "Precision-Recall Curve — TEST partition",
) -> None:
    """Precision-Recall curve (important for imbalanced data)."""
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, A_t)
    pr_auc = average_precision_score(y_true, A_t)
    baseline_rate = float(y_true.mean())

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(rec_vals, prec_vals, lw=2, color="#2ca02c",
                label=f"LightGBM baseline (AP = {pr_auc:.4f})")
        ax.axhline(baseline_rate, color="k", linestyle="--", lw=1,
                   label=f"No-skill baseline ({baseline_rate:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    logger.info("PR curve saved → %s", save_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    A_t: np.ndarray,
    threshold: float,
    save_path: Path,
    title: str = "Confusion Matrix — TEST partition",
) -> None:
    """Annotated heatmap confusion matrix."""
    y_pred = (A_t >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Legit", "Predicted Fraud"],
            yticklabels=["Actual Legit", "Actual Fraud"],
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(f"{title}\n(threshold={threshold:.4f})")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    logger.info("Confusion matrix saved → %s", save_path)


def plot_risk_score_distribution(
    y_true: np.ndarray,
    A_t: np.ndarray,
    threshold: float,
    save_path: Path,
    title: str = "A_t Distribution — Fraud vs Legitimate",
) -> None:
    """
    Histogram of predicted risk scores A_t, separated by true class.
    Shows how well the model separates fraud from legitimate transactions.
    """
    A_t_fraud  = A_t[y_true == 1]
    A_t_legit  = A_t[y_true == 0]

    with plt.style.context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(A_t_legit, bins=100, alpha=0.6, color="#1f77b4",
                label=f"Legitimate (n={len(A_t_legit):,})", density=True, range=(0, 1))
        ax.hist(A_t_fraud, bins=100, alpha=0.7, color="#d62728",
                label=f"Fraud (n={len(A_t_fraud):,})", density=True, range=(0, 1))
        ax.axvline(threshold, color="black", linestyle="--", lw=1.5,
                   label=f"Threshold = {threshold:.4f}")
        ax.set_xlabel("A_t  =  P(isFraud = 1 | X_t)")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    logger.info("Risk score distribution saved → %s", save_path)


def generate_all_plots(
    y_true: np.ndarray,
    A_t: np.ndarray,
    threshold: float,
    plots_dir: Path,
    partition: str = "test",
) -> Dict[str, str]:
    """
    Generate and save all four required evaluation plots.

    Returns a dict mapping plot name → file path string.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    tag = partition.upper()

    paths = {}

    p = plots_dir / "roc_curve.png"
    plot_roc_curve(y_true, A_t, p, title=f"ROC Curve — {tag} partition")
    paths["roc_curve"] = str(p)

    p = plots_dir / "pr_curve.png"
    plot_pr_curve(y_true, A_t, p, title=f"Precision-Recall Curve — {tag} partition")
    paths["pr_curve"] = str(p)

    p = plots_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, A_t, threshold, p, title=f"Confusion Matrix — {tag} partition")
    paths["confusion_matrix"] = str(p)

    p = plots_dir / "risk_score_distribution.png"
    plot_risk_score_distribution(y_true, A_t, threshold, p,
                                 title=f"A_t Distribution — {tag} Fraud vs Legitimate")
    paths["risk_score_distribution"] = str(p)

    return paths
