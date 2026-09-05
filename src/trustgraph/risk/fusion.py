"""
fusion.py — Mathematical Risk Fusion Engine (Phase 3)
=====================================================

Implements and compares four candidate fusion formulations that combine
A_t (XGBoost transaction-level fraud probability) and G_t (knowledge-graph
contextual risk score) into a final risk score R_t ∈ [0, 1].

Formulations
------------
F1 — Additive
    R_t = clip(A_t + α·G_t, 0, 1)
    Rationale: simple baseline; assumes G_t is already on probability scale.
    Risk: can push R_t well above 1 if both signals are high.

F2 — Conditional graph contribution (PREFERRED)
    R_t = clip(A_t + β·G_t·(1 − A_t), 0, 1)
    Rationale: G_t can only ADD residual risk above what A_t already captures.
    When A_t → 1, G_t contribution → 0 (already maximum risk).
    When A_t → 0, G_t can contribute up to β of residual space.
    Preserves A_t calibration while adding context-sensitive uplift.

F3 — Conditional multi-signal
    R_t = clip(A_t + α·P_t·(1 − A_t) + β·G_t·(1 − A_t), 0, 1)
    Skipped: P_t (temporal entity memory) is not available as a separate
    validated signal in the Phase 1/2 architecture. The legacy relational
    engine uses different feature scales and is not a valid posterior estimate.

F4 — Conservative maximum
    R_t = clip(max(A_t, c·G_t), 0, 1)
    Rationale: take the most alarming signal. Never dilutes a high-confidence
    ML fraud signal with low graph context, but graph alone can trigger action.

Parameter Tuning
----------------
All parameters (α, β, c) are selected by grid search on VALIDATION F1/ROC-AUC
maximization. TEST labels are never used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class FusionFormula(str, Enum):
    F1 = "F1_additive"
    F2 = "F2_conditional"
    F4 = "F4_conservative_max"


@dataclass
class FusionResult:
    """Per-transaction fusion output."""
    base_risk: float          # A_t (calibrated if applicable)
    graph_risk: float         # G_t (calibrated if applicable)
    final_risk: float         # R_t
    formula: FusionFormula
    graph_contribution: float # G_t's signed contribution to R_t


@dataclass
class TuningResult:
    """Validation-set tuning outcome for one formula."""
    formula: FusionFormula
    params: Dict[str, float]
    val_roc_auc: float
    val_pr_auc: float
    val_f1: float
    val_threshold: float     # optimal classification threshold on VAL


# ---------------------------------------------------------------------------
# Fusion equations (vectorized, parameter-free)
# ---------------------------------------------------------------------------

def _f1_additive(a: np.ndarray, g: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(a + alpha * g, 0.0, 1.0)


def _f2_conditional(a: np.ndarray, g: np.ndarray, beta: float) -> np.ndarray:
    return np.clip(a + beta * g * (1.0 - a), 0.0, 1.0)


def _f4_conservative_max(a: np.ndarray, g: np.ndarray, c: float) -> np.ndarray:
    return np.clip(np.maximum(a, c * g), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Threshold search (VALIDATION ONLY)
# ---------------------------------------------------------------------------

def _best_threshold_by_f1(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[float, float]:
    """Return (threshold, f1) that maximises F1 on the given partition."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = np.where(
        (precision + recall) == 0,
        0.0,
        2 * precision * recall / (precision + recall),
    )
    best_idx = int(np.argmax(f1_scores))
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


# ---------------------------------------------------------------------------
# Main FusionEngine
# ---------------------------------------------------------------------------

class FusionEngine:
    """
    Tunes and applies fusion formulas using VALIDATION data only.

    Usage::

        engine = FusionEngine()
        engine.tune(a_val, g_val, y_val)          # select best formula + params
        r_t = engine.fuse_batch(a_test, g_test)    # apply to TEST
        result = engine.fuse_single(a, g)          # single-transaction
    """

    # Parameter grids for VALIDATION grid search
    ALPHA_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    BETA_GRID  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    C_GRID     = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20]

    def __init__(self) -> None:
        self.best_formula: Optional[FusionFormula] = None
        self.best_params: Dict[str, float] = {}
        self.best_threshold: float = 0.5
        self.tuning_results: list[TuningResult] = []
        self.fitted = False

    # -----------------------------------------------------------------------
    # Internal: evaluate one formula + param on VAL
    # -----------------------------------------------------------------------

    def _evaluate(
        self,
        a_val: np.ndarray,
        g_val: np.ndarray,
        y_val: np.ndarray,
        formula: FusionFormula,
        params: Dict[str, float],
    ) -> TuningResult:
        r = self._apply(a_val, g_val, formula, params)
        roc = float(roc_auc_score(y_val, r))
        pr  = float(average_precision_score(y_val, r))
        tau, f1 = _best_threshold_by_f1(y_val, r)
        return TuningResult(
            formula=formula,
            params=params,
            val_roc_auc=round(roc, 6),
            val_pr_auc=round(pr, 6),
            val_f1=round(f1, 6),
            val_threshold=round(tau, 6),
        )

    def _apply(
        self,
        a: np.ndarray,
        g: np.ndarray,
        formula: FusionFormula,
        params: Dict[str, float],
    ) -> np.ndarray:
        if formula == FusionFormula.F1:
            return _f1_additive(a, g, params["alpha"])
        elif formula == FusionFormula.F2:
            return _f2_conditional(a, g, params["beta"])
        elif formula == FusionFormula.F4:
            return _f4_conservative_max(a, g, params["c"])
        else:
            raise ValueError(f"Unknown formula: {formula}")

    # -----------------------------------------------------------------------
    # Tune — VALIDATION ONLY
    # -----------------------------------------------------------------------

    def tune(
        self,
        a_val: np.ndarray,
        g_val: np.ndarray,
        y_val: np.ndarray,
        selection_metric: str = "pr_auc",
    ) -> "FusionEngine":
        """
        Grid-search parameters for F1, F2, F4 on VALIDATION.

        selection_metric : "pr_auc" | "roc_auc" | "f1"
            Primary metric used to pick the best formula+params combination.
            PR-AUC is preferred for imbalanced fraud detection.
        """
        logger.info("Tuning fusion parameters on VALIDATION (N=%d)...", len(y_val))
        all_results: list[TuningResult] = []

        # F1 grid
        for alpha in self.ALPHA_GRID:
            res = self._evaluate(a_val, g_val, y_val, FusionFormula.F1, {"alpha": alpha})
            all_results.append(res)

        # F2 grid
        for beta in self.BETA_GRID:
            res = self._evaluate(a_val, g_val, y_val, FusionFormula.F2, {"beta": beta})
            all_results.append(res)

        # F4 grid
        for c in self.C_GRID:
            res = self._evaluate(a_val, g_val, y_val, FusionFormula.F4, {"c": c})
            all_results.append(res)

        self.tuning_results = all_results

        # Select best
        metric_key = {"pr_auc": "val_pr_auc", "roc_auc": "val_roc_auc", "f1": "val_f1"}[selection_metric]
        best = max(all_results, key=lambda r: getattr(r, metric_key))

        self.best_formula = best.formula
        self.best_params  = best.params
        self.best_threshold = best.val_threshold
        self.fitted = True

        logger.info(
            "Selected: %s | params=%s | %s=%.4f | τ=%.4f",
            best.formula, best.params, selection_metric, getattr(best, metric_key), best.val_threshold
        )
        return self

    # -----------------------------------------------------------------------
    # Public — apply frozen fusion
    # -----------------------------------------------------------------------

    def fuse_batch(
        self, a: np.ndarray, g: np.ndarray
    ) -> np.ndarray:
        """Apply frozen fusion to a batch of (A_t, G_t) signals."""
        if not self.fitted:
            raise RuntimeError("FusionEngine must be tuned before fuse_batch().")
        return self._apply(a, g, self.best_formula, self.best_params)

    def fuse_single(self, a: float, g: float) -> FusionResult:
        """Apply frozen fusion to a single (A_t, G_t) and return full result."""
        if not self.fitted:
            raise RuntimeError("FusionEngine must be tuned before fuse_single().")
        r = float(self._apply(
            np.array([a], dtype=np.float64),
            np.array([g], dtype=np.float64),
            self.best_formula,
            self.best_params,
        )[0])
        # Compute graph contribution (signed)
        if self.best_formula == FusionFormula.F1:
            contrib = self.best_params["alpha"] * g
        elif self.best_formula == FusionFormula.F2:
            contrib = self.best_params["beta"] * g * (1.0 - a)
        elif self.best_formula == FusionFormula.F4:
            contrib = max(0.0, self.best_params["c"] * g - a)
        else:
            contrib = 0.0
        return FusionResult(
            base_risk=round(a, 6),
            graph_risk=round(g, 6),
            final_risk=round(r, 6),
            formula=self.best_formula,
            graph_contribution=round(contrib, 6),
        )

    def apply_formula(
        self,
        a: np.ndarray,
        g: np.ndarray,
        formula: FusionFormula,
        params: Dict[str, float],
    ) -> np.ndarray:
        """Apply a specific formula with specific params (for comparison reporting)."""
        return self._apply(a, g, formula, params)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def summary(self) -> dict:
        if not self.fitted:
            return {"fitted": False}
        return {
            "fitted": True,
            "selected_formula": self.best_formula.value if self.best_formula else None,
            "selected_params": self.best_params,
            "selected_threshold": self.best_threshold,
            "tuning_results_count": len(self.tuning_results),
        }

    def best_per_formula(self) -> Dict[str, TuningResult]:
        """Return the best result for each formula family."""
        out: Dict[str, TuningResult] = {}
        for formula in FusionFormula:
            candidates = [r for r in self.tuning_results if r.formula == formula]
            if candidates:
                out[formula.value] = max(candidates, key=lambda r: r.val_pr_auc)
        return out
