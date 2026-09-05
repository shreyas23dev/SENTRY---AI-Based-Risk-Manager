"""
calibration.py — Signal Calibration Layer (Phase 3)
====================================================

Evaluates whether A_t and G_t require calibration before fusion and
applies Platt scaling (logistic) or isotonic regression if warranted.

Calibration is fitted on TRAIN only and evaluated on VALIDATION.
Test partition is NEVER used for calibration fitting.

Design decisions:
  - A_t (XGBoost) is generally well-calibrated for tree ensembles.
  - G_t (graph risk) is a hand-crafted [0,1] score, NOT a posterior probability.
    It needs calibration before being treated as a probability-like quantity.
  - Calibration is selective: only applied if it meaningfully improves ECE.
  - Calibration does NOT change signal rank order (monotone transforms only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    Lower is better. 0.0 = perfect calibration.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


# ---------------------------------------------------------------------------
# PlattCalibrator — thin wrapper around logistic regression on raw scores
# ---------------------------------------------------------------------------

class PlattCalibrator:
    """
    Fits a monotone logistic regression (Platt scaling) on raw signal values.

    Parameters learned: slope + intercept of sigmoid transformation.
    Guarantees output is in [0, 1].
    """

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
        self.fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "PlattCalibrator":
        self._lr.fit(scores.reshape(-1, 1), labels)
        self.fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("PlattCalibrator must be fitted before transform.")
        return self._lr.predict_proba(scores.reshape(-1, 1))[:, 1]


# ---------------------------------------------------------------------------
# IsotonicCalibrator
# ---------------------------------------------------------------------------

class IsotonicCalibrator:
    """
    Monotone isotonic regression calibration.
    More flexible than Platt but requires more data.
    """

    def __init__(self) -> None:
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self.fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        self._iso.fit(scores, labels)
        self.fitted = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("IsotonicCalibrator must be fitted before transform.")
        return np.clip(self._iso.predict(scores), 0.0, 1.0)


# ---------------------------------------------------------------------------
# SignalCalibrator — decides whether to calibrate and which method
# ---------------------------------------------------------------------------

@dataclass
class CalibrationReport:
    """Records calibration decision and before/after ECE."""
    signal_name: str
    ece_before: float
    ece_after: float
    method: str          # "none", "platt", "isotonic"
    calibrated: bool
    improvement: float   # ece_before - ece_after


class SignalCalibrator:
    """
    Evaluates calibration need and applies selected method.

    Calibration is applied only if ECE improves by at least `min_ece_gain`.

    Usage::

        cal = SignalCalibrator()
        cal.fit(a_train, g_train, y_train)
        a_cal, g_cal = cal.transform(a_val, g_val)
        report = cal.calibration_report
    """

    ECE_IMPROVEMENT_THRESHOLD = 0.005   # 0.5 pp minimum gain to justify calibration

    def __init__(self) -> None:
        self._a_calibrator: Optional[PlattCalibrator | IsotonicCalibrator] = None
        self._g_calibrator: Optional[PlattCalibrator | IsotonicCalibrator] = None
        self._calibrate_a = False
        self._calibrate_g = False
        self.calibration_report: dict[str, CalibrationReport] = {}
        self.fitted = False

    def fit(
        self,
        a_train: np.ndarray,
        g_train: np.ndarray,
        y_train: np.ndarray,
        a_val: np.ndarray,
        g_val: np.ndarray,
        y_val: np.ndarray,
        method: Literal["platt", "isotonic"] = "platt",
    ) -> "SignalCalibrator":
        """
        Fit calibrators on TRAIN and evaluate gain on VALIDATION.

        Parameters
        ----------
        method : "platt" | "isotonic"
            Calibration method to try.
        """
        for signal_name, (scores_train, scores_val, labels_train, labels_val) in [
            ("A_t", (a_train, a_val, y_train, y_val)),
            ("G_t", (g_train, g_val, y_train, y_val)),
        ]:
            ece_before = expected_calibration_error(labels_val, scores_val)

            # Fit calibrator on TRAIN
            if method == "platt":
                cal = PlattCalibrator().fit(scores_train, labels_train)
            else:
                cal = IsotonicCalibrator().fit(scores_train, labels_train)

            # Evaluate on VAL
            scores_cal = cal.transform(scores_val)
            ece_after = expected_calibration_error(labels_val, scores_cal)

            improvement = ece_before - ece_after
            should_calibrate = improvement >= self.ECE_IMPROVEMENT_THRESHOLD

            report = CalibrationReport(
                signal_name=signal_name,
                ece_before=round(ece_before, 6),
                ece_after=round(ece_after, 6),
                method=method if should_calibrate else "none",
                calibrated=should_calibrate,
                improvement=round(improvement, 6),
            )
            self.calibration_report[signal_name] = report

            if signal_name == "A_t":
                self._calibrate_a = should_calibrate
                if should_calibrate:
                    # Re-fit on full train for production use
                    if method == "platt":
                        self._a_calibrator = PlattCalibrator().fit(scores_train, labels_train)
                    else:
                        self._a_calibrator = IsotonicCalibrator().fit(scores_train, labels_train)
            else:
                self._calibrate_g = should_calibrate
                if should_calibrate:
                    if method == "platt":
                        self._g_calibrator = PlattCalibrator().fit(scores_train, labels_train)
                    else:
                        self._g_calibrator = IsotonicCalibrator().fit(scores_train, labels_train)

            logger.info(
                "[Calibration] %s: ECE %.4f → %.4f (Δ=%.4f) — %s",
                signal_name, ece_before, ece_after, improvement,
                "APPLIED" if should_calibrate else "SKIPPED",
            )

        self.fitted = True
        return self

    def transform(
        self, a: np.ndarray, g: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply calibration (if fitted) to signals."""
        if not self.fitted:
            raise RuntimeError("SignalCalibrator must be fitted before transform.")
        a_out = self._a_calibrator.transform(a) if self._calibrate_a else a.copy()
        g_out = self._g_calibrator.transform(g) if self._calibrate_g else g.copy()
        return a_out, g_out

    def transform_single(self, a: float, g: float) -> Tuple[float, float]:
        """Single-transaction calibration transform."""
        a_arr, g_arr = self.transform(
            np.array([a], dtype=np.float64),
            np.array([g], dtype=np.float64),
        )
        return float(a_arr[0]), float(g_arr[0])

    def summary(self) -> dict:
        return {
            name: {
                "ece_before": r.ece_before,
                "ece_after": r.ece_after,
                "method": r.method,
                "calibrated": r.calibrated,
                "improvement": r.improvement,
            }
            for name, r in self.calibration_report.items()
        }
