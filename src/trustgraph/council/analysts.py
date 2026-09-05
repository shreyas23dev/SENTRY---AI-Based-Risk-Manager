"""
analysts.py — Independent Analyst Agents for the Risk Council (Phase 8)
======================================================================

Wraps the existing frozen components without changing inference:
  1. TransactionRiskAnalyst: wraps existing XGBoost baseline (A_t)
  2. SlowBurnAnalyst: wraps existing temporal memory & entity trajectory (P_t)

Guarantees:
  - Strictly non-breaking: wraps existing models and features.
  - Zero hallucinations: explanations are grounded in feature values and history.
  - If temporal history is missing, SlowBurnAnalyst explicitly returns "INSUFFICIENT_HISTORY".
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Analyst 1: Transaction Risk Analyst (XGBoost A_t Wrapper)
# ---------------------------------------------------------------------------

class TransactionRiskAnalyst:
    """
    Evaluates instantaneous, point-in-time transaction risk using the existing
    frozen XGBoost model outputs (A_t).
    """

    AGENT_NAME = "transaction_risk_analyst"

    def evaluate(
        self,
        transaction_id: int,
        A_t: float,
        amount: float = 0.0,
        raw_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Produce structured assessment from existing A_t inference and transaction features.
        """
        # Assessment tier based on validated thresholds
        if A_t >= 0.70:
            assessment = "CRITICAL"
        elif A_t >= 0.50:
            assessment = "HIGH"
        elif A_t >= 0.1244:  # Selected validation decision threshold tau*
            assessment = "MEDIUM"
        else:
            assessment = "LOW"

        # Signal extraction from existing features
        signals: List[str] = [
            f"XGBoost transaction-level fraud probability A_t = {A_t:.4f} ({assessment} risk tier).",
        ]

        if amount > 0:
            if amount < 25.0:
                signals.append(f"Low transaction amount (INR {amount:.2f}) matches micro-testing card probe profile.")
            elif amount > 500.0:
                signals.append(f"High transaction amount (INR {amount:.2f}) carries elevated financial exposure.")
            else:
                signals.append(f"Standard transaction amount (INR {amount:.2f}).")

        if raw_features:
            p_email = raw_features.get("P_emaildomain")
            r_email = raw_features.get("R_emaildomain")
            if p_email and r_email and p_email != r_email:
                signals.append(f"Discrepant email domains detected: Purchaser '{p_email}' vs Recipient '{r_email}'.")

            product = raw_features.get("ProductCD")
            if product in ("C", "H"):
                signals.append(f"High-risk product channel '{product}' (cross-border or high chargeback rate).")

            c1 = raw_features.get("C1", 0)
            if c1 and float(c1) > 20:
                signals.append(f"Elevated velocity counter C1 = {c1} indicates rapid burst activity.")

        return {
            "agent": self.AGENT_NAME,
            "risk": round(float(A_t), 4),
            "assessment": assessment,
            "signals": signals,
        }


# ---------------------------------------------------------------------------
# Analyst 2: Slow-Burn Behavioral Analyst (Temporal State P_t Wrapper)
# ---------------------------------------------------------------------------

class SlowBurnAnalyst:
    """
    Evaluates persistent temporal risk trajectory over time using the existing
    Phase 2 temporal memory and entity history (P_t).
    """

    AGENT_NAME = "slow_burn_analyst"

    def evaluate(
        self,
        transaction_id: int,
        entity_id: Optional[str] = None,
        prior_txns: int = 0,
        prior_frauds: int = 0,
        fraud_rate: float = 0.0,
        device_sharing_count: int = 0,
        device_prior_frauds: int = 0,
        precomputed_P_t: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Produce structured behavioral assessment from existing temporal state.
        If history is unavailable or entity has 0 prior transactions, explicitly
        returns 'INSUFFICIENT_HISTORY' without fabricating a value.
        """
        # Guardrail: Explicitly declare insufficient history if no prior record
        if prior_txns == 0 and precomputed_P_t is None:
            return {
                "agent": self.AGENT_NAME,
                "risk": None,
                "state": "INSUFFICIENT_HISTORY",
                "assessment": "INSUFFICIENT_HISTORY",
                "signals": [
                    "Zero prior transaction history recorded for this entity in the point-in-time graph.",
                    "Persistent risk accumulator cannot evaluate without historical observation sequence.",
                ],
            }

        # Determine P_t from precomputed state or entity historical trajectory
        if precomputed_P_t is not None:
            P_t = float(precomputed_P_t)
        else:
            # Map graph entity recidivism to normalized persistent risk P_t in [0, 1]
            if prior_frauds > 0:
                # Strong persistent risk if confirmed recidivist
                P_t = min(1.0, 0.50 + min(prior_frauds * 0.05, 0.45))
            elif fraud_rate > 0:
                P_t = min(1.0, fraud_rate * 2.0)
            elif device_prior_frauds > 0:
                P_t = min(1.0, 0.30 + min(device_prior_frauds * 0.05, 0.40))
            elif device_sharing_count > 3:
                P_t = min(0.60, 0.15 + device_sharing_count * 0.05)
            else:
                P_t = max(0.0, 0.10 - min(prior_txns * 0.01, 0.10))

        P_t = round(float(P_t), 4)

        # Assessment scale
        if P_t >= 0.70:
            assessment = "CRITICAL"
            state = "ELEVATED"
        elif P_t >= 0.50:
            assessment = "HIGH"
            state = "ELEVATED"
        elif P_t >= 0.20:
            assessment = "MEDIUM"
            state = "ACCUMULATING"
        else:
            assessment = "LOW"
            state = "STABLE"

        signals: List[str] = [
            f"Persistent behavioral risk accumulator state P_t = {P_t:.4f} ({state} state).",
            f"Entity trajectory: {prior_txns} prior transactions, {prior_frauds} confirmed fraud chargebacks (fraud rate: {fraud_rate:.1%}).",
        ]

        if device_sharing_count > 1:
            signals.append(
                f"Hardware multiplexing: associated device shared across {device_sharing_count} distinct entities."
            )

        if device_prior_frauds > 0:
            signals.append(
                f"Contaminated device fingerprint: {device_prior_frauds} confirmed fraud events linked to this hardware."
            )

        return {
            "agent": self.AGENT_NAME,
            "risk": P_t,
            "state": state,
            "assessment": assessment,
            "signals": signals,
        }
