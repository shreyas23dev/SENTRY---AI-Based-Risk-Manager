"""
cost_model.py — Merchant-Configurable Cost Model (Phase 3)
===========================================================

Models the expected financial and friction costs of each possible action
given the posterior fraud risk R_t.

Cost Model Design
-----------------
For a transaction with fraud probability R_t and transaction amount TxnAmt:

  ALLOW:
    E[cost | ALLOW, R_t] = R_t * C_fraud * TxnAmt

    If fraud goes undetected, merchant bears the full chargeback (transaction
    amount) plus a chargeback fee.

  VERIFY:
    E[cost | VERIFY, R_t] = R_t * C_fraud_after_verify * TxnAmt
                           + (1 - R_t) * C_fp_friction
                           + C_verify_fixed

    Verification reduces fraud risk by a factor (1 - verify_fraud_reduction).
    Legitimate customers bear friction cost proportional to UX degradation.
    A fixed operational cost covers SMS OTP / email verification system cost.

  THROTTLE:
    E[cost | THROTTLE, R_t] = R_t * C_fraud_after_throttle * TxnAmt
                             + (1 - R_t) * C_throttle_friction
                             + C_throttle_fixed

    Throttling limits velocity/amount but doesn't block. Reduces expected
    fraud by a partial factor, causes less friction than VERIFY.

  BLOCK:
    E[cost | BLOCK, R_t] = (1 - R_t) * C_fp_block

    If we block a legitimate transaction, we lose merchant revenue and
    damage customer trust. If we block fraud, cost = 0 (prevented).

Action Selection
----------------
    action* = argmin E[cost | action, R_t]

All parameters are merchant-configurable and documented explicitly.
No parameter is tuned using TEST labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost Scenario
# ---------------------------------------------------------------------------

@dataclass
class CostScenario:
    """
    A named set of merchant cost parameters.

    All monetary values are in normalised units (fraction of transaction amount)
    or absolute INR amounts, depending on the field.

    Attributes
    ----------
    name : str
        Scenario identifier (e.g. "conservative", "balanced", "aggressive").

    C_fraud_rate : float
        Fraction of transaction amount lost to fraud chargeback (0.0 – 1.0).
        Typically 1.0 (full amount + chargeback fee).
        Default = 1.0.

    C_chargeback_fee : float
        Fixed INR penalty per chargeback (e.g. ₹1,000 Razorpay chargeback fee).
        Default = 1000.

    verify_fraud_reduction : float
        Fraction of fraud risk eliminated by VERIFY (e.g. OTP). (0.0 – 1.0)
        Default = 0.70 (70% of fraud cases caught by OTP).

    throttle_fraud_reduction : float
        Fraction of fraud risk eliminated by THROTTLE. (0.0 – 1.0)
        Default = 0.30.

    C_fp_friction_verify : float
        Expected friction cost per falsely verified legitimate customer (INR).
        Represents brand damage / cart abandonment.
        Default = 150 (₹150 per interrupted legitimate txn).

    C_fp_friction_throttle : float
        Friction cost per throttled legitimate customer (INR).
        Default = 50 (less friction than VERIFY).

    C_fp_block : float
        Cost per blocked legitimate transaction (INR).
        Includes lost GMV commission + brand damage.
        Default = 500.

    C_verify_fixed : float
        Fixed operational cost of running VERIFY flow (INR).
        Covers SMS/email OTP infrastructure cost.
        Default = 2 (₹2 per OTP sent).

    C_throttle_fixed : float
        Fixed cost of THROTTLE (INR). Usually near zero.
        Default = 0.

    avg_transaction_amount : float
        Assumed average transaction amount (INR) when TxnAmt not available.
        Default = 3500 (reasonable for Indian e-commerce).
    """
    name: str = "balanced"
    C_fraud_rate: float = 1.0
    C_chargeback_fee: float = 1000.0
    verify_fraud_reduction: float = 0.70
    throttle_fraud_reduction: float = 0.30
    C_fp_friction_verify: float = 150.0
    C_fp_friction_throttle: float = 50.0
    C_fp_block: float = 500.0
    C_verify_fixed: float = 2.0
    C_throttle_fixed: float = 0.0
    avg_transaction_amount: float = 3500.0

    def validate(self) -> None:
        assert 0.0 <= self.C_fraud_rate <= 1.0, "C_fraud_rate must be in [0, 1]"
        assert 0.0 <= self.verify_fraud_reduction <= 1.0
        assert 0.0 <= self.throttle_fraud_reduction <= 1.0
        assert self.C_fp_friction_verify >= 0
        assert self.C_fp_friction_throttle >= 0
        assert self.C_fp_block >= 0


# ---------------------------------------------------------------------------
# Pre-defined scenarios
# ---------------------------------------------------------------------------

DEFAULT_SCENARIOS: Dict[str, CostScenario] = {
    "conservative": CostScenario(
        name="conservative",
        # High fraud tolerance — minimise false-positive friction (e.g. luxury goods)
        C_fraud_rate=1.0,
        C_chargeback_fee=1000.0,
        verify_fraud_reduction=0.70,
        throttle_fraud_reduction=0.30,
        C_fp_friction_verify=400.0,    # High: customer is premium/sensitive
        C_fp_friction_throttle=150.0,
        C_fp_block=2000.0,
        C_verify_fixed=2.0,
        C_throttle_fixed=0.0,
        avg_transaction_amount=8000.0,
    ),
    "balanced": CostScenario(
        name="balanced",
        # Balanced — typical e-commerce merchant
        C_fraud_rate=1.0,
        C_chargeback_fee=1000.0,
        verify_fraud_reduction=0.70,
        throttle_fraud_reduction=0.30,
        C_fp_friction_verify=150.0,
        C_fp_friction_throttle=50.0,
        C_fp_block=500.0,
        C_verify_fixed=2.0,
        C_throttle_fixed=0.0,
        avg_transaction_amount=3500.0,
    ),
    "aggressive": CostScenario(
        name="aggressive",
        # Low fraud tolerance — financial services / high-value
        C_fraud_rate=1.0,
        C_chargeback_fee=2000.0,
        verify_fraud_reduction=0.80,
        throttle_fraud_reduction=0.40,
        C_fp_friction_verify=75.0,     # Low: willing to create friction
        C_fp_friction_throttle=25.0,
        C_fp_block=200.0,
        C_verify_fixed=2.0,
        C_throttle_fixed=0.0,
        avg_transaction_amount=2000.0,
    ),
}


# ---------------------------------------------------------------------------
# Cost Model
# ---------------------------------------------------------------------------

@dataclass
class ActionCosts:
    """Per-action expected costs for a single transaction."""
    allow: float
    verify: float
    throttle: float
    block: float
    risk: float
    txn_amount: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "allow": round(self.allow, 4),
            "verify": round(self.verify, 4),
            "throttle": round(self.throttle, 4),
            "block": round(self.block, 4),
            "risk": round(self.risk, 6),
            "txn_amount": round(self.txn_amount, 2),
        }

    def optimal_action(self) -> str:
        costs = {
            "ALLOW": self.allow,
            "VERIFY": self.verify,
            "THROTTLE": self.throttle,
            "BLOCK": self.block,
        }
        return min(costs, key=costs.get)


class MerchantCostModel:
    """
    Computes expected costs per action using a CostScenario.

    All formulas are documented above. Parameters are merchant-configurable.
    No cost parameter is tuned using TEST labels.

    Usage::

        model = MerchantCostModel(DEFAULT_SCENARIOS["balanced"])
        costs = model.compute_action_costs(risk=0.72, txn_amount=5000.0)
        action = costs.optimal_action()
    """

    def __init__(self, scenario: CostScenario) -> None:
        scenario.validate()
        self.scenario = scenario

    def compute_action_costs(
        self,
        risk: float,
        txn_amount: Optional[float] = None,
    ) -> ActionCosts:
        """
        Compute expected cost for each action given R_t and transaction amount.

        Parameters
        ----------
        risk : float
            Final fused risk score R_t ∈ [0, 1].
        txn_amount : float or None
            Transaction amount in INR. If None, uses scenario average.
        """
        if txn_amount is None or np.isnan(txn_amount) or txn_amount <= 0:
            txn_amount = self.scenario.avg_transaction_amount

        s = self.scenario
        r = float(np.clip(risk, 0.0, 1.0))
        amt = float(txn_amount)

        # ALLOW: expected fraud loss
        cost_allow = r * (s.C_fraud_rate * amt + s.C_chargeback_fee)

        # VERIFY: residual fraud + FP friction + fixed OTP cost
        residual_fraud_verify = r * (1.0 - s.verify_fraud_reduction)
        cost_verify = (
            residual_fraud_verify * (s.C_fraud_rate * amt + s.C_chargeback_fee)
            + (1.0 - r) * s.C_fp_friction_verify
            + s.C_verify_fixed
        )

        # THROTTLE: partial fraud reduction + friction
        residual_fraud_throttle = r * (1.0 - s.throttle_fraud_reduction)
        cost_throttle = (
            residual_fraud_throttle * (s.C_fraud_rate * amt + s.C_chargeback_fee)
            + (1.0 - r) * s.C_fp_friction_throttle
            + s.C_throttle_fixed
        )

        # BLOCK: legitimate customer loss (fraud blocked = ₹0 cost)
        cost_block = (1.0 - r) * s.C_fp_block

        return ActionCosts(
            allow=cost_allow,
            verify=cost_verify,
            throttle=cost_throttle,
            block=cost_block,
            risk=r,
            txn_amount=amt,
        )

    def compute_batch_costs(
        self,
        risks: np.ndarray,
        txn_amounts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Vectorized batch computation of optimal actions.

        Returns
        -------
        np.ndarray of str with shape (N,): optimal action per transaction.
        """
        if txn_amounts is None:
            txn_amounts = np.full(len(risks), self.scenario.avg_transaction_amount)

        s = self.scenario
        r = np.clip(risks, 0.0, 1.0)
        amt = np.where(
            np.isnan(txn_amounts) | (txn_amounts <= 0),
            s.avg_transaction_amount,
            txn_amounts,
        )
        fraud_cost_base = s.C_fraud_rate * amt + s.C_chargeback_fee

        cost_allow = r * fraud_cost_base
        cost_verify = (
            r * (1.0 - s.verify_fraud_reduction) * fraud_cost_base
            + (1.0 - r) * s.C_fp_friction_verify
            + s.C_verify_fixed
        )
        cost_throttle = (
            r * (1.0 - s.throttle_fraud_reduction) * fraud_cost_base
            + (1.0 - r) * s.C_fp_friction_throttle
            + s.C_throttle_fixed
        )
        cost_block = (1.0 - r) * s.C_fp_block

        # Stack and argmin
        stack = np.stack([cost_allow, cost_verify, cost_throttle, cost_block], axis=1)
        idx = np.argmin(stack, axis=1)
        action_labels = np.array(["ALLOW", "VERIFY", "THROTTLE", "BLOCK"])
        return action_labels[idx]
