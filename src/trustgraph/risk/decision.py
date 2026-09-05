"""
decision.py — Cost-Aware Intervention Decision Engine (Phase 3)
===============================================================

Wraps the FusionEngine + MerchantCostModel into a single decision interface
that produces a full, explainable risk decision for each transaction.

Output contract
---------------
Every decision returns::

    {
        "transaction_id": int,
        "base_risk":       float,    # A_t (calibrated)
        "graph_risk":      float,    # G_t (calibrated)
        "final_risk":      float,    # R_t from fusion
        "action":          str,      # ALLOW | VERIFY | THROTTLE | BLOCK
        "expected_cost":   float,    # INR expected cost of chosen action
        "action_costs":    dict,     # all four action expected costs
        "risk_contributors": list,   # human-readable explanation items
        "explanation":     str,      # formatted human-readable summary
    }

This output is the input to the future GraphRAG investigator (Phase 4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from .calibration import SignalCalibrator
from .cost_model import ActionCosts, CostScenario, MerchantCostModel
from .fusion import FusionEngine, FusionFormula, FusionResult

logger = logging.getLogger(__name__)


class Action(str, Enum):
    ALLOW    = "ALLOW"
    VERIFY   = "VERIFY"
    THROTTLE = "THROTTLE"
    BLOCK    = "BLOCK"


@dataclass
class ActionResult:
    """Full decision record for a single transaction."""
    transaction_id: int
    base_risk: float
    graph_risk: float
    final_risk: float
    action: Action
    expected_cost: float
    action_costs: Dict[str, float]
    risk_contributors: List[str]
    explanation: str
    formula: FusionFormula
    graph_contribution: float
    scenario_name: str

    def to_dict(self) -> dict:
        return {
            "transaction_id": self.transaction_id,
            "base_risk": self.base_risk,
            "graph_risk": self.graph_risk,
            "final_risk": self.final_risk,
            "action": self.action.value,
            "expected_cost": self.expected_cost,
            "action_costs": self.action_costs,
            "risk_contributors": self.risk_contributors,
            "explanation": self.explanation,
            "formula": self.formula.value,
            "graph_contribution": self.graph_contribution,
            "scenario_name": self.scenario_name,
        }


class DecisionEngine:
    """
    Combines calibration → fusion → cost-aware action selection.

    The engine is fully deterministic: same inputs → same outputs.

    Parameters
    ----------
    calibrator : SignalCalibrator
        Fitted calibrator (or identity if calibration not needed).
    fusion_engine : FusionEngine
        Tuned and frozen fusion engine (VALIDATION-selected params).
    cost_model : MerchantCostModel
        Merchant cost model with configurable parameters.

    Usage::

        engine = DecisionEngine(calibrator, fusion_engine, cost_model)
        result = engine.decide(txn_id=123, a_t=0.41, g_t=0.73, txn_amount=5000.0)
        print(result.explanation)
    """

    def __init__(
        self,
        calibrator: SignalCalibrator,
        fusion_engine: FusionEngine,
        cost_model: MerchantCostModel,
    ) -> None:
        self.calibrator = calibrator
        self.fusion_engine = fusion_engine
        self.cost_model = cost_model

    # -----------------------------------------------------------------------
    # Single-transaction decision
    # -----------------------------------------------------------------------

    def decide(
        self,
        txn_id: int,
        a_t: float,
        g_t: float,
        txn_amount: Optional[float] = None,
    ) -> ActionResult:
        """
        Produce a full risk decision for one transaction.

        Parameters
        ----------
        txn_id : int
        a_t : float — raw (uncalibrated) A_t from XGBoost
        g_t : float — raw (uncalibrated) G_t from knowledge graph
        txn_amount : float or None — transaction amount in INR
        """
        # 1. Calibrate
        a_cal, g_cal = self.calibrator.transform_single(a_t, g_t) if self.calibrator.fitted else (a_t, g_t)

        # 2. Fuse
        fusion_result: FusionResult = self.fusion_engine.fuse_single(a_cal, g_cal)
        r_t = fusion_result.final_risk

        # 3. Compute action costs
        action_costs: ActionCosts = self.cost_model.compute_action_costs(
            risk=r_t, txn_amount=txn_amount
        )
        optimal = action_costs.optimal_action()
        action = Action(optimal)
        chosen_cost = getattr(action_costs, optimal.lower())

        # 4. Build explanation
        contributors, explanation = self._explain(
            txn_id=txn_id,
            a_raw=a_t, g_raw=g_t,
            a_cal=a_cal, g_cal=g_cal,
            fusion=fusion_result,
            action=action,
            action_costs=action_costs,
            txn_amount=txn_amount or self.cost_model.scenario.avg_transaction_amount,
        )

        return ActionResult(
            transaction_id=txn_id,
            base_risk=round(a_cal, 6),
            graph_risk=round(g_cal, 6),
            final_risk=round(r_t, 6),
            action=action,
            expected_cost=round(chosen_cost, 4),
            action_costs=action_costs.as_dict(),
            risk_contributors=contributors,
            explanation=explanation,
            formula=fusion_result.formula,
            graph_contribution=round(fusion_result.graph_contribution, 6),
            scenario_name=self.cost_model.scenario.name,
        )

    # -----------------------------------------------------------------------
    # Batch decision (vectorized)
    # -----------------------------------------------------------------------

    def decide_batch(
        self,
        txn_ids: np.ndarray,
        a_t_batch: np.ndarray,
        g_t_batch: np.ndarray,
        txn_amounts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Fast vectorized batch decision. Returns array of action strings.
        Does not produce full ActionResult objects (use decide() for those).
        """
        if self.calibrator.fitted:
            a_cal, g_cal = self.calibrator.transform(a_t_batch, g_t_batch)
        else:
            a_cal, g_cal = a_t_batch.copy(), g_t_batch.copy()

        r_t = self.fusion_engine.fuse_batch(a_cal, g_cal)
        actions = self.cost_model.compute_batch_costs(r_t, txn_amounts)
        return actions

    # -----------------------------------------------------------------------
    # Explanation builder
    # -----------------------------------------------------------------------

    def _explain(
        self,
        txn_id: int,
        a_raw: float,
        g_raw: float,
        a_cal: float,
        g_cal: float,
        fusion: FusionResult,
        action: Action,
        action_costs: ActionCosts,
        txn_amount: float,
    ):
        contributors = []

        # Base ML signal
        if a_cal > 0.5:
            contributors.append(f"High ML fraud probability (A_t = {a_cal:.4f} > 0.50)")
        elif a_cal > 0.2:
            contributors.append(f"Elevated ML fraud probability (A_t = {a_cal:.4f})")
        else:
            contributors.append(f"Low ML fraud probability (A_t = {a_cal:.4f})")

        # Graph signal
        if g_cal > 0.5:
            contributors.append(f"High graph risk (G_t = {g_cal:.4f}): entity/device has confirmed fraud history")
        elif g_cal > 0.2:
            contributors.append(f"Moderate graph risk (G_t = {g_cal:.4f}): some suspicious graph context")
        elif g_cal == 0.0:
            contributors.append("No graph context (cold-start entity, G_t = 0.0)")
        else:
            contributors.append(f"Low graph risk (G_t = {g_cal:.4f})")

        # Graph contribution
        if fusion.graph_contribution > 0.05:
            contributors.append(
                f"Graph contextual uplift: +{fusion.graph_contribution:.4f} (formula: {fusion.formula.value})"
            )

        # Final risk level
        if fusion.final_risk >= 0.7:
            contributors.append(f"Final risk R_t = {fusion.final_risk:.4f} [HIGH RISK]")
        elif fusion.final_risk >= 0.4:
            contributors.append(f"Final risk R_t = {fusion.final_risk:.4f} [MEDIUM RISK]")
        else:
            contributors.append(f"Final risk R_t = {fusion.final_risk:.4f} [LOW RISK]")

        # Format explanation text
        cal_note_a = f" (calibrated from {a_raw:.4f})" if abs(a_cal - a_raw) > 0.001 else ""
        cal_note_g = f" (calibrated from {g_raw:.4f})" if abs(g_cal - g_raw) > 0.001 else ""

        explanation = (
            f"Transaction: {txn_id}\n"
            f"\n"
            f"  Base ML risk (A_t):    {a_cal:.4f}{cal_note_a}\n"
            f"  Graph risk (G_t):      {g_cal:.4f}{cal_note_g}\n"
            f"  Graph contribution:    +{fusion.graph_contribution:.4f}  [{fusion.formula.value}]\n"
            f"  Final risk (R_t):      {fusion.final_risk:.4f}\n"
            f"\n"
            f"  Merchant cost model ({self.cost_model.scenario.name}):\n"
            f"    ALLOW expected cost:    ₹{action_costs.allow:.0f}\n"
            f"    VERIFY expected cost:   ₹{action_costs.verify:.0f}\n"
            f"    THROTTLE expected cost: ₹{action_costs.throttle:.0f}\n"
            f"    BLOCK expected cost:    ₹{action_costs.block:.0f}\n"
            f"\n"
            f"  Decision: {action.value}  (min expected cost: ₹{getattr(action_costs, action.value.lower()):.0f})\n"
        )

        return contributors, explanation
