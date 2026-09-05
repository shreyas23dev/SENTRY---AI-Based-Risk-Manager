"""
engine_service.py — Clean Service Layer Coordinating Scorer and API
====================================================================

Separates API transport from the underlying ML/risk logic.
Reuses existing RuntimeScorer and PolicyThresholds without modifying them.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from trustgraph.runtime.scorer import RuntimeScorer, ScoringResult
from trustgraph.policy.config import PolicyAction
from trustgraph.service.schemas import (
    PolicyDecision,
    RiskLevel,
    SignalBreakdown,
    TransactionRiskRequest,
    TransactionRiskResponse,
)
from trustgraph.service.explanation import generate_signal_explanations
from trustgraph.service.store import TransactionStore

logger = logging.getLogger("trustgraph.service")


class RiskEngineService:
    """
    Production-grade service managing RuntimeScorer lifecycle,
    request evaluation, explanation generation, and decision caching.
    """

    def __init__(
        self,
        scorer: RuntimeScorer,
        store: Optional[TransactionStore] = None,
    ) -> None:
        self.scorer = scorer
        self.store = store or TransactionStore()
        self._lock = threading.Lock()
        logger.info("RiskEngineService initialized successfully.")

    @classmethod
    def create(cls, artifacts_root: Optional[Path] = None) -> "RiskEngineService":
        """Factory loader using project root artifacts directory."""
        if artifacts_root is None:
            artifacts_root = Path(__file__).resolve().parents[3] / "artifacts"
        logger.info("Loading RuntimeScorer from %s...", artifacts_root)
        scorer = RuntimeScorer.load(artifacts_root)
        return cls(scorer=scorer)

    def evaluate_transaction(self, request: TransactionRiskRequest) -> TransactionRiskResponse:
        """
        Evaluate risk for an incoming transaction request.

        Thread-safe execution ensuring causal consistency and state updates.
        """
        raw_dict = request.to_raw_feature_dict()

        with self._lock:
            # Score transaction using existing RuntimeScorer
            # Note: score_transaction handles missing fields safely
            result: ScoringResult = self.scorer.score_transaction(
                raw_dict=raw_dict,
                timestamp=request.transaction_dt,
                transaction_id=raw_dict.get("TransactionID"),
            )

        R_t = float(result.R_t)
        A_t = float(result.A_t)
        P_t = float(result.P_t)
        G_t = float(result.G_t)

        # Map to standard risk level and policy decision
        t = self.scorer._thresholds
        if R_t >= t.tau_block:
            decision: PolicyDecision = "BLOCK"
            risk_level: RiskLevel = "CRITICAL"
        elif R_t >= t.tau_throttle:
            decision = "THROTTLE"
            risk_level = "HIGH"
        elif R_t >= t.tau_verify:
            decision = "VERIFY"
            risk_level = "MEDIUM"
        else:
            decision = "ALLOW"
            risk_level = "LOW"

        # Generate human-readable, auditable explanations strictly from signals
        explanations = generate_signal_explanations(
            A_t=A_t,
            P_t=P_t,
            G_t=G_t,
            R_t=R_t,
            d_t=result.d_t,
            v_t=result.v_t,
            decision=decision,
            device_info=request.device_info,
            threshold_verify=t.tau_verify,
            threshold_block=t.tau_block,
        )

        # Build signals breakdown
        signals = SignalBreakdown(
            baseline_risk=round(A_t, 6),
            temporal_risk=round(P_t, 6),
            graph_risk=round(G_t, 6),
            fusion_risk=round(R_t, 6),
        )

        # Rich metadata for audit and observability
        metadata = {
            "entity_id": result.entity_id,
            "timestamp": request.transaction_dt,
            "amount": request.amount,
            "device_connected_entities": result.d_t,
            "device_recent_velocity": result.v_t,
            "normalized_degree": round(float(result.D_t), 4),
            "normalized_velocity": round(float(result.V_t), 4),
            "contextual_uplift": round(max(0.0, R_t - A_t), 6),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        response = TransactionRiskResponse(
            transaction_id=str(request.transaction_id),
            risk_score=round(R_t, 6),
            risk_level=risk_level,
            decision=decision,
            signals=signals,
            explanation=explanations,
            metadata=metadata,
        )

        # Cache in store for subsequent lookup
        self.store.save(response)
        return response

    def get_transaction(self, transaction_id: str) -> Optional[TransactionRiskResponse]:
        """Fetch previously evaluated transaction by ID."""
        return self.store.get(transaction_id)

    def get_health_status(self) -> Dict[str, Any]:
        """Inspect component readiness and parameters."""
        t = self.scorer._thresholds
        return {
            "status": "healthy",
            "engine": "TRUSTGRAPH Unified Risk Decision Engine",
            "version": "1.0.0",
            "model_readiness": {
                "baseline_model_loaded": self.scorer._model is not None,
                "preprocessor_loaded": self.scorer._fast_prep is not None,
                "temporal_engine_ready": self.scorer._temp_engine is not None,
                "relational_graph_ready": self.scorer._graph_engine is not None,
                "policy_thresholds_loaded": self.scorer._thresholds is not None,
            },
            "parameters": {
                "baseline_threshold": 0.594298,
                "policy_thresholds": {
                    "tau_verify": t.tau_verify,
                    "tau_throttle": t.tau_throttle,
                    "tau_block": t.tau_block,
                },
                "fusion_rule": "M0: clip(A_t + 1.0 * P_t + 0.05 * G_t, 0.0, 1.0)",
            },
            "stored_transactions": self.store.count(),
        }
