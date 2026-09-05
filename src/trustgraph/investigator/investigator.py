"""
investigator.py — Grounded GraphRAG AI Risk Investigator Engine (Phase 4)
========================================================================

Orchestrates:
  1. Deterministic evidence retrieval from PaymentKnowledgeGraph
  2. Mathematical risk provenance binding (Phase 3)
  3. Grounded explanation synthesis via LLMProvider
  4. Strict anti-hallucination verification
  5. Multi-level caching (transaction, evidence version, question)
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from trustgraph.graph.temporal_graph import PaymentKnowledgeGraph
from trustgraph.investigator.llm_provider import LLMProvider, get_llm_provider
from trustgraph.investigator.retriever import EvidenceRetriever
from trustgraph.investigator.schema import (
    AskQueryResponse,
    EvidenceItem,
    GraphNeighborhoodView,
    GroundedReason,
    InvestigationReport,
)

logger = logging.getLogger(__name__)


class GraphRAGInvestigator:
    """
    Grounded AI Risk Investigator combining deterministic knowledge graph retrieval
    with an anti-hallucination language model layer.
    """

    def __init__(
        self,
        graph: PaymentKnowledgeGraph,
        llm_provider: Optional[LLMProvider] = None,
        max_hops: int = 2,
    ) -> None:
        self.graph = graph
        self.retriever = EvidenceRetriever(graph)
        self.llm = llm_provider or get_llm_provider()
        self.max_hops = max_hops

        # In-memory LRU caches
        self._report_cache: Dict[str, InvestigationReport] = {}
        self._ask_cache: Dict[str, AskQueryResponse] = {}

    def _evidence_hash(self, evidence_items: List[EvidenceItem]) -> str:
        """Compute stable hash of retrieved evidence items."""
        keys = sorted([f"{e.evidence_id}:{e.evidence_type.value}:{e.risk_weight}" for e in evidence_items])
        return hashlib.sha256(";".join(keys).encode("utf-8")).hexdigest()[:16]

    def investigate_transaction(
        self,
        transaction_id: int,
        transaction_dict: Optional[Dict[str, Any]] = None,
        base_risk: float = 0.0,
        graph_risk: float = 0.0,
        final_risk: float = 0.0,
        action: str = "ALLOW",
        expected_cost: float = 0.0,
        beta: float = 0.05,
        scenario_name: str = "balanced",
        force_refresh: bool = False,
    ) -> Tuple[InvestigationReport, GraphNeighborhoodView]:
        """
        Produce a full grounded investigation report and graph visualization view.
        """
        # 1. Deterministic Evidence Retrieval from Knowledge Graph
        evidence_items, neighborhood = self.retriever.retrieve_evidence(
            transaction_id=transaction_id,
            transaction_dict=transaction_dict,
            base_risk=base_risk,
            graph_risk=graph_risk,
            final_risk=final_risk,
            action=action,
            expected_cost=expected_cost,
            beta=beta,
            max_hops=self.max_hops,
        )

        # Cache check (incorporates risk inputs to avoid stale hits across different risk values)
        ev_hash = self._evidence_hash(evidence_items)
        cache_key = f"{transaction_id}_{base_risk:.4f}_{graph_risk:.4f}_{final_risk:.4f}_{action}_{scenario_name}_{ev_hash}"
        if not force_refresh and cache_key in self._report_cache:
            logger.debug("Returning cached investigation report for transaction %d.", transaction_id)
            return self._report_cache[cache_key], neighborhood

        # 2. Generate Grounded Explanation via LLM Provider
        reasons, summary_text = self.llm.generate_investigation(
            transaction_id=transaction_id,
            base_risk_a=base_risk,
            graph_risk_g=graph_risk,
            final_risk_r=final_risk,
            action=action,
            expected_cost=expected_cost,
            evidence_items=evidence_items,
        )

        # 3. Post-Generation Grounding Verification
        valid_evidence_ids = {e.evidence_id for e in evidence_items}
        verified_reasons: List[GroundedReason] = []
        for r in reasons:
            # Filter cited evidence IDs to strictly existing ones
            clean_ids = [eid for eid in r.evidence_ids if eid in valid_evidence_ids]
            if not clean_ids:
                clean_ids = ["RISK_ENGINE"]
            verified_reasons.append(GroundedReason(
                statement=r.statement,
                evidence_ids=clean_ids,
                category=r.category,
            ))

        # 4. Assemble Graph Summary for Dashboard
        graph_summary = {
            "total_nodes": len(neighborhood.nodes),
            "total_edges": len(neighborhood.edges),
            "suspicious_paths_count": len(neighborhood.suspicious_paths),
            "max_hops": neighborhood.max_hops,
            "has_fraud_links": any(n.is_fraud for n in neighborhood.nodes),
            "high_risk_nodes_count": sum(1 for n in neighborhood.nodes if n.is_high_risk),
        }

        # 5. Build Final Report
        report = InvestigationReport(
            transaction_id=transaction_id,
            timestamp=float(transaction_dict.get("TransactionDT", 0.0)) if transaction_dict else 0.0,
            amount=float(transaction_dict.get("TransactionAmt", 0.0)) if transaction_dict else 0.0,
            base_risk_a=base_risk,
            graph_risk_g=graph_risk,
            final_risk_r=final_risk,
            action=action,
            expected_cost=expected_cost,
            scenario_name=scenario_name,
            reasons=verified_reasons,
            evidence_items=evidence_items,
            graph_summary=graph_summary,
            provider=self.llm.name,
            is_fallback=self.llm.is_fallback,
            confidence=round(0.95 if any(e.risk_weight >= 0.8 for e in evidence_items) else 0.85, 2),
            created_at=datetime.now(timezone.utc).isoformat(),
            narrative_summary=summary_text,
        )

        self._report_cache[cache_key] = report
        return report, neighborhood

    def ask_question(
        self,
        transaction_id: int,
        question: str,
        transaction_dict: Optional[Dict[str, Any]] = None,
        base_risk: float = 0.0,
        graph_risk: float = 0.0,
        final_risk: float = 0.0,
        action: str = "ALLOW",
        expected_cost: float = 0.0,
        beta: float = 0.05,
    ) -> AskQueryResponse:
        """
        Answer an analyst question grounded in the retrieved graph evidence.
        """
        clean_q = question.strip()
        if not clean_q:
            return AskQueryResponse(
                transaction_id=transaction_id,
                question="",
                answer="Please enter a valid investigation question.",
                cited_evidence_ids=[],
                grounded=False,
                provider=self.llm.name,
            )

        # Retrieve evidence
        evidence_items, _ = self.retriever.retrieve_evidence(
            transaction_id=transaction_id,
            transaction_dict=transaction_dict,
            base_risk=base_risk,
            graph_risk=graph_risk,
            final_risk=final_risk,
            action=action,
            expected_cost=expected_cost,
            beta=beta,
            max_hops=self.max_hops,
        )

        ev_hash = self._evidence_hash(evidence_items)
        q_hash = hashlib.md5(clean_q.lower().encode("utf-8")).hexdigest()[:12]
        cache_key = f"{transaction_id}_{ev_hash}_{q_hash}"

        if cache_key in self._ask_cache:
            res = self._ask_cache[cache_key]
            return AskQueryResponse(
                transaction_id=res.transaction_id,
                question=res.question,
                answer=res.answer,
                cited_evidence_ids=res.cited_evidence_ids,
                grounded=res.grounded,
                provider=res.provider,
                cached=True,
            )

        # Call LLM
        answer, cited_ids, is_grounded = self.llm.answer_question(
            transaction_id=transaction_id,
            base_risk_a=base_risk,
            graph_risk_g=graph_risk,
            final_risk_r=final_risk,
            action=action,
            expected_cost=expected_cost,
            evidence_items=evidence_items,
            question=clean_q,
        )

        # Grounding validation: ensure cited IDs are valid
        valid_ids = {e.evidence_id for e in evidence_items} | {"RISK_ENGINE", "ENGINE", "COST_MODEL"}
        valid_cited = [cid for cid in cited_ids if cid in valid_ids]

        response = AskQueryResponse(
            transaction_id=transaction_id,
            question=clean_q,
            answer=answer,
            cited_evidence_ids=valid_cited,
            grounded=is_grounded and (len(valid_cited) > 0 or "insufficient" in answer.lower()),
            provider=self.llm.name,
            cached=False,
        )

        self._ask_cache[cache_key] = response
        return response
