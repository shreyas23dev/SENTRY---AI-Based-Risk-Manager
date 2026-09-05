"""
schema.py — Data Models for GraphRAG AI Risk Investigator & Interactive Graph
=============================================================================

Defines structured evidence items with complete provenance, graph visualization
neighborhood models, investigation reports, and question-answering contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EvidenceType(str, Enum):
    """Categorisation of retrieved risk evidence."""
    DIRECT_FRAUD = "DIRECT_FRAUD"            # Confirmed fraud on same entity
    DEVICE_SHARING = "DEVICE_SHARING"        # Multiple entities on same device
    DEVICE_FRAUD = "DEVICE_FRAUD"            # Confirmed fraud on same device
    NETWORK_SHARING = "NETWORK_SHARING"      # Shared browser / network fingerprint
    CARD_SHARING = "CARD_SHARING"            # Shared card across distinct entities
    VELOCITY_BURST = "VELOCITY_BURST"        # Transaction rate anomaly
    HOP2_CONTAMINATION = "HOP2_CONTAMINATION"# 2-hop connected entity with fraud
    COLD_START = "COLD_START"                # New entity without historical graph context
    RISK_ENGINE = "RISK_ENGINE"              # Mathematical fusion and cost decision provenance


@dataclass
class EvidenceItem:
    """
    Structured evidence record with strict provenance tracking.
    Guarantees that every factual claim is auditable back to the knowledge graph.
    """
    evidence_id: str                      # e.g. "E1", "E2", "RISK_ENGINE"
    evidence_type: EvidenceType
    title: str
    description: str
    risk_weight: float                    # [0, 1] relevance score
    source_node: str                      # e.g. "txn_3504259"
    target_node: Optional[str] = None     # e.g. "dev_SM-G950F" or "txn_3488910"
    relationship_path: List[str] = field(default_factory=list) # e.g. ["3504259", "dev_X", "3488910"]
    provenance: Dict[str, Any] = field(default_factory=dict)   # Timestamp, raw values, confirmed fraud status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type.value,
            "title": self.title,
            "description": self.description,
            "risk_weight": round(self.risk_weight, 4),
            "source_node": self.source_node,
            "target_node": self.target_node,
            "relationship_path": self.relationship_path,
            "provenance": self.provenance,
        }


@dataclass
class GraphNodeView:
    """Frontend-ready node representation for force-directed D3 visualization."""
    id: str
    label: str
    node_type: str                         # Transaction, CustomerEntity, Card, Device, Address, Email, Network
    risk_score: float = 0.0                # [0, 1]
    is_target: bool = False                # True if this is the focal transaction
    is_fraud: bool = False                 # True if confirmed historical fraud
    is_high_risk: bool = False             # True if node exhibits elevated risk
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "node_type": self.node_type,
            "risk_score": round(self.risk_score, 4),
            "is_target": self.is_target,
            "is_fraud": self.is_fraud,
            "is_high_risk": self.is_high_risk,
            "properties": self.properties,
        }


@dataclass
class GraphEdgeView:
    """Frontend-ready edge representation for force-directed D3 visualization."""
    source: str
    target: str
    edge_type: str                         # MADE_BY, USED_CARD, USED_DEVICE, SHIPPED_TO, HAS_EMAIL, etc.
    label: str = ""
    timestamp: float = 0.0
    is_suspicious: bool = False            # True if this edge lies on a fraud evidence path
    provenance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "label": self.label or self.edge_type,
            "timestamp": self.timestamp,
            "is_suspicious": self.is_suspicious,
            "provenance": self.provenance,
        }


@dataclass
class GraphNeighborhoodView:
    """Complete force-directed graph neighborhood payload."""
    transaction_id: int
    nodes: List[GraphNodeView]
    edges: List[GraphEdgeView]
    suspicious_paths: List[List[str]] = field(default_factory=list)
    max_hops: int = 2
    total_nodes: int = 0
    total_edges: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "suspicious_paths": self.suspicious_paths,
            "max_hops": self.max_hops,
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
        }


@dataclass
class GroundedReason:
    """A single reason explaining the risk decision, linked to evidence IDs."""
    statement: str
    evidence_ids: List[str]
    category: str                          # "ML_BASELINE", "GRAPH_RELATION", "MATHEMATICAL_FUSION", "COST_ENGINE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "evidence_ids": self.evidence_ids,
            "category": self.category,
        }


@dataclass
class InvestigationReport:
    """Comprehensive investigation output combining math + graph evidence + grounded explanation."""
    transaction_id: int
    timestamp: float
    amount: float
    base_risk_a: float
    graph_risk_g: float
    final_risk_r: float
    action: str                            # ALLOW, VERIFY, THROTTLE, BLOCK
    expected_cost: float
    scenario_name: str
    reasons: List[GroundedReason]
    evidence_items: List[EvidenceItem]
    graph_summary: Dict[str, Any]
    provider: str                          # "deterministic_fallback", "gemini", "groq"
    is_fallback: bool
    confidence: float
    created_at: str
    narrative_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp,
            "amount": self.amount,
            "base_risk_a": round(self.base_risk_a, 4),
            "graph_risk_g": round(self.graph_risk_g, 4),
            "final_risk_r": round(self.final_risk_r, 4),
            "action": self.action,
            "expected_cost": round(self.expected_cost, 2),
            "scenario_name": self.scenario_name,
            "reasons": [r.to_dict() for r in self.reasons],
            "evidence_items": [e.to_dict() for e in self.evidence_items],
            "graph_summary": self.graph_summary,
            "provider": self.provider,
            "is_fallback": self.is_fallback,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "narrative_summary": self.narrative_summary,
        }


@dataclass
class AskQueryResponse:
    """Response payload for analyst question-answering."""
    transaction_id: int
    question: str
    answer: str
    cited_evidence_ids: List[str]
    grounded: bool
    provider: str
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "question": self.question,
            "answer": self.answer,
            "cited_evidence_ids": self.cited_evidence_ids,
            "grounded": self.grounded,
            "provider": self.provider,
            "cached": self.cached,
        }
