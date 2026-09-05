"""
schema.py — Payment Knowledge Graph Schema and Data Models
===========================================================

Defines typed nodes, edges, contextual feature records, and evidence paths
for the point-in-time payment knowledge graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class NodeType(str, Enum):
    """Real entity types supported directly by the IEEE-CIS payment schema."""
    TRANSACTION = "Transaction"
    CUSTOMER_ENTITY = "CustomerEntity"  # Composite proxy (card1 + addr1 + P_emaildomain / UID)
    CARD = "Card"                      # Card identifier (card1, card2, card3, card5)
    DEVICE = "Device"                  # Device identifier (DeviceInfo)
    ADDRESS = "Address"                # Billing/shipping location (addr1, addr2)
    EMAIL = "Email"                    # Purchaser email domain (P_emaildomain)
    MERCHANT = "Merchant"              # Recipient email domain or ProductCD
    NETWORK = "Network"                # Browser / OS agent (id_30, id_31)


class EdgeType(str, Enum):
    """Causal payment relationship types."""
    MADE_BY = "MADE_BY"                # Transaction -> CustomerEntity
    USED_CARD = "USED_CARD"            # Transaction -> Card
    USED_DEVICE = "USED_DEVICE"        # Transaction -> Device
    SHIPPED_TO = "SHIPPED_TO"          # Transaction -> Address
    HAS_EMAIL = "HAS_EMAIL"            # Transaction -> Email
    SENT_TO = "SENT_TO"                # Transaction -> Merchant
    ACCESSED_VIA = "ACCESSED_VIA"      # Transaction -> Network


@dataclass(slots=True)
class GraphNode:
    """A node in the payment knowledge graph."""
    node_id: str
    node_type: NodeType
    created_at: float
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphEdge:
    """A directed, timestamped edge in the payment knowledge graph."""
    src_id: str
    dst_id: str
    edge_type: EdgeType
    timestamp: float
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphFeatureRecord:
    """
    Point-in-time graph feature record for a single transaction.
    Every feature is computed strictly from events occurring prior to transaction timestamp t.
    """
    transaction_id: int
    timestamp: float
    entity_id: str

    # Direct Entity Historical Metrics (prior to t)
    prior_entity_txns: int = 0
    prior_entity_frauds: int = 0
    entity_fraud_rate: float = 0.0
    entity_velocity_1h: int = 0
    entity_velocity_24h: int = 0

    # Device Multiplexing & Fraud History (prior to t)
    device_id: Optional[str] = None
    device_entity_count: int = 0
    device_prior_txns: int = 0
    device_prior_frauds: int = 0
    device_fraud_rate: float = 0.0
    device_velocity_24h: int = 0
    unusual_device_sharing: int = 0  # 1 if >= 3 entities on same device

    # Card & Address Sharing (prior to t)
    card_entity_count: int = 0
    address_entity_count: int = 0
    network_entity_count: int = 0

    # 2-Hop Network Contamination (prior to t)
    hop2_linked_frauds: int = 0
    hop2_distinct_fraud_entities: int = 0

    # Context & Cold Start Flags
    has_graph_context: int = 0  # 1 if entity/device had prior history, 0 for cold-start

    # Deterministic Graph Risk Output
    graph_risk: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialisation or dataframe creation."""
        return {
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp,
            "entity_id": self.entity_id,
            "prior_entity_txns": self.prior_entity_txns,
            "prior_entity_frauds": self.prior_entity_frauds,
            "entity_fraud_rate": round(self.entity_fraud_rate, 4),
            "entity_velocity_1h": self.entity_velocity_1h,
            "entity_velocity_24h": self.entity_velocity_24h,
            "device_id": self.device_id,
            "device_entity_count": self.device_entity_count,
            "device_prior_txns": self.device_prior_txns,
            "device_prior_frauds": self.device_prior_frauds,
            "device_fraud_rate": round(self.device_fraud_rate, 4),
            "device_velocity_24h": self.device_velocity_24h,
            "unusual_device_sharing": self.unusual_device_sharing,
            "card_entity_count": self.card_entity_count,
            "address_entity_count": self.address_entity_count,
            "network_entity_count": self.network_entity_count,
            "hop2_linked_frauds": self.hop2_linked_frauds,
            "hop2_distinct_fraud_entities": self.hop2_distinct_fraud_entities,
            "has_graph_context": self.has_graph_context,
            "graph_risk": round(self.graph_risk, 6),
        }


@dataclass
class EvidencePath:
    """An auditable evidence path connecting a transaction to historical risk factors."""
    path_str: str              # e.g. "Txn 3489000 -> Device SM-G950F -> Txn 3488500 -> Confirmed Fraud"
    hops: List[str]            # Node sequence
    risk_contribution: float   # Weight of this path in overall risk assessment
    description: str           # Human-readable explanation


@dataclass
class TransactionEvidence:
    """
    Comprehensive queryable graph intelligence output for a transaction.
    Designed for human investigation and Phase 4 GraphRAG retrieval.
    """
    transaction_id: int
    timestamp: float
    graph_risk: float
    entity_id: str
    connected_entities: Dict[str, List[str]]
    historical_summary: Dict[str, Any]
    suspicious_relationships: List[str]
    evidence_paths: List[EvidencePath]
    risk_factors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp,
            "graph_risk": round(self.graph_risk, 6),
            "entity_id": self.entity_id,
            "connected_entities": self.connected_entities,
            "historical_summary": self.historical_summary,
            "suspicious_relationships": self.suspicious_relationships,
            "evidence_paths": [
                {
                    "path": p.path_str,
                    "risk_contribution": round(p.risk_contribution, 4),
                    "description": p.description,
                }
                for p in self.evidence_paths
            ],
            "risk_factors": self.risk_factors,
        }
