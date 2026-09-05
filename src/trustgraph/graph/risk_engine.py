"""
risk_engine.py — Contextual Entity Graph Risk Engine and Evidence Generator
============================================================================

Calculates deterministic graph risk G_t in [0.0, 1.0] and produces
point-in-time evidence paths for human investigation and GraphRAG retrieval.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from trustgraph.graph.schema import (
    EvidencePath,
    GraphFeatureRecord,
    TransactionEvidence,
)
from trustgraph.graph.temporal_graph import PaymentKnowledgeGraph

logger = logging.getLogger(__name__)


class EntityGraphRiskEngine:
    """
    Computes deterministic relational/graph risk G_t in [0.0, 1.0].

    G_t is formulated as an interpretable composite of:
      1. S_entity: Direct historical entity fraud momentum
      2. S_device: Device multiplexing and fraud contamination
      3. S_network: 2-hop connected network fraud contamination
      4. S_velocity: Rapid transaction bursts across entity and device
    """

    def __init__(
        self,
        w_entity: float = 0.40,
        w_device: float = 0.30,
        w_network: float = 0.15,
        w_velocity: float = 0.15,
    ) -> None:
        self.w_entity = w_entity
        self.w_device = w_device
        self.w_network = w_network
        self.w_velocity = w_velocity

    def compute_graph_risk(self, record: GraphFeatureRecord) -> float:
        """
        Compute continuous graph risk score G_t in [0.0, 1.0].
        """
        # 1. Direct Entity Historical Fraud Risk
        s_entity = 0.0
        if record.prior_entity_frauds > 0:
            s_entity = min(1.0, 0.4 * record.prior_entity_frauds + 0.6 * record.entity_fraud_rate)

        # 2. Device Multiplexing & Fraud Contamination Risk
        s_dev_fraud = 0.0
        if record.device_prior_frauds > 0:
            s_dev_fraud = min(1.0, 0.4 * record.device_prior_frauds + 0.6 * record.device_fraud_rate)

        s_dev_mult = 0.0
        if record.device_entity_count >= 2:
            s_dev_mult = min(1.0, record.device_entity_count / 5.0)

        s_device = min(1.0, 0.65 * s_dev_fraud + 0.35 * s_dev_mult)

        # 3. 2-Hop Network Contamination Risk
        s_network = 0.0
        if record.hop2_linked_frauds > 0:
            s_network = min(
                1.0,
                0.3 * record.hop2_linked_frauds + 0.35 * record.hop2_distinct_fraud_entities
            )

        # 4. Velocity Burst Risk
        s_v1h = min(1.0, max(0, record.entity_velocity_1h - 1) / 3.0)
        s_v24h = min(1.0, max(0, record.entity_velocity_24h - 2) / 6.0)
        s_dev_v24h = min(1.0, max(0, record.device_velocity_24h - 3) / 8.0)
        s_velocity = min(1.0, 0.5 * s_v1h + 0.25 * s_v24h + 0.25 * s_dev_v24h)

        # Combined Weighted Formulation
        g_raw = (
            self.w_entity * s_entity
            + self.w_device * s_device
            + self.w_network * s_network
            + self.w_velocity * s_velocity
        )

        G_t = float(np.clip(g_raw, 0.0, 1.0))
        record.graph_risk = G_t
        return G_t

    def generate_transaction_evidence(
        self,
        transaction_id: int,
        graph: PaymentKnowledgeGraph,
        record: GraphFeatureRecord,
    ) -> TransactionEvidence:
        """
        Produce auditable graph evidence for a transaction.
        Ready for human investigation and Phase 4 GraphRAG synthesis.
        """
        timestamp = record.timestamp
        entity_id = record.entity_id
        device_id = record.device_id

        # Connected entities map
        connected_ents: Dict[str, List[str]] = {
            "entity": [entity_id],
            "devices": [device_id] if device_id else [],
            "cards": list(graph.entity_cards.get(entity_id, set())),
            "addresses": list(graph.entity_addresses.get(entity_id, set())),
        }

        # Find other entities sharing the same device prior to t
        if device_id and device_id in graph.device_entities:
            other_dev_ents = [
                ent for ent in graph.device_entities[device_id]
                if ent != entity_id
            ]
            connected_ents["other_entities_sharing_device"] = other_dev_ents[:10]

        # Historical summary
        hist_summary = {
            "prior_entity_transactions": record.prior_entity_txns,
            "prior_entity_frauds": record.prior_entity_frauds,
            "entity_fraud_rate": round(record.entity_fraud_rate, 4),
            "entity_1h_velocity": record.entity_velocity_1h,
            "entity_24h_velocity": record.entity_velocity_24h,
            "device_prior_transactions": record.device_prior_txns,
            "device_prior_frauds": record.device_prior_frauds,
            "device_connected_entities": record.device_entity_count,
            "hop2_linked_fraud_count": record.hop2_linked_frauds,
        }

        # Suspicious relationships & risk factors
        suspicious_rels: List[str] = []
        risk_factors: List[str] = []
        evidence_paths: List[EvidencePath] = []

        # 1. Direct Entity Fraud Path
        if record.prior_entity_frauds > 0:
            msg = f"Customer entity has {record.prior_entity_frauds} confirmed historical fraud transactions (fraud rate: {record.entity_fraud_rate:.1%})"
            suspicious_rels.append(msg)
            risk_factors.append(f"Known fraudulent entity history ({record.prior_entity_frauds} prior frauds)")

            # Find representative prior fraud transaction ID
            prior_fraud_ids = graph.entity_fraud_txns.get(entity_id, [])
            if prior_fraud_ids:
                past_txn_id = prior_fraud_ids[-1]
                evidence_paths.append(
                    EvidencePath(
                        path_str=f"Txn {transaction_id} → Entity {entity_id} → Prior Txn {past_txn_id} → Confirmed Fraud",
                        hops=[f"txn_{transaction_id}", f"ent_{entity_id}", f"txn_{past_txn_id}", "Fraud_Status:1"],
                        risk_contribution=0.40,
                        description=f"Direct entity recidivism: entity had confirmed fraud on transaction {past_txn_id}.",
                    )
                )

        # 2. Device Multiplexing & Device Fraud Path
        if record.device_entity_count > 1:
            msg = f"Device '{device_id}' was previously associated with {record.device_entity_count} distinct entities"
            suspicious_rels.append(msg)
            risk_factors.append(f"Device multiplexing across {record.device_entity_count} customer entities")

        if record.device_prior_frauds > 0:
            msg = f"{record.device_prior_frauds} previous transactions from device '{device_id}' were confirmed fraud"
            suspicious_rels.append(msg)
            risk_factors.append(f"Device linked to prior fraud events ({record.device_prior_frauds} frauds)")

            dev_fraud_records = graph.device_fraud_txns.get(device_id, [])
            if dev_fraud_records:
                past_txn_id, other_ent = dev_fraud_records[-1]
                evidence_paths.append(
                    EvidencePath(
                        path_str=f"Txn {transaction_id} → Device {device_id} → Prior Txn {past_txn_id} (Entity {other_ent}) → Confirmed Fraud",
                        hops=[f"txn_{transaction_id}", f"dev_{device_id}", f"txn_{past_txn_id}", f"ent_{other_ent}", "Fraud_Status:1"],
                        risk_contribution=0.30,
                        description=f"Device contamination: device {device_id} was used in fraudulent transaction {past_txn_id}.",
                    )
                )

        # 3. Velocity Burst
        if record.entity_velocity_1h >= 2:
            msg = f"{record.entity_velocity_1h} transactions occurred from this customer entity in the previous hour"
            suspicious_rels.append(msg)
            risk_factors.append(f"High transaction burst velocity ({record.entity_velocity_1h} txns in 1 hour)")

        # 4. 2-Hop Network Path
        if record.hop2_linked_frauds > 0:
            msg = f"{record.hop2_linked_frauds} confirmed frauds detected across 2-hop connected entity neighborhood"
            suspicious_rels.append(msg)
            risk_factors.append(f"2-hop network contamination ({record.hop2_distinct_fraud_entities} linked entities with fraud)")

        if not risk_factors:
            risk_factors.append("No suspicious relational patterns or prior fraud history detected in payment graph.")

        return TransactionEvidence(
            transaction_id=transaction_id,
            timestamp=timestamp,
            graph_risk=record.graph_risk,
            entity_id=entity_id,
            connected_entities=connected_ents,
            historical_summary=hist_summary,
            suspicious_relationships=suspicious_rels,
            evidence_paths=evidence_paths,
            risk_factors=risk_factors,
        )
