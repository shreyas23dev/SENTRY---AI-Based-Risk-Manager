"""
retriever.py — Deterministic Knowledge Graph Evidence Retriever (Phase 4)
=========================================================================

Traverses the actual Phase 2 PaymentKnowledgeGraph to extract a bounded 1-2 hop
causal neighborhood for a given transaction. Produces ranked, provenance-backed
EvidenceItem objects and a GraphNeighborhoodView for force-directed visualization.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from trustgraph.graph.schema import EdgeType, NodeType
from trustgraph.graph.temporal_graph import MAX_RING_DEGREE, PaymentKnowledgeGraph
from trustgraph.investigator.schema import (
    EvidenceItem,
    EvidenceType,
    GraphEdgeView,
    GraphNeighborhoodView,
    GraphNodeView,
)

logger = logging.getLogger(__name__)


class EvidenceRetriever:
    """
    Deterministic retriever extracting 1-hop and 2-hop risk evidence
    from the production PaymentKnowledgeGraph.
    """

    def __init__(self, graph: PaymentKnowledgeGraph, max_neighborhood_nodes: int = 50) -> None:
        self.graph = graph
        self.max_nodes = max_neighborhood_nodes

    def retrieve_evidence(
        self,
        transaction_id: int,
        transaction_dict: Optional[Dict[str, Any]] = None,
        base_risk: float = 0.0,
        graph_risk: float = 0.0,
        final_risk: float = 0.0,
        action: str = "ALLOW",
        expected_cost: float = 0.0,
        beta: float = 0.05,
        max_hops: int = 2,
    ) -> Tuple[List[EvidenceItem], GraphNeighborhoodView]:
        """
        Extract ranked evidence items and graph neighborhood view for a transaction.
        """
        # Resolve transaction metadata
        txn_meta = self.graph.transactions.get(transaction_id)
        if txn_meta is None and transaction_dict is not None:
            txn_meta = transaction_dict

        if txn_meta is None:
            # Fallback for unknown transaction: empty graph, cold-start evidence
            logger.warning("Transaction %d not found in Knowledge Graph index.", transaction_id)
            return self._build_empty_evidence(transaction_id, base_risk, graph_risk, final_risk, action, expected_cost, beta)

        timestamp = float(txn_meta.get("timestamp", txn_meta.get("TransactionDT", 0.0)))
        entity_id = str(txn_meta.get("entity_id", txn_meta.get("card1", "unknown")))
        device_id = txn_meta.get("device_id", txn_meta.get("DeviceInfo"))
        if device_id and str(device_id).lower() in ("nan", "none", ""):
            device_id = None
        device_id = str(device_id) if device_id else None

        card_id = str(txn_meta.get("card_id", txn_meta.get("card1", "")))
        addr_id = txn_meta.get("addr_id", txn_meta.get("addr1"))
        addr_id = str(addr_id) if addr_id and str(addr_id).lower() not in ("nan", "none", "") else None
        email_id = txn_meta.get("email_id", txn_meta.get("P_emaildomain"))
        email_id = str(email_id) if email_id and str(email_id).lower() not in ("nan", "none", "") else None
        net_id = txn_meta.get("network_id", txn_meta.get("id_31"))
        net_id = str(net_id) if net_id and str(net_id).lower() not in ("nan", "none", "") else None
        amt = float(txn_meta.get("amount", txn_meta.get("TransactionAmt", 0.0)))

        # -------------------------------------------------------------------
        # Build Graph Nodes & Edges (1-Hop & 2-Hop)
        # -------------------------------------------------------------------
        nodes: Dict[str, GraphNodeView] = {}
        edges: List[GraphEdgeView] = []
        suspicious_paths: List[List[str]] = []

        # Target Transaction Node
        target_node_id = f"txn_{transaction_id}"
        nodes[target_node_id] = GraphNodeView(
            id=target_node_id,
            label=f"Txn #{transaction_id}",
            node_type=NodeType.TRANSACTION.value,
            risk_score=final_risk,
            is_target=True,
            is_fraud=False,
            is_high_risk=final_risk >= 0.5,
            properties={"amount": amt, "timestamp": timestamp, "decision": action},
        )

        # 1-Hop: Customer Entity
        ent_node_id = f"ent_{entity_id}"
        prior_txns, prior_frauds, v1h, v24h = self.graph.get_prior_entity_history(entity_id, timestamp)
        entity_is_fraud = prior_frauds > 0
        nodes[ent_node_id] = GraphNodeView(
            id=ent_node_id,
            label=f"Entity {entity_id[:16]}",
            node_type=NodeType.CUSTOMER_ENTITY.value,
            risk_score=min(1.0, prior_frauds * 0.4 + (prior_frauds / max(prior_txns, 1)) * 0.6),
            is_fraud=entity_is_fraud,
            is_high_risk=prior_frauds > 0 or prior_txns > 10,
            properties={
                "prior_transactions": prior_txns,
                "prior_frauds": prior_frauds,
                "fraud_rate": round(prior_frauds / max(prior_txns, 1), 4),
                "velocity_24h": v24h,
            },
        )
        edges.append(GraphEdgeView(
            source=target_node_id,
            target=ent_node_id,
            edge_type=EdgeType.MADE_BY.value,
            label="MADE_BY",
            timestamp=timestamp,
            is_suspicious=entity_is_fraud,
            provenance=f"Associated via card1+addr1 entity proxy at timestamp {timestamp}",
        ))

        # 1-Hop: Card
        if card_id:
            card_node_id = f"card_{card_id}"
            card_entities = self.graph.card_entities.get(card_id, set())
            nodes[card_node_id] = GraphNodeView(
                id=card_node_id,
                label=f"Card {card_id}",
                node_type=NodeType.CARD.value,
                properties={"distinct_entities": len(card_entities)},
            )
            edges.append(GraphEdgeView(
                source=target_node_id,
                target=card_node_id,
                edge_type=EdgeType.USED_CARD.value,
                label="USED_CARD",
                timestamp=timestamp,
                provenance=f"Card {card_id} presented during payment",
            ))

        # 1-Hop: Device (if available)
        dev_node_id = None
        dev_entities: Set[str] = set()
        dev_prior_frauds = 0
        if device_id:
            dev_node_id = f"dev_{device_id}"
            distinct_ents, dev_txns, dev_prior_frauds, dev_v24 = self.graph.get_prior_device_history(
                device_id, timestamp, current_entity=entity_id
            )
            dev_entities = self.graph.device_entities.get(device_id, set())
            dev_is_fraud = dev_prior_frauds > 0
            nodes[dev_node_id] = GraphNodeView(
                id=dev_node_id,
                label=f"Device {device_id[:16]}",
                node_type=NodeType.DEVICE.value,
                risk_score=min(1.0, dev_prior_frauds * 0.4 + (distinct_ents / 5.0) * 0.35),
                is_fraud=dev_is_fraud,
                is_high_risk=dev_prior_frauds > 0 or distinct_ents >= 3,
                properties={
                    "prior_transactions": dev_txns,
                    "prior_frauds": dev_prior_frauds,
                    "distinct_entities": distinct_ents,
                },
            )
            edges.append(GraphEdgeView(
                source=target_node_id,
                target=dev_node_id,
                edge_type=EdgeType.USED_DEVICE.value,
                label="USED_DEVICE",
                timestamp=timestamp,
                is_suspicious=dev_is_fraud or distinct_ents >= 3,
                provenance=f"Client DeviceInfo: {device_id}",
            ))

        # 1-Hop: Address
        if addr_id:
            addr_node_id = f"addr_{addr_id}"
            nodes[addr_node_id] = GraphNodeView(
                id=addr_node_id,
                label=f"Addr {addr_id}",
                node_type=NodeType.ADDRESS.value,
                properties={"addr1": addr_id},
            )
            edges.append(GraphEdgeView(
                source=target_node_id,
                target=addr_node_id,
                edge_type=EdgeType.SHIPPED_TO.value,
                label="SHIPPED_TO",
                timestamp=timestamp,
                provenance=f"Billing/shipping postal code: {addr_id}",
            ))

        # 1-Hop: Email
        if email_id:
            email_node_id = f"email_{email_id}"
            nodes[email_node_id] = GraphNodeView(
                id=email_node_id,
                label=f"Email {email_id}",
                node_type=NodeType.EMAIL.value,
                properties={"domain": email_id},
            )
            edges.append(GraphEdgeView(
                source=target_node_id,
                target=email_node_id,
                edge_type=EdgeType.HAS_EMAIL.value,
                label="HAS_EMAIL",
                timestamp=timestamp,
                provenance=f"Purchaser email domain: {email_id}",
            ))

        # 1-Hop: Network / Browser
        if net_id:
            net_node_id = f"net_{net_id}"
            nodes[net_node_id] = GraphNodeView(
                id=net_node_id,
                label=f"Net {net_id[:16]}",
                node_type=NodeType.NETWORK.value,
                properties={"browser_os": net_id},
            )
            edges.append(GraphEdgeView(
                source=target_node_id,
                target=net_node_id,
                edge_type=EdgeType.ACCESSED_VIA.value,
                label="ACCESSED_VIA",
                timestamp=timestamp,
                provenance=f"Browser user agent id_31: {net_id}",
            ))

        # -------------------------------------------------------------------
        # 2-Hop Traversal (if max_hops >= 2)
        # -------------------------------------------------------------------
        evidence_items: List[EvidenceItem] = []
        evidence_counter = 1

        if max_hops >= 2:
            # 2-Hop A: Historical Fraud Transactions of this Entity
            past_entity_frauds = self.graph.entity_fraud_txns.get(entity_id, [])
            for past_fid in past_entity_frauds[-3:]:  # limit to top 3
                if len(nodes) >= self.max_nodes:
                    break
                p_node_id = f"past_fraud_{past_fid}"
                nodes[p_node_id] = GraphNodeView(
                    id=p_node_id,
                    label=f"Fraud Txn #{past_fid}",
                    node_type=NodeType.TRANSACTION.value,
                    risk_score=1.0,
                    is_fraud=True,
                    is_high_risk=True,
                    properties={"isFraud": 1, "prior_confirmed": True},
                )
                edges.append(GraphEdgeView(
                    source=ent_node_id,
                    target=p_node_id,
                    edge_type="PRIOR_FRAUD_BY",
                    label="PRIOR_FRAUD",
                    timestamp=0.0,
                    is_suspicious=True,
                    provenance=f"Historical confirmed fraud event #{past_fid} by Entity {entity_id}",
                ))
                path = [target_node_id, ent_node_id, p_node_id]
                suspicious_paths.append(path)

                evidence_items.append(EvidenceItem(
                    evidence_id=f"E{evidence_counter}",
                    evidence_type=EvidenceType.DIRECT_FRAUD,
                    title="Confirmed Recidivist Entity Fraud History",
                    description=(
                        f"Customer Entity '{entity_id}' has {prior_frauds} confirmed prior fraudulent transactions "
                        f"(e.g. Transaction #{past_fid}). Historical entity fraud rate is {prior_frauds/max(prior_txns, 1):.1%}."
                    ),
                    risk_weight=0.95,
                    source_node=target_node_id,
                    target_node=p_node_id,
                    relationship_path=path,
                    provenance={
                        "prior_fraud_txn_id": past_fid,
                        "entity_id": entity_id,
                        "total_prior_txns": prior_txns,
                        "total_prior_frauds": prior_frauds,
                    },
                ))
                evidence_counter += 1

            # 2-Hop B: Device Multiplexing & Fraud Connections
            if device_id and dev_node_id:
                # Prior fraud on device
                past_dev_frauds = self.graph.device_fraud_txns.get(device_id, [])
                for past_fid, other_ent in past_dev_frauds[-3:]:
                    if len(nodes) >= self.max_nodes:
                        break
                    d_fraud_node = f"dev_fraud_{past_fid}"
                    nodes[d_fraud_node] = GraphNodeView(
                        id=d_fraud_node,
                        label=f"Dev Fraud #{past_fid}",
                        node_type=NodeType.TRANSACTION.value,
                        risk_score=1.0,
                        is_fraud=True,
                        is_high_risk=True,
                        properties={"isFraud": 1, "origin_entity": other_ent},
                    )
                    edges.append(GraphEdgeView(
                        source=dev_node_id,
                        target=d_fraud_node,
                        edge_type="DEVICE_FRAUD_LINK",
                        label="FRAUD_ON_DEVICE",
                        timestamp=0.0,
                        is_suspicious=True,
                        provenance=f"Device {device_id} was previously used in fraud transaction #{past_fid}",
                    ))
                    path = [target_node_id, dev_node_id, d_fraud_node]
                    suspicious_paths.append(path)

                    evidence_items.append(EvidenceItem(
                        evidence_id=f"E{evidence_counter}",
                        evidence_type=EvidenceType.DEVICE_FRAUD,
                        title="Contaminated Device Fraud Link",
                        description=(
                            f"Device '{device_id}' is directly linked to historical fraudulent transaction #{past_fid} "
                            f"(conducted by Entity '{other_ent}'). Total device fraud count is {dev_prior_frauds}."
                        ),
                        risk_weight=0.90,
                        source_node=target_node_id,
                        target_node=d_fraud_node,
                        relationship_path=path,
                        provenance={
                            "device_id": device_id,
                            "fraud_txn_id": past_fid,
                            "other_entity": other_ent,
                            "device_fraud_count": dev_prior_frauds,
                        },
                    ))
                    evidence_counter += 1

                # Device multiplexing (other entities sharing device)
                other_dev_entities = [e for e in dev_entities if e != entity_id]
                if len(other_dev_entities) >= 2:
                    for other_e in other_dev_entities[:4]:
                        if len(nodes) >= self.max_nodes:
                            break
                        oe_node_id = f"other_ent_{other_e[:12]}"
                        o_txns, o_frauds, _, _ = self.graph.get_prior_entity_history(other_e, timestamp)
                        nodes[oe_node_id] = GraphNodeView(
                            id=oe_node_id,
                            label=f"Co-Entity {other_e[:12]}",
                            node_type=NodeType.CUSTOMER_ENTITY.value,
                            risk_score=0.7 if o_frauds > 0 else 0.3,
                            is_fraud=o_frauds > 0,
                            is_high_risk=o_frauds > 0,
                            properties={"prior_txns": o_txns, "prior_frauds": o_frauds},
                        )
                        edges.append(GraphEdgeView(
                            source=dev_node_id,
                            target=oe_node_id,
                            edge_type="SHARED_DEVICE_BY",
                            label="SHARED_DEVICE",
                            timestamp=0.0,
                            is_suspicious=o_frauds > 0,
                            provenance=f"Entity {other_e} accessed via same device {device_id}",
                        ))
                    evidence_items.append(EvidenceItem(
                        evidence_id=f"E{evidence_counter}",
                        evidence_type=EvidenceType.DEVICE_SHARING,
                        title="High-Degree Device Multiplexing Anomaly",
                        description=(
                            f"Device '{device_id}' has been shared across {len(dev_entities)} distinct customer entities "
                            f"prior to this transaction, indicating shared hardware activity across multiple accounts."
                        ),
                        risk_weight=0.75,
                        source_node=target_node_id,
                        target_node=dev_node_id,
                        relationship_path=[target_node_id, dev_node_id],
                        provenance={"device_id": device_id, "distinct_entities_count": len(dev_entities)},
                    ))
                    evidence_counter += 1

            # 2-Hop C: 2-Hop Network Contamination
            hop2_frauds, hop2_distinct_ents = self.graph.get_prior_2hop_fraud_history(
                entity_id, device_id, card_id
            )
            if hop2_frauds > 0 and not any(e.evidence_type == EvidenceType.DIRECT_FRAUD for e in evidence_items):
                evidence_items.append(EvidenceItem(
                    evidence_id=f"E{evidence_counter}",
                    evidence_type=EvidenceType.HOP2_CONTAMINATION,
                    title="2-Hop Network Fraud Contamination",
                    description=(
                        f"Target entity shares infrastructure (card or device) with {hop2_distinct_ents} distinct "
                        f"entities that have {hop2_frauds} confirmed fraud cases on record."
                    ),
                    risk_weight=0.70,
                    source_node=target_node_id,
                    target_node=ent_node_id,
                    relationship_path=[target_node_id, ent_node_id],
                    provenance={"hop2_frauds": hop2_frauds, "hop2_entities": hop2_distinct_ents},
                ))
                evidence_counter += 1

        # -------------------------------------------------------------------
        # Velocity & Activity Evidence
        # -------------------------------------------------------------------
        if v1h >= 2 or v24h >= 4:
            evidence_items.append(EvidenceItem(
                evidence_id=f"E{evidence_counter}",
                evidence_type=EvidenceType.VELOCITY_BURST,
                title="Rapid Velocity Burst Detected",
                description=(
                    f"Entity '{entity_id}' executed {v1h} transactions in the preceding 1 hour and {v24h} "
                    f"transactions in the preceding 24 hours, indicating high-frequency payment activity."
                ),
                risk_weight=0.60,
                source_node=target_node_id,
                target_node=ent_node_id,
                relationship_path=[target_node_id, ent_node_id],
                provenance={"velocity_1h": v1h, "velocity_24h": v24h},
            ))
            evidence_counter += 1

        # If zero historical signals found, record clean history or cold-start
        if prior_txns == 0:
            evidence_items.append(EvidenceItem(
                evidence_id=f"E{evidence_counter}",
                evidence_type=EvidenceType.COLD_START,
                title="Cold-Start Customer Entity",
                description=f"First time observing entity '{entity_id}' in the knowledge graph. Zero prior historical record.",
                risk_weight=0.10,
                source_node=target_node_id,
                target_node=ent_node_id,
                relationship_path=[target_node_id, ent_node_id],
                provenance={"prior_txns": 0, "prior_frauds": 0},
            ))
            evidence_counter += 1
        elif prior_frauds == 0 and dev_prior_frauds == 0 and not any(e.evidence_type == EvidenceType.HOP2_CONTAMINATION for e in evidence_items):
            evidence_items.append(EvidenceItem(
                evidence_id=f"E{evidence_counter}",
                evidence_type=EvidenceType.CARD_SHARING,
                title="Clean Historical Payment Record",
                description=(
                    f"Entity '{entity_id}' has {prior_txns} previous successful transactions with zero reported frauds. "
                    f"No device contamination observed."
                ),
                risk_weight=0.05,
                source_node=target_node_id,
                target_node=ent_node_id,
                relationship_path=[target_node_id, ent_node_id],
                provenance={"prior_txns": prior_txns, "prior_frauds": 0},
            ))
            evidence_counter += 1

        # -------------------------------------------------------------------
        # Mathematical Engine Provenance Evidence (always present)
        # -------------------------------------------------------------------
        uplift = round(final_risk - base_risk, 4)
        evidence_items.append(EvidenceItem(
            evidence_id="RISK_ENGINE",
            evidence_type=EvidenceType.RISK_ENGINE,
            title="Phase 3 Mathematical Risk Engine & Cost Decision",
            description=(
                f"Base ML probability A_t = {base_risk:.4f} (XGBoost). "
                f"Calibrated Graph Risk G_t = {graph_risk:.4f}. "
                f"F2 Conditional Fusion (beta = {beta}) assigned Final Risk R_t = {final_risk:.4f} (uplift: {uplift:+.4f}). "
                f"Expected cost minimisation selected decision '{action}' with expected loss of INR {expected_cost:.2f}."
            ),
            risk_weight=0.85,
            source_node=target_node_id,
            target_node=None,
            relationship_path=[target_node_id],
            provenance={
                "A_t": base_risk,
                "G_t": graph_risk,
                "R_t": final_risk,
                "formula": "F2_conditional",
                "beta": beta,
                "action": action,
                "expected_cost": expected_cost,
            },
        ))

        # Rank evidence items by risk_weight descending
        evidence_items.sort(key=lambda e: e.risk_weight, reverse=True)

        neighborhood = GraphNeighborhoodView(
            transaction_id=transaction_id,
            nodes=list(nodes.values()),
            edges=edges,
            suspicious_paths=suspicious_paths,
            max_hops=max_hops,
        )

        return evidence_items, neighborhood

    def _build_empty_evidence(
        self,
        transaction_id: int,
        base_risk: float,
        graph_risk: float,
        final_risk: float,
        action: str,
        expected_cost: float,
        beta: float,
    ) -> Tuple[List[EvidenceItem], GraphNeighborhoodView]:
        """Fallback when transaction is completely unknown to graph index."""
        target_node_id = f"txn_{transaction_id}"
        nodes = [GraphNodeView(
            id=target_node_id,
            label=f"Txn #{transaction_id}",
            node_type=NodeType.TRANSACTION.value,
            risk_score=final_risk,
            is_target=True,
        )]
        evidence = [
            EvidenceItem(
                evidence_id="E1",
                evidence_type=EvidenceType.COLD_START,
                title="Unknown / Cold-Start Transaction",
                description=f"Transaction #{transaction_id} has no historical context registered in the knowledge graph.",
                risk_weight=0.10,
                source_node=target_node_id,
            ),
            EvidenceItem(
                evidence_id="RISK_ENGINE",
                evidence_type=EvidenceType.RISK_ENGINE,
                title="Phase 3 Mathematical Risk Engine",
                description=f"A_t={base_risk:.4f}, G_t={graph_risk:.4f} -> R_t={final_risk:.4f}. Action: {action}.",
                risk_weight=0.85,
                source_node=target_node_id,
                provenance={"A_t": base_risk, "G_t": graph_risk, "R_t": final_risk, "action": action},
            ),
        ]
        return evidence, GraphNeighborhoodView(transaction_id=transaction_id, nodes=nodes, edges=[])
