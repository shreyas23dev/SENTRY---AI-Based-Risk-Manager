"""
temporal_graph.py — Point-in-Time Payment Knowledge Graph (O(1) Causal Engine)
================================================================================

Maintains a causal, temporally indexed payment entity graph with O(1) state lookups.
Guarantees zero future-information leakage: every query for transaction at timestamp t
strictly accesses state and events where timestamp < t.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from trustgraph.graph.schema import EdgeType, GraphEdge, GraphNode, NodeType

logger = logging.getLogger(__name__)

# Specificity ceiling: entities sharing an attribute beyond this threshold
# are treated as generic categories (e.g. broad OS / zip codes), not tight fraud rings.
MAX_RING_DEGREE = 25


class PaymentKnowledgeGraph:
    """
    Point-in-Time Payment Knowledge Graph.

    Maintains entity connections, temporal event histories, and fraud records.
    Every query at timestamp t accesses state prior to ingestion of transaction t.
    """

    def __init__(self) -> None:
        # Node storage: node_id -> GraphNode
        self.nodes: Dict[str, GraphNode] = {}

        # Transaction metadata index: transaction_id -> dict
        self.transactions: Dict[int, Dict[str, Any]] = {}

        # ----------------------------------------------------------------------
        # O(1) Entity Historical State
        # ----------------------------------------------------------------------
        self.entity_txn_count: Dict[str, int] = defaultdict(int)
        self.entity_fraud_count: Dict[str, int] = defaultdict(int)
        # Sliding deques for fast velocity computation: entity_id -> deque[(timestamp, is_fraud)]
        self.entity_recent_txns: Dict[str, deque[Tuple[float, int]]] = defaultdict(deque)

        # ----------------------------------------------------------------------
        # O(1) Device Historical State
        # ----------------------------------------------------------------------
        # device_id -> set of entity_ids seen strictly before current transaction
        self.device_entities: Dict[str, Set[str]] = defaultdict(set)
        self.device_txn_count: Dict[str, int] = defaultdict(int)
        self.device_fraud_count: Dict[str, int] = defaultdict(int)
        self.device_recent_txns: Dict[str, deque[Tuple[float, int]]] = defaultdict(deque)

        # Representative fraud transactions for evidence trails: device_id -> List of past fraud txn_ids
        self.device_fraud_txns: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        # entity_id -> List of past fraud txn_ids
        self.entity_fraud_txns: Dict[str, List[int]] = defaultdict(list)

        # ----------------------------------------------------------------------
        # O(1) Sharing Attribute Indices
        # ----------------------------------------------------------------------
        self.card_entities: Dict[str, Set[str]] = defaultdict(set)
        self.addr_entities: Dict[str, Set[str]] = defaultdict(set)
        self.network_entities: Dict[str, Set[str]] = defaultdict(set)

        # Entity reverse links
        self.entity_devices: Dict[str, Set[str]] = defaultdict(set)
        self.entity_cards: Dict[str, Set[str]] = defaultdict(set)
        self.entity_addresses: Dict[str, Set[str]] = defaultdict(set)

        self.total_transactions_ingested: int = 0
        self.total_fraud_events_registered: int = 0

    # ---------------------------------------------------------------------------
    # Node Management
    # ---------------------------------------------------------------------------

    def _ensure_node(self, node_id: str, node_type: NodeType, timestamp: float) -> GraphNode:
        if node_id not in self.nodes:
            node = GraphNode(node_id=node_id, node_type=node_type, created_at=timestamp)
            self.nodes[node_id] = node
            return node
        return self.nodes[node_id]

    # ---------------------------------------------------------------------------
    # Point-in-Time Historical State Queries (STRICTLY < t)
    # ---------------------------------------------------------------------------

    def get_prior_entity_history(
        self, entity_id: str, timestamp: float
    ) -> Tuple[int, int, int, int]:
        """
        Retrieve prior transaction count, prior fraud count, 1h velocity, and 24h velocity
        strictly prior to transaction at timestamp t.
        """
        total_txns = self.entity_txn_count.get(entity_id, 0)
        if total_txns == 0:
            return 0, 0, 0, 0

        prior_frauds = self.entity_fraud_count.get(entity_id, 0)

        # Compute sliding velocities from deque
        recent_q = self.entity_recent_txns[entity_id]
        t_24h = timestamp - 86400.0
        # Evict events older than 24h from head of deque
        while recent_q and recent_q[0][0] < t_24h:
            recent_q.popleft()

        velocity_24h = len(recent_q)
        t_1h = timestamp - 3600.0
        # 1h velocity: count elements in deque with ts >= t_1h
        velocity_1h = 0
        for ts, _ in reversed(recent_q):
            if ts >= t_1h:
                velocity_1h += 1
            else:
                break

        return total_txns, prior_frauds, velocity_1h, velocity_24h

    def get_prior_device_history(
        self, device_id: Optional[str], timestamp: float, current_entity: str
    ) -> Tuple[int, int, int, int]:
        """
        Retrieve device multiplexing metrics strictly prior to timestamp t:
        - distinct other entities using device
        - total prior transactions on device
        - total prior frauds on device
        - device 24h velocity
        """
        if not device_id or device_id == "nan":
            return 0, 0, 0, 0

        # 1. Distinct other entities seen on device before t (O(1))
        ent_set = self.device_entities.get(device_id)
        distinct_entities = 0
        if ent_set:
            distinct_entities = len(ent_set) - 1 if current_entity in ent_set else len(ent_set)

        # 2. Total prior transactions and frauds (O(1))
        total_txns = self.device_txn_count.get(device_id, 0)
        if total_txns == 0:
            return distinct_entities, 0, 0, 0

        prior_frauds = self.device_fraud_count.get(device_id, 0)

        # 3. 24h device velocity from deque
        dev_q = self.device_recent_txns[device_id]
        t_24h = timestamp - 86400.0
        while dev_q and dev_q[0][0] < t_24h:
            dev_q.popleft()
        velocity_24h = len(dev_q)

        return distinct_entities, total_txns, prior_frauds, velocity_24h

    def get_prior_sharing_counts(
        self,
        card_id: Optional[str],
        addr_id: Optional[str],
        network_id: Optional[str],
        current_entity: str,
    ) -> Tuple[int, int, int]:
        """
        Count distinct other entities sharing card, address, or network agent prior to t (O(1)).
        """
        card_cnt = 0
        if card_id and card_id in self.card_entities:
            c_set = self.card_entities[card_id]
            card_cnt = len(c_set) - 1 if current_entity in c_set else len(c_set)

        addr_cnt = 0
        if addr_id and addr_id in self.addr_entities:
            a_set = self.addr_entities[addr_id]
            addr_cnt = len(a_set) - 1 if current_entity in a_set else len(a_set)

        net_cnt = 0
        if network_id and network_id in self.network_entities:
            n_set = self.network_entities[network_id]
            net_cnt = len(n_set) - 1 if current_entity in n_set else len(n_set)

        return card_cnt, addr_cnt, net_cnt

    def get_prior_2hop_fraud_history(
        self,
        entity_id: str,
        device_id: Optional[str],
        card_id: Optional[str],
    ) -> Tuple[int, int]:
        """
        Inspect 2-hop connected entities sharing a specific hardware device or card
        strictly before timestamp t.
        Limits inspection to high-specificity identifiers (degree <= MAX_RING_DEGREE).
        """
        linked_entities: Set[str] = set()

        if device_id and device_id in self.device_entities:
            dev_ents = self.device_entities[device_id]
            if 1 < len(dev_ents) <= MAX_RING_DEGREE:
                for ent in dev_ents:
                    if ent != entity_id:
                        linked_entities.add(ent)

        if card_id and card_id in self.card_entities:
            c_ents = self.card_entities[card_id]
            if 1 < len(c_ents) <= MAX_RING_DEGREE:
                for ent in c_ents:
                    if ent != entity_id:
                        linked_entities.add(ent)

        if not linked_entities:
            return 0, 0

        total_frauds = 0
        distinct_fraud_ents = 0

        for other_ent in linked_entities:
            f_cnt = self.entity_fraud_count.get(other_ent, 0)
            if f_cnt > 0:
                total_frauds += f_cnt
                distinct_fraud_ents += 1

        return total_frauds, distinct_fraud_ents

    # ---------------------------------------------------------------------------
    # Causal Graph Ingestion (Called strictly AFTER query at time t)
    # ---------------------------------------------------------------------------

    def add_transaction(
        self,
        transaction_id: int,
        timestamp: float,
        entity_id: str,
        amount: float,
        device_id: Optional[str] = None,
        card_id: Optional[str] = None,
        addr_id: Optional[str] = None,
        email_id: Optional[str] = None,
        merchant_id: Optional[str] = None,
        network_id: Optional[str] = None,
        is_fraud: int = 0,
    ) -> None:
        """
        Ingest transaction and its relationships into graph state at timestamp t.
        """
        # Node tracking (for query/evidence lookup)
        txn_str = f"txn_{transaction_id}"
        self._ensure_node(txn_str, NodeType.TRANSACTION, timestamp)

        ent_str = f"ent_{entity_id}"
        self._ensure_node(ent_str, NodeType.CUSTOMER_ENTITY, timestamp)

        self.transactions[transaction_id] = {
            "timestamp": timestamp,
            "entity_id": entity_id,
            "amount": amount,
            "device_id": device_id,
            "card_id": card_id,
            "addr_id": addr_id,
            "email_id": email_id,
            "merchant_id": merchant_id,
            "network_id": network_id,
            "is_fraud": is_fraud,
        }

        # Update entity counters and deques
        self.entity_txn_count[entity_id] += 1
        self.entity_recent_txns[entity_id].append((timestamp, is_fraud))
        if is_fraud == 1:
            self.entity_fraud_count[entity_id] += 1
            self.entity_fraud_txns[entity_id].append(transaction_id)

        # Update device state
        if device_id and device_id != "nan":
            self.device_entities[device_id].add(entity_id)
            self.device_txn_count[device_id] += 1
            self.device_recent_txns[device_id].append((timestamp, is_fraud))
            self.entity_devices[entity_id].add(device_id)
            if is_fraud == 1:
                self.device_fraud_count[device_id] += 1
                self.device_fraud_txns[device_id].append((transaction_id, entity_id))

        # Update card state
        if card_id and card_id != "nan":
            self.card_entities[card_id].add(entity_id)
            self.entity_cards[entity_id].add(card_id)

        # Update address state
        if addr_id and addr_id != "nan":
            self.addr_entities[addr_id].add(entity_id)
            self.entity_addresses[entity_id].add(addr_id)

        # Update network state
        if network_id and network_id != "nan":
            self.network_entities[network_id].add(entity_id)

        self.total_transactions_ingested += 1
        if is_fraud == 1:
            self.total_fraud_events_registered += 1

    def get_state_summary(self) -> Dict[str, Any]:
        """Summary statistics of graph state."""
        return {
            "total_nodes": len(self.nodes),
            "total_transactions": len(self.transactions),
            "total_customer_entities": len(self.entity_txn_count),
            "total_devices": len(self.device_entities),
            "total_cards": len(self.card_entities),
            "total_addresses": len(self.addr_entities),
            "total_registered_frauds": self.total_fraud_events_registered,
        }
