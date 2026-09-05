"""
features.py — Contextual Graph Feature Extractor
=================================================

Extracts point-in-time contextual feature records from the payment knowledge graph
strictly prior to transaction timestamp t.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from trustgraph.graph.schema import GraphFeatureRecord
from trustgraph.graph.temporal_graph import PaymentKnowledgeGraph


def resolve_customer_entity_key(
    row: Union[Dict[str, Any], pd.Series], txn_id: int
) -> str:
    """
    Derive the primary composite customer entity proxy (card1 + addr1 + P_emaildomain).
    """
    card1 = row.get("card1")
    addr1 = row.get("addr1")
    email = row.get("P_emaildomain")

    c1_valid = card1 is not None and not _is_nan(card1)
    a1_valid = addr1 is not None and not _is_nan(addr1)
    em_valid = email is not None and not _is_nan(email)

    if c1_valid and a1_valid and em_valid:
        return f"{card1}_{addr1}_{email}"
    elif c1_valid and a1_valid:
        return f"{card1}_{addr1}"
    elif c1_valid:
        return f"{card1}"
    return f"unresolved_{txn_id}"


def _is_nan(val: Any) -> bool:
    if val is None:
        return True
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False


def _clean_str(val: Any) -> Optional[str]:
    if val is None or _is_nan(val):
        return None
    s = str(val).strip()
    return s if s and s.lower() != "nan" else None


class GraphFeatureExtractor:
    """
    Extracts point-in-time relational signals from PaymentKnowledgeGraph.
    """

    def __init__(self, graph: PaymentKnowledgeGraph) -> None:
        self.graph = graph

    def extract_features(
        self,
        transaction_id: int,
        timestamp: float,
        row_dict: Dict[str, Any],
        entity_id: Optional[str] = None,
    ) -> GraphFeatureRecord:
        """
        Extract causal graph features strictly at timestamp t (using events < t).
        """
        if entity_id is None:
            entity_id = resolve_customer_entity_key(row_dict, transaction_id)

        device_id = _clean_str(row_dict.get("DeviceInfo"))
        card_id = _clean_str(row_dict.get("card1"))
        addr_id = _clean_str(row_dict.get("addr1"))
        network_id = _clean_str(row_dict.get("id_31"))

        # 1. Entity History (< t)
        p_txns, p_frauds, v_1h, v_24h = self.graph.get_prior_entity_history(
            entity_id, timestamp
        )
        ent_fraud_rate = (p_frauds / p_txns) if p_txns > 0 else 0.0

        # 2. Device History & Multiplexing (< t)
        d_ents, d_txns, d_frauds, d_v24h = self.graph.get_prior_device_history(
            device_id, timestamp, entity_id
        )
        dev_fraud_rate = (d_frauds / d_txns) if d_txns > 0 else 0.0
        unusual_device = 1 if d_ents >= 3 else 0

        # 3. Sharing Attributes (< t)
        card_ents, addr_ents, net_ents = self.graph.get_prior_sharing_counts(
            card_id, addr_id, network_id, entity_id
        )

        # 4. 2-Hop Network Contamination (< t)
        h2_frauds, h2_fraud_ents = self.graph.get_prior_2hop_fraud_history(
            entity_id, device_id, card_id
        )

        # 5. Cold-Start / Coverage Indicator
        has_context = 1 if (p_txns > 0 or d_txns > 0 or card_ents > 0 or addr_ents > 0) else 0

        return GraphFeatureRecord(
            transaction_id=transaction_id,
            timestamp=timestamp,
            entity_id=entity_id,
            prior_entity_txns=p_txns,
            prior_entity_frauds=p_frauds,
            entity_fraud_rate=ent_fraud_rate,
            entity_velocity_1h=v_1h,
            entity_velocity_24h=v_24h,
            device_id=device_id,
            device_entity_count=d_ents,
            device_prior_txns=d_txns,
            device_prior_frauds=d_frauds,
            device_fraud_rate=dev_fraud_rate,
            device_velocity_24h=d_v24h,
            unusual_device_sharing=unusual_device,
            card_entity_count=card_ents,
            address_entity_count=addr_ents,
            network_entity_count=net_ents,
            hop2_linked_frauds=h2_frauds,
            hop2_distinct_fraud_entities=h2_fraud_ents,
            has_graph_context=has_context,
            graph_risk=0.0,
        )
