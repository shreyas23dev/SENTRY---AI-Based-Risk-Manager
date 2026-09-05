"""
builder.py — Graph Stream Pipeline Builder
==========================================

Streams chronologically ordered transactions through the PaymentKnowledgeGraph,
extracting point-in-time contextual features and computing G_t causally.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from trustgraph.graph.features import (
    GraphFeatureExtractor,
    _clean_str,
    resolve_customer_entity_key,
)
from trustgraph.graph.risk_engine import EntityGraphRiskEngine
from trustgraph.graph.schema import GraphFeatureRecord, TransactionEvidence
from trustgraph.graph.temporal_graph import PaymentKnowledgeGraph

logger = logging.getLogger(__name__)


class GraphPipelineBuilder:
    """
    Coordinates causal graph ingestion, feature extraction, and G_t calculation.
    """

    def __init__(
        self,
        graph: Optional[PaymentKnowledgeGraph] = None,
        risk_engine: Optional[EntityGraphRiskEngine] = None,
    ) -> None:
        self.graph = graph or PaymentKnowledgeGraph()
        self.extractor = GraphFeatureExtractor(self.graph)
        self.risk_engine = risk_engine or EntityGraphRiskEngine()
        # Cache of evaluated records for fast evidence retrieval: txn_id -> GraphFeatureRecord
        self._record_cache: Dict[int, GraphFeatureRecord] = {}

    def score_and_ingest_transaction(
        self,
        transaction_id: int,
        timestamp: float,
        row_dict: Dict[str, Any],
        is_fraud: int = 0,
        update_label: bool = True,
    ) -> GraphFeatureRecord:
        """
        Causal step:
          1. Extract features strictly using events < t.
          2. Compute G_t.
          3. Ingest transaction into graph at timestamp t.
        """
        entity_id = resolve_customer_entity_key(row_dict, transaction_id)

        # 1. Query point-in-time features (< t)
        record = self.extractor.extract_features(
            transaction_id=transaction_id,
            timestamp=timestamp,
            row_dict=row_dict,
            entity_id=entity_id,
        )

        # 2. Compute G_t
        self.risk_engine.compute_graph_risk(record)
        self._record_cache[transaction_id] = record

        # 3. Causal update (register transaction into graph state)
        amount = float(row_dict.get("TransactionAmt", 0.0))
        device_id = _clean_str(row_dict.get("DeviceInfo"))
        card_id = _clean_str(row_dict.get("card1"))
        addr_id = _clean_str(row_dict.get("addr1"))
        email_id = _clean_str(row_dict.get("P_emaildomain"))
        merchant_id = _clean_str(row_dict.get("R_emaildomain")) or _clean_str(row_dict.get("ProductCD"))
        network_id = _clean_str(row_dict.get("id_31"))

        label_to_record = is_fraud if update_label else 0

        self.graph.add_transaction(
            transaction_id=transaction_id,
            timestamp=timestamp,
            entity_id=entity_id,
            amount=amount,
            device_id=device_id,
            card_id=card_id,
            addr_id=addr_id,
            email_id=email_id,
            merchant_id=merchant_id,
            network_id=network_id,
            is_fraud=label_to_record,
        )

        return record

    def process_dataframe_stream(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
        log_interval: int = 50_000,
    ) -> pd.DataFrame:
        """
        Stream an entire partition through the graph in chronological order.
        Returns a DataFrame of graph features and G_t for each transaction.
        """
        logger.info(
            "Streaming %d transactions through PaymentKnowledgeGraph (is_train=%s)...",
            len(df), is_train
        )
        t0 = time.perf_counter()

        txn_ids = df["TransactionID"].values
        timestamps = df["TransactionDT"].values
        has_labels = "isFraud" in df.columns
        labels = df["isFraud"].values if has_labels else np.zeros(len(df), dtype=int)

        # Pre-extract common column arrays for speed
        card1_arr = df["card1"].values if "card1" in df.columns else [None] * len(df)
        addr1_arr = df["addr1"].values if "addr1" in df.columns else [None] * len(df)
        pem_arr = df["P_emaildomain"].values if "P_emaildomain" in df.columns else [None] * len(df)
        dev_arr = df["DeviceInfo"].values if "DeviceInfo" in df.columns else [None] * len(df)
        id31_arr = df["id_31"].values if "id_31" in df.columns else [None] * len(df)
        amt_arr = df["TransactionAmt"].values if "TransactionAmt" in df.columns else [0.0] * len(df)
        rem_arr = df["R_emaildomain"].values if "R_emaildomain" in df.columns else [None] * len(df)
        prod_arr = df["ProductCD"].values if "ProductCD" in df.columns else [None] * len(df)

        records_list: List[Dict[str, Any]] = []

        for i in range(len(df)):
            t_id = int(txn_ids[i])
            t_ts = float(timestamps[i])
            is_fr = int(labels[i]) if has_labels else 0

            row_dict = {
                "card1": card1_arr[i],
                "addr1": addr1_arr[i],
                "P_emaildomain": pem_arr[i],
                "DeviceInfo": dev_arr[i],
                "id_31": id31_arr[i],
                "TransactionAmt": amt_arr[i],
                "R_emaildomain": rem_arr[i],
                "ProductCD": prod_arr[i],
            }

            # In training, labels are known historically. In test, labels are NOT recorded into graph.
            rec = self.score_and_ingest_transaction(
                transaction_id=t_id,
                timestamp=t_ts,
                row_dict=row_dict,
                is_fraud=is_fr,
                update_label=is_train,
            )
            records_list.append(rec.to_dict())

            if (i + 1) % log_interval == 0:
                elapsed = time.perf_counter() - t0
                speed = (i + 1) / elapsed
                logger.info(
                    "Processed %d/%d txns (%.1f txns/s)...",
                    i + 1, len(df), speed
                )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Finished streaming %d txns in %.2f s (%.1f txns/s).",
            len(df), elapsed, len(df) / max(elapsed, 1e-6)
        )

        out_df = pd.DataFrame(records_list)
        return out_df

    def get_transaction_evidence(self, transaction_id: int) -> Optional[TransactionEvidence]:
        """
        Query causal evidence for a previously scored transaction.
        """
        record = self._record_cache.get(transaction_id)
        if not record:
            txn_meta = self.graph.transactions.get(transaction_id)
            if not txn_meta:
                return None
            # Reconstruct record from metadata
            record = self.extractor.extract_features(
                transaction_id=transaction_id,
                timestamp=txn_meta["timestamp"],
                row_dict=txn_meta,
                entity_id=txn_meta["entity_id"],
            )
            self.risk_engine.compute_graph_risk(record)

        return self.risk_engine.generate_transaction_evidence(
            transaction_id=transaction_id,
            graph=self.graph,
            record=record,
        )
