"""
graph_engine.py — TRUSTGRAPH Phase 3 Lightweight Relational Graph
==================================================================

Implements a causal in-memory bipartite-style graph between entity proxies
and attribute values, computing relational degree (D_t) and relationship
velocity (V_t) strictly from history prior to each transaction.

Design constraints:
  - Graph state is NEVER reset at split boundaries (persists TRAIN → VAL → TEST).
  - Attribute-frequency ceiling (k_attr_max) is computed once on TRAIN only.
  - Mandatory causal update order: query → evaluate → insert.
  - No GNNs, no graph embeddings. Pure dictionary lookups.
  - addr1 and P_emaildomain can be passed as extra_attrs for ablation
    experiments only. DeviceInfo is the primary relational attribute.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RelationalRecord:
    """Per-transaction relational risk output."""
    transaction_id: int
    entity_id: str
    timestamp: float

    # Degree components
    d_t: int           # distinct other entities connected via specific attrs (before t)
    D_t: float         # normalised degree [0, 1]

    # Velocity components
    v_t: int           # distinct NEW entity pairs first formed in [t-W, t)
    V_t: float         # normalised velocity [0, 1]

    # Combined relational risk
    G_t: float

    # Breakdown by attribute (for ablation/explanation)
    attr_d_t: Dict[str, int] = field(default_factory=dict)
    attr_v_t: Dict[str, int] = field(default_factory=dict)
    filtered_attrs: List[str] = field(default_factory=list)


@dataclass
class GraphParameters:
    """Fully specified set of relational parameters."""
    k_attr_max: int    = 25
    window_sec: float  = 86_400.0   # 24 h
    d_ref: float       = 5.0
    v_ref: float       = 3.0
    w_D: float         = 0.5
    w_V: float         = 0.5
    relational_attrs:  Tuple[str, ...] = ("DeviceInfo",)

    def __post_init__(self) -> None:
        if abs(self.w_D + self.w_V - 1.0) > 1e-9:
            raise ValueError("w_D + w_V must equal 1.0")


# ---------------------------------------------------------------------------
# Core graph engine
# ---------------------------------------------------------------------------

class LightweightRelationalGraph:
    """
    Causal bipartite relational graph for entity-proxy ↔ attribute connections.

    Usage pattern:
        engine = LightweightRelationalGraph(params)
        engine.fit_attribute_frequency_ceiling(train_df)
        for t in transactions:
            record = engine.score(t.entity_id, t.timestamp,
                                  t.transaction_id, attr_dict)
            # ... evaluate decisions ...
            engine.update(t.entity_id, t.timestamp, attr_dict)
    """

    def __init__(self, params: GraphParameters) -> None:
        self.params = params

        # ------------------------------------------------------------------
        # String intern registries — convert str keys to compact int IDs.
        # Human-readable strings remain available via the reverse lists.
        # All internal adjacency structures use int IDs to avoid repeated
        # Python str hashing and frozenset() object allocation overhead.
        # ------------------------------------------------------------------
        # entity string -> int ID
        self._entity_to_id: Dict[str, int] = {}
        self._id_to_entity: List[str] = []
        # (attr_name, attr_val) string tuple -> int ID
        self._attrkey_to_id: Dict[Tuple[str, str], int] = {}
        self._id_to_attrkey: List[Tuple[str, str]] = []

        # Bipartite state using integer IDs:
        # entity_to_attrs[eid] = set of attrkey int IDs
        self._entity_to_attrs: Dict[int, Set[int]] = defaultdict(set)
        # attr_to_entities[akid] = set of entity int IDs
        self._attr_to_entities: Dict[int, Set[int]] = defaultdict(set)
        # entity_neighbors[eid] = set of neighbor entity int IDs
        self._entity_neighbors: Dict[int, Set[int]] = defaultdict(set)

        # Relationship first-seen: packed int64 pair key -> timestamp
        # key = (min_eid << 32) | max_eid  — unique for any ordered int pair
        self._relationship_first_seen: Dict[int, float] = {}

        # Per-entity velocity deque: eid -> deque[(timestamp, neighbor_eid)]
        self._entity_velocity_events: Dict[int, deque] = defaultdict(deque)

        # Attribute-frequency ceiling (fitted on TRAIN) — still uses str tuples
        self._blocked_attr_values: Set[Tuple[str, str]] = set()
        self._attr_freq_cache_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers: intern strings to int IDs
    # ------------------------------------------------------------------

    def _intern_entity(self, entity_str: str) -> int:
        """Return existing or newly assigned int ID for an entity string."""
        eid = self._entity_to_id.get(entity_str)
        if eid is None:
            eid = len(self._id_to_entity)
            self._entity_to_id[entity_str] = eid
            self._id_to_entity.append(entity_str)
        return eid

    def _intern_attrkey(self, key: Tuple[str, str]) -> int:
        """Return existing or newly assigned int ID for an (attr_name, attr_val) pair."""
        akid = self._attrkey_to_id.get(key)
        if akid is None:
            akid = len(self._id_to_attrkey)
            self._attrkey_to_id[key] = akid
            self._id_to_attrkey.append(key)
        return akid

    @staticmethod
    def _pair_key(eid_a: int, eid_b: int) -> int:
        """Pack two int IDs into a single int64 relationship key (order-independent)."""
        lo, hi = (eid_a, eid_b) if eid_a < eid_b else (eid_b, eid_a)
        return (lo << 32) | hi

    # ------------------------------------------------------------------
    # Phase-1 only: attribute-frequency ceiling (must run on TRAIN data)
    # ------------------------------------------------------------------

    def fit_attribute_frequency_ceiling(
        self,
        df,
        entity_col: str = "entity_proxy",
    ) -> Dict[str, Any]:
        """
        Compute per-attribute value distinct-entity counts on TRAIN.
        Populate _blocked_attr_values for values exceeding k_attr_max.

        Parameters
        ----------
        df : pd.DataFrame — TRAIN partition with entity_proxy column.
        entity_col : str — column name for entity proxy.

        Returns
        -------
        Diagnostic dict with blocked counts per attribute.
        """
        k = self.params.k_attr_max
        diagnostics: Dict[str, Any] = {}
        self._blocked_attr_values = set()

        for attr in self.params.relational_attrs:
            if attr not in df.columns:
                diagnostics[attr] = {"error": "column not found"}
                continue

            non_null = df[df[attr].notna() & (~df[entity_col].str.startswith("unresolved_"))]
            freq = non_null.groupby(attr)[entity_col].nunique()
            blocked = freq[freq > k]
            for val in blocked.index:
                self._blocked_attr_values.add((attr, str(val)))
            diagnostics[attr] = {
                "total_values": int(len(freq)),
                "blocked_count": int(len(blocked)),
                "blocked_values": sorted(blocked.index.tolist()),
                "pct_blocked": round(100.0 * len(blocked) / len(freq), 2) if len(freq) else 0,
            }

        self._attr_freq_cache_fitted = True
        return diagnostics

    def load_blocked_values(self, blocked: Set[Tuple[str, str]]) -> None:
        """Restore a previously serialised blocked-values set."""
        self._blocked_attr_values = blocked
        self._attr_freq_cache_fitted = True

    # ------------------------------------------------------------------
    # Helper: resolve non-blocked attribute pairs for a row
    # ------------------------------------------------------------------

    def _resolve_attrs(
        self, attr_dict: Dict[str, Optional[str]]
    ) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Return (valid_pairs, filtered_labels).
        valid_pairs   — (attr_name, attr_val) tuples to use for graph edges.
        filtered_labels — human-readable descriptions of skipped values.
        """
        valid, filtered = [], []
        for attr in self.params.relational_attrs:
            val = attr_dict.get(attr)
            if val is None or (isinstance(val, float) and __import__("math").isnan(val)):
                continue
            val_str = str(val)
            key = (attr, val_str)
            if key in self._blocked_attr_values:
                filtered.append(f"{attr}={val_str}")
            else:
                valid.append(key)
        return valid, filtered

    # ------------------------------------------------------------------
    # Score — MUST be called BEFORE update for the same transaction
    # ------------------------------------------------------------------

    def score(
        self,
        entity_id: str,
        timestamp: float,
        transaction_id: int,
        attr_dict: Dict[str, Optional[str]],
    ) -> RelationalRecord:
        """
        Compute relational risk from the graph state PRIOR to this transaction.
        Does NOT modify graph state.
        """
        valid_attrs, filtered = self._resolve_attrs(attr_dict)

        # --- Intern entity id ---
        eid = self._intern_entity(entity_id)

        # --- Degree: distinct other entity IDs connected via any valid attr ---
        other_eids: Set[int] = set()
        attr_d_t: Dict[str, int] = {}
        for str_key in valid_attrs:
            attr_name = str_key[0]
            akid = self._attrkey_to_id.get(str_key)  # may not exist yet (new attr val)
            if akid is not None:
                neighbors = self._attr_to_entities.get(akid, set())
                others = neighbors - {eid}
                attr_d_t[attr_name] = attr_d_t.get(attr_name, 0) + len(others)
                other_eids.update(others)
        d_t = len(other_eids)
        D_t = min(1.0, d_t / self.params.d_ref) if self.params.d_ref > 0 else 0.0

        # --- Velocity: distinct NEW pairs first seen in [t-W, t) ---
        W = self.params.window_sec
        window_start = timestamp - W

        # Prune entity's velocity events older than window_start
        q = self._entity_velocity_events.get(eid)
        if q:
            while q and q[0][0] < window_start:
                q.popleft()
            v_t = len(q)
        else:
            v_t = 0

        V_t = min(1.0, v_t / self.params.v_ref) if self.params.v_ref > 0 else 0.0
        G_t = self.params.w_D * D_t + self.params.w_V * V_t

        return RelationalRecord(
            transaction_id=transaction_id,
            entity_id=entity_id,
            timestamp=timestamp,
            d_t=d_t,
            D_t=D_t,
            v_t=v_t,
            V_t=V_t,
            G_t=G_t,
            attr_d_t=attr_d_t,
            attr_v_t={},
            filtered_attrs=filtered,
        )

    # ------------------------------------------------------------------
    # Update — MUST be called AFTER score for the same transaction
    # ------------------------------------------------------------------

    def update(
        self,
        entity_id: str,
        timestamp: float,
        attr_dict: Dict[str, Optional[str]],
    ) -> None:
        """
        Insert this transaction's entity-attribute relationships into the graph.
        Must be called AFTER score() for the same transaction.
        """
        valid_attrs, _ = self._resolve_attrs(attr_dict)
        eid = self._intern_entity(entity_id)
        existing_neighbors = self._entity_neighbors.get(eid, set())
        new_eids: Set[int] = set()

        for str_key in valid_attrs:
            # Intern attribute key
            akid = self._intern_attrkey(str_key)
            existing_entities = self._attr_to_entities.get(akid, set())
            new_others = existing_entities - {eid} - existing_neighbors
            new_eids.update(new_others)

            # Register new relationships for velocity tracking
            for other_eid in new_others:
                pair = self._pair_key(eid, other_eid)
                if pair not in self._relationship_first_seen:
                    self._relationship_first_seen[pair] = timestamp
                    self._entity_velocity_events[eid].append((timestamp, other_eid))
                    self._entity_velocity_events[other_eid].append((timestamp, eid))

            # Insert entity <-> attr edge (int IDs)
            self._entity_to_attrs[eid].add(akid)
            self._attr_to_entities[akid].add(eid)

        # Update neighbour set incrementally
        if new_eids:
            self._entity_neighbors[eid].update(new_eids)
            for nb_eid in new_eids:
                self._entity_neighbors[nb_eid].add(eid)

    # ------------------------------------------------------------------
    # Persistence helpers (for checkpointing across splits if needed)
    # ------------------------------------------------------------------

    def get_state_summary(self) -> Dict[str, Any]:
        """Lightweight summary for logging."""
        return {
            "total_entities": len(self._entity_to_id),
            "total_attr_values": len(self._attrkey_to_id),
            "total_known_relationships": len(self._relationship_first_seen),
            "total_tracked_velocity_entities": len(self._entity_velocity_events),
            "blocked_attr_value_count": len(self._blocked_attr_values),
        }

    def export_blocked_values(self) -> Set[Tuple[str, str]]:
        return set(self._blocked_attr_values)


# ---------------------------------------------------------------------------
# Batch processing utilities
# ---------------------------------------------------------------------------

def build_attr_dict(row, attrs: List[str]) -> Dict[str, Optional[str]]:
    """Extract attribute values from a DataFrame row into a plain dict."""
    d = {}
    for a in attrs:
        val = row.get(a)
        import math
        if val is None or (isinstance(val, float) and math.isnan(val)):
            d[a] = None
        else:
            d[a] = str(val)
    return d


def process_partition(
    df,
    engine: LightweightRelationalGraph,
    entity_col: str = "entity_proxy",
    timestamp_col: str = "TransactionDT",
    id_col: str = "TransactionID",
    label_col: Optional[str] = None,
) -> List[RelationalRecord]:
    """
    Process a partition chronologically through the graph engine.
    Returns a list of RelationalRecord (one per transaction).

    CAUSAL ORDER (enforced):
      For each row:
        1. score()    <- query graph state BEFORE t
        2. [collect record]
        3. update()   <- insert AFTER scoring
    """
    attrs = list(engine.params.relational_attrs)
    records = []

    entities = df[entity_col].astype(str).tolist()
    timestamps = df[timestamp_col].astype(float).tolist()
    txn_ids = df[id_col].astype(int).tolist()
    attr_cols = [df[a].tolist() for a in attrs]
    n_rows = len(df)

    for i in range(n_rows):
        entity_id = entities[i]
        timestamp = timestamps[i]
        txn_id = txn_ids[i]
        attr_dict = {}
        for j, a in enumerate(attrs):
            val = attr_cols[j][i]
            if val is not None and not (isinstance(val, float) and (val != val)):
                attr_dict[a] = str(val)
            else:
                attr_dict[a] = None

        rec = engine.score(entity_id, timestamp, txn_id, attr_dict)
        records.append(rec)
        engine.update(entity_id, timestamp, attr_dict)

    return records
