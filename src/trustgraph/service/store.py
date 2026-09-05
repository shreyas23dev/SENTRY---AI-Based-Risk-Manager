"""
store.py — Thread-Safe In-Memory Transaction Decision Store
============================================================

Maintains the state and evaluation history of processed transactions,
enabling O(1) query by transaction_id via GET /api/v1/risk/transactions/{transaction_id}.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Dict, Optional

from trustgraph.service.schemas import TransactionRiskResponse


class TransactionStore:
    """
    Thread-safe, bounded in-memory LRU store for evaluated transactions.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self._capacity = capacity
        self._store: OrderedDict[str, TransactionRiskResponse] = OrderedDict()
        self._lock = threading.Lock()

    def save(self, response: TransactionRiskResponse) -> None:
        """Save or update an evaluation result."""
        with self._lock:
            key = str(response.transaction_id)
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = response
            if len(self._store) > self._capacity:
                self._store.popitem(last=False)  # Evict oldest

    def get(self, transaction_id: str) -> Optional[TransactionRiskResponse]:
        """Retrieve the latest evaluated decision for a transaction_id."""
        with self._lock:
            key = str(transaction_id)
            if key not in self._store:
                return None
            self._store.move_to_end(key)
            return self._store[key]

    def count(self) -> int:
        """Total number of stored transactions."""
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Reset the store (for test isolation)."""
        with self._lock:
            self._store.clear()
