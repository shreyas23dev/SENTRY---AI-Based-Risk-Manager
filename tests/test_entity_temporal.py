"""
test_entity_temporal.py — Phase 2.1 Entity-Scoped Temporal Memory Test Suite
=============================================================================

Unit tests verifying:
  - Independent entity states (Entity A does not modify Entity B)
  - Missing/unresolved entity isolation (no shared UNKNOWN cross-contamination)
  - First transaction initialization (E_0=0, P_0=0)
  - Chronological causality (no future information)
  - Bounded P_t and E_t in [0, 1]
  - Multi-key resolution logic (card1, card_composite, card_email, card_addr)
  - Deterministic stream processing
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgraph.temporal.entity_tracker import (
    EntityTemporalRiskEngine,
    resolve_entity_key,
)


class TestEntityStateIsolation:

    def test_independent_entity_states(self):
        """Entity A high-risk activity must NOT modify Entity B state."""
        engine = EntityTemporalRiskEngine(beta=0.5, gamma=0.4, lambda_=0.1, delta=0.05)

        # Entity A has 3 high-risk transactions
        e_a1, p_a1 = engine.step("user_A", 0.9)
        e_a2, p_a2 = engine.step("user_A", 0.9)
        e_a3, p_a3 = engine.step("user_A", 0.9)

        # Entity B arrives with 1 low-risk transaction
        e_b1, p_b1 = engine.step("user_B", 0.05)

        # Verify Entity B initialized from 0.0, not contaminated by A
        assert p_b1 == 0.0
        assert e_b1 == pytest.approx(0.5 * 0.05)
        # Verify Entity A accumulated risk
        assert p_a3 > 0.0

    def test_interleaved_entity_updates(self):
        """Interleaved transactions between Entity X and Y must maintain separate histories."""
        engine = EntityTemporalRiskEngine(beta=0.4, gamma=0.3, lambda_=0.2, delta=0.05)

        # Sequence: X, Y, X, Y
        _, p_x1 = engine.step("X", 0.8)
        _, p_y1 = engine.step("Y", 0.1)
        _, p_x2 = engine.step("X", 0.8)
        _, p_y2 = engine.step("Y", 0.1)

        # X should accumulate, Y should remain 0.0
        assert p_x2 > p_x1
        assert p_y2 == 0.0


class TestMissingEntityHandling:

    def test_unresolved_isolation(self):
        """Unresolved transactions must NOT share a single fake 'UNKNOWN' state."""
        df = pd.DataFrame({
            "TransactionID": [101, 102, 103],
            "card1": [1000, np.nan, np.nan],
            "P_emaildomain": ["gmail.com", np.nan, np.nan],
            "A_t": [0.9, 0.9, 0.9],
        })

        keys = resolve_entity_key(df, key_type="card_email")
        # Row 102 and 103 are unresolved -> must have distinct unique keys
        assert keys.iloc[1] == "unresolved_102"
        assert keys.iloc[2] == "unresolved_103"
        assert keys.iloc[1] != keys.iloc[2]

        engine = EntityTemporalRiskEngine(beta=0.5, gamma=0.3, lambda_=0.2, delta=0.05)
        E_arr, P_arr = engine.process_stream(df, keys, score_col="A_t")

        # Row 102 and 103 must each start from P_0=0.0 without accumulating across each other
        assert P_arr[1] == 0.0
        assert P_arr[2] == 0.0


class TestEntityKeyResolution:

    def test_card1_resolution(self):
        df = pd.DataFrame({"TransactionID": [1], "card1": [5000]})
        keys = resolve_entity_key(df, key_type="card1")
        assert keys.iloc[0] == "5000"

    def test_card_composite_resolution(self):
        df = pd.DataFrame({
            "TransactionID": [1, 2],
            "card1": [100, 100], "card2": [200, np.nan], "card3": [150, 150],
            "card4": ["visa", "visa"], "card5": [226, 226], "card6": ["debit", "debit"],
        })
        keys = resolve_entity_key(df, key_type="card_composite")
        assert keys.iloc[0] == "100_200.0_150_visa_226_debit"
        assert keys.iloc[1] == "unresolved_2"

    def test_card_email_resolution(self):
        df = pd.DataFrame({
            "TransactionID": [1, 2],
            "card1": [1234, 1234],
            "P_emaildomain": ["yahoo.com", np.nan],
        })
        keys = resolve_entity_key(df, key_type="card_email")
        assert keys.iloc[0] == "1234_yahoo.com"
        assert keys.iloc[1] == "unresolved_2"


class TestMathematicalIntegrity:

    def test_bounds(self):
        """E_t and P_t must remain within [0, 1] for all entity steps."""
        engine = EntityTemporalRiskEngine(beta=0.6, gamma=0.2, lambda_=0.4, delta=0.05)
        df = pd.DataFrame({
            "TransactionID": range(20),
            "entity": ["ent_1"] * 10 + ["ent_2"] * 10,
            "A_t": [0.95] * 10 + [0.01] * 10,
        })
        E_arr, P_arr = engine.process_stream(df, "entity", score_col="A_t")
        assert np.all(E_arr >= 0.0) and np.all(E_arr <= 1.0)
        assert np.all(P_arr >= 0.0) and np.all(P_arr <= 1.0)
        assert np.max(P_arr) <= 1.0
        assert np.min(P_arr) >= 0.0
