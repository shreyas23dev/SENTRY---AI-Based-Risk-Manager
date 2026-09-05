"""
test_risk_council.py — Unit & Integration Tests for Phase 8 Risk Council
========================================================================

Verifies:
  1. ML Analyst wrapper (A_t assessment & signals)
  2. Slow-Burn Analyst wrapper (P_t assessment & state)
  3. Agreement condition (both high/critical or both low)
  4. Disagreement condition (ML low, Slow-Burn high/critical or vice versa)
  5. Insufficient history condition (no prior entity history)
  6. Evidence grounding (all citations exist in evidence set)
  7. Council API endpoint (GET /api/v1/risk/{id}/council)
  8. Existing decision remains authoritative and unchanged
  9. Existing R_t remains unchanged
 10. Existing endpoints remain unchanged and operational
"""

import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgraph.council.analysts import SlowBurnAnalyst, TransactionRiskAnalyst
from trustgraph.council.council import RiskCouncil, get_risk_council
from trustgraph.council.officer import AIRiskOfficer
from trustgraph.service.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def council():
    return get_risk_council()


# ===========================================================================
# 1. ML Analyst Wrapper Tests
# ===========================================================================

class TestTransactionRiskAnalyst:

    def test_critical_assessment(self):
        analyst = TransactionRiskAnalyst()
        res = analyst.evaluate(transaction_id=101, A_t=0.85, amount=120.0)
        assert res["agent"] == "transaction_risk_analyst"
        assert res["risk"] == 0.85
        assert res["assessment"] == "CRITICAL"
        assert len(res["signals"]) > 0

    def test_low_assessment(self):
        analyst = TransactionRiskAnalyst()
        res = analyst.evaluate(transaction_id=102, A_t=0.01, amount=15.0)
        assert res["risk"] == 0.01
        assert res["assessment"] == "LOW"
        assert any("micro-testing" in s.lower() for s in res["signals"])


# ===========================================================================
# 2. Slow-Burn Analyst Wrapper Tests
# ===========================================================================

class TestSlowBurnAnalyst:

    def test_insufficient_history_when_zero_prior_txns(self):
        analyst = SlowBurnAnalyst()
        res = analyst.evaluate(transaction_id=201, prior_txns=0, prior_frauds=0)
        assert res["agent"] == "slow_burn_analyst"
        assert res["risk"] is None
        assert res["assessment"] == "INSUFFICIENT_HISTORY"
        assert res["state"] == "INSUFFICIENT_HISTORY"

    def test_high_assessment_on_recidivist(self):
        analyst = SlowBurnAnalyst()
        res = analyst.evaluate(
            transaction_id=202,
            prior_txns=50,
            prior_frauds=10,
            fraud_rate=0.20,
            device_sharing_count=4,
        )
        assert res["risk"] is not None
        assert res["risk"] >= 0.50
        assert res["assessment"] in ("HIGH", "CRITICAL")
        assert res["state"] == "ELEVATED"


# ===========================================================================
# 3. Council Agreement / Disagreement / Insufficient History Tests
# ===========================================================================

class TestCouncilScenarios:

    def test_agreement_on_high_risk_fraud(self, council):
        # Txn 3570805: Recidivist fraud
        res = council.evaluate(3570805)
        assert res["transaction_id"] == "3570805"
        assert res["transaction_analyst"]["assessment"] in ("HIGH", "CRITICAL")
        assert res["slow_burn_analyst"]["assessment"] in ("HIGH", "CRITICAL")
        assert res["council"]["status"] == "AGREEMENT"

    def test_disagreement_on_slow_burn_recovered_case(self, council):
        # Txn 3531382: Low ML risk (A_t < 0.10) but High persistent risk P_t
        res = council.evaluate(3531382)
        assert res["transaction_analyst"]["assessment"] == "LOW"
        assert res["slow_burn_analyst"]["assessment"] in ("HIGH", "CRITICAL")
        assert res["council"]["status"] in ("DISAGREEMENT", "SLOW_BURN_ONLY")
        assert res["council"]["relationship_type"] == "SLOW_BURN_ONLY"

    def test_insufficient_history_on_cold_start(self, council):
        # Txn 3488970: True cold start with 0 prior transactions
        res = council.evaluate(3488970)
        assert res["slow_burn_analyst"]["assessment"] == "INSUFFICIENT_HISTORY"
        assert res["council"]["status"] == "INSUFFICIENT_HISTORY"


# ===========================================================================
# 4. Evidence Grounding & Decision Invariance Tests
# ===========================================================================

class TestCouncilGroundingAndInvariance:

    def test_evidence_grounding_citations(self, council):
        res = council.evaluate(3570805)
        valid_eids = {e["evidence_id"] for e in res["evidence"]}
        valid_eids.add("RISK_ENGINE")
        for cid in res["citations"]:
            assert cid in valid_eids

    def test_existing_decision_remains_unchanged(self, council):
        # Decision must match the authoritative decision from risk engine
        res = council.evaluate(3504259)
        assert res["risk_engine"]["decision"] == "BLOCK"
        assert res["cost"]["optimal_action"] == "BLOCK"

    def test_existing_r_t_remains_unchanged(self, council):
        res = council.evaluate(3504259)
        assert abs(res["risk_engine"]["R_t"] - 0.5229) < 0.001


# ===========================================================================
# 5. API Endpoint Tests
# ===========================================================================

class TestCouncilApiEndpoints:

    def test_get_council_case_endpoint(self, client):
        resp = client.get("/api/v1/risk/3504259/council")
        assert resp.status_code == 200
        data = resp.json()
        assert "transaction_analyst" in data
        assert "slow_burn_analyst" in data
        assert "council" in data
        assert "officer_synthesis" in data
        assert "risk_engine" in data
        assert data["risk_engine"]["decision"] == "BLOCK"

    def test_existing_investigate_endpoint_unaffected(self, client):
        resp = client.get("/api/v1/risk/3504259/investigate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "BLOCK"

    def test_existing_graph_endpoint_unaffected(self, client):
        resp = client.get("/api/v1/risk/3504259/graph")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data


# ===========================================================================
# 6. Phase 9 AI Risk Officer (Groq LLM Reasoning & Guardrails) Tests
# ===========================================================================

import json
from unittest.mock import MagicMock, patch
from trustgraph.council.officer import SENTINEL_RISK_COUNCIL_SYSTEM_PROMPT, ALLOWED_COUNCIL_STATUSES
from trustgraph.investigator.schema import EvidenceItem, EvidenceType


class TestPhase9AIRiskOfficerGroqReasoning:
    """
    Verifies Section 12 requirements:
      1. Verified context only (no leakage)
      2. System prompt applied
      3. Structured schema enforced
      4. Citation filtering (anti-hallucination defense)
      5. Immutability of A_t, P_t, G_t, R_t
      6. Action invariance (LLM cannot change deterministic decision)
      7. Malformed JSON / error handling (graceful fallback)
      8. Agreement / disagreement / insufficient history scenarios
    """

    def test_verified_context_no_secrets_and_clean(self):
        officer = AIRiskOfficer()
        ev = [
            EvidenceItem(
                evidence_id="E1",
                evidence_type=EvidenceType.DIRECT_FRAUD,
                title="Direct Fraud",
                description="Entity has 3 prior chargebacks.",
                risk_weight=0.9,
                source_node="card_123",
            )
        ]
        context = officer.build_transaction_context(
            transaction_id=9999,
            transaction_analyst={"risk": 0.88, "assessment": "CRITICAL", "signals": ["Extreme transaction amount"]},
            slow_burn_analyst={"risk": 0.75, "assessment": "HIGH", "state": "ELEVATED", "signals": ["Prior fraud history"]},
            graph_risk_g=0.65,
            final_risk_r=0.8878,
            action="BLOCK",
            expected_cost=150.0,
            evidence_items=ev,
            amount=5000.0,
            timestamp=123456.0,
        )
        # Context must contain all verified values
        assert "9999" in context
        assert "A_t: 0.8800" in context
        assert "P_t: 0.7500" in context
        assert "G_t: 0.6500" in context
        assert "R_t: 0.8878" in context
        assert "BLOCK" in context
        assert "INR 5000.00" in context
        assert "[E1]" in context
        # Context must NOT contain secrets or internal objects
        assert "gsk_" not in context
        assert "api_key" not in context.lower()
        assert "<object" not in context
        assert "password" not in context.lower()

    def test_system_prompt_applied_to_llm_caller(self):
        mock_provider = MagicMock()
        mock_provider.name = "mock-groq"
        mock_provider.is_fallback = False
        mock_provider.model_name = "groq/compound-mini"

        officer = AIRiskOfficer(llm_provider=mock_provider)

        llm_reply = json.dumps({
            "council_status": "AGREEMENT",
            "reasoning": "Both analysts detect critical risk [RISK_ENGINE].",
            "transaction_analyst_interpretation": "A_t is elevated [RISK_ENGINE].",
            "slow_burn_interpretation": "P_t is elevated [E1].",
            "graph_interpretation": "G_t provides context.",
            "key_evidence": ["E1", "RISK_ENGINE"],
            "risk_engine_consistency": True,
        })

        ev = [EvidenceItem(evidence_id="E1", evidence_type=EvidenceType.DIRECT_FRAUD, title="Fraud", description="Desc", risk_weight=0.9, source_node="card_123")]

        with patch.object(officer, "_call_groq_council", return_value=llm_reply) as mock_call:
            res = officer.reason_council(
                transaction_id=12345,
                transaction_analyst={"risk": 0.9, "assessment": "CRITICAL", "signals": ["High amount"]},
                slow_burn_analyst={"risk": 0.85, "assessment": "HIGH", "state": "ELEVATED", "signals": ["Past fraud"]},
                graph_risk_g=0.5,
                final_risk_r=0.91,
                action="BLOCK",
                expected_cost=100.0,
                evidence_items=ev,
            )
            # Verify system prompt was passed to _call_groq_council
            assert mock_call.called
            passed_sys_prompt = mock_call.call_args[1].get("system_prompt") if mock_call.call_args[1] else mock_call.call_args[0][0]
            assert passed_sys_prompt == SENTINEL_RISK_COUNCIL_SYSTEM_PROMPT
            assert res["council_status"] == "AGREEMENT"
            assert res["llm_execution"]["invoked"] is True
            assert res["llm_execution"]["is_fallback"] is False

    def test_structured_schema_enforced(self):
        mock_provider = MagicMock()
        mock_provider.name = "mock-groq"
        mock_provider.is_fallback = False
        officer = AIRiskOfficer(llm_provider=mock_provider)

        llm_reply = json.dumps({
            "council_status": "DISAGREEMENT",
            "reasoning": "Transaction risk is low while slow-burn risk is high [RISK_ENGINE].",
            "transaction_analyst_interpretation": "Instantaneous signals appear benign.",
            "slow_burn_interpretation": "Longitudinal history reveals persistent fraud risk [E1].",
            "graph_interpretation": "Graph links to flagged device.",
            "key_evidence": ["E1"],
            "risk_engine_consistency": True,
        })
        ev = [EvidenceItem(evidence_id="E1", evidence_type=EvidenceType.DEVICE_SHARING, title="Device", description="Shared", risk_weight=0.8, source_node="dev_123")]

        with patch.object(officer, "_call_groq_council", return_value=llm_reply):
            res = officer.reason_council(
                transaction_id=54321,
                transaction_analyst={"risk": 0.05, "assessment": "LOW", "signals": []},
                slow_burn_analyst={"risk": 0.85, "assessment": "HIGH", "state": "ELEVATED", "signals": []},
                graph_risk_g=0.7,
                final_risk_r=0.82,
                action="VERIFY",
                expected_cost=40.0,
                evidence_items=ev,
            )
            required_keys = {
                "council_status",
                "reasoning",
                "transaction_analyst_interpretation",
                "slow_burn_interpretation",
                "graph_interpretation",
                "key_evidence",
                "risk_engine_consistency",
                "llm_execution",
            }
            assert required_keys.issubset(res.keys())
            assert res["council_status"] in ALLOWED_COUNCIL_STATUSES

    def test_citation_filtering_strips_hallucinated_ids(self):
        mock_provider = MagicMock()
        mock_provider.name = "mock-groq"
        mock_provider.is_fallback = False
        officer = AIRiskOfficer(llm_provider=mock_provider)

        # LLM attempts to cite non-existent evidence tags
        llm_reply = json.dumps({
            "council_status": "AGREEMENT",
            "reasoning": "Hallucinated evidence cited [E999] and [FAKE_ID].",
            "transaction_analyst_interpretation": "A_t is high.",
            "slow_burn_interpretation": "P_t is high.",
            "graph_interpretation": "Graph supports.",
            "key_evidence": ["E1", "E999_FAKE", "[HALLUCINATED_ID]"],
            "risk_engine_consistency": True,
        })
        # Only E1 is valid
        ev = [EvidenceItem(evidence_id="E1", evidence_type=EvidenceType.DIRECT_FRAUD, title="T", description="D", risk_weight=0.8, source_node="card_123")]

        with patch.object(officer, "_call_groq_council", return_value=llm_reply):
            res = officer.reason_council(
                transaction_id=111,
                transaction_analyst={"risk": 0.8, "assessment": "HIGH", "signals": []},
                slow_burn_analyst={"risk": 0.8, "assessment": "HIGH", "state": "ELEVATED", "signals": []},
                graph_risk_g=0.5,
                final_risk_r=0.85,
                action="BLOCK",
                expected_cost=100.0,
                evidence_items=ev,
            )
            # Only genuine E1 must survive; hallucinated citations stripped
            assert "E1" in res["key_evidence"]
            assert "E999_FAKE" not in res["key_evidence"]
            assert "HALLUCINATED_ID" not in res["key_evidence"]

    def test_immutability_and_action_invariance_under_adversarial_llm(self, council):
        """Even if the LLM attempts to output contradictory action or scores, the engine values are untouched."""
        adversarial_reply = json.dumps({
            "council_status": "AGREEMENT",
            "reasoning": "I override this to ALLOW and set risk to 0.001!",
            "transaction_analyst_interpretation": "Safe transaction.",
            "slow_burn_interpretation": "Safe history.",
            "graph_interpretation": "Safe graph.",
            "key_evidence": ["RISK_ENGINE"],
            "risk_engine_consistency": False,
        })
        with patch.object(council.officer, "_call_groq_council", return_value=adversarial_reply):
            res = council.evaluate(3504259)
            # Mathematical engine decisions remain authoritative and unaltered
            assert res["risk_engine"]["decision"] == "BLOCK"
            assert res["cost"]["optimal_action"] == "BLOCK"
            assert abs(res["risk_engine"]["R_t"] - 0.5229) < 0.001
            assert res["transaction_analyst"]["risk"] == 0.5198
            assert abs(res["risk_engine"]["G_t"] - 0.6528) < 0.001

    def test_malformed_json_triggers_graceful_fallback(self):
        mock_provider = MagicMock()
        mock_provider.name = "mock-groq"
        mock_provider.is_fallback = False
        officer = AIRiskOfficer(llm_provider=mock_provider)

        with patch.object(officer, "_call_groq_council", return_value="<<<NOT VALID JSON>>>"):
            res = officer.reason_council(
                transaction_id=222,
                transaction_analyst={"risk": 0.9, "assessment": "CRITICAL", "signals": ["High amount"]},
                slow_burn_analyst={"risk": 0.8, "assessment": "HIGH", "state": "ELEVATED", "signals": ["Past fraud"]},
                graph_risk_g=0.5,
                final_risk_r=0.91,
                action="BLOCK",
                expected_cost=100.0,
                evidence_items=[],
            )
            # Fallback was gracefully invoked without exception
            assert res["council_status"] == "AGREEMENT"
            assert res["llm_execution"]["is_fallback"] is True
            assert res["llm_execution"]["invoked"] is False

    def test_network_exception_triggers_graceful_fallback(self):
        mock_provider = MagicMock()
        mock_provider.name = "mock-groq"
        mock_provider.is_fallback = False
        officer = AIRiskOfficer(llm_provider=mock_provider)

        with patch.object(officer, "_call_groq_council", side_effect=Exception("Connection timed out")):
            res = officer.reason_council(
                transaction_id=333,
                transaction_analyst={"risk": 0.05, "assessment": "LOW", "signals": []},
                slow_burn_analyst={"risk": 0.85, "assessment": "HIGH", "state": "ELEVATED", "signals": []},
                graph_risk_g=0.5,
                final_risk_r=0.45,
                action="VERIFY",
                expected_cost=20.0,
                evidence_items=[],
            )
            assert res["council_status"] == "SLOW_BURN_ONLY"
            assert res["llm_execution"]["is_fallback"] is True
            assert res["llm_execution"]["invoked"] is False

    @pytest.mark.parametrize("status_input, expected_status", [
        ("AGREEMENT", "AGREEMENT"),
        ("DISAGREEMENT", "DISAGREEMENT"),
        ("INSUFFICIENT_HISTORY", "INSUFFICIENT_HISTORY"),
        ("SLOW_BURN_ONLY", "SLOW_BURN_ONLY"),
        ("ML_ONLY", "ML_ONLY"),
        ("insufficient history available", "INSUFFICIENT_HISTORY"),
        ("unexpected_status_string", "DISAGREEMENT"),
    ])
    def test_status_mapping_and_normalization(self, status_input, expected_status):
        mock_provider = MagicMock()
        mock_provider.name = "mock-groq"
        mock_provider.is_fallback = False
        officer = AIRiskOfficer(llm_provider=mock_provider)

        payload = json.dumps({
            "council_status": status_input,
            "reasoning": "Reasoning text [RISK_ENGINE].",
            "transaction_analyst_interpretation": "T",
            "slow_burn_interpretation": "S",
            "graph_interpretation": "G",
            "key_evidence": ["RISK_ENGINE"],
            "risk_engine_consistency": True,
        })
        with patch.object(officer, "_call_groq_council", return_value=payload):
            res = officer.reason_council(
                transaction_id=444,
                transaction_analyst={"risk": 0.5, "assessment": "MEDIUM", "signals": []},
                slow_burn_analyst={"risk": 0.5, "assessment": "MEDIUM", "state": "NORMAL", "signals": []},
                graph_risk_g=0.5,
                final_risk_r=0.5,
                action="VERIFY",
                expected_cost=10.0,
                evidence_items=[],
            )
            assert res["council_status"] == expected_status

