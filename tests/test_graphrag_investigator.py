"""
test_graphrag_investigator.py — Unit & Integration Tests for Phase 4 GraphRAG
=============================================================================

Covers:
  1. Retrieval: correct transaction, neighbors, hop depth, ranking, provenance
  2. Grounding & Anti-Hallucination: evidence citations, rejection of out-of-domain facts
  3. Graph API: node types, edges, fraud markers, target flags
  4. Risk Integration: A_t, G_t, R_t, and Decision consistency
  5. Security: immutability, zero graph mutation, no secret leaks
"""

import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustgraph.graph.schema import EdgeType, NodeType
from trustgraph.graph.temporal_graph import PaymentKnowledgeGraph
from trustgraph.investigator.investigator import GraphRAGInvestigator
from trustgraph.investigator.llm_provider import DeterministicFallbackProvider
from trustgraph.investigator.retriever import EvidenceRetriever
from trustgraph.investigator.schema import EvidenceType
from trustgraph.investigator.service import InvestigatorService
from trustgraph.service.app import app


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def seeded_graph():
    """Build a deterministic test payment graph with known fraud patterns."""
    graph = PaymentKnowledgeGraph()

    # 1. Target Transaction 1001 (Fraud recidivist entity)
    graph.transactions[1001] = {
        "transaction_id": 1001,
        "timestamp": 100000.0,
        "amount": 250.0,
        "entity_id": "ent_bad_1",
        "device_id": "dev_bad_1",
        "card_id": "card_bad_1",
        "addr_id": "addr_90210",
        "email_id": "fraudmail.com",
    }
    graph.entity_txn_count["ent_bad_1"] = 12
    graph.entity_fraud_count["ent_bad_1"] = 3
    graph.entity_fraud_txns["ent_bad_1"] = [800, 850, 900]
    for past_id in [800, 850, 900]:
        graph.transactions[past_id] = {
            "transaction_id": past_id,
            "timestamp": 90000.0,
            "amount": 100.0,
            "entity_id": "ent_bad_1",
            "isFraud": 1,
        }

    # Device multiplexing on dev_bad_1
    graph.device_entities["dev_bad_1"].update(["ent_bad_1", "ent_syndicate_2", "ent_syndicate_3"])
    graph.device_fraud_count["dev_bad_1"] = 2
    graph.device_fraud_txns["dev_bad_1"] = [(850, "ent_bad_1"), (750, "ent_syndicate_2")]

    graph.card_entities["card_bad_1"].add("ent_bad_1")
    graph.entity_cards["ent_bad_1"].add("card_bad_1")

    # 2. Target Transaction 2002 (Clean legitimate user)
    graph.transactions[2002] = {
        "transaction_id": 2002,
        "timestamp": 100000.0,
        "amount": 50.0,
        "entity_id": "ent_clean_1",
        "device_id": "dev_clean_1",
        "card_id": "card_clean_1",
    }
    graph.entity_txn_count["ent_clean_1"] = 15
    graph.entity_fraud_count["ent_clean_1"] = 0
    graph.device_entities["dev_clean_1"].add("ent_clean_1")

    return graph


@pytest.fixture(scope="module")
def investigator_engine(seeded_graph):
    return GraphRAGInvestigator(graph=seeded_graph)


@pytest.fixture(scope="module")
def test_client():
    return TestClient(app)


# ===========================================================================
# 1. Retrieval Tests
# ===========================================================================

class TestEvidenceRetrieval:

    def test_retrieve_correct_transaction(self, seeded_graph):
        retriever = EvidenceRetriever(seeded_graph)
        items, view = retriever.retrieve_evidence(
            transaction_id=1001,
            base_risk=0.55,
            graph_risk=0.80,
            final_risk=0.58,
            action="BLOCK",
            expected_cost=150.0,
        )
        assert view.transaction_id == 1001
        assert any(n.id == "txn_1001" and n.is_target for n in view.nodes)

    def test_retrieve_1hop_neighbors(self, seeded_graph):
        retriever = EvidenceRetriever(seeded_graph)
        _, view = retriever.retrieve_evidence(
            transaction_id=1001,
            base_risk=0.5,
            graph_risk=0.5,
            final_risk=0.5,
            action="BLOCK",
            max_hops=1,
        )
        node_ids = {n.id for n in view.nodes}
        assert "ent_ent_bad_1" in node_ids
        assert "dev_dev_bad_1" in node_ids
        assert "card_card_bad_1" in node_ids
        # In 1-hop, past fraud transaction nodes should not be present
        assert not any("past_fraud_" in nid for nid in node_ids)

    def test_retrieve_2hop_neighbors(self, seeded_graph):
        retriever = EvidenceRetriever(seeded_graph)
        items, view = retriever.retrieve_evidence(
            transaction_id=1001,
            base_risk=0.5,
            graph_risk=0.5,
            final_risk=0.5,
            action="BLOCK",
            max_hops=2,
        )
        node_ids = {n.id for n in view.nodes}
        # 2-hop should include past fraud transactions
        assert any("past_fraud_" in nid for nid in node_ids)
        assert len(view.suspicious_paths) > 0

    def test_evidence_ranking(self, seeded_graph):
        retriever = EvidenceRetriever(seeded_graph)
        items, _ = retriever.retrieve_evidence(
            transaction_id=1001,
            base_risk=0.5,
            graph_risk=0.8,
            final_risk=0.6,
            action="BLOCK",
        )
        weights = [e.risk_weight for e in items]
        # Must be monotonically non-increasing (ranked by weight descending)
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1]

    def test_provenance_attached(self, seeded_graph):
        retriever = EvidenceRetriever(seeded_graph)
        items, _ = retriever.retrieve_evidence(
            transaction_id=1001,
            base_risk=0.5,
            graph_risk=0.8,
            final_risk=0.6,
            action="BLOCK",
        )
        for e in items:
            assert isinstance(e.provenance, dict)
            assert e.evidence_id.startswith("E") or e.evidence_id == "RISK_ENGINE"


# ===========================================================================
# 2. Grounding & Anti-Hallucination Tests
# ===========================================================================

class TestGroundingAndAntiHallucination:

    def test_every_reason_has_valid_citation(self, investigator_engine):
        report, _ = investigator_engine.investigate_transaction(
            transaction_id=1001,
            base_risk=0.52,
            graph_risk=0.65,
            final_risk=0.53,
            action="BLOCK",
            expected_cost=200.0,
        )
        valid_ids = {e.evidence_id for e in report.evidence_items}
        for r in report.reasons:
            assert len(r.evidence_ids) > 0
            for cid in r.evidence_ids:
                assert cid in valid_ids

    def test_out_of_domain_query_rejected(self, investigator_engine):
        ans = investigator_engine.ask_question(
            transaction_id=1001,
            question="What is the cardholder's favorite restaurant and weather outside?",
            base_risk=0.5,
            graph_risk=0.5,
            final_risk=0.5,
            action="BLOCK",
        )
        assert "insufficient evidence" in ans.answer.lower()
        assert not ans.grounded
        assert len(ans.cited_evidence_ids) == 0

    def test_unsupported_factual_claim_rejected(self, investigator_engine):
        ans = investigator_engine.ask_question(
            transaction_id=1001,
            question="What is the customer's bank balance and credit score?",
            base_risk=0.5,
            graph_risk=0.5,
            final_risk=0.5,
            action="BLOCK",
        )
        assert "insufficient evidence" in ans.answer.lower()

    def test_deterministic_output_stability(self, investigator_engine):
        rep1, _ = investigator_engine.investigate_transaction(
            transaction_id=1001, base_risk=0.5, graph_risk=0.7, final_risk=0.55, action="BLOCK", force_refresh=True
        )
        rep2, _ = investigator_engine.investigate_transaction(
            transaction_id=1001, base_risk=0.5, graph_risk=0.7, final_risk=0.55, action="BLOCK", force_refresh=True
        )
        assert len(rep1.reasons) == len(rep2.reasons)
        for r1, r2 in zip(rep1.reasons, rep2.reasons):
            assert r1.statement == r2.statement
            assert r1.evidence_ids == r2.evidence_ids


# ===========================================================================
# 3. Graph API & Visualization Schema Tests
# ===========================================================================

class TestGraphApiSchema:

    def test_node_types_valid(self, seeded_graph):
        retriever = EvidenceRetriever(seeded_graph)
        _, view = retriever.retrieve_evidence(
            transaction_id=1001, base_risk=0.5, graph_risk=0.5, final_risk=0.5, action="BLOCK"
        )
        valid_types = {t.value for t in NodeType}
        for n in view.nodes:
            assert n.node_type in valid_types

    def test_edge_types_valid(self, seeded_graph):
        retriever = EvidenceRetriever(seeded_graph)
        _, view = retriever.retrieve_evidence(
            transaction_id=1001, base_risk=0.5, graph_risk=0.5, final_risk=0.5, action="BLOCK"
        )
        valid_edges = {
            EdgeType.MADE_BY.value, EdgeType.USED_CARD.value, EdgeType.USED_DEVICE.value,
            EdgeType.SHIPPED_TO.value, EdgeType.HAS_EMAIL.value, EdgeType.SENT_TO.value,
            EdgeType.ACCESSED_VIA.value, "PRIOR_FRAUD_BY", "DEVICE_FRAUD_LINK", "SHARED_DEVICE_BY"
        }
        for e in view.edges:
            assert e.edge_type in valid_edges

    def test_fraud_nodes_marked(self, seeded_graph):
        retriever = EvidenceRetriever(seeded_graph)
        _, view = retriever.retrieve_evidence(
            transaction_id=1001, base_risk=0.5, graph_risk=0.5, final_risk=0.5, action="BLOCK"
        )
        fraud_nodes = [n for n in view.nodes if n.is_fraud]
        assert len(fraud_nodes) > 0
        for fn in fraud_nodes:
            assert fn.risk_score >= 0.7 or fn.properties.get("isFraud") == 1


# ===========================================================================
# 4. Risk Integration Tests
# ===========================================================================

class TestRiskIntegration:

    def test_math_engine_evidence_present(self, seeded_graph):
        retriever = EvidenceRetriever(seeded_graph)
        items, _ = retriever.retrieve_evidence(
            transaction_id=1001,
            base_risk=0.5198,
            graph_risk=0.6528,
            final_risk=0.5229,
            action="BLOCK",
            expected_cost=238.5,
            beta=0.05,
        )
        math_items = [e for e in items if e.evidence_id == "RISK_ENGINE"]
        assert len(math_items) == 1
        p = math_items[0].provenance
        assert p["A_t"] == 0.5198
        assert p["G_t"] == 0.6528
        assert p["R_t"] == 0.5229
        assert p["formula"] == "F2_conditional"
        assert p["action"] == "BLOCK"

    def test_clean_transaction_gets_allow(self, seeded_graph):
        retriever = EvidenceRetriever(seeded_graph)
        items, view = retriever.retrieve_evidence(
            transaction_id=2002,
            base_risk=0.001,
            graph_risk=0.0,
            final_risk=0.001,
            action="ALLOW",
        )
        assert not any(n.is_fraud for n in view.nodes)
        assert any(e.evidence_type == EvidenceType.CARD_SHARING for e in items)


# ===========================================================================
# 5. Security & Immutability Tests
# ===========================================================================

class TestSecurityAndImmutability:

    def test_retrieval_does_not_mutate_graph(self, seeded_graph):
        initial_node_count = len(seeded_graph.nodes)
        initial_fraud_count = seeded_graph.total_fraud_events_registered

        retriever = EvidenceRetriever(seeded_graph)
        for _ in range(5):
            retriever.retrieve_evidence(
                transaction_id=1001, base_risk=0.5, graph_risk=0.5, final_risk=0.5, action="BLOCK"
            )

        assert len(seeded_graph.nodes) == initial_node_count
        assert seeded_graph.total_fraud_events_registered == initial_fraud_count

    def test_llm_cannot_modify_risk_score(self, investigator_engine):
        report, _ = investigator_engine.investigate_transaction(
            transaction_id=1001,
            base_risk=0.45,
            graph_risk=0.30,
            final_risk=0.46,
            action="VERIFY",
            expected_cost=120.0,
        )
        # Final risk must strictly match what was passed from math engine
        assert report.final_risk_r == 0.46
        assert report.action == "VERIFY"

    def test_no_secrets_in_api_response(self, test_client):
        res = test_client.get("/api/v1/risk/3504259/investigate")
        if res.status_code == 200:
            data_str = res.text
            assert "api_key" not in data_str.lower()
            assert "secret" not in data_str.lower()
            assert "token" not in data_str.lower()


# ===========================================================================
# 6. Full FastAPI Endpoint Integration Tests
# ===========================================================================

class TestFastApiInvestigationEndpoints:

    def test_list_demo_transactions(self, test_client):
        res = test_client.get("/api/v1/investigation/demo-transactions")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        assert len(data) >= 4
        ids = [d["transaction_id"] for d in data]
        assert 3504259 in ids

    def test_get_risk_summary_endpoint(self, test_client):
        res = test_client.get("/api/v1/risk/3504259")
        assert res.status_code == 200
        data = res.json()
        assert data["transaction_id"] == 3504259
        assert "base_risk" in data
        assert "final_risk" in data

    def test_get_graph_endpoint(self, test_client):
        res = test_client.get("/api/v1/risk/3504259/graph?max_hops=2")
        assert res.status_code == 200
        data = res.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) > 0

    def test_get_evidence_endpoint(self, test_client):
        res = test_client.get("/api/v1/risk/3504259/evidence")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)
        assert len(data) > 0
        assert "evidence_id" in data[0]

    def test_investigate_endpoint(self, test_client):
        res = test_client.get("/api/v1/risk/3504259/investigate")
        assert res.status_code == 200
        data = res.json()
        assert data["action"] == "BLOCK"
        assert len(data["reasons"]) > 0
        assert "confidence" in data

    def test_ask_endpoint_valid_question(self, test_client):
        res = test_client.post(
            "/api/v1/risk/3504259/ask",
            json={"question": "Why was this transaction blocked?"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["grounded"] is True
        assert len(data["cited_evidence_ids"]) > 0

    def test_ask_endpoint_unsupported_question(self, test_client):
        res = test_client.post(
            "/api/v1/risk/3504259/ask",
            json={"question": "What was the weather?"},
        )
        assert res.status_code == 200
        data = res.json()
        assert "insufficient evidence" in data["answer"].lower()

    def test_serve_dashboard_html(self, test_client):
        res = test_client.get("/dashboard")
        assert res.status_code == 200
        assert "<title>TRUSTGRAPH" in res.text
        assert "d3.v7.min.js" in res.text
