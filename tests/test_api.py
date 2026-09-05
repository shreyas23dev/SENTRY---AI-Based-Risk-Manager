"""
test_api.py — Unit and Integration Tests for TRUSTGRAPH Risk Decision API
==========================================================================

Covers all 8 required engineering test categories:
  1. ALLOW decision evaluation
  2. VERIFY decision evaluation
  3. THROTTLE decision evaluation
  4. BLOCK decision evaluation
  5. Missing graph context (safe degradation when DeviceInfo is missing)
  6. Malformed request validation (422 response)
  7. Deterministic repeated evaluation
  8. Explanation consistency with signals
Plus:
  - GET /api/v1/health
  - GET /api/v1/risk/transactions/{transaction_id} (found and 404)
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from starlette.testclient import TestClient

from trustgraph.service.app import app, get_engine_service


FIXTURES_PATH = Path(__file__).resolve().parent / "fixtures_sample_txns.json"


@pytest.fixture(scope="module")
def fixtures_data():
    """Load sample transaction fixtures keyed by transaction ID."""
    if not FIXTURES_PATH.exists():
        pytest.skip("Fixture file tests/fixtures_sample_txns.json not found")
    with open(FIXTURES_PATH) as f:
        rows = json.load(f)
    out = {}
    for r in rows:
        clean = {k: v for k, v in r.items() if v is not None and str(v) != "nan"}
        out[int(clean["TransactionID"])] = clean
    return out


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with pre-loaded engine service."""
    _ = get_engine_service()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health Check Test
# ---------------------------------------------------------------------------

def test_health_check_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["engine"] == "TRUSTGRAPH Unified Risk Decision Engine"
    assert data["model_readiness"]["baseline_model_loaded"] is True
    assert data["model_readiness"]["temporal_engine_ready"] is True
    assert data["model_readiness"]["relational_graph_ready"] is True
    assert data["parameters"]["baseline_threshold"] == 0.594298
    assert data["parameters"]["policy_thresholds"]["tau_verify"] == 0.60
    assert data["parameters"]["policy_thresholds"]["tau_throttle"] == 0.65
    assert data["parameters"]["policy_thresholds"]["tau_block"] == 0.80


# ---------------------------------------------------------------------------
# 1. ALLOW Decision Test (< 0.60)
# ---------------------------------------------------------------------------

def test_evaluate_allow_decision(client, fixtures_data):
    """Clean, low-risk transaction yields ALLOW decision and LOW risk level."""
    row = fixtures_data[3488960]
    payload = {
        "transaction_id": "tx_allow_3488960",
        "transaction_dt": row.get("TransactionDT", 100.0),
        "amount": row["TransactionAmt"],
        "card1": row.get("card1"),
        "card2": row.get("card2"),
        "addr1": row.get("addr1"),
        "P_emaildomain": row.get("P_emaildomain"),
        "DeviceInfo": row.get("DeviceInfo"),
        "features": row,
    }
    response = client.post("/api/v1/risk/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == "tx_allow_3488960"
    assert data["risk_score"] < 0.60
    assert data["risk_level"] == "LOW"
    assert data["decision"] == "ALLOW"
    assert "baseline_risk" in data["signals"]
    assert "temporal_risk" in data["signals"]
    assert "graph_risk" in data["signals"]
    assert "fusion_risk" in data["signals"]
    assert isinstance(data["explanation"], list)
    assert len(data["explanation"]) > 0


# ---------------------------------------------------------------------------
# 2. VERIFY Decision Test (0.60 <= R < 0.65)
# ---------------------------------------------------------------------------

def test_evaluate_verify_decision(client, fixtures_data):
    """Transaction landing in [0.60, 0.65) triggers VERIFY decision and MEDIUM risk level."""
    row = fixtures_data[3489013]
    payload = {
        "transaction_id": "tx_verify_3489013",
        "transaction_dt": row.get("TransactionDT", 200.0),
        "amount": row["TransactionAmt"],
        "card1": row.get("card1"),
        "card2": row.get("card2"),
        "addr1": row.get("addr1"),
        "P_emaildomain": row.get("P_emaildomain"),
        "DeviceInfo": row.get("DeviceInfo"),
        "features": row,
    }
    response = client.post("/api/v1/risk/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == "tx_verify_3489013"
    assert 0.60 <= data["risk_score"] < 0.65
    assert data["decision"] == "VERIFY"
    assert data["risk_level"] == "MEDIUM"


# ---------------------------------------------------------------------------
# 3. THROTTLE Decision Test (0.65 <= R < 0.80)
# ---------------------------------------------------------------------------

def test_evaluate_throttle_decision(client, fixtures_data):
    """Transaction landing in [0.65, 0.80) triggers THROTTLE decision and HIGH risk level."""
    row = fixtures_data[3489068]
    payload = {
        "transaction_id": "tx_throttle_3489068",
        "transaction_dt": row.get("TransactionDT", 300.0),
        "amount": row["TransactionAmt"],
        "card1": row.get("card1"),
        "card2": row.get("card2"),
        "addr1": row.get("addr1"),
        "P_emaildomain": row.get("P_emaildomain"),
        "DeviceInfo": row.get("DeviceInfo"),
        "features": row,
    }
    response = client.post("/api/v1/risk/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == "tx_throttle_3489068"
    assert 0.65 <= data["risk_score"] < 0.80
    assert data["decision"] == "THROTTLE"
    assert data["risk_level"] == "HIGH"


# ---------------------------------------------------------------------------
# 4. BLOCK Decision Test (R >= 0.80)
# ---------------------------------------------------------------------------

def test_evaluate_block_decision(client, fixtures_data):
    """High risk transaction triggers BLOCK decision and CRITICAL risk level."""
    row = fixtures_data[3489048]
    payload = {
        "transaction_id": "tx_block_3489048",
        "transaction_dt": row.get("TransactionDT", 400.0),
        "amount": row["TransactionAmt"],
        "card1": row.get("card1"),
        "card2": row.get("card2"),
        "addr1": row.get("addr1"),
        "P_emaildomain": row.get("P_emaildomain"),
        "DeviceInfo": row.get("DeviceInfo"),
        "features": row,
    }
    response = client.post("/api/v1/risk/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == "tx_block_3489048"
    assert data["risk_score"] >= 0.80
    assert data["decision"] == "BLOCK"
    assert data["risk_level"] == "CRITICAL"
    explanation_str = " ".join(data["explanation"])
    assert "risk" in explanation_str.lower()


# ---------------------------------------------------------------------------
# 5. Missing Graph Context Test (DeviceInfo is None / absent)
# ---------------------------------------------------------------------------

def test_evaluate_missing_graph_context(client):
    """Engine gracefully handles missing DeviceInfo without throwing errors."""
    payload = {
        "transaction_id": "tx_missing_graph_001",
        "transaction_dt": 20000.0,
        "amount": 49.95,
        "card1": 12345,
        "addr1": 120,
        "P_emaildomain": "yahoo.com",
        # DeviceInfo explicitly omitted
    }
    response = client.post("/api/v1/risk/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["signals"]["graph_risk"] == 0.0
    assert data["metadata"]["device_connected_entities"] == 0
    assert 0.0 <= data["risk_score"] <= 1.0


# ---------------------------------------------------------------------------
# 6. Malformed Request Validation Tests (HTTP 422)
# ---------------------------------------------------------------------------

def test_malformed_request_negative_amount(client):
    """Negative amount must fail validation with 422."""
    payload = {
        "transaction_id": "tx_bad_01",
        "amount": -50.0,
    }
    response = client.post("/api/v1/risk/evaluate", json=payload)
    assert response.status_code == 422


def test_malformed_request_zero_amount(client):
    """Zero amount must fail validation with 422."""
    payload = {
        "transaction_id": "tx_bad_02",
        "amount": 0.0,
    }
    response = client.post("/api/v1/risk/evaluate", json=payload)
    assert response.status_code == 422


def test_malformed_request_missing_id(client):
    """Missing transaction_id must fail validation with 422."""
    payload = {
        "amount": 100.0,
    }
    response = client.post("/api/v1/risk/evaluate", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 7. Deterministic Repeated Evaluation Test
# ---------------------------------------------------------------------------

def test_deterministic_evaluation(client):
    """Two identical isolated evaluation calls produce identical point-wise baseline risk."""
    payload1 = {
        "transaction_id": "tx_det_001",
        "transaction_dt": 30000.0,
        "amount": 125.0,
        "card1": 77777,
        "addr1": 888,
        "P_emaildomain": "hotmail.com",
    }
    payload2 = {
        "transaction_id": "tx_det_002",
        "transaction_dt": 30001.0,
        "amount": 125.0,
        "card1": 77777,
        "addr1": 888,
        "P_emaildomain": "hotmail.com",
    }
    res1 = client.post("/api/v1/risk/evaluate", json=payload1).json()
    res2 = client.post("/api/v1/risk/evaluate", json=payload2).json()

    assert "signals" in res1 and "signals" in res2
    assert res1["signals"]["baseline_risk"] == pytest.approx(res2["signals"]["baseline_risk"], abs=1e-5)


# ---------------------------------------------------------------------------
# 8. Explanation Consistency Test
# ---------------------------------------------------------------------------

def test_explanation_consistency(client):
    """Every explanation string maps strictly to true active signals."""
    payload = {
        "transaction_id": "tx_exp_001",
        "transaction_dt": 40000.0,
        "amount": 15.0,
        "card1": 11223,
        "addr1": 101,
        "P_emaildomain": "gmail.com",
    }
    res = client.post("/api/v1/risk/evaluate", json=payload).json()
    assert "explanation" in res
    reasons = res["explanation"]
    assert isinstance(reasons, list)
    assert len(reasons) >= 1
    for r in reasons:
        assert isinstance(r, str)
        assert len(r) > 5


# ---------------------------------------------------------------------------
# 9. GET /api/v1/risk/transactions/{transaction_id} Tests
# ---------------------------------------------------------------------------

def test_get_transaction_by_id_found(client):
    """Successfully query an evaluated transaction by its ID."""
    txn_id = "tx_query_test_456"
    payload = {
        "transaction_id": txn_id,
        "transaction_dt": 50000.0,
        "amount": 75.0,
        "card1": 22334,
        "addr1": 150,
        "P_emaildomain": "icloud.com",
    }
    eval_res = client.post("/api/v1/risk/evaluate", json=payload).json()

    # Query by ID
    get_res = client.get(f"/api/v1/risk/transactions/{txn_id}")
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["transaction_id"] == txn_id
    assert data["risk_score"] == eval_res["risk_score"]
    assert data["decision"] == eval_res["decision"]
    assert data["signals"] == eval_res["signals"]


def test_get_transaction_by_id_not_found(client):
    """Unknown transaction ID returns 404 Not Found."""
    get_res = client.get("/api/v1/risk/transactions/non_existent_tx_999999")
    assert get_res.status_code == 404
    err = get_res.json()
    assert "not found" in err["detail"].lower()
