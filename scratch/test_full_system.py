"""
test_full_system.py — Live End-to-End System Verification
=========================================================
Tests both live running servers:
  1. Backend API on http://localhost:8000
  2. Frontend Static Server on http://localhost:5173
"""

import json
import urllib.request
import urllib.error
import time

BACKEND_BASE = "http://localhost:8000"
FRONTEND_BASE = "http://localhost:5173"

def http_get(url):
    t0 = time.perf_counter()
    req = urllib.request.Request(url, headers={"User-Agent": "Sentinel-Audit/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = resp.read().decode("utf-8")
        latency = (time.perf_counter() - t0) * 1000
        return resp.status, body, latency

def http_post(url, data_dict):
    t0 = time.perf_counter()
    data_bytes = json.dumps(data_dict).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"User-Agent": "Sentinel-Audit/1.0", "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = resp.read().decode("utf-8")
        latency = (time.perf_counter() - t0) * 1000
        return resp.status, body, latency

def run_tests():
    print("=" * 80)
    print("  SENTINEL (AI Payment Risk Manager) — LIVE END-TO-END SYSTEM TEST")
    print("=" * 80)
    
    # 1. Health check
    print("\n[1] Testing Backend Health Check (/api/v1/health)...")
    status, body, lat = http_get(f"{BACKEND_BASE}/api/v1/health")
    data = json.loads(body)
    print(f"    -> HTTP {status} in {lat:.1f}ms | Status: {data.get('status')} | Service: {data.get('service')}")
    assert status == 200
    assert data.get("status") == "healthy"

    # 2. Demo transactions
    print("\n[2] Testing Demo Manifest Retrieval (/api/v1/investigation/demo-transactions)...")
    status, body, lat = http_get(f"{BACKEND_BASE}/api/v1/investigation/demo-transactions")
    demos = json.loads(body)
    print(f"    -> HTTP {status} in {lat:.1f}ms | Loaded {len(demos)} demo transaction cases")
    demo_ids = [d["transaction_id"] for d in demos]
    print(f"    -> Demo Transaction IDs: {demo_ids}")
    assert status == 200
    assert len(demos) >= 4

    # 3. Test Risk Scoring on each case
    print("\n[3] Testing Live Risk Engine Scoring across Key Cases...")
    test_cases = [3570805, 3504259, 3488964, 3512832]
    for txn_id in test_cases:
        status, body, lat = http_get(f"{BACKEND_BASE}/api/v1/risk/{txn_id}")
        r_data = json.loads(body)
        a_t = r_data.get("base_risk") if r_data.get("base_risk") is not None else r_data.get("A_t", 0.0)
        g_t = r_data.get("graph_risk") if r_data.get("graph_risk") is not None else r_data.get("G_t", 0.0)
        r_t = r_data.get("final_risk") if r_data.get("final_risk") is not None else r_data.get("R_t", 0.0)
        action = r_data.get("action", "UNKNOWN")
        amt = r_data.get("amount", 0.0)
        print(f"    -> Txn #{txn_id} (INR {amt:.2f}): Base_ML={a_t:.4f}, Graph_G={g_t:.4f}, Final_R={r_t:.4f} => [{action}] ({lat:.1f}ms)")
        assert status == 200
        assert action in ("ALLOW", "VERIFY", "THROTTLE", "BLOCK")

    # 4. Knowledge Graph Endpoints
    print("\n[4] Testing Knowledge Graph Topology & Relational Subgraph (/api/v1/risk/3570805/graph)...")
    status, body, lat = http_get(f"{BACKEND_BASE}/api/v1/risk/3570805/graph?max_hops=2")
    g_data = json.loads(body)
    nodes = g_data.get("nodes", [])
    edges = g_data.get("edges", [])
    categories = set(n.get("category") for n in nodes)
    print(f"    -> HTTP {status} in {lat:.1f}ms | Graph has {len(nodes)} nodes, {len(edges)} edges")
    print(f"    -> Entity Categories represented: {list(categories)}")
    assert status == 200
    assert len(nodes) > 0

    # 5. Evidence Items
    print("\n[5] Testing Grounded Evidence Extraction (/api/v1/risk/3570805/evidence)...")
    status, body, lat = http_get(f"{BACKEND_BASE}/api/v1/risk/3570805/evidence")
    e_data = json.loads(body)
    items = e_data if isinstance(e_data, list) else e_data.get("evidence", [])
    print(f"    -> HTTP {status} in {lat:.1f}ms | Generated {len(items)} evidence items:")
    for item in items[:3]:
        print(f"       * [{item.get('evidence_id')}] {item.get('title')} (Risk: {item.get('risk_weight')*100:.0f}%)")
    assert status == 200
    assert len(items) > 0

    # 6. Risk Council Multi-Agent Deliberation
    print("\n[6] Testing Risk Council Multi-Agent Deliberation (/api/v1/risk/3570805/council)...")
    status, body, lat = http_get(f"{BACKEND_BASE}/api/v1/risk/3570805/council")
    c_data = json.loads(body)
    t_analyst = c_data.get("transaction_analyst", {})
    s_analyst = c_data.get("slow_burn_analyst", {})
    council_box = c_data.get("council", {})
    print(f"    -> HTTP {status} in {lat:.1f}ms")
    print(f"    -> Transaction Analyst: {t_analyst.get('assessment')} (A_t={t_analyst.get('risk')})")
    print(f"    -> Slow-Burn Analyst: {s_analyst.get('assessment')} (P_t={s_analyst.get('risk')})")
    print(f"    -> Council Status: {council_box.get('status')}")
    print(f"    -> Presiding Officer Reasoning: {council_box.get('summary')[:120]}...")
    assert status == 200
    assert council_box.get("status") in ("AGREEMENT", "DISAGREEMENT", "SLOW_BURN_ONLY", "ML_ONLY", "INSUFFICIENT_HISTORY")

    # 7. AI Investigator Grounded Case Report
    print("\n[7] Testing AI Investigator Report (/api/v1/risk/3570805/investigate)...")
    status, body, lat = http_post(f"{BACKEND_BASE}/api/v1/risk/3570805/investigate?scenario=balanced", {})
    inv_data = json.loads(body)
    summary = inv_data.get("narrative_summary", "")
    reasons = inv_data.get("reasons", [])
    conf = inv_data.get("confidence", 0.95)
    print(f"    -> HTTP {status} in {lat:.1f}ms | Confidence: {conf*100:.0f}%")
    print(f"    -> Grounded Reasons Count: {len(reasons)}")
    for r in reasons[:2]:
        print(f"       * {r.get('statement')[:80]}... [Evidence: {r.get('evidence_ids')}]")
    assert status == 200
    assert len(reasons) > 0

    # 8. Interactive AI Q&A
    print("\n[8] Testing Live Interactive Q&A (/api/v1/risk/3570805/ask)...")
    q_payload = {"question": "Why was this transaction blocked?"}
    status, body, lat = http_post(f"{BACKEND_BASE}/api/v1/risk/3570805/ask", q_payload)
    q_data = json.loads(body)
    answer = q_data.get("answer", "")
    citations = q_data.get("cited_evidence_ids", [])
    grounded = q_data.get("grounded", True)
    print(f"    -> HTTP {status} in {lat:.1f}ms | Grounded: {grounded}")
    print(f"    -> Answer snippet: {answer[:130]}...")
    print(f"    -> Evidence citations: {citations}")
    assert status == 200
    assert len(citations) > 0

    # 9. Frontend Serving & Asset Verification
    print("\n[9] Testing Frontend Static Web Server (http://localhost:5173)...")
    pages = [
        ("index.html", "SENTRY", "REAL HELD-OUT TEST DATA"),
        ("overview.html", "SENTRY", "REAL HELD-OUT TEST DATA"),
        ("transactions.html", "SENTRY", "REAL HELD-OUT TEST DATA"),
        ("risk-engine.html", "SENTRY", "REAL HELD-OUT TEST DATA"),
        ("investigations.html", "SENTRY", "REAL HELD-OUT TEST DATA"),
    ]
    for page, brand_str, badge_str in pages:
        status, body, lat = http_get(f"{FRONTEND_BASE}/{page}")
        has_brand = brand_str in body
        has_badge = badge_str in body
        has_kaggle = "Kaggle" in body or "kaggle" in body
        print(f"    -> {page:18s} HTTP {status} in {lat:.1f}ms | Brand '{brand_str}': {has_brand} | Badge '{badge_str}': {has_badge} | Zero Kaggle: {not has_kaggle}")
        assert status == 200
        assert has_brand
        assert has_badge
        assert not has_kaggle, f"Found unexpected Kaggle in {page}"

    # 10. Assets
    print("\n[10] Testing Frontend Logo & Static JavaScript Modules...")
    assets = [
        "public/sentinel-shield.svg",
        "js/api.js",
        "js/state.js",
        "js/graph.js",
        "js/overview.js",
        "js/transactions.js",
        "js/risk-engine.js",
        "js/investigations.js",
        "js/demo-mode.js",
    ]
    for asset in assets:
        status, body, lat = http_get(f"{FRONTEND_BASE}/{asset}")
        print(f"    -> {asset:30s} HTTP {status} in {lat:.1f}ms (Length: {len(body)} B)")
        assert status == 200

    print("\n" + "=" * 80)
    print("  ALL 10 END-TO-END LIVE TESTS PASSED WITH ZERO ERRORS!")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
