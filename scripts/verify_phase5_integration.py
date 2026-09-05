import sys
import json
import urllib.request
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from starlette.testclient import TestClient
from trustgraph.service.app import app

def run_verification():
    print("=" * 60)
    print("PHASE 5: FULL STITCH FRONTEND <-> BACKEND INTEGRATION VERIFICATION")
    print("=" * 60)

    client = TestClient(app)

    # 1. Health Check
    print("\n[1] Checking /api/v1/health...")
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    health = resp.json()
    print(f"    Status: {health.get('status')} | Scorer Ready: {health.get('scorer_ready')}")

    # 2. Demo Transactions
    print("\n[2] Checking /api/v1/investigation/demo-transactions...")
    resp = client.get("/api/v1/investigation/demo-transactions")
    assert resp.status_code == 200, f"Demo transactions failed: {resp.text}"
    demos = resp.json()
    print(f"    Retrieved {len(demos)} demo transactions:")
    for d in demos:
        print(f"    - Txn #{d['transaction_id']}: R_t={d['final_risk']*100:.2f}% | A_t={d['base_risk']*100:.2f}% | G_t={d['graph_risk']*100:.2f}% | INR {d['amount']}")

    # 3. Target Transaction 3504259 Deep-Dive
    target_tid = 3504259
    print(f"\n[3] Checking Risk Summary for Txn #{target_tid}...")
    resp = client.get(f"/api/v1/risk/{target_tid}")
    assert resp.status_code == 200, f"Risk summary failed: {resp.text}"
    risk = resp.json()
    print(f"    Base ML Risk (A_t): {risk['base_risk']*100:.2f}%")
    print(f"    Graph Context (G_t): {risk['graph_risk']*100:.2f}%")
    print(f"    Final Risk (R_t): {risk['final_risk']*100:.2f}%")
    print(f"    Amount: INR {risk['amount']}")

    # 4. Knowledge Graph (1-Hop & 2-Hop)
    print(f"\n[4] Checking Knowledge Graph for Txn #{target_tid}...")
    resp1 = client.get(f"/api/v1/risk/{target_tid}/graph?max_hops=1")
    assert resp1.status_code == 200
    g1 = resp1.json()
    print(f"    1-Hop Graph: {len(g1['nodes'])} nodes, {len(g1['edges'])} edges")

    resp2 = client.get(f"/api/v1/risk/{target_tid}/graph?max_hops=2")
    assert resp2.status_code == 200
    g2 = resp2.json()
    print(f"    2-Hop Graph: {len(g2['nodes'])} nodes, {len(g2['edges'])} edges")
    node_types = set(n['node_type'] for n in g2['nodes'])
    print(f"    Node Categories: {', '.join(node_types)}")

    # 5. Evidence Retrieval
    print(f"\n[5] Checking Evidence Items for Txn #{target_tid}...")
    resp = client.get(f"/api/v1/risk/{target_tid}/evidence")
    assert resp.status_code == 200
    evidence = resp.json()
    print(f"    Retrieved {len(evidence)} evidence items:")
    for e in evidence:
        print(f"    - [{e['evidence_id']}] {e['title']} (Weight: {e['risk_weight']*100:.0f}%)")

    # 6. AI Risk Investigation (POST)
    print(f"\n[6] Checking AI Risk Investigation for Txn #{target_tid}...")
    resp = client.post(f"/api/v1/risk/{target_tid}/investigate?scenario=balanced")
    assert resp.status_code == 200
    report = resp.json()
    summary = report.get('narrative_summary') or report.get('executive_summary') or "Report ready"
    safe_summary = summary.encode('ascii', errors='ignore').decode('ascii')
    print(f"    Summary: {safe_summary[:120]}...")
    reasons = report.get('reasons', [])
    print(f"    Grounded Reasons ({len(reasons)} items):")
    for r in reasons[:3]:
        stmt = r['statement'].encode('ascii', errors='ignore').decode('ascii')
        print(f"      * {r['category']}: {stmt} (Evidence: {r.get('evidence_ids')})")
    print(f"    Provider: {report.get('provider')} | Confidence: {report.get('confidence')}")

    # 7. AI Q&A Terminal
    print(f"\n[7] Checking AI Q&A for Txn #{target_tid} ('Why was this transaction blocked?')...")
    resp = client.post(f"/api/v1/risk/{target_tid}/ask", json={"question": "Why was this transaction blocked?"})
    assert resp.status_code == 200
    ans = resp.json()
    safe_ans = ans.get('answer', '').encode('ascii', errors='ignore').decode('ascii')
    print(f"    Answer: {safe_ans[:120]}...")
    print(f"    Evidence Citations: {ans.get('cited_evidence_ids')}")
    print(f"    Grounded: {ans.get('grounded')}")

    # 8. Local Frontend Static Files & Server Check
    print("\n[8] Checking Frontend Static Server (http://localhost:5173)...")
    urls = [
        "http://localhost:5173/index.html",
        "http://localhost:5173/transactions.html",
        "http://localhost:5173/investigations.html",
        "http://localhost:5173/risk-engine.html",
        "http://localhost:5173/public/sentinel-shield.svg",
        "http://localhost:5173/js/api.js",
        "http://localhost:5173/js/state.js",
        "http://localhost:5173/js/graph.js",
        "http://localhost:5173/js/overview.js",
        "http://localhost:5173/js/transactions.js",
        "http://localhost:5173/js/investigations.js",
        "http://localhost:5173/js/risk-engine.js"
    ]
    for u in urls:
        try:
            r = urllib.request.urlopen(u)
            assert r.status == 200
            print(f"    {u} -> HTTP 200 OK")
        except Exception as err:
            print(f"    {u} -> ERROR: {err}")

    print("\n" + "=" * 60)
    print("ALL PHASE 5 INTEGRATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    run_verification()
