import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from starlette.testclient import TestClient
from trustgraph.service.app import app

def audit_system():
    print("=" * 80)
    print("PHASE 6: BUILDATHON AUDIT & VALIDATION ON 10+ REAL TRANSACTIONS")
    print("=" * 80)

    client = TestClient(app)

    # 1. Retrieve Demo Transactions
    resp = client.get("/api/v1/investigation/demo-transactions")
    assert resp.status_code == 200
    demo_txns = resp.json()
    print(f"Retrieved {len(demo_txns)} core demo transactions.")

    # 2. Test 10 Transactions
    # We test demo transactions + additional transactions from risk scores matrix if available
    txn_ids_to_test = [d["transaction_id"] for d in demo_txns]
    # Add additional real IDs from fixtures
    additional_ids = [3000000, 3000001, 3000002, 3000003, 3000004, 3000005]
    for tid in additional_ids:
        if tid not in txn_ids_to_test:
            txn_ids_to_test.append(tid)

    tested_records = []
    print("\n[Auditing Individual Transactions]:")
    for tid in txn_ids_to_test[:10]:
        resp = client.get(f"/api/v1/risk/{tid}")
        assert resp.status_code == 200, f"Failed for {tid}"
        data = resp.json()

        # Get evidence
        ev_resp = client.get(f"/api/v1/risk/{tid}/evidence")
        ev_items = ev_resp.json() if ev_resp.status_code == 200 else []

        # Get graph
        g_resp = client.get(f"/api/v1/risk/{tid}/graph?max_hops=2")
        graph = g_resp.json() if g_resp.status_code == 200 else {"nodes": [], "edges": []}

        record = {
            "transaction_id": tid,
            "A_t": round(data["base_risk"] * 100, 2),
            "G_t": round(data["graph_risk"] * 100, 2),
            "R_t": round(data["final_risk"] * 100, 2),
            "amount": round(data["amount"], 2),
            "node_count": len(graph["nodes"]),
            "edge_count": len(graph["edges"]),
            "evidence_count": len(ev_items),
        }
        tested_records.append(record)
        print(f"  Txn #{tid}: A_t={record['A_t']}% | G_t={record['G_t']}% | R_t={record['R_t']}% | Amount=INR {record['amount']} | Nodes={record['node_count']} | Evidence={record['evidence_count']}")

    # 3. AI Question Quality Testing on 6 Specified Questions
    print("\n" + "=" * 80)
    print("TESTING AI INVESTIGATOR ON 6 SPECIFIED CORE QUESTIONS")
    print("=" * 80)

    target_id = 3504259
    test_questions = [
        "Why was this transaction blocked?",
        "What evidence contributed most to the decision?",
        "What graph relationships make this transaction suspicious?",
        "Is this entity associated with previous fraud?",
        "What would happen if graph risk were removed?",
        "Why is this transaction different from a legitimate transaction?"
    ]

    for q in test_questions:
        resp = client.post(f"/api/v1/risk/{target_id}/ask", json={"question": q})
        assert resp.status_code == 200, f"Ask query failed for: {q}"
        ans = resp.json()
        safe_ans = ans.get('answer', '').encode('ascii', errors='ignore').decode('ascii')
        citations = ans.get('cited_evidence_ids', [])
        grounded = ans.get('grounded', False)
        print(f"\nQ: \"{q}\"")
        print(f"A: {safe_ans[:140]}...")
        print(f"   [Citations]: {citations} | Grounded: {grounded}")

    print("\n" + "=" * 80)
    print("AUDIT & VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == '__main__':
    audit_system()
