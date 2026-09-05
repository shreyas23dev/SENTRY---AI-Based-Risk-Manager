"""
validate_demo_cases.py — Strict Reproducibility & Integrity Validator (Phase 10 Pre-11)
========================================================================================

Verifies:
  1. Loads all 4 demo case files from data/demo_cases/.
  2. Resolves each transaction against the real held-out TEST dataset parquet.
  3. Verifies that the transaction actually exists in the real dataset.
  4. Verifies that it belongs to the intended split ('TEST').
  5. Runs the transaction through the existing frozen Sentinel pipeline.
  6. Recomputes all relevant outputs (A_t, P_t, G_t, R_t, Council, Decision).
  7. Compares the newly generated outputs against the stored demo_manifest.json.
  8. Fails loudly if anything changed unexpectedly.
  9. Confirms that no synthetic or fabricated transaction is used.
 10. Prints a clear validation summary table.
"""

import sys
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trustgraph.council.council import get_risk_council


def validate():
    print("=" * 88)
    print("SENTINEL DEMO CASE VALIDATION -- REAL DATASET REPRODUCIBILITY AUDIT")
    print("=" * 88)

    cases_dir = PROJECT_ROOT / "data" / "demo_cases"
    manifest_file = cases_dir / "demo_manifest.json"
    test_parquet = PROJECT_ROOT / "artifacts" / "risk" / "test_risk_scores.parquet"

    assert manifest_file.exists(), f"Manifest not found: {manifest_file}"
    assert test_parquet.exists(), f"Test parquet not found: {test_parquet}"

    # 1. Load manifest and test dataset
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    test_df = pd.read_parquet(test_parquet)
    total_test_rows = len(test_df)
    print(f"Held-out TEST dataset loaded: {total_test_rows:,} real transactions.")

    council = get_risk_council()
    results = []

    case_files = [
        "case_01_agreement_block.json",
        "case_02_slow_burn_disagreement.json",
        "case_03_insufficient_history.json",
        "case_04_clean_allow.json",
        "case_05_verify.json",
        "case_06_throttle.json",
    ]

    for cf in case_files:
        c_path = cases_dir / cf
        assert c_path.exists(), f"Case file missing: {c_path}"
        with open(c_path, "r", encoding="utf-8") as f:
            c_data = json.load(f)

        case_id = c_data["case_id"]
        tid = int(c_data["transaction_id"])

        # 2 & 3. Verify real dataset existence
        matches = test_df[test_df["TransactionID"] == tid]
        assert len(matches) == 1, f"Transaction {tid} NOT FOUND in real test dataset!"
        row = matches.iloc[0]

        # 4. Verify split and label integrity
        assert c_data["dataset_split"] == "TEST", f"Invalid split for {tid}"
        real_label = int(row["isFraud"])
        assert c_data["label"] == real_label, f"Label mismatch for {tid}: {c_data['label']} vs {real_label}"

        # 5 & 6. Run through actual frozen pipeline
        res = council.evaluate(tid)
        actual_at = res["risk_engine"]["A_t"]
        actual_gt = res["risk_engine"]["G_t"]
        actual_rt = res["risk_engine"]["R_t"]
        actual_pt = res["slow_burn_analyst"]["risk"]
        actual_decision = res["risk_engine"]["decision"]
        actual_status = res["council"]["status"]

        # 7 & 8. Compare against manifest
        manifest_match = [m for m in manifest["cases"] if m["case_id"] == case_id]
        assert len(manifest_match) == 1, f"Manifest entry missing for {case_id}"
        m_entry = manifest_match[0]

        assert m_entry["transaction_id"] == tid, f"Transaction ID mismatch in manifest"
        assert abs(m_entry["A_t"] - actual_at) < 1e-4, f"A_t changed: {m_entry['A_t']} vs {actual_at}"
        assert abs(m_entry["G_t"] - actual_gt) < 1e-4, f"G_t changed: {m_entry['G_t']} vs {actual_gt}"
        assert abs(m_entry["R_t"] - actual_rt) < 1e-4, f"R_t changed: {m_entry['R_t']} vs {actual_rt}"
        if m_entry["P_t"] is not None:
            assert abs(m_entry["P_t"] - actual_pt) < 1e-4, f"P_t changed: {m_entry['P_t']} vs {actual_pt}"
        else:
            assert actual_pt is None, f"Expected P_t=None for cold start, got {actual_pt}"

        assert m_entry["final_risk_engine_decision"] == actual_decision, f"Decision changed for {tid}"
        assert m_entry["risk_council_classification"] == actual_status, f"Status changed for {tid}"

        pt_str = f"{actual_pt:.2f}" if actual_pt is not None else "N/A"
        case_num = case_id.split("_")[1]
        results.append({
            "case": f"CASE {int(case_num)}",
            "txn": tid,
            "at": f"{actual_at:.2f}",
            "pt": pt_str,
            "gt": f"{actual_gt:.2f}",
            "rt": f"{actual_rt:.2f}",
            "council": actual_status,
            "decision": actual_decision,
            "status": "PASS",
        })

    # 10. Print clean summary table
    print("\n" + "-" * 88)
    print(f"{'CASE':<8} | {'TXN':<8} | {'A_t':<5} | {'P_t':<5} | {'G_t':<5} | {'R_t':<5} | {'COUNCIL':<21} | {'DECISION':<8} | {'STATUS':<6}")
    print("-" * 88)
    for r in results:
        print(f"{r['case']:<8} | {r['txn']:<8} | {r['at']:<5} | {r['pt']:<5} | {r['gt']:<5} | {r['rt']:<5} | {r['council']:<21} | {r['decision']:<8} | {r['status']:<6}")
    print("-" * 88)

    print("\n[VERIFIED] All 6 demo cases resolved against real test dataset rows.")
    print("[VERIFIED] Pipeline outputs strictly match stored manifest.")
    print("[VERIFIED] Zero synthetic/fabricated data used.")
    print("=" * 88)


if __name__ == "__main__":
    validate()
