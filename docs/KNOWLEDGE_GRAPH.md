# TRUSTGRAPH Phase 2: Point-in-Time Payment Knowledge Graph

## Overview

Phase 2 builds a **causal, temporally indexed payment knowledge graph** that tracks relationships between transaction entities and historical fraud patterns. This layer produces a continuous graph risk score $G_t \in [0.0, 1.0]$ for every transaction that captures relational intelligence that a transaction-level ML model cannot see.

---

## Graph Schema

### Node Types

| Node Type | Proxy Key | Populated Rate (TEST) | Description |
|:---|:---|:---:|:---|
| `Transaction` | `txn_{TransactionID}` | 100% | Individual payment event |
| `CustomerEntity` | `{card1}_{addr1}_{P_emaildomain}` | 100% | Composite customer identity proxy |
| `Card` | `card1` | 100% | Payment card identifier |
| `Device` | `DeviceInfo` | 17.0% | Device fingerprint |
| `Address` | `addr1` | 89.5% | Billing/shipping postal code |
| `Email` | `P_emaildomain` | 81.9% | Purchaser email domain |
| `Merchant` | `R_emaildomain` or `ProductCD` | 20.9% | Recipient or product domain |
| `Network` | `id_31` | 20.1% | Browser or OS agent string |

### Edge Types

| Edge | Direction | Source Field | Description |
|:---|:---|:---|:---|
| `MADE_BY` | Transaction → CustomerEntity | `card1+addr1+P_emaildomain` | Customer entity ownership |
| `USED_CARD` | Transaction → Card | `card1` | Payment card used |
| `USED_DEVICE` | Transaction → Device | `DeviceInfo` | Device fingerprint used |
| `SHIPPED_TO` | Transaction → Address | `addr1` | Billing address |
| `HAS_EMAIL` | Transaction → Email | `P_emaildomain` | Purchaser email domain |
| `SENT_TO` | Transaction → Merchant | `R_emaildomain` | Recipient domain |
| `ACCESSED_VIA` | Transaction → Network | `id_31` | Browser/OS agent |

---

## Customer Entity Key Construction

The primary entity proxy is derived by combining the most stable identifiers:

```
entity_id = f"{card1}_{addr1}_{P_emaildomain}"
```

**Fallback hierarchy:**
1. `{card1}_{addr1}_{P_emaildomain}` — all three available (most specific)
2. `{card1}_{addr1}` — email missing
3. `{card1}` — only card available
4. `"unresolved_{transaction_id}"` — no stable identifier

---

## Temporal Semantics

> **Every feature computed for transaction $t$ uses strictly only information available before $t$.**

### Causal Ingest Protocol

For every transaction processed in chronological order:

```
1. QUERY  → get_prior_entity_history(entity_id, timestamp_t)
2. QUERY  → get_prior_device_history(device_id, timestamp_t)
3. QUERY  → get_prior_sharing_counts(...)
4. QUERY  → get_prior_2hop_fraud_history(...)
5. COMPUTE → G_t = compute_graph_risk(features)
6. UPDATE → add_transaction(... is_fraud=label)  ← label ingested AFTER scoring
```

Step 6 only occurs during TRAIN/VALIDATION streaming. During TEST evaluation:
- `is_train=False`: test labels are **never ingested** into the graph.
- Historical state after TRAIN+VALIDATION provides the prior-knowledge baseline.

### O(1) State Design

Unlike bisect-based temporal queries, the engine maintains:
- **`entity_txn_count[entity_id]`**: cumulative transaction counter
- **`entity_fraud_count[entity_id]`**: cumulative fraud counter
- **`entity_recent_txns[entity_id]`**: sliding deque for velocity windows (1h, 24h)
- **`device_entities[device_id]`**: set of entity_ids that used the device
- **`device_fraud_txns[device_id]`**: list of `(txn_id, entity_id)` fraud records

Temporal isolation is guaranteed by calling the queries *before* calling `add_transaction()` — not by bisecting into time-indexed lists.

---

## Graph Feature Definitions

The following 20 features are extracted per transaction for downstream scoring:

| Feature | Type | Description |
|:---|:---:|:---|
| `prior_entity_txns` | int | Number of prior transactions by this entity (before t) |
| `prior_entity_frauds` | int | Number of confirmed frauds by this entity (before t) |
| `entity_fraud_rate` | float | Historical fraud rate for this entity (before t) |
| `entity_velocity_1h` | int | Transactions from this entity in the preceding 1 hour |
| `entity_velocity_24h` | int | Transactions from this entity in the preceding 24 hours |
| `device_id` | str | Device fingerprint (may be None) |
| `device_entity_count` | int | Number of distinct other entities sharing this device |
| `device_prior_txns` | int | Total prior transactions on this device |
| `device_prior_frauds` | int | Total confirmed frauds on this device |
| `device_fraud_rate` | float | Historical fraud rate across all device transactions |
| `device_velocity_24h` | int | Transactions from this device in the preceding 24 hours |
| `unusual_device_sharing` | int | 1 if device was used by ≥ 3 distinct entities |
| `card_entity_count` | int | Number of distinct other entities sharing this card |
| `address_entity_count` | int | Number of distinct other entities sharing this address |
| `network_entity_count` | int | Number of distinct other entities sharing this browser/OS |
| `hop2_linked_frauds` | int | Total confirmed frauds among 2-hop connected entities |
| `hop2_distinct_fraud_entities` | int | Count of distinct 2-hop linked entities with ≥ 1 prior fraud |
| `has_graph_context` | int | 1 if entity/device has any prior history; 0 for cold-start |
| `graph_risk` | float | Deterministic combined graph risk G_t ∈ [0, 1] |

---

## $G_t$ Risk Calculation Methodology

$G_t$ is a deterministic, weighted combination of four interpretable sub-signals:

$$G_t = w_e \cdot S_\text{entity} + w_d \cdot S_\text{device} + w_n \cdot S_\text{network} + w_v \cdot S_\text{velocity}$$

Where:
- $w_e = 0.40$ (entity historical fraud momentum)
- $w_d = 0.30$ (device contamination and multiplexing)
- $w_n = 0.15$ (2-hop network contamination)
- $w_v = 0.15$ (transaction velocity bursts)

### Component Formulas

**Entity Signal:**
$$S_\text{entity} = \min\!\left(1, \; 0.4 \cdot N_\text{entity\_frauds} + 0.6 \cdot r_\text{entity}\right)$$

**Device Signal:**
$$S_\text{device} = \min\!\left(1, \; 0.65 \cdot S_\text{dev\_fraud} + 0.35 \cdot S_\text{dev\_mult}\right)$$

where:
$$S_\text{dev\_fraud} = \min\!\left(1, \; 0.4 \cdot N_\text{dev\_frauds} + 0.6 \cdot r_\text{device}\right)$$
$$S_\text{dev\_mult} = \min\!\left(1, \; \frac{N_\text{dev\_entities}}{5}\right)$$

**Network Signal:**
$$S_\text{network} = \min\!\left(1, \; 0.3 \cdot N_\text{hop2\_frauds} + 0.35 \cdot N_\text{hop2\_fraud\_ents}\right)$$

**Velocity Signal:**
$$S_\text{velocity} = \min\!\left(1, \; 0.5 \cdot S_{1h} + 0.25 \cdot S_{24h} + 0.25 \cdot S_\text{dev24h}\right)$$

where:
$$S_{1h} = \min\!\left(1, \; \frac{\max(0, v_{1h} - 1)}{3}\right)$$

**Weight Tuning:** Weights were selected analytically based on expected real-time fraud ring behavior. No test-set optimization was performed. Weights should be formally tuned on VALIDATION in Phase 3.

---

## Leakage Prevention

### What is prevented:

| Risk | Mitigation |
|:---|:---|
| **Future fraud labels leaking into G_t** | Labels are added AFTER scoring. `is_train=False` never adds labels. |
| **Future transactions visible** | Entity and device counters are only incremented after the causal scoring step. |
| **Test-set distribution leakage** | Feature pipeline is fit on TRAIN only; all test queries use TRAIN-derived historical state. |
| **2-hop speculation ceiling** | `MAX_RING_DEGREE = 25` prevents generic hub nodes (e.g., popular browsers) being flagged as ring indicators. |

### What is documented as a known limitation:

- **Velocity deques**: For performance, the velocity sliding windows use a deque that is evicted lazily on each query. This means that if a transaction is queried for a timestamp far before the last event, the deque may include events outside the target window. This does not affect correctness because the streaming order guarantees `query(t) → add(t)` causality.
- **Card re-use across unrelated users**: `card_entity_count > 0` includes legitimate card re-use (e.g., household sharing). High-specificity flags like `unusual_device_sharing` are more diagnostic.

---

## Evaluation Results on Held-Out TEST ($N = 88,580$)

**Validation-Selected Threshold:** $\tau_G = 0.49$ (selected via F1 maximization on VALIDATION only)

### Standalone Graph Risk vs. XGBoost Baseline

| Metric | $A_t$ Only (XGBoost, $\tau=0.12$) | $G_t$ Only (Graph, $\tau=0.49$) |
|:---|:---:|:---:|
| **ROC-AUC** | **0.8892** | 0.7480 |
| **PR-AUC** | **0.5658** | 0.1101 |
| **Precision** | **73.02%** | 12.74% |
| **Recall** | 47.49% | **39.09%** |
| **F1-Score** | **0.5755** | 0.1921 |
| **False Positive Rate** | **0.63%** | 9.66% |
| **True Positives** | **1,464** | 1,205 |
| **False Positives** | **541** | 8,255 |
| **Fraud Capture** | 47.49% | 39.09% |
| **Fraud Enrichment** | **20.98x** | 3.66x |

> **Important interpretation:** $G_t$ standalone performance is intentionally lower precision than $A_t$ because $G_t$ is a network-based relational signal, not a transaction classifier. Its value is incremental enrichment when combined with $A_t$, not standalone discrimination.

### Analytical Combination ($0.70 \cdot A_t + 0.30 \cdot G_t$, illustrative only)

The analytical combination substantially increases recall (from 47.49% to 74.70%) but at the cost of a major increase in false positives (541 → 24,001). This demonstrates that $G_t$ carries complementary recall signal. Proper fusion weights should be tuned via the FUSION phase (Phase 3), not applied directly.

---

## Coverage & Cold-Start Analysis

| Metric | TEST ($N = 88,580$) |
|:---|:---:|
| **Graph Context Coverage** | **99.92%** |
| **Cold-Start Rate** | **0.08%** (≈ 71 transactions) |

Nearly all test transactions had prior history in the graph because TRAIN and VALIDATION span 15 months of data before the TEST partition boundary. Cold-start transactions are assigned $G_t = 0.0$.

---

## Sample Evidence Lookup

For transaction `3488979` (confirmed fraud, $G_t = 0.498$):

```json
{
  "transaction_id": 3488979,
  "timestamp": 13153151.0,
  "graph_risk": 0.4975,
  "entity_id": "14260",
  "connected_entities": {
    "entity": ["14260"],
    "cards": ["14260"],
    "addresses": []
  },
  "historical_summary": {
    "prior_entity_transactions": 76,
    "prior_entity_frauds": 6,
    "entity_fraud_rate": 0.0789,
    "entity_1h_velocity": 0,
    "entity_24h_velocity": 1,
    "device_prior_transactions": 0,
    "device_prior_frauds": 0,
    "device_connected_entities": 0,
    "hop2_linked_fraud_count": 1
  },
  "suspicious_relationships": [
    "Customer entity has 6 confirmed historical fraud transactions (fraud rate: 7.9%)",
    "1 confirmed fraud detected across 2-hop connected entity neighborhood"
  ],
  "evidence_paths": [
    {
      "path": "Txn 3488979 → Entity 14260 → Prior Txn 3488910 → Confirmed Fraud",
      "risk_contribution": 0.40,
      "description": "Direct entity recidivism: entity had confirmed fraud on transaction 3488910."
    }
  ],
  "risk_factors": [
    "Known fraudulent entity history (6 prior frauds)",
    "2-hop network contamination (1 linked entities with fraud)"
  ]
}
```

---

## Computational Observations

| Metric | Value |
|:---|:---:|
| **TRAIN streaming speed** | ~16,073 txns/s |
| **VALIDATION streaming speed** | ~20,190 txns/s |
| **TEST streaming speed** | ~17,930 txns/s |
| **TRAIN partition time** | 25.7 s (N = 413,379) |
| **VALIDATION partition time** | 4.4 s (N = 88,581) |
| **TEST partition time** | 4.9 s (N = 88,580) |
| **Single-transaction latency** | < 0.1 ms |
| **Total G_t evaluation wall time** | < 5 min (including dataset loading and XGBoost comparison) |

The O(1) lookup design enables production-grade throughput without an external graph database.

---

## Limitations & Phase 3 Considerations

1. **Fusion weights not tuned:** The analytical $0.70 A_t + 0.30 G_t$ combination is for analysis only. Phase 3 should formally optimize fusion weights on VALIDATION.
2. **Policy thresholds not adapted:** The progressive ALLOW/VERIFY/THROTTLE/BLOCK policy still uses LightGBM-era thresholds. These must be recalibrated for XGBoost+Graph probability scales.
3. **Entity key resolution depends on addr1 (88.5% populated):** When `addr1` is missing, entity disambiguation collapses to `card1` alone, increasing collisions between distinct cardholders.
4. **No persistent graph storage:** The graph state is in-memory. For production, this should be serialized to disk or a graph database (Neo4j, TigerGraph, Neptune).
5. **2-hop exploration only:** The current implementation uses 1-hop and 2-hop contamination signals. Phase 4 GraphRAG can extend this to multi-hop ring detection.
