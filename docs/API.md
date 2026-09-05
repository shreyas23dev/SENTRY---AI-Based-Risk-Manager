# TRUSTGRAPH: Transaction Risk Decision API Specification
**Version:** `1.0.0`  
**Base Path:** `/api/v1`  
**Protocol:** HTTP/1.1 REST JSON  
**Engine:** TRUSTGRAPH Unified Risk Engine (Frozen $M_0$ Fusion Formulation)

---

## 1. Overview & Architecture

The TRUSTGRAPH Risk Decision API productizes the frozen TRUSTGRAPH machine-learning and graph-intelligence pipeline for real-time payment gateway integration (e.g. Razorpay Buildathon).

The service evaluates transactions synchronously with low latency ($< 10\text{ ms}$ processing time), combining:
1. **$A_t$ (Baseline Tabular Risk):** Point-wise LightGBM prediction over 432 features.
2. **$P_t$ (Temporal Entity Risk):** Causal Exponential Moving Average and velocity burst memory.
3. **$G_t$ (Relational Graph Risk):** Bipartite device-sharing graph evaluating entity multiplexing ($d_t$) and burst velocity ($v_t$).
4. **$R_t$ (Fused Risk Formulation $M_0$):**
   $$R_t = \text{clip}(A_t + 1.0 \cdot P_t + 0.05 \cdot G_t, 0.0, 1.0)$$
5. **Progressive Policy Action & Explanation:** Causal routing into 4 operational tiers with auditable signal-backed explanations.

---

## 2. API Endpoints

### 2.1 Health Check
- **Endpoint:** `GET /api/v1/health`
- **Description:** Verifies service health, component readiness flags, and active frozen parameter configuration.
- **Response Status:** `200 OK`

#### Response Example
```json
{
  "status": "healthy",
  "engine": "TRUSTGRAPH Unified Risk Decision Engine",
  "version": "1.0.0",
  "model_readiness": {
    "baseline_model_loaded": true,
    "preprocessor_loaded": true,
    "temporal_engine_ready": true,
    "relational_graph_ready": true,
    "policy_thresholds_loaded": true
  },
  "parameters": {
    "baseline_threshold": 0.594298,
    "policy_thresholds": {
      "tau_verify": 0.60,
      "tau_throttle": 0.65,
      "tau_block": 0.80
    },
    "fusion_rule": "M0: clip(A_t + 1.0 * P_t + 0.05 * G_t, 0.0, 1.0)"
  },
  "stored_transactions": 4
}
```

---

### 2.2 Evaluate Transaction Risk
- **Endpoint:** `POST /api/v1/risk/evaluate`
- **Description:** Evaluates risk for an incoming payment transaction. Supports both top-level payment entity attributes and optional IEEE-CIS tabular features.
- **Response Status:** `200 OK` (or `422 Unprocessable Content` on validation failure)

#### Request Schema (`application/json`)
| Field | Type | Required | Description |
|:---|:---|:---:|:---|
| `transaction_id` | `string \| integer` | **Yes** | Unique transaction identifier (payment reference or order ID). Cannot be empty. |
| `amount` | `float` | **Yes** | Monetary value of the transaction (`TransactionAmt`). Must be strictly $> 0.0$. |
| `transaction_dt` | `float` | No | Timestamp in seconds (`TransactionDT`). Defaults to `0.0` or server time if omitted. |
| `card1` | `string \| integer` | No | Primary payment card identifier (BIN / issuer prefix). |
| `card2` | `string \| integer` | No | Secondary card type / routing code. |
| `addr1` | `string \| integer` | No | Billing / shipping geographic postal/region code. |
| `P_emaildomain` | `string` | No | Purchaser email domain (e.g. `gmail.com`, `yahoo.com`). |
| `DeviceInfo` | `string` | No | Client device model / user-agent fingerprint. |
| `features` | `object` | No | Optional dictionary containing raw IEEE-CIS tabular features (`C1-C14`, `D1-D15`, `V1-V339`, etc.). |

#### Response Schema (`application/json`)
| Field | Type | Description |
|:---|:---|:---|
| `transaction_id` | `string` | Echoed transaction ID. |
| `risk_score` | `float` | Combined continuous risk score $R_t \in [0.0, 1.0]$ derived from $M_0$. |
| `risk_level` | `string` | Categorical risk severity: `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL`. |
| `decision` | `string` | Operational action recommendation: `ALLOW`, `VERIFY`, `THROTTLE`, or `BLOCK`. |
| `signals` | `object` | Component signal breakdown: `baseline_risk`, `temporal_risk`, `graph_risk`, `fusion_risk`. |
| `explanation` | `array of string` | Non-fabricated, human-readable reasons mapping strictly to active features and signals. |
| `metadata` | `object` | Topological network metrics ($d_t$, $v_t$), entity proxy ID, and evaluation timestamp. |

---

## 3. Decision Semantics & Operating Thresholds

The API implements the frozen progressive 4-tier risk decision policy:

| Combined Risk Score ($R_t$) | Risk Level | Policy Decision | Operational Semantics |
|:---|:---:|:---:|:---|
| $R_t < 0.60$ | **`LOW`** | **`ALLOW`** | Frictionless processing; low fraudulent probability. |
| $0.60 \le R_t < 0.65$ | **`MEDIUM`** | **`VERIFY`** | Step-up 3-D Secure OTP / biometric challenge required. |
| $0.65 \le R_t < 0.80$ | **`HIGH`** | **`THROTTLE`** | Velocity dampening, reduced limits, or secondary review queue. |
| $R_t \ge 0.80$ | **`CRITICAL`** | **`BLOCK`** | Immediate hard decline. High-confidence fraud ring / entity abuse. |

---

## 4. Explanation Semantics

Explanations are strictly causal and derived without fabrication:
- **Baseline Risk:**
  - $A_t \ge 0.80$: `"Severe baseline transaction risk (point-wise tabular score: {A_t})"`
  - $A_t \ge 0.60$: `"Elevated baseline transaction risk (point-wise tabular score: {A_t})"`
- **Temporal Entity Risk:**
  - $P_t \ge 0.30$: `"Elevated recent entity risk with persistent longitudinal velocity (P_t: {P_t})"`
  - $P_t > 0.0$: `"Mild recent entity risk detected on historical proxy (P_t: {P_t})"`
- **Relational Device Context:**
  - $G_t \ge 0.50$: `"High relational risk across device network (G_t: {G_t})"`
  - $d_t > 1$: `"Device linked to multiple entities ({d_t} distinct entities connected)"`
  - $v_t > 1$: `"High device transaction velocity ({v_t} new connection events in 24h window)"`
- **Contextual Uplift Dynamics:**
  - If $R_t - A_t \ge 0.01$: `"Risk increased by contextual evidence (+{uplift} combined temporal and graph uplift)"`
  - If $A_t < 0.60$ and $R_t \ge 0.60$: `"Contextual evidence escalated decision from baseline ALLOW to {decision}"`

---

## 5. Query Transaction Decision
- **Endpoint:** `GET /api/v1/risk/transactions/{transaction_id}`
- **Description:** Queries the latest evaluation and explanation for a previously evaluated transaction.
- **Status Codes:**
  - `200 OK`: Found. Returns the complete `TransactionRiskResponse`.
  - `404 Not Found`: Transaction has not been evaluated or was evicted from LRU cache.

---

## 6. End-to-End Examples

### 6.1 ALLOW Example (Low Risk)

**Request:**
```http
POST /api/v1/risk/evaluate HTTP/1.1
Content-Type: application/json

{
  "transaction_id": "tx_clean_1001",
  "transaction_dt": 86450.0,
  "amount": 29.99,
  "card1": 13926,
  "card2": 150,
  "addr1": 315,
  "P_emaildomain": "gmail.com",
  "DeviceInfo": "iOS Device"
}
```

**Response (`200 OK`):**
```json
{
  "transaction_id": "tx_clean_1001",
  "risk_score": 0.03111,
  "risk_level": "LOW",
  "decision": "ALLOW",
  "signals": {
    "baseline_risk": 0.03111,
    "temporal_risk": 0.0,
    "graph_risk": 0.0,
    "fusion_risk": 0.03111
  },
  "explanation": [
    "All baseline and contextual risk signals within low-risk operating thresholds"
  ],
  "metadata": {
    "entity_id": "13926_315_gmail.com",
    "timestamp": 86450.0,
    "amount": 29.99,
    "device_connected_entities": 0,
    "device_recent_velocity": 0,
    "normalized_degree": 0.0,
    "normalized_velocity": 0.0,
    "contextual_uplift": 0.0,
    "evaluated_at": "2026-09-03T06:21:00.123456+00:00"
  }
}
```

---

### 6.2 BLOCK Example (Critical Risk with Contextual Escalation)

**Request:**
```http
POST /api/v1/risk/evaluate HTTP/1.1
Content-Type: application/json

{
  "transaction_id": "tx_fraud_8801",
  "transaction_dt": 13155000.0,
  "amount": 450.00,
  "card1": 11223,
  "addr1": 450,
  "P_emaildomain": "burner_mail.com",
  "DeviceInfo": "SM-G950F",
  "features": {
    "C1": 125.0,
    "C2": 110.0,
    "V258": 4.0,
    "V294": 8.0
  }
}
```

**Response (`200 OK`):**
```json
{
  "transaction_id": "tx_fraud_8801",
  "risk_score": 0.861633,
  "risk_level": "CRITICAL",
  "decision": "BLOCK",
  "signals": {
    "baseline_risk": 0.851633,
    "temporal_risk": 0.01,
    "graph_risk": 0.0,
    "fusion_risk": 0.861633
  },
  "explanation": [
    "Severe baseline transaction risk (point-wise tabular score: 0.8516)",
    "Mild recent entity risk detected on historical proxy (P_t: 0.0100)",
    "Risk increased by contextual evidence (+0.0100 combined temporal and graph uplift)"
  ],
  "metadata": {
    "entity_id": "11223_450_burner_mail.com",
    "timestamp": 13155000.0,
    "amount": 450.0,
    "device_connected_entities": 0,
    "device_recent_velocity": 0,
    "normalized_degree": 0.0,
    "normalized_velocity": 0.0,
    "contextual_uplift": 0.01,
    "evaluated_at": "2026-09-03T06:21:05.987654+00:00"
  }
}
```

---

## 7. Limitations & Operational Notes

1. **Causal Stream Ordering:**
   - The engine's internal state (temporal risk $P_t$ and relational bipartite graph $G_t$) evolves causally. In production streaming environments, transactions should be presented in non-decreasing order of `transaction_dt`.
2. **Missing Device Context:**
   - If `DeviceInfo` is omitted or unpopulated, the engine safely sets graph degree $d_t = 0$, velocity $v_t = 0$, and graph risk $G_t = 0.0$. The transaction is evaluated purely via $A_t$ and $P_t$ without exceptions.
3. **Transaction State Cache:**
   - Evaluated decisions queried via `GET /api/v1/risk/transactions/{transaction_id}` are held in a high-speed, thread-safe in-memory LRU store (default capacity: 100,000 transactions).
