# TRUSTGRAPH Phase 4: Grounded GraphRAG AI Risk Investigator & Interactive Risk Graph

## 1. Overview & Architecture

Phase 4 introduces the **AI Risk Investigator** on top of the frozen Phase 1–3 architecture:

```text
                           TRANSACTION
                                │
                                ▼
                       Existing Risk Engine
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
                   Aₜ          Gₜ          Rₜ
               (XGBoost)     (Graph)    (F2 Fusion)
                    │           │           │
                    └───────────┼───────────┘
                                │
                                ▼
                       Evidence Retriever
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            Graph Evidence            Risk Evidence
          (1-2 Hop Subgraph)      (Decision & Formula)
                    │                       │
                    └───────────┬───────────┘
                                │
                                ▼
                           LLM / RAG
                  (Deterministic / Gemini)
                                │
                                ▼
                       AI Risk Investigator
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
       Grounded Explanation              Evidence References
        (Structured Reasons)               ([E1], [E2], ...)
```

The LLM is strictly an **explanation and investigation layer**; it is **NOT** the decision maker. The mathematical risk engine ($A_t, G_t, R_t$) and cost-minimising policy remain the authoritative source of truth.

---

## 2. Real Knowledge Graph Retrieval Pipeline

The retriever queries the production Phase 2 `PaymentKnowledgeGraph` directly:
- **No Secondary Graph**: GraphRAG traverses the exact same graph state that computed $G_t$.
- **Bounded Neighborhood ($k \le 2$ hops)**:
  - **1-Hop Neighbors**: CustomerEntity (`MADE_BY`), Device (`USED_DEVICE`), Card (`USED_CARD`), Address (`SHIPPED_TO`), Email (`HAS_EMAIL`), Network (`ACCESSED_VIA`).
  - **2-Hop Neighbors**: Prior transactions from the entity, other entities sharing the device, confirmed fraud events linked to the device, 2-hop contamination rings.
  - **Hub Damping**: Attributes with degree $> 25$ (broad OS / zip codes) are pruned to prevent combinatorial explosions. Total nodes capped at 50 for visualization.
- **Deterministic Evidence Ranking**:
  1. `DIRECT_FRAUD` (weight: 0.95): Confirmed prior fraud transactions on the customer entity.
  2. `DEVICE_FRAUD` (weight: 0.90): Prior confirmed fraud transactions originating from the same device.
  3. `DEVICE_SHARING` (weight: 0.75): Multiple distinct entities multiplexing on the same device.
  4. `HOP2_CONTAMINATION` (weight: 0.70): Shared infrastructure with known fraudulent rings.
  5. `VELOCITY_BURST` (weight: 0.60): Elevated 1h / 24h transaction volume.
  6. `CARD_SHARING` / `COLD_START` (weight: 0.05–0.10): Clean history or first-time entity.
  7. `RISK_ENGINE` (weight: 0.85): Mathematical fusion and cost decision provenance.

---

## 3. Structured Evidence Schema & Provenance

Every retrieved evidence item is represented as a strongly-typed `EvidenceItem`:

```json
{
  "evidence_id": "E1",
  "evidence_type": "DIRECT_FRAUD",
  "title": "Confirmed Recidivist Entity Fraud History",
  "description": "Customer Entity '10568' has 159 confirmed prior fraudulent transactions (e.g. Transaction #3503959). Historical entity fraud rate is 20.5%.",
  "risk_weight": 0.95,
  "source_node": "txn_3504259",
  "target_node": "past_fraud_3503959",
  "relationship_path": ["txn_3504259", "ent_10568", "past_fraud_3503959"],
  "provenance": {
    "prior_fraud_txn_id": 3503959,
    "entity_id": "10568",
    "total_prior_txns": 775,
    "total_prior_frauds": 159
  }
}
```

Provenance fields link every claim back to raw transaction IDs, timestamps, and confirmed historical labels.

---

## 4. Strict Anti-Hallucination & Grounding Design

The anti-hallucination layer enforces four strict invariants:

1. **Explicit Citation Requirement**:
   - Every factual claim produced by the AI investigator must be tagged with `[EVIDENCE: E...]` or `[EVIDENCE: RISK_ENGINE]`.
2. **Post-Generation Grounding Validator**:
   - The engine validates cited IDs against the retrieved `EvidenceItem` set. Any citation to non-existent evidence is stripped or flagged.
3. **Out-of-Domain Guardrail**:
   - If an analyst asks a question whose facts are not present in the graph (e.g., weather, credit score, personal details), the engine strictly responds:
     > *"Insufficient evidence to determine this. The payment knowledge graph and transaction record do not contain this information. Factual claims cannot be substantiated from retrieved evidence."*
4. **Mathematical Engine Immutability**:
   - The LLM never recalculates risk. $A_t, G_t, R_t$ are injected from the frozen Phase 3 outputs and cannot be overridden.

---

## 5. LLM Provider Abstraction & Fallback Behavior

```text
LLMProvider (Abstract Base Class)
    ├── DeterministicFallbackProvider (Default / Zero Dependencies)
    └── GeminiProvider (Native HTTP via GEMINI_API_KEY)
```

- **Deterministic Fallback (Default)**:
  - Generates rule-based, perfectly structured investigation reports and answers.
  - Zero hallucination risk.
  - Sub-millisecond latency (0.12 ms).
  - Ensures the platform runs out-of-the-box without requiring external paid API keys.
- **Gemini Provider**:
  - Activated when `GEMINI_API_KEY` is present.
  - Uses native HTTP requests with low temperature (0.1) and strict anti-hallucination system instructions.
  - Falls back automatically to `DeterministicFallbackProvider` if network requests fail or timeout.

---

## 6. Interactive Graph Visualization (Dashboard)

The frontend console (`/dashboard`) serves an interactive force-directed network graph built with D3.js (v7):

- **Data-Driven Real Graph**: Renders the exact 1-hop and 2-hop neighborhood retrieved from the backend.
- **Node Semantics**:
  - `Transaction`: Blue target circle (prominent, $r=24$).
  - `CustomerEntity`: Emerald if clean, orange if elevated risk ($r=18$).
  - `Device`: Purple circle ($r=14$).
  - `Card`: Yellow circle ($r=14$).
  - `Confirmed Fraud`: Red pulsing circle ($r=20$) with border stroke.
- **Edge Semantics**:
  - Real Phase 2 relationships: `MADE_BY`, `USED_CARD`, `USED_DEVICE`, `SHIPPED_TO`, `HAS_EMAIL`, `ACCESSED_VIA`.
  - Suspicious fraud paths marked with red dashed lines.
- **Controls & Interactivity**:
  - **Zoom & Pan**: Full smooth pan/zoom canvas.
  - **Node Click Inspector**: Displays node type, risk score, fraud status, and properties.
  - **Edge Click**: Shows relationship type and causal provenance.
  - **Node Filters**: Checkboxes to toggle Transactions, Entities, Devices, and Cards.
  - **Depth Toggle**: Switch dynamically between 1-Hop and 2-Hop exploration.
  - **Highlight Fraud Paths**: Toggles high-contrast glow along paths leading to confirmed historical frauds.

---

## 7. API Endpoints

All endpoints return structured JSON:

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/api/v1/investigation/demo-transactions` | List the 4 canonical test demo transactions. |
| `GET` | `/api/v1/risk/{transaction_id}` | Retrieve $A_t, G_t, R_t$, and cost decision summary. |
| `GET` | `/api/v1/risk/{transaction_id}/graph?max_hops=2` | Retrieve nodes, edges, and suspicious paths for D3. |
| `GET` | `/api/v1/risk/{transaction_id}/evidence` | Retrieve ranked list of `EvidenceItem` records with provenance. |
| `POST` / `GET` | `/api/v1/risk/{transaction_id}/investigate` | Generate full grounded investigation report with citations. |
| `POST` | `/api/v1/risk/{transaction_id}/ask` | Q&A endpoint: answer analyst questions grounded in evidence. |
| `GET` | `/dashboard` or `/` | Serve interactive D3.js force-directed console. |

---

## 8. Performance & Latency Benchmarks

Measured on the 88k transaction test partition:

| Component | Latency |
|:---|:---:|
| **Knowledge Graph Retrieval (2-Hop)** | **0.102 ms** |
| **Evidence Construction & Ranking** | **0.108 ms** |
| **Investigation Report Synthesis (Fallback)** | **0.126 ms** |
| **FastAPI `/api/v1/risk/{id}/investigate`** | **5.378 ms** |
| **FastAPI `/api/v1/risk/{id}/ask` (Q&A)** | **4.771 ms** |
| **Graph Rendering Performance** | **60 FPS** (D3 force simulation on canvas) |

---

## 9. Representative Demonstration Cases

The system provides 4 pre-configured test transactions covering diverse risk dynamics:

1. **Transaction #3504259 — Borderline Fraud Escalated by Graph Context**:
   - $A_t = 0.5198 \to R_t = 0.5229$.
   - **Action Transition**: `VERIFY` $\to$ **`BLOCK`**.
   - **Graph Evidence**: Entity has 159 prior frauds; device is shared across 5 entities with 1,156 2-hop fraud links.
2. **Transaction #3570805 — High-Risk Recidivist Fraud**:
   - $A_t = 0.9921, G_t = 0.7375 \to R_t = 0.9922$.
   - **Action**: **`BLOCK`** (Expected Loss: ₹4 vs ₹1,074 for ALLOW).
   - **Graph Evidence**: 25 direct entity frauds, 2,607 device-linked frauds.
3. **Transaction #3512832 — Recidivist Entity on Contaminated Device**:
   - $A_t = 0.1683, G_t = 0.9625 \to R_t = 0.1825$.
   - **Action**: **`THROTTLE`**.
   - **Graph Evidence**: 343 entity frauds, device shared across 23 entities with 8 confirmed fraud events.
4. **Transaction #3488964 — Clean Legitimate Customer**:
   - $A_t = 0.0014, G_t = 0.4000 \to R_t = 0.0039$.
   - **Action**: **`ALLOW`**.
   - **Graph Evidence**: Zero fraud history, no device sharing, fast-path approved.

---

## 10. Limitations

1. **Sub-graph Node Truncation**: High-degree nodes (e.g. `DeviceInfo = "Windows"` or `ProductCD = "W"`) are capped to 50 nodes to avoid visual freezing in D3.js.
2. **Local Fallback Template Depth**: The deterministic template covers the top 6 investigation question patterns; highly complex multi-hop compositional natural language queries benefit from setting `GEMINI_API_KEY`.
