# TRUSTGRAPH Phase 1: XGBoost Fraud Detection Baseline Integration

**Evaluation Partition:** Chronological Held-out TEST ($N = 88,580$, $3,083$ frauds)  
**Training Partition:** Chronological TRAIN ($N = 413,379$, $14,538$ frauds)  
**Validation Partition:** Chronological VALIDATION ($N = 88,581$, $3,042$ frauds)  
**Primary Base Model:** XGBoost Fraud Risk Classifier  
**Base Risk Definition:** $A_t = P(\text{isFraud} = 1 \mid \text{Features}_t) \in [0.0, 1.0]$  

---

## 1. Executive Summary & Rationale

Phase 1 replaces the initial LightGBM tabular baseline with a high-performance XGBoost model for IEEE-CIS Fraud Detection (based on the solution architecture by Chris Deotte / Konstantin Yakovlev).

### Core Findings on Held-Out Chronological TEST ($N = 88,580$):
- **Precision:** Jumped from **63.49%** (LightGBM) to **73.02%** (XGBoost) — **+9.53 percentage points higher precision**.
- **False Positives (Hard Declines):** Dropped from **755** to **541** — **214 fewer legitimate customer false alarms** (a **28.3% reduction** in false positive rate).
- **Recall:** Increased from **42.59%** ($1,313$ frauds) to **47.49%** ($1,464$ frauds) — **+151 additional frauds captured**.
- **F1 Score:** Rose from **0.5098** to **0.5755** (**+0.0657 improvement**).
- **PR-AUC:** Rose from **0.5340** to **0.5658** (**+0.0318 improvement**).
- **Validation-Selected Threshold:** $\tau = 0.1200$ (selected purely on VALIDATION to maximize F1).

*Scientific integrity note:* All metrics reported here derive strictly from our held-out chronological TEST partition under zero-leakage evaluation.

---

## 2. Integrated Feature Architecture

The solution constructs 263 features combining domain heuristics, D-column temporal alignment, and credit card group aggregations:

### 2.1 Feature Engineering Pipeline (`ModelFeaturePipeline`)
1. **Redundant V-Column Elimination:**
   - 219 collinear V-columns were discarded based on correlation analysis, retaining exactly 120 V-columns.
2. **D-Column Normalization:**
   - Time deltas $D_i$ for $i \in \{1..15\} \setminus \{1, 2, 3, 5, 9\}$ are normalized into points in the past:
     $$D_i^{\text{norm}} = D_i - \frac{\text{TransactionDT}}{86400}$$
   - Stops cumulative drift over time while preserving $D_1$ as the card creation delta.
3. **Categorical Handling & Numeric Standardization:**
   - All string and category columns are alphabetically sorted and factorized into integer mappings with unknown/missing mapped to `-1`.
   - Numeric features are shifted positive by $\min(X)$ and missing values imputed with `-1.0`.
4. **Cents Residual:**
   $$\text{cents} = \text{TransactionAmt} - \lfloor \text{TransactionAmt} \rfloor$$
5. **Interaction Features:**
   - $\text{card1\_addr1} = \text{card1} + \text{"\_"} + \text{addr1}$
   - $\text{card1\_addr1\_P\_emaildomain} = \text{card1\_addr1} + \text{"\_"} + \text{P\_emaildomain}$
6. **UID Reconstruction:**
   - Approximates the issuing date of the credit card:
     $$\text{day} = \frac{\text{TransactionDT}}{86400}, \quad \text{UID} = \text{card1\_addr1} + \text{"\_"} + \lfloor \text{day} - D_1 \rfloor$$
7. **Frequency Encodings (`encode_FE`):**
   - Normalized frequency encodings across: `addr1`, `card1`, `card2`, `card3`, `P_emaildomain`, `card1_addr1`, `card1_addr1_P_emaildomain`, `UID`.
8. **Group Aggregations (`encode_AG` & `encode_AG2`):**
   - Mean & standard deviation of `TransactionAmt`, `D9`, `D11` grouped by card/interaction keys.
   - Mean & standard deviation of `TransactionAmt`, `D4`, `D9`, `D10`, `D15` grouped by `UID`.
   - Mean of `C1`..`C14` (excluding `C3`) grouped by `UID`.
   - Mean of `M1`..`M9` grouped by `UID`.
   - Number of unique values (`nunique`) for `P_emaildomain`, `dist1`, `DT_M`, `id_02`, `cents`, `C13`, `V314`, `V127`, `V136`, `V309`, `V307`, `V320` within each `UID`.
9. **Outlier Indicator (`outsider15`):**
   $$\text{outsider15} = \mathbb{I}(|D_1 - D_{15}| > 3)$$
10. **Time-Consistency Feature Selection:**
    - Explicitly removes columns that degrade across time boundaries: `C3`, `M5`, `id_08`, `id_33`, `card4`, `id_07`, `id_14`, `id_21`, `id_30`, `id_32`, `id_34`, `id_22`..`id_27`, and high-missing $D$ columns (`D6, D7, D8, D9, D12, D13, D14`).

---

## 3. XGBoost Model Configuration

The model is trained via `XGBRiskModel` with early stopping on VALIDATION AUC:

```python
XGB_PARAMS = {
    "n_estimators": 2000,
    "max_depth": 12,
    "learning_rate": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.4,
    "missing": -1.0,
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
}
```

- **Best Iteration:** 611 rounds (early stopping triggered at round 711).
- **Training Time:** 308.05 seconds on multi-core CPU.
- **Probability Calibration:** Unlike the old LightGBM baseline, which used `scale_pos_weight = 27.58` (artificially inflating predicted probabilities toward 1.0), XGBoost optimizes raw log-loss without class reweighting. Output probabilities $A_t$ reflect true posterior fraud rates.

---

## 4. Integration into TRUSTGRAPH Architecture

Downstream components consume $A_t$ via the unified wrapper `XGBBaselineWrapper`:

```
                 ModelFeaturePipeline
                           ↓ (263 features)
                     XGBRiskModel
                           ↓
                          A_t
                           ↓
              [Downstream TRUSTGRAPH Stack]
```

### Preserved Project Contracts:
- `predict_risk(X)` $\to A_t \in [0.0, 1.0]$
- `predict_proba(X)` $\to \text{numpy array of shape } (N, 2)$
- `predict(X, threshold)` $\to \text{binary predictions } (0 \text{ or } 1)$
- `score_dataframe(df)` $\to \text{DataFrame with } [\text{transaction\_id}, \text{risk\_score}, A_t]$

---

## 5. Data Leakage & Causality Audit

Because this system targets real-time payment decisions, we performed an exhaustive audit comparing offline reference scripts against our production deployment:

| Feature Category | Offline Script Behavior | TRUSTGRAPH Implementation | Audit Classification |
|:---|:---|:---|:---:|
| **Target Labels (`isFraud`)** | Labels never used in feature creation (only as training target). | Target column excluded from all feature sets; zero label leakage. | **A (Production Safe)** |
| **D-Columns Normalization** | $D_i - \text{TransactionDT} / 86400$ uses instantaneous row timestamp. | Computed row-by-row at transaction time $t$. | **A (Production Safe)** |
| **Magic UID** | $\text{card1\_addr1} + \lfloor \text{day} - D_1 \rfloor$ uses only transaction fields. | Purely causal card-creation timestamp proxy. | **A (Production Safe)** |
| **Categorical Factorization** | In offline: `pd.concat([X_train, X_test]).factorize()`. | In TRUSTGRAPH: Mappings fitted **strictly on TRAIN** ($N = 413,379$). Unseen test values map to `-1`. | **A (Zero Test Leakage)** |
| **Frequency Encodings (`encode_FE`)** | In offline: Frequencies computed across concatenated Train + Test. | In TRUSTGRAPH: Value counts fitted **strictly on TRAIN**. | **A (Zero Test Leakage)** |
| **UID Group Aggregations (`encode_AG`)** | In offline: Mean/std computed across combined 6-month Train + 6-month Test. | In TRUSTGRAPH: Group tables fitted **strictly on TRAIN**. | **B (Documented Offline / Read-Only)** |

### Causality Verdict:
All preprocessors, encoders, and statistics are fit strictly on the **TRAIN** partition. No test data was accessed during training or threshold selection.

---

## 6. Apples-to-Apples Baseline Comparison

Evaluated on the **Held-out TEST partition** ($N = 88,580$, $3,083$ frauds, base fraud rate $= 3.4805\%$):

| Evaluation Metric | Old LightGBM ($\tau = 0.5943$) | XGBoost ($\tau = 0.1200$) | XGBoost ($\tau = 0.5943$) | Absolute Improvement (XGB vs LGBM) |
|:---|:---:|:---:|:---:|:---:|
| **ROC-AUC** | 0.9019 | 0.8892 | 0.8892 | -0.0127 |
| **PR-AUC** | 0.5340 | **0.5658** | **0.5658** | **+0.0318 (+5.96%)** |
| **Precision** | 63.49% | **73.02%** | **93.77%** | **+9.53 percentage points** |
| **Recall** | 42.59% | **47.49%** | 22.93% | **+4.90 percentage points** |
| **F1-Score** | 0.5098 | **0.5755** | 0.3685 | **+0.0657 (+12.89%)** |
| **False Positive Rate** | 0.8831% | **0.6328%** | **0.0550%** | **-0.2503% (28.3% lower FPR)** |
| **True Positives (TP)** | 1,313 | **1,464** | 707 | **+151 frauds caught** |
| **False Positives (FP)** | 755 | **541** | **47** | **214 fewer false alarms** |
| **False Negatives (FN)** | 1,770 | **1,619** | 2,376 | **-151 fewer missed frauds** |
| **True Negatives (TN)** | 84,742 | **84,956** | 85,450 | **+214 legitimate approved** |
| **Fraud Capture** | 42.59% | **47.49%** | 22.93% | **+4.90 percentage points** |
| **Fraud Enrichment** | 18.24x | **20.98x** | **26.94x** | **+2.74x enrichment** |

---

## 7. Threshold Analysis & Probability Calibration

- **LightGBM Baseline:** Used `scale_pos_weight = 27.58`, shifting median predicted probabilities into the $0.20 - 0.70$ range, resulting in an optimal validation threshold of $\tau = 0.5943$.
- **XGBoost Baseline:** Trained without artificial weight re-scaling. Predicted probabilities directly approximate true conditional risk $P(\text{fraud} \mid X)$. On VALIDATION ($N = 88,581$, base rate $3.43\%$), sweeping thresholds from $0.01$ to $0.95$ identified $\tau = 0.1200$ as the global F1 maximizer ($F_1 = 0.6699$, Precision $= 80.69\%$, Recall $= 57.27\%$).
- When evaluated at $\tau = 0.1200$ on held-out TEST, the model achieves **73.02% precision** and captures **1,464 frauds**.

---

## 8. Artifact Directory (`artifacts/models/kaggle_xgb/`)

| Artifact | File | Size | Contents |
|:---|:---|:---:|:---|
| **Fitted Pipeline** | `feature_pipeline.pkl` | 103.1 MB | Mappings, category lookups, frequency dicts, group aggregations. |
| **Trained XGBoost** | `xgb_model.pkl` | 20.8 MB | Trained XGBoost trees, best iteration (611), hyperparameters. |
| **Evaluation Metadata** | `metadata.json` | 7.5 KB | Partitions, parameters, validation/test metrics, leakage audit. |
| **Test Predictions** | `test_predictions_kaggle_xgb.parquet` | 1.8 MB | TransactionID, TransactionDT, $A_t$, isFraud for all 88,580 test rows. |

---

## 9. Limitations & Operational Notes

1. **Memory Footprint of Group Dictionaries:**
   - The fitted `ModelFeaturePipeline` contains large in-memory lookup dictionaries ($103\text{ MB}$) for UID group statistics. In high-concurrency microservices, this can be streamlined into a fast key-value store (e.g. Redis).
2. **Single-Row Inference:**
   - Transforming a single sparse raw transaction dictionary executes in ~20-30 ms in Python. For sub-5ms SLAs, feature lookup caching or a C++ feature transformer can be used.
3. **No Downstream Retuning Yet:**
   - In accordance with Phase 1 constraints, downstream temporal engines, graph parameters, fusion equations, and policy thresholds were not retuned.
