# Bhadra Q — Quantum-Enhanced UPI Fraud Detection

> A multi-model fraud detection system combining Classical ML, Quantum Kernel SVM, and RAG-based explainability, fused with dynamic confidence-weighted scoring.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Models](#models)
5. [Fusion Engine](#fusion-engine)
6. [RAG Explainer](#rag-explainer)
7. [API Reference](#api-reference)
8. [Setup & Running](#setup--running)
9. [Evaluation Results](#evaluation-results)
10. [Dataset](#dataset)

---

## Overview

Bhadra Q detects fraudulent UPI transactions by running three independent models in parallel and fusing their outputs into a single verdict. Each model brings a different perspective:

| Model | Strength |
|-------|----------|
| Logistic Regression + XGBoost | Fast, high-AUC classical detection |
| Quantum Kernel SVM | Non-linear quantum feature space separation |
| RAG Pattern Matcher | Semantic similarity to known fraud patterns |

The final score is a **confidence-weighted fusion** — models that are more certain about their prediction get higher influence on the result.

---

## Architecture

```
Transaction Input
       │
       ▼
┌─────────────────┐
│   Preprocessor  │  Label encoding → Feature engineering → StandardScaler → PCA (10 components)
└────────┬────────┘
         │
    ┌────┴─────────────────────────┐
    │             │                │
    ▼             ▼                ▼
┌────────┐  ┌─────────┐   ┌──────────────┐
│  LR +  │  │ Quantum │   │     RAG      │
│ XGBoost│  │ Kernel  │   │  Explainer   │
│        │  │   SVM   │   │ (FAISS+SBERT)│
└───┬────┘  └────┬────┘   └──────┬───────┘
    │             │               │
  score+conf   score+conf      score+conf
    │             │               │
    └─────────────┴───────────────┘
                  │
                  ▼
        ┌──────────────────┐
        │  Fusion Engine   │  softmax(confidences) → weighted average
        └────────┬─────────┘
                 │
                 ▼
        Final Score + Verdict + Explanation
```

---

## Project Structure

```
fraud_detection/
├── train.py            # One-shot training pipeline
├── evaluate.py         # Metrics report (AUC, precision, recall, F1)
├── server.py           # Flask backend (SSE streaming API)
├── app.py              # CLI interface
│
├── preprocessor.py     # Data loading, feature engineering, PCA
├── classical_model.py  # Logistic Regression + XGBoost
├── qml_model.py        # Quantum Kernel SVM (PennyLane)
├── rag_explainer.py    # FAISS + sentence-transformers RAG
├── fusion_engine.py    # Dynamic confidence-weighted fusion
│
├── models/
│   ├── logistic_regression.pkl
│   ├── xgboost.pkl
│   └── qsvm.pkl
│
├── data/
│   ├── scaler.pkl
│   ├── pca.pkl
│   └── encoders.pkl
│
├── static/
│   └── index.html      # Bhadra Q frontend
│
└── requirements.txt
```

---

## Models

### 1. Classical Models (`classical_model.py`)

Two models trained on PCA-reduced features, soft-ensembled at inference.

**Logistic Regression**
- `C=0.1`, `class_weight='balanced'`, `max_iter=1000`
- Serves as a linear baseline — strong at capturing linear fraud signals

**XGBoost**
- `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`
- `scale_pos_weight` auto-computed from class ratio
- Handles non-linear interactions between features

**Ensemble**
```python
ensemble_score = 0.4 * lr_score + 0.6 * xgb_score
```

---

### 2. Quantum Kernel SVM (`qml_model.py`)

Uses a ZZ feature map circuit to encode PCA features into quantum states, then computes a fidelity kernel for SVM classification.

**Circuit (ZZ Feature Map)**
```
For each qubit i:
  H(i) → RZ(2x[i], i)
For each pair (i, i+1):
  CNOT(i, i+1) → RZ(2(π-x[i])(π-x[i+1]), i+1) → CNOT(i, i+1)
```

**Quantum Kernel**
```
K(x₁, x₂) = |⟨φ(x₁)|φ(x₂)⟩|²
```

**Specs**
- 4 qubits, 4 PCA features, simulated via PennyLane `default.qubit`
- Training subset: 200 balanced samples (quantum simulation constraint)
- SVM: `kernel='precomputed'`, `class_weight='balanced'`

---

### 3. RAG Explainer (`rag_explainer.py`)

Retrieval-Augmented Generation system that matches a transaction against a knowledge base of 12 fraud patterns.

**Components**
- Embedding model: `all-MiniLM-L6-v2` (sentence-transformers)
- Vector store: FAISS `IndexFlatIP` (cosine similarity on normalized vectors)
- Knowledge base: 12 curated UPI fraud patterns with descriptions and recommendations

**Inference**
1. Transaction features → natural language query string
2. Query embedded → cosine similarity search over fraud pattern index
3. Top-k patterns retrieved → explanation + recommendation generated
4. Top relevance score → converted to soft fraud probability

```python
rag_score = 0.3 + top_relevance * 0.7   # maps [0,1] cosine → [0.3, 1.0]
rag_confidence = top_relevance
```

---

## Fusion Engine

**File:** `fusion_engine.py`

Each model outputs a fraud score `s ∈ [0,1]` and a confidence `c ∈ [0,1]`.

```python
confidence = abs(score - 0.5) * 2
```

Dynamic weights are computed via softmax over confidences:

```python
weights = softmax([c_xgb, c_qml, c_rag])
final_score = w_xgb * s_xgb + w_qml * s_qml + w_rag * s_rag
prediction = FRAUD if final_score >= 0.5 else LEGIT
```

**Why softmax?** It amplifies differences — a model with 0.9 confidence gets significantly more weight than one with 0.3, without any manual tuning.

**Example (fraud transaction):**

| Model | Score | Confidence | Weight |
|-------|-------|------------|--------|
| XGBoost | 40.1% | 19.8% | 0.215 |
| QSVM | 89.8% | 79.6% | 0.391 |
| RAG | 86.4% | 80.6% | 0.395 |
| **Final** | **77.8%** | **55.6%** | — |

---

## API Reference

### `POST /analyze`

Accepts a transaction JSON, streams results via Server-Sent Events (SSE).

**Request body:**
```json
{
  "transaction type": "P2P",
  "merchant_category": "Entertainment",
  "amount (INR)": 49500,
  "transaction_status": "SUCCESS",
  "sender_age_group": "18-25",
  "receiver_age_group": "26-35",
  "sender_state": "Delhi",
  "sender_bank": "Axis",
  "receiver_bank": "SBI",
  "device_type": "Android",
  "network_type": "3G",
  "hour_of_day": 2,
  "day_of_week": "Sunday",
  "is_weekend": 1
}
```

**SSE Events streamed:**

| Event | Payload |
|-------|---------|
| `progress` | `{ step, label, pct }` |
| `classical` | `{ lr_score, xgb_score, ensemble_score, confidence, prediction }` |
| `qml` | `{ score, confidence }` |
| `rag` | `{ score, confidence, explanation, recommendation, patterns[] }` |
| `fusion` | `{ final_score, final_confidence, final_prediction, weights, component_scores }` |
| `error` | `{ message }` |

---

## Setup & Running

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train models (once)
```bash
cd fraud_detection
python train.py
```

### Run the web app
```bash
python server.py
```
Open **http://127.0.0.1:5000** in your browser.

### Run evaluation report
```bash
python evaluate.py
```

### Run CLI demo
```bash
python app.py
```

---

## Evaluation Results

Evaluated on 50,000 test samples (0.5% fraud rate = 250 fraud cases).

---

### Classical Models — Full Test Set (50,000 samples)

**Logistic Regression**

| Metric | Legit | Fraud |
|--------|-------|-------|
| Precision | 1.000 | 0.272 |
| Recall | 0.987 | 0.996 |
| F1-Score | 0.993 | 0.427 |

```
AUC-ROC : 0.9980
Avg Precision : 0.6946
Threshold : 0.3

Confusion Matrix:
              Predicted
              Legit   Fraud
Actual Legit  49083     667
Actual Fraud      1     249
```

---

**XGBoost**

| Metric | Legit | Fraud |
|--------|-------|-------|
| Precision | 1.000 | 0.376 |
| Recall | 0.992 | 0.960 |
| F1-Score | 0.996 | 0.541 |

```
AUC-ROC : 0.9974
Avg Precision : 0.6322
Threshold : 0.3

Confusion Matrix:
              Predicted
              Legit   Fraud
Actual Legit  49352     398
Actual Fraud     10     240
```

---

### Quantum SVM — Balanced Subset (200 samples)

| Metric | Legit | Fraud |
|--------|-------|-------|
| Precision | 0.942 | 0.979 |
| Recall | 0.980 | 0.940 |
| F1-Score | 0.961 | 0.959 |

```
AUC-ROC : 0.9898
Avg Precision : 0.9892
Threshold : 0.5
Accuracy : 96.0%

Confusion Matrix:
              Predicted
              Legit   Fraud
Actual Legit     98       2
Actual Fraud      6      94
```

> QSVM is evaluated on a balanced 200-sample subset due to O(n²) quantum kernel simulation cost. On real quantum hardware this constraint does not apply.

---

### Summary

| Model | AUC-ROC | Avg Precision | Fraud Recall | Test Set |
|-------|---------|---------------|--------------|----------|
| Logistic Regression | 0.9980 | 0.6946 | 99.6% | 50,000 samples |
| XGBoost | 0.9974 | 0.6322 | 96.0% | 50,000 samples |
| Quantum SVM | 0.9898 | 0.9892 | 94.0% | 200-sample balanced subset |

**PCA:** 10 components, 64.2% variance explained  
**SMOTE:** Applied during training to balance classes  
**Fusion:** Dynamic confidence-weighted ensemble of all 3 models

---

## Dataset

**File:** `upi_transactions_2024.csv`  
**Size:** 250,000 transactions  
**Features used:** 14 raw + 7 engineered = 21 total → reduced to 10 via PCA

**Engineered features:**

| Feature | Description |
|---------|-------------|
| `is_late_night` | Hour between 22:00–05:00 |
| `is_high_amount` | Amount above 75th percentile |
| `is_slow_network` | Network is 2G or 3G |
| `is_p2p_high_risk` | P2P + late night + high amount combo |
| `is_cross_bank` | Sender and receiver on different banks |
| `amount_log` | log(1 + amount) to reduce skew |
| `risk_score` | Composite rule-based risk signal (used as feature) |

> Note: The original dataset's `fraud_flag` had near-zero correlation with features (synthetically random labels). Fraud labels were reassigned using a rule-based scoring function to inject realistic signal for model training.
