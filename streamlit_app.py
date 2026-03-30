"""
streamlit_app.py  —  Bhadra Q Streamlit frontend
Run: streamlit run streamlit_app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fraud_detection'))

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import time

st.set_page_config(page_title="Bhadra Q", page_icon="⚛️", layout="centered")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0a0a0f; }
  .block-container { padding-top: 2rem; }
  h1 { background: linear-gradient(135deg, #7c6af7, #4fc3f7);
       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
       font-size: 2.8rem !important; font-weight: 800 !important; text-align: center; }
  .tagline { text-align: center; color: #6b6b8a; font-size: 0.95rem; margin-top: -10px; margin-bottom: 20px; }
  .verdict-fraud { background: #2d0a0a; border: 1px solid #f06292; border-radius: 12px; padding: 20px; text-align: center; }
  .verdict-legit { background: #0a2d1a; border: 1px solid #4caf82; border-radius: 12px; padding: 20px; text-align: center; }
  .model-card { background: #1a1a26; border: 1px solid #2a2a3d; border-radius: 12px; padding: 16px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1>Bhadra Q</h1>", unsafe_allow_html=True)
st.markdown('<p class="tagline">⚛️ Quantum-Enhanced UPI Fraud Detection · Classical · Quantum · RAG · Fusion</p>', unsafe_allow_html=True)
st.divider()

# ── Load models once ──────────────────────────────────────────────────────────
@st.cache_resource
def load_rag():
    from rag_explainer import RAGExplainer
    return RAGExplainer()

# ── Input Form ────────────────────────────────────────────────────────────────
st.subheader("Transaction Details")

col1, col2, col3 = st.columns(3)
with col1:
    txn_type     = st.selectbox("Transaction Type", ["P2P", "P2M", "Recharge", "Bill Payment"])
    merchant     = st.selectbox("Merchant Category", ["Grocery","Entertainment","Food","Fuel","Healthcare","Education","Travel","Shopping","Utilities","Gaming"])
    amount       = st.number_input("Amount (INR ₹)", min_value=1, value=4950)
    status       = st.selectbox("Transaction Status", ["SUCCESS", "FAILED", "PENDING"])

with col2:
    sender_age   = st.selectbox("Sender Age Group",   ["18-25","26-35","36-45","46-60","60+"])
    receiver_age = st.selectbox("Receiver Age Group", ["18-25","26-35","36-45","46-60","60+"])
    sender_state = st.selectbox("Sender State", ["Delhi","Maharashtra","Karnataka","Tamil Nadu","Gujarat","Rajasthan","West Bengal","Uttar Pradesh","Telangana","Kerala","Punjab","Haryana"])

with col3:
    sender_bank  = st.selectbox("Sender Bank",   ["SBI","HDFC","ICICI","Axis","PNB","Kotak","BOB","Canara"])
    receiver_bank= st.selectbox("Receiver Bank", ["HDFC","SBI","ICICI","Axis","PNB","Kotak","BOB","Canara"])
    device       = st.selectbox("Device Type",   ["Android","iOS"])
    network      = st.selectbox("Network Type",  ["4G","5G","3G","2G","WiFi"])

col4, col5 = st.columns(2)
with col4:
    hour         = st.slider("Hour of Day", 0, 23, 14)
with col5:
    day          = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

is_weekend = 1 if day in ["Saturday","Sunday"] else 0

txn = {
    "transaction type":   txn_type,
    "merchant_category":  merchant,
    "amount (INR)":       amount,
    "transaction_status": status,
    "sender_age_group":   sender_age,
    "receiver_age_group": receiver_age,
    "sender_state":       sender_state,
    "sender_bank":        sender_bank,
    "receiver_bank":      receiver_bank,
    "device_type":        device,
    "network_type":       network,
    "hour_of_day":        hour,
    "day_of_week":        day,
    "is_weekend":         is_weekend
}

st.divider()
analyze = st.button("⚛️ Analyze Transaction", use_container_width=True, type="primary")

# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze:
    from preprocessor import transform_single
    from classical_model import predict_classical
    from qml_model import predict_qml
    from fusion_engine import fuse_scores, rag_pattern_to_score

    rag = load_rag()

    progress = st.progress(0, text="Starting analysis...")
    status_text = st.empty()

    # Step 1
    progress.progress(10, text="Preprocessing transaction...")
    X_pca = transform_single(txn)

    # Step 2
    progress.progress(30, text="Running Classical Models (LR + XGBoost)...")
    time.sleep(0.3)
    classical = predict_classical(X_pca)

    # Step 3
    progress.progress(60, text="Running Quantum Kernel SVM...")
    qml = predict_qml(X_pca)

    # Step 4
    progress.progress(80, text="Running RAG Explainer...")
    time.sleep(0.2)
    preview = (classical['ensemble_score'] + qml['qml_score']) / 2
    rag_result = rag.explain(txn, preview)
    rag_score, rag_conf = rag_pattern_to_score(rag_result['retrieved_patterns'])

    # Step 5
    progress.progress(95, text="Fusing scores with dynamic weights...")
    time.sleep(0.2)
    fusion = fuse_scores(
        xgb_score=classical['ensemble_score'],
        xgb_confidence=classical['xgb_confidence'],
        qml_score=qml['qml_score'],
        qml_confidence=qml['qml_confidence'],
        rag_score=rag_score,
        rag_confidence=rag_conf,
        verbose=False
    )
    progress.progress(100, text="Analysis complete.")

    st.divider()

    # ── Verdict ──────────────────────────────────────────────────────────────
    is_fraud = fusion['final_prediction'] == 1
    if is_fraud:
        st.markdown(f"""<div class="verdict-fraud">
            <h2 style="color:#f06292">🚨 FRAUD DETECTED</h2>
            <p style="color:#e8e8f0;font-size:1.1rem">Fraud Score: <b>{fusion['final_score']*100:.1f}%</b> &nbsp;|&nbsp; Confidence: <b>{fusion['final_confidence']*100:.1f}%</b></p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="verdict-legit">
            <h2 style="color:#4caf82">✅ LEGITIMATE</h2>
            <p style="color:#e8e8f0;font-size:1.1rem">Fraud Score: <b>{fusion['final_score']*100:.1f}%</b> &nbsp;|&nbsp; Confidence: <b>{fusion['final_confidence']*100:.1f}%</b></p>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Model Scores ─────────────────────────────────────────────────────────
    st.subheader("Model Scores")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Classical (LR+XGB)", f"{classical['ensemble_score']*100:.1f}%", f"Confidence: {classical['xgb_confidence']*100:.1f}%")
        st.progress(classical['ensemble_score'])
    with c2:
        st.metric("Quantum SVM", f"{qml['qml_score']*100:.1f}%", f"Confidence: {qml['qml_confidence']*100:.1f}%")
        st.progress(qml['qml_score'])
    with c3:
        st.metric("RAG Pattern Match", f"{rag_score*100:.1f}%", f"Confidence: {rag_conf*100:.1f}%")
        st.progress(rag_score)

    # ── Fusion Weights ────────────────────────────────────────────────────────
    st.subheader("Dynamic Fusion Weights")
    wc1, wc2, wc3 = st.columns(3)
    weights = fusion['weights']
    labels  = list(weights.keys())
    with wc1: st.metric(labels[0], f"{weights[labels[0]]*100:.1f}%")
    with wc2: st.metric(labels[1], f"{weights[labels[1]]*100:.1f}%")
    with wc3: st.metric(labels[2], f"{weights[labels[2]]*100:.1f}%")

    st.divider()

    # ── RAG Explanation ───────────────────────────────────────────────────────
    st.subheader("Why this transaction was flagged")
    st.info(rag_result['explanation'])

    st.subheader("Matched Fraud Patterns")
    for p in rag_result['retrieved_patterns']:
        with st.expander(f"🔍 {p['pattern']}  —  Relevance: {p['relevance']*100:.1f}%"):
            st.write(p['description'])

    st.subheader("Recommendation")
    st.success(rag_result['top_recommendation'])
