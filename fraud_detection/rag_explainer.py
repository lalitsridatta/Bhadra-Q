"""
rag_explainer.py
RAG-based fraud explanation system.
"""
import warnings
warnings.filterwarnings('ignore')
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
EMBED_MODEL = 'all-MiniLM-L6-v2'

# ── Fraud knowledge base ─────────────────────────────────────────────────────

FRAUD_KNOWLEDGE = [
    {
        "pattern": "Late night high-value P2P transfer",
        "description": "Transactions above ₹5000 sent peer-to-peer between 11 PM and 5 AM are strongly associated with account takeover fraud.",
        "recommendation": "Verify your identity via OTP and check if you initiated this transfer. Contact your bank immediately if suspicious."
    },
    {
        "pattern": "Unusual merchant category for sender profile",
        "description": "A transaction to a merchant category never used before by the sender (e.g., Fuel for a user who only shops online) is a red flag.",
        "recommendation": "Review recent transactions in your banking app. If unrecognized, raise a dispute with your bank."
    },
    {
        "pattern": "Multiple rapid transactions in short window",
        "description": "Several transactions within minutes from the same account suggest automated fraud or SIM-swap attack.",
        "recommendation": "Freeze your UPI immediately via your bank app and report to the National Cyber Crime portal (cybercrime.gov.in)."
    },
    {
        "pattern": "Cross-state transaction with new receiver",
        "description": "Sending money to a receiver in a different state than usual, especially to a new account, is a common mule account pattern.",
        "recommendation": "Confirm the receiver's identity before proceeding. Avoid sending money to unknown accounts."
    },
    {
        "pattern": "Weekend high-amount transaction on 3G/2G network",
        "description": "Fraudsters exploit weekends when bank support is limited. Transactions on slow networks may indicate SIM cloning.",
        "recommendation": "Avoid large transactions on weekends via slow networks. Use WiFi or 4G/5G for high-value transfers."
    },
    {
        "pattern": "Failed transaction followed by immediate retry",
        "description": "A FAILED status followed by a SUCCESS for the same amount is a common pattern in replay attacks.",
        "recommendation": "Check your transaction history for duplicate charges. Report to your bank's fraud helpline."
    },
    {
        "pattern": "Young sender age group with very high amount",
        "description": "Users aged 18-25 making transactions above ₹10,000 to unknown receivers show elevated fraud risk.",
        "recommendation": "Enable transaction limits for your age group in your bank app. Use UPI PIN carefully."
    },
    {
        "pattern": "P2M transaction to Entertainment/Gaming category at odd hours",
        "description": "Payments to entertainment or gaming merchants late at night are frequently linked to social engineering scams.",
        "recommendation": "Be cautious of unsolicited payment requests. Legitimate businesses don't ask for UPI payments at odd hours."
    },
    {
        "pattern": "Sender and receiver using same bank with high amount",
        "description": "Intra-bank high-value transfers to new payees can indicate internal fraud or phishing.",
        "recommendation": "Verify the payee's VPA (UPI ID) carefully before confirming. Fraudsters use similar-looking IDs."
    },
    {
        "pattern": "iOS device on WiFi with international-pattern amount",
        "description": "Transactions from iOS devices on WiFi with round-number amounts (e.g., ₹9999) are linked to phishing app fraud.",
        "recommendation": "Only use official bank apps from the App Store. Never share your UPI PIN or OTP with anyone."
    },
    {
        "pattern": "High transaction amount on Sunday",
        "description": "Fraudulent transactions are disproportionately observed on Sundays when fraud monitoring teams have reduced capacity.",
        "recommendation": "Set daily transaction limits in your UPI app. Enable SMS/email alerts for all transactions."
    },
    {
        "pattern": "Receiver from high-risk state with new account",
        "description": "Certain states show higher rates of mule accounts used to receive and quickly withdraw fraudulent funds.",
        "recommendation": "Verify receiver details through a secondary channel (phone call) before sending large amounts."
    }
]


class RAGExplainer:
    def __init__(self):
        print("[RAG] Loading sentence transformer...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self._build_index()

    def _build_index(self):
        texts = [f"{k['pattern']}. {k['description']}" for k in FRAUD_KNOWLEDGE]
        embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine on normalized vecs
        self.index.add(embeddings.astype(np.float32))
        self.knowledge = FRAUD_KNOWLEDGE
        print(f"[RAG] Index built with {len(FRAUD_KNOWLEDGE)} fraud patterns.")

    def _transaction_to_query(self, transaction: dict, fraud_score: float) -> str:
        parts = []
        amount = transaction.get('amount (INR)', 0)
        hour = transaction.get('hour_of_day', 12)
        txn_type = transaction.get('transaction type', '')
        merchant = transaction.get('merchant_category', '')
        network = transaction.get('network_type', '')
        device = transaction.get('device_type', '')
        is_weekend = transaction.get('is_weekend', 0)
        sender_age = transaction.get('sender_age_group', '')
        sender_state = transaction.get('sender_state', '')
        status = transaction.get('transaction_status', '')

        parts.append(f"{txn_type} transaction of ₹{amount}")
        if hour < 6 or hour >= 22:
            parts.append("at late night/early morning hours")
        if is_weekend:
            parts.append("on a weekend")
        if merchant:
            parts.append(f"to {merchant} merchant")
        if network in ['3G', '2G']:
            parts.append(f"on slow {network} network")
        if device:
            parts.append(f"from {device} device")
        if sender_age:
            parts.append(f"by {sender_age} age group sender")
        if amount > 5000:
            parts.append("high value transfer")
        if status == 'FAILED':
            parts.append("with failed status")

        query = " ".join(parts) + f". Fraud risk score: {fraud_score:.2f}"
        return query

    def explain(self, transaction: dict, fraud_score: float, top_k: int = 3) -> dict:
        query = self._transaction_to_query(transaction, fraud_score)
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)

        retrieved = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                retrieved.append({
                    'pattern': self.knowledge[idx]['pattern'],
                    'description': self.knowledge[idx]['description'],
                    'recommendation': self.knowledge[idx]['recommendation'],
                    'relevance': float(score)
                })

        # Generate explanation
        explanation = self._generate_explanation(transaction, fraud_score, retrieved)
        top_recommendation = retrieved[0]['recommendation'] if retrieved else "Monitor your account closely."

        return {
            'query': query,
            'retrieved_patterns': retrieved,
            'explanation': explanation,
            'top_recommendation': top_recommendation
        }

    def _generate_explanation(self, transaction: dict, fraud_score: float, patterns: list) -> str:
        amount = transaction.get('amount (INR)', 0)
        hour = transaction.get('hour_of_day', 12)
        txn_type = transaction.get('transaction type', 'Unknown')
        merchant = transaction.get('merchant_category', 'Unknown')
        network = transaction.get('network_type', 'Unknown')
        is_weekend = transaction.get('is_weekend', 0)
        sender_state = transaction.get('sender_state', 'Unknown')

        risk_level = "HIGH" if fraud_score > 0.7 else "MEDIUM" if fraud_score > 0.4 else "LOW"
        time_desc = "late night" if (hour >= 22 or hour < 6) else "daytime"
        weekend_desc = "weekend" if is_weekend else "weekday"

        pattern_names = [p['pattern'] for p in patterns[:2]]
        pattern_str = " and ".join(f'"{p}"' for p in pattern_names)

        explanation = (
            f"This {txn_type} transaction of ₹{amount} to a {merchant} merchant "
            f"at {hour}:00 ({time_desc}, {weekend_desc}) from {sender_state} via {network} "
            f"has been flagged with a {risk_level} fraud risk score of {fraud_score:.1%}. "
            f"The transaction matches known fraud patterns: {pattern_str}. "
        )

        if fraud_score > 0.7:
            explanation += (
                "The combination of transaction characteristics strongly resembles "
                "fraudulent activity in our historical data. Immediate action is recommended."
            )
        elif fraud_score > 0.4:
            explanation += (
                "While not definitively fraudulent, several risk factors are present. "
                "Please verify this transaction carefully."
            )
        else:
            explanation += (
                "The risk is relatively low, but the system flagged some minor anomalies "
                "worth monitoring."
            )

        return explanation
