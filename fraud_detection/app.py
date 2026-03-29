"""
app.py
Interactive fraud detection CLI.
Accepts a transaction, runs all three models, fuses scores, and explains.
"""
import warnings
warnings.filterwarnings('ignore')
import json
import numpy as np
from preprocessor import transform_single
from classical_model import predict_classical
from qml_model import predict_qml
from rag_explainer import RAGExplainer
from fusion_engine import fuse_scores, rag_pattern_to_score

BANNER = """
╔══════════════════════════════════════════════════════════╗
║       UPI Fraud Detection System  v1.0                  ║
║  Classical + Quantum ML + RAG Explanation + Fusion      ║
╚══════════════════════════════════════════════════════════╝
"""

SAMPLE_TRANSACTIONS = [
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
    },
    {
        "transaction type": "P2M",
        "merchant_category": "Grocery",
        "amount (INR)": 450,
        "transaction_status": "SUCCESS",
        "sender_age_group": "36-45",
        "receiver_age_group": "26-35",
        "sender_state": "Maharashtra",
        "sender_bank": "HDFC",
        "receiver_bank": "ICICI",
        "device_type": "iOS",
        "network_type": "4G",
        "hour_of_day": 11,
        "day_of_week": "Wednesday",
        "is_weekend": 0
    }
]


def analyze_transaction(transaction: dict, rag: RAGExplainer, verbose: bool = True) -> dict:
    # 1. Preprocess
    X_pca = transform_single(transaction)

    # 2. Classical models
    classical = predict_classical(X_pca)

    # 3. QML model
    qml = predict_qml(X_pca)

    # 4. RAG explanation
    combined_score_preview = (classical['ensemble_score'] + qml['qml_score']) / 2
    rag_result = rag.explain(transaction, combined_score_preview)
    rag_score, rag_conf = rag_pattern_to_score(rag_result['retrieved_patterns'])

    # 5. Dynamic fusion
    fusion = fuse_scores(
        xgb_score=classical['ensemble_score'],
        xgb_confidence=classical['xgb_confidence'],
        qml_score=qml['qml_score'],
        qml_confidence=qml['qml_confidence'],
        rag_score=rag_score,
        rag_confidence=rag_conf,
        verbose=verbose
    )

    result = {
        'transaction': transaction,
        'classical': classical,
        'qml': qml,
        'rag': rag_result,
        'fusion': fusion
    }

    if verbose:
        _print_result(result)

    return result


def _print_result(result: dict):
    fusion = result['fusion']
    rag = result['rag']
    txn = result['transaction']

    verdict = "🚨 FRAUD DETECTED" if fusion['final_prediction'] == 1 else "✅ LEGITIMATE"
    risk_pct = fusion['final_score'] * 100
    conf_pct = fusion['final_confidence'] * 100

    print("\n" + "─" * 60)
    print(f"  VERDICT: {verdict}")
    print(f"  Final Fraud Score : {risk_pct:.1f}%")
    print(f"  Final Confidence  : {conf_pct:.1f}%")
    print("─" * 60)

    print("\n  Model Scores:")
    for model, score in result['fusion']['component_scores'].items():
        conf = result['fusion']['component_confidences'][model]
        w = result['fusion']['weights'][model]
        print(f"    {model:10s} | fraud={score*100:.1f}% | conf={conf*100:.1f}% | weight={w:.3f}")
    c = result['classical']
    print(f"    {'(LR raw)':10s} | fraud={c['lr_score']*100:.1f}%")
    print(f"    {'(XGB raw)':10s} | fraud={c['xgb_score']*100:.1f}%")

    print("\n  RAG Explanation:")
    print(f"    {rag['explanation']}")

    print("\n  Top Matched Fraud Pattern:")
    if rag['retrieved_patterns']:
        top = rag['retrieved_patterns'][0]
        print(f"    Pattern     : {top['pattern']}")
        print(f"    Relevance   : {top['relevance']:.3f}")

    print("\n  Recommendation:")
    print(f"    {rag['top_recommendation']}")
    print("─" * 60)


def interactive_mode(rag: RAGExplainer):
    print("\nEnter transaction details (or press Enter to use sample transactions):")
    print("  [1] Sample FRAUD transaction")
    print("  [2] Sample LEGIT transaction")
    print("  [3] Enter custom transaction as JSON")
    print("  [q] Quit\n")

    while True:
        choice = input("Choice: ").strip().lower()

        if choice == 'q':
            print("Goodbye.")
            break
        elif choice == '1':
            analyze_transaction(SAMPLE_TRANSACTIONS[0], rag)
        elif choice == '2':
            analyze_transaction(SAMPLE_TRANSACTIONS[1], rag)
        elif choice == '3':
            raw = input("Paste transaction JSON: ").strip()
            try:
                txn = json.loads(raw)
                analyze_transaction(txn, rag)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON: {e}")
        else:
            print("Invalid choice. Enter 1, 2, 3, or q.")


if __name__ == '__main__':
    print(BANNER)
    rag = RAGExplainer()

    print("\nRunning demo on both sample transactions...\n")
    for i, txn in enumerate(SAMPLE_TRANSACTIONS):
        print(f"\n{'='*60}")
        print(f"  Transaction {i+1}: ₹{txn['amount (INR)']} | {txn['transaction type']} | {txn['merchant_category']}")
        print(f"{'='*60}")
        analyze_transaction(txn, rag)

    print("\n" + "="*60)
    interactive_mode(rag)
