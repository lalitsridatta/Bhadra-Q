"""
evaluate.py
Clean metrics report for judges — AUC, precision, recall, F1, confusion matrix.
Run: python evaluate.py
"""
import numpy as np
import joblib
import os
from preprocessor import load_and_preprocess
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def evaluate_model(name, y_test, y_prob, threshold=0.3):
    y_pred = (y_prob >= threshold).astype(int)
    auc    = roc_auc_score(y_test, y_prob)
    ap     = average_precision_score(y_test, y_prob)
    cm     = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Model : {name}")
    print(f"  AUC-ROC            : {auc:.4f}")
    print(f"  Avg Precision (AP) : {ap:.4f}")
    print(f"  Threshold used     : {threshold}")
    print(f"\n  Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Legit   Fraud")
    print(f"  Actual Legit  {tn:5d}   {fp:5d}")
    print(f"  Actual Fraud  {fn:5d}   {tp:5d}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'], digits=3))
    return auc, ap

if __name__ == '__main__':
    print_section("UPI Fraud Detection — Model Evaluation Report")

    print("\n  Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, *_ = load_and_preprocess(apply_smote=False)
    print(f"  Test set: {len(y_test)} samples | Fraud: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

    # ── Logistic Regression ──────────────────────────────────────────────────
    print_section("1. Logistic Regression (Baseline)")
    lr = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    lr_prob = lr.predict_proba(X_test)[:, 1]
    lr_auc, lr_ap = evaluate_model("Logistic Regression", y_test, lr_prob, threshold=0.3)

    # ── XGBoost ──────────────────────────────────────────────────────────────
    print_section("2. XGBoost (Boosting)")
    xgb = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))
    xgb_prob = xgb.predict_proba(X_test)[:, 1]
    xgb_auc, xgb_ap = evaluate_model("XGBoost", y_test, xgb_prob, threshold=0.3)

    # ── QSVM ─────────────────────────────────────────────────────────────────
    print_section("3. Quantum Kernel SVM (QML)")
    from qml_model import QML_FEATURES, _build_kernel_matrix
    artifact = joblib.load(os.path.join(MODELS_DIR, 'qsvm.pkl'))
    svm      = artifact['svm']
    qscaler  = artifact['qscaler']
    X_support = artifact['X_support']

    # Balanced subsample of test set for QSVM eval
    fraud_idx = np.where(y_test == 1)[0]
    legit_idx = np.where(y_test == 0)[0]
    half = min(100, len(fraud_idx))
    sel = np.concatenate([
        np.random.choice(fraud_idx, half, replace=False),
        np.random.choice(legit_idx, half, replace=False)
    ])
    np.random.shuffle(sel)
    X_q = X_test[sel, :QML_FEATURES]
    y_q = y_test[sel]
    X_q_s = qscaler.transform(X_q)

    print(f"\n  Building QSVM test kernel ({len(X_q_s)}x{len(X_support)})...")
    K_test = _build_kernel_matrix(X_q_s, X_support)
    qsvm_prob = svm.predict_proba(K_test)[:, 1]
    qsvm_auc, qsvm_ap = evaluate_model("Quantum SVM", y_q, qsvm_prob, threshold=0.5)

    # ── Summary ──────────────────────────────────────────────────────────────
    print_section("Summary Comparison")
    print(f"\n  {'Model':<25} {'AUC-ROC':>10} {'Avg Precision':>15}")
    print(f"  {'-'*52}")
    print(f"  {'Logistic Regression':<25} {lr_auc:>10.4f} {lr_ap:>15.4f}")
    print(f"  {'XGBoost':<25} {xgb_auc:>10.4f} {xgb_ap:>15.4f}")
    print(f"  {'Quantum SVM (subset)':<25} {qsvm_auc:>10.4f} {qsvm_ap:>15.4f}")
    print(f"\n  Note: QSVM evaluated on balanced 200-sample subset (quantum simulation constraint).")
    print(f"  All models use PCA-reduced features (10 components, {0.642*100:.1f}% variance explained).")
    print(f"  Fusion engine dynamically weights all 3 models by confidence at inference time.\n")
