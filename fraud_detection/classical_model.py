"""
classical_model.py
Logistic Regression (baseline) + XGBoost (boosting) for fraud detection.
Returns fraud probability and confidence.
"""
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def train_logistic(X_train, y_train, X_test, y_test):
    print("\n[Classical] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1, random_state=42)
    lr.fit(X_train, y_train)

    y_prob = lr.predict_proba(X_test)[:, 1]
    # Use lower threshold since fraud is rare in test set
    threshold = 0.3
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    print(f"[LR] AUC: {auc:.4f} (threshold={threshold})")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

    joblib.dump(lr, os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    return lr


def train_xgboost(X_train, y_train, X_test, y_test):
    print("\n[Classical] Training XGBoost...")
    scale_pos = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    # Lower threshold for better fraud recall
    threshold = 0.3
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    print(f"[XGB] AUC: {auc:.4f} (threshold={threshold})")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

    joblib.dump(xgb, os.path.join(MODELS_DIR, 'xgboost.pkl'))
    return xgb


def predict_classical(X_pca: np.ndarray):
    """
    Returns dict with lr_score, xgb_score, ensemble_score, xgb_confidence, prediction.
    Uses a soft ensemble of LR + XGBoost for the final classical score.
    """
    lr  = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression.pkl'))
    xgb = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))

    lr_prob  = lr.predict_proba(X_pca)[0][1]
    xgb_prob = xgb.predict_proba(X_pca)[0][1]

    # Soft ensemble: LR catches what XGB misses and vice versa
    ensemble_score = 0.4 * lr_prob + 0.6 * xgb_prob
    confidence     = abs(ensemble_score - 0.5) * 2  # 0..1

    return {
        'lr_score':       float(lr_prob),
        'xgb_score':      float(xgb_prob),
        'ensemble_score': float(ensemble_score),
        'xgb_confidence': float(confidence),
        'prediction':     int(ensemble_score >= 0.5)
    }
