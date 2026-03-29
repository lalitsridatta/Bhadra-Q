"""
train.py
One-shot training pipeline: preprocess -> LR -> XGBoost -> QSVM
Run this once before using app.py
"""
import numpy as np
from preprocessor import load_and_preprocess
from classical_model import train_logistic, train_xgboost
from qml_model import train_qsvm

if __name__ == '__main__':
    print("=" * 60)
    print("  UPI Fraud Detection - Training Pipeline")
    print("=" * 60)

    # Step 1: Preprocess + PCA + SMOTE
    X_train, X_test, y_train, y_test, scaler, pca, encoders = load_and_preprocess()

    # Step 2: Logistic Regression (baseline)
    train_logistic(X_train, y_train, X_test, y_test)

    # Step 3: XGBoost (boosting)
    train_xgboost(X_train, y_train, X_test, y_test)

    # Step 4: Quantum SVM (subsample for tractability)
    train_qsvm(X_train, y_train, X_test, y_test)

    print("\n[Done] All models trained and saved to models/")
