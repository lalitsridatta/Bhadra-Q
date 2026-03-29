"""
qml_model.py
Quantum Kernel + SVM for fraud detection using PennyLane.
Uses ZZFeatureMap-style angle encoding on a subset of PCA features.
"""
import numpy as np
import joblib
import os
import pennylane as qml
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
N_QUBITS = 4          # keep small for simulation speed
QML_FEATURES = 4      # use first 4 PCA components
QML_TRAIN_LIMIT = 200 # subsample for kernel matrix computation (simulation speed)


# ── Quantum circuit ──────────────────────────────────────────────────────────

dev = qml.device("default.qubit", wires=N_QUBITS)

#_feature_map is the function that creates the fingerprint.
@qml.qnode(dev)
def _feature_map(x):
    """ZZ-style angle encoding feature map."""
    for i in range(N_QUBITS):
        qml.Hadamard(wires=i)
        qml.RZ(2.0 * x[i], wires=i)
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=i + 1)
        qml.CNOT(wires=[i, i + 1])
    return qml.state()


def _quantum_kernel(x1, x2):
    """Fidelity kernel: |<phi(x1)|phi(x2)>|^2"""
    s1 = _feature_map(x1)
    s2 = _feature_map(x2)
    return float(np.abs(np.dot(np.conj(s1), s2)) ** 2)


def _build_kernel_matrix(X1, X2):
    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = _quantum_kernel(X1[i], X2[j])
        if i % 50 == 0:
            print(f"  [QKernel] row {i}/{n1}")
    return K


# ── Training ─────────────────────────────────────────────────────────────────

def train_qsvm(X_train, y_train, X_test, y_test):
    print("\n[QML] Training Quantum Kernel SVM...")

    # Use only first QML_FEATURES PCA dims
    X_tr = X_train[:, :QML_FEATURES]
    X_te = X_test[:, :QML_FEATURES]

    # Subsample training set (balanced)
    fraud_idx = np.where(y_train == 1)[0]
    legit_idx = np.where(y_train == 0)[0]
    half = QML_TRAIN_LIMIT // 2
    sel_train = np.concatenate([
        np.random.choice(fraud_idx, min(half, len(fraud_idx)), replace=False),
        np.random.choice(legit_idx, min(half, len(legit_idx)), replace=False)
    ])
    np.random.shuffle(sel_train)
    X_sub, y_sub = X_tr[sel_train], y_train[sel_train]

    # Subsample test set for evaluation only (keep tractable)
    QML_TEST_LIMIT = 200
    fraud_te_idx = np.where(y_test == 1)[0]
    legit_te_idx = np.where(y_test == 0)[0]
    half_te = QML_TEST_LIMIT // 2
    sel_test = np.concatenate([
        np.random.choice(fraud_te_idx, min(half_te, len(fraud_te_idx)), replace=False),
        np.random.choice(legit_te_idx, min(half_te, len(legit_te_idx)), replace=False)
    ])
    np.random.shuffle(sel_test)
    X_te_sub, y_te_sub = X_te[sel_test], y_test[sel_test]

    # Scale to [0, pi]
    qscaler = MinMaxScaler(feature_range=(0, np.pi))
    X_sub_s = qscaler.fit_transform(X_sub)
    X_te_s = qscaler.transform(X_te_sub)

    print(f"  [QML] Building train kernel ({len(X_sub)}x{len(X_sub)})...")
    K_train = _build_kernel_matrix(X_sub_s, X_sub_s)

    svm = SVC(kernel='precomputed', probability=True, class_weight='balanced', random_state=42)
    svm.fit(K_train, y_sub)

    print(f"  [QML] Building test kernel ({len(X_te_s)}x{len(X_sub_s)})...")
    K_test = _build_kernel_matrix(X_te_s, X_sub_s)

    y_pred = svm.predict(K_test)
    y_prob = svm.predict_proba(K_test)[:, 1]
    auc = roc_auc_score(y_te_sub, y_prob)
    print(f"[QSVM] AUC: {auc:.4f}")
    print(classification_report(y_te_sub, y_pred, target_names=['Legit', 'Fraud']))

    joblib.dump({'svm': svm, 'qscaler': qscaler, 'X_support': X_sub_s},
                os.path.join(MODELS_DIR, 'qsvm.pkl'))
    return svm, qscaler, X_sub_s


def predict_qml(X_pca: np.ndarray):
    """Returns qml_score and qml_confidence."""
    artifact = joblib.load(os.path.join(MODELS_DIR, 'qsvm.pkl'))
    svm = artifact['svm']
    qscaler = artifact['qscaler']
    X_support = artifact['X_support']

    X_q = X_pca[:, :QML_FEATURES]
    X_q_s = qscaler.transform(X_q)

    K = _build_kernel_matrix(X_q_s, X_support)
    prob = svm.predict_proba(K)[0][1]
    confidence = abs(prob - 0.5) * 2

    return {
        'qml_score': float(prob),
        'qml_confidence': float(confidence)
    }
