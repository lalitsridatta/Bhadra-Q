"""
fusion_engine.py
Dynamic weight fusion of classical (XGBoost), QML (QSVM), and RAG scores.
Weights are computed based on per-model confidence scores.
"""
import numpy as np


def _softmax(x: list) -> np.ndarray:
    e = np.exp(np.array(x) - np.max(x))
    return e / e.sum()


def fuse_scores(
    xgb_score: float,
    xgb_confidence: float,
    qml_score: float,
    qml_confidence: float,
    rag_score: float = None,
    rag_confidence: float = None,
    verbose: bool = True
) -> dict:
    """
    Dynamically weight each model's fraud score by its confidence.
    
    - xgb_score / qml_score: fraud probability [0,1]
    - *_confidence: how confident the model is [0,1] (distance from 0.5 boundary)
    - rag_score: optional soft score derived from RAG pattern relevance [0,1]
    
    Returns final_score, final_prediction, weights used.
    """
    scores = [xgb_score, qml_score]
    confidences = [xgb_confidence, qml_confidence]
    labels = ['XGBoost', 'QSVM']

    if rag_score is not None and rag_confidence is not None:
        scores.append(rag_score)
        confidences.append(rag_confidence)
        labels.append('RAG')

    # Dynamic weights via softmax over confidence values
    weights = _softmax(confidences)

    final_score = float(np.dot(weights, scores))
    final_prediction = int(final_score >= 0.5)
    final_confidence = abs(final_score - 0.5) * 2  # 0..1

    if verbose:
        print("\n[Fusion] Score breakdown:")
        for label, score, conf, w in zip(labels, scores, confidences, weights):
            print(f"  {label:10s} | score={score:.4f} | confidence={conf:.4f} | weight={w:.4f}")
        print(f"  {'FINAL':10s} | score={final_score:.4f} | confidence={final_confidence:.4f} | prediction={'FRAUD' if final_prediction else 'LEGIT'}")

    return {
        'final_score': final_score,
        'final_prediction': final_prediction,
        'final_confidence': final_confidence,
        'weights': {label: float(w) for label, w in zip(labels, weights)},
        'component_scores': {label: float(s) for label, s in zip(labels, scores)},
        'component_confidences': {label: float(c) for label, c in zip(labels, confidences)}
    }


def rag_pattern_to_score(retrieved_patterns: list) -> tuple:
    """
    Convert RAG retrieved pattern relevance scores into a soft fraud score.
    Returns (rag_score, rag_confidence).
    """
    if not retrieved_patterns:
        return 0.5, 0.0

    # Weighted average of relevance scores (cosine similarity, 0..1)
    relevances = [p['relevance'] for p in retrieved_patterns]
    top_relevance = relevances[0] if relevances else 0.0

    # Map cosine similarity to fraud probability
    # High relevance to fraud patterns => higher fraud score
    rag_score = min(0.95, 0.3 + top_relevance * 0.7)
    rag_confidence = top_relevance  # use relevance as confidence proxy

    return float(rag_score), float(rag_confidence)
