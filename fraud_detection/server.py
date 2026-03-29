"""
server.py  —  Flask backend for Bhadra Q frontend
"""
import warnings
warnings.filterwarnings('ignore')

import os, sys, json, time, threading
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

from preprocessor import transform_single
from classical_model import predict_classical
from qml_model import predict_qml
from rag_explainer import RAGExplainer
from fusion_engine import fuse_scores, rag_pattern_to_score

app = Flask(__name__, static_folder='static')
CORS(app)

# Load RAG once at startup
print("[Server] Loading RAG explainer...")
rag = RAGExplainer()
print("[Server] Ready.")


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    txn = request.json
#we send each models result
    def stream():
        try:
            # ── Step 1: Preprocess ──────────────────────────────────────
            yield _event('progress', {'step': 'preprocess', 'label': 'Preprocessing transaction...', 'pct': 10})
            X_pca = transform_single(txn)

            # ── Step 2: Classical ───────────────────────────────────────
            yield _event('progress', {'step': 'classical', 'label': 'Running Classical Models (LR + XGBoost)...', 'pct': 30})
            time.sleep(0.4)
            classical = predict_classical(X_pca)
            yield _event('classical', {
                'lr_score':       round(classical['lr_score'] * 100, 1),
                'xgb_score':      round(classical['xgb_score'] * 100, 1),
                'ensemble_score': round(classical['ensemble_score'] * 100, 1),
                'confidence':     round(classical['xgb_confidence'] * 100, 1),
                'prediction':     classical['prediction']
            })

            # ── Step 3: QML ─────────────────────────────────────────────
            yield _event('progress', {'step': 'qml', 'label': 'Running Quantum Kernel SVM...', 'pct': 60})
            qml = predict_qml(X_pca)
            yield _event('qml', {
                'score':      round(qml['qml_score'] * 100, 1),
                'confidence': round(qml['qml_confidence'] * 100, 1)
            })

            # ── Step 4: RAG ─────────────────────────────────────────────
            yield _event('progress', {'step': 'rag', 'label': 'Running RAG Explainer...', 'pct': 80})
            time.sleep(0.3)
            preview = (classical['ensemble_score'] + qml['qml_score']) / 2
            rag_result = rag.explain(txn, preview)
            rag_score, rag_conf = rag_pattern_to_score(rag_result['retrieved_patterns'])
            yield _event('rag', {
                'score':      round(rag_score * 100, 1),
                'confidence': round(rag_conf * 100, 1),
                'explanation': rag_result['explanation'],
                'recommendation': rag_result['top_recommendation'],
                'patterns': [
                    {'pattern': p['pattern'], 'relevance': round(p['relevance'] * 100, 1)}
                    for p in rag_result['retrieved_patterns']
                ]
            })

            # ── Step 5: Fusion ──────────────────────────────────────────
            yield _event('progress', {'step': 'fusion', 'label': 'Fusing scores with dynamic weights...', 'pct': 95})
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
            yield _event('fusion', {
                'final_score':      round(fusion['final_score'] * 100, 1),
                'final_confidence': round(fusion['final_confidence'] * 100, 1),
                'final_prediction': fusion['final_prediction'],
                'weights': {k: round(v * 100, 1) for k, v in fusion['weights'].items()},
                'component_scores': {k: round(v * 100, 1) for k, v in fusion['component_scores'].items()},
            })

            yield _event('progress', {'step': 'done', 'label': 'Analysis complete.', 'pct': 100})

        except Exception as e:
            yield _event('error', {'message': str(e)})

    return Response(stream(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


def _event(name, data):
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


if __name__ == '__main__':
    app.run(debug=False, port=5000, threaded=True)
