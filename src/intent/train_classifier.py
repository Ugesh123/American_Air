"""
Final intent classifier for the deployed agent. Same weak-supervision training
approach as the baseline (see DECISIONS.md), but with two real, modest
improvements over baselines/simple_ml_baseline.py:
  1. TF-IDF features include unigrams+bigrams (vs unigrams only), more max_features.
  2. Calibrated LinearSVC (vs plain Logistic Regression) -- LinearSVC's margin-based
     decision often handles the class imbalance in weak-labeled data a bit better;
     CalibratedClassifierCV gives us back well-formed probabilities for
     intent_confidence, which the escalation policy needs.

This is deliberately NOT a large architectural leap over the baseline -- the
assignment brief explicitly asks us to prefer simple systems with strong
evaluation over complex ones. The comparison table in evaluation/metrics
reports honestly whether this actually beats the baseline; if it doesn't, that
is reported as-is, not hidden.

Usage:
    python3 src/intent/train_classifier.py
"""
import json
import os
import sys

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from intent.taxonomy_keywords import weak_label

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_PATH = os.path.join(ROOT, 'data', 'processed', 'train.jsonl')
MODEL_DIR = os.path.join(ROOT, 'results', 'models')


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding='utf-8')]


def train():
    train = load_jsonl(TRAIN_PATH)
    texts = [r['customer_text_clean'] for r in train]
    labels = [weak_label(t) for t in texts]

    print(f'Training on {len(texts)} weak-supervised pairs (pseudo-labels via keyword heuristic)')

    vec = TfidfVectorizer(max_features=8000, min_df=2, ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(texts)

    base_clf = LinearSVC(C=1.0, max_iter=5000)
    clf = CalibratedClassifierCV(base_clf, cv=3)
    clf.fit(X, labels)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vec, os.path.join(MODEL_DIR, 'intent_vectorizer.joblib'))
    joblib.dump(clf, os.path.join(MODEL_DIR, 'intent_classifier.joblib'))
    print(f'Saved model + vectorizer to {MODEL_DIR}')
    return vec, clf


def predict(vec, clf, texts):
    X = vec.transform(texts)
    probs = clf.predict_proba(X)
    classes = list(clf.classes_)
    preds = [classes[p.argmax()] for p in probs]
    confs = [float(p.max()) for p in probs]
    return preds, confs


if __name__ == '__main__':
    train()
