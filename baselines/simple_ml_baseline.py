"""
Baseline 3 (simple ML, non-LLM): TF-IDF + Logistic Regression intent classifier,
trained via weak supervision (see src/intent/taxonomy_keywords.py) on the 25,451
unlabeled train pairs, evaluated on the 200 real golden-set labels. Reply
"generation" here is literally the single nearest historical AA reply by TF-IDF
cosine similarity -- no LLM, no generation, just retrieval. Escalation is a
simple keyword rule (not the final deterministic policy in src/escalation/).

Usage:
    python3 baselines/simple_ml_baseline.py
"""
import json
import os
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from intent.taxonomy_keywords import weak_label, ALL_INTENTS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(ROOT, 'data', 'processed', 'train.jsonl')
GOLDEN_CANDIDATES = os.path.join(ROOT, 'data', 'golden_set', 'candidates_200.jsonl')
GOLDEN_LABELS = os.path.join(ROOT, 'data', 'golden_set', 'labels.json')

SIMPLE_ESCALATION_KEYWORDS_RE = re.compile(
    r'\b(lawyer|sue|discriminat\w*|assault|threat|emergency|unsafe|refund|compensat\w*)\b', re.IGNORECASE)


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding='utf-8')]


def load_golden_set():
    candidates = {json.loads(l)['pair_id']: json.loads(l) for l in open(GOLDEN_CANDIDATES, encoding='utf-8')}
    labels = json.load(open(GOLDEN_LABELS, encoding='utf-8'))
    return [{**candidates[pid], **lab} for pid, lab in labels.items()]


def run():
    train = load_jsonl(TRAIN_PATH)
    golden = load_golden_set()

    print(f'Train pairs (unlabeled -> weak-supervised): {len(train)}')
    print(f'Golden set (real labels, eval only): {len(golden)}')

    train_texts = [r['customer_text_clean'] for r in train]
    weak_labels = [weak_label(t) for t in train_texts]

    # -------- intent classifier: TF-IDF unigrams + Logistic Regression (baseline config) --------
    vec = TfidfVectorizer(max_features=3000, min_df=2)
    X_train = vec.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, weak_labels)

    golden_texts = [g['customer_text_clean'] for g in golden]
    X_golden = vec.transform(golden_texts)
    pred_intents = clf.predict(X_golden)
    pred_probs = clf.predict_proba(X_golden)
    class_order = list(clf.classes_)

    # -------- nearest-historical-reply retrieval (TF-IDF cosine similarity, no generation) --------
    sims_full = cosine_similarity(X_golden, X_train)
    top1_idx = sims_full.argmax(axis=1)
    top1_sim = sims_full.max(axis=1)

    results = []
    for i, g in enumerate(golden):
        conf = float(pred_probs[i][class_order.index(pred_intents[i])])
        nn_idx = int(top1_idx[i])
        results.append({
            'pair_id': g['pair_id'],
            'gold_intent': g['gold_intent'],
            'pred_intent': pred_intents[i],
            'pred_intent_confidence': conf,
            'gold_escalation': g['gold_escalation'],
            'pred_escalation': bool(SIMPLE_ESCALATION_KEYWORDS_RE.search(g['customer_text_clean'])),
            'retrieved_reply': train[nn_idx]['aa_text_clean'],
            'retrieved_customer_msg': train[nn_idx]['customer_text_clean'],
            'retrieval_similarity': float(top1_sim[i]),
        })

    out_path = os.path.join(ROOT, 'results', 'simple_ml_baseline_predictions.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote {out_path}')

    acc = np.mean([r['pred_intent'] == r['gold_intent'] for r in results])
    print(f'Quick check -- raw intent accuracy vs golden set: {acc:.3f} '
          f'(full metrics computed separately by evaluation/metrics/intent_metrics.py)')
    return results


if __name__ == '__main__':
    run()
