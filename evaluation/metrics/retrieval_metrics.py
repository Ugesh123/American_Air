"""
Retrieval quality evaluation. IMPORTANT LIMITATION (stated here, not buried):
train.jsonl has no gold intent labels (see DECISIONS.md), so we cannot compute
a true "did retrieval return topically-correct evidence" metric against real
labels on the train side. What we CAN measure honestly:
  1. The similarity-score distribution on real golden-set queries (are we
     finding decent matches at all, or mostly near-zero noise?).
  2. A PROXY topical-match rate: does the top-1 retrieved case's weak-label
     (keyword heuristic -- same caveat as classifier training) match the
     golden-set query's REAL gold_intent? This is explicitly a proxy, bounded
     by the weak labeler's own ~53-66% accuracy (see intent_classifier_metrics),
     not a clean ground-truth retrieval metric.

Usage: python3 evaluation/metrics/retrieval_metrics.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
from intent.taxonomy_keywords import weak_label

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    pipeline_out = json.load(open(os.path.join(ROOT, 'results', 'pipeline_outputs_golden_set.json'), encoding='utf-8'))

    top1_sims = []
    n_evidence = []
    proxy_match = []
    n_zero_evidence = 0
    for r in pipeline_out:
        ev = r['evidence']
        n_evidence.append(len(ev))
        if not ev:
            n_zero_evidence += 1
            continue
        top1_sims.append(ev[0]['similarity'])
        top1_weak_intent = weak_label(ev[0]['historical_customer_msg'])
        proxy_match.append(top1_weak_intent == r['gold_intent'])

    top1_sims = np.array(top1_sims)
    print('=== Retrieval quality (TF-IDF backend) ===')
    print(f'Golden-set queries: {len(pipeline_out)}')
    print(f'Queries with ZERO evidence above min_similarity threshold: {n_zero_evidence} '
          f'({100*n_zero_evidence/len(pipeline_out):.1f}%)')
    print(f'Avg evidence count returned (when non-zero): {np.mean(n_evidence):.2f}')
    print(f'Top-1 similarity: min={top1_sims.min():.3f} p25={np.percentile(top1_sims,25):.3f} '
          f'median={np.percentile(top1_sims,50):.3f} p75={np.percentile(top1_sims,75):.3f} max={top1_sims.max():.3f}')
    print(f'\nPROXY topical-match rate (top-1 retrieved case\'s weak-label == query\'s real gold_intent): '
          f'{sum(proxy_match)}/{len(proxy_match)} ({100*sum(proxy_match)/len(proxy_match):.1f}%)')
    print('(Bounded above by the weak labeler\'s own ~53-66% accuracy -- see intent_classifier_metrics.json. '
          'This is a proxy, not a clean ground-truth retrieval metric.)')

    result = {
        'n_queries': len(pipeline_out),
        'n_zero_evidence': n_zero_evidence,
        'avg_evidence_count': float(np.mean(n_evidence)),
        'top1_similarity': {
            'min': float(top1_sims.min()), 'p25': float(np.percentile(top1_sims, 25)),
            'median': float(np.percentile(top1_sims, 50)), 'p75': float(np.percentile(top1_sims, 75)),
            'max': float(top1_sims.max()),
        },
        'proxy_topical_match_rate': sum(proxy_match) / len(proxy_match),
        'proxy_match_caveat': "Bounded by weak labeler's own accuracy (~53-66%), not a clean retrieval metric.",
    }
    out_path = os.path.join(ROOT, 'results', 'retrieval_metrics.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f'\nWrote {out_path}')
    return result


if __name__ == '__main__':
    main()
