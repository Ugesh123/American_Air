"""
Intent classifier evaluation against the real 200-example golden set.
Compares baselines/simple_ml_baseline.py vs the final classifier's predictions
(from results/pipeline_outputs_golden_set.json), all against real gold_intent
labels -- nothing here is estimated or invented.

Usage:
    python3 evaluation/metrics/intent_metrics.py
"""
import json
import os

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALL_INTENTS = ['FLIGHT_STATUS_DELAY_CANCEL', 'BOOKING_CHANGE', 'BAGGAGE', 'REFUND_COMPENSATION',
               'CHECKIN_BOARDING', 'LOYALTY_MILES', 'ACCOUNT_WEBSITE_APP', 'SAFETY_SECURITY',
               'COMPLIMENT', 'COMPLAINT_GENERAL', 'OTHER_UNCLEAR']


def evaluate(gold, pred, labels=ALL_INTENTS, system_name=''):
    acc = accuracy_score(gold, pred)
    macro_f1 = f1_score(gold, pred, labels=labels, average='macro', zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(gold, pred, labels=labels)

    per_class = {}
    for i, lab in enumerate(labels):
        per_class[lab] = {
            'precision': round(float(precision[i]), 3),
            'recall': round(float(recall[i]), 3),
            'f1': round(float(f1[i]), 3),
            'support': int(support[i]),
        }

    result = {
        'system': system_name,
        'n': len(gold),
        'accuracy': round(float(acc), 4),
        'macro_f1': round(float(macro_f1), 4),
        'per_class': per_class,
        'confusion_matrix': {'labels': labels, 'matrix': cm.tolist()},
    }
    return result


def print_summary(result):
    print(f"\n=== {result['system']} ===")
    print(f"n={result['n']}  accuracy={result['accuracy']:.4f}  macro_f1={result['macro_f1']:.4f}")
    print(f"{'class':<28} {'P':>6} {'R':>6} {'F1':>6} {'n':>4}")
    for cls, m in result['per_class'].items():
        print(f"{cls:<28} {m['precision']:>6.2f} {m['recall']:>6.2f} {m['f1']:>6.2f} {m['support']:>4}")


def main():
    # baseline
    baseline_preds = json.load(open(os.path.join(ROOT, 'results', 'simple_ml_baseline_predictions.json'), encoding='utf-8'))
    gold_b = [r['gold_intent'] for r in baseline_preds]
    pred_b = [r['pred_intent'] for r in baseline_preds]
    baseline_result = evaluate(gold_b, pred_b, system_name='Baseline 3: TF-IDF unigrams + LogisticRegression (weak-supervised)')
    print_summary(baseline_result)

    # trivial baseline 1 (majority)
    trivial = json.load(open(os.path.join(ROOT, 'results', 'trivial_baseline_predictions.json'), encoding='utf-8'))
    t1 = trivial['baseline_1_majority_always_auto']
    gold_t = [r['gold_intent'] for r in t1]
    pred_t = [r['pred_intent'] for r in t1]
    trivial_result = evaluate(gold_t, pred_t, system_name='Baseline 1: Majority-class intent')
    print_summary(trivial_result)

    # final classifier (from pipeline output)
    pipeline_out = json.load(open(os.path.join(ROOT, 'results', 'pipeline_outputs_golden_set.json'), encoding='utf-8'))
    gold_f = [r['gold_intent'] for r in pipeline_out]
    pred_f = [r['intent'] for r in pipeline_out]
    final_result = evaluate(gold_f, pred_f, system_name='Final classifier: TF-IDF (uni+bi) + Calibrated LinearSVC (weak-supervised)')
    print_summary(final_result)

    out = {
        'trivial_majority': trivial_result,
        'simple_ml_baseline': baseline_result,
        'final_classifier': final_result,
    }
    out_path = os.path.join(ROOT, 'results', 'intent_classifier_metrics.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {out_path}')
    return out


if __name__ == '__main__':
    main()
