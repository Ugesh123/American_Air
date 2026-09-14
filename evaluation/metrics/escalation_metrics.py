"""
Escalation policy evaluation against the real 200-example golden set.
Usage: python3 evaluation/metrics/escalation_metrics.py
"""
import json
import os
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    pipeline_out = json.load(open(os.path.join(ROOT, 'results', 'pipeline_outputs_golden_set.json'), encoding='utf-8'))

    gold = [bool(r['gold_escalation']) for r in pipeline_out]
    pred = [r['decision'] == 'ESCALATE' for r in pipeline_out]

    precision, recall, f1, support = precision_recall_fscore_support(gold, pred, labels=[True, False], zero_division=0)
    cm = confusion_matrix(gold, pred, labels=[True, False])

    print('=== Escalation policy: overall ===')
    print(f'n={len(gold)}  gold escalation rate: {sum(gold)}/{len(gold)} ({100*sum(gold)/len(gold):.1f}%)')
    print(f'predicted escalation rate: {sum(pred)}/{len(pred)} ({100*sum(pred)/len(pred):.1f}%)')
    print(f'ESCALATE class:   precision={precision[0]:.3f}  recall={recall[0]:.3f}  f1={f1[0]:.3f}  (n={support[0]})')
    print(f'AUTO_HANDLE class: precision={precision[1]:.3f}  recall={recall[1]:.3f}  f1={f1[1]:.3f}  (n={support[1]})')
    print(f'confusion matrix [gold x pred], labels=[ESCALATE, AUTO_HANDLE]:')
    print(cm)

    # recall specifically on SAFETY_SECURITY and other high-stakes gold-escalation cases
    safety_rows = [r for r in pipeline_out if r['gold_intent'] == 'SAFETY_SECURITY']
    safety_gold_escalate = [r for r in safety_rows if r['gold_escalation']]
    safety_recall = (sum(1 for r in safety_gold_escalate if r['decision'] == 'ESCALATE') / len(safety_gold_escalate)
                      if safety_gold_escalate else None)
    print(f'\nSAFETY_SECURITY (gold-labeled) examples: {len(safety_rows)}, of which '
          f'{len(safety_gold_escalate)} are gold_escalation=True')
    print(f'Escalation recall on those: {safety_recall}')

    # dangerous failures: gold_escalation=True but system said AUTO_HANDLE
    dangerous = [r for r in pipeline_out if r['gold_escalation'] and r['decision'] == 'AUTO_HANDLE']
    print(f'\nDANGEROUS FAILURES (gold says escalate, system said AUTO_HANDLE): {len(dangerous)} / {len(gold)} '
          f'({100*len(dangerous)/len(gold):.1f}%)')
    for d in dangerous[:10]:
        print(f"  pair_id={d['pair_id']}  gold_intent={d['gold_intent']}  pred_intent={d['intent']}  "
              f"reason_given={d['reason']!r}")

    result = {
        'n': len(gold),
        'gold_escalation_rate': sum(gold) / len(gold),
        'pred_escalation_rate': sum(pred) / len(pred),
        'escalate_precision': float(precision[0]), 'escalate_recall': float(recall[0]), 'escalate_f1': float(f1[0]),
        'auto_handle_precision': float(precision[1]), 'auto_handle_recall': float(recall[1]), 'auto_handle_f1': float(f1[1]),
        'confusion_matrix_labels': ['ESCALATE', 'AUTO_HANDLE'],
        'confusion_matrix': cm.tolist(),
        'safety_security_escalation_recall': safety_recall,
        'dangerous_failure_count': len(dangerous),
        'dangerous_failure_rate': len(dangerous) / len(gold),
        'dangerous_failure_pair_ids': [d['pair_id'] for d in dangerous],
    }
    out_path = os.path.join(ROOT, 'results', 'escalation_metrics.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f'\nWrote {out_path}')
    return result


if __name__ == '__main__':
    main()
