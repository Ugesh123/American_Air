"""
Baselines 1 & 2 (trivial): majority-class intent prediction, paired with two
different fixed escalation policies, to bound the problem from both directions.

Baseline 1 (majority + always-AUTO_HANDLE): the naive "just automate everything"
floor -- shows what recall on escalation looks like if you never escalate.

Baseline 2 (majority + always-ESCALATE): the naive "when in doubt, always use a
human" safety-reference floor -- perfect escalation recall, zero automation.

IMPORTANT (documented in DECISIONS.md): the "majority class" is computed from
the golden set's OWN gold_intent distribution, because no other intent-labeled
data exists (train/dev are unlabeled -- see the intent classifier's use of weak
supervision for why). This makes the majority-class baseline a same-set floor,
not an independently-fit model. That's a real, disclosed limitation of a
majority-class baseline under this data constraint, not an error.

Usage:
    python3 baselines/trivial_baseline.py
"""
import json
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_CANDIDATES = os.path.join(ROOT, 'data', 'golden_set', 'candidates_200.jsonl')
GOLDEN_LABELS = os.path.join(ROOT, 'data', 'golden_set', 'labels.json')

CANNED_REPLIES = {
    'FLIGHT_STATUS_DELAY_CANCEL': "Thanks for reaching out -- we're looking into your flight's status and will update you shortly.",
    'BOOKING_CHANGE': "We can help with your booking change -- please DM your confirmation number and we'll take a look.",
    'BAGGAGE': "Sorry about your baggage issue -- please DM your bag claim details so we can help track this down.",
    'REFUND_COMPENSATION': "We understand you're looking for a refund/compensation -- please DM your confirmation number so we can review.",
    'CHECKIN_BOARDING': "Sorry for the check-in/boarding trouble -- please DM your confirmation number so we can assist.",
    'LOYALTY_MILES': "Thanks for being an AAdvantage member -- please DM your account details so we can look into this.",
    'ACCOUNT_WEBSITE_APP': "Sorry you're running into a technical issue -- please DM details so we can help troubleshoot.",
    'SAFETY_SECURITY': "We take this very seriously -- please DM us right away so we can connect you with the right team.",
    'COMPLIMENT': "Thank you so much for the kind words -- we'll be sure to share this with the team!",
    'COMPLAINT_GENERAL': "We're sorry to hear about your experience -- please DM us more details so we can make this right.",
    'OTHER_UNCLEAR': "Thanks for reaching out -- could you share a bit more detail so we can help?",
}


def load_golden_set():
    candidates = {}
    with open(GOLDEN_CANDIDATES, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            candidates[r['pair_id']] = r
    labels = json.load(open(GOLDEN_LABELS, encoding='utf-8'))
    merged = []
    for pid, lab in labels.items():
        c = candidates[pid]
        merged.append({**c, **lab})
    return merged


def run():
    golden = load_golden_set()
    intent_counts = Counter(g['gold_intent'] for g in golden)
    majority_intent = intent_counts.most_common(1)[0][0]
    print(f'Majority intent (from golden set\'s own distribution): {majority_intent} '
          f'({intent_counts[majority_intent]}/{len(golden)} = '
          f'{100*intent_counts[majority_intent]/len(golden):.1f}%)')

    results = {'baseline_1_majority_always_auto': [], 'baseline_2_majority_always_escalate': []}
    for g in golden:
        pred_intent = majority_intent
        pred_reply = CANNED_REPLIES[majority_intent]
        results['baseline_1_majority_always_auto'].append({
            'pair_id': g['pair_id'], 'gold_intent': g['gold_intent'], 'pred_intent': pred_intent,
            'gold_escalation': g['gold_escalation'], 'pred_escalation': False, 'pred_reply': pred_reply,
        })
        results['baseline_2_majority_always_escalate'].append({
            'pair_id': g['pair_id'], 'gold_intent': g['gold_intent'], 'pred_intent': pred_intent,
            'gold_escalation': g['gold_escalation'], 'pred_escalation': True, 'pred_reply': None,
        })

    out_path = os.path.join(ROOT, 'results', 'trivial_baseline_predictions.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'Wrote {out_path}')
    return results


if __name__ == '__main__':
    run()
