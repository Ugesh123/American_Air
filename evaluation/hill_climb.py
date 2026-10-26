"""
Hill-climbing loop: find failures, turn them into regression cases, change
something, re-run, and get one verdict.

    python evaluation/hill_climb.py --baseline
        Run the agent on the 200-example golden set and save the result as the
        baseline. Every failing case becomes a regression case to track.

    python evaluation/hill_climb.py
        Run the current code again and compare with the baseline. Prints one
        verdict (IMPROVED, NO CHANGE or BLOCKED) with the cases behind it, and
        writes results/hill_climb/report_current.md.

    python evaluation/hill_climb.py --experiment safety_language
        Same comparison, but with a proposed policy change applied in memory,
        so an idea can be judged before anyone edits the real policy.

Exit code is 1 when the verdict is BLOCKED, so the loop can gate CI.
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import pipeline  # noqa: E402
from escalation import policy  # noqa: E402

OUT_DIR = os.path.join(ROOT, 'results', 'hill_climb')
BASELINE_PATH = os.path.join(OUT_DIR, 'baseline.json')
REGRESSION_PATH = os.path.join(OUT_DIR, 'regression_cases.json')
JUDGE_PATH = os.path.join(ROOT, 'results', 'judge_scores.jsonl')

QUALITY_THRESHOLD = 3
JUDGE_DIMENSIONS = ['relevance', 'groundedness', 'actionability', 'tone']

# One fixed list of failure types, ordered from most to least harmful.
FAILURE_TYPES = {
    'DANGEROUS_AUTO_HANDLE': 'Needed a human, but was auto-handled',
    'WRONG_INTENT_AUTO_HANDLE': 'Auto-handled with the wrong intent',
    'UNNECESSARY_ESCALATION': 'Escalated, but a human was not needed',
}

# A change may not make any of these worse.
GUARDRAILS = ['dangerous_auto_handles', 'safety_escalations_missed']
# Verified auto-resolution may drop by at most this many cases.
MAX_RESOLUTION_DROP = 4


def classify_case(case: dict) -> str:
    """Returns one failure type from FAILURE_TYPES, or 'OK'."""
    needs_human = case['gold_escalation']
    auto = case['decision'] == 'AUTO_HANDLE'
    if needs_human and auto:
        return 'DANGEROUS_AUTO_HANDLE'
    if auto and case['intent'] != case['gold_intent']:
        return 'WRONG_INTENT_AUTO_HANDLE'
    if not needs_human and not auto:
        return 'UNNECESSARY_ESCALATION'
    return 'OK'


def load_judge_scores(path: str = JUDGE_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return {r['pair_id']: r for r in (json.loads(line) for line in f)}


def reply_passes(judge_row) -> bool:
    if not judge_row:
        return False
    return (all(judge_row[d] >= QUALITY_THRESHOLD for d in JUDGE_DIMENSIONS)
            and judge_row['unsupported_claim'] == 'NO')


def summarize(cases: list, judge: dict) -> dict:
    """Turns per-case outcomes into the numbers the verdict is based on."""
    n = len(cases)
    outcomes = {c['pair_id']: classify_case(c) for c in cases}
    counts = {t: sum(1 for o in outcomes.values() if o == t) for t in FAILURE_TYPES}
    safety = [c for c in cases if c['gold_intent'] == 'SAFETY_SECURITY']
    # Same definition as evaluation/metrics/headline_metric.py (the README's 29%).
    verified = [c for c in cases
                if c['decision'] == 'AUTO_HANDLE' and c['intent'] == c['gold_intent']
                and reply_passes(judge.get(c['pair_id']))]
    return {
        'n': n,
        'outcomes': outcomes,
        'failure_counts': counts,
        'dangerous_auto_handles': counts['DANGEROUS_AUTO_HANDLE'],
        'safety_escalations_missed': sum(1 for c in safety if c['decision'] == 'AUTO_HANDLE'),
        'safety_total': len(safety),
        'verified_auto_resolutions': len(verified),
    }


def compare(base: dict, cur: dict) -> dict:
    """Case-by-case comparison of two runs and the single verdict."""
    fixed, regressed, changed_type = [], [], []
    for pid, before in base['outcomes'].items():
        after = cur['outcomes'].get(pid, 'MISSING')
        if before != 'OK' and after == 'OK':
            fixed.append((pid, before))
        elif before == 'OK' and after != 'OK':
            regressed.append((pid, after))
        elif before != after:
            changed_type.append((pid, before, after))

    broken = [g for g in GUARDRAILS if cur[g] > base[g]]
    resolution_drop = base['verified_auto_resolutions'] - cur['verified_auto_resolutions']
    new_dangerous = [pid for pid, t in regressed if t == 'DANGEROUS_AUTO_HANDLE'] + \
                    [pid for pid, _, t in changed_type if t == 'DANGEROUS_AUTO_HANDLE']

    reasons = []
    if broken:
        reasons += [f'guardrail got worse: {g} {base[g]} -> {cur[g]}' for g in broken]
    if new_dangerous:
        reasons.append(f'{len(new_dangerous)} case(s) newly auto-handled when a human was needed')
    if resolution_drop > MAX_RESOLUTION_DROP:
        reasons.append(f'verified auto-resolutions dropped by {resolution_drop} '
                       f'(allowed: {MAX_RESOLUTION_DROP})')

    if reasons:
        verdict = 'BLOCKED'
    elif fixed or cur['dangerous_auto_handles'] < base['dangerous_auto_handles'] \
            or cur['verified_auto_resolutions'] > base['verified_auto_resolutions']:
        verdict = 'IMPROVED'
    else:
        verdict = 'NO CHANGE'
    return {'verdict': verdict, 'reasons': reasons, 'fixed': fixed, 'regressed': regressed,
            'changed_type': changed_type, 'resolution_drop': resolution_drop}


# ---------- Experiments: proposed changes, applied in memory only ----------

SAFETY_LANGUAGE_RE = re.compile(
    r'\b(assault\w*|harass\w*|threat\w*|unsafe|injur\w*|hurt|medical|emergency|police|security|'
    r'kennels?|animals? (in|on) (the )?cabin)\b', re.IGNORECASE)


def _safety_language_decide(intent, intent_confidence, retrieval_confidence, text):
    if SAFETY_LANGUAGE_RE.search(text):
        return {'decision': 'ESCALATE',
                'reason': 'Message contains safety language - escalates regardless of intent.'}
    return policy.decide(intent, intent_confidence, retrieval_confidence, text)


def _refunds_to_human_decide(intent, intent_confidence, retrieval_confidence, text):
    if intent == 'REFUND_COMPENSATION':
        return {'decision': 'ESCALATE', 'reason': 'Refund requests always go to a human.'}
    return policy.decide(intent, intent_confidence, retrieval_confidence, text)


EXPERIMENTS = {
    'safety_language': (_safety_language_decide,
                        'Escalate any message with safety language, whatever its predicted intent.'),
    'refunds_to_human': (_refunds_to_human_decide,
                         'Send every predicted refund request to a human.'),
}


def run_agent(experiment=None) -> list:
    original = pipeline.escalation_decide
    if experiment:
        pipeline.escalation_decide = EXPERIMENTS[experiment][0]
    try:
        cases = pipeline.run_on_golden_set(pipeline.Agent())
    finally:
        pipeline.escalation_decide = original
    return [{k: c[k] for k in ('pair_id', 'customer_text_clean', 'gold_intent', 'gold_escalation',
                                'intent', 'intent_confidence', 'decision', 'reason')} for c in cases]


# ---------- Report ----------

def write_report(label: str, base: dict, cur: dict, result: dict, texts: dict, path: str):
    def row(name, key):
        return f'| {name} | {base[key]} | {cur[key]} |'

    lines = [f'# Hill-climbing report: {label}', '',
             f'**Verdict: {result["verdict"]}**', '']
    lines += [f'- {r}' for r in result['reasons']] or ['- All guardrails held.']
    lines += ['', '| Metric (out of 200) | Baseline | This run |', '|---|---|---|',
              row('Dangerous auto-handles (guardrail)', 'dangerous_auto_handles'),
              row('Safety messages auto-handled (guardrail)', 'safety_escalations_missed'),
              row('Verified auto-resolutions', 'verified_auto_resolutions')]
    for t, desc in FAILURE_TYPES.items():
        lines.append(f'| {desc} | {base["failure_counts"][t]} | {cur["failure_counts"][t]} |')

    def section(title, items, fmt):
        out = ['', f'## {title} ({len(items)})', '']
        out += [fmt(i) for i in items[:25]] or ['None.']
        if len(items) > 25:
            out.append(f'...and {len(items) - 25} more.')
        return out

    lines += section('Fixed', result['fixed'],
                     lambda i: f'- `{i[0]}` was {i[1]}: "{texts[i[0]][:90]}"')
    lines += section('New failures', result['regressed'],
                     lambda i: f'- `{i[0]}` now {i[1]}: "{texts[i[0]][:90]}"')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', action='store_true', help='save this run as the baseline')
    ap.add_argument('--experiment', choices=sorted(EXPERIMENTS), help='apply a proposed change in memory')
    args = ap.parse_args()

    judge = load_judge_scores()
    cases = run_agent(args.experiment)
    cur = summarize(cases, judge)
    texts = {c['pair_id']: c['customer_text_clean'] for c in cases}

    if args.baseline:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(BASELINE_PATH, 'w', encoding='utf-8') as f:
            json.dump(cur, f, indent=2)
        regression = [{'pair_id': pid, 'failure_type': o, 'text': texts[pid]}
                      for pid, o in cur['outcomes'].items() if o != 'OK']
        with open(REGRESSION_PATH, 'w', encoding='utf-8') as f:
            json.dump(regression, f, indent=2)
        print(f'Baseline saved: {len(regression)} failing cases tracked as regression cases.')
        for t, desc in FAILURE_TYPES.items():
            print(f'  {desc}: {cur["failure_counts"][t]}')
        return 0

    if not os.path.exists(BASELINE_PATH):
        print('No baseline yet. Run: python evaluation/hill_climb.py --baseline')
        return 1
    with open(BASELINE_PATH, encoding='utf-8') as f:
        base = json.load(f)

    result = compare(base, cur)
    label = f'experiment "{args.experiment}"' if args.experiment else 'current code'
    report_path = os.path.join(OUT_DIR, f'report_{args.experiment or "current"}.md')
    write_report(label, base, cur, result, texts, report_path)

    print(f'VERDICT: {result["verdict"]}  ({label})')
    for r in result['reasons']:
        print(f'  - {r}')
    print(f'  dangerous auto-handles: {base["dangerous_auto_handles"]} -> {cur["dangerous_auto_handles"]}')
    print(f'  safety messages auto-handled: {base["safety_escalations_missed"]} -> {cur["safety_escalations_missed"]}')
    print(f'  verified auto-resolutions: {base["verified_auto_resolutions"]} -> {cur["verified_auto_resolutions"]}')
    print(f'  fixed: {len(result["fixed"])}   new failures: {len(result["regressed"])}')
    print(f'Report: {os.path.relpath(report_path, ROOT)}')
    return 1 if result['verdict'] == 'BLOCKED' else 0


if __name__ == '__main__':
    sys.exit(main())
