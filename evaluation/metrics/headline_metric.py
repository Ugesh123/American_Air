"""
Headline metric: Verified Auto-Resolution Rate.

Full definition (per the assignment brief): an example counts as verified
auto-resolution only if (1) intent is correct, (2) system chose AUTO_HANDLE,
(3) the generated reply passes a quality threshold, (4) the reply is grounded,
and (5) no unsupported claims exist.

THIS SCRIPT COMPUTES ONLY (1) AND (2) -- called "Auto-Handle Candidates" here,
not the full metric -- because (3)-(5) require a generated reply and judge
score, neither of which exist without an LLM API key (see
src/generation/generate_replies.py, evaluation/judge/run_judge.py). This is
stated as a ceiling, not the real number, in REPORT.md.

If results/judge_scores.jsonl exists (i.e. you've run generation+judging),
this script automatically computes the FULL metric instead.

Usage: python3 evaluation/metrics/headline_metric.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PIPELINE_OUT = os.path.join(ROOT, 'results', 'pipeline_outputs_golden_set.json')
JUDGE_SCORES = os.path.join(ROOT, 'results', 'judge_scores.jsonl')
QUALITY_THRESHOLD = 3  # min score (1-5) on relevance/groundedness/actionability/tone to "pass"


def main():
    pipeline_out = json.load(open(PIPELINE_OUT, encoding='utf-8'))
    intent_correct_and_auto = [r for r in pipeline_out if r['intent'] == r['gold_intent'] and r['decision'] == 'AUTO_HANDLE']

    dangerous = [r for r in pipeline_out if r['gold_escalation'] and r['decision'] == 'AUTO_HANDLE']
    dangerous_rate = len(dangerous) / len(pipeline_out)

    if os.path.exists(JUDGE_SCORES):
        judge = {r['pair_id']: r for r in (json.loads(l) for l in open(JUDGE_SCORES, encoding='utf-8'))}
        verified = 0
        for r in intent_correct_and_auto:
            j = judge.get(r['pair_id'])
            if not j:
                continue
            passes_quality = all(j[dim] >= QUALITY_THRESHOLD for dim in
                                  ['relevance', 'groundedness', 'actionability', 'tone'])
            no_unsupported = j['unsupported_claim'] == 'NO'
            if passes_quality and no_unsupported:
                verified += 1
        rate = verified / len(pipeline_out)
        print(f'FULL Verified Auto-Resolution Rate: {verified}/{len(pipeline_out)} = {rate:.3f}')
        result = {'metric': 'Verified Auto-Resolution Rate (FULL)', 'verified_count': verified,
                   'n': len(pipeline_out), 'rate': rate,
                   'dangerous_failure_rate': dangerous_rate, 'dangerous_failure_count': len(dangerous)}
    else:
        rate = len(intent_correct_and_auto) / len(pipeline_out)
        print(f'PARTIAL metric -- "Auto-Handle Candidates" (intent correct AND AUTO_HANDLE): '
              f'{len(intent_correct_and_auto)}/{len(pipeline_out)} = {rate:.3f}')
        print('This is a CEILING on the true Verified Auto-Resolution Rate -- '
              'run generate_replies.py + run_judge.py (requires your own API key) for the real metric.')
        result = {'metric': 'Auto-Handle Candidates (PARTIAL -- ceiling, not real metric)',
                   'count': len(intent_correct_and_auto), 'n': len(pipeline_out), 'rate': rate,
                   'note': 'Requires results/judge_scores.jsonl for the full metric (needs LLM API access).',
                   'dangerous_failure_rate': dangerous_rate, 'dangerous_failure_count': len(dangerous)}

    print(f'Dangerous failure rate (gold_escalation=True but AUTO_HANDLE): '
          f'{len(dangerous)}/{len(pipeline_out)} = {dangerous_rate:.3f}')

    out_path = os.path.join(ROOT, 'results', 'headline_metric.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
