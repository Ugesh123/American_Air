"""
Computes human-vs-LLM-judge agreement. Requires BOTH:
  - results/judge_scores.jsonl (from evaluation/judge/run_judge.py)
  - data/judge_agreement/human_judge_scores.json (from tools/judge_agreement_tool.py)
Neither exists in this sandbox (both require your own LLM API access / your own
scoring time), so this has never been run here. Run once both exist:

    python3 evaluation/metrics/judge_agreement.py
"""
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JUDGE_PATH = os.path.join(ROOT, 'results', 'judge_scores.jsonl')
HUMAN_PATH = os.path.join(ROOT, 'data', 'judge_agreement', 'human_judge_scores.json')

DIMENSIONS = ['relevance', 'groundedness', 'actionability', 'tone']


def main():
    if not os.path.exists(JUDGE_PATH) or not os.path.exists(HUMAN_PATH):
        raise SystemExit(
            'Missing judge_scores.jsonl and/or human_judge_scores.json -- run '
            'evaluation/judge/run_judge.py and tools/judge_agreement_tool.py first '
            '(both require your own LLM API access, not available in this sandbox).')

    judge = {r['pair_id']: r for r in (json.loads(l) for l in open(JUDGE_PATH, encoding='utf-8'))}
    human = json.load(open(HUMAN_PATH, encoding='utf-8'))

    common_ids = sorted(set(judge.keys()) & set(human.keys()))
    print(f'Comparing {len(common_ids)} examples scored by both human and judge')

    results = {'n': len(common_ids), 'per_dimension': {}}
    for dim in DIMENSIONS:
        j_scores = [judge[pid][dim] for pid in common_ids]
        h_scores = [human[pid][dim] for pid in common_ids]

        exact = np.mean([j == h for j, h in zip(j_scores, h_scores)])
        within1 = np.mean([abs(j - h) <= 1 for j, h in zip(j_scores, h_scores)])
        rho, pval = spearmanr(j_scores, h_scores)

        results['per_dimension'][dim] = {
            'exact_agreement': float(exact), 'within_1_agreement': float(within1),
            'spearman_rho': float(rho), 'spearman_pvalue': float(pval),
        }
        print(f'{dim}: exact={exact:.3f}  within_1={within1:.3f}  spearman_rho={rho:.3f} (p={pval:.4f})')

    # binary unsupported_claim -- Cohen's kappa
    j_bin = [judge[pid]['unsupported_claim'] == 'YES' for pid in common_ids]
    h_bin = [human[pid]['unsupported_claim'] == 'YES' for pid in common_ids]
    kappa = cohen_kappa_score(j_bin, h_bin)
    results['unsupported_claim_cohens_kappa'] = float(kappa)
    print(f'\nunsupported_claim Cohen\'s kappa: {kappa:.3f}')

    out_path = os.path.join(ROOT, 'results', 'judge_agreement_metrics.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
