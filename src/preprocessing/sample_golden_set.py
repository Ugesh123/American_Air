"""
Build the 200-example golden-set CANDIDATE pool (labels are added later by hand)
from data/processed/eval_pool.jsonl, per the approved sampling plan.

Deterministic / reproducible: single fixed RNG seed, no external randomness.

Usage:
    python3 src/preprocessing/sample_golden_set.py \
        --eval_pool data/processed/eval_pool.jsonl \
        --train data/processed/train.jsonl \
        --output data/golden_set/candidates_200.jsonl
"""
import argparse, json, random, re
from collections import defaultdict, Counter
from datetime import datetime

SEED = 42

TAXONOMY_PRIORITY = [
    'SAFETY_SECURITY', 'REFUND_COMPENSATION', 'FLIGHT_STATUS_DELAY_CANCEL', 'BAGGAGE',
    'BOOKING_CHANGE', 'CHECKIN_BOARDING', 'ACCOUNT_WEBSITE_APP', 'LOYALTY_MILES',
    'COMPLAINT_GENERAL', 'COMPLIMENT',
]

# NOTE: these keyword patterns are a SIZING/CANDIDATE-SELECTION proxy only.
# They are never used as the gold label - the human annotator assigns gold_intent.
BUCKETS = {
    'FLIGHT_STATUS_DELAY_CANCEL': r'\b(delay(ed|s)?|cancel(led|ed|lation|s)?|on time|status|diverted|gate|departure|arriv\w*|affected flights?|flights? affected|scheduling (glitch|issue)|no pilot|missing pilot|eta|updates? on (my |the )?flight|flight (is|has) a (crew|pilot))\b',
    'BOOKING_CHANGE': r'\b(rebook|re-book|change my flight|reschedul\w*|seat change|upgrade|book(ing)?|reservation|pnr|record locator)\b',
    'REFUND_COMPENSATION': r'\b(refund|reimburs\w*|compensat\w*|voucher|credit (me|back)|money back)\b',
    'BAGGAGE': r'\b(bag(s|gage)?|luggage|suitcase|left (my |a )?\w+ (on|in) (the )?(flight|plane|seat)|lost (my )?(bag|item)|damaged bag)\b',
    'CHECKIN_BOARDING': r'\b(check[- ]?in|boarding pass|boarding|gate agent|standby|precheck|pre-check|ktn|known traveler)\b',
    'LOYALTY_MILES': r'\b(aadvantage|miles|loyalty|elite status|frequent flyer|systemwide upgrade)\b',
    'ACCOUNT_WEBSITE_APP': r'\b(app|website|log ?in|password|account (access|locked)|error message)\b',
    # Tightened per real false-positives found in the discovery pass (safety VIDEOS/hashtags != incidents)
    'SAFETY_SECURITY': r'\b(medical emergency|paramedic|ambulance|assault(ed)?|threat(en(ed)?)?|discriminat\w*|racis\w*|allerg(y|ic|ies)|tripped over|felt unsafe|security (threat|concern|incident)|groped|harass(ed|ment)?)\b',
    'COMPLIMENT': r'\b(thank you|thanks|great (service|job|crew|flight)|appreciate|awesome|amazing|kudos|top notch|so nice|so helpful|favorite (person|crew|flight attendant))\b',
    'COMPLAINT_GENERAL': r'\b(rude|terrible|worst|horrible|awful|disgust\w*|never fly|unacceptable|disappoint\w*|sucks|shame on you|verbally abused|smh|on hold|spent \d+ hours|customer service)\b',
}
COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in BUCKETS.items()}
ES_RE = re.compile(r'\b(que|para|con|muy|está|gracias|vuelo|maleta|por|los|las|del|una|así|más|porque|señorita)\b', re.IGNORECASE)

# safety-specific false-positive exclusion, applied on top of the tightened pattern above
SAFETY_FALSE_POSITIVE_RE = re.compile(
    r'\b(safety video|safety demonstration|#safetyfirst|safety is (your |our )?top priority|budget.*safety video)\b',
    re.IGNORECASE)

ESCALATION_EDGE_RE = re.compile(
    r'\b(lawyer|sue|legal action|attorney|discriminat\w*|racis\w*|never (fly|flying) (you|with you) again|report(ed)? (this|you) to|BBB|department of transportation|\bdot\b)\b',
    re.IGNORECASE)

STRATA_TARGETS = {
    'FLIGHT_STATUS_DELAY_CANCEL': 28, 'COMPLIMENT': 22, 'COMPLAINT_GENERAL': 20,
    'BAGGAGE': 18, 'BOOKING_CHANGE': 16, 'CHECKIN_BOARDING': 14, 'REFUND_COMPENSATION': 14,
    'LOYALTY_MILES': 10, 'ACCOUNT_WEBSITE_APP': 10, 'SAFETY_SECURITY': 12, 'OTHER_UNCLEAR': 16,
}
HARD_CASE_TARGETS = {'weak_retrieval': 6, 'multi_intent': 5, 'short_noisy': 4, 'escalation_edge': 5}


def load_jsonl(path):
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            out.append(json.loads(line))
    return out


def primary_bucket_hits(text):
    return [name for name in TAXONOMY_PRIORITY if COMPILED[name].search(text)]


def build(eval_pool_path, train_path, output_path):
    rng = random.Random(SEED)
    eval_pool = load_jsonl(eval_pool_path)
    train = load_jsonl(train_path)

    # ---- exclusions ----
    eligible = [r for r in eval_pool if not r['dup_customer_text_in_train'] and not r['dup_aa_text_in_train']]
    n_after_dup = len(eligible)
    eligible = [r for r in eligible if not ES_RE.search(r['customer_text_clean'])]
    n_after_lang = len(eligible)
    print(f'eval_pool={len(eval_pool)} -> after dup-flag exclusion={n_after_dup} -> after non-English exclusion={n_after_lang}')

    # index by thread for context assembly (all members of a thread live in the SAME split by construction)
    by_thread = defaultdict(list)
    for r in eval_pool:  # use full eval_pool (not just eligible) so context is complete even if a context turn was itself dup-flagged
        by_thread[r['thread_id']].append(r)
    TS_FMT = '%a %b %d %H:%M:%S %z %Y'
    for tid in by_thread:
        by_thread[tid].sort(key=lambda r: datetime.strptime(r['customer_created_at_raw'], TS_FMT))

    # ---- assign a PROXY primary bucket to every eligible record (selection aid only) ----
    proxy_bucket = {}
    multi_hit_ids = set()
    safety_true_positive_ids = set()
    for r in eligible:
        t = r['customer_text_clean']
        hits = primary_bucket_hits(t)
        if len(hits) >= 2:
            multi_hit_ids.add(r['pair_id'])
        if not hits:
            proxy_bucket[r['pair_id']] = 'OTHER_UNCLEAR'
        else:
            proxy_bucket[r['pair_id']] = hits[0]
        # genuine safety check: matches tightened pattern AND not a known false-positive phrase
        if COMPILED['SAFETY_SECURITY'].search(t) and not SAFETY_FALSE_POSITIVE_RE.search(t):
            safety_true_positive_ids.add(r['pair_id'])

    by_bucket = defaultdict(list)
    for r in eligible:
        by_bucket[proxy_bucket[r['pair_id']]].append(r)

    print('\nEligible-pool sizes by proxy bucket:')
    for k in list(STRATA_TARGETS.keys()):
        print(f'  {k}: {len(by_bucket.get(k, []))}  (target {STRATA_TARGETS[k]})')
    print(f'\nSAFETY_SECURITY genuine (non-false-positive) candidates available: {len(safety_true_positive_ids)}')

    selected = {}  # pair_id -> record (augmented)
    selection_log = []

    def take(records, n, reason, bucket_label):
        pool = [r for r in records if r['pair_id'] not in selected]
        rng.shuffle(pool)
        chosen = pool[:n]
        for r in chosen:
            selected[r['pair_id']] = r
            selection_log.append({'pair_id': r['pair_id'], 'stratum': bucket_label, 'reason': reason})
        return len(chosen)

    # ---- stratified intent sampling ----
    strata_actual = {}
    for bucket, target in STRATA_TARGETS.items():
        if bucket == 'SAFETY_SECURITY':
            candidates = [r for r in by_bucket.get(bucket, []) if r['pair_id'] in safety_true_positive_ids]
        else:
            candidates = by_bucket.get(bucket, [])
        got = take(candidates, target, 'stratified_intent', bucket)
        strata_actual[bucket] = got

    # ---- hard-case supplemental slots ----
    # weak retrieval: TF-IDF max-sim to train, meaningful length only, excludes already-selected
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    train_texts = [r['customer_text_clean'] for r in train]
    remaining = [r for r in eligible if r['pair_id'] not in selected and len(r['customer_text_clean'].split()) >= 6]
    vec = TfidfVectorizer(max_features=5000, min_df=2)
    train_X = vec.fit_transform(train_texts)
    rem_X = vec.transform([r['customer_text_clean'] for r in remaining])
    sims = np.zeros(rem_X.shape[0])
    for i in range(0, rem_X.shape[0], 500):
        sims[i:i+500] = cosine_similarity(rem_X[i:i+500], train_X).max(axis=1)
    order = np.argsort(sims)
    weak_candidates = [remaining[i] for i in order]
    got_weak = take(weak_candidates, HARD_CASE_TARGETS['weak_retrieval'], 'weak_retrieval', 'HARD_CASE')

    # multi-intent
    multi_candidates = [r for r in eligible if r['pair_id'] in multi_hit_ids and r['pair_id'] not in selected]
    got_multi = take(multi_candidates, HARD_CASE_TARGETS['multi_intent'], 'multi_intent', 'HARD_CASE')

    # short/noisy (<=5 words), excluding pure-empty (already filtered upstream)
    short_candidates = [r for r in eligible if len(r['customer_text_clean'].split()) <= 5 and r['pair_id'] not in selected]
    got_short = take(short_candidates, HARD_CASE_TARGETS['short_noisy'], 'short_noisy', 'HARD_CASE')

    # escalation edge cases outside SAFETY_SECURITY
    escalation_candidates = [r for r in eligible if ESCALATION_EDGE_RE.search(r['customer_text_clean'])
                              and r['pair_id'] not in selected]
    got_escalation = take(escalation_candidates, HARD_CASE_TARGETS['escalation_edge'], 'escalation_edge', 'HARD_CASE')

    hard_actual = {'weak_retrieval': got_weak, 'multi_intent': got_multi,
                   'short_noisy': got_short, 'escalation_edge': got_escalation}

    print('\nHard-case supplemental slots filled:')
    for k, v in hard_actual.items():
        print(f'  {k}: {v} (target {HARD_CASE_TARGETS[k]})')

    print(f'\nTotal selected: {len(selected)}')

    # ---- assemble thread context + write output ----
    # Shuffle final presentation order (same seeded RNG) so strata aren't blocked together
    # during manual labeling - purely a UX/anti-fatigue-bias choice, doesn't affect which
    # 200 examples were chosen.
    ordered_pids = list(selected.keys())
    rng.shuffle(ordered_pids)

    with open(output_path, 'w', encoding='utf-8') as f:
        for pid in ordered_pids:
            r = selected[pid]
            thread_members = by_thread[r['thread_id']]
            context = []
            for m in thread_members:
                if m['pair_id'] == pid:
                    break
                context.append({
                    'pair_id': m['pair_id'],
                    'customer_text_clean': m['customer_text_clean'],
                    'aa_text_clean': m['aa_text_clean'],
                })
            log_entry = next(x for x in selection_log if x['pair_id'] == pid)
            record = {
                'pair_id': r['pair_id'],
                'thread_id': r['thread_id'],
                'customer_tweet_id': r['customer_tweet_id'],
                'aa_tweet_id': r['aa_tweet_id'],
                'customer_text_clean': r['customer_text_clean'],
                'customer_text_raw': r['customer_text_raw'],
                'aa_text_clean_for_reference': r['aa_text_clean'],
                'is_first_turn': r['is_first_turn'],
                'thread_context': context,
                'selection_stratum': log_entry['stratum'],
                'selection_reason': log_entry['reason'],
                'proxy_intent_hint': proxy_bucket.get(pid, 'OTHER_UNCLEAR'),
                # gold fields - intentionally NULL, to be filled by the human annotator
                'gold_intent': None, 'gold_intent_confidence_note': None, 'gold_action': None,
                'gold_escalation': None, 'gold_escalation_reason': None,
                'difficulty': None, 'notes': None, 'labeled': False,
            }
            f.write(json.dumps(record) + '\n')

    return strata_actual, hard_actual, len(selected), len(eligible)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval_pool', default='data/processed/eval_pool.jsonl')
    ap.add_argument('--train', default='data/processed/train.jsonl')
    ap.add_argument('--output', default='data/golden_set/candidates_200.jsonl')
    args = ap.parse_args()
    build(args.eval_pool, args.train, args.output)
