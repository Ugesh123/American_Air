# Golden Evaluation Set - Methodology

## What this is

`data/golden_set/candidates_200.jsonl` contains 200 **candidate** examples sampled
from the held-out `eval_pool.jsonl` (5,453 customer→AmericanAir pairs, Nov 24 – Dec 3
2017). All 200 were labeled by hand with `tools/label_tool.py`; the labels live in
`data/golden_set/labels.json`. This file explains how the 200 were chosen.

## Isolation from training / retrieval

The 200 candidates are drawn exclusively from `eval_pool.jsonl`, which was assigned
its own set of **threads** (not just tweets) during the time-based split in
`src/preprocessing/build_dataset.py`, verified with 0 threads spanning multiple splits. None of these threads, or any
tweet within them, were used to build the training set or the retrieval corpus. Two further conservative exclusions were applied on top of that
thread-level isolation before candidates were even considered:

1. **Cross-split duplicate-flagged records excluded**: any eval-pool record whose
   normalized customer or AA-reply text exactly matches something seen in `train`
   (`dup_customer_text_in_train` / `dup_aa_text_in_train`, computed during the dataset build) was
   dropped from the candidate pool - 5,453 → 5,236 (96.0% retained). This is a belt-
   and-suspenders safeguard on top of thread isolation, in case a boilerplate phrase
   recurs across otherwise-unrelated threads.
2. **Heuristically non-English excluded**: a Spanish-stopword heuristic flagged
   ~1% of records (5,236 → 5,184 final eligible pool). The system as scoped is
   English-only; non-English support is out of scope and documented as a limitation
   rather than silently mishandled.

## Why these strata

The taxonomy and per-class targets came from an actual discovery pass over the real
data (TF-IDF/KMeans clustering of the 48.6% of eval-pool messages that a draft
keyword taxonomy failed to match, plus manual reading of samples) - not an assumed
distribution. In short, the discovery pass confirmed the original 10 topical intents were sound but under-
specified in phrasing, added no new classes, tightened `SAFETY_SECURITY`'s
definition after finding real false positives ("safety video" ≠ a safety incident),
and formalized `OTHER_UNCLEAR` as covering noise / non-English / context-dependent
fragments.

Per-class targets are **not proportional to raw prevalence** - common classes
(FLIGHT_STATUS_DELAY_CANCEL, COMPLIMENT) got a healthy allocation but rare,
high-stakes classes (SAFETY_SECURITY, ACCOUNT_WEBSITE_APP, LOYALTY_MILES) were
deliberately over-sampled relative to their true frequency, because a golden set
sized 12-30 examples-per-class needs enough of each class to compute a meaningful
per-class precision/recall, and the rare classes matter disproportionately for the
escalation-recall metric.

20 additional slots were reserved for deliberately hard cases, chosen from signals
already computed on real data rather than invented: weak-retrieval risk (TF-IDF
max-cosine-similarity to the train corpus - a proxy, since the
retrieval index did not exist yet when candidates were sampled), multi-intent (2+ keyword-bucket
matches), short/noisy (≤5 words), and escalation-edge language (legal/regulatory
threats, discrimination) outside the SAFETY_SECURITY stratum itself.

## SAFETY_SECURITY - transparency on the target

The plan called for 12 SAFETY_SECURITY examples. The keyword-based *candidate
sizing* proxy initially suggested more were available, but a first pass surfaced
real false positives (casual mentions of "safety video," "#SafetyFirst" hashtags).
The pattern was tightened to require language describing an actual incident
(assault, harassment, discrimination, medical emergency, allergy, threat), and
re-run against the eligible pool: **28 genuine candidates were found, comfortably
above the target of 12** - no shortfall, no keyword-false-positive backfilling was
needed. Had fewer than 12 genuine examples existed, this document would report the
true number and the target would have been reduced accordingly, per your
instruction - that adjustment simply wasn't necessary here.

## Reproducibility

`src/preprocessing/sample_golden_set.py` is deterministic: a single fixed RNG seed
(42) controls every random draw (per-stratum sampling and the final presentation-
order shuffle). Re-running the script against the same `eval_pool.jsonl` /
`train.jsonl` reproduces the identical 200 `pair_id`s in the identical order.

```bash
python3 src/preprocessing/sample_golden_set.py \
    --eval_pool data/processed/eval_pool.jsonl \
    --train data/processed/train.jsonl \
    --output data/golden_set/candidates_200.jsonl
```

## Annotation procedure

1. Run `python3 tools/label_tool.py`, open `http://localhost:8420`.
2. For each example: read `customer_text_clean` (raw original text and thread
   context, if any, are available via toggles/panels). Consult the taxonomy
   reference panel on the right for inclusion/exclusion criteria.
3. Assign exactly one `gold_intent` (single-label, per the primary-intent +
   escalation-flag decision - see below), a `gold_action`, a `gold_escalation`
   decision with a required reason if escalating, a `difficulty` rating, and free-
   text `notes` for anything relevant (e.g. secondary intent, sarcasm, sensitive
   content).
4. Progress saves to `data/golden_set/labels.json` after every example - the tool
   can be closed and reopened without losing work.

**Multi-intent handling reminder**: label the single **dominant actionable topic**
as `gold_intent`; use `notes` to record a secondary topic if the message genuinely
raises two. Escalation is judged independently of intent - a FLIGHT_STATUS message
can still be `gold_escalation=True` if the underlying situation is urgent.

**A content note**: this is real historical data. A small number of the
SAFETY_SECURITY-stratum examples describe genuine past incidents, including one
disclosure of a historical assault. They are included because a real support
system needs to correctly escalate this class of message, not because of any
editorial choice - they were labeled with the same care anyone reaching out would deserve.

## What happens after labeling

`labels.json` (200 entries) is merged onto `candidates_200.jsonl` to form the
frozen golden set used for every metric in this project: intent classification,
the escalation policy and the LLM-judge human-agreement study. No modeling,
retrieval or generation work started before the labels were finished, so none of
it could influence them.
