# REPORT - AmericanAir AI Support Agent

## 1. Problem framing

Given real customer↔brand Twitter support conversations, build an agent that
classifies a customer's intent, drafts a reply grounded in how the brand has
historically resolved similar issues, and decides whether to auto-handle or
escalate to a human - with a stated reason. The guiding principle is that the
**proof matters more than the system**: a simple architecture with rigorous,
honest evaluation beats a complex one with vague claims.

## 2. What "good" means for this system

Not "high accuracy" alone. A support agent that's confidently wrong or that
auto-handles something a human should see is worse than one that escalates
more and automates less. "Good" here means: (a) intent classification good
enough to inform a decision, not perfect; (b) an escalation policy whose
**recall on human-required cases** (safety, legal threats, account
security) is the metric that matters most, even at the cost of precision; (c)
replies that never invent policy, refunds, or procedures beyond what real
historical data supports; (d) every claimed number traceable to an actual run
against real human labels, never estimated.

## 3. What was intentionally not built

- No fine-tuned/transformer classifier - TF-IDF + linear models, evaluated
  well; simple and well-proven first.
- No multi-agent framework, LangChain/LangGraph, Docker/K8s, or cloud infra.
- No multi-turn dialogue manager - thread context is read-only for grounding,
  not a stateful conversation manager.
- No live ticketing/CRM integration.
- No full-dataset processing - a documented, reproducible subsample (36,357
  cleaned pairs from ~2.8M raw rows) for one brand.
- No multi-label intent classification (see DECISIONS.md).
- No LLM-decided escalation - a deterministic, explainable policy instead.

## 4. System architecture

```
Customer message
  -> Intent classification (TF-IDF + calibrated LinearSVC, weak-supervised)
  -> Historical retrieval (TF-IDF cosine similarity over 25,451 train pairs)
  -> Escalation decision (deterministic rules over intent/retrieval confidence
     + intent-sensitivity + legal/safety keyword override)
  -> [Grounded reply generation - LLM, requires your own API key, see §11]
  -> Structured output: {intent, intent_confidence, reply, decision, reason, evidence}
```

Retrieval is pluggable: a `TfidfRetriever` (used for every number below) and an
optional `EmbeddingRetriever` (sentence-transformers + FAISS, not yet
benchmarked) share one interface - see DECISIONS.md.

## 5. Dataset and sampling

Real, verified numbers (not assumed) from the actual `twcs.csv` (2,811,774
rows): AmericanAir has 36,764 outbound tweets, ranking #7 by volume among all
brands. After thread reconstruction, cleaning (HTML unescape, URL/mention
normalization, empty-content filtering), and exact-duplicate removal:
**36,357 usable customer→AmericanAir pairs** across **26,298 threads**. A
thread-level, chronological 70/15/15 split produced train (25,451 pairs),
dev (5,453), and eval-pool (5,453), with **0 threads spanning multiple
splits** (programmatically verified) and small, documented cross-split
text-overlap rates (50-54 boilerplate phrases per split, mostly generic
acknowledgments - see `data/step2_stats.json`).

## 6. Golden evaluation methodology

200 examples, stratified across the 11 finalized intents plus 20 deliberately
hard-case slots (weak-retrieval, multi-intent, short/noisy, escalation-edge
language), sampled from the held-out eval pool with two additional exclusions
(cross-split-duplicate-flagged records, heuristically non-English) - see
`GOLDEN_SET.md` for full methodology. All 200 are **real human labels**,
manually reviewed and confirmed via a local labeling tool (`tools/label_tool.py`).

## 7. Baselines

1. **Trivial (majority-class, always-AUTO_HANDLE)** - floor.
2. **Trivial (majority-class, always-ESCALATE)** - safety-reference floor.
3. **Simple ML (TF-IDF unigrams + Logistic Regression, weak-supervised)** - real non-LLM
   baseline.
4. **Final system (TF-IDF uni+bigrams + calibrated LinearSVC + TF-IDF
   retrieval + deterministic escalation policy)**.

## 8. Results (real numbers, computed against the 200-example golden set)

### Intent classification

| System | Accuracy | Macro F1 |
|---|---|---|
| Majority-class (trivial) | 0.120 | 0.020 |
| TF-IDF unigrams + LogReg (simple ML baseline) | 0.535 | 0.500 |
| **Final: TF-IDF uni+bi + calibrated LinearSVC** | **0.665** | **0.650** |

Per-class detail, confusion matrices: `results/intent_classifier_metrics.json`.
Notable weak spot: **SAFETY_SECURITY recall is only 0.23** even in the final
classifier (0.00 in the baseline) - flagged prominently in §10 and §11, not
hidden by the headline accuracy number.

### Escalation policy

| Metric | Value |
|---|---|
| Gold escalation rate | 30.0% (60/200) |
| Predicted escalation rate | 14.5% (29/200) |
| ESCALATE precision / recall / F1 | 0.552 / 0.267 / 0.360 |
| AUTO_HANDLE precision / recall / F1 | 0.743 / 0.907 / 0.817 |
| SAFETY_SECURITY escalation recall | 0.538 (7/13) |
| **Dangerous failure rate** (gold says escalate, system auto-handled) | **22.0% (44/200)** |

This is the single most important number in this report and it is not good.
See §10.

### Retrieval quality

| Metric | Value |
|---|---|
| Queries with zero evidence above similarity threshold | 0/200 |
| Top-1 similarity (median) | 0.335 |
| Proxy topical-match rate | 29.5% (bounded by weak-labeler accuracy, not a clean metric) |

### Headline metric: Verified Auto-Resolution Rate

An example counts only if (1) the intent is correct, (2) the system chose
AUTO_HANDLE, (3) the LLM judge scored the reply at least 3/5 on relevance,
groundedness, actionability and tone, and (4) the reply has no unsupported
claims.

| Stage | Count |
|---|---|
| Intent correct AND AUTO_HANDLE ("Auto-Handle Candidates", the ceiling) | 120/200 = 60.0% |
| **...and the reply passes quality + has no unsupported claims** | **58/200 = 29.0%** |

Reply quality across all 200 judged replies (mean, 1-5): relevance 3.94,
groundedness 3.83, **actionability 2.88**, tone 4.17. 34/200 replies (17%)
contained at least one unsupported claim. Actionability is the weakest
dimension: replies are polite and on-topic but often don't tell the customer
what to do next.

## 9. Failure analysis - top 5 real failure modes

**1. SAFETY_SECURITY is systematically under-recognized by the classifier.**
Example `aa_2344855_2344853`: *"Coming back was better... waited 40 minutes..."*
- gold SAFETY_SECURITY, predicted OTHER_UNCLEAR (confidence 0.85). 7 of 13
gold SAFETY_SECURITY examples were misclassified as OTHER_UNCLEAR. Hypothesis:
these incidents are described in narrative, indirect language ("waited 40
minutes," not "I was hurt") that doesn't match any keyword-derived pseudo-label
pattern used in weak supervision - the classifier never saw enough
SAFETY_SECURITY-like training signal because the weak labeler itself
under-detects this exact phrasing style. Improvement: hand-label a larger
SAFETY_SECURITY-focused seed set rather than relying on keyword pseudo-labels
for this specific high-stakes class.

**2. Confident-but-wrong intent predictions bypass the escalation policy's
confidence-based safety net.** Example `aa_2864248_2864249`: *"But we still
are at the gate"* - gold CHECKIN_BOARDING/escalate (customer possibly stuck
mid-boarding), predicted FLIGHT_STATUS_DELAY_CANCEL at 0.99 confidence - so
confident it cleared the escalation threshold, resulting in a dangerous
AUTO_HANDLE. Hypothesis: high classifier confidence on short, ambiguous
fragments is poorly calibrated - the model is "sure" precisely because the
weak-labeled training data has very few examples this short. Improvement:
down-weight confidence for very short messages, or add message-length as an
explicit escalation-policy signal.

**3. ACCOUNT_WEBSITE_APP is confused with COMPLIMENT and OTHER_UNCLEAR.**
Example `aa_2867972_2867974`: *"it should. But it hasn't been. I do appreciate
all the great flight attendants..."* - gold ACCOUNT_WEBSITE_APP, predicted
COMPLIMENT (0.98 confidence). This is a hard example: it's a
later-turn thread fragment mixing a technical complaint with unrelated praise.
Hypothesis: ACCOUNT_WEBSITE_APP has the smallest weak-supervision training
signal (its keyword pattern is broad but rare in practice) and later-turn
fragments without full thread context are inherently ambiguous even to a
careful reader. Improvement: incorporate `thread_context` into the classifier
input for later-turn messages, not just the focal message.

**4. Repeated-complaint sarcasm confuses topic classification.** Example
`aa_2865587_2865586`: *"Another lost bag with. You're batting 100 whenever I
fly with you!!! Great job guys!"* - gold COMPLAINT_GENERAL (the point is
frustration, not a fresh baggage claim), predicted BAGGAGE. Hypothesis: the
weak-label keyword heuristic (and by extension the classifier trained on it)
treats the literal word "bag" as decisive, missing that the message's actual
intent is a complaint *about* a recurring baggage problem, not a new baggage
report. Improvement: sarcasm/tone cues (exclamation patterns, "another,"
"every time") as classifier features, or as a rule that downgrades BAGGAGE
confidence when COMPLAINT_GENERAL-style intensifiers co-occur.

**5. Escalation thresholds were deliberately not tuned to the golden set,
producing an honest but weak 26.7% escalation recall.** This isn't a single
example but a systemic finding: 44/200 dangerous failures, cutting across
nearly every intent class, share the same root cause - the intent-confidence
and retrieval-confidence thresholds (0.55/0.70/0.20) were set from reasoning
about the architecture, not fit to this data (see DECISIONS.md for why). The
real, uncomfortable result is that the classifier is too often confidently
wrong, and confidence-based escalation doesn't catch it. Improvement: this is
the single highest-value next step - likely a dedicated escalation-risk model
trained on gold_escalation labels directly (once more than 200 are available),
rather than confidence thresholds on a model trained for a different task.

## 10. "What is misleading about my headline number?"

The 29.0% Verified Auto-Resolution Rate is still optimistic if read alone,
for several real, disclosed reasons:

- **The ceiling halves once replies are checked.** 60.0% of cases look
  automatable on intent and decision alone, but only 29.0% survive the reply
  quality check. Quoting the 60% would double-count the opportunity.
- **The quality check itself is an LLM judge that agrees poorly with humans**
  (exact agreement 17-33% per dimension; Cohen's kappa -0.10 on unsupported
  claims across 60 hand-scored replies). The 29% depends on that judge.
- **The escalation policy it depends on has only 26.7% recall and a 22.0%
  dangerous-failure rate.** A headline number built on top of a policy that
  misses 3 out of 4 cases that should escalate is not describing a
  trustworthy system, even if the auto-handled subset looks fine in isolation.
- **The golden set is 200 examples.** Per-class metrics for rare intents
  (SAFETY_SECURITY n=13, ACCOUNT_WEBSITE_APP n=23) have wide uncertainty;
  a couple of different examples would meaningfully move these numbers.
- **The weak-supervision training labels bound everything downstream.** The
  intent classifier was trained on keyword-heuristic pseudo-labels, not real
  ground truth. Its real accuracy (66.5%) is itself the ceiling for anything
  built on top of it (retrieval relevance, escalation confidence).
- **Historical AA replies are evidence of past behavior, not correctness.**
  Nothing here validates that the 2017 AmericanAir agent replies being
  retrieved as "grounding evidence" were actually the *right* thing to say,
  only that they're what was historically said.
- **The held-out eval window (Nov 24–Dec 3, 2017) is a narrow 10-day slice**
  that includes a real, quantified event (a scheduling disruption around
  Christmas 2017) at a 10x higher rate than in train (0.35% vs 0.03% of
  messages) - a small effect in absolute terms, but a reminder that this
  window isn't necessarily "typical" traffic.
- **Distribution shift**: this is all 2017 Twitter data; a 2026 support agent
  built this way would need re-validation against current channels/complaint
  patterns, not just a bigger sample of the same era.

## 11. Limitations

- The LLM judge is not reliable enough to be the final word: the 60-reply
  human study shows weak agreement, so judge-based numbers are a rough signal.
- The human scores in the agreement study come from one reviewer; a second
  reviewer would show how much humans themselves disagree.
- Retrieval uses TF-IDF only; the interface-compatible `EmbeddingRetriever`
  is not benchmarked yet.
- 2 of 200 golden-set records are missing `gold_action` (disclosed, not
  patched - see DECISIONS.md).

## 12. What I would do with one more week

1. Calibrate the judge before trusting it: tighten the rubric with examples,
   re-score the same 60 replies, and only use it once agreement improves.
2. Build a dedicated escalation-risk model trained directly on
   `gold_escalation` labels (once the golden set or a larger hand-labeled
   sample supports it) rather than reusing intent-classifier confidence.
3. Hand-label a larger SAFETY_SECURITY-focused sample to fix the classifier's
   worst failure mode directly, rather than relying on weak supervision for
   this one high-stakes class.
4. Feed `thread_context` into the classifier for later-turn messages instead
   of only the focal message.
5. Swap in the real embedding-based retriever and re-run every metric to see
   how much of the current ceiling was a TF-IDF limitation vs. a genuine data
   limitation.
6. Improve reply actionability (the weakest judged dimension, 2.88/5) by
   prompting for one concrete next step grounded in the retrieved evidence.
