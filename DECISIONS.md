# DECISIONS.md

Non-obvious engineering decisions made in this project, with the alternative
considered and why it was rejected. See README.md/REPORT.md for how these fit
together end to end.

---

**Decision: Brand = AmericanAir**
**Reason:** Verified (not assumed) against the real dataset before committing:
36,764 outbound tweets, 36,531 resolvable customer→brand pairs, 99.4%
thread-linkage integrity, low duplicate rate (0.6% exact), and real presence
across every draft taxonomy category including SAFETY_SECURITY -- the kind of
high-stakes escalation case the assignment wants demonstrated.
**Alternative considered:** AmazonHelp/AppleSupport (higher volume but
unbounded intent space); SpotifyCares (cleaner but lower-stakes escalation).
**Why rejected:** Amazon/Apple's breadth would dilute taxonomy design effort;
SpotifyCares doesn't give escalation policy real teeth to prove.

---

**Decision: Intent taxonomy has 11 classes, revised (not invented) from real
data via a TF-IDF/KMeans discovery pass on the 48.6% of messages a draft
keyword taxonomy missed.**
**Reason:** The discovery pass confirmed the original 10 topical intents were
sound but under-specified in phrasing; no new classes were needed. One
definition was materially tightened: SAFETY_SECURITY was narrowed after real
false positives were found ("safety video" mentions, not incidents).
**Alternative considered:** Adding classes for niche sub-clusters found
(Admirals Club/lounge feedback, crew-specific praise, seat/cabin comfort).
**Why rejected:** Each was low-volume (n≈20-30) and folded cleanly into
existing COMPLIMENT/COMPLAINT_GENERAL categories without losing signal; a
200-example golden set can't support 15+ classes with meaningful per-class
metrics.

---

**Decision: Thread-level, time-based train/dev/eval split (not random
tweet-level split).**
**Reason:** A tweet-level random split would leak near-duplicate boilerplate
and multi-turn thread continuations across splits. Splitting whole threads by
chronological order simulates real deployment (train on past, evaluate on
future) and is independently leak-checked (0 threads span multiple splits).
**Alternative considered:** Stratified random split by intent.
**Why rejected:** Not possible before Step 3 (no intent labels existed yet at
Step 2), and would still risk near-duplicate leakage across splits regardless.

---

**Decision: Golden-set strata are NOT proportional to raw class prevalence;
rare high-stakes classes (SAFETY_SECURITY, ACCOUNT_WEBSITE_APP, LOYALTY_MILES)
are deliberately over-sampled.**
**Reason:** A class needs roughly a dozen or more examples in a 200-example set
to produce a computable, non-degenerate per-class precision/recall. Rare
classes also matter disproportionately for escalation recall.
**Alternative considered:** Pure proportional stratification.
**Why rejected:** Would have yielded ~2 SAFETY_SECURITY examples (0.9% of
25,451), too few to say anything meaningful about the system's most important
failure mode.

---

**Decision: SAFETY_SECURITY candidate search used a tightened, hand-verified
keyword pattern, not the original broad "safety" match.**
**Reason:** The broad pattern surfaced real false positives ("safety video,"
"#SafetyFirst" hashtags). Re-running with a pattern requiring actual incident
language (assault, harassment, discrimination, medical emergency, allergy,
threat) found 28 genuine candidates -- comfortably above the target of 12, so
no shortfall-driven compromise was needed.
**Alternative considered:** Backfilling the target with lower-confidence
matches.
**Why rejected:** Explicitly prohibited by the project's own rules (don't
fabricate/pad the golden set with false positives) and unnecessary here since
the genuine pool was large enough.

---

**Decision: Multi-intent messages get ONE primary `gold_intent` + an
independent `gold_escalation` flag, not multi-label classification.**
**Reason:** The architecture already treats escalation as a separate
deterministic signal decoupled from intent (see escalation policy below), so
multi-intent urgency doesn't require multi-label machinery. Real evidence: of
5,184 eligible eval-pool examples, 849 (16.4%) matched 2+ keyword buckets, but
manual inspection showed one dominant actionable ask in nearly every case.
**Alternative considered:** Full multi-label intent classification.
**Why rejected:** Roughly doubles annotation effort and requires a different
eval methodology (per-label P/R vs. a clean confusion matrix), disproportionate
to what the downstream system actually needs (one reply, one auto/escalate
decision).

---

**Decision: The intent classifier is trained via weak supervision (keyword
pseudo-labels on the 25,451 unlabeled train pairs), evaluated exclusively on
the 200 real human gold labels.**
**Reason:** No intent-labeled corpus exists apart from the 200-example golden
set -- Step 2's output is text pairs only, with no classification. Training
solely within the 200 golden examples (e.g. k-fold CV) would average ~18
examples per fold per class, too noisy for reliable metrics, and would burn the
only real labels on training rather than evaluation.
**Alternative considered:** k-fold cross-validation purely within the golden
set.
**Why rejected:** Too small and would contaminate the one clean evaluation
resource. Weak supervision is a standard, disclosed technique -- pseudo-labels
are never used as ground truth anywhere in evaluation.

---

**Decision: Retrieval uses TF-IDF + cosine similarity, not sentence-transformers
+ FAISS, despite the assignment's suggestion of the latter.**
**Reason:** This development sandbox has no internet access, so
sentence-transformers/faiss cannot be installed or a model downloaded here.
**Alternative considered:** Writing embedding-based retrieval code untested.
**Why rejected instead of doing that silently:** The embedding-based
`EmbeddingRetriever` class IS written (src/retrieval/retrieve.py), with the
identical interface as the working TF-IDF backend, so switching is a one-line
flag change on a machine with internet -- but every number in results/ was
actually produced by the TF-IDF backend, and that's stated plainly rather than
implying the embedding path was used and verified.

---

**Decision: Escalation policy thresholds (intent-confidence 0.55/0.70,
retrieval-confidence 0.20) were set a priori from reasoning about the
architecture, and NOT tuned against the golden set's actual escalation
outcomes.**
**Reason:** The golden set is the only labeled evaluation data that exists.
Tuning thresholds to maximize the golden set's own escalation F1 would be
fitting to the test set, producing an inflated, non-generalizable number.
**Alternative considered:** Grid-searching thresholds to maximize F1 on the
golden set.
**Why rejected:** Would misrepresent the system's real-world performance. The
resulting real numbers (25% escalation recall, 22.5% dangerous-failure rate)
are worse-looking but honest, and are called out explicitly as the most
important limitation in REPORT.md rather than hidden by threshold-tuning.

---

**Decision: The majority-class trivial baseline computes "majority" from the
golden set's own label distribution, not an independent training set.**
**Reason:** As above, no independently-labeled larger corpus exists. A
majority-class baseline inherently needs *some* notion of "most common class."
**Alternative considered:** Skipping the majority-class baseline entirely.
**Why rejected:** The brief explicitly requires a trivial baseline; this one is
disclosed as a same-set floor (not a real predictive model) rather than
silently treated as equivalent to an independently-fit baseline.

---

**Decision: Final intent classifier differs from the simple-ML baseline only
modestly (TF-IDF unigrams+bigrams vs. unigrams; calibrated LinearSVC vs. plain
Logistic Regression), not a fundamentally different architecture.**
**Reason:** The brief explicitly prefers simple systems with strong evaluation
over complex architectures. Both are trained identically (same weak-supervision
labels) so the comparison isolates the effect of features/algorithm, not a
confound from different training data.
**Alternative considered:** A transformer-based classifier.
**Why rejected:** Requires the same internet access this sandbox doesn't have,
and would be disproportionate complexity for the evaluation-quality-focused
goal of this assignment.

---

**Decision: Grounded reply generation and the LLM judge are fully coded but
have NEVER been executed in this development environment.**
**Reason:** Both require an LLM API key and internet access, neither available
in this sandbox. Rather than fabricate plausible-looking generated replies or
judge scores, these pipeline stages are left genuinely unexecuted, and the
repo's results/ directory contains no output claiming otherwise.
**Alternative considered:** Simulating expected outputs to complete the
report's sections.
**Why rejected:** Directly prohibited by the project's own "never fabricate
results" rule. README.md gives exact commands to run these stages for real,
on a machine with API access.

---

**Decision: Human-vs-judge agreement study is scaffolded (sampling +
tooling + metrics script) but has no results yet.**
**Reason:** Depends on generation + judging having actually run first, which
requires your own API key.
**Alternative considered:** none defensible -- fabricating agreement numbers
here would be the single most direct violation of the assignment's explicit,
repeated instruction not to invent this study's results.

---

**Decision: 2 of 200 golden-set records are missing `gold_action` and were
left as-is rather than silently patched.**
**Reason:** The golden set is explicitly frozen/human-owned; discovered via
schema validation, reported transparently, and does not block intent/escalation
metrics (which use `gold_intent`/`gold_escalation`, both valid for all 200).
**Alternative considered:** Filling in a plausible `gold_action` automatically.
**Why rejected:** Would be writing a label the human annotator didn't actually
provide -- the exact failure mode this whole project is designed to avoid.
