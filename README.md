# AI Customer Support Agent: Automate the Routine, Escalate the Risky

![tests](https://github.com/Ugesh123/American_Air/actions/workflows/tests.yml/badge.svg)

**An AI support agent that knows when to answer and when to hand over to a human.**
Built on **36,357 real AmericanAir customer-support conversations**, it detects what the customer needs, finds how similar issues were resolved before, drafts a grounded reply, and decides whether to **auto-handle or escalate**, always with a stated reason.

**At a glance:** 11 intents · 200-case human-labeled golden set · 0.665 intent accuracy (vs 0.535 baseline) · **29% verified auto-resolution** (60% before checking reply quality) · 200 LLM replies judged · 60-reply human-vs-judge study · 19 tests in CI

> Started from a take-home brief by Hiver whose core idea shaped the whole project: *rigorous, honest evaluation matters more than a complex system.*

---

## Understanding the Problem

### 1. Context

| | |
|---|---|
| **About the company** | **AmericanAir** handles a very high volume of customer-support requests on Twitter (36,764 outbound replies in the dataset, ranking #7 among brands by volume). |
| **About the industry** | Airline customer support: delays, cancellations, baggage, refunds and loyalty questions, where some messages involve **safety, legal or account-security** issues. |
| **Stakeholders** | **Customers** (fast, correct answers), **support agents** (less repetitive work), **support leads** (quality and risk) and the **brand** (trust). |
| **Research** | Cleaned 36,531 raw conversation pairs into 36,357, split them by thread and time into **leakage-free** train, dev and eval sets, and ran a clustering-based discovery pass to finalise **11 intents** from the real data rather than assumptions. |

### 2. Diagnosis

| | |
|---|---|
| **Pain points** | Support teams face huge volume, but a wrong automated reply to a safety, legal or account-security message is far worse than a slow one. |
| **Key takeaways** | *"High accuracy"* alone is the wrong goal. A support agent that is **confidently wrong** is worse than one that escalates more and automates less. |
| **Benchmark view** | Measured against simple baselines: a majority-class guess scores **0.120** accuracy and a TF-IDF + logistic regression model **0.535**. The final classifier reaches **0.665** (macro F1 0.650). |
| **Trade-off** | Automate more vs. stay safe. The policy deliberately favours **catching risky cases** (escalation recall) over automation rate, and escalation is decided by **explainable rules**, not by an LLM. |

### 3. Strategy

| | |
|---|---|
| **User persona** | **Meera, a support team lead** *(illustrative persona)*. She wants routine questions handled automatically and anything risky flagged to a human, with a reason she can audit. |
| **Key decisions** | Simple, well-evaluated models over complex ones · replies grounded only in how the brand actually resolved similar issues (no invented policies or refunds) · every number traceable to a real run on human labels. |
| **Solution** | Intent classification → retrieval of similar resolved cases → grounded reply drafting → **auto-handle or escalate**, returning `{intent, confidence, reply, decision, reason, evidence}`. |
| **Implementation** | Python + scikit-learn (TF-IDF + calibrated LinearSVC), TF-IDF retrieval, rule-based escalation policy, LLM reply generation and judging via OpenRouter, and a labeling tool for the golden set. |

### 4. Impact

| | |
|---|---|
| **Business impact** | **29%** of messages (58 of 200) are **verified auto-resolutions**: correct intent, auto-handled, and a reply that passes quality checks with no unsupported claims. Before checking replies, 60% looked automatable, so reply quality halves the real opportunity. |
| **Quality check** | **200 generated replies** were scored by an LLM judge (mean out of 5: relevance 3.94, groundedness 3.83, **actionability 2.88**, tone 4.17); 17% had an unsupported claim. A **60-reply human-vs-judge study** checked whether the judge itself can be trusted. |
| **Risk & mitigation** | Escalation recall is only **26.7%** today, and **22%** of cases are "dangerous failures" (should have escalated but were auto-handled). The judge also agrees poorly with humans (Cohen's κ ≈ −0.10 on unsupported claims). **So this is not ready for unsupervised rollout**: next come a dedicated escalation-risk model, better "next step" guidance in replies, and a calibrated judge. |
| **Annexures** | [`REPORT.md`](REPORT.md) evaluation · [`DECISIONS.md`](DECISIONS.md) engineering decisions · [`GOLDEN_SET.md`](GOLDEN_SET.md) golden-set method · [`results/`](results/) all outputs |

---

## How It Works

```
Customer message
   │
Intent classification ─ TF-IDF (uni+bigrams) + calibrated LinearSVC → 11 intents
   │
Retrieval ─ finds similar historical cases and how the brand resolved them
   │
Escalation policy ─ deterministic rules on intent + confidence + retrieval strength
   │                 + safety / legal keyword override
   │
Reply generation (optional) ─ LLM drafts a reply grounded only in retrieved evidence
   │
Output ─ {intent, intent_confidence, reply, decision, reason, evidence}
```

## Results (200-example human-labeled golden set)

| System | Accuracy | Macro F1 |
|---|---|---|
| Majority-class baseline | 0.120 | 0.020 |
| TF-IDF + Logistic Regression | 0.535 | 0.500 |
| **Final: TF-IDF uni+bi + calibrated LinearSVC** | **0.665** | **0.650** |

| Escalation metric | Value |
|---|---|
| ESCALATE precision / recall | 0.552 / 0.267 |
| AUTO_HANDLE precision / recall | 0.743 / 0.907 |
| SAFETY_SECURITY escalation recall | 0.538 (7/13) |
| Dangerous failures (should escalate, auto-handled) | 22.0% (44/200) |

| Verified auto-resolution funnel | Count |
|---|---|
| Golden-set messages | 200 |
| Intent correct and auto-handled (ceiling) | 120 (60%) |
| ...and reply passes quality with no unsupported claims | **58 (29%)** |

Full per-class metrics, confusion matrices and judge outputs are in [`results/`](results/).

## Evaluation Method

- **Leakage-free splits:** thread-level, time-based split with **0 threads** shared across train, dev and eval.
- **Golden set:** 200 human-labeled examples from held-out data, excluding text duplicated in training and non-English records, with rare high-risk classes over-sampled and **20 deliberately hard cases**.
- **Honest reporting:** weak spots (e.g. SAFETY_SECURITY recall) are shown, not hidden behind the headline number.

## Tech Stack

Python · scikit-learn · NumPy · joblib · TF-IDF retrieval · OpenRouter (LLM reply generation and judging) · pytest

## Project Structure

```
src/preprocessing/   cleaning, dataset build, golden-set sampling
src/intent/          taxonomy and classifier training
src/retrieval/       historical case retrieval
src/escalation/      rule-based escalation policy
src/generation/      grounded reply generation (LLM)
src/pipeline.py      end-to-end pipeline
evaluation/          metrics and LLM judge
tools/               labeling and judge-agreement tools
results/             all evaluation outputs
```

## How to Run

```bash
pip install -r requirements.txt

# Classify, retrieve and decide for a single message (runs offline)
python3 src/pipeline.py --text "my flight got delayed and nobody told me"

# Run over all 200 golden-set examples
python3 src/pipeline.py --golden_set

# Optional: grounded reply generation (needs OPENROUTER_API_KEY and OPENROUTER_MODEL)
python3 src/pipeline.py --text "my bag never arrived" --generate_reply
```

## Tests

```bash
python -m unittest discover -s tests -v
```

**19 tests** cover text cleaning, the escalation policy, data integrity (no thread leaks across splits, 200 labeled golden examples) and an end-to-end smoke test. They run on every push through GitHub Actions.

No API keys or secrets are committed to this repository.
