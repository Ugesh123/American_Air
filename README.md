# AI Customer Support Agent: Automate the Routine, Escalate the Risky

![tests](https://github.com/Ugesh123/ai-support-agent-eval/actions/workflows/tests.yml/badge.svg)

- **Problem:** airline support teams get huge message volume, but a confident wrong auto-reply to a safety, legal or account-security message is far worse than a slow one.
- **Decision:** build an agent that auto-handles only what it can answer safely, escalates the rest with a stated reason, and is judged on real outcomes, not accuracy alone.
- **What I learned:** 60% of messages looked automatable, but only **29%** survived a reply-quality check. Reply quality, not intent accuracy, is what limits automation.

**At a glance:** 36,357 real AmericanAir support conversations · 11 intents · 200-example human-labeled golden set · 0.665 intent accuracy (vs 0.535 baseline) · 200 LLM-judged replies · 60-reply human-vs-judge study · 19 tests in CI

---

## Demo: one message in, one decision out

Real decisions from `src/pipeline.py`, shown in short form (the script prints JSON, including the full evidence list):

```text
$ python src/pipeline.py --text "my flight got delayed and nobody told me"
  intent:     FLIGHT_STATUS_DELAY_CANCEL (confidence 0.995)
  decision:   AUTO_HANDLE
  reason:     Intent confidence 1.00 and retrieval confidence 0.45 both clear
              their thresholds; intent is not high-stakes.
  evidence:   5 similar past cases and how AmericanAir replied

$ python src/pipeline.py --text "I will be contacting my lawyer about the way your staff treated me"
  intent:     OTHER_UNCLEAR (confidence 0.946)
  decision:   ESCALATE
  reason:     Message contains legal/regulatory/discrimination language -
              escalates regardless of intent.

$ python src/pipeline.py --text "Your app logged me out and now someone else booked a flight on my account"
  intent:     ACCOUNT_WEBSITE_APP (confidence 0.982)
  decision:   AUTO_HANDLE      <- wrong: a likely account takeover should go to a human
  reason:     Intent confidence 0.98 and retrieval confidence 0.28 both clear
              their thresholds; intent is not high-stakes.
```

The third case is the project's main finding in one example: the model is **confident and wrong about risk**, so confidence thresholds alone can't catch it (see [Limitations](#limitations-and-next-steps)).

---

## What I built

```
Customer message
   │
Intent classification ─ TF-IDF (uni+bigrams) + calibrated LinearSVC → 11 intents
   │
Retrieval ─ similar past cases and how the brand resolved them
   │
Escalation policy ─ explainable rules: intent, confidence, retrieval strength,
   │                 plus a safety/legal keyword override
   │
Reply generation ─ LLM drafts a reply grounded only in the retrieved cases
   │
Output ─ {intent, intent_confidence, reply, decision, reason, evidence}
```

- **Leakage-free data:** conversations split by thread and time, with 0 threads shared between train, dev and eval.
- **Golden set:** 200 hand-labeled examples from held-out data, with rare high-risk intents over-sampled and 20 deliberately hard cases ([GOLDEN_SET.md](GOLDEN_SET.md)).
- **Evaluation of the evaluator:** an LLM judge scored all 200 replies, and 60 of them were scored by hand to check whether the judge can be trusted.
- **Engineering decisions,** each with the alternative I rejected: [DECISIONS.md](DECISIONS.md).

## Key results

| Intent classification | Accuracy | Macro F1 |
|---|---|---|
| Majority-class baseline | 0.120 | 0.020 |
| TF-IDF + Logistic Regression | 0.535 | 0.500 |
| **Final: TF-IDF uni+bi + calibrated LinearSVC** | **0.665** | **0.650** |

| Verified auto-resolution funnel | Count |
|---|---|
| Golden-set messages | 200 |
| Intent correct and auto-handled | 120 (60%) |
| ...and the reply passes quality with no unsupported claims | **58 (29%)** |

Reply quality (LLM judge, mean out of 5): relevance 3.94 · groundedness 3.83 · **actionability 2.88** · tone 4.17. Replies are polite and on-topic but often don't tell the customer what to do next. Full metrics: [REPORT.md](REPORT.md) and [`results/`](results/).

## Product thinking

The full product plan is in [PRD.md](PRD.md): users, north-star metric, guardrails, launch bars, rollout and priorities.

**User research.** The persona in the PRD (Meera, a support team lead) is **illustrative: no real user research was done**. The first interviews I would run:
1. **Support team leads:** which ticket types they would trust automating, and what error rate they would accept.
2. **Frontline agents:** which tickets are the most repetitive, and what they need to see when a ticket is escalated to them.
3. **Compliance or QA:** which message types must never get an automated reply.
4. **Customers who complained publicly:** what a useful first reply looks like to them.
5. **Support operations:** real ticket volumes and cost per ticket, to replace the assumptions below.

**Business case (assumption-based).** Monthly savings ≈ tickets per month × cost per ticket × verified auto-resolution rate. For example, *if* a team handled 50,000 tickets a month at ₹100 each (illustrative numbers, not data), each point of verified auto-resolution would be worth about ₹50,000 a month, and today's 29% about ₹14.5 lakh. That only holds if the guardrails hold: one mishandled safety or legal message can cost more than the savings.

**What I didn't build, and why.**
- **No transformer or fine-tuned model:** a simple, well-evaluated model first, so every gain is provable.
- **No LLM-decided escalation:** escalation must be explainable and auditable.
- **No actions** (rebooking, refunds) **and no multi-turn chat:** out of scope until the decision layer is safe.
- **English only;** non-English messages were excluded and documented.

## Limitations and next steps

- **Escalation misses too much:** escalation recall is **26.7%**, and **22%** of cases (44/200) were auto-handled when a human was needed. All 44 had intent confidence ≥ 0.7, so thresholds can't catch them. **Next:** a dedicated escalation-risk model trained on the escalation labels.
- **The judge isn't trustworthy yet:** it agreed poorly with human scores (exact agreement 17-33% per dimension; Cohen's κ ≈ −0.10 on unsupported claims), so the 29% depends on a weak judge. **Next:** a tighter rubric with examples, re-scoring the same 60 replies, and using judge scores only once agreement is high.
- **Small sample:** 200 examples, with only 13 safety cases, so per-class numbers have wide error margins. **Next:** a larger hand-labeled set, starting with safety messages.
- **Old data:** 2017 Twitter conversations; patterns and channels have changed since.
- **Replies lack next steps:** actionability is the weakest score. **Next:** prompt for one concrete next step grounded in the retrieved cases.

Full failure analysis: [REPORT.md](REPORT.md), sections 9-12.

## Run it

```bash
pip install -r requirements.txt

python src/pipeline.py --text "my flight got delayed and nobody told me"   # one message, offline
python src/pipeline.py --golden_set                                        # all 200 golden examples
python -m unittest discover -s tests -v                                    # 19 tests
```

Reply generation and judging need `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` (see [`.env.example`](.env.example)). No API keys are committed to this repository.

**Code:** `src/` (pipeline, classifier, retrieval, escalation, generation) · `evaluation/` (metrics, LLM judge) · `baselines/` · `tools/` (labeling and judge-check tools) · `tests/`

**Stack:** Python · scikit-learn · NumPy · SciPy · OpenRouter · unittest · GitHub Actions

## Credits

- **Data:** the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset (Thought Vector, via Kaggle); this project uses the AmericanAir conversations.
- The project started from a take-home brief by Hiver, whose core idea shaped it: rigorous, honest evaluation matters more than a complex system.
