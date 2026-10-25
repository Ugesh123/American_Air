# PRD: AI Support Agent for Airline Customer Support

**Status:** Prototype evaluated on 200 human-labeled examples. Not ready for unsupervised rollout.
**Owner:** Yada Ugesh
**Related:** [README.md](README.md) · [REPORT.md](REPORT.md) · [DECISIONS.md](DECISIONS.md)

---

## 1. Problem

AmericanAir's support team answers a very high volume of customer messages on Twitter. Most are routine (delays, baggage, booking changes), but a small share involve safety, legal threats or account security. Handling everything by hand is slow; automating everything is risky, because one confident wrong reply to a safety or legal message costs far more than a slow reply.

**The question this product answers:** which messages can be answered automatically and safely, and which must go to a human, with a reason a support lead can audit?

## 2. Users

| User | What they need |
|---|---|
| **Support team lead** *(primary)* | Routine volume handled automatically, risky messages flagged, and a stated reason for every decision |
| **Support agent** | Less repetitive work, and a draft reply plus similar past cases when a message is escalated to them |
| **Customer** | A fast, correct answer that tells them what to do next |

## 3. Goals and non-goals

**Goals**
- Auto-handle routine messages only when the reply is correct and grounded.
- Never auto-handle safety, legal or account-security messages.
- Explain every decision (`reason` plus the past cases used as evidence).

**Non-goals (for this version)**
- Taking actions such as rebooking or issuing refunds.
- Multi-turn conversations.
- Languages other than English.

## 4. Success metrics

| Type | Metric | Current (200-example golden set) | Proposed launch bar |
|---|---|---|---|
| **North star** | Verified auto-resolution rate: correct intent, auto-handled, and the reply passes quality with no unsupported claims | **29%** (58/200) | Grow without breaking the guardrails |
| **Guardrail** | Dangerous failure rate: should have escalated, was auto-handled | **22%** (44/200) | ≤ 2% |
| **Guardrail** | SAFETY_SECURITY escalation recall | **54%** (7/13) | 100% |
| **Guardrail** | Replies with an unsupported claim | **17%** (34/200) | ≤ 3% |
| **Quality** | Reply actionability (LLM judge, 1-5) | **2.88** | ≥ 4.0 |
| **Trust in the judge** | Human-vs-judge agreement on unsupported claims (Cohen's κ) | **−0.10** | ≥ 0.6 before judge scores gate anything |

*Current values are measured results. Launch bars are my proposed targets, not results.*

## 5. What the evaluation showed

1. **The opportunity is half of what it first looks like.** 60% of messages look automatable on intent and decision alone, but only 29% survive a reply-quality check.
2. **Confidence thresholds can't catch the risky cases.** All 44 dangerous failures had intent confidence of 0.7 or higher. In 27 of them the intent was even correct: the message needed a human for reasons the intent label doesn't capture. This is why a separate escalation-risk model is the top priority.
3. **Replies are polite but not useful enough.** Tone scores 4.17/5, but actionability scores 2.88/5: replies often don't tell the customer what to do next.
4. **The automated judge can't be trusted yet.** It agrees poorly with human scores, so judge-based numbers are a rough signal only.

## 6. Proposed rollout

| Phase | What the agent does | Exit criteria to move on |
|---|---|---|
| **1. Shadow mode** | Runs on live traffic; humans reply as usual; decisions are logged and compared | Dangerous failure rate ≤ 2% on a fresh labeled sample |
| **2. Agent assist** | Drafts replies and shows evidence; a human approves every reply | Agents accept most drafts with light edits; unsupported claims ≤ 3% |
| **3. Limited auto-handle** | Auto-replies only for the lowest-risk intents (for example COMPLIMENT, simple delay updates); everything else stays with humans | Guardrails hold for several weeks; support leads sign off |

Safety, legal and account-security messages stay with humans in every phase.

## 7. Prioritized next steps

| # | Work | Why first | Effort |
|---|---|---|---|
| 1 | Dedicated escalation-risk model trained on `gold_escalation` labels | Fixes the biggest risk (22% dangerous failures) | Medium |
| 2 | Hand-label more SAFETY_SECURITY examples | Highest-stakes class with the weakest recall | Low |
| 3 | Prompt replies to include one concrete next step from the evidence | Actionability is the weakest quality score | Low |
| 4 | Calibrate the judge with a tighter rubric and examples, then re-check agreement | The 29% depends on a judge that is not trustworthy yet | Medium |
| 5 | Benchmark embedding retrieval against TF-IDF | May improve grounding; lower risk impact than 1-4 | Medium |

## 8. Risks and open questions

- **Old data:** this is 2017 Twitter data; complaint patterns and channels have changed since.
- **Small test set:** 200 examples, with only 13 SAFETY_SECURITY cases, so per-class numbers have wide uncertainty.
- **Past replies are not policy:** retrieved historical replies show what agents said, not necessarily what was correct.
- **Open question:** what dangerous-failure rate would a support lead actually accept? The ≤ 2% bar above needs validating with real support leads.
