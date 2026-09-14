# AmericanAir AI Support Agent — Hiver SDE Take-Home

An evaluated AI support agent built on historical AmericanAir customer-service conversations. The system classifies customer intent, retrieves relevant historical resolutions, drafts grounded replies, and decides whether to auto-handle or escalate.

See `REPORT.md` for the evaluation and findings, `DECISIONS.md` for engineering decisions, and `GOLDEN_SET.md` for the golden-set methodology.

## What was actually evaluated

The preprocessing, baselines, intent classifier, retrieval, escalation policy, reply generation, LLM judging, human-agreement study, and test suite were executed during development.

The generated outputs and evaluation artifacts are included under `results/`.

The LLM-based reply generation and judging were run through OpenRouter using an external API key. No API keys or secrets are committed to this repository.

## 1. Setup

```bash
pip install -r requirements.txt