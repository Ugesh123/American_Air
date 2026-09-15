# AmericanAir AI Support Agent — Hiver SDE Take-Home
<img width="2582" height="1604" alt="Screenshot 2026-09-14 174222" src="https://github.com/user-attachments/assets/16a03ff7-6b4e-4e2a-803a-323e13fcdcaf" />


An evaluated AI support agent built on historical AmericanAir customer-service conversations. The system classifies customer intent, retrieves relevant historical resolutions, drafts grounded replies, and decides whether to auto-handle or escalate.
<img width="2570" height="1624" alt="Screenshot 2026-09-14 174214" src="https://github.com/user-attachments/assets/182be818-af78-4976-90c8-58295ef52399" />


See `REPORT.md` for the evaluation and findings, `DECISIONS.md` for engineering decisions, and `GOLDEN_SET.md` for the golden-set methodology.

## What was actually evaluated

The preprocessing, baselines, intent classifier, retrieval, escalation policy, reply generation, LLM judging, human-agreement study, and test suite were executed during development.

The generated outputs and evaluation artifacts are included under `results/`.

The LLM-based reply generation and judging were run through OpenRouter using an external API key. No API keys or secrets are committed to this repository.

## 1. Setup

```bash
pip install -r requirements.txt
