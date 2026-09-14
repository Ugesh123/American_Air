"""
LLM-as-judge for reply quality using OpenRouter.

Required environment variables:
    OPENROUTER_API_KEY
    OPENROUTER_MODEL

Generated replies must already exist:
    results/generated_replies.jsonl

Windows CMD:
    set OPENROUTER_API_KEY=your_key
    set OPENROUTER_MODEL=google/gemini-2.5-flash

Then:
    python3 evaluation\judge\run_judge.py
"""

import json
import os
import urllib.error
import urllib.request


JUDGE_SYSTEM_PROMPT = """You are grading a customer-support reply against a fixed rubric.
You will see the customer's message, the historical evidence the support agent had
access to, and the agent's generated reply. Score the reply ONLY on the following 5
dimensions, using ONLY the information given (do not assume any additional context):

1. Relevance (1-5): does the reply address what the customer actually asked/said?
2. Groundedness (1-5): is everything the reply claims actually supported by the
historical evidence provided (not invented)?
3. Actionability (1-5): does the reply give the customer something concrete/useful?
4. Tone/professionalism (1-5): is the tone appropriate for a professional support reply?
5. Unsupported_claim (YES/NO): does the reply state any specific fact (policy, refund
amount, timeline, procedure) that is NOT directly supported by the evidence?

Respond ONLY with a JSON object with exactly these keys:
relevance, groundedness, actionability, tone, unsupported_claim

No markdown. No explanation."""


JUDGE_USER_TEMPLATE = """Customer message: {customer_text}

Evidence available to the agent:
{evidence_block}

Agent's generated reply:
{reply}

Score it now."""


def format_evidence(evidence: list) -> str:
    if not evidence:
        return "(none retrieved)"

    lines = []

    for i, e in enumerate(evidence, 1):
        customer_msg = e.get("historical_customer_msg", "")
        aa_reply = e.get("historical_aa_reply", "")

        lines.append(
            f'{i}. Customer said: "{customer_msg}" -> '
            f'AA replied: "{aa_reply}"'
        )

    return "\n".join(lines)


def _extract_json_object(text: str) -> dict:
    """Extract a JSON object from normal or fenced model output."""

    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "", 1)
        text = text.replace("```", "", 1)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise RuntimeError(
                f"Judge returned non-JSON output: {text[:2000]}"
            )

        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Judge returned invalid JSON: {text[:2000]}"
            ) from exc


def _validate_score(score: dict) -> dict:
    required = {
        "relevance",
        "groundedness",
        "actionability",
        "tone",
        "unsupported_claim",
    }

    missing = required - set(score.keys())

    if missing:
        raise RuntimeError(
            f"Judge response missing keys: {sorted(missing)}"
        )

    for key in (
        "relevance",
        "groundedness",
        "actionability",
        "tone",
    ):
        try:
            value = int(score[key])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Judge score for {key} is not an integer: {score[key]!r}"
            ) from exc

        if value < 1 or value > 5:
            raise RuntimeError(
                f"Judge score for {key} must be between 1 and 5, got {value}"
            )

        score[key] = value

    unsupported = str(score["unsupported_claim"]).strip().upper()

    if unsupported not in {"YES", "NO"}:
        raise RuntimeError(
            "Judge unsupported_claim must be YES or NO, "
            f"got {score['unsupported_claim']!r}"
        )

    score["unsupported_claim"] = unsupported

    return score


def judge_reply(
    customer_text: str,
    evidence: list,
    reply: str,
) -> dict:
    """Judge one generated reply through OpenRouter."""

    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set."
        )

    model = os.environ.get("OPENROUTER_MODEL")

    if not model:
        raise RuntimeError(
            "OPENROUTER_MODEL is not set."
        )

    user_msg = JUDGE_USER_TEMPLATE.format(
        customer_text=customer_text,
        evidence_block=format_evidence(evidence),
        reply=reply,
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": JUDGE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_msg,
            },
        ],
        "max_tokens": 200,
        "temperature": 0,
    }

    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Hiver AmericanAir Support Agent Judge",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")

    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        raise RuntimeError(
            f"OpenRouter judge API error ({exc.code}): {error_body}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to OpenRouter: {exc.reason}"
        ) from exc

    except TimeoutError as exc:
        raise RuntimeError(
            "OpenRouter judge request timed out."
        ) from exc

    try:
        result = json.loads(raw)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"OpenRouter returned invalid JSON: {raw[:2000]}"
        ) from exc

    try:
        choices = result.get("choices")

        if not choices:
            raise RuntimeError(
                "OpenRouter returned no choices: "
                f"{json.dumps(result)[:2000]}"
            )

        message = choices[0].get("message") or {}
        content = message.get("content")

        if content is None:
            raise RuntimeError(
                "OpenRouter judge returned no final text content. "
                f"Full response: {json.dumps(result)[:2000]}"
            )

        if isinstance(content, list):
            text_parts = []

            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                ):
                    text_parts.append(
                        block.get("text", "")
                    )

            content = "".join(text_parts)

        if not isinstance(content, str):
            content = str(content)

        text = content.strip()

        if not text:
            raise RuntimeError(
                "OpenRouter judge returned an empty response."
            )

        score = _extract_json_object(text)

        return _validate_score(score)

    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected OpenRouter judge response: {raw[:2000]}"
        ) from exc


if __name__ == "__main__":
    ROOT = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )

    generated_path = os.path.join(
        ROOT,
        "results",
        "generated_replies.jsonl",
    )

    if not os.path.exists(generated_path):
        raise SystemExit(
            f"{generated_path} does not exist yet. "
            "Run src/generation/generate_replies.py first."
        )

    outputs = []

    with open(
        generated_path,
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            r = json.loads(line)

            if not r.get("reply"):
                continue

            score = judge_reply(
                r["customer_text_clean"],
                r.get("evidence", []),
                r["reply"],
            )

            outputs.append(
                {
                    "pair_id": r["pair_id"],
                    **score,
                }
            )

            print(
                f"Judged {len(outputs)} replies...",
                end="\r",
            )

    out_path = os.path.join(
        ROOT,
        "results",
        "judge_scores.jsonl",
    )

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:
        for output in outputs:
            f.write(
                json.dumps(
                    output,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print(
        f"Judged {len(outputs)} replies -> {out_path}"
    )