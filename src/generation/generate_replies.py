"""
Grounded reply generation using OpenRouter.

Required environment variables:
    OPENROUTER_API_KEY
    OPENROUTER_MODEL

Windows CMD:
    set OPENROUTER_API_KEY=your_key
    set OPENROUTER_MODEL=openrouter/free

Then:
    python3 src\generation\generate_replies.py
"""

import json
import os
import urllib.error
import urllib.request


SYSTEM_PROMPT = """You are drafting a customer-support reply for AmericanAir on
behalf of a human-supervised automation system. You will be given a customer
message, its classified intent, and up to 5 historical AmericanAir replies to
similar past customer messages (your only source of "how we've handled this
before").

STRICT RULES:
1. Do not invent policies, refund amounts, procedures, compensation offers, or
URLs that are not present in the retrieved historical evidence.
2. If the evidence does not clearly support a specific answer, write a reply
that acknowledges the issue and says a team member will follow up with
specifics -- do NOT guess at specifics.
3. Be concise, professional, and actionable, in the voice the historical
replies demonstrate.
4. Do not promise anything (refund amount, timeline, compensation) that isn't
directly supported by the evidence.

Respond with the reply text only, nothing else."""


USER_TEMPLATE = """Customer message: {customer_text}
Classified intent: {intent}

Historical evidence (past AmericanAir replies to similar customer messages):
{evidence_block}

Draft the reply now."""


def format_evidence(evidence: list) -> str:
    if not evidence:
        return (
            "(No sufficiently similar historical case was found. "
            "If you cannot answer safely without more context, "
            "say so rather than guessing.)"
        )

    lines = []

    for i, e in enumerate(evidence, 1):
        lines.append(
            f'{i}. [similarity={e.similarity:.2f}] '
            f'Customer said: "{e.customer_text}"\n'
            f'   AmericanAir replied: "{e.aa_reply_text}"'
        )

    return "\n".join(lines)


def generate_grounded_reply(
    customer_text: str,
    intent: str,
    evidence: list,
) -> str:
    """Generate a grounded reply through OpenRouter."""

    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. "
            "Set it in the terminal before running generation."
        )

    model = os.environ.get("OPENROUTER_MODEL")

    if not model:
        raise RuntimeError(
            "OPENROUTER_MODEL is not set. "
            "Set it to a valid OpenRouter model ID."
        )

    user_msg = USER_TEMPLATE.format(
        customer_text=customer_text,
        intent=intent,
        evidence_block=format_evidence(evidence),
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_msg,
            },
        ],
        "max_tokens": 300,
        "temperature": 0.2,
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
            "X-Title": "Hiver AmericanAir Support Agent",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")

    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenRouter API error ({exc.code}): {error_body}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not connect to OpenRouter: {exc.reason}"
        ) from exc

    except TimeoutError as exc:
        raise RuntimeError(
            "OpenRouter request timed out."
        ) from exc

    try:
        result = json.loads(raw)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"OpenRouter returned invalid JSON: {raw[:1000]}"
        ) from exc

    try:
        choices = result.get("choices")

        if not choices:
            raise RuntimeError(
                f"OpenRouter returned no choices: {json.dumps(result)[:2000]}"
            )

        message = choices[0].get("message") or {}
        content = message.get("content")

        # Some OpenRouter models/providers may return null content.
        if content is None:
            # Fallback if content is represented as structured blocks.
            reasoning = message.get("reasoning")
            if reasoning and isinstance(reasoning, str):
                raise RuntimeError(
                    "OpenRouter returned reasoning but no final text content: "
                    f"{reasoning[:1000]}"
                )

            raise RuntimeError(
                "OpenRouter returned no text content. "
                f"Full response: {json.dumps(result)[:2000]}"
            )

        if isinstance(content, list):
            text_parts = []

            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

            content = "".join(text_parts)

        if not isinstance(content, str):
            content = str(content)

        reply = content.strip()

    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected OpenRouter response: {raw[:2000]}"
        ) from exc

    if not reply:
        raise RuntimeError(
            f"OpenRouter returned an empty reply: {json.dumps(result)[:2000]}"
        )

    return reply


if __name__ == "__main__":
    import sys

    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        ),
    )

    from pipeline import Agent, ROOT, run_on_golden_set

    agent = Agent()

    outputs = run_on_golden_set(
        agent,
        generate_reply=True,
    )

    out_path = os.path.join(
        ROOT,
        "results",
        "generated_replies.jsonl",
    )

    with open(out_path, "w", encoding="utf-8") as f:
        for output in outputs:
            f.write(
                json.dumps(
                    output,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"Generated replies for {len(outputs)} golden-set examples "
        f"-> {out_path}"
    )