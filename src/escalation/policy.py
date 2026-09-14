"""
Deterministic AUTO_HANDLE vs ESCALATE policy. Per the assignment brief, this is
explicitly NOT delegated to an LLM -- every decision must be explainable from a
small set of measurable signals, each with a human-readable reason string.

Signals used:
  - intent: SAFETY_SECURITY always escalates, no exceptions.
  - intent_confidence: below INTENT_CONF_THRESHOLD -> escalate (the classifier
    itself isn't sure what this is about).
  - retrieval_confidence (top-1 similarity): below RETRIEVAL_CONF_THRESHOLD ->
    escalate (we don't have good historical grounding to draft a safe reply).
  - sensitive_intents: REFUND_COMPENSATION and ACCOUNT_WEBSITE_APP are treated
    as inherently higher-stakes (money / account access) and escalate at a
    higher confidence threshold than other intents.
  - escalation_edge language: legal/regulatory/discrimination keywords force
    escalation regardless of intent or confidence (a customer threatening legal
    action should never get an auto-reply, whatever topic it's nominally filed
    under).

Recall is deliberately prioritized over precision for SAFETY_SECURITY and the
escalation-edge keyword override, per the brief ("prioritize high recall for
genuinely human-required cases") -- false escalations cost an agent's time;
false auto-handles of a safety issue are the actually dangerous failure mode.
"""
import re

INTENT_CONF_THRESHOLD = 0.55
RETRIEVAL_CONF_THRESHOLD = 0.20
SENSITIVE_INTENTS = {'REFUND_COMPENSATION', 'ACCOUNT_WEBSITE_APP'}
SENSITIVE_INTENT_CONF_THRESHOLD = 0.70  # stricter bar for sensitive intents

ESCALATION_EDGE_RE = re.compile(
    r'\b(lawyer|sue|legal action|attorney|discriminat\w*|racis\w*|never (fly|flying) (you|with you) again|'
    r'report(ed)? (this|you) to|BBB|department of transportation|\bdot\b)\b', re.IGNORECASE)


def decide(intent: str, intent_confidence: float, retrieval_confidence: float, customer_text: str) -> dict:
    """Returns {'decision': 'AUTO_HANDLE'|'ESCALATE', 'reason': str}."""

    if intent == 'SAFETY_SECURITY':
        return {'decision': 'ESCALATE', 'reason': 'Intent is SAFETY_SECURITY -- always escalates by policy.'}

    if ESCALATION_EDGE_RE.search(customer_text):
        return {'decision': 'ESCALATE',
                'reason': 'Message contains legal/regulatory/discrimination language -- escalates regardless of intent.'}

    conf_threshold = SENSITIVE_INTENT_CONF_THRESHOLD if intent in SENSITIVE_INTENTS else INTENT_CONF_THRESHOLD
    if intent_confidence < conf_threshold:
        return {'decision': 'ESCALATE',
                'reason': f'Intent confidence {intent_confidence:.2f} is below the '
                          f'{"sensitive-intent " if intent in SENSITIVE_INTENTS else ""}threshold {conf_threshold:.2f}.'}

    if retrieval_confidence < RETRIEVAL_CONF_THRESHOLD:
        return {'decision': 'ESCALATE',
                'reason': f'Retrieval confidence {retrieval_confidence:.2f} is below {RETRIEVAL_CONF_THRESHOLD:.2f} '
                          f'-- insufficient historical grounding to draft a safe reply.'}

    return {'decision': 'AUTO_HANDLE',
            'reason': f'Intent confidence {intent_confidence:.2f} and retrieval confidence '
                      f'{retrieval_confidence:.2f} both clear their thresholds; intent is not high-stakes.'}
