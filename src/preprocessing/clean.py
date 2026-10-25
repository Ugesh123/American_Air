"""
Text cleaning utilities for the AmericanAir customer-support pairs.

Design decisions (see DECISIONS.md for the full rationale once written):
- We always keep the RAW original text untouched alongside a CLEANED version,
  so every example is traceable back to the exact original tweet.
- HTML entities (&amp;, &gt;, etc.) are unescaped - Twitter's export leaves these raw.
- URLs are replaced with a [URL] placeholder in the cleaned text (not deleted),
  since "customer included a link" is sometimes a real signal (e.g. attached a
  screenshot of an error), but the literal t.co URL has no reusable content.
- The leading brand mention (@AmericanAir) is stripped from the cleaned customer
  text since it's purely an addressing token, not content - it would otherwise
  dominate every single example and add no discriminative signal.
- Other @mentions (co-mentioned users, other handles) are replaced with [USER]
  to reduce sparsity while preserving the fact that a mention occurred.
- Whitespace is collapsed; leading/trailing whitespace stripped.
"""
import html
import re

URL_RE = re.compile(r'https?://\S+')
BRAND_MENTION_RE = re.compile(r'@AmericanAir\b', re.IGNORECASE)
OTHER_MENTION_RE = re.compile(r'@\w+')
WHITESPACE_RE = re.compile(r'\s+')


def clean_customer_text(raw_text: str) -> str:
    t = html.unescape(raw_text)
    t = URL_RE.sub('[URL]', t)
    t = BRAND_MENTION_RE.sub('', t)
    t = OTHER_MENTION_RE.sub('[USER]', t)
    t = WHITESPACE_RE.sub(' ', t).strip()
    return t


def clean_agent_text(raw_text: str) -> str:
    """Historical AmericanAir reply text. Same cleaning, but we do NOT strip the
    leading customer mention (e.g. '@105904') since in the agent reply that
    mention IS the addressing token and stripping the customer's [USER] mention
    still preserves the fact a reply was directed at someone."""
    t = html.unescape(raw_text)
    t = URL_RE.sub('[URL]', t)
    t = OTHER_MENTION_RE.sub('[USER]', t)
    t = WHITESPACE_RE.sub(' ', t).strip()
    return t


def normalize_for_dedup(clean_text: str) -> str:
    """Aggressive normalization used ONLY for duplicate/near-duplicate detection,
    never for modeling. Lowercases and strips all non-alphanumeric characters."""
    t = clean_text.lower()
    t = re.sub(r'\[url\]|\[user\]', '', t)
    t = re.sub(r'[^a-z0-9\s]', '', t)
    t = WHITESPACE_RE.sub(' ', t).strip()
    return t


PLACEHOLDER_RE = re.compile(r'\[URL\]|\[USER\]')


def is_effectively_empty(clean_text: str, min_alnum_chars: int = 2) -> bool:
    """True if, after cleaning, there's essentially no real content left
    (e.g. the tweet was just '@AmericanAir' plus a photo/link).

    IMPORTANT: placeholder tokens ([URL], [USER]) are stripped BEFORE counting
    alphanumeric characters - otherwise a link-only or mention-only tweet would
    count "URL"/"USER" as real content and never get flagged as empty."""
    without_placeholders = PLACEHOLDER_RE.sub('', clean_text)
    alnum = re.sub(r'[^a-zA-Z0-9]', '', without_placeholders)
    return len(alnum) < min_alnum_chars
