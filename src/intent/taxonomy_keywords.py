"""
Shared taxonomy keyword patterns, used to generate WEAK-SUPERVISION pseudo-labels
over the unlabeled train/dev corpus (25,451 / 5,453 pairs). These patterns are
copied from src/preprocessing/sample_golden_set.py (kept in sync manually --
that file is the approved, frozen Step 3 artifact and is not modified here).

IMPORTANT: pseudo-labels produced by this module are NEVER used as evaluation
ground truth. They are only used as noisy training signal for the intent
classifier. All evaluation happens against data/golden_set/labels.json (real
human labels).
"""
import re

TAXONOMY_PRIORITY = [
    'SAFETY_SECURITY', 'REFUND_COMPENSATION', 'FLIGHT_STATUS_DELAY_CANCEL', 'BAGGAGE',
    'BOOKING_CHANGE', 'CHECKIN_BOARDING', 'ACCOUNT_WEBSITE_APP', 'LOYALTY_MILES',
    'COMPLAINT_GENERAL', 'COMPLIMENT',
]
ALL_INTENTS = TAXONOMY_PRIORITY + ['OTHER_UNCLEAR']

BUCKETS = {
    'FLIGHT_STATUS_DELAY_CANCEL': r'\b(delay(ed|s)?|cancel(led|ed|lation|s)?|on time|status|diverted|gate|departure|arriv\w*|affected flights?|flights? affected|scheduling (glitch|issue)|no pilot|missing pilot|eta|updates? on (my |the )?flight|flight (is|has) a (crew|pilot))\b',
    'BOOKING_CHANGE': r'\b(rebook|re-book|change my flight|reschedul\w*|seat change|upgrade|book(ing)?|reservation|pnr|record locator)\b',
    'REFUND_COMPENSATION': r'\b(refund|reimburs\w*|compensat\w*|voucher|credit (me|back)|money back)\b',
    'BAGGAGE': r'\b(bag(s|gage)?|luggage|suitcase|left (my |a )?\w+ (on|in) (the )?(flight|plane|seat)|lost (my )?(bag|item)|damaged bag)\b',
    'CHECKIN_BOARDING': r'\b(check[- ]?in|boarding pass|boarding|gate agent|standby|precheck|pre-check|ktn|known traveler)\b',
    'LOYALTY_MILES': r'\b(aadvantage|miles|loyalty|elite status|frequent flyer|systemwide upgrade)\b',
    'ACCOUNT_WEBSITE_APP': r'\b(app|website|log ?in|password|account (access|locked)|error message)\b',
    'SAFETY_SECURITY': r'\b(medical emergency|paramedic|ambulance|assault(ed)?|threat(en(ed)?)?|discriminat\w*|racis\w*|allerg(y|ic|ies)|tripped over|felt unsafe|security (threat|concern|incident)|groped|harass(ed|ment)?)\b',
    'COMPLIMENT': r'\b(thank you|thanks|great (service|job|crew|flight)|appreciate|awesome|amazing|kudos|top notch|so nice|so helpful|favorite (person|crew|flight attendant))\b',
    'COMPLAINT_GENERAL': r'\b(rude|terrible|worst|horrible|awful|disgust\w*|never fly|unacceptable|disappoint\w*|sucks|shame on you|verbally abused|smh|on hold|spent \d+ hours|customer service)\b',
}
COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in BUCKETS.items()}
SAFETY_FALSE_POSITIVE_RE = re.compile(
    r'\b(safety video|safety demonstration|#safetyfirst|safety is (your |our )?top priority|budget.*safety video)\b',
    re.IGNORECASE)


def weak_label(text: str) -> str:
    """Priority-ordered keyword match -> single pseudo-intent label. Falls back
    to OTHER_UNCLEAR if nothing matches. SAFETY_SECURITY additionally excludes
    known false-positive phrasing (see sample_golden_set.py discovery notes)."""
    for name in TAXONOMY_PRIORITY:
        if COMPILED[name].search(text):
            if name == 'SAFETY_SECURITY' and SAFETY_FALSE_POSITIVE_RE.search(text):
                continue
            return name
    return 'OTHER_UNCLEAR'
