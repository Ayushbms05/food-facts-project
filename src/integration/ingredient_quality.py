"""I8.7a: conservative text triage, not food recognition or OCR accuracy."""
import re
from .nova_service import InputError, validate_text

RULE_VERSION = '0.1'
# Narrow list for common English short ingredient names; not a food dictionary.
SHORT_INGREDIENTS = {'oil', 'tea', 'egg', 'pea', 'rye', 'soy', 'dal', 'yam', 'nut', 'oat', 'gel', 'ice', 'fig', 'ghee', 'rai'}
HEADING = re.compile(r'^\s*ingredients?\b\s*[:\-–—]?\s*', re.I)
METADATA = re.compile(r'(?im)^\s*(?:nutrition(?:al)?\s+(?:facts|information|values)|best\s+before|expiry\s+date|batch\s+(?:no\.?|number)|manufactured\s+by|net\s+weight|mrp)\b')


def assess_ingredients(text, source='manual', reviewed=False):
    """Return accept/review/reject; never rewrite inference text or run the model."""
    try:
        clean = validate_text(text, source, reviewed)
    except InputError as exc:
        return {'decision': 'reject', 'reason': 'basic_validation', 'message': str(exc)}
    # Remove only an initial heading for inspection, not from model input.
    body = HEADING.sub('', clean, count=1).strip()
    if not any(c.isalpha() for c in body):
        return {'decision': 'reject', 'reason': 'heading_without_ingredients'}
    if METADATA.search(clean):
        return {'decision': 'review', 'reason': 'possible_non_ingredient_label_text'}
    letters = ''.join(c for c in body if c.isalpha())
    # One tiny alphabetic fragment is suspicious, not proof of invalidity.
    if len(letters) <= 3 and letters.casefold() not in SHORT_INGREDIENTS:
        return {'decision': 'review', 'reason': 'unrecognized_short_fragment'}
    return {'decision': 'accept', 'reason': 'no_obvious_issue_detected'}
