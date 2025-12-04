import re
from typing import Tuple


def clean_and_validate_plate(raw_text: str, country: str = "IN") -> Tuple[str, bool]:
    """
    Clean raw OCR text and optionally validate against a simple regex pattern.

    Returns:
        cleaned_text: uppercase alphanumeric, no spaces/hyphens
        is_valid: whether it matches a simple country-specific pattern
    """
    if raw_text is None:
        return "", False

    # Remove spaces and non-alphanumeric, make uppercase
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()

    if not cleaned:
        return "", False

    is_valid = False

    if country == "IN":
        # Very simple Indian plate pattern:
        # e.g. TN10AB1234, KA01AA0001, etc.
        # 2 letters + 2 digits + 1-2 letters + 4 digits
        pattern = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$")
        if pattern.match(cleaned):
            is_valid = True
    else:
        # For other countries, just accept cleaned text for now
        is_valid = True

    return cleaned, is_valid
