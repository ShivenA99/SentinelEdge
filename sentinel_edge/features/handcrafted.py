"""Handcrafted feature extraction for fraud detection.

Extracts 18 linguistic and structural features from a single sentence
of transcript text, designed to capture common phone/SMS fraud patterns.
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Word lists for categorical counting
# ---------------------------------------------------------------------------

_URGENCY_WORDS: list[str] = [
    "urgent", "immediately", "now", "act", "deadline", "expire",
    "hurry", "quickly", "asap", "right away", "limited time", "last chance",
]

_ACTION_WORDS: list[str] = [
    "click", "verify", "confirm", "validate", "login", "update",
    "provide", "submit", "enter", "activate", "download", "install",
]

_FINANCIAL_WORDS: list[str] = [
    "bank", "account", "credit", "payment", "wire", "transfer",
    "routing", "deposit", "withdrawal", "balance", "transaction", "invoice",
]

_IMPERSONATION_WORDS: list[str] = [
    "irs", "fbi", "ssa", "microsoft", "apple", "amazon",
    "google", "police", "government", "federal", "agent", "department",
    "officer",
]

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns
# ---------------------------------------------------------------------------

_URL_PATTERN = re.compile(
    r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE
)

_SHORTENED_URL_PATTERN = re.compile(
    r"(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|is\.gd|buff\.ly"
    r"|rebrand\.ly|cutt\.ly|shorturl\.at)[/\s]?",
    re.IGNORECASE,
)

_VERIFY_PATTERN = re.compile(r"\b(?:verify|confirm|validate)\b", re.IGNORECASE)
_THREAT_PATTERN = re.compile(
    r"\b(?:arrest|suspend|terminate|cancel|lock|seize|revoke|prosecute)",
    re.IGNORECASE,
)
_PRIZE_PATTERN = re.compile(
    r"\b(?:won|winner|prize|congratulations|selected|lucky|reward)\b",
    re.IGNORECASE,
)
_ACCOUNT_REF_PATTERN = re.compile(
    r"\b(?:account|password|pin|ssn|social\s+security|credentials)\b",
    re.IGNORECASE,
)
_PHONE_NUMBER_PATTERN = re.compile(
    r"(?:\+?1[-.\s]?)?"              # optional country code
    r"(?:\(?\d{3}\)?[-.\s]?)"        # area code
    r"\d{3}[-.\s]?\d{4}"             # subscriber number
)


# ---------------------------------------------------------------------------
# Helper: count occurrences of words/phrases in text
# ---------------------------------------------------------------------------

def _count_word_occurrences(text_lower: str, word_list: list[str]) -> int:
    """Count how many times any word in *word_list* appears in *text_lower*.

    Multi-word phrases (e.g. "right away") are matched as substrings.
    Single words are matched on word boundaries to avoid partial matches.
    """
    count = 0
    for word in word_list:
        if " " in word:
            # Multi-word phrase: simple substring count
            count += text_lower.count(word)
        else:
            # Single word: use word boundary regex for precision
            count += len(re.findall(rf"\b{re.escape(word)}\b", text_lower))
    return count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_handcrafted_features(text: str) -> dict[str, float]:
    """Extract all 18 handcrafted features from a single sentence.

    Parameters
    ----------
    text : str
        A single sentence (or short text segment) from a transcript.

    Returns
    -------
    dict[str, float]
        Dictionary mapping feature names to their numeric values.
        Binary features are represented as 0.0 or 1.0.
    """
    text_lower = text.lower()

    # --- Count-based features ---
    urgency_count = _count_word_occurrences(text_lower, _URGENCY_WORDS)
    action_count = _count_word_occurrences(text_lower, _ACTION_WORDS)
    financial_count = _count_word_occurrences(text_lower, _FINANCIAL_WORDS)
    impersonation_count = _count_word_occurrences(text_lower, _IMPERSONATION_WORDS)

    # --- Binary pattern features ---
    urls_found = _URL_PATTERN.findall(text)
    has_url = 1.0 if urls_found else 0.0
    has_shortened_url = 1.0 if _SHORTENED_URL_PATTERN.search(text_lower) else 0.0
    has_verify_pattern = 1.0 if _VERIFY_PATTERN.search(text) else 0.0
    has_threat = 1.0 if _THREAT_PATTERN.search(text) else 0.0
    has_prize = 1.0 if _PRIZE_PATTERN.search(text) else 0.0
    has_account_ref = 1.0 if _ACCOUNT_REF_PATTERN.search(text) else 0.0
    dollar_sign = 1.0 if "$" in text else 0.0
    has_phone_number = 1.0 if _PHONE_NUMBER_PATTERN.search(text) else 0.0

    # --- Structural / statistical features ---
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    avg_word_len = (
        sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
    )
    url_count = len(urls_found)
    exclamation_count = text.count("!")

    alpha_chars = [c for c in text if c.isalpha()]
    caps_ratio = (
        sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if alpha_chars
        else 0.0
    )

    return {
        "urgency_count": float(urgency_count),
        "action_count": float(action_count),
        "financial_count": float(financial_count),
        "impersonation_count": float(impersonation_count),
        "has_url": has_url,
        "has_shortened_url": has_shortened_url,
        "has_verify_pattern": has_verify_pattern,
        "has_threat": has_threat,
        "has_prize": has_prize,
        "has_account_ref": has_account_ref,
        "dollar_sign": dollar_sign,
        "has_phone_number": has_phone_number,
        "char_count": float(char_count),
        "word_count": float(word_count),
        "avg_word_len": avg_word_len,
        "url_count": float(url_count),
        "exclamation_count": float(exclamation_count),
        "caps_ratio": caps_ratio,
    }
