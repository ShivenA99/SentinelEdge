"""URL-specific feature extraction for phishing URL detection.

Extracts 14 structural features from a URL string that are commonly
indicative of phishing or malicious links.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse, parse_qs


# Suspicious TLDs frequently abused for phishing
_SUSPICIOUS_TLDS: frozenset[str] = frozenset({
    ".tk", ".ml", ".ga", ".cf", ".gq",  # Freenom free TLDs
    ".xyz", ".top", ".buzz", ".club", ".work",
    ".info", ".zip", ".mov", ".su", ".pw",
    ".cc", ".ws", ".icu", ".cyou", ".rest",
})

# Regex for detecting raw IP addresses in the hostname
_IP_PATTERN = re.compile(
    r"^(?:\d{1,3}\.){3}\d{1,3}$"   # IPv4
    r"|^\[?[0-9a-fA-F:]+\]?$"      # IPv6 (with optional brackets)
)


def _shannon_entropy(s: str) -> float:
    """Compute Shannon entropy (in bits) of a string.

    Parameters
    ----------
    s : str
        Input string (e.g. the domain portion of a URL).

    Returns
    -------
    float
        Entropy value.  Higher values suggest randomised domain names
        often used by DGAs (domain generation algorithms).
    """
    if not s:
        return 0.0
    freq = Counter(s)
    length = len(s)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in freq.values()
    )


def extract_url_features(url: str) -> dict[str, float]:
    """Extract 14 structural features from a URL.

    Parameters
    ----------
    url : str
        A URL string.  If no scheme is present ``https://`` is assumed.

    Returns
    -------
    dict[str, float]
        Feature name -> numeric value mapping.
    """
    # Normalise: ensure a scheme exists so urlparse works correctly.
    normalised = url.strip()
    if not re.match(r"^[a-zA-Z]+://", normalised):
        normalised = "https://" + normalised

    parsed = urlparse(normalised)

    hostname = parsed.hostname or ""
    path = parsed.path or ""

    # --- 1. url_length ---
    url_length = float(len(url))

    # --- 2. dot_count ---
    dot_count = float(hostname.count("."))

    # --- 3. hyphen_count ---
    hyphen_count = float(hostname.count("-"))

    # --- 4. at_symbol ---
    at_symbol = 1.0 if "@" in url else 0.0

    # --- 5. has_ip ---
    has_ip = 1.0 if _IP_PATTERN.match(hostname) else 0.0

    # --- 6. digit_ratio ---
    alpha_digits = [c for c in hostname if c.isalnum()]
    digit_ratio = (
        sum(1 for c in alpha_digits if c.isdigit()) / len(alpha_digits)
        if alpha_digits
        else 0.0
    )

    # --- 7. subdomain_count ---
    # "www.sub.example.com" -> parts = ["www", "sub", "example", "com"]
    # subdomains = total parts - 2 (domain + TLD), but at least 0
    parts = hostname.split(".") if hostname else []
    subdomain_count = float(max(len(parts) - 2, 0))

    # --- 8. path_depth ---
    # "/a/b/c" -> depth = 3
    path_segments = [seg for seg in path.split("/") if seg]
    path_depth = float(len(path_segments))

    # --- 9. has_https ---
    has_https = 1.0 if parsed.scheme.lower() == "https" else 0.0

    # --- 10. domain_entropy ---
    domain_entropy = _shannon_entropy(hostname)

    # --- 11. suspicious_tld ---
    suspicious_tld = 0.0
    if hostname:
        last_dot = hostname.rfind(".")
        if last_dot != -1:
            tld = hostname[last_dot:]
            suspicious_tld = 1.0 if tld.lower() in _SUSPICIOUS_TLDS else 0.0

    # --- 12. has_port ---
    has_port = 1.0 if parsed.port is not None else 0.0

    # --- 13. query_param_count ---
    query_params = parse_qs(parsed.query)
    query_param_count = float(len(query_params))

    # --- 14. fragment_present ---
    fragment_present = 1.0 if parsed.fragment else 0.0

    return {
        "url_length": url_length,
        "dot_count": dot_count,
        "hyphen_count": hyphen_count,
        "at_symbol": at_symbol,
        "has_ip": has_ip,
        "digit_ratio": digit_ratio,
        "subdomain_count": subdomain_count,
        "path_depth": path_depth,
        "has_https": has_https,
        "domain_entropy": domain_entropy,
        "suspicious_tld": suspicious_tld,
        "has_port": has_port,
        "query_param_count": query_param_count,
        "fragment_present": fragment_present,
    }
