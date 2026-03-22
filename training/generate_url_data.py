"""Generate synthetic phishing and legitimate URL data for training.

Produces a CSV file with columns: url, label
  - label: 1 = phishing, 0 = legitimate
  - ~5,000 phishing and ~5,000 legitimate URLs (~10,000 total)

Usage
-----
    python training/generate_url_data.py [--output data/raw/phishing_urls/urls.csv]
                                          [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import string
import sys


# ======================================================================
# Legitimate URL patterns
# ======================================================================

_LEGIT_DOMAINS = [
    # Search engines
    "https://www.google.com/search?q={query}",
    "https://www.bing.com/search?q={query}",
    "https://search.yahoo.com/search?p={query}",
    "https://duckduckgo.com/?q={query}",

    # Social media
    "https://www.facebook.com/{username}",
    "https://twitter.com/{username}",
    "https://www.instagram.com/{username}/",
    "https://www.linkedin.com/in/{username}",
    "https://www.reddit.com/r/{subreddit}",
    "https://www.reddit.com/r/{subreddit}/comments/{code}",
    "https://www.tiktok.com/@{username}",
    "https://www.pinterest.com/{username}/",
    "https://www.youtube.com/watch?v={videocode}",
    "https://www.youtube.com/channel/{code}",
    "https://www.twitch.tv/{username}",

    # Tech / developer
    "https://github.com/{username}/{repo}",
    "https://github.com/{username}/{repo}/issues/{number}",
    "https://github.com/{username}/{repo}/pull/{number}",
    "https://gitlab.com/{username}/{repo}",
    "https://stackoverflow.com/questions/{number}/{slug}",
    "https://docs.python.org/3/library/{module}.html",
    "https://docs.python.org/3/tutorial/{topic}.html",
    "https://developer.mozilla.org/en-US/docs/Web/{topic}",
    "https://www.npmjs.com/package/{package}",
    "https://pypi.org/project/{package}/",
    "https://crates.io/crates/{package}",
    "https://hub.docker.com/_/{package}",
    "https://medium.com/@{username}/{slug}",
    "https://dev.to/{username}/{slug}",
    "https://hackernews.com/item?id={number}",

    # Reference / education
    "https://en.wikipedia.org/wiki/{article}",
    "https://www.britannica.com/topic/{article}",
    "https://www.khanacademy.org/math/{topic}",
    "https://www.coursera.org/learn/{course}",
    "https://www.edx.org/course/{course}",
    "https://www.udemy.com/course/{course}/",
    "https://scholar.google.com/scholar?q={query}",
    "https://arxiv.org/abs/{arxiv_id}",

    # News
    "https://www.nytimes.com/{year}/{month}/{day}/{section}/{slug}.html",
    "https://www.bbc.com/news/{section}-{number}",
    "https://www.cnn.com/{year}/{month}/{day}/{section}/{slug}/index.html",
    "https://www.reuters.com/{section}/{slug}",
    "https://www.theguardian.com/{section}/{slug}",
    "https://www.washingtonpost.com/{section}/{slug}/",
    "https://apnews.com/article/{code}",

    # E-commerce / business
    "https://www.amazon.com/dp/{asin}",
    "https://www.amazon.com/{slug}/dp/{asin}",
    "https://www.ebay.com/itm/{number}",
    "https://www.etsy.com/listing/{number}/{slug}",
    "https://www.walmart.com/ip/{slug}/{number}",
    "https://www.target.com/p/{slug}/-/A-{number}",
    "https://www.bestbuy.com/site/{slug}/{number}.p",
    "https://www.shopify.com/{section}",
    "https://www.apple.com/{product}/",
    "https://www.samsung.com/us/{product}/",
    "https://store.google.com/product/{product}",

    # Government / institutional
    "https://www.usa.gov/{section}",
    "https://www.irs.gov/{section}/{topic}",
    "https://www.cdc.gov/{section}/{topic}",
    "https://www.nih.gov/{section}/{topic}",
    "https://www.nasa.gov/{section}/{topic}",
    "https://www.whitehouse.gov/{section}/",
    "https://www.sec.gov/{section}/{topic}",
    "https://www.fda.gov/{section}/{topic}",
    "https://www.ed.gov/{section}",
    "https://www.state.gov/{section}/{topic}",

    # Services
    "https://mail.google.com/mail/u/0/",
    "https://outlook.live.com/mail/0/",
    "https://calendar.google.com/calendar/u/0/r",
    "https://drive.google.com/drive/my-drive",
    "https://docs.google.com/document/d/{code}/edit",
    "https://docs.google.com/spreadsheets/d/{code}/edit",
    "https://www.dropbox.com/home",
    "https://www.icloud.com/mail/",
    "https://app.slack.com/client/{code}/{code}",
    "https://zoom.us/j/{number}",
    "https://meet.google.com/{code}",

    # Entertainment
    "https://www.netflix.com/browse",
    "https://www.spotify.com/us/",
    "https://music.apple.com/us/album/{slug}/{number}",
    "https://www.imdb.com/title/{imdb_id}/",
    "https://www.rottentomatoes.com/m/{slug}",
    "https://store.steampowered.com/app/{number}/{slug}/",

    # Simple domains
    "https://www.google.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.amazon.com",
    "https://www.facebook.com",
    "https://www.twitter.com",
    "https://www.github.com",
    "https://www.stackoverflow.com",
    "https://www.wikipedia.org",
    "https://www.python.org",
    "https://www.rust-lang.org",
    "https://www.nodejs.org",
    "https://www.cloudflare.com",
    "https://www.digitalocean.com",
    "https://www.heroku.com",
    "https://www.netlify.com",
    "https://www.vercel.com",
]


# ======================================================================
# Phishing URL patterns
# ======================================================================

_SUSPICIOUS_TLDS = [".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".pw",
                    ".cc", ".club", ".work", ".date", ".racing", ".stream",
                    ".download", ".win", ".bid", ".trade", ".webcam", ".review",
                    ".party", ".cricket", ".science", ".click", ".link", ".info",
                    ".zip", ".mov", ".icu", ".buzz", ".rest", ".cyou", ".su"]

_TARGET_BRANDS = [
    "paypal", "bankofamerica", "chase", "wellsfargo", "citibank",
    "apple", "microsoft", "google", "amazon", "facebook", "instagram",
    "netflix", "spotify", "dropbox", "linkedin", "twitter", "yahoo",
    "outlook", "icloud", "usps", "fedex", "ups", "dhl",
    "walmart", "bestbuy", "ebay", "coinbase", "blockchain",
]

_MISSPELLED_BRANDS = [
    "paypa1", "paypai", "payp4l", "pay-pal", "paypaI",
    "bankofamerika", "bank0famerica", "bankofamercia",
    "ch4se", "chas3", "chasse",
    "well5fargo", "wellsf4rgo", "weIlsfargo",
    "app1e", "appie", "4pple",
    "micros0ft", "microsooft", "microsft", "mlcrosoft",
    "gooogle", "g00gle", "googIe", "go0gle",
    "amaz0n", "amazom", "arnazon", "amaz0m",
    "faceb00k", "faecbook", "fac3book",
    "netf1ix", "netfIix", "netflx",
    "dropb0x", "dr0pbox",
    "lnstagram", "1nstagram", "instagr4m",
    "linkedln", "l1nkedin",
    "tw1tter", "twtter",
    "yah00", "yaho0",
]

_PHISH_PATH_PARTS = [
    "login", "signin", "verify", "secure", "update", "confirm",
    "account", "authenticate", "validate", "auth", "security",
    "password", "reset", "recover", "unlock", "billing",
    "payment", "checkout", "invoice", "receipt", "refund",
    "support", "help", "service", "customer", "admin",
    "webmail", "portal", "access", "session", "token",
]

_PHISH_PARAMS = [
    "id", "token", "session", "ref", "user", "email", "uid",
    "action", "redirect", "return", "callback", "next", "verify",
    "confirm", "code", "key", "hash", "sig", "auth",
]


def _random_hex(length: int) -> str:
    return "".join(random.choices("0123456789abcdef", k=length))


def _random_alnum(length: int) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def _random_encoded_chars() -> str:
    """Generate some percent-encoded characters."""
    encoded = ["%20", "%3D", "%26", "%3F", "%2F", "%23", "%40", "%3A"]
    return "".join(random.choices(encoded, k=random.randint(1, 3)))


def _gen_phishing_url_brand_subdomain() -> str:
    """Brand-impersonation via subdomains: secure-paypal.evil.tk"""
    brand = random.choice(_TARGET_BRANDS)
    tld = random.choice(_SUSPICIOUS_TLDS)
    prefix = random.choice(["secure", "login", "verify", "account", "update", "auth", "alert"])
    evil_word = _random_alnum(random.randint(4, 8))
    path = random.choice(_PHISH_PATH_PARTS)
    sep = random.choice(["-", ".", ""])

    patterns = [
        f"http://{prefix}{sep}{brand}{sep}login.{evil_word}{tld}/{path}",
        f"http://{prefix}-{brand}.{evil_word}{tld}/{path}?id={_random_hex(16)}",
        f"http://{brand}-{prefix}.{evil_word}{tld}/{path}/{_random_alnum(8)}",
        f"http://{brand}.{prefix}.{evil_word}{tld}/{path}?token={_random_hex(24)}",
        f"http://{prefix}.{brand}-service.{evil_word}{tld}/{path}",
    ]
    return random.choice(patterns)


def _gen_phishing_url_misspelled() -> str:
    """Misspelled brand in domain."""
    brand = random.choice(_MISSPELLED_BRANDS)
    tld = random.choice([".com", ".net", ".org"] + _SUSPICIOUS_TLDS[:10])
    path = random.choice(_PHISH_PATH_PARTS)
    param = random.choice(_PHISH_PARAMS)

    patterns = [
        f"http://{brand}{tld}/{path}",
        f"https://www.{brand}{tld}/{path}?{param}={_random_hex(12)}",
        f"http://{brand}{tld}/index.php?{param}={_random_alnum(16)}",
        f"https://{brand}{tld}/{path}/form.html",
        f"http://www.{brand}{tld}/{path}?redirect={_random_alnum(10)}",
    ]
    return random.choice(patterns)


def _gen_phishing_url_ip_address() -> str:
    """URLs using IP addresses."""
    ip = f"{random.randint(1,254)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    port = random.choice(["", ":8080", ":8443", ":3000", ":443", ":4443", ":9090"])
    brand = random.choice(_TARGET_BRANDS)
    path = random.choice(_PHISH_PATH_PARTS)

    patterns = [
        f"http://{ip}{port}/{brand}/{path}",
        f"http://{ip}{port}/{path}.php?user={_random_alnum(8)}",
        f"http://{ip}{port}/~{brand}/{path}/index.html",
        f"http://{ip}{port}/cgi-bin/{path}.cgi?id={random.randint(10000,99999)}",
        f"https://{ip}{port}/{brand}-{path}/",
        f"http://{ip}{port}/{_random_alnum(6)}/{path}",
    ]
    return random.choice(patterns)


def _gen_phishing_url_long_subdomain() -> str:
    """Long URLs with many subdomains."""
    brand = random.choice(_TARGET_BRANDS)
    tld = random.choice(_SUSPICIOUS_TLDS[:8])
    n_sub = random.randint(3, 6)
    words = random.choices(["secure", "login", "account", "verify", "update", "auth",
                            "service", "alert", "help", "support", "portal"], k=n_sub)
    subdomains = ".".join(words)
    path = "/".join(random.choices(_PHISH_PATH_PARTS, k=random.randint(2, 4)))
    evil = _random_alnum(random.randint(4, 7))

    return f"http://{brand}.{subdomains}.{evil}{tld}/{path}?session={_random_hex(20)}"


def _gen_phishing_url_encoded() -> str:
    """URLs with encoded characters."""
    brand = random.choice(_TARGET_BRANDS)
    tld = random.choice(_SUSPICIOUS_TLDS[:6])
    evil = _random_alnum(random.randint(5, 8))
    path = random.choice(_PHISH_PATH_PARTS)
    encoded = _random_encoded_chars()

    patterns = [
        f"http://{brand}-{path}.{evil}{tld}/index{encoded}.html",
        f"http://{evil}{tld}/{brand}/{path}{encoded}",
        f"http://{brand}.{evil}{tld}/{path}?redirect={encoded}{_random_alnum(8)}",
        f"http://{evil}{tld}/@{brand}/{path}{encoded}?id={_random_hex(8)}",
    ]
    return random.choice(patterns)


def _gen_phishing_url_at_symbol() -> str:
    """URLs with @ symbol for credential trick."""
    brand = random.choice(_TARGET_BRANDS)
    tld = random.choice(_SUSPICIOUS_TLDS[:6])
    evil = _random_alnum(random.randint(4, 8))
    path = random.choice(_PHISH_PATH_PARTS)

    patterns = [
        f"http://{brand}.com@{evil}{tld}/{path}",
        f"http://login.{brand}.com@{evil}{tld}/{path}?id={_random_hex(8)}",
        f"http://secure.{brand}.com@{evil}{tld}/{path}",
        f"http://www.{brand}.com:login@{evil}{tld}/{path}",
    ]
    return random.choice(patterns)


def _gen_phishing_url_data_uri() -> str:
    """URLs with unusual schemes or data-like paths."""
    brand = random.choice(_TARGET_BRANDS)
    evil = _random_alnum(random.randint(5, 8))
    tld = random.choice(_SUSPICIOUS_TLDS[:8])
    path = random.choice(_PHISH_PATH_PARTS)
    hex_str = _random_hex(random.randint(32, 64))

    patterns = [
        f"http://{evil}{tld}/{brand}/{path}/{hex_str}",
        f"http://{evil}{tld}/{hex_str}/{brand}.html",
        f"http://{evil}{tld}/d/{hex_str}?brand={brand}",
        f"http://{brand}-{evil}{tld}/token/{hex_str}",
    ]
    return random.choice(patterns)


def _gen_phishing_url_hyphenated() -> str:
    """Heavily hyphenated domain names."""
    brand = random.choice(_TARGET_BRANDS)
    tld = random.choice(_SUSPICIOUS_TLDS[:10])
    words = random.choices(["secure", "login", "update", "verify", "account",
                            "online", "web", "portal", "client", "user"], k=random.randint(2, 4))
    domain = "-".join([brand] + words)
    path = random.choice(_PHISH_PATH_PARTS)

    return f"http://{domain}{tld}/{path}?ref={_random_alnum(10)}"


# ======================================================================
# Legitimate URL generation with substitution
# ======================================================================

_SEARCH_QUERIES = [
    "weather+today", "python+tutorial", "best+restaurants+near+me",
    "how+to+cook+pasta", "machine+learning+basics", "stock+market+news",
    "covid+vaccine+schedule", "recipe+chocolate+cake", "flight+status",
    "movie+showtimes", "diy+home+repair", "yoga+for+beginners",
    "best+hiking+trails", "used+cars+for+sale", "apartments+for+rent",
    "cat+adoption+near+me", "budget+travel+tips", "healthy+meal+prep",
    "remote+jobs+2024", "baby+names+popular",
]

_SUBREDDITS = [
    "python", "javascript", "AskReddit", "todayilearned", "science",
    "worldnews", "technology", "gaming", "movies", "books",
    "fitness", "cooking", "gardening", "photography", "DIY",
    "datascience", "machinelearning", "personalfinance", "travel", "music",
]

_USERNAMES = [
    "john_doe", "techguru42", "sarah_dev", "codingmaster", "data_wizard",
    "pixel_artist", "green_thumb", "bookworm99", "fitness_fan", "traveler_x",
    "jane_smith", "devops_ninja", "cloud_arch", "ml_researcher", "web_builder",
]

_REPOS = [
    "awesome-python", "tensorflow", "react", "next.js", "fastapi",
    "flask", "django", "express", "rust-lang", "go-ethereum",
    "vscode", "terminal", "cli-tool", "data-pipeline", "ml-model",
]

_PY_MODULES = [
    "os", "sys", "json", "re", "math", "random", "datetime",
    "collections", "itertools", "functools", "pathlib", "typing",
    "argparse", "logging", "unittest", "dataclasses", "enum",
]

_NPM_PACKAGES = [
    "react", "express", "lodash", "axios", "next", "typescript",
    "webpack", "eslint", "prettier", "jest", "mocha", "tailwindcss",
]

_WIKI_ARTICLES = [
    "Machine_learning", "Python_(programming_language)", "Artificial_intelligence",
    "Deep_learning", "Computer_science", "Data_science", "Internet",
    "Climate_change", "Quantum_computing", "Cryptocurrency",
    "World_War_II", "United_States", "Solar_System", "DNA",
    "Olympic_Games", "Mathematics", "Philosophy", "Evolution",
]

_ARXIV_IDS = [
    "2301.07041", "2312.00752", "2310.06825", "2309.16609",
    "2308.08155", "2307.09288", "2306.15595", "2305.18290",
]

_IMDB_IDS = [
    "tt0111161", "tt0068646", "tt0071562", "tt0468569",
    "tt0108052", "tt0167260", "tt0110912", "tt0137523",
]

_ASINS = [
    "B08N5WRWNW", "B09V3KXJPB", "B0BSHF7WHW", "B07XJ8C8F5",
    "B0C2J3KT4N", "B09G9FPHY6", "B0BT2KG7J5", "B07ZPKN6YR",
]

_NEWS_SECTIONS = ["politics", "business", "technology", "science", "health", "sports", "world"]
_NEWS_SLUGS = [
    "new-climate-report-released", "tech-earnings-beat-expectations",
    "ai-breakthrough-study", "market-rally-continues",
    "healthcare-reform-debate", "space-mission-update",
    "election-results-analysis", "economic-outlook-report",
]

_PRODUCTS = [
    "iphone-15-pro", "macbook-air", "ipad-pro", "airpods-pro",
    "galaxy-s24", "pixel-8", "surface-pro", "xbox-series-x",
]

_COURSES = [
    "machine-learning", "python-for-everybody", "data-science-specialization",
    "deep-learning-ai", "web-development-bootcamp", "algorithms-part1",
]


def _fill_legit_template(template: str) -> str:
    """Replace placeholders in a legitimate URL template."""
    result = template
    while "{" in result:
        old = result
        result = result.replace("{query}", random.choice(_SEARCH_QUERIES), 1)
        result = result.replace("{username}", random.choice(_USERNAMES), 1)
        result = result.replace("{subreddit}", random.choice(_SUBREDDITS), 1)
        result = result.replace("{repo}", random.choice(_REPOS), 1)
        result = result.replace("{number}", str(random.randint(100000, 9999999)), 1)
        result = result.replace("{code}", _random_alnum(random.randint(8, 16)), 1)
        result = result.replace("{videocode}", _random_alnum(11), 1)
        result = result.replace("{slug}", random.choice(_NEWS_SLUGS + [_random_alnum(random.randint(8, 20))]), 1)
        result = result.replace("{module}", random.choice(_PY_MODULES), 1)
        result = result.replace("{topic}", random.choice(["html", "css", "javascript", "api", "http"]), 1)
        result = result.replace("{package}", random.choice(_NPM_PACKAGES + _PY_MODULES[:5]), 1)
        result = result.replace("{article}", random.choice(_WIKI_ARTICLES), 1)
        result = result.replace("{course}", random.choice(_COURSES), 1)
        result = result.replace("{arxiv_id}", random.choice(_ARXIV_IDS), 1)
        result = result.replace("{year}", str(random.choice([2023, 2024, 2025])), 1)
        result = result.replace("{month}", f"{random.randint(1,12):02d}", 1)
        result = result.replace("{day}", f"{random.randint(1,28):02d}", 1)
        result = result.replace("{section}", random.choice(_NEWS_SECTIONS), 1)
        result = result.replace("{asin}", random.choice(_ASINS), 1)
        result = result.replace("{product}", random.choice(_PRODUCTS), 1)
        result = result.replace("{imdb_id}", random.choice(_IMDB_IDS), 1)
        if result == old:
            break
    return result


# ======================================================================
# Generation
# ======================================================================

# Phishing URL generator functions with weights
_PHISHING_GENERATORS = [
    (_gen_phishing_url_brand_subdomain, 0.25),
    (_gen_phishing_url_misspelled, 0.15),
    (_gen_phishing_url_ip_address, 0.15),
    (_gen_phishing_url_long_subdomain, 0.10),
    (_gen_phishing_url_encoded, 0.10),
    (_gen_phishing_url_at_symbol, 0.08),
    (_gen_phishing_url_data_uri, 0.07),
    (_gen_phishing_url_hyphenated, 0.10),
]


def generate_url_data(
    n_phishing: int = 5000,
    n_legit: int = 5000,
    seed: int = 42,
) -> list[tuple[str, int]]:
    """Generate synthetic URL data.

    Returns list of (url, label) tuples.
    """
    random.seed(seed)
    rows: list[tuple[str, int]] = []

    # ---- Generate phishing URLs ----
    generators = [g for g, _ in _PHISHING_GENERATORS]
    weights = [w for _, w in _PHISHING_GENERATORS]

    for _ in range(n_phishing):
        gen = random.choices(generators, weights=weights, k=1)[0]
        url = gen()
        rows.append((url, 1))

    # ---- Generate legitimate URLs ----
    for _ in range(n_legit):
        template = random.choice(_LEGIT_DOMAINS)
        url = _fill_legit_template(template)
        rows.append((url, 0))

    random.shuffle(rows)
    return rows


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic phishing/legitimate URL data."
    )
    parser.add_argument(
        "--output",
        default="data/raw/phishing_urls/urls.csv",
        help="Output CSV path",
    )
    parser.add_argument("--n-phishing", type=int, default=5000)
    parser.add_argument("--n-legit", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = generate_url_data(
        n_phishing=args.n_phishing,
        n_legit=args.n_legit,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        for url, label in rows:
            writer.writerow([url, label])

    n_phish = sum(1 for _, l in rows if l == 1)
    n_leg = sum(1 for _, l in rows if l == 0)
    print(f"Generated {len(rows):,} URLs:")
    print(f"  Legitimate: {n_leg:,}")
    print(f"  Phishing:   {n_phish:,}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
