"""Rule-based relation extractor (the "homemade" one).

Reads the article table from stage 2 and emits firm-to-firm edges using
hand-written rules: a junk filter, a company-name dictionary, and keyword
patterns per relation type.

Scope: only detects universe-universe edges, because the name dictionary
only knows the 100 universe firms. Edges to outside firms (e.g. PEP ->
SodaStream) require recognizing arbitrary company names — that is the LLM
extractor's advantage, and part of the comparison.

Output: data/processed/edges_rules.csv
        (date, ticker_a, ticker_b, relation, headline)
"""

import re
from itertools import combinations
from pathlib import Path

import pandas as pd

from news import load_articles
from data import PROCESSED, ROOT

NAMES_FILE = ROOT / "labels" / "company_names.csv"


# Transcribed from the 30 random-sample headlines that were all noise.
JUNK_PATTERNS = [
    r"earnings scheduled",
    r"week ahead",
    r"roundup",
    r"primer",
    r"pre-?market",
    r"mid-?morning",
    r"market in 5 minutes",
    r"wall street breakfast",
    r"movers",
    r"ranked",
    r"price target",
    r"\betf\b",
    r"podcast",
    r"13f",
    r"\bstakes?\b",
    r"to advise on",
    r"advisor to",
    r"advising",
]
# "Sterling Capital Management LLC Buys McKesson..." is a
# portfolio trade, not an acquisition. A fund-like word near Buys/Sells.
FUND_BUYER = re.compile(
    r"\b(management|partners|capital|advisors|investments?|fund|llc|ltd)\b"
    r".{0,40}\b(buys|sells|adds|trims)\b",
    re.IGNORECASE,
)

RELATION_PATTERNS = [
    ("acquisition", re.compile(
        r"acquir|merger|merges? with|takeover|buy-?out", re.IGNORECASE)),
    ("partnership", re.compile(
        r"partner(s|ship|ing)? with|joint venture|teams? up with|"
        r"alliance with|collaborat", re.IGNORECASE)),
    ("supplier", re.compile(
        r"supplier|supply (deal|agreement|contract)|"
        r"awards? .{0,30}contract|contract with", re.IGNORECASE)),
    ("competitor", re.compile(
        r"\brivals?\b|competitors?|competes? with|price war", re.IGNORECASE)),
]

#some names too generic to match alone ("Southern Company" to "southern" which matches "Southern Africa")
BLOCKED_NAMES = {"southern", "gap", "merch &", "deere &"}
NAME_EXCLUSIONS = {
    "MRK": re.compile(r"\s*(kgaa|darmstadt)", re.IGNORECASE),
                      "JNPR": re.compile(r"\s*(pharmaceutical)", re.IGNORECASE),
                      }
                     

ALLOW_RUMORS = True


def _clean_name(name: str) -> str:
    """Strip legal suffixes so 'PepsiCo, Inc.' matches 'PepsiCo'."""
    name = re.sub(r"\s*\(the\)\s*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^the\s+", "", name, flags=re.IGNORECASE)
    suffix = r",?\s+(incorporated|inc\.?|corporation|corp\.?|company|co\.?|ltd\.?|plc|llc|l\.p\.|group)$"
    while re.search(suffix, name, flags=re.IGNORECASE):
        name = re.sub(suffix, "", name, flags=re.IGNORECASE)
    return name.strip()


def load_name_map() -> dict[str, str]:
    """lowercase name variant -> ticker, for all universe firms."""
    df = pd.read_csv(NAMES_FILE)
    name_map = {}
    for _, row in df.iterrows():
        variants = [row["official_name"], _clean_name(row["official_name"])]
        if isinstance(row.get("variants"), str):
            variants += row["variants"].split("|")
        for v in variants:
            v = v.strip().lower()
            if len(v) >= 4 and v not in BLOCKED_NAMES:  # too-short names false-match everywhere
                name_map[v] = row["ticker"]
    return name_map

def load_short_names() -> dict[str, str]:
    """Two- and three-character names, kept case-sensitive: 'IBM' -> IBM.

    These are dropped from load_name_map because lowercasing them creates
    false matches against common words. Matching them as written recovers
    12 firms — IBM, GE, J&J, 3M, P&G, CVS, UPS, and others — that headlines
    almost always refer to by abbreviation.
    """
    df = pd.read_csv(NAMES_FILE)
    short = {}
    for _, row in df.iterrows():
        variants = [row["official_name"], _clean_name(row["official_name"])]
        if isinstance(row.get("variants"), str):
            variants += row["variants"].split("|")
        for v in variants:
            v = v.strip()
            if 2 <= len(v) <= 3 and v.lower() not in BLOCKED_NAMES:
                short[v] = row["ticker"]
    return short


MAX_GAP = 60
MAX_FIRMS =4
HYPHEN_PARTNER = re.compile(r"[-/]\s*([A-Z][A-Za-z&.]+)")

def find_firm_mentions(headline: str, name_map: dict[str, str]) -> list[tuple[int, int, str]]:
    """Where each universe firm is mentioned: (start, end, ticker), in order.

    Longer names win overlaps, so 'Norfolk Southern' claims its span before
    'Southern' can. Parenthesized tickers like '(MMM)' also count.
    """
    text = headline.lower()
    mentions: list[tuple[int, int, str]] = []
    for name in sorted(name_map, key=len, reverse=True):
        for m in re.finditer(r"\b" + re.escape(name) + r"\b", text):
            ticker = name_map[name]
            excl = NAME_EXCLUSIONS.get(ticker)
            if excl and excl.match(text[m.end():]):
                continue
            if not any(m.start() < e and m.end() > s for s, e, _ in mentions):
                mentions.append((m.start(), m.end(), ticker))
    # Short names (IBM, GE, J&J, 3M) are matched case-sensitively against the
    # original text. Lowercased they would collide with ordinary words —
    # "ups" in "ups and downs", "pg" in a page reference — which is why they
    # are excluded from the length-filtered map above.
    for name, ticker in load_short_names().items():
        for m in re.finditer(r"\b" + re.escape(name) + r"\b", headline):
            if not any(m.start() < e and m.end() > s for s, e, _ in mentions):
                mentions.append((m.start(), m.end(), ticker))

    tickers = set(pd.read_csv(NAMES_FILE)["ticker"])
    for m in re.finditer(r"\(([A-Z]{1,5})\)", headline):
        if m.group(1) in tickers and not any(
                m.start() < e and m.end() > s for s, e, _ in mentions):
            mentions.append((m.start(), m.end(), m.group(1)))
    return sorted(mentions)


def is_spoken_for(headline: str, start: int, end: int, mentions) -> bool:
    """True if this firm is hyphen-joined to a company outside the universe.

    'Aetna-Humana' means Humana's deal partner is Aetna, not whichever other
    universe firm happens to appear later in the headline.
    """
    after = headline[end:end + 30]
    m = HYPHEN_PARTNER.match(after)
    if m and not any(end <= s < end + len(m.group(0)) + 1 for s, _, _ in mentions):
        return True
    before = headline[max(0, start - 30):start]
    m2 = re.search(r"([A-Z][A-Za-z&.]+)\s*[-/]$", before)
    if m2 and not any(start - len(m2.group(0)) <= s < start for s, _, _ in mentions):
        return True
    return False


def pair_firms(headline: str, mentions) -> list[tuple[str, str]]:
    """Adjacent, nearby, unspoken-for mentions become edges."""
    if len(mentions) > MAX_FIRMS:
        return []
    keep = [mn for mn in mentions
            if not is_spoken_for(headline, mn[0], mn[1], mentions)]
    pairs = []
    for (_, e1, t1), (s2, _, t2) in zip(keep, keep[1:]):
        if t1 != t2 and s2 - e1 <= MAX_GAP:
            pairs.append(tuple(sorted((t1, t2))))
    return pairs

def is_junk(headline: str) -> bool:
    text = headline.lower()
    if any(re.search(p, text) for p in JUNK_PATTERNS):
        return True
    return bool(FUND_BUYER.search(headline))


def classify(headline: str) -> str | None:
    if not ALLOW_RUMORS and re.search(r"in talks|reportedly|rumor",
                                      headline, re.IGNORECASE):
        return None
    for relation, pattern in RELATION_PATTERNS:
        if pattern.search(headline):
            return relation
    return None


def extract_edges(articles: pd.DataFrame) -> pd.DataFrame:
    name_map = load_name_map()
    rows = []
    for _, art in articles.iterrows():
        title = art["title"]
        if not isinstance(title, str) or is_junk(title):
            continue
        relation = classify(title)
        if relation is None:
            continue
        mentions = find_firm_mentions(title, name_map)
        for a, b in pair_firms(title, mentions):
            rows.append({"date": art["date"], "ticker_a": a, "ticker_b": b,
                         "relation": relation, "headline": title})
    return pd.DataFrame(rows)

def main():
    # Same candidate pool as the LLM extractor, so the two graphs are
    # comparable. Imported inside the function because extraction_llm
    # imports from this module.
    from extraction_llm import candidates

    articles = candidates()
    print(f"candidate headlines: {len(articles)}")
    edges = extract_edges(articles)
    edges.to_csv(PROCESSED / "edges_rules.csv", index=False)
    print(f"edges found: {len(edges)}")
    if len(edges):
        print(edges["relation"].value_counts().to_string())
        print(edges.sample(min(10, len(edges)), random_state=0)
              [["date", "ticker_a", "ticker_b", "relation", "headline"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()