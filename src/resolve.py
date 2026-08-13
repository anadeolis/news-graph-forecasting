"""Map company names from the LLM extractor back to universe tickers.

The rules extractor emits tickers; the LLM emits names as written in the
headline ("Marriott International", "Bristol", "J&J"). Comparing the two
graphs, or building either one, needs the names resolved to tickers.

Matching is word-boundary based and longest-match-wins, the same discipline
as find_firm_mentions. A naive substring test is not safe here: "GE" appears
inside "General Dynamics", so a careless resolver maps GE to GD.
"""

import re

import pandas as pd

from data import PROCESSED
from extraction_rules import load_name_map

# Suffixes journalists append that should not block a match.
SUFFIX = re.compile(
    r"[,\s]+(inc|inc\.|incorporated|corp|corp\.|corporation|co|co\.|company|"
    r"ltd|ltd\.|plc|llc|lp|l\.p\.|group|holdings?|nv|n\.v\.|sa|s\.a\.|ag)$",
    re.IGNORECASE,
)


def _clean(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"^the\s+", "", name)
    while SUFFIX.search(name):
        name = SUFFIX.sub("", name)
    return name.strip(" .,'\"")


def resolve(name: str, name_map: dict[str, str]) -> str | None:
    """Ticker for this company name, or None if it is outside the universe.

    Accepts both directions of partial naming: the headline may write more
    than the dictionary entry ("Marriott International" vs "Marriott") or
    less ("Bristol" vs "Bristol-Myers Squibb"). Requires a whole-word match
    so short names cannot match inside longer ones.
    """
    text = _clean(name)
    if not text:
        return None
    best_len, best_ticker = 0, None
    for variant, ticker in name_map.items():
        if len(variant) < 4:
            continue
        long, short = (text, variant) if len(text) >= len(variant) else (variant, text)
        if re.search(r"\b" + re.escape(short) + r"\b", long) and len(short) > best_len:
            best_len, best_ticker = len(short), ticker
    return best_ticker


def resolve_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Add ticker_a / ticker_b columns to an LLM edge table."""
    name_map = load_name_map()
    cache: dict[str, str | None] = {}

    def lookup(name):
        if name not in cache:
            cache[name] = resolve(name, name_map)
        return cache[name]

    out = edges.copy()
    out["ticker_a"] = out["firm_a"].apply(lookup)
    out["ticker_b"] = out["firm_b"].apply(lookup)
    out["n_universe"] = out[["ticker_a", "ticker_b"]].notna().sum(axis=1)
    return out


def load_llm_edges(universe_only: bool = False) -> pd.DataFrame:
    edges = resolve_edges(pd.read_csv(PROCESSED / "edges_llm.csv"))
    return edges[edges["n_universe"] == 2] if universe_only else edges


if __name__ == "__main__":
    edges = load_llm_edges()
    print(f"total relations: {len(edges)}")
    print(edges["n_universe"].value_counts().sort_index().rename(
        index={0: "neither firm in universe", 1: "one firm in universe",
               2: "both firms in universe"}).to_string())
