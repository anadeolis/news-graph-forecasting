"""LLM relation extractor (the API one).

Same job and same output format as extraction_rules.py, but an LLM reads the
headline instead of keyword patterns. Where the rules extractor needs an
explicit trigger word and a name in its dictionary, this one can read
"Boeing taps Spirit AeroSystems to build fuselage sections" as a supplier
link and name a counterparty outside the 100-firm universe.


Output: data/processed/edges_llm.csv
        (date, ticker_a, ticker_b, relation, headline)
"""

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from pydantic import BaseModel, Field

from data import PROCESSED
from news import load_articles

MODEL = "claude-opus-4-8"
BATCH_SIZE = 20
CACHE_FILE = PROCESSED / "llm_cache.jsonl"

RELATIONS = ["supplier", "customer", "competitor", "acquisition", "partnership"]

# ── EXTRACTION INSTRUCTIONS ─────────────────────────────────
# These encode the same judgment calls as the hand labels: the relation
# must be stated in the headline, co-mention is not a relationship, and
# an investor buying shares is not an acquisition.
SYSTEM = f"""You extract firm-to-firm business relationships from financial news headlines.

For each headline, decide whether it states a relationship between two specific
companies. Relation types: {", ".join(RELATIONS)}.

Rules:
- The relationship must be stated or clearly implied by the headline itself. Do
  not use background knowledge about which firms are related.
- Two companies merely appearing in the same headline is NOT a relationship.
  Earnings calendars, market roundups, stock lists, and analyst price-target
  lists contain no relationships.
- An investment fund buying or selling SHARES of a company is NOT an
  acquisition. Ignore 13F filings and portfolio-holding reports.
- A bank advising on someone else's deal is not a party to that deal.
- BOTH parties must be companies. Cities, governments, agencies, universities,
  people, products, and stock indices are not companies — skip those pairs.
- Litigation is not a relationship. Suing, settling a dispute, or facing the
  same regulator is not a partnership.
- Name both companies as they appear in the headline, even if one is small,
  private, or foreign. Use the company name, not a ticker.
- "acquisition" means firm_a is acquiring or merging with firm_b, so order
  matters. For the symmetric types the order does not matter.
- Reported, rumored, and in-talks deals count; label them with rumor=true.
- A headline may contain zero, one, or several relationships.

Return one entry per relationship found. Return an empty list for headlines
with no relationship."""


class Relation(BaseModel):
    headline_id: int = Field(description="the id of the headline this came from")
    firm_a: str = Field(description="first company, as named in the headline")
    firm_b: str = Field(description="second company, as named in the headline")
    relation: str = Field(description=f"one of: {', '.join(RELATIONS)}")
    rumor: bool = Field(description="true if reported/rumored/in talks rather than confirmed")


class Extraction(BaseModel):
    relations: list[Relation]


MIN_FIRMS_IN_TEXT = 1


def candidates() -> pd.DataFrame:
    """Headlines worth sending to the LLM.

    Two filters, in order. FNSPID's ticker tags narrow 517k articles to the
    24,358 tagged with 2+ universe firms — but 79% of those never name a
    universe firm in the headline text, because the tags are attached
    loosely. Requiring the name to appear in the text drops the pool to
    ~5,200 and cuts the API bill by the same proportion.

    MIN_FIRMS_IN_TEXT = 1 deliberately keeps headlines naming one universe
    firm and one outsider ("Marriott To Acquire Starwood"): those edges are
    still information about the universe firm. Raise to 2 for a smaller,
    universe-only pool.
    """
    from extraction_rules import find_firm_mentions, load_name_map

    articles = load_articles()
    universe = set(pd.read_csv(PROCESSED / "universe_top100.csv")["ticker"])
    n_tagged = articles["tickers"].apply(lambda ts: sum(t in universe for t in ts))
    articles = articles[n_tagged >= 2]

    name_map = load_name_map()
    n_named = articles["title"].apply(
        lambda t: len(find_firm_mentions(t, name_map)) if isinstance(t, str) else 0)
    return articles[n_named >= MIN_FIRMS_IN_TEXT].reset_index(drop=True)


def _key(headline: str) -> str:
    return hashlib.sha1(headline.encode()).hexdigest()[:16]


def load_cache() -> dict[str, list[dict]]:
    if not CACHE_FILE.exists():
        return {}
    cache = {}
    with open(CACHE_FILE) as f:
        for line in f:
            row = json.loads(line)
            cache[row["key"]] = row["relations"]
    return cache


def extract_batch(client: Anthropic, headlines: list[str]) -> list[dict]:
    """Send one batch of headlines, return the relations found."""
    numbered = "\n".join(f"{i}. {h}" for i, h in enumerate(headlines))
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM,
        output_config={"effort": "low"},
        output_format=Extraction,
        messages=[{"role": "user", "content": f"Headlines:\n\n{numbered}"}],
    )
    out = response.parsed_output
    return [r.model_dump() for r in out.relations] if out else []


def run(limit: int | None = None, seed: int = 0) -> None:
    """Extract relations. limit=N runs a random N-headline pilot.

    The pilot samples at random rather than taking the first N, which would
    only cover 2009-2010 — the thinnest years of news coverage.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("set ANTHROPIC_API_KEY first (see module docstring)")

    client = Anthropic()
    articles = candidates()
    if limit:
        articles = articles.sample(min(limit, len(articles)), random_state=seed)
        articles = articles.sort_values("date").reset_index(drop=True)

    cache = load_cache()
    todo = [h for h in articles["title"] if _key(h) not in cache]
    print(f"{len(articles)} headlines, {len(todo)} not yet extracted")

    with open(CACHE_FILE, "a") as cache_file:
        for start in range(0, len(todo), BATCH_SIZE):
            batch = todo[start:start + BATCH_SIZE]
            try:
                found = extract_batch(client, batch)
            except Exception as exc:  # keep going; the batch stays uncached
                print(f"  batch at {start} failed: {exc}")
                continue
            by_headline: dict[int, list[dict]] = {}
            for rel in found:
                by_headline.setdefault(rel.pop("headline_id"), []).append(rel)
            for i, headline in enumerate(batch):
                row = {"key": _key(headline), "relations": by_headline.get(i, [])}
                cache_file.write(json.dumps(row) + "\n")
                cache[row["key"]] = row["relations"]
            cache_file.flush()
            print(f"  {start + len(batch)}/{len(todo)} headlines, "
                  f"{sum(len(v) for v in cache.values())} relations so far")

    rows = []
    for _, art in articles.iterrows():
        for rel in cache.get(_key(art["title"]), []):
            rows.append({"date": art["date"], "firm_a": rel["firm_a"],
                         "firm_b": rel["firm_b"], "relation": rel["relation"],
                         "rumor": rel["rumor"], "headline": art["title"]})
    edges = pd.DataFrame(rows)
    edges.to_csv(PROCESSED / "edges_llm.csv", index=False)

    print(f"\nrelations found: {len(edges)}")
    if len(edges):
        print(edges["relation"].value_counts().to_string())


if __name__ == "__main__":
    run()  # full pool; pass limit=200 for a pilot
