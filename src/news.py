"""2: Filter FNSPID news down to the project's firms.

Raw input (data/raw/, not committed):
  fnspid_all_external.csv   FNSPID "All_external" news, ~5.7 GB
      (Dong et al. 2024, https://huggingface.co/datasets/Zihan1004/FNSPID)
      one row per (article, tagged ticker); columns include Date,
      Article_title, Stock_symbol, Url, Publisher

Outputs (data/processed/):
  news_rows.parquet   rows tagged with any of the 695 panel tickers
  articles.parquet    one row per article: date, title, list of panel
                      tickers tagged, n_tickers — stage 3 reads this
"""

from pathlib import Path

import pandas as pd

from data import PROCESSED, RAW

USECOLS = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher"]


def panel_tickers() -> set[str]:
    return set(pd.read_csv(PROCESSED / "sectors.csv")["ticker"])


def filter_news(chunksize: int = 2_000_000) -> pd.DataFrame:
    """Stream the raw CSV and keep rows tagged with a panel ticker."""
    tickers = panel_tickers()
    kept = []
    reader = pd.read_csv(RAW / "fnspid_all_external.csv", usecols=USECOLS,
                         chunksize=chunksize)
    for i, chunk in enumerate(reader):
        kept.append(chunk[chunk["Stock_symbol"].isin(tickers)])
        print(f"chunk {i + 1}: scanned {chunksize * (i + 1):,} rows, "
              f"kept {sum(len(k) for k in kept):,}")
    news = pd.concat(kept, ignore_index=True)
    news["Date"] = pd.to_datetime(news["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    news = news.dropna(subset=["Date", "Article_title"])
    news = news[news["Date"] <= "2020-12-31"]  # returns panel ends Dec 2020
    return news


def build_article_table(news: pd.DataFrame) -> pd.DataFrame:
    """Collapse (article, ticker) rows to one row per article.

    The same article appears once per tagged ticker; the URL identifies the
    article. Articles tagged with 2+ panel tickers are the prime candidates
    for relation extraction in stage 3.
    """
    key = news["Url"].fillna(news["Article_title"])
    articles = (
        news.assign(key=key)
        .groupby("key")
        .agg(date=("Date", "min"),
             title=("Article_title", "first"),
             publisher=("Publisher", "first"),
             tickers=("Stock_symbol", lambda s: sorted(set(s))))
        .reset_index(drop=True)
    )
    articles["n_tickers"] = articles["tickers"].str.len()
    return articles.sort_values("date").reset_index(drop=True)


def build_news() -> None:
    news = filter_news()
    articles = build_article_table(news)

    news.to_parquet(PROCESSED / "news_rows.parquet")
    articles.to_parquet(PROCESSED / "articles.parquet")

    print(f"\nkept rows: {len(news):,}")
    print(f"unique articles: {len(articles):,} "
          f"({articles['date'].min().date()} to {articles['date'].max().date()})")
    print(f"articles tagged with 2+ panel firms: {(articles['n_tickers'] >= 2).sum():,}")


def load_articles() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "articles.parquet")


if __name__ == "__main__":
    build_news()
