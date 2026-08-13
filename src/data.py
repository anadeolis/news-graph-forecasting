"""
Raw inputs (data/raw/):
  pvCLCL_20000103_20201231.csv  close-to-close daily returns, 695 stocks x 5279 days
  volMM_20000103_20201231.csv   daily dollar volume (USD millions), same shape
  Sectors_SP500_YahooNWikipedia.csv, Sectors_SP1500.csv  ticker -> sector
  nyse_business_days_1990_2026.csv  NYSE trading-day calendar (YYYYMMDD, one per line)

Outputs (data/processed/):
  returns_daily.parquet   dates x tickers matrix of close-to-close returns
  sectors.csv             ticker, sector (GICS-style labels)
  universe_top100.csv     100 most liquid tickers by average dollar volume
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def load_matrix(variable: str) -> pd.DataFrame:
    """Load one of the matrix files as a dates x tickers DataFrame.

    The raw file has tickers as rows and day columns named like 'X20000103',
    so the date is embedded in each column name.
    """
    path = RAW / f"{variable}_20000103_20201231.csv"
    df = pd.read_csv(path)
    df = df.set_index(df.columns[0])
    df.index.name = "ticker"
    df.columns = pd.to_datetime([c.lstrip("X") for c in df.columns], format="%Y%m%d")
    panel = df.T.sort_index()
    panel.index.name = "date"
    return panel


def load_calendar() -> pd.DatetimeIndex:
    cal = pd.read_csv(RAW / "nyse_business_days_1990_2026.csv", header=None)[0]
    return pd.DatetimeIndex(pd.to_datetime(cal, format="%Y%m%d"))


def load_sectors(tickers) -> pd.Series:
    """Map each ticker to a sector, preferring the S&P 500 Wikipedia labels
    and filling the rest from the S&P 1500 file. Unmatched tickers -> 'Unknown'.
    """
    sp500 = pd.read_csv(RAW / "Sectors_SP500_YahooNWikipedia.csv")
    m500 = sp500.set_index("Ticker")["Sector_Wikipedia"]

    sp1500 = pd.read_csv(RAW / "Sectors_SP1500.csv", header=None)
    m1500 = sp1500.set_index(2)[3]
    m1500 = m1500[m1500 != "SPY"]

    sectors = pd.Series(index=pd.Index(tickers, name="ticker"), dtype=object, name="sector")
    sectors.update(m1500)
    sectors.update(m500)
    return sectors.fillna("Unknown")


def build_universe(volmm: pd.DataFrame, sectors: pd.Series, n: int = 100) -> list[str]:
    """The n most liquid tickers by full-sample average daily dollar volume.

    Restricted to tickers with a known sector: ETFs and closed-end funds
    (SPY, EWU, ...) have no sector label and must not enter a firm graph.
    """
    firms = sectors[sectors != "Unknown"].index
    return volmm[firms.intersection(volmm.columns)].mean().nlargest(n).index.tolist()


def build_panel(top_n: int = 100) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    returns = load_matrix("pvCLCL")
    volmm = load_matrix("volMM")

    calendar = load_calendar()
    expected = calendar[(calendar >= returns.index[0]) & (calendar <= returns.index[-1])]
    if not returns.index.difference(expected).empty:
        raise ValueError("panel contains dates that are not NYSE trading days")
    dropped = expected.difference(returns.index)
    if len(dropped) > 10:
        raise ValueError(f"panel is missing {len(dropped)} trading days")
    if len(dropped):
        # vendor drops the first day back after unscheduled closures (9/11,
        # funerals, Hurricane Sandy) since those close-to-close returns span
        # multiple days
        print(f"note: {len(dropped)} post-closure days absent from panel: "
              f"{', '.join(dropped.strftime('%Y-%m-%d'))}")
    n_missing = int(returns.isna().sum().sum())
    if n_missing > 50:
        raise ValueError(f"returns panel has {n_missing} missing values — investigate")
    if n_missing:
        # a handful of isolated single-day gaps; treat as zero return
        print(f"note: {n_missing} isolated missing returns set to 0")
        returns = returns.fillna(0.0)

    sectors = load_sectors(returns.columns)
    universe = build_universe(volmm, sectors, n=top_n)

    returns.to_parquet(PROCESSED / "returns_daily.parquet")
    sectors.to_csv(PROCESSED / "sectors.csv")
    pd.Series(universe, name="ticker").to_csv(PROCESSED / f"universe_top{top_n}.csv", index=False)

    print(f"returns panel: {returns.shape[0]} days x {returns.shape[1]} tickers "
          f"({returns.index[0].date()} to {returns.index[-1].date()})")
    print(f"sector coverage: {(sectors != 'Unknown').mean():.0%}")
    print(f"universe saved: top {top_n} by avg dollar volume "
          f"(e.g., {', '.join(universe[:5])} ...)")


def load_returns(universe: str | None = "top100") -> pd.DataFrame:
    """Load the processed panel. Every later stage starts here.

    universe='top100' restricts to the liquid universe; None returns all 695.
    """
    returns = pd.read_parquet(PROCESSED / "returns_daily.parquet")
    if universe is not None:
        tickers = pd.read_csv(PROCESSED / f"universe_{universe}.csv")["ticker"]
        returns = returns[tickers.tolist()]
    return returns


def load_sector_map() -> pd.Series:
    return pd.read_csv(PROCESSED / "sectors.csv", index_col=0)["sector"]


if __name__ == "__main__":
    build_panel()
