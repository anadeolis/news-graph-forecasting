"""turn the knowledge graph into matrices aligned to the returns panel.

Primary source is the prebuilt graph from DylanSand/financial-news-kg, which
ships rolling-window adjacency matrices as sparse COO arrays:

    <KG_ROOT>/data/output/graphs/
        tickers.json                  global index, 36,897 tickers
        w{20,60,252}/
            unified_{YYYYMMDD}.npz    all relation types summed
            multiplex_{YYYYMMDD}.npz  one layer per relation type

Edge weight is (mention count x average extraction confidence), symmetric,
zero diagonal.

Two things this module handles that the raw files do not:

1. Snapshots are roughly every 20 trading days, not daily. Each trading day
   is mapped to the most recent snapshot STRICTLY BEFORE it, so the graph
   used to predict day t never contains news from day t. Same convention as
   the returns window in the backtest, and the same idea as Capponi et al.
   holding an annual supply-chain graph fixed across the months that follow.

2. SECTOR_PEER is 35% of all edges within the top-100 universe and is
   essentially a sector-membership relation. Since the research question is
   whether news adds information beyond sector labels, it is excluded from
   the news graph by default and available separately as the sector
   benchmark.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from data import PROCESSED

KG_ROOT = Path.home() / "Documents" / "financial-news-kg"
GRAPHS = KG_ROOT / "data" / "output" / "graphs"

# Relations that encode sector membership or analyst coverage rather than a
# business relationship between the two firms. Kept out of the news graph;
# SECTOR_PEER is exposed separately as the placebo benchmark.
SECTOR_TYPES = ["SECTOR_PEER"]
NON_BUSINESS_TYPES = ["ANALYST_COVERS", "EARNINGS_COMPARED_TO", "REGULATES"]

# The extractor emitted a small tail of malformed type strings; fold them in.
TYPE_FIXES = {
    "ANALyst_COVERS": "ANALYST_COVERS",
    "ANALYTIC_COVERS": "ANALYST_COVERS",
    "ANALYT_COVERS": "ANALYST_COVERS",
    "ANALYSIS_COVERS": "ANALYST_COVERS",
    "ANALYZES": "ANALYST_COVERS",
    "ANALYSIS": "ANALYST_COVERS",
}


@lru_cache(maxsize=1)
def global_tickers() -> list[str]:
    return json.load(open(GRAPHS / "tickers.json"))


def snapshot_dates(window: int = 60) -> pd.DatetimeIndex:
    """Dates for which a graph snapshot exists, ascending."""
    files = sorted((GRAPHS / f"w{window}").glob("unified_*.npz"))
    return pd.DatetimeIndex(
        [pd.Timestamp(f.stem.split("_")[1]) for f in files]).sort_values()


def universe_tickers(name: str = "top100") -> list[str]:
    return pd.read_csv(PROCESSED / f"universe_{name}.csv")["ticker"].tolist()


def _coo_to_dense(row, col, data, keep: dict[int, int], n: int) -> np.ndarray:
    """Sparse triplets on the global index -> dense matrix on the universe."""
    A = np.zeros((n, n), dtype=np.float32)
    for r, c, v in zip(row, col, data):
        i, j = keep.get(int(r)), keep.get(int(c))
        if i is not None and j is not None and i != j:
            A[i, j] = v
            A[j, i] = v          # schema says symmetric; enforce it
    return A


def load_snapshot(date, window: int = 60, universe: str = "top100",
                  types: list[str] | None = None,
                  exclude: list[str] | None = None) -> np.ndarray:
    """Adjacency matrix for one snapshot date, restricted to the universe.

    types    restrict to these relation types (None = all)
    exclude  drop these relation types (defaults to sector + non-business)
    """
    tickers = universe_tickers(universe)
    gidx = {t: i for i, t in enumerate(global_tickers())}
    keep = {gidx[t]: i for i, t in enumerate(tickers) if t in gidx}
    n = len(tickers)
    date = pd.Timestamp(date)

    if exclude is None and types is None:
        exclude = SECTOR_TYPES + NON_BUSINESS_TYPES

    stem = "unified" if (types is None and not exclude) else "multiplex"
    path = GRAPHS / f"w{window}" / f"{stem}_{date:%Y%m%d}.npz"
    if not path.exists():
        return np.zeros((n, n), dtype=np.float32)

    f = np.load(path, allow_pickle=True)
    if stem == "unified":
        return _coo_to_dense(f["row"], f["col"], f["data"], keep, n)

    A = np.zeros((n, n), dtype=np.float32)
    layers = {k.rsplit("__", 1)[0] for k in f.files if "__" in k}
    for layer in layers:
        canon = TYPE_FIXES.get(layer, layer)
        if types is not None and canon not in types:
            continue
        if exclude and canon in exclude:
            continue
        A += _coo_to_dense(f[f"{layer}__row"], f[f"{layer}__col"],
                           f[f"{layer}__data"], keep, n)
    return A


def graph_for_days(dates, window: int = 60, universe: str = "top100",
                   max_age_days: int = 183, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Graph aligned to a daily index, forward-filled from snapshots.

    Returns (stack, which) where stack is (n_snapshots, N, N) holding each
    distinct snapshot used, and which[t] indexes the snapshot in force on
    trading day dates[t]. Storing it this way avoids materialising one dense
    matrix per trading day.

    A snapshot is only used on days strictly after its date, so the graph
    predicting day t contains no news from day t itself.
    """
    dates = pd.DatetimeIndex(dates)
    snaps = snapshot_dates(window)
    pos = np.searchsorted(snaps.values, dates.values, side="left") - 1
    # a snapshot only counts if it's recent. Without this, the last 2014
    # snapshot wld be applied to every day of 2015-2016
    for i, p in enumerate(pos):
        if p>= 0 and (dates[i] - snaps[p]).days > max_age_days:
            pos[i] = -1
    used = sorted({p for p in pos if p >= 0})
    stack = np.stack([load_snapshot(snaps[p], window, universe, **kw)
                      for p in used]) if used else np.zeros((0, 0, 0))
    remap = {p: i for i, p in enumerate(used)}
    which = np.array([remap.get(p, -1) for p in pos])
    return stack, which


def neighbour_mask(A: np.ndarray, include_self: bool = True) -> np.ndarray:
    """Boolean (N, N): may firm j be used to predict firm i?

    This is the hard graph restriction. Row i is True for firm i's graph
    neighbours; the diagonal is kept so a firm can always use its own lag,
    which makes the model nest AR rather than being able to underperform it.
    """
    M = A > 0
    if include_self:
        np.fill_diagonal(M, True)
    return M


def penalty_weights(A: np.ndarray, linked: float = 0.1, unlinked: float = 1.0,
                    include_self: bool = True) -> np.ndarray:
    """Per-coefficient penalty multipliers for the soft graph constraint.

    Feeds SparseVAR(penalty_weights=...). Linked pairs get a small weight, so
    their coefficients are penalised lightly and survive shrinkage; unlinked
    pairs get the full penalty. Unlike the hard mask this lets the data
    override the graph when the evidence is strong enough.
    """
    W = np.where(A > 0, linked, unlinked).astype(float)
    if include_self:
        np.fill_diagonal(W, linked)
    return W


def summary(window: int = 60, universe: str = "top100") -> pd.DataFrame:
    """Per-snapshot edge counts and density"""
    tickers = universe_tickers(universe)
    n = len(tickers)
    rows = []
    for d in snapshot_dates(window):
        A = load_snapshot(d, window, universe)
        deg = (A > 0).sum(axis=1)
        rows.append({"date": d, "edges": int((A > 0).sum() // 2),
                     "mean_degree": deg.mean(),
                     "isolated": int((deg == 0).sum()),
                     "density": (A > 0).sum() / (n * (n - 1))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    s = summary()
    print(f"snapshots: {len(s)}   {s['date'].min().date()} .. {s['date'].max().date()}")
    print(s[["edges", "mean_degree", "isolated", "density"]].describe().round(3).to_string())
