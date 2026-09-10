#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 17:34:06 2026

@author: anasantana
"""

import numpy as np
import pandas as pd
import networkx as nx

from data import ROOT, load_returns, load_sector_map
from graph import (SECTOR_TYPES, load_snapshot, snapshot_dates,
                   universe_tickers)

WINDOW = 252
UNIVERSE = "top100"
START, END = "2007-01-01", "2020-12-31"
DENSE_END = "2014-12-31"
MAX_AGE_DAYS = 183

def sector_pairs(tickers) -> np.ndarray:
    sec = load_sector_map().reindex(tickers).fillna("Unknown").values
    return sec[:, None] == sec[None, :]

def edge_set(B, iu) -> set:
    """the pairs present in a boolean adjacency, as a set of (i,j)"""
    return set(zip(*[x[B[iu]] for x in iu]))

def describe(B, iu, same_sector) -> dict:
    """Structural metrics for one boolean adjacency matrix."""
    n = B.shape[0]
    deg = B.sum(1)
    G = nx.from_numpy_array(B.astype(int))
    G.remove_nodes_from([v for v in G.nodes if G.degree(v) == 0])
    return {
        "edges": int(B.sum() // 2),
        "density": B.sum() / (n * (n - 1)),
        "mean_degree": deg.mean(),
        "max_degree": int(deg.max()),
        "isolated": int((deg == 0).sum()),
        "components": nx.number_connected_components(G) if len(G) else 0,
        "largest_comp": len(max(nx.connected_components(G), key=len)) if len(G) else 0,
        "clustering": nx.average_clustering(G) if len(G) else 0.0,
        "pct_within_sector": (100 * same_sector[iu][B[iu]].mean()
                              if B[iu].any() else np.nan),
    }


def snapshot_series(window, start, end, iu, same_sector, **kw) -> pd.DataFrame:
    """Structure of every snapshot in a date range, plus edge persistence."""
    snaps = snapshot_dates(window)
    snaps = snaps[(snaps >= pd.Timestamp(start)) & (snaps <= pd.Timestamp(end))]
    rows, prev = [], None
    for d in snaps:
        B = load_snapshot(d, window, UNIVERSE, **kw) > 0
        e = edge_set(B, iu)
        rows.append({"date": d, **describe(B, iu, same_sector),
                     "edge_persistence": (len(e & prev) / max(len(prev), 1)
                                          if prev is not None else np.nan)})
        prev = e
    return pd.DataFrame(rows)


def corr_graph(d, k, returns, window, iu):
    """Density-matched correlation graph: the k most correlated pairs."""
    n = len(returns.columns)
    A = np.zeros((n, n), dtype=bool)
    if k <= 0:
        return A
    end = returns.index.searchsorted(d)
    W = returns.iloc[max(0, end - window):end]
    C = np.abs(np.nan_to_num(np.corrcoef(W.values.T), nan=0.0))
    np.fill_diagonal(C, 0.0)
    top = np.argsort(C[iu])[-k:]
    A[iu[0][top], iu[1][top]] = A[iu[1][top], iu[0][top]] = True
    return A


def time_audit(dates, window):
    """Every day must use a snapshot dated strictly before it."""
    snaps = snapshot_dates(window)
    pos = np.searchsorted(snaps.values, dates.values, side="left") - 1
    ages, violations = [], 0
    for i, p in enumerate(pos):
        if p < 0:
            continue
        age = (dates[i] - snaps[p]).days
        if age > MAX_AGE_DAYS:          # staleness guard drops these
            continue
        if snaps[p] >= dates[i]:
            violations += 1
        ages.append(age)
    return np.array(ages), violations


def main():
    tickers = universe_tickers(UNIVERSE)
    n = len(tickers)
    iu = np.triu_indices(n, 1)
    same_sector = sector_pairs(tickers)
    returns = load_returns(UNIVERSE)
    out = ROOT / "reports"

    # ----structure of the news graph -------------------------------
    S = snapshot_series(WINDOW, START, END, iu, same_sector)
    cols = ["edges", "density", "mean_degree", "max_degree", "isolated",
            "components", "largest_comp", "clustering", "pct_within_sector",
            "edge_persistence"]
    print("=" * 76)
    print(f"1. NEWS GRAPH STRUCTURE  ({UNIVERSE}, w{WINDOW})")
    print("=" * 76)
    print(f"{len(S)} snapshots, {S.date.min().date()} .. {S.date.max().date()}\n")
    print(S[cols].describe().loc[["mean", "std", "min", "50%", "max"]]
          .round(3).to_string())

    S["year"] = S.date.dt.year
    print("\nby year:")
    print(S.groupby("year")[["edges", "mean_degree", "isolated", "largest_comp",
                             "clustering", "pct_within_sector"]]
          .mean().round(2).to_string())
    S.to_csv(out / "graph_structure.csv", index=False)

    # ----  window length --------------------------------------------
    print("\n" + "=" * 76)
    print(f"2. WINDOW LENGTH  (dense period only, {START[:4]}-{DENSE_END[:4]})")
    print("=" * 76)
    wins = []
    for w in (20, 60, 252):
        T = snapshot_series(w, START, DENSE_END, iu, same_sector)
        wins.append({"window": w, "snapshots": len(T),
                     "edges": T.edges.mean(), "isolated": T.isolated.mean(),
                     "pct_within_sector": T.pct_within_sector.mean(),
                     "edge_persistence": T.edge_persistence.mean()})
    W = pd.DataFrame(wins)
    print(W.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    W.to_csv(out / "graph_windows.csv", index=False)

    # ---- benchmark graphs -----------------------------------------
    print("\n" + "=" * 76)
    print("3. NEWS vs SECTOR vs CORRELATION  (density-matched)")
    print("=" * 76)
    snaps = snapshot_dates(WINDOW)
    snaps = snaps[(snaps >= pd.Timestamp(START)) & (snaps <= pd.Timestamp(END))]
    rows = []
    for d in snaps[::12]:                       # non-overlapping windows
        news = load_snapshot(d, WINDOW, UNIVERSE) > 0
        sector = load_snapshot(d, WINDOW, UNIVERSE,
                               types=SECTOR_TYPES, exclude=[]) > 0
        corr = corr_graph(d, int(news.sum() // 2), returns, WINDOW, iu)
        for name, A in [("news", news), ("sector", sector), ("corr", corr)]:
            rows.append({"graph": name, **describe(A, iu, same_sector),
                         "overlap_with_news":
                             100 * (A & news)[iu].sum() / max(news[iu].sum(), 1)})
    B = pd.DataFrame(rows).groupby("graph").mean()
    print(B.round(2).to_string())
    B.to_csv(out / "graph_benchmarks.csv")

    # ---- time consistency -----------------------------------------
    print("\n" + "=" * 76)
    print("4. TIME-CONSISTENCY AUDIT")
    print("=" * 76)
    days = returns.loc[START:END].index
    ages, violations = time_audit(days, WINDOW)
    print(f"trading days evaluated       : {len(days)}")
    print(f"days with a graph in force   : {len(ages)} "
          f"({100 * len(ages) / len(days):.1f}%)")
    print(f"days with no graph (empty)   : {len(days) - len(ages)}")
    print(f"VIOLATIONS (snapshot >= day) : {violations}")
    print(f"snapshot age, days           : min {ages.min()}, "
          f"median {int(np.median(ages))}, max {ages.max()} "
          f"(cap {MAX_AGE_DAYS})")

    print(f"\nsaved -> {out / 'graph_structure.csv'}")
    print(f"saved -> {out / 'graph_windows.csv'}")
    print(f"saved -> {out / 'graph_benchmarks.csv'}")


if __name__ == "__main__":
    main()