#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 08:07:45 2026

@author: anasantana



Here we run every model on the SAME window (2007-2020, where the knowledge graph
has coverage) so the numbers are directly comparable. Then we can see if the 
network built from news beats the alternatives 

The sector graph is the control. It is built from SECTOR_PEER edges,
which have almost identical density to the news graph (mean degree 1.46 vs
1.54), so a performance difference cannot be explained by one graph simply
having more connections.

CorrGraphVAR is a second control bc it builds a correlation graph matched edge
for edge to the news graph on the same date, isolating "which pairs" from
"how many pairs".

"""

import numpy as np
import pandas as pd

from backtest import (diebold_mariano, economic_metrics, excess_returns,
                      run_backtest, statistical_metrics)
from data import ROOT
from graph import SECTOR_TYPES, graph_for_days
from models import AR, NIRVAR, GraphVAR, SparseVAR

START, END = "2007-01-01", "2020-12-31"   # the graph's usable coverage
WINDOW = 252                              # 252-day graph, mean degree ~1.3
HORIZON = 1                               # 1 = daily, 5 = weekly, 21 = monthly
TARGET = "volume"
REFIT_EVERY = 20


class CorrGraphVAR(GraphVAR):
    """Receives the news graph to count edges, then builds its own
    network from the strongest correlations in the training window, keeping
    exactly the same number of pairs. If the news graph wins against this,
    the advantage is about WHICH firms are linked instead of how many.
    """

    @property
    def name(self):
        return f"CorrGraphVAR-{self.mode}"

    def fit(self, W, A=None):
        n = W.shape[1]
        k = int((A > 0).sum() // 2) if A is not None else 0
        A_corr = np.zeros((n, n))
        if k > 0:
            C = np.abs(np.nan_to_num(np.corrcoef(W.T), nan=0.0))
            np.fill_diagonal(C, 0.0)
            iu = np.triu_indices(n, 1)
            strongest = np.argsort(C[iu])[-k:]        # top k pairs by |corr|
            rows, cols = iu[0][strongest], iu[1][strongest]
            A_corr[rows, cols] = A_corr[cols, rows] = 1.0
        return super().fit(W, A_corr)


def aggregate(returns, horizon):
    """Make daily returns into weekly or monthly ones by adding them up, so 
    can check whether firms affect each other over weeks instead of overnight.
    """
    if horizon == 1:
        return returns
    blocks = np.arange(len(returns)) // horizon
    out = returns.groupby(blocks).sum()
    out.index = returns.index[::horizon][:len(out)]
    return out


def build_graphs(dates):
    """News and sector adjacency stacks aligned to the trading days."""
    news_stack, which = graph_for_days(dates, window=WINDOW)
    sector_stack, _ = graph_for_days(dates, window=WINDOW,
                                     types=SECTOR_TYPES, exclude=[])
    return news_stack, sector_stack, which


def main():
    if TARGET == "returns":
        returns = excess_returns("top100")
    else:
        from targets import TARGETS
        returns = TARGETS[TARGET]()
    lookback = max(60, int(4 * 252 / HORIZON))     
    first = returns.index.searchsorted(pd.Timestamp(START))
    returns = returns.iloc[max(0, first - 4 * 252 - 10):
                           returns.index.searchsorted(pd.Timestamp(END)) + 1]
    returns = aggregate(returns, HORIZON)

    news_stack, sector_stack, which = build_graphs(returns.index)
    print(f"window   : {returns.index[0].date()} .. {returns.index[-1].date()}"
          f"  ({len(returns)} periods, horizon {HORIZON}d)")
    print(f"graphs   : {len(news_stack)} news snapshots, "
          f"{len(sector_stack)} sector snapshots")
    if len(news_stack):
        deg = (news_stack > 0).sum(axis=2).mean()
        print(f"news graph mean degree {deg:.2f}, "
              f"isolated firms {(news_stack > 0).sum(axis=2).mean(axis=0).__eq__(0).sum()}")

    runs = [
        ("AR",              AR,                                    None),
        ("SparseVAR-LASSO", lambda: SparseVAR(alpha=3.16e-5),      None),
        ("NIRVAR",          NIRVAR,                                None),
        ("GraphVAR-news-hard",   lambda: GraphVAR("hard"),         news_stack),
        ("GraphVAR-news-soft",   lambda: GraphVAR("soft", alpha=1e-4), news_stack),
        ("GraphVAR-sector-hard", lambda: GraphVAR("hard"),         sector_stack),
        ("GraphVAR-sector-soft", lambda: GraphVAR("soft", alpha=1e-4), sector_stack),
        ("CorrGraphVAR-hard",    lambda: CorrGraphVAR("hard"),     news_stack),
    ]

    rows, preds = [], {}
    for label, make, stack in runs:
        print(f"  {label} ...")
        p = run_backtest(make, returns, lookback=lookback,
                         refit_every=REFIT_EVERY, graph_stack=stack,
                         graph_which=which if stack is not None else None,
                         start=START, verbose=False)
        preds[label] = p
        rows.append({"model": label, **statistical_metrics(p, returns),
                     **{k: v for k, v in economic_metrics(p, returns, periods_per_year=252 / HORIZON).items()
                        if k != "pnl"}})

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False,
                                 float_format=lambda v: f"{v:.4f}"))

    out = ROOT / "reports" / f"graph_comparison_{TARGET}_h{HORIZON}.csv"
    table.to_csv(out, index=False)


    print("\nDiebold-Mariano tests (negative favours the news graph):")
    dm_rows = []
    for a, b in [("GraphVAR-news-hard", "AR"),
                 ("GraphVAR-news-soft", "AR"),
                 ("GraphVAR-news-soft", "NIRVAR"),
                 ("GraphVAR-news-soft", "GraphVAR-sector-soft"),
                 ("GraphVAR-news-hard", "GraphVAR-sector-hard"),
                 ("GraphVAR-news-hard", "CorrGraphVAR-hard")]:
        if a in preds and b in preds:
            dm, pv = diebold_mariano(preds[a], preds[b], returns)
            verdict = ("model_a better" if dm < 0 and pv < 0.05 else
                       "model_b better" if dm > 0 and pv < 0.05 else
                       "no significant difference")
            dm_rows.append({"model_a": a, "model_b": b, "DM": dm,
                            "p_value": pv, "verdict": verdict})
            print(f"  {a:22s} vs {b:22s} DM = {dm:7.3f}  p = {pv:.4f}  {verdict}")

    dm_out = ROOT / "reports" / f"graph_comparison__{TARGET}_h{HORIZON}_dm_tests.csv"
    pd.DataFrame(dm_rows).to_csv(dm_out, index=False)

    print(f"\nsaved -> {out}")
    print(f"saved -> {dm_out}")


if __name__ == "__main__":
    main()