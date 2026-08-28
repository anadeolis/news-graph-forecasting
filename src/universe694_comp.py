#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 20:23:43 2026

@author: anasantana


Instead of doing the graph compariosn over the top 100, we should also do on 
the full pabel of 694 firms.

Same experiment as graph_comp.py, run on every stock in the panel. Written
in a seperate script so that the main results and robustness results can't
overwrite each other.

Only the fast models  are included. Also note that Dylan's graphs only reach
542 of the 694, so a large share of firms are isolated here than in the top 100
run. So expect the graph models to be closer to AR.

Out: reports/universe694_comparison_{HORIZON}.csv (+ _dm_tests)

"""

import numpy as np
import pandas as pd

from backtest import (diebold_mariano, economic_metrics, excess_returns,
                      run_backtest, statistical_metrics)
from data import PROCESSED, ROOT
from graph import SECTOR_TYPES, graph_for_days
from graph_comp import CorrGraphVAR
from models import AR, NIRVAR, VAR, GraphVAR

START, END = "2007-01-01", "2020-12-31"
WINDOW = 252
HORIZON = 1
REFIT_EVERY =20
UNIVERSE = "all694"

def ensure_universe_file():
    """Create universe_all694.csv if missing"""
    path = PROCESSED/ f"universe_{UNIVERSE}.csv"
    if not path.exists():
        panel = pd.read_parquet(PROCESSED / "returns_daily.parquet")
        tickers = [c for c in panel.columns if c!= "SPY"]
        pd.Series(tickers, name = "ticker").to_csv(path, index = False)
        print(f"created {path.name} with {len(tickers)} tickers")
        
def main():
    ensure_universe_file()
    returns = excess_returns(UNIVERSE)

    first = returns.index.searchsorted(pd.Timestamp(START))
    returns = returns.iloc[max(0, first - 4 * 252 - 10):
                           returns.index.searchsorted(pd.Timestamp(END)) + 1]

    news_stack, which = graph_for_days(returns.index, window=WINDOW,
                                       universe=UNIVERSE)
    sector_stack, _ = graph_for_days(returns.index, window=WINDOW,
                                     universe=UNIVERSE,
                                     types=SECTOR_TYPES, exclude=[])
    covered = int(((news_stack > 0).sum(axis=(0, 2)) > 0).sum())
    print(f"universe {returns.shape[1]} stocks, "
          f"{covered} ever appear in the news graph")

    runs = [
        ("AR",                   AR,                            None),
        ("VAR",                  VAR,                           None),
        ("NIRVAR",               NIRVAR,                        None),
        ("GraphVAR-news-hard",   lambda: GraphVAR("hard"),      news_stack),
        ("GraphVAR-sector-hard", lambda: GraphVAR("hard"),      sector_stack),
        ("CorrGraphVAR-hard",    lambda: CorrGraphVAR("hard"),  news_stack),
    ]

    rows, preds = [], {}
    for label, make, stack in runs:
        print(f"  {label} ...")
        p = run_backtest(make, returns, lookback=4 * 252,
                         refit_every=REFIT_EVERY, graph_stack=stack,
                         graph_which=which if stack is not None else None,
                         start=START, verbose=False)
        preds[label] = p
        rows.append({"model": label, **statistical_metrics(p, returns),
                     **{k: v for k, v in economic_metrics(p, returns).items()
                        if k != "pnl"}})

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False,
                                 float_format=lambda v: f"{v:.4f}"))
    out = ROOT / "reports" / f"universe694_comparison_h{HORIZON}.csv"
    table.to_csv(out, index=False)

    print("\nDiebold-Mariano tests:")
    dm_rows = []
    for a, b in [("GraphVAR-news-hard", "AR"),
                 ("GraphVAR-news-hard", "GraphVAR-sector-hard"),
                 ("GraphVAR-news-hard", "CorrGraphVAR-hard"),
                 ("VAR", "AR")]:
        dm, pv = diebold_mariano(preds[a], preds[b], returns)
        verdict = ("model_a better" if dm < 0 and pv < 0.05 else
                   "model_b better" if dm > 0 and pv < 0.05 else
                   "no significant difference")
        dm_rows.append({"model_a": a, "model_b": b, "DM": dm,
                        "p_value": pv, "verdict": verdict})
        print(f"  {a:22s} vs {b:22s} DM = {dm:7.3f}  p = {pv:.4f}  {verdict}")

    dm_out = ROOT / "reports" / f"universe694_comparison_h{HORIZON}_dm_tests.csv"
    pd.DataFrame(dm_rows).to_csv(dm_out, index=False)
    print(f"\nsaved -> {out}\nsaved -> {dm_out}")


if __name__ == "__main__":
    main()

