#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 14:04:41 2026

@author: anasantana

Walk fwd backtest and evaluation
Loop is same one from var_pipeline.py and it's generalized so that any model
in models.py can be used, plus some statistical metrics 

For each prediction on day t: 
    take LOOKBACK days strictly before t (training window)
    fit model on that window only
    predict t
    step forward
    
    
"""

import time
import numpy as np
import pandas as pd
from scipy import stats
from data import ROOT, load_returns

LOOKBACK = 4*252
WINSOR = 0.01
REFIT_EVERY = 20 #refitting everyday is expensive, so only do every month.
                 #SparseVAR costs ~4.4s per refit, so at 5 the full run takes
                 #~2.5h and at 20 it takes ~40min. Use 5 for the final results.
MKT = "SPY"

#data
def excess_returns(universe: str = "top100", hedge: bool= True):
    """ add l8r
    """
    panel = load_returns(universe)
    rets= np.log1p(panel)
    if not hedge:
        return rets
    spy = np.log1p(load_returns(None)[MKT])
    return rets.sub(spy.reindex(rets.index), axis=0)

def winsorize_window(W, q=WINSOR):
    lo = np.quantile(W, q, axis=0)
    hi = np.quantile(W, 1-q, axis=0)
    return np.clip(W, lo, hi)


#backtesting

def run_backtest(make_model, returns, lookback = LOOKBACK, refit_every=REFIT_EVERY,
                 graph_stack=None, graph_which=None, start = None, verbose=True):
    """add l8r"""
    X = returns.values
    dates=returns.index
    T, n=X.shape
    
    t0 = lookback if start is None else max(
        lookback, dates.searchsorted(pd.Timestamp(start)))
    preds=np.full((T,n),np.nan)
    model,clock = None, time.perf_counter()
    
    for t in range(t0, T):
        if model is None or (t-t0) % refit_every ==0:
            W = winsorize_window(X[t - lookback:t])
            model = make_model()
            if graph_stack is not None:
                k = graph_which[t]
                A = graph_stack[k] if k>=0 else np.zeros((n,n))
                model.fit(W,A)
            else:
                model.fit(W)
        preds[t] = model.predict(X[t-1])
        if verbose and (t-t0)%500 ==0:
            print(f"    day {t - t0}/{T - t0}  ({time.perf_counter() - clock:.0f}s)")
            
    return pd.DataFrame(preds, index= dates,
                        columns = returns.columns).dropna(how="all")

#stat metrics
def statistical_metrics(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """ add l8r"""
    p = pred.values
    y = truth.loc[pred.index, pred.columns].values
    ok = ~np.isnan(p) & ~np.isnan(y)
    p, y = p[ok], y[ok]
    return {
        "oos_r2": 1 - np.sum((y - p) ** 2) / np.sum(y ** 2),
        "mse": np.mean((y - p) ** 2),
        "mae": np.mean(np.abs(y - p)),
        "dir_acc": np.mean(np.sign(p[p != 0]) == np.sign(y[p != 0])),
    }

#economic metrics 

def sharpe_test(v):
    """.
    """
    v = np.asarray(v)
    sr = v.mean() / v.std(ddof=1)
    T = len(v)
    g3 = stats.skew(v)
    g4 = stats.kurtosis(v, fisher=False)
    p1 = stats.norm.cdf(
        sr / np.sqrt((1 - g3 * sr + (g4 - 1) * sr ** 2 / 4) / (T - 1)))
    return sr, min(p1, 1 - p1) * 2


def economic_metrics(pred: pd.DataFrame, truth: pd.DataFrame,
                     top_frac: float = 1.0, periods_per_year: float = 252) -> dict:
    """PnL markout: positions are sign(forecast), equal $1 bets."""
    s = np.nan_to_num(pred.values.copy(), nan=0.0)
    f = np.nan_to_num(truth.loc[pred.index, pred.columns].values, nan=0.0)

    if top_frac < 1.0:                       # trade only the strongest signals
        k = max(1, int(np.ceil(top_frac * s.shape[1])))
        thresh = np.sort(np.abs(s), axis=1)[:, -k][:, None]
        s = np.where(np.abs(s) >= thresh, s, 0.0)

    pos = np.sign(s)
    pnl = (pos * f).sum(axis=1)
    ntrades = (pos != 0).sum(axis=1)
    traded = pos != 0
    _, pval = sharpe_test(pnl)
    return {
        "ann_sharpe": pnl.mean() / pnl.std(ddof=1) * np.sqrt(periods_per_year),
        "sharpe_p": pval,
        "ppt_bps": pnl.sum() / max(ntrades.sum(), 1) * 1e4,
        "hit_ratio": (np.sign(f)[traded] == pos[traded]).mean(),
        "pnl": pd.Series(pnl, index=pred.index),
    }


def diebold_mariano(pred_a, pred_b, truth):
    """Test whether model A's squared errors are smaller than model B's.
    Returns (DM statistic, p-value). Negative statistic favours A.
    """
    idx = pred_a.index.intersection(pred_b.index)
    cols = pred_a.columns
    y = truth.loc[idx, cols].values
    ea = (pred_a.loc[idx, cols].values - y) ** 2
    eb = (pred_b.loc[idx, cols].values - y) ** 2
    d = np.nanmean(ea - eb, axis=1)          # loss differential per day
    d = d[~np.isnan(d)]
    dm = d.mean() / np.sqrt(np.var(d, ddof=1) / len(d))
    return dm, 2 * (1 - stats.norm.cdf(abs(dm)))


#runner

def evaluate_all(models: dict, returns, **kw):
    """Run every model and collect both metric families in one table."""
    preds, rows = {}, []
    for name, make in models.items():
        print(f"  {name} ...")
        p = run_backtest(make, returns, **kw)
        preds[name] = p
        rows.append({"model": name, **statistical_metrics(p, returns),
                     **{k: v for k, v in economic_metrics(p, returns).items()
                        if k != "pnl"}})
    return pd.DataFrame(rows), preds


def main():
    from models import BASELINES

    returns = excess_returns("top100")
    print(f"panel: {returns.shape[0]} days x {returns.shape[1]} stocks "
          f"({returns.index[0].date()} .. {returns.index[-1].date()})")
    print(f"lookback {LOOKBACK}d, refit every {REFIT_EVERY}d\n")

    table, preds = evaluate_all(BASELINES, returns)

    print("\n" + table.to_string(index=False,
                                 float_format=lambda v: f"{v:.4f}"))
    out = ROOT / "reports" / "baseline_results.csv"
    table.to_csv(out, index=False)

    print("\nDiebold-Mariano vs AR "
          "(negative favours the model, p < 0.05 significant):")
    for name, p in preds.items():
        if name == "AR":
            continue
        dm, pv = diebold_mariano(p, preds["AR"], returns)
        print(f"  {name:20s} DM = {dm:7.3f}   p = {pv:.4f}")

    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()