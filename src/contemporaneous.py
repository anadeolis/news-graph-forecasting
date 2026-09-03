#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:29:22 2026

@author: anasantana


"""

import numpy as np
import pandas as pd
from scipy import stats

from data import ROOT, load_returns, load_sector_map
from graph import load_snapshot, snapshot_dates, universe_tickers

WINDOW = 252
UNIVERSE = "top100"
START, END = "2007-01-01", "2020-12-31"

LAYERS = ["COMPETES_WITH", "PARTNERS_WITH", "SUPPLIES_TO", "INVESTS_IN",
          "ACQUIRES", "ADVISES", "COUNTERPARTY_OF", "LENDER_TO",
          "EXECUTIVE_MOVES", "BIDS_FOR", "CONTAGION_TO", "SPINS_OFF"]


def sector_pair_matrix(tickers) -> np.ndarray:
    """Boolean (N, N): do these two firms share a GICS sector?"""
    sec = load_sector_map().reindex(tickers).fillna("Unknown").values
    return sec[:, None] == sec[None, :]


def snapshot_stats(A, C, same_sector, iu):
    """Mean |corr| gap, linked minus unlinked, stratified by sector."""
    linked = A[iu] > 0
    if linked.sum() < 5:
        return None
    corr, same = C[iu], same_sector[iu]

    out = {"n_linked": int(linked.sum()),
           "linked_corr": corr[linked].mean(),
           "unlinked_corr": corr[~linked].mean()}
    out["overall"] = out["linked_corr"] - out["unlinked_corr"]
    for name, mask in [("same_sector", same), ("cross_sector", ~same)]:
        l, u = corr[mask & linked], corr[mask & ~linked]
        out[name] = (l.mean() - u.mean()) if len(l) >= 3 and len(u) >= 3 else np.nan
    return out


def main():
    tickers = universe_tickers(UNIVERSE)
    rets = load_returns(UNIVERSE)
    same_sector = sector_pair_matrix(tickers)
    iu = np.triu_indices(len(tickers), 1)

    snaps = snapshot_dates(WINDOW)
    snaps = snaps[(snaps >= pd.Timestamp(START)) & (snaps <= pd.Timestamp(END))]
    snaps = snaps[::max(1, WINDOW // 20)]        # non-overlapping windows
    print(f"{len(snaps)} non-overlapping snapshots, "
          f"{snaps[0].date()} .. {snaps[-1].date()}\n")

    rows = []
    for layer in ["ALL_NEWS"] + LAYERS:
        per_snap = []
        for d in snaps:
            end = rets.index.searchsorted(d)
            W = rets.iloc[max(0, end - WINDOW):end]
            if len(W) < 60:
                continue
            kw = {} if layer == "ALL_NEWS" else {"types": [layer], "exclude": []}
            A = load_snapshot(d, WINDOW, UNIVERSE, **kw)
            C = np.abs(np.nan_to_num(np.corrcoef(W.values.T), nan=0.0))
            s = snapshot_stats(A, C, same_sector, iu)
            if s:
                per_snap.append(s)

        if len(per_snap) < 5:
            print(f"  {layer:16s} too few usable snapshots, skipped")
            continue

        df = pd.DataFrame(per_snap)
        row = {"layer": layer, "n_snapshots": len(df),
               "mean_linked_pairs": df["n_linked"].mean(),
               "corr_linked": df["linked_corr"].mean(),
               "corr_unlinked": df["unlinked_corr"].mean()}
        for col in ["overall", "same_sector", "cross_sector"]:
            v = df[col].dropna().values
            t, p = stats.ttest_1samp(v, 0.0) if len(v) >= 5 else (np.nan, np.nan)
            row[f"d_{col}"], row[f"p_{col}"] = (v.mean() if len(v) else np.nan), p
        rows.append(row)
        print(f"  {layer:16s} overall d={row['d_overall']:+.4f} "
              f"p={row['p_overall']:.4f}   cross-sector d={row['d_cross_sector']:+.4f} "
              f"p={row['p_cross_sector']:.4f}")

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out = ROOT / "reports" / "contemporaneous_correlation.csv"
    table.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
