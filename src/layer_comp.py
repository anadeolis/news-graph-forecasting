"""Run the models on each relation layer separately.


"""

import numpy as np
import pandas as pd

from backtest import (diebold_mariano, excess_returns, run_backtest,
                      statistical_metrics)
from data import ROOT
from graph import graph_for_days
from graph_comp import aggregate
from models import AR, GraphVAR

START, END = "2007-01-01", "2020-12-31"
WINDOW = 252
HORIZON = 1
TARGET = "returns"
REFIT_EVERY = 20
UNIVERSE = "top100"

LAYERS = ["COMPETES_WITH", "PARTNERS_WITH", "SUPPLIES_TO", "INVESTS_IN",
          "ACQUIRES", "ADVISES", "COUNTERPARTY_OF", "LENDER_TO",
          "EXECUTIVE_MOVES", "BIDS_FOR", "CONTAGION_TO", "SPINS_OFF"]


def panel():
    if TARGET == "returns":
        r = excess_returns(UNIVERSE)
    else:
        from targets import TARGETS
        r = TARGETS[TARGET]()
    first = r.index.searchsorted(pd.Timestamp(START))
    r = r.iloc[max(0, first - 4 * 252 - 10):
               r.index.searchsorted(pd.Timestamp(END)) + 1]
    return aggregate(r, HORIZON)


def main():
    returns = panel()
    lookback = max(60, int(4 * 252 / HORIZON))
    print(f"window: {returns.index[0].date()} .. {returns.index[-1].date()}"
          f"  ({len(returns)} periods, {returns.shape[1]} firms)")

    print("\nAR (shared baseline) ...")
    ar = run_backtest(AR, returns, lookback=lookback,
                      refit_every=REFIT_EVERY, start=START, verbose=False)

    rows = []
    for layer in LAYERS:
        stack, which = graph_for_days(returns.index, window=WINDOW,
                                      universe=UNIVERSE, types=[layer],
                                      exclude=[])
        if stack.size == 0:
            print(f"  {layer}: no snapshots, skipped")
            continue

        covered = (stack > 0).any(axis=(0, 1))     # firms with any edge, ever
        cols = returns.columns[covered]
        if len(cols) < 3:
            print(f"  {layer}: only {len(cols)} firms connected, skipped")
            continue

        print(f"  {layer} ... {len(cols)} connected firms")
        g = run_backtest(lambda: GraphVAR("hard"), returns, lookback=lookback,
                         refit_every=REFIT_EVERY, graph_stack=stack,
                         graph_which=which, start=START, verbose=False)

        truth = returns[cols]
        gm = statistical_metrics(g[cols], truth)
        am = statistical_metrics(ar.loc[g.index, cols], truth)
        dm, pv = diebold_mariano(g[cols], ar.loc[g.index, cols], truth)

        rows.append({
            "layer": layer, "n_firms": len(cols),
            "edges_per_day": float(np.mean([(stack[w] > 0).sum() // 2
                                            for w in which if w >= 0])),
            "graph_r2": gm["oos_r2"], "ar_r2": am["oos_r2"],
            "graph_dir_acc": gm["dir_acc"], "ar_dir_acc": am["dir_acc"],
            "DM": dm, "p_value": pv,
            "verdict": ("graph better" if dm < 0 and pv < 0.05 else
                        "AR better" if dm > 0 and pv < 0.05 else
                        "no significant difference"),
        })

    table = pd.DataFrame(rows)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out = ROOT / "reports" / f"layer_comparison_{TARGET}_h{HORIZON}.csv"
    table.to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()