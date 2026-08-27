"""Stage 4: choose the sparse-VAR penalty by time-series cross-validation.

The project brief asks for the regularised models to be "tuned by
cross-validation". For time series that means TimeSeriesSplit, never
KFold: folds must be contiguous and the validation block must always come
after the training block. Shuffled CV would train on future returns to
predict past ones and pick a far too small penalty.

Concretely, a window of returns is split into k expanding folds:

    fold 1   train [-----]  validate [--]
    fold 2   train [--------]  validate [--]
    fold 3   train [-----------]  validate [--]

For each candidate alpha the model is fit on the training part and scored on
the validation part, and the alpha with the best average out-of-sample R2
wins. Because the panel is 21 years long and the model is refit on a rolling
window anyway, tuning is done on a handful of windows spread across the
sample rather than at every refit, and the median winner is used.

Run:  python src/tune.py
Out:  reports/tuning_alpha.csv
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from data import ROOT, load_returns
from models import SparseVAR

LOOKBACK = 4 * 252
ALPHAS = np.logspace(-6, -2, 9)
L1_RATIOS = [1.0, 0.5]           # LASSO, elastic net
N_WINDOWS = 6                    # tuning windows spread across the sample
N_SPLITS = 4


def oos_r2(pred: np.ndarray, truth: np.ndarray) -> float:
    """Out-of-sample R2 against a zero forecast.

    Zero, not the historical mean, is the benchmark — the convention in
    Gu, Kelly & Xiu (2020) and Capponi et al. The sample mean of daily
    returns is estimated so noisily that it is usually a worse forecast
    than simply predicting no change.
    """
    return 1.0 - np.sum((truth - pred) ** 2) / np.sum(truth ** 2)


def directional_accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    """Share of forecasts whose sign matches the realised return.

    This is the criterion the portfolio actually depends on: positions are
    sign(forecast) with equal bets, so forecast magnitude is discarded.
    Selecting alpha by squared error instead picks a model with almost no
    surviving coefficients, because at daily frequency predicting zero is
    near-optimal in MSE terms while being useless for taking positions.
    """
    mask = pred != 0
    if not mask.any():
        return 0.5
    return float((np.sign(pred[mask]) == np.sign(truth[mask])).mean())


SCORERS = {"r2": oos_r2, "dir_acc": directional_accuracy}


def score_alpha(W: np.ndarray, alpha: float, l1_ratio: float,
                n_splits: int = N_SPLITS, criterion: str = "dir_acc") -> float:
    """Mean validation score for one alpha on one window, via TimeSeriesSplit."""
    scorer = SCORERS[criterion]
    Z, Y = W[:-1], W[1:]
    scores = []
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(Z):
        model = SparseVAR(alpha=alpha, l1_ratio=l1_ratio)
        # fit() expects a contiguous window and forms its own lag pairs, so
        # hand it the raw slice covering the training rows plus one extra
        model.fit(W[tr[0]:tr[-1] + 2])
        pred = np.array([model.predict(Z[i]) for i in va])
        scores.append(scorer(pred, Y[va]))
    return float(np.mean(scores))


def tune(universe: str = "top100") -> pd.DataFrame:
    returns = load_returns(universe)
    X = np.log1p(returns).values
    T = len(X)
    starts = np.linspace(LOOKBACK, T - 1, N_WINDOWS, dtype=int)

    rows = []
    for l1 in L1_RATIOS:
        for t in starts:
            W = X[t - LOOKBACK:t]
            for a in ALPHAS:
                row = {"l1_ratio": l1, "window_end": returns.index[t - 1],
                       "alpha": a}
                for crit in SCORERS:
                    row[crit] = score_alpha(W, a, l1, criterion=crit)
                row["nonzero_per_stock"] = (
                    SparseVAR(alpha=a, l1_ratio=l1).fit(W).B_ != 0).sum() / X.shape[1]
                rows.append(row)
            print(f"  l1={l1} window ending {returns.index[t-1].date()} done")
    return pd.DataFrame(rows)


def main():
    scores = tune()
    out = ROOT / "reports" / "tuning_alpha.csv"
    scores.to_csv(out, index=False)

    for crit in SCORERS:
        print(f"\nmean validation {crit} by alpha:")
        pivot = scores.pivot_table(index="alpha", columns="l1_ratio",
                                   values=crit, aggfunc="mean")
        pivot["coefs/stock"] = scores.groupby("alpha")["nonzero_per_stock"].mean()
        print(pivot.to_string(float_format=lambda v: f"{v:.5f}"))

    # Take the peak of the averaged curve rather than the median of
    # per-window winners: individual windows are noisy, and the median can
    # land between grid points where no model was actually fitted.
    print("\nchosen by directional accuracy (peak of mean curve):")
    for l1, grp in scores.groupby("l1_ratio"):
        curve = grp.groupby("alpha")["dir_acc"].mean()
        alpha = float(curve.idxmax())
        kind = "LASSO" if l1 == 1.0 else f"elastic net (l1_ratio={l1})"
        n = grp[grp.alpha == alpha]["nonzero_per_stock"].mean()
        print(f"  {kind:32s} alpha = {alpha:.2e}   "
              f"({n:.1f} coefs/stock, dir_acc {curve.max():.4f})")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
