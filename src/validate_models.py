"""Validate every model on synthetic data with a known structure.

1. Lead-lag chain. Firm i+1 follows firm i with a one-day lag, and there is
   no own-lag structure at all. Models that use cross-firm information should
   score well; AR (own past only) and FactorModel (common movements only)
   should score near zero, because pairwise lead-lag is invisible to them.

2. Graph quality. GraphVAR is given the TRUE graph and a deliberately WRONG
   graph of identical density. This separates the hard constraint, which is
   only as good as the graph, from the soft penalty, which can override a
   bad prior.

"""

import numpy as np
import pandas as pd

from data import ROOT
from models import AR, VAR, FactorModel, GraphVAR, NIRVAR, SparseVAR

LOOKBACK = 400
STEP = 5            # evaluate every 5th day to keep the run quick


def lead_lag_panel(T=700, N=15, strength=0.5, n_linked=None, seed=1):
    """Returns where firm i+1 follows firm i, with no own-lag structure.

    n_linked limits the chain to the first n_linked firms, leaving the rest
    as pure noise — used by experiment 2 so a "wrong" graph can point at
    firms that genuinely have no signal.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.01, (T, N))
    end = N if n_linked is None else n_linked
    for t in range(1, T):
        X[t, 1:end] += strength * X[t - 1, 0:end - 1]
    return X


def score(make_model, X, A=None):
    """Walk forward over the synthetic panel and score the forecasts."""
    preds, truth = [], []
    for t in range(LOOKBACK, len(X), STEP):
        model = make_model()
        W = X[t - LOOKBACK:t]                  # strictly before day t
        model.fit(W, A) if A is not None else model.fit(W)
        preds.append(model.predict(X[t - 1]))
        truth.append(X[t])
    P, Y = np.array(preds), np.array(truth)
    return {
        "oos_r2": 1 - ((Y - P) ** 2).sum() / (Y ** 2).sum(),
        "dir_acc": float((np.sign(P) == np.sign(Y)).mean()),
    }


def experiment_1():
    """Can each model see a pairwise lead-lag structure?"""
    X = lead_lag_panel(T=700, N=15, seed=1)
    models = {
        "AR": AR,
        "VAR": VAR,
        "SparseVAR-LASSO": lambda: SparseVAR(alpha=2e-5, l1_ratio=1.0),
        "SparseVAR-EN": lambda: SparseVAR(alpha=2e-5, l1_ratio=0.5),
        "Factor(k=3)": lambda: FactorModel(k=3),
        "NIRVAR": NIRVAR,
    }
    expected = {
        "AR": "~0 (no own-lag structure exists)",
        "VAR": "high (sees cross-firm lags)",
        "SparseVAR-LASSO": "high, above VAR (drops noise coefficients)",
        "SparseVAR-EN": "high, above VAR",
        "Factor(k=3)": "~0 (pairwise links are not a common factor)",
        "NIRVAR": "partial (true structure is a chain, not clusters)",
    }
    return pd.DataFrame([
        {"experiment": "1: lead-lag chain", "model": name,
         **score(make, X), "expected": expected[name]}
        for name, make in models.items()
    ])


def experiment_2():
    """Does GraphVAR depend on the graph being correct?"""
    N, n_linked = 20, 10
    X = lead_lag_panel(T=700, N=N, n_linked=n_linked, seed=3)

    def chain(first, last):
        A = np.zeros((N, N))
        for i in range(first + 1, last):
            A[i, i - 1] = A[i - 1, i] = 1.0
        return A

    A_true = chain(0, n_linked)        # points at the firms that matter
    A_wrong = chain(n_linked + 1, N)   # same density, wrong firms

    cases = [
        ("AR (no graph)", AR, None, "baseline"),
        ("GraphVAR-hard", lambda: GraphVAR("hard"), A_true, "high"),
        ("GraphVAR-soft", lambda: GraphVAR("soft", alpha=1e-5), A_true, "high"),
        ("GraphVAR-hard", lambda: GraphVAR("hard"), A_wrong,
         "collapses to AR: constraint forbids the useful firms"),
        ("GraphVAR-soft", lambda: GraphVAR("soft", alpha=1e-5), A_wrong,
         "still works: penalty can be overridden by evidence"),
    ]
    return pd.DataFrame([
        {"experiment": "2: graph quality", "model": name,
         "graph": "none" if A is None else ("true" if A is A_true else "wrong"),
         **score(make, X, A), "expected": exp}
        for name, make, A, exp in cases
    ])


def main():
    one, two = experiment_1(), experiment_2()

    print("Experiment 1 — can each model see a pairwise lead-lag structure?")
    print(one[["model", "oos_r2", "dir_acc", "expected"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nExperiment 2 — does GraphVAR depend on the graph being correct?")
    print(two[["model", "graph", "oos_r2", "dir_acc", "expected"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    out = ROOT / "reports" / "model_validation.csv"
    pd.concat([one, two], ignore_index=True).to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()