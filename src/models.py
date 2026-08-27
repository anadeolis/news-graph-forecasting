"""
Every model implements:
    fit(W)            W is a (T_window x N) array of past returns, the most
                      recent row last. 
    predict(x_last)   x_last is the most recent day's returns (N,).
                      Returns an (N,) forecast for the next day.

"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.mixture import GaussianMixture


def _lagged(W):
    """yesterday predicts td"""
    return W[:-1], W[1:]


class Model:
    

    name = "model"

    def fit(self, W):
        raise NotImplementedError

    def predict(self, x_last):
        raise NotImplementedError


class AR(Model):

    name = "AR"

    def __init__(self, p: int = 1):
        self.p = p

    def fit(self, W):
        Z, Y = _lagged(W)
        n = W.shape[1]
        self.coef_ = np.zeros(n)
        self.intercept_ = np.zeros(n)
        for i in range(n):
            z = np.column_stack([np.ones(len(Z)), Z[:, i]])
            beta, *_ = np.linalg.lstsq(z, Y[:, i], rcond=None)
            self.intercept_[i], self.coef_[i] = beta
        return self

    def predict(self, x_last):
        return self.intercept_ + self.coef_ * x_last


class VAR(Model):

    name = "VAR"

    def fit(self, W):
        Z, Y = _lagged(W)
        Zd = np.column_stack([np.ones(len(Z)), Z])
        self.B_, *_ = np.linalg.lstsq(Zd, Y, rcond=None)
        return self

    def predict(self, x_last):
        return np.concatenate([[1.0], x_last]) @ self.B_


class SparseVAR(Model):
    """VAR(1) with an L1 (LASSO) or L1+L2 (elastic net) penalty per stock.
    Each stock's equation is fit separately so coefficient matrix is
    estimated row by row. The penalty drives most coefficients to exactly
    zero 
    """

    def __init__(self, alpha=1e-4, l1_ratio=1.0, penalty_weights=None,
                 max_iter=2000):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.penalty_weights = penalty_weights
        self.max_iter = max_iter

    @property
    def name(self):
        kind = "LASSO" if self.l1_ratio == 1.0 else f"EN({self.l1_ratio})"
        return f"SparseVAR-{kind}"

    def _estimator(self):
        if self.l1_ratio == 1.0:
            return Lasso(alpha=self.alpha, max_iter=self.max_iter)
        return ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                          max_iter=self.max_iter)

    def fit(self, W):
        Z, Y = _lagged(W)
        n = W.shape[1]
        self.B_ = np.zeros((n, n))
        self.intercept_ = np.zeros(n)
        for i in range(n):
            Zi, scale = Z, None
            if self.penalty_weights is not None:
                # A per-coefficient penalty w_j is equivalent to dividing
                # column j by w_j, fitting an ordinary penalised model, then
                # dividing the coefficient back. This is the standard
                # adaptive-LASSO trick and avoids writing a custom solver.
                scale = np.where(self.penalty_weights[i] > 0,
                                 1.0 / self.penalty_weights[i], 0.0)
                Zi = Z * scale
            est = self._estimator().fit(Zi, Y[:, i])
            coef = est.coef_ if scale is None else est.coef_ * scale
            self.B_[i] = coef
            self.intercept_[i] = est.intercept_
        return self

    def predict(self, x_last):
        return self.intercept_ + self.B_ @ x_last


class FactorModel(Model):
    """PCA factors, forecast by AR(1), projected back to stocks.
    """

    name = "Factor"

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, W):
        self.mean_ = W.mean(axis=0)
        Wc = W - self.mean_
        # economy SVD: columns of V are the loadings, U*S the factor paths
        U, S, Vt = np.linalg.svd(Wc, full_matrices=False)
        k = min(self.k, Vt.shape[0])
        self.loadings_ = Vt[:k].T                      # N x k
        F = Wc @ self.loadings_                        # T x k factor series
        self.phi_ = np.zeros(k)
        self.c_ = np.zeros(k)
        for j in range(k):
            z = np.column_stack([np.ones(len(F) - 1), F[:-1, j]])
            beta, *_ = np.linalg.lstsq(z, F[1:, j], rcond=None)
            self.c_[j], self.phi_[j] = beta
        return self

    def predict(self, x_last):
        f_last = (x_last - self.mean_) @ self.loadings_
        f_next = self.c_ + self.phi_ * f_last
        return self.mean_ + self.loadings_ @ f_next


class NIRVAR(Model):

    name = "NIRVAR"

    def __init__(self, seed: int = 4436):
        self.seed = seed

    def fit(self, W):
        T_w, N = W.shape
        C = np.nan_to_num(np.corrcoef(W.T), nan=0.0)
        np.fill_diagonal(C, 1.0)
        evals, evecs = np.linalg.eigh(C)                   # ascending
        cutoff = (1 + np.sqrt(N / T_w)) ** 2                # Marchenko-Pastur
        d = max(int((evals > cutoff).sum()), 2)
        emb = evecs[:, -d:] * np.sqrt(evals[-d:])
        try:
            labels = GaussianMixture(n_components=d, random_state=self.seed,
                                     init_params="k-means++").fit_predict(emb)
        except Exception:
            labels = KMeans(n_clusters=d, random_state=self.seed,
                            n_init="auto").fit_predict(emb)
        Z, Y = _lagged(W)
        self.blocks_ = []
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            Phi_c, *_ = np.linalg.lstsq(Z[:, idx], Y[:, idx], rcond=None)
            self.blocks_.append((idx, Phi_c))              # no intercept
        self.d_ = d
        self.labels_ = labels
        return self

    def predict(self, x_last):
        s = np.zeros(len(x_last))
        for idx, Phi_c in self.blocks_:
            s[idx] = x_last[idx] @ Phi_c
        return s


class GraphVAR(Model):
    """

    hard   firm i is regressed only on its graph neighbours (plus itself).
           Unlinked firms cannot contribute even if strong the in-sample 
           evidence. 

    soft   every firm may predict every other, but the LASSO penalty is
           lighter on linked pairs than unlinked ones. 

    Firms with no neighbours fall back to their own lag alone, which makes
    the model nest AR rather than predicting zero for isolated firms. 
    at the observed graph density roughly 40% of firms are
    isolated in any given snapshot.
    """

    def __init__(self, mode: str = "hard", alpha: float = 1e-4,
                 l1_ratio: float = 1.0, linked_penalty: float = 0.1):
        if mode not in ("hard", "soft"):
            raise ValueError("mode must be 'hard' or 'soft'")
        self.mode = mode
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.linked_penalty = linked_penalty

    @property
    def name(self):
        return f"GraphVAR-{self.mode}"

    def fit(self, W, A=None):
        """W is the return window; A is the adjacency matrix for this date."""
        n = W.shape[1]
        if A is None:
            A = np.zeros((n, n))
        self.n_ = n

        if self.mode == "soft":
            weights = np.where(A > 0, self.linked_penalty, 1.0).astype(float)
            np.fill_diagonal(weights, self.linked_penalty)
            self._inner = SparseVAR(alpha=self.alpha, l1_ratio=self.l1_ratio,
                                    penalty_weights=weights).fit(W)
            return self

        mask = (A > 0)
        np.fill_diagonal(mask, True)          # a firm may always use its own lag
        Z, Y = _lagged(W)
        self.B_ = np.zeros((n, n))
        self.intercept_ = np.zeros(n)
        self.n_predictors_ = mask.sum(axis=1)
        for i in range(n):
            cols = np.where(mask[i])[0]
            Zi = np.column_stack([np.ones(len(Z)), Z[:, cols]])
            beta, *_ = np.linalg.lstsq(Zi, Y[:, i], rcond=None)
            self.intercept_[i] = beta[0]
            self.B_[i, cols] = beta[1:]
        return self

    def predict(self, x_last):
        if self.mode == "soft":
            return self._inner.predict(x_last)
        return self.intercept_ + self.B_ @ x_last


BASELINES = {
    "AR": AR,
    "VAR": VAR,
    "SparseVAR-LASSO": lambda: SparseVAR(l1_ratio=1.0, alpha=3.16e-5),
    "SparseVAR-EN": lambda: SparseVAR(l1_ratio=0.5, alpha=1e-4),
    "Factor": FactorModel,
    "NIRVAR": NIRVAR,
}

# Graph models take an adjacency matrix in fit(); the backtest supplies it.
GRAPH_MODELS = {
    "GraphVAR-hard": lambda: GraphVAR(mode="hard"),
    "GraphVAR-soft": lambda: GraphVAR(mode="soft", alpha=1e-4),
}
