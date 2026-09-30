"""Independent check of the statsmodels fits (thesis part D).

`ri_lme` is a from-scratch random-intercept linear mixed model: for a given
variance ratio lambda = tau^2 / sigma^2 the GLS estimate of beta and sigma^2 are
closed-form (V_i = sigma^2 (I + lambda J)), so the log-likelihood is profiled
onto lambda and maximised with a bounded scalar search (plus the lambda = 0
boundary). It shares no code with statsmodels, and is used to re-derive the
coefficients, standard errors, Wald chi-square statistics, variance components
and ICCs reported in the thesis.
"""

import numpy as np
import pandas as pd
import patsy
from scipy import optimize, stats


def ri_lme(formula: str, data: pd.DataFrame, group: str, reml: bool = False) -> dict:
    y, X = patsy.dmatrices(formula, data, return_type="dataframe")
    names = list(X.columns)
    y = y.to_numpy(float).ravel()
    X = X.to_numpy(float)
    g = pd.factorize(data.loc[:, group].to_numpy())[0]
    N, p = X.shape
    G = g.max() + 1
    n_i = np.bincount(g, minlength=G).astype(float)
    Sx = np.zeros((G, p))
    np.add.at(Sx, g, X)
    Sy = np.bincount(g, weights=y, minlength=G)
    XtX, Xty, yty = X.T @ X, X.T @ y, y @ y

    def pieces(lam):
        c = lam / (1.0 + n_i * lam)
        A = XtX - (Sx * c[:, None]).T @ Sx
        bvec = Xty - (Sx * c[:, None]).T @ Sy
        beta = np.linalg.solve(A, bvec)
        rss = (yty - np.sum(c * Sy ** 2)) - beta @ bvec
        logdetH = np.sum(np.log1p(n_i * lam))
        return A, beta, rss, logdetH

    def negll(log_lam):
        lam = np.exp(log_lam) if np.isfinite(log_lam) else 0.0
        return -_ll(lam)

    def _ll(lam):
        A, beta, rss, logdetH = pieces(lam)
        if reml:
            s2 = rss / (N - p)
            _, logdetA = np.linalg.slogdet(A)
            return -0.5 * ((N - p) * np.log(2 * np.pi * s2) + logdetH + logdetA + (N - p))
        s2 = rss / N
        return -0.5 * (N * np.log(2 * np.pi * s2) + logdetH + N)

    opt = optimize.minimize_scalar(negll, bounds=(-25, 10), method="bounded",
                                   options={"xatol": 1e-10, "maxiter": 1000})
    cand = [(np.exp(opt.x), -opt.fun), (0.0, _ll(0.0))]
    lam, llf = max(cand, key=lambda t: t[1])
    A, beta, rss, _ = pieces(lam)
    s2 = rss / ((N - p) if reml else N)
    cov = s2 * np.linalg.inv(A)
    se = np.sqrt(np.diag(cov))
    tau2 = lam * s2
    return {"names": names, "beta": pd.Series(beta, index=names), "se": pd.Series(se, index=names),
            "cov": pd.DataFrame(cov, index=names, columns=names), "sigma2": s2, "tau2": tau2,
            "icc": tau2 / (tau2 + s2), "llf": llf, "n": N, "groups": G}


def wald(fit: dict, prefix: str) -> tuple[float, int, float]:
    idx = [n for n in fit["names"] if n.startswith(prefix)]
    b = fit["beta"][idx].to_numpy()
    V = fit["cov"].loc[idx, idx].to_numpy()
    w = float(b @ np.linalg.solve(V, b))
    return w, len(idx), float(stats.chi2.sf(w, len(idx)))


def holm(p) -> np.ndarray:
    """Hand-written Holm step-down adjustment."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def rel_diff(a, b) -> float:
    a, b = float(a), float(b)
    return abs(a - b) / max(abs(a), abs(b), 1e-12)
