"""lstar.py — Logistic Smooth Transition Autoregressive model."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression


class LSTAR:
    """Logistic STAR: G(z) = 1 / (1 + exp(-gamma * (z - c)))."""

    def __init__(self, p: int = 1, d: int = 1) -> None:
        self.p = p
        self.d = d

    def _make_lags(self, y: np.ndarray) -> np.ndarray:
        T = len(y)
        X = np.zeros((T - self.p - self.d, self.p))
        for i in range(self.p):
            X[:, i] = y[self.p + self.d - 1 - i : T - 1 - i]
        return np.column_stack([np.ones(len(X)), X])

    def _transition(self, z: np.ndarray, gamma: float, c: float) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-gamma * (z - c)))

    def _nll(
        self,
        params: np.ndarray,
        y_target: np.ndarray,
        X: np.ndarray,
        z: np.ndarray,
    ) -> float:
        gamma, c = params[0], params[1]
        phi1 = params[2 : 2 + X.shape[1]]
        phi2 = params[2 + X.shape[1] : 2 + 2 * X.shape[1]]
        sigma2 = np.exp(params[-1])
        G = self._transition(z, gamma, c)
        y_pred = (1 - G) * (X @ phi1) + G * (X @ phi2)
        return (
            0.5 * len(y_target) * np.log(2 * np.pi * sigma2)
            + 0.5 * np.sum((y_target - y_pred) ** 2) / sigma2
        )

    def fit(self, y: np.ndarray) -> "LSTAR":
        self.y_ = y
        X = self._make_lags(y)
        z = y[self.p + self.d - 1 : -1]
        y_target = y[self.p + self.d :]

        # Initialise from OLS on full sample
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y_target)
        phi_init = reg.coef_

        # Grid-search initial threshold from percentiles
        best_ssr = np.inf
        best_c = float(np.median(z))
        for pct in range(20, 81, 10):
            c0 = float(np.percentile(z, pct))
            G0 = self._transition(z, 5.0, c0)
            y_pred0 = (1 - G0) * (X @ phi_init) + G0 * (X @ phi_init * 0.5)
            ssr = float(np.sum((y_target - y_pred0) ** 2))
            if ssr < best_ssr:
                best_ssr = ssr
                best_c = c0

        n_params = 2 + 2 * X.shape[1] + 1
        x0 = np.zeros(n_params)
        x0[0] = 5.0
        x0[1] = best_c
        x0[2 : 2 + X.shape[1]] = phi_init
        x0[2 + X.shape[1] : 2 + 2 * X.shape[1]] = phi_init * 0.5
        x0[-1] = np.log(max(np.var(y_target), 1e-8))

        bounds = (
            [(0.1, 50.0), (float(z.min()), float(z.max()))]
            + [(None, None)] * (2 * X.shape[1])
            + [(None, None)]
        )

        result = minimize(
            self._nll,
            x0,
            args=(y_target, X, z),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000},
        )

        self.params_ = result.x
        self.gamma_ = float(result.x[0])
        self.c_ = float(result.x[1])
        self.phi1_ = result.x[2 : 2 + X.shape[1]]
        self.phi2_ = result.x[2 + X.shape[1] : 2 + 2 * X.shape[1]]
        self.sigma_ = float(np.sqrt(np.exp(result.x[-1])))
        return self

    def predict(self, y_hist: np.ndarray) -> float:
        X_new = np.array([1.0] + list(y_hist[-self.p :][::-1])).reshape(1, -1)
        z = float(y_hist[-self.d])
        G = float(self._transition(np.array([z]), self.gamma_, self.c_)[0])
        return float((1 - G) * (X_new @ self.phi1_) + G * (X_new @ self.phi2_))
