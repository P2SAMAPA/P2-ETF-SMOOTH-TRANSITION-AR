import numpy as np
from sklearn.linear_model import LinearRegression

class SETAR:
    def __init__(self, p=1, d=1):
        self.p = p
        self.d = d
        
    def _make_lags(self, y):
        T = len(y)
        X = np.zeros((T - self.p - self.d, self.p))
        for i in range(self.p):
            X[:, i] = y[self.p + self.d - 1 - i:T - 1 - i]
        X = np.column_stack([np.ones(len(X)), X])
        return X
    
    def _fit_for_threshold(self, y, X, c):
        z = y[self.p + self.d - 1:-1]
        mask_low = z <= c
        mask_high = z > c
        y_target = y[self.p + self.d:]
        if mask_low.sum() < self.p + 1 or mask_high.sum() < self.p + 1:
            return np.inf, None, None
        reg1 = LinearRegression(fit_intercept=False)
        reg1.fit(X[mask_low], y_target[mask_low])
        ssr1 = np.sum((y_target[mask_low] - reg1.predict(X[mask_low]))**2)
        reg2 = LinearRegression(fit_intercept=False)
        reg2.fit(X[mask_high], y_target[mask_high])
        ssr2 = np.sum((y_target[mask_high] - reg2.predict(X[mask_high]))**2)
        return ssr1 + ssr2, reg1.coef_, reg2.coef_
    
    def fit(self, y):
        self.y_ = y
        X = self._make_lags(y)
        z = y[self.p + self.d - 1:-1]
        c_grid = np.percentile(z, np.arange(10, 91, 2))
        best_ssr = np.inf
        for c in c_grid:
            ssr, phi1, phi2 = self._fit_for_threshold(y, X, c)
            if ssr < best_ssr:
                best_ssr = ssr
                self.c_ = c
                self.phi1_ = phi1
                self.phi2_ = phi2
        _, self.phi1_, self.phi2_ = self._fit_for_threshold(y, X, self.c_)
        # Store regressors for predict
        z = y[self.p + self.d - 1:-1]
        y_target = y[self.p + self.d:]
        mask_low = z <= self.c_
        mask_high = z > self.c_
        self.reg1_ = LinearRegression(fit_intercept=False)
        self.reg1_.fit(X[mask_low], y_target[mask_low])
        self.reg2_ = LinearRegression(fit_intercept=False)
        self.reg2_.fit(X[mask_high], y_target[mask_high])
        # Compute AIC/BIC
        n = len(y_target)
        k = 2 * (self.p + 1) + 1  # two regimes + threshold
        self.aic_ = n * np.log(best_ssr / n) + 2 * k
        self.bic_ = n * np.log(best_ssr / n) + k * np.log(n)
        self.regime_props_ = {'regime1_pct': mask_low.mean(), 'regime2_pct': mask_high.mean(), 'threshold': self.c_}
        return self
    
    def predict(self, y_hist):
        X_new = np.array([1] + y_hist[-self.p:][::-1].tolist()).reshape(1, -1)
        z = y_hist[-self.d]
        if z <= self.c_:
            return self.reg1_.predict(X_new)[0]
        else:
            return self.reg2_.predict(X_new)[0]
