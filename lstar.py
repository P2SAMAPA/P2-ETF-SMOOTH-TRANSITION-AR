import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

class LSTAR:
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
    
    def _transition(self, z, gamma, c):
        return 1 / (1 + np.exp(-gamma * (z - c)))
    
    def _nll(self, params, y, X, z):
        gamma, c = params[0], params[1]
        phi1 = params[2:2 + X.shape[1]]
        phi2 = params[2 + X.shape[1]:2 + 2 * X.shape[1]]
        sigma2 = np.exp(params[-1])
        G = self._transition(z, gamma, c)
        y_pred = (1 - G) * (X @ phi1) + G * (X @ phi2)
        nll = 0.5 * len(y) * np.log(2 * np.pi * sigma2) + 0.5 * np.sum((y - y_pred)**2) / sigma2
        return nll
    
    def fit(self, y):
        self.y_ = y
        X = self._make_lags(y)
        z = y[self.p + self.d - 1:-1]
        y_target = y[self.p + self.d:]
        # initial guess from SETAR
        from setar import SETAR
        s = SETAR(p=self.p, d=self.d)
        s.fit(y)
        n_params = 2 + 2 * X.shape[1] + 1
        x0 = np.zeros(n_params)
        x0[0] = 5.0
        x0[1] = s.c_
        x0[2:2 + X.shape[1]] = s.phi1_
        x0[2 + X.shape[1]:2 + 2 * X.shape[1]] = s.phi2_
        x0[-1] = np.log(np.var(y_target))
        bounds = [(0.1, 50), (np.min(z), np.max(z))] + [(None, None)] * (2 * X.shape[1]) + [(None, None)]
        result = minimize(self._nll, x0, args=(y_target, X, z), method='L-BFGS-B', bounds=bounds)
        self.params_ = result.x
        self.gamma_ = result.x[0]
        self.c_ = result.x[1]
        self.phi1_ = result.x[2:2 + X.shape[1]]
        self.phi2_ = result.x[2 + X.shape[1]:2 + 2 * X.shape[1]]
        self.sigma_ = np.sqrt(np.exp(result.x[-1]))
        return self
    
    def predict(self, y_hist):
        X_new = np.array([1] + y_hist[-self.p:][::-1].tolist()).reshape(1, -1)
        z = y_hist[-self.d]
        G = self._transition(z, self.gamma_, self.c_)
        return (1 - G) * (X_new @ self.phi1_) + G * (X_new @ self.phi2_)
