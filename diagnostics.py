import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

def lm_linearity_test(y, p=2, d=1):
    T = len(y)
    X = np.zeros((T - p - d, p))
    for i in range(p):
        X[:, i] = y[p + d - 1 - i:T - 1 - i]
    X = np.column_stack([np.ones(len(X)), X])
    y_target = y[p + d:]
    z = y[p + d - 1:-1]
    Z_aux = np.column_stack([X, X[:,1:]*z[:,None], X[:,1:]*(z**2)[:,None], X[:,1:]*(z**3)[:,None]])
    reg_r = LinearRegression(fit_intercept=False)
    reg_r.fit(X, y_target)
    u_r = y_target - reg_r.predict(X)
    ssr_r = np.sum(u_r**2)
    reg_ur = LinearRegression(fit_intercept=False)
    reg_ur.fit(Z_aux, y_target)
    u_ur = y_target - reg_ur.predict(Z_aux)
    ssr_ur = np.sum(u_ur**2)
    n = len(y_target)
    lm = n * (ssr_r - ssr_ur) / ssr_r
    df = 3 * p
    pvalue = 1 - stats.chi2.cdf(lm, df)
    return lm, pvalue

def reset_test(y, p=2):
    T = len(y)
    X = np.zeros((T - p, p))
    for i in range(p):
        X[:, i] = y[p - 1 - i:T - 1 - i]
    X = np.column_stack([np.ones(len(X)), X])
    y_target = y[p:]
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, y_target)
    yhat = reg.predict(X)
    X_aug = np.column_stack([X, yhat**2, yhat**3])
    reg_r = LinearRegression(fit_intercept=False).fit(X, y_target)
    ssr_r = np.sum((y_target - reg_r.predict(X))**2)
    reg_ur = LinearRegression(fit_intercept=False).fit(X_aug, y_target)
    ssr_ur = np.sum((y_target - reg_ur.predict(X_aug))**2)
    n = len(y_target)
    reset = n * (ssr_r - ssr_ur) / ssr_r
    pvalue = 1 - stats.f.cdf(reset, 2, n - p - 3)
    return reset, pvalue

def tsay_test(y, p=2, d=1):
    T = len(y)
    X = np.zeros((T - p - d, p))
    for i in range(p):
        X[:, i] = y[p + d - 1 - i:T - 1 - i]
    X = np.column_stack([np.ones(len(X)), X])
    y_target = y[p + d:]
    z = y[p + d - 1:-1]
    order = np.argsort(z)
    X_sorted = X[order]
    y_sorted = y_target[order]
    n = len(y_target)
    w = np.zeros(n)
    for t in range(p+1, n):
        X_t = X_sorted[:t]
        y_t = y_sorted[:t]
        reg = LinearRegression(fit_intercept=False).fit(X_t, y_t)
        y_pred = reg.predict(X_sorted[t:t+1])
        w[t] = y_sorted[t] - y_pred[0]
    W = np.cumsum(w)
    S = np.std(w[p+1:])
    tsay = np.max(np.abs(W)) / (S * np.sqrt(n))
    pvalue = 2 * (1 - stats.norm.cdf(tsay))
    return tsay, pvalue
