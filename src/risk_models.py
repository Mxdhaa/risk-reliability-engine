import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm

def rolling_historical_var(returns: pd.Series, alpha: float, window: int) -> pd.Series:
    loss = -returns.astype(float)
    return loss.rolling(window).quantile(alpha)

def ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    r = returns.astype(float).fillna(0.0)
    sigma2 = np.zeros(len(r))
    sigma2[0] = float(r.var())
    for i in range(1, len(r)):
        sigma2[i] = lam * sigma2[i-1] + (1 - lam) * (r.iloc[i-1] ** 2)
    return pd.Series(np.sqrt(sigma2), index=r.index)

def garch_var_with_fallback(returns: pd.Series, alpha: float, window: int) -> pd.Series:
    z = float(norm.ppf(alpha))
    out = pd.Series(index=returns.index, dtype=float)

    r = returns.astype(float)
    hs_fallback = rolling_historical_var(r, alpha, window)

    for t in range(window, len(r)):
        window_r = (r.iloc[t-window:t] * 100.0).dropna()
        if len(window_r) < window:
            out.iloc[t] = hs_fallback.iloc[t]
            continue
        try:
            am = arch_model(window_r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
            res = am.fit(disp="off")
            f = res.forecast(horizon=1, reindex=False)
            sigma = float(np.sqrt(f.variance.values[-1, 0]) / 100.0)
            out.iloc[t] = z * sigma
        except Exception:
            out.iloc[t] = hs_fallback.iloc[t]

    return out
