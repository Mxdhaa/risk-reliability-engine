import numpy as np
import pandas as pd

def kurtosis_rolling(x: pd.Series, window: int) -> pd.Series:
    def _k(a: np.ndarray) -> float:
        if a.size < 4:
            return np.nan
        m = a.mean()
        s2 = ((a - m) ** 2).mean()
        if s2 <= 0:
            return np.nan
        m4 = ((a - m) ** 4).mean()
        return m4 / (s2 ** 2) - 3.0
    return x.rolling(window).apply(lambda a: _k(a.values), raw=False)

# def build_reliability_features(
#     returns: pd.Series,
#     reported_risk: pd.Series,
#     sigma_hat: pd.Series,
#     exceed_k: int,
#     kurt_window: int,
#     vol_mom_lag: int
# ) -> pd.DataFrame:
#     df = pd.DataFrame(index=returns.index)
#     L = -returns.astype(float)

#     df["sigma_hat"] = sigma_hat
#     df["vol_mom"] = (sigma_hat - sigma_hat.shift(vol_mom_lag)) / (sigma_hat.shift(vol_mom_lag).replace(0, np.nan))
#     df["risk_mismatch"] = (reported_risk - sigma_hat)

#     df["exceed_count"] = (L > reported_risk).rolling(exceed_k).sum()
#     df["kurtosis"] = kurtosis_rolling(returns, kurt_window)

#     df["loss_minus_risk"] = (L - reported_risk)
#     df["abs_loss"] = L.abs()

#     return df
def build_reliability_features(
    returns: pd.Series,
    reported_risk: pd.Series,
    sigma_hat: pd.Series,
    exceed_k: int,
    kurt_window: int,
    vol_mom_lag: int
) -> pd.DataFrame:
    df = pd.DataFrame(index=returns.index)
    L = -returns.astype(float)
    df["sigma_hat"]      = sigma_hat
    df["vol_mom"]        = (sigma_hat - sigma_hat.shift(vol_mom_lag)) / (sigma_hat.shift(vol_mom_lag).replace(0, np.nan))
    df["risk_mismatch"]  = (reported_risk - sigma_hat)
    df["loss_minus_risk"] = (L - reported_risk)
    df["abs_loss"]       = L.abs()
    return df