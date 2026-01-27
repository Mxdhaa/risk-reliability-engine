from data_factory import fetch_ohlcv, compute_returns
from risk_models import rolling_historical_var, ewma_vol

df = compute_returns(fetch_ohlcv("^GSPC", "2018-01-01", "2020-01-01"))
r = df["logret"]

var_hs = rolling_historical_var(r.shift(1), 0.99, 252)
sig = ewma_vol(r.shift(1).fillna(0.0))

print("var_hs last:", var_hs.dropna().iloc[-1])
print("sigma last:", sig.dropna().iloc[-1])
