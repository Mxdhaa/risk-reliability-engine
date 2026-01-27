from config import Config
from data_factory import fetch_ohlcv, compute_returns
from risk_models import rolling_historical_var, ewma_vol
from features import build_reliability_features
from reliability import make_labels_next_day

cfg = Config()

df = compute_returns(fetch_ohlcv(cfg.symbol, "2015-01-01", "2020-01-01"))
r = df["logret"]

r_delayed = r.shift(cfg.delay)
df["var_base"] = rolling_historical_var(r_delayed, cfg.alpha, cfg.window)
df["sigma_hat"] = ewma_vol(r_delayed.fillna(0.0))

X = build_reliability_features(r, df["var_base"], df["sigma_hat"], cfg.exceed_k, cfg.kurt_window, cfg.vol_mom_lag)
y = make_labels_next_day(r, df["var_base"], cfg.gamma)

z = X.copy()
z["y"] = y
z = z.dropna()

print("X shape:", z.drop(columns=["y"]).shape)
print("y pos rate:", z["y"].mean())
