from config import Config
from data_factory import fetch_ohlcv, compute_returns
from risk_models import rolling_historical_var, ewma_vol
from features import build_reliability_features
from reliability import make_labels_next_day, train_reliability_xgb
from backtest import gate_policy, compute_metrics_v2
import pandas as pd

cfg = Config()

df = compute_returns(fetch_ohlcv(cfg.symbol, "2005-01-01", "2020-01-01"))
r = df["logret"].astype(float)

r_delayed = r.shift(cfg.delay)
df["var_base"] = rolling_historical_var(r_delayed, cfg.alpha, cfg.window)
df["sigma_hat"] = ewma_vol(r_delayed.fillna(0.0))

X = build_reliability_features(r, df["var_base"], df["sigma_hat"], cfg.exceed_k, cfg.kurt_window, cfg.vol_mom_lag)
y = make_labels_next_day(r, df["var_base"], cfg.gamma)

model, fit_metrics = train_reliability_xgb(X, y, cfg)

Xn = X.dropna()
s = pd.Series(index=X.index, dtype=float)
s.loc[Xn.index] = model.predict_proba(Xn)[:, 1]
df["s_score"] = s

df["pi"] = gate_policy(df["s_score"], cfg.tau_low, cfg.tau_high, cfg.phi)

n = len(df)
test_start = int(n * (cfg.train_ratio + cfg.val_ratio))
test_df = df.iloc[test_start:].copy()

m = compute_metrics_v2(test_df["logret"], test_df["var_base"], test_df["pi"], cfg.alpha)
print("AUC test:", fit_metrics["auc_test"])
print("Decision metrics:", m)
