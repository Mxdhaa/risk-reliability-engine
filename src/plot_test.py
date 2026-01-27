from config import Config
from data_factory import fetch_ohlcv, compute_returns
from risk_models import rolling_historical_var, ewma_vol
from features import build_reliability_features
from reliability import make_labels_next_day, train_reliability_xgb
from backtest import gate_policy
from viz import killer_plot
import pandas as pd

cfg = Config()

df = fetch_ohlcv(cfg.symbol, cfg.start, cfg.end)
df = compute_returns(df)

r = df["logret"].astype(float)
r_delayed = r.shift(cfg.delay)

df["var_base"] = rolling_historical_var(r_delayed, cfg.alpha, cfg.window)
df["sigma_hat"] = ewma_vol(r_delayed.fillna(0.0))

X = build_reliability_features(r, df["var_base"], df["sigma_hat"], cfg.exceed_k, cfg.kurt_window, cfg.vol_mom_lag)
y = make_labels_next_day(r, df["var_base"], cfg.gamma)

model, _ = train_reliability_xgb(X, y, cfg)

Xn = X.dropna()
s = pd.Series(index=X.index, dtype=float)
s.loc[Xn.index] = model.predict_proba(Xn)[:, 1]
df["s_score"] = s

df["pi"] = gate_policy(df["s_score"], cfg.tau_low, cfg.tau_high, cfg.phi)

killer_plot(
    df,
    start="2020-02-01",
    end="2020-05-15",
    tau_high=cfg.tau_high,
    title="COVID Crash: Price vs Baseline VaR vs Reliability (Detection Zone Shaded)"
)
