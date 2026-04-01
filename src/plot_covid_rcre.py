"""
plot_covid_rcre.py  --  Figure 4: COVID-19 RCRE case study
Run from src/: python plot_covid_rcre.py
Saves: artifacts/fig4_covid_rcre.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from config import Config
from data_factory import fetch_ohlcv, compute_returns
from risk_models import rolling_historical_var, ewma_vol
from features import build_reliability_features
from reliability import make_labels_next_day, train_reliability_xgb, predict_calibrated
from rcre import RCREModel
from regime_detector import RegimeDetector


def plot_covid_rcre(save_dir="artifacts"):
    os.makedirs(save_dir, exist_ok=True)
    cfg = Config()

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("Fetching data...")
    raw = fetch_ohlcv(cfg.symbol, cfg.start, cfg.end)
    if raw.empty:
        raise RuntimeError("fetch_ohlcv returned empty dataframe. Check network/yfinance.")
    df = compute_returns(raw)
    r  = df["logret"].astype(float)
    r_delayed = r.shift(cfg.delay)

    df["var_base"]  = rolling_historical_var(r_delayed, cfg.alpha, cfg.window)
    df["sigma_hat"] = ewma_vol(r_delayed.fillna(0.0))

    X = build_reliability_features(
        r, df["var_base"], df["sigma_hat"],
        cfg.exceed_k, cfg.kurt_window, cfg.vol_mom_lag
    )
    y = make_labels_next_day(r, df["var_base"], cfg.gamma)

    # ── 2. Regime labels ─────────────────────────────────────────────────────
    print("Detecting regimes...")
    detector = RegimeDetector(n_regimes=cfg.n_regimes, vol_series=df["sigma_hat"], cusum_h=cfg.cusum_h)
    detector.fit()
    regimes  = detector.predict(df["sigma_hat"])
    regimes  = regimes.reindex(X.index).ffill().fillna(0).astype(int)

    # ── 3. Train RCRE ────────────────────────────────────────────────────────
    print("Training RCRE...")
    rcre = RCREModel(
        n_regimes=cfg.n_regimes,
        cfg=cfg,
        mixing_window=cfg.mixing_window,
        mixing_tau=cfg.mixing_tau
    )
    rcre.fit(X, y, regimes)

    print("Generating RCRE predictions...")
    s_rcre = rcre.predict(X, regimes).fillna(0)

    # ── 4. XGBoost baseline ──────────────────────────────────────────────────
    print("Training XGBoost baseline...")
    xgb_model, xgb_cal, _, _ = train_reliability_xgb(X, y, cfg)
    s_xgb = pd.Series(predict_calibrated(xgb_model, xgb_cal, X), index=X.index).fillna(0)

    # ── 5. COVID window ──────────────────────────────────────────────────────
    covid_start = "2019-10-01"
    covid_end   = "2020-09-01"
    shock_start = "2020-02-20"
    shock_end   = "2020-04-30"

    mask = (df.index >= covid_start) & (df.index <= covid_end)

    raw_df = fetch_ohlcv(cfg.symbol, covid_start, covid_end)
    # get close price robustly
    close_cols = [c for c in raw_df.columns if "close" in c.lower()]
    price = raw_df[close_cols[0]] if close_cols else raw_df.iloc[:, 3]

    s_rcre_c = s_rcre[mask]
    s_xgb_c  = s_xgb[mask]
    var_c    = df["var_base"][mask]
    breach_dates = y[mask][y[mask] == 1].index

    # ── 6. Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1, 1.4]})
    fig.suptitle(
        "Figure 4: COVID-19 Stress Test  |  RCRE Reliability Monitor  |  S&P 500",
        fontsize=12, fontweight="bold", y=0.99
    )
    sc = "#AEC6CF"  # shock shading colour

    # Panel A: Price
    ax1 = axes[0]
    ax1.plot(price.index, price.values, color="#1a1a2e", linewidth=1.2)
    ax1.axvspan(shock_start, shock_end, alpha=0.20, color=sc, label="Acute shock window")
    ax1.set_ylabel("S&P 500 Price (USD)", fontsize=9)
    ax1.legend(fontsize=8, loc="lower left")
    ax1.set_title("Panel A: S&P 500 Price", fontsize=9, loc="left", style="italic")
    for s in ["top", "right"]: ax1.spines[s].set_visible(False)

    # Panel B: Baseline VaR
    ax2 = axes[1]
    ax2.plot(var_c.index, var_c.values, color="#E74C3C", linewidth=1.1,
             label="Baseline HS-VaR (delayed, alpha=0.99)")
    ax2.axvspan(shock_start, shock_end, alpha=0.20, color=sc)
    ax2.set_ylabel("VaR (portfolio fraction)", fontsize=9)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.set_title("Panel B: Baseline HS-VaR  (252-day rolling, stale during shock)",
                  fontsize=9, loc="left", style="italic")
    for s in ["top", "right"]: ax2.spines[s].set_visible(False)

    # Panel C: RCRE vs XGB scores
    ax3 = axes[2]
    ax3.fill_between(s_rcre_c.index, 0, s_rcre_c.values, color="#2980B9", alpha=0.28)
    ax3.plot(s_rcre_c.index, s_rcre_c.values, color="#2980B9",
             linewidth=1.3, label="RCRE failure prob.")
    ax3.plot(s_xgb_c.index, s_xgb_c.values, color="#27AE60",
             linewidth=0.9, linestyle="--", alpha=0.75, label="XGBoost failure prob.")

    ax3.axhline(cfg.tau_low,  color="orange",  linewidth=1.0, linestyle="--",
                label=f"tau_low={cfg.tau_low}  (de-risk)")
    ax3.axhline(cfg.tau_high, color="#E74C3C", linewidth=1.0, linestyle="--",
                label=f"tau_high={cfg.tau_high}  (halt)")

    ax3.fill_between(s_rcre_c.index, cfg.tau_low, cfg.tau_high,
                     where=(s_rcre_c >= cfg.tau_low) & (s_rcre_c < cfg.tau_high),
                     color="orange", alpha=0.10)
    ax3.fill_between(s_rcre_c.index, cfg.tau_high,
                     s_rcre_c.clip(lower=cfg.tau_high),
                     where=(s_rcre_c >= cfg.tau_high),
                     color="#E74C3C", alpha=0.15)

    for bd in breach_dates:
        if bd in s_rcre_c.index:
            ax3.axvline(bd, color="#E74C3C", linewidth=0.5, alpha=0.35)

    ax3.axvspan(shock_start, shock_end, alpha=0.20, color=sc)
    ax3.set_ylabel("Failure probability", fontsize=9)
    ax3.set_xlabel("Date", fontsize=9)
    ax3.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.85)
    ax3.set_title(
        "Panel C: RCRE Score  (red verticals = actual breach days, shading = gating zones)",
        fontsize=9, loc="left", style="italic"
    )
    for s in ["top", "right"]: ax3.spines[s].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = f"{save_dir}/fig4_covid_rcre.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    plot_covid_rcre()