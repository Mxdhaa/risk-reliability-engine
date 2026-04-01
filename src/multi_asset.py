"""
multi_asset.py
--------------
Runs the full reliability monitoring pipeline across multiple assets,
producing a cross-asset results table for the paper.

Why this matters:
  A single-asset evaluation (S&P 500 only) is the weakest point of the
  original paper. Reviewers at ICAIF or AAAI will immediately ask:
  "Does this generalise?" This module answers that.

Assets covered (all available via Yahoo Finance):
  Equities  : ^GSPC (S&P 500), ^FTSE (FTSE 100), ^N225 (Nikkei),
              ^GDAXI (DAX), ^HSI (Hang Seng)
  EM        : EEM (iShares EM ETF)
  Volatility: ^VIX as a feature input (not a target)

Usage:
  from multi_asset import run_multi_asset
  results_df = run_multi_asset(cfg)
  results_df.to_csv("artifacts/multi_asset_results.csv", index=False)
"""

import pandas as pd
import numpy as np
from dataclasses import replace

from config import Config
from data_factory import fetch_ohlcv, compute_returns
from risk_models import rolling_historical_var, ewma_vol
from features import build_reliability_features
from reliability import make_labels_next_day, train_reliability_xgb, predict_calibrated
from baselines import run_all_baselines
from regime_detector import RegimeDetector
from rcre import RCREModel
from backtest import gate_policy, compute_metrics_v2


ASSETS = {
    "S&P 500":   "^GSPC",
    "FTSE 100":  "^FTSE",
    "Nikkei 225": "^N225",
    "DAX":       "^GDAXI",
    "Hang Seng": "^HSI",
    "EM ETF":    "EEM",
}


def run_single_asset(ticker: str, cfg: Config) -> dict:
    """
    Full pipeline for one ticker. Returns a dict of key metrics.
    """
    try:
        df = compute_returns(fetch_ohlcv(ticker, cfg.start, cfg.end))
    except Exception as e:
        print(f"  [{ticker}] Data fetch failed: {e}")
        return {"ticker": ticker, "error": str(e)}

    r = df["logret"].astype(float)
    r_delayed = r.shift(cfg.delay)

    df["var_base"]  = rolling_historical_var(r_delayed, cfg.alpha, cfg.window)
    df["sigma_hat"] = ewma_vol(r_delayed.fillna(0.0))

    X = build_reliability_features(
        r, df["var_base"], df["sigma_hat"],
        cfg.exceed_k, cfg.kurt_window, cfg.vol_mom_lag
    )
    y = make_labels_next_day(r, df["var_base"], cfg.gamma)

    # Drop NaNs — some assets have shorter histories
    valid = X.dropna().index.intersection(y.dropna().index)
    if len(valid) < 500:
        return {"ticker": ticker, "error": f"Insufficient data: {len(valid)} rows"}

    X = X.loc[valid]
    y = y.loc[valid]

    # XGBoost + calibration
    try:
        model, calibrator, fit_metrics, splits = train_reliability_xgb(X, y, cfg)
    except Exception as e:
        return {"ticker": ticker, "error": f"XGB training failed: {e}"}

    s = predict_calibrated(model, calibrator, X)

    # Baselines
    try:
        baseline_results, _, _ = run_all_baselines(X, y, df["sigma_hat"].loc[valid], cfg)
        lr_auc = baseline_results["LogisticRegression"]["auc_test"]
        vt_auc = baseline_results["VolThreshold"]["auc_test"]
    except Exception:
        lr_auc = float("nan")
        vt_auc = float("nan")

    # RCRE
    try:
        rd = RegimeDetector(n_regimes=cfg.n_regimes, vol_series=df["sigma_hat"].loc[valid],
                            train_ratio=cfg.train_ratio)
        rd.fit()
        regimes = rd.predict(df["sigma_hat"].loc[valid])
        rcre = RCREModel(n_regimes=cfg.n_regimes, cfg=cfg)
        rcre.fit(X, y, regimes)
        s_rcre = rcre.predict(X, regimes)
        rcre_metrics = rcre.evaluate(X, y, regimes, cfg)
        rcre_auc = rcre_metrics["auc_test"]
    except Exception as e:
        print(f"  [{ticker}] RCRE failed: {e}")
        s_rcre = s.copy()
        rcre_auc = float("nan")

    # Gating metrics (XGBoost)
    df_valid = df.loc[valid].copy()
    df_valid["s_score"] = s
    df_valid["pi"]      = gate_policy(s, cfg.tau_low, cfg.tau_high, cfg.phi)

    n = len(df_valid)
    test_start = int(n * (cfg.train_ratio + cfg.val_ratio))
    test_df = df_valid.iloc[test_start:]

    try:
        dm = compute_metrics_v2(test_df["logret"], test_df["var_base"], test_df["pi"], cfg.alpha)
        esb_base  = dm["esb_base"]
        esb_gated = dm["esb_gated"]
        avg_exp   = dm["avg_exposure"]
    except Exception:
        esb_base = esb_gated = avg_exp = float("nan")

    return {
        "ticker":       ticker,
        "n_days":       len(valid),
        "pos_rate":     round(float(y.mean()), 4),
        "auc_volthresh": round(vt_auc, 3),
        "auc_lr":        round(lr_auc, 3),
        "auc_xgb":       round(fit_metrics["auc_test"], 3),
        "auc_rcre":      round(rcre_auc, 3),
        "brier_xgb":     round(fit_metrics["brier_test"], 4),
        "esb_base":      round(esb_base,  4),
        "esb_gated":     round(esb_gated, 4),
        "esb_reduction": round((esb_base - esb_gated) / max(esb_base, 1e-8), 3),
        "avg_exposure":  round(avg_exp, 3),
    }


def run_multi_asset(cfg: Config, assets: dict = None) -> pd.DataFrame:
    """
    Run the pipeline across all assets and return a summary DataFrame.

    Parameters
    ----------
    cfg    : Config object (symbol field is overridden per asset)
    assets : dict mapping display_name -> ticker. Defaults to ASSETS.
    """
    if assets is None:
        assets = ASSETS

    rows = []
    for name, ticker in assets.items():
        print(f"\n=== {name} ({ticker}) ===")
        # Override the symbol in config for this asset
        asset_cfg = replace(cfg, symbol=ticker)
        result = run_single_asset(ticker, asset_cfg)
        result["asset"] = name
        rows.append(result)
        if "error" not in result:
            print(f"  XGB AUC: {result['auc_xgb']:.3f}  RCRE AUC: {result['auc_rcre']:.3f}  ESB reduction: {result['esb_reduction']:.1%}")

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    import os
    cfg = Config()
    os.makedirs("artifacts", exist_ok=True)
    results = run_multi_asset(cfg)
    results.to_csv("artifacts/multi_asset_results.csv", index=False)
    print("\n=== MULTI-ASSET SUMMARY ===")
    print(results[["asset", "auc_volthresh", "auc_lr", "auc_xgb", "auc_rcre", "esb_reduction"]].to_string(index=False))