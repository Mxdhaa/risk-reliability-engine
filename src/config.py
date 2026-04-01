from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ── Data ──────────────────────────────────────────────────
    symbol: str  = "^GSPC"
    start:  str  = "2000-01-01"
    end:    str | None = None

    # ── VaR / ES ──────────────────────────────────────────────
    alpha:  float = 0.99
    window: int   = 252
    delay:  int   = 1

    # ── Reliability label ─────────────────────────────────────
    # gamma > 1 targets SEVERE failures (losses > 1.5x VaR)
    # This focuses the monitor on operationally consequential events
    gamma: float = 0.8

    # ── Features ──────────────────────────────────────────────
    exceed_k:    int = 20   # rolling breach count window
    kurt_window: int = 60   # kurtosis estimation window
    vol_mom_lag: int = 5    # volatility momentum lag

    # ── Train / Val / Test split ───────────────────────────────
    train_ratio: float = 0.6
    val_ratio:   float = 0.2
    # test_ratio is implicitly 0.2

    # ── Gating policy ─────────────────────────────────────────
    tau_low:  float = 0.019  # 70th percentile of XGB scores
    tau_high: float = 0.064  # 90th percentile of XGB scores
    phi:      float = 0.50   # mid-band exposure multiplier

    # ── XGBoost ───────────────────────────────────────────────
    xgb_max_depth:        int   = 4
    xgb_n_estimators:     int   = 400
    xgb_learning_rate:    float = 0.03
    xgb_subsample:        float = 0.9
    xgb_colsample_bytree: float = 0.9
    xgb_reg_lambda:       float = 1.0
    xgb_min_child_weight: float = 5.0

    # ── RCRE (Regime-Conditional Reliability Estimator) ───────
    n_regimes:      int   = 3      # calm / transitional / crisis
    mixing_window:  int   = 10     # soft-mixing lookback W
    mixing_tau:     float = 3.0    # Gaussian kernel bandwidth τ
    cusum_h:        float = 4.0    # CUSUM detection threshold