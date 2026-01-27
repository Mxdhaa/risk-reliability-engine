from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    symbol: str = "^GSPC"
    start: str = "2000-01-01"
    end: str | None = None

    # VaR / ES
    alpha: float = 0.99
    window: int = 252
    delay: int = 1

    # Reliability label
    gamma: float = 1.5

    # Features
    exceed_k: int = 20
    kurt_window: int = 60
    vol_mom_lag: int = 5

    # Train/Val/Test split
    train_ratio: float = 0.6
    val_ratio: float = 0.2

    # Gating
    tau_high: float = 0.7
    tau_low: float = 0.3
    phi: float = 0.5

    # XGBoost
    xgb_max_depth: int = 4
    xgb_n_estimators: int = 400
    xgb_learning_rate: float = 0.03
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    xgb_reg_lambda: float = 1.0
    xgb_min_child_weight: float = 5
