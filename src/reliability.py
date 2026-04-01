import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def make_labels_next_day(returns: pd.Series, reported_risk: pd.Series, gamma: float) -> pd.Series:
    L = -returns.astype(float)
    y_t = (L > gamma * reported_risk).astype(int)
    return y_t.shift(-1)


def chrono_split(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    n = len(df)
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + val_ratio))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def isotonic_calibrate(raw_probs: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression calibrator on validation data."""
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(raw_probs, y_true)
    return ir


def train_reliability_xgb(X: pd.DataFrame, y: pd.Series, cfg):
    data = X.copy()
    data["y"] = y
    data = data.dropna()
    y = data.pop("y").astype(int)
    X = data

    pack = pd.concat([X, y.rename("y")], axis=1)
    train_df, val_df, test_df = chrono_split(pack, cfg.train_ratio, cfg.val_ratio)

    X_tr, y_tr = train_df.drop(columns=["y"]), train_df["y"].astype(int)
    X_va, y_va = val_df.drop(columns=["y"]), val_df["y"].astype(int)
    X_te, y_te = test_df.drop(columns=["y"]), test_df["y"].astype(int)

    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    scale_pos_weight = neg / max(pos, 1)

    model = XGBClassifier(
        n_estimators=cfg.xgb_n_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample_bytree,
        reg_lambda=cfg.xgb_reg_lambda,
        min_child_weight=cfg.xgb_min_child_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    # Raw probabilities
    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]

    # --- ISOTONIC CALIBRATION (fit on val, apply to test) ---
    calibrator = isotonic_calibrate(p_va, y_va.values)
    s_va = calibrator.predict(p_va)
    s_te = calibrator.predict(p_te)

    metrics = {
        # Discrimination
        "auc_val":  float(roc_auc_score(y_va, p_va)) if y_va.nunique() > 1 else float("nan"),
        "auc_test": float(roc_auc_score(y_te, p_te)) if y_te.nunique() > 1 else float("nan"),
        "prauc_val":  float(average_precision_score(y_va, p_va)) if y_va.nunique() > 1 else float("nan"),
        "prauc_test": float(average_precision_score(y_te, p_te)) if y_te.nunique() > 1 else float("nan"),
        # Calibration quality (on calibrated scores)
        "brier_val":  float(brier_score_loss(y_va, s_va)),
        "brier_test": float(brier_score_loss(y_te, s_te)),
        # Split sizes
        "n_train": int(len(X_tr)),
        "n_val":   int(len(X_va)),
        "n_test":  int(len(X_te)),
        "pos_rate_train": float(y_tr.mean()),
        "pos_rate_val":   float(y_va.mean()),
        "pos_rate_test":  float(y_te.mean()),
    }

    return model, calibrator, metrics, (X_tr, X_va, X_te, y_tr, y_va, y_te)


def predict_calibrated(model, calibrator, X: pd.DataFrame) -> pd.Series:
    """Return calibrated failure probabilities for any feature DataFrame."""
    Xn = X.dropna()
    raw = model.predict_proba(Xn)[:, 1]
    cal = calibrator.predict(raw)
    s = pd.Series(index=X.index, dtype=float)
    s.loc[Xn.index] = cal
    return s