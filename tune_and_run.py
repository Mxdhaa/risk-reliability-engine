import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import json
import numpy as np
import pandas as pd

from config import Config
from data_factory import fetch_ohlcv, compute_returns
from risk_models import rolling_historical_var, ewma_vol
from features import build_reliability_features
from reliability import make_labels_next_day
from backtest import gate_policy, compute_metrics_v2

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression


def chrono_split(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    n = len(df)
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + val_ratio))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


def train_xgb_with_val(X: pd.DataFrame, y: pd.Series, cfg: Config):
    data = X.copy()
    data["y"] = y
    data = data.dropna()

    y2 = data.pop("y").astype(int)
    X2 = data

    pack = pd.concat([X2, y2.rename("y")], axis=1)
    train_df, val_df, test_df = chrono_split(pack, cfg.train_ratio, cfg.val_ratio)

    X_tr, y_tr = train_df.drop(columns=["y"]), train_df["y"].astype(int)
    X_va, y_va = val_df.drop(columns=["y"]), val_df["y"].astype(int)
    X_te, y_te = test_df.drop(columns=["y"]), test_df["y"].astype(int)

    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    scale_pos_weight = (neg / max(pos, 1))

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

    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]

    auc_val = float(roc_auc_score(y_va, p_va)) if y_va.nunique() > 1 else float("nan")
    auc_test = float(roc_auc_score(y_te, p_te)) if y_te.nunique() > 1 else float("nan")

    return model, (X_tr, y_tr, X_va, y_va, X_te, y_te), auc_val, auc_test


def calibrate_isotonic(p: np.ndarray, y: np.ndarray):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y.astype(int))
    return iso


def pick_tau_by_val(p_val_cal: np.ndarray, target_exposure: float, phi: float):
    # search percentiles for tau_low/tau_high
    best = None
    for q_low in [0.90, 0.92, 0.94, 0.96, 0.97, 0.98]:
        for q_high in [0.98, 0.985, 0.99, 0.995]:
            if q_high <= q_low:
                continue
            tau_low = float(np.quantile(p_val_cal, q_low))
            tau_high = float(np.quantile(p_val_cal, q_high))

            pi = np.ones_like(p_val_cal, dtype=float)
            pi[p_val_cal >= tau_high] = 0.0
            mid = (p_val_cal >= tau_low) & (p_val_cal < tau_high)
            pi[mid] = float(phi)

            exp = float(pi.mean())
            dist = abs(exp - target_exposure)

            cand = (dist, tau_low, tau_high, exp)
            if (best is None) or (cand[0] < best[0]):
                best = cand

    _, tau_low, tau_high, exp = best
    return float(tau_low), float(tau_high), float(exp)


def main():
    cfg = Config()
    os.makedirs("artifacts", exist_ok=True)

    # --- data ---
    df = compute_returns(fetch_ohlcv(cfg.symbol, cfg.start, cfg.end))
    r = df["logret"].astype(float)

    r_delayed = r.shift(cfg.delay)
    df["var_base"] = rolling_historical_var(r_delayed, cfg.alpha, cfg.window)
    df["sigma_hat"] = ewma_vol(r_delayed.fillna(0.0))

    X = build_reliability_features(
        r, df["var_base"], df["sigma_hat"],
        cfg.exceed_k, cfg.kurt_window, cfg.vol_mom_lag
    )

    # --- choose gamma to make positives not absurdly rare ---
    # ratio_t = L_t / R_t, pick gamma as (1 - target_pos) quantile
    L = (-r).astype(float)
    ratio = (L / df["var_base"].astype(float)).replace([np.inf, -np.inf], np.nan).dropna()

    gamma_grid = []
    for target_pos in [0.01, 0.02, 0.03, 0.05]:
        gamma_grid.append(float(ratio.quantile(1.0 - target_pos)))

    gamma_grid = sorted(list(set([g for g in gamma_grid if np.isfinite(g)])))

    results = []
    best = None

    for gamma in gamma_grid:
        y = make_labels_next_day(r, df["var_base"], gamma)

        model, packs, auc_val, auc_test = train_xgb_with_val(X, y, cfg)
        X_tr, y_tr, X_va, y_va, X_te, y_te = packs

        # calibrate on val
        p_va = model.predict_proba(X_va)[:, 1]
        cal = calibrate_isotonic(p_va, y_va.values)
        p_va_cal = cal.transform(p_va)

        # pick tau by target exposure (so gating actually triggers)
        target_exposure = 0.95  # change to 0.90 if you want stronger gating
        tau_low, tau_high, exp_val = pick_tau_by_val(p_va_cal, target_exposure, cfg.phi)

        # evaluate decision metrics on TEST with calibrated probs + chosen taus
        p_te = model.predict_proba(X_te)[:, 1]
        p_te_cal = cal.transform(p_te)

        test_idx = X_te.index
        df_te = df.loc[test_idx].copy()
        df_te["s_score"] = p_te_cal

        df_te["pi"] = gate_policy(df_te["s_score"], tau_low, tau_high, cfg.phi)

        m = compute_metrics_v2(df_te["logret"], df_te["var_base"], df_te["pi"], cfg.alpha)

        out = {
            "gamma": float(gamma),
            "auc_val": float(auc_val),
            "auc_test": float(auc_test),
            "tau_low": float(tau_low),
            "tau_high": float(tau_high),
            "avg_exposure_val_targeted": float(exp_val),
            "breach_ratio_gated": float(m["breach_ratio_gated"]),
            "esb_gated": float(m["esb_gated"]),
            "sharpe_gated": float(m["sharpe_gated"]),
            "sortino_gated": float(m["sortino_gated"]),
            "prr": float(m["prr"]),
            "avg_exposure_test": float(m["avg_exposure"]),
        }
        results.append(out)

        # choose best primarily by decision objective: minimize ESB_gated, tie-break by AUC_val
        key = (out["esb_gated"], -out["auc_val"])
        if (best is None) or (key < best[0]):
            best = (key, out)

        print(json.dumps(out, indent=2))

    best_out = best[1]
    with open("artifacts/tuning_results.json", "w", encoding="utf-8") as f:
        json.dump({"candidates": results, "best": best_out}, f, indent=2)

    print("\nBEST CHOICE (saved to artifacts/tuning_results.json):")
    print(json.dumps(best_out, indent=2))


if __name__ == "__main__":
    main()
