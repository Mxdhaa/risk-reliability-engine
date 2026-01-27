import os
import pandas as pd

from config import Config
from data_factory import fetch_ohlcv, compute_returns
from risk_models import rolling_historical_var, ewma_vol
from features import build_reliability_features
from reliability import make_labels_next_day, train_reliability_xgb
from backtest import gate_policy, compute_metrics_v2
from viz import killer_plot
from logger import save_json, save_text, make_results_latex


def main():
    cfg = Config()

    df = compute_returns(fetch_ohlcv(cfg.symbol, cfg.start, cfg.end))
    r = df["logret"].astype(float)
    r_delayed = r.shift(cfg.delay)

    df["var_base"] = rolling_historical_var(r_delayed, cfg.alpha, cfg.window)
    df["sigma_hat"] = ewma_vol(r_delayed.fillna(0.0))

    X = build_reliability_features(
        r, df["var_base"], df["sigma_hat"],
        cfg.exceed_k, cfg.kurt_window, cfg.vol_mom_lag
    )
    y = make_labels_next_day(r, df["var_base"], cfg.gamma)

    model, fit_metrics = train_reliability_xgb(X, y, cfg)

    Xn = X.dropna()
    s = pd.Series(index=X.index, dtype=float)
    s.loc[Xn.index] = model.predict_proba(Xn)[:, 1]

    os.makedirs("artifacts", exist_ok=True)

    preds_df = pd.DataFrame({
        "t": Xn.index,
        "p": s.loc[Xn.index].values,
        "y": y.loc[Xn.index].values,
        "L": r.loc[Xn.index].values,
        "R": df.loc[Xn.index, "var_base"].values,
    })
    preds_df.to_csv("artifacts/preds.csv", index=False)
    print("Wrote artifacts/preds.csv", preds_df.shape, list(preds_df.columns))

    df["s_score"] = s
    df["pi"] = gate_policy(df["s_score"], cfg.tau_low, cfg.tau_high, cfg.phi)

    n = len(df)
    test_start = int(n * (cfg.train_ratio + cfg.val_ratio))
    test_df = df.iloc[test_start:].copy()

    decision_metrics = compute_metrics_v2(
        test_df["logret"],
        test_df["var_base"],
        test_df["pi"],
        cfg.alpha
    )

    save_json("artifacts/fit_metrics.json", fit_metrics)
    save_json("artifacts/decision_metrics.json", decision_metrics)

    rows = [{
        "model": "HS-VaR + Reliability",
        "auc_test": fit_metrics["auc_test"],
        "br_base": decision_metrics["breach_ratio_base"],
        "br_gated": decision_metrics["breach_ratio_gated"],
        "esb_base": decision_metrics["esb_base"],
        "esb_gated": decision_metrics["esb_gated"],
        "sharpe_base": decision_metrics["sharpe_base"],
        "sharpe_gated": decision_metrics["sharpe_gated"],
        "sortino_base": decision_metrics["sortino_base"],
        "sortino_gated": decision_metrics["sortino_gated"],
        "prr": decision_metrics["prr"],
        "avg_exposure": decision_metrics["avg_exposure"],
    }]

    tex = make_results_latex(
        rows,
        caption="Reliability-gated decision utility on the test set.",
        label="tab:decision_utility"
    )
    save_text("artifacts/results_table.tex", tex)

    killer_plot(
        df,
        start="2020-02-01",
        end="2020-05-15",
        tau_high=cfg.tau_high,
        title="COVID Crash: Reliability detects VaR breakdown"
    )


if __name__ == "__main__":
    main()
