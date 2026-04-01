import os
import pandas as pd
from config import Config
from data_factory import fetch_ohlcv, compute_returns
from risk_models import rolling_historical_var, ewma_vol
from features import build_reliability_features
from reliability import make_labels_next_day, train_reliability_xgb, predict_calibrated
from baselines import run_all_baselines
from regime_detector import RegimeDetector
from rcre import RCREModel
from backtest import gate_policy, compute_metrics_v2
from ablation import run_ablation
from viz import killer_plot
from logger import save_json, save_text, make_results_latex


def main():
    cfg = Config()

    # ── 1. Data ──────────────────────────────────────────────
    df = compute_returns(fetch_ohlcv(cfg.symbol, cfg.start, cfg.end))
    r = df["logret"].astype(float)
    r_delayed = r.shift(cfg.delay)

    df["var_base"]  = rolling_historical_var(r_delayed, cfg.alpha, cfg.window)
    df["sigma_hat"] = ewma_vol(r_delayed.fillna(0.0))

    X = build_reliability_features(
        r, df["var_base"], df["sigma_hat"],
        cfg.exceed_k, cfg.kurt_window, cfg.vol_mom_lag
    )
    y = make_labels_next_day(r, df["var_base"], cfg.gamma)

    os.makedirs("artifacts", exist_ok=True)

    # ── 2. XGBoost + Isotonic Calibration (existing model) ───
    print("Training XGBoost reliability monitor...")
    model, calibrator, fit_metrics, splits = train_reliability_xgb(X, y, cfg)
    X_tr, X_va, X_te, y_tr, y_va, y_te = splits

    s = predict_calibrated(model, calibrator, X)

    # ── 3. Baselines ─────────────────────────────────────────
    print("Training baselines...")
    baseline_results, lr_model, vt_model = run_all_baselines(X, y, df["sigma_hat"], cfg)
    save_json("artifacts/baseline_metrics.json", baseline_results)

    # ── 4. Regime Detector ───────────────────────────────────
    print("Fitting regime detector...")
    regime_detector = RegimeDetector(
        n_regimes=cfg.n_regimes,
        vol_series=df["sigma_hat"],
        train_ratio=cfg.train_ratio,
    )
    regime_detector.fit()
    regimes = regime_detector.predict(df["sigma_hat"])
    df["regime"] = regimes

    # ── 5. RCRE (Novel Algorithm) ────────────────────────────
    print("Training RCRE model (novel contribution)...")
    rcre = RCREModel(n_regimes=cfg.n_regimes, cfg=cfg)
    rcre.fit(X, y, regimes)
    s_rcre = rcre.predict(X, regimes)
    rcre_metrics = rcre.evaluate(X, y, regimes, cfg)
    save_json("artifacts/rcre_metrics.json", rcre_metrics)

    # ── 6. Feature Ablation ──────────────────────────────────
    print("Running feature ablation...")
    ablation_results = run_ablation(X, y, cfg)
    save_json("artifacts/ablation_metrics.json", ablation_results)

    # ── 7. Save predictions (XGBoost) ────────────────────────
    Xn = X.dropna()
    preds_df = pd.DataFrame({
        "t":      Xn.index,
        "s_xgb":  s.loc[Xn.index].values,
        "s_rcre": s_rcre.reindex(Xn.index).values,
        "y":      y.loc[Xn.index].values,
        "L":      r.loc[Xn.index].values,
        "R":      df.loc[Xn.index, "var_base"].values,
        "regime": regimes.reindex(Xn.index).values,
    })
    preds_df.to_csv("artifacts/preds.csv", index=False)
    print(f"Wrote artifacts/preds.csv  {preds_df.shape}")

    # ── 8. Gating Policy ─────────────────────────────────────
    # Evaluate both XGBoost-gated and RCRE-gated policies
    df["s_score"]    = s
    df["s_score_rcre"] = s_rcre
    df["pi_xgb"]  = gate_policy(df["s_score"],      cfg.tau_low, cfg.tau_high, cfg.phi)
    df["pi_rcre"] = gate_policy(df["s_score_rcre"],  cfg.tau_low, cfg.tau_high, cfg.phi)

    n = len(df)
    test_start = int(n * (cfg.train_ratio + cfg.val_ratio))
    test_df = df.iloc[test_start:].copy()

    decision_metrics_xgb = compute_metrics_v2(
        test_df["logret"], test_df["var_base"], test_df["pi_xgb"],  cfg.alpha
    )
    decision_metrics_rcre = compute_metrics_v2(
        test_df["logret"], test_df["var_base"], test_df["pi_rcre"], cfg.alpha
    )

    save_json("artifacts/fit_metrics.json",          fit_metrics)
    save_json("artifacts/decision_metrics_xgb.json", decision_metrics_xgb)
    save_json("artifacts/decision_metrics_rcre.json", decision_metrics_rcre)

    # ── 9. Results Table ─────────────────────────────────────
    rows = [
        {
            "model":        "VolThreshold (baseline)",
            "auc_test":     baseline_results["VolThreshold"]["auc_test"],
            "prauc_test":   baseline_results["VolThreshold"]["prauc_test"],
            "brier_test":   "—",
            "esb_base":     "—",
            "esb_gated":    "—",
            "avg_exposure": "—",
        },
        {
            "model":        "LogisticRegression (baseline)",
            "auc_test":     baseline_results["LogisticRegression"]["auc_test"],
            "prauc_test":   baseline_results["LogisticRegression"]["prauc_test"],
            "brier_test":   baseline_results["LogisticRegression"]["brier_test"],
            "esb_base":     decision_metrics_xgb["esb_base"],   # same for all
            "esb_gated":    "—",
            "avg_exposure": "—",
        },
        {
            "model":        "XGBoost (ours)",
            "auc_test":     fit_metrics["auc_test"],
            "prauc_test":   fit_metrics["prauc_test"],
            "brier_test":   fit_metrics["brier_test"],
            "esb_base":     decision_metrics_xgb["esb_base"],
            "esb_gated":    decision_metrics_xgb["esb_gated"],
            "avg_exposure": decision_metrics_xgb["avg_exposure"],
        },
        {
            "model":        "RCRE (ours, novel)",
            "auc_test":     rcre_metrics["auc_test"],
            "prauc_test":   rcre_metrics["prauc_test"],
            "brier_test":   rcre_metrics["brier_test"],
            "esb_base":     decision_metrics_rcre["esb_base"],
            "esb_gated":    decision_metrics_rcre["esb_gated"],
            "avg_exposure": decision_metrics_rcre["avg_exposure"],
        },
    ]

    tex = make_results_latex(
        rows,
        caption="Discrimination, calibration, and operational metrics on the test block.",
        label="tab:main_results"
    )
    save_text("artifacts/results_table.tex", tex)

    # ── 10. Figures ──────────────────────────────────────────
    killer_plot(
        df,
        start="2020-02-01",
        end="2020-05-15",
        tau_high=cfg.tau_high,
        title="COVID Crash: Reliability detects VaR breakdown"
    )

    print("\n=== RESULTS SUMMARY ===")
    print(f"XGBoost  — AUC: {fit_metrics['auc_test']:.3f}  PR-AUC: {fit_metrics['prauc_test']:.3f}  Brier: {fit_metrics['brier_test']:.4f}")
    print(f"RCRE     — AUC: {rcre_metrics['auc_test']:.3f}  PR-AUC: {rcre_metrics['prauc_test']:.3f}  Brier: {rcre_metrics['brier_test']:.4f}")
    print(f"LR       — AUC: {baseline_results['LogisticRegression']['auc_test']:.3f}")
    print(f"VolThresh— AUC: {baseline_results['VolThreshold']['auc_test']:.3f}")
    print(f"XGBoost ESB:  {decision_metrics_xgb['esb_base']:.4f} → {decision_metrics_xgb['esb_gated']:.4f}")
    print(f"RCRE    ESB:  {decision_metrics_rcre['esb_base']:.4f} → {decision_metrics_rcre['esb_gated']:.4f}")

if __name__ == "__main__":
    main()