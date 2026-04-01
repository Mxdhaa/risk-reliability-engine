"""
shap_analysis.py
----------------
SHAP (SHapley Additive exPlanations) analysis for the XGBoost reliability
monitor. Generates four outputs:

  1. SHAP summary plot (beeswarm)  — overall feature importance
  2. SHAP bar plot                 — mean |SHAP| per feature
  3. SHAP dependence plot          — risk_mismatch vs SHAP value
  4. artifacts/shap_values.csv     — raw SHAP values for audit

Why SHAP matters here:
  - Ablation (leave-one-out) tells you WHICH features matter globally.
  - SHAP tells you HOW each feature affects each individual prediction.
  - For a risk system, explainability is not optional — regulators and
    risk managers need to understand why the monitor fires.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import shap
import os

from config import Config
from data_factory import fetch_ohlcv, compute_returns
from risk_models import rolling_historical_var, ewma_vol
from features import build_reliability_features
from reliability import make_labels_next_day, train_reliability_xgb, predict_calibrated


def run_shap_analysis(save_dir: str = "artifacts") -> dict:
    """
    Full SHAP analysis pipeline.
    Returns dict with shap_values array and feature names.
    """
    os.makedirs(save_dir, exist_ok=True)
    cfg = Config()

    # ── 1. Data ──────────────────────────────────────────────
    print("Fetching data...")
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

    # ── 2. Train model ───────────────────────────────────────
    print("Training XGBoost...")
    model, calibrator, fit_metrics, splits = train_reliability_xgb(X, y, cfg)
    X_tr, X_va, X_te, y_tr, y_va, y_te = splits

    # ── 3. SHAP on test set ──────────────────────────────────
    print("Computing SHAP values on test set...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te)

    # Feature display names — cleaner for plots
    feature_names = {
        "sigma_hat":     "EWMA Volatility (σ̂)",
        "vol_mom":       "Volatility Momentum (Δσ̂)",
        "risk_mismatch": "Risk–Vol Divergence (R̂−σ̂)",
        "loss_minus_risk": "Loss–Risk Divergence (L−R̂)",
        "abs_loss":      "Absolute Loss (|L|)",
    }
    display_names = [feature_names.get(c, c) for c in X_te.columns]

    # ── 4. Summary Plot (beeswarm) ───────────────────────────
    print("Generating SHAP summary plot...")
    fig, ax = plt.subplots(figsize=(9, 5))
    shap.summary_plot(
        shap_values, X_te,
        feature_names=display_names,
        show=False,
        plot_size=None,
        color_bar_label="Feature value"
    )
    plt.title("SHAP Summary: Feature Impact on Failure Probability",
              fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(f"{save_dir}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir}/shap_summary.png")

    # ── 5. Bar Plot (mean |SHAP|) ────────────────────────────
    print("Generating SHAP bar plot...")
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#C0392B" if i == 0 else "#2980B9" for i in range(len(order))]
    ax.barh(
        [display_names[i] for i in order[::-1]],
        mean_abs[order[::-1]],
        color=colors[::-1],
        edgecolor="white", linewidth=0.5
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Feature Importance (Mean Absolute SHAP Value)",
                 fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{save_dir}/shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_dir}/shap_bar.png")

    # ── 6. Dependence Plot: risk_mismatch ────────────────────
    print("Generating SHAP dependence plot for risk_mismatch...")
    if "risk_mismatch" in X_te.columns:
        feat_idx = list(X_te.columns).index("risk_mismatch")
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(
            X_te["risk_mismatch"].values,
            shap_values[:, feat_idx],
            c=X_te["sigma_hat"].values,
            cmap="RdYlBu_r",
            alpha=0.5,
            s=12,
            rasterized=True
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("EWMA Volatility (σ̂)", fontsize=10)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Risk–Volatility Divergence (R̂ − σ̂)", fontsize=11)
        ax.set_ylabel("SHAP value", fontsize=11)
        ax.set_title(
            "SHAP Dependence: Risk–Vol Divergence\n"
            "(colour = EWMA volatility level)",
            fontsize=13, fontweight="bold"
        )
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        fig.savefig(f"{save_dir}/shap_dependence_risk_mismatch.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_dir}/shap_dependence_risk_mismatch.png")

    # ── 7. COVID window SHAP ─────────────────────────────────
    print("Generating COVID stress-period SHAP plot...")
    covid_mask = (X_te.index >= "2020-02-01") & (X_te.index <= "2020-05-15")
    if covid_mask.sum() > 10:
        X_covid   = X_te[covid_mask]
        sv_covid  = shap_values[covid_mask.values]

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        # Top: calibrated score over COVID window
        s_all = predict_calibrated(model, calibrator, X)
        s_covid = s_all[covid_mask]
        axes[0].plot(X_covid.index, s_covid.values, color="#C0392B", linewidth=1.5)
        axes[0].axhline(cfg.tau_high, color="gray", linestyle="--",
                        linewidth=0.9, label=f"τ_high={cfg.tau_high}")
        axes[0].set_ylabel("Reliability Score", fontsize=10)
        axes[0].set_title("COVID Crash: SHAP Feature Contributions",
                          fontsize=13, fontweight="bold")
        axes[0].legend(fontsize=9)

        # Bottom: stacked SHAP contributions
        bottom = np.zeros(len(X_covid))
        palette = ["#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6"]
        for i, (col, dname) in enumerate(zip(X_te.columns, display_names)):
            contrib = sv_covid[:, i]
            axes[1].bar(X_covid.index, contrib, bottom=bottom,
                        label=dname, color=palette[i % len(palette)],
                        alpha=0.85, width=1)
            bottom += contrib
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_ylabel("SHAP contribution", fontsize=10)
        axes[1].set_xlabel("Date", fontsize=10)
        axes[1].legend(fontsize=8, loc="upper left", ncol=2)

        plt.tight_layout()
        fig.savefig(f"{save_dir}/shap_covid_contributions.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_dir}/shap_covid_contributions.png")

    # ── 8. Save raw SHAP values ──────────────────────────────
    shap_df = pd.DataFrame(shap_values, columns=display_names, index=X_te.index)
    shap_df.to_csv(f"{save_dir}/shap_values.csv")
    print(f"  Saved: {save_dir}/shap_values.csv")

    # ── 9. Print summary ─────────────────────────────────────
    print("\n=== SHAP FEATURE IMPORTANCE ===")
    for i in order:
        print(f"  {display_names[i]:<35} mean|SHAP| = {mean_abs[i]:.6f}")

    return {
        "shap_values":    shap_values,
        "feature_names":  display_names,
        "mean_abs_shap":  dict(zip(display_names, mean_abs)),
        "explainer":      explainer,
    }


if __name__ == "__main__":
    results = run_shap_analysis()