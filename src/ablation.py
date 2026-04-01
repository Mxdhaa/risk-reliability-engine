"""
ablation.py
-----------
Leave-one-out feature ablation for the reliability classifier.

For each feature in the feature map, we:
  1. Drop that feature from X
  2. Re-train XGBoost with identical hyperparameters
  3. Record ROC-AUC and PR-AUC on the test block
  4. Report ΔAUC = AUC_full − AUC_ablated

A large ΔAUC means the feature carries non-redundant signal.
This directly answers the reviewer question: "which features matter?"
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from reliability import chrono_split, isotonic_calibrate


def _train_and_eval(X: pd.DataFrame, y: pd.Series, cfg) -> dict:
    """Train XGBoost on X, return test AUC and PR-AUC."""
    pack = pd.concat([X, y.rename("y")], axis=1).dropna()
    y_c  = pack.pop("y").astype(int)
    X_c  = pack

    full = pd.concat([X_c, y_c.rename("y")], axis=1)
    train_df, val_df, test_df = chrono_split(full, cfg.train_ratio, cfg.val_ratio)

    X_tr, y_tr = train_df.drop(columns=["y"]), train_df["y"].astype(int)
    X_va, y_va = val_df.drop(columns=["y"]),   val_df["y"].astype(int)
    X_te, y_te = test_df.drop(columns=["y"]),  test_df["y"].astype(int)

    pos = int(y_tr.sum()); neg = int(len(y_tr) - pos)
    spw = neg / max(pos, 1)

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
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]

    # Isotonic calibration on val
    cal = isotonic_calibrate(p_va, y_va.values)
    s_te = cal.predict(p_te)

    auc   = float(roc_auc_score(y_te, s_te))   if y_te.nunique() > 1 else float("nan")
    prauc = float(average_precision_score(y_te, s_te)) if y_te.nunique() > 1 else float("nan")
    return {"auc": auc, "prauc": prauc}


def run_ablation(X: pd.DataFrame, y: pd.Series, cfg) -> list[dict]:
    """
    Leave-one-out ablation over all features in X.

    Returns a list of dicts (one per feature + one for the full model),
    sorted by ΔAUC descending — ready to paste into a LaTeX table.
    """
    features = list(X.columns)

    print("Ablation: training full model...")
    full_metrics = _train_and_eval(X, y, cfg)
    full_auc   = full_metrics["auc"]
    full_prauc = full_metrics["prauc"]

    rows = [{
        "feature_removed": "None (full model)",
        "auc":   round(full_auc,   3),
        "prauc": round(full_prauc, 3),
        "delta_auc":   0.0,
        "delta_prauc": 0.0,
    }]

    for feat in features:
        print(f"  Ablating: {feat}")
        X_ablated = X.drop(columns=[feat])
        m = _train_and_eval(X_ablated, y, cfg)
        rows.append({
            "feature_removed": feat,
            "auc":   round(m["auc"],   3),
            "prauc": round(m["prauc"], 3),
            "delta_auc":   round(full_auc   - m["auc"],   3),
            "delta_prauc": round(full_prauc - m["prauc"], 3),
        })

    # Sort by importance (largest ΔAUC first), keep full model at top
    header = rows[:1]
    body   = sorted(rows[1:], key=lambda r: r["delta_auc"], reverse=True)
    return header + body