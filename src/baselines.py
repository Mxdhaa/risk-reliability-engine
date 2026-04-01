"""
baselines.py
------------
Two comparison baselines for the reliability prediction task:

1. LogisticReliability  — L2-regularised logistic regression + isotonic calibration
2. VolThresholdBaseline — simple rule: flag top-decile volatility days as unreliable

Both expose the same interface as the XGBoost monitor so they can be
evaluated identically in run_pipeline.py.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from reliability import chrono_split, isotonic_calibrate


# ─────────────────────────────────────────────────────────────
# 1. Logistic Regression Baseline
# ─────────────────────────────────────────────────────────────

class LogisticReliability:
    """
    L2-regularised logistic regression with isotonic calibration.
    Same chrono split as XGBoost for fair comparison.
    """

    def __init__(self, C: float = 0.1):
        self.C = C
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=C, max_iter=1000, random_state=42, n_jobs=-1)
        self.calibrator: IsotonicRegression | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, cfg) -> dict:
        pack = pd.concat([X, y.rename("y")], axis=1).dropna()
        y_clean = pack.pop("y").astype(int)
        X_clean = pack

        train_df, val_df, test_df = chrono_split(
            pd.concat([X_clean, y_clean.rename("y")], axis=1),
            cfg.train_ratio, cfg.val_ratio
        )
        X_tr, y_tr = train_df.drop(columns=["y"]), train_df["y"].astype(int)
        X_va, y_va = val_df.drop(columns=["y"]),   val_df["y"].astype(int)
        X_te, y_te = test_df.drop(columns=["y"]),  test_df["y"].astype(int)

        # Scale features (important for LR)
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_va_s = self.scaler.transform(X_va)
        X_te_s = self.scaler.transform(X_te)

        self.model.fit(X_tr_s, y_tr)

        p_va = self.model.predict_proba(X_va_s)[:, 1]
        p_te = self.model.predict_proba(X_te_s)[:, 1]

        # Isotonic calibration on validation
        self.calibrator = isotonic_calibrate(p_va, y_va.values)
        s_va = self.calibrator.predict(p_va)
        s_te = self.calibrator.predict(p_te)

        # Store test index for predict()
        self._test_index = X_te.index

        return {
            "model": "LogisticRegression",
            "auc_val":    float(roc_auc_score(y_va, p_va)) if y_va.nunique() > 1 else float("nan"),
            "auc_test":   float(roc_auc_score(y_te, p_te)) if y_te.nunique() > 1 else float("nan"),
            "prauc_val":  float(average_precision_score(y_va, p_va)) if y_va.nunique() > 1 else float("nan"),
            "prauc_test": float(average_precision_score(y_te, p_te)) if y_te.nunique() > 1 else float("nan"),
            "brier_val":  float(brier_score_loss(y_va, s_va)),
            "brier_test": float(brier_score_loss(y_te, s_te)),
        }

    def predict_calibrated(self, X: pd.DataFrame) -> pd.Series:
        Xn = X.dropna()
        Xs = self.scaler.transform(Xn)
        raw = self.model.predict_proba(Xs)[:, 1]
        cal = self.calibrator.predict(raw)
        s = pd.Series(index=X.index, dtype=float)
        s.loc[Xn.index] = cal
        return s


# ─────────────────────────────────────────────────────────────
# 2. Volatility Threshold Baseline
# ─────────────────────────────────────────────────────────────

class VolThresholdBaseline:
    """
    Flags a day as high-unreliability if EWMA volatility exceeds the
    training-set 90th percentile.  Returns a binary {0, 1} score.

    This is the most natural operational heuristic — if reviewers ask
    'why not just use volatility?', this baseline answers that question.
    """

    def __init__(self, quantile: float = 0.90):
        self.quantile = quantile
        self._threshold: float | None = None

    def fit(self, sigma_hat: pd.Series, cfg) -> dict:
        """
        Fit the threshold on the training portion only.
        sigma_hat: full-length EWMA volatility series.
        """
        n = len(sigma_hat)
        i1 = int(n * cfg.train_ratio)
        train_sigma = sigma_hat.iloc[:i1].dropna()
        self._threshold = float(np.quantile(train_sigma, self.quantile))
        return {"vol_threshold": self._threshold, "quantile": self.quantile}

    def predict_score(self, sigma_hat: pd.Series) -> pd.Series:
        """Returns binary series: 1 if vol >= threshold, else 0."""
        if self._threshold is None:
            raise RuntimeError("Call fit() before predict_score().")
        return (sigma_hat >= self._threshold).astype(float)

    def evaluate(self, sigma_hat: pd.Series, y: pd.Series, cfg) -> dict:
        """Compute discrimination metrics on the test block."""
        n = len(sigma_hat)
        i2 = int(n * (cfg.train_ratio + cfg.val_ratio))
        s_te = self.predict_score(sigma_hat).iloc[i2:]
        y_te = y.iloc[i2:].dropna().astype(int)
        s_te = s_te.reindex(y_te.index)

        if y_te.nunique() < 2:
            return {"model": "VolThreshold", "auc_test": float("nan"), "prauc_test": float("nan")}

        return {
            "model": "VolThreshold",
            "auc_test":   float(roc_auc_score(y_te, s_te)),
            "prauc_test": float(average_precision_score(y_te, s_te)),
            "brier_test": float("nan"),  # not a probability, skip
        }


# ─────────────────────────────────────────────────────────────
# Convenience: run all baselines and return comparison dict
# ─────────────────────────────────────────────────────────────

def run_all_baselines(X: pd.DataFrame, y: pd.Series, sigma_hat: pd.Series, cfg) -> dict:
    """
    Train and evaluate all baselines. Returns a dict of metric dicts
    keyed by model name, ready for table generation.
    """
    results = {}

    # Logistic Regression
    lr = LogisticReliability(C=0.1)
    results["LogisticRegression"] = lr.fit(X, y, cfg)

    # Volatility Threshold
    vt = VolThresholdBaseline(quantile=0.90)
    vt.fit(sigma_hat, cfg)
    results["VolThreshold"] = vt.evaluate(sigma_hat, y, cfg)

    return results, lr, vt