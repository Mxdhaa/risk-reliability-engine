"""
rcre.py — Regime-Conditional Reliability Estimator (RCRE)
----------------------------------------------------------

CORE IDEA
---------
A single classifier trained on all market regimes conflates calm-market
signal with stress-market signal. This is problematic because:

  1. Feature distributions shift dramatically across regimes (non-stationarity).
  2. Isotonic calibration fit on pooled data inherits this distributional mix.
  3. The most safety-critical predictions (crisis regime) are dominated by
     the numerically larger calm-regime training examples.

RCRE addresses this by:

  (a) Partitioning training data into K regime-specific subsets using the
      RegimeDetector's lag-consistent labels.
  (b) Training a separate XGBoost classifier per regime, each with its own
      isotonic calibrator fit on regime-specific validation data.
  (c) At prediction time, using a SOFT MIXING weight w_k(t) that blends
      regime-specific probabilities based on the posterior regime probability
      (smoothed via a rolling transition window).

SOFT MIXING (the theoretical contribution)
------------------------------------------
Hard regime assignment introduces discontinuities at transition boundaries.
RCRE instead computes a soft weight vector w(t) ∈ Δ^{K-1} (probability
simplex) using a rolling Gaussian kernel over recent regime history:

    w_k(t) = Σ_{i=0}^{W-1} K(i) · 𝟙{regime_{t-i} = k}  / Z

where K(i) = exp(-i² / 2τ²) is a recency kernel and Z normalises.
This produces smooth probability transitions and avoids cliff-edge effects.

The final prediction is:

    p̂_RCRE(t) = Σ_k w_k(t) · ŝ_k(φ_t)

where ŝ_k is the calibrated regime-k classifier output.

TEMPORAL CAUSALITY
------------------
- Regime labels used for training are assigned from training-set centroids only.
- Regime labels at prediction time use the same frozen centroids.
- The mixing window W looks only backward (no future regimes used).
- All calibrators are fit on validation data only.

This ensures strict F_t measurability throughout.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from reliability import chrono_split, isotonic_calibrate


class RCREModel:
    """
    Regime-Conditional Reliability Estimator.

    Parameters
    ----------
    n_regimes : int    Number of volatility regimes (must match RegimeDetector).
    cfg       : Config Configuration object.
    mixing_window : int  Lookback window W for soft regime mixing (default 10).
    mixing_tau    : float Gaussian kernel bandwidth τ (default 3.0).
    """

    def __init__(self, n_regimes: int, cfg, mixing_window: int = 10, mixing_tau: float = 3.0):
        self.n_regimes     = n_regimes
        self.cfg           = cfg
        self.mixing_window = mixing_window
        self.mixing_tau    = mixing_tau

        # One classifier + calibrator per regime
        self._models:      list[XGBClassifier]   = []
        self._calibrators: list[IsotonicRegression] = []
        self._fitted = False

        # Precompute Gaussian kernel weights
        w = np.array([np.exp(-(i**2) / (2 * mixing_tau**2)) for i in range(mixing_window)])
        self._kernel = w / w.sum()   # shape (mixing_window,)

    # ── internal helpers ───────────────────────────────────────────────────

    def _regime_mask(self, regimes: pd.Series, k: int) -> pd.Series:
        return (regimes == k)

    def _soft_weights(self, regimes: pd.Series) -> pd.DataFrame:
        """
        Compute soft mixing weights w_k(t) for every t using backward kernel.
        Returns DataFrame of shape (T, K) with rows summing to 1.
        """
        T = len(regimes)
        W = self.mixing_window
        weights = np.zeros((T, self.n_regimes))

        reg_arr = regimes.values.astype(int)

        for t in range(T):
            counts = np.zeros(self.n_regimes)
            for lag in range(min(W, t + 1)):
                k = reg_arr[t - lag]
                counts[k] += self._kernel[lag]
            total = counts.sum()
            weights[t] = counts / total if total > 0 else np.ones(self.n_regimes) / self.n_regimes

        return pd.DataFrame(weights, index=regimes.index,
                            columns=[f"w_{k}" for k in range(self.n_regimes)])

    # ── public API ─────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series, regimes: pd.Series) -> "RCREModel":
        """
        Train one XGBoost + isotonic calibrator per regime.

        Parameters
        ----------
        X       : feature DataFrame (full timeline)
        y       : next-day failure labels (full timeline)
        regimes : integer regime labels from RegimeDetector.predict()
        """
        cfg = self.cfg
        self._models      = []
        self._calibrators = []

        # Global chrono split indices (same as XGBoost baseline for fairness)
        pack = pd.concat([X, y.rename("y"), regimes.rename("regime")], axis=1).dropna()
        n = len(pack)
        i1 = int(n * cfg.train_ratio)
        i2 = int(n * (cfg.train_ratio + cfg.val_ratio))

        pack_tr  = pack.iloc[:i1]
        pack_va  = pack.iloc[i1:i2]

        for k in range(self.n_regimes):
            # Training data for regime k
            tr_k = pack_tr[pack_tr["regime"] == k]
            va_k = pack_va[pack_va["regime"] == k]

            X_tr_k = tr_k.drop(columns=["y", "regime"])
            y_tr_k = tr_k["y"].astype(int)
            X_va_k = va_k.drop(columns=["y", "regime"])
            y_va_k = va_k["y"].astype(int)

            pos = int(y_tr_k.sum())
            neg = int(len(y_tr_k) - pos)
            spw = neg / max(pos, 1)

            model_k = XGBClassifier(
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
                random_state=42 + k,   # different seed per regime
                n_jobs=-1,
            )

            if len(X_tr_k) < 10 or y_tr_k.nunique() < 2:
                # Insufficient data for this regime — use global fallback
                # We'll fill with a dummy model trained on all training data
                X_tr_all = pack_tr.drop(columns=["y", "regime"])
                y_tr_all = pack_tr["y"].astype(int)
                pos_all = int(y_tr_all.sum()); neg_all = int(len(y_tr_all) - pos_all)
                model_k.set_params(scale_pos_weight=neg_all / max(pos_all, 1))
                model_k.fit(X_tr_all, y_tr_all, verbose=False)
                # Calibrate on all validation data
                X_va_all = pack_va.drop(columns=["y", "regime"])
                y_va_all = pack_va["y"].astype(int)
                p_va_all = model_k.predict_proba(X_va_all)[:, 1]
                cal_k = isotonic_calibrate(p_va_all, y_va_all.values)
            else:
                eval_set = [(X_va_k, y_va_k)] if len(X_va_k) > 0 else []
                model_k.fit(X_tr_k, y_tr_k,
                            eval_set=eval_set if eval_set else None,
                            verbose=False)
                if len(X_va_k) >= 5 and y_va_k.nunique() > 1:
                    p_va_k = model_k.predict_proba(X_va_k)[:, 1]
                    cal_k = isotonic_calibrate(p_va_k, y_va_k.values)
                else:
                    # Fallback: calibrate on all val data
                    X_va_all = pack_va.drop(columns=["y", "regime"])
                    y_va_all = pack_va["y"].astype(int)
                    p_va_all = model_k.predict_proba(X_va_all)[:, 1]
                    cal_k = isotonic_calibrate(p_va_all, y_va_all.values)

            self._models.append(model_k)
            self._calibrators.append(cal_k)

        self._fitted = True
        print(f"RCRE: fitted {self.n_regimes} regime-specific models.")
        return self

    def predict(self, X: pd.DataFrame, regimes: pd.Series) -> pd.Series:
        """
        Predict calibrated failure probabilities using soft regime mixing.

        Returns pd.Series of same index as X.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        Xn = X.dropna()
        soft_w = self._soft_weights(regimes.reindex(X.index).ffill().fillna(0))

        # Get calibrated probability from each regime model
        prob_matrix = np.zeros((len(Xn), self.n_regimes))
        for k in range(self.n_regimes):
            raw_k = self._models[k].predict_proba(Xn)[:, 1]
            prob_matrix[:, k] = self._calibrators[k].predict(raw_k)

        # Soft-mix using weights at each time step
        w_matrix = soft_w.reindex(Xn.index).values   # shape (T_valid, K)
        mixed    = (prob_matrix * w_matrix).sum(axis=1)

        s = pd.Series(index=X.index, dtype=float)
        s.loc[Xn.index] = mixed
        return s

    def evaluate(self, X: pd.DataFrame, y: pd.Series,
                 regimes: pd.Series, cfg) -> dict:
        """
        Compute discrimination and calibration metrics on the test block.
        """
        pack = pd.concat([X, y.rename("y"), regimes.rename("regime")], axis=1).dropna()
        n = len(pack)
        i2 = int(n * (cfg.train_ratio + cfg.val_ratio))
        test_pack = pack.iloc[i2:]

        X_te  = test_pack.drop(columns=["y", "regime"])
        y_te  = test_pack["y"].astype(int)
        reg_te = test_pack["regime"]

        s_te = self.predict(X_te, reg_te)
        s_te = s_te.reindex(y_te.index).dropna()
        y_te = y_te.reindex(s_te.index)

        if y_te.nunique() < 2:
            return {"model": "RCRE", "auc_test": float("nan"),
                    "prauc_test": float("nan"), "brier_test": float("nan")}

        return {
            "model":      "RCRE",
            "auc_test":   float(roc_auc_score(y_te,  s_te)),
            "prauc_test": float(average_precision_score(y_te, s_te)),
            "brier_test": float(brier_score_loss(y_te, s_te)),
        }

    def regime_specific_auc(self, X: pd.DataFrame, y: pd.Series,
                             regimes: pd.Series, cfg) -> pd.DataFrame:
        """
        Per-regime AUC on the test block.
        Useful for Table: 'RCRE discriminates best in the crisis regime'.
        """
        pack = pd.concat([X, y.rename("y"), regimes.rename("regime")], axis=1).dropna()
        n = len(pack)
        i2 = int(n * (cfg.train_ratio + cfg.val_ratio))
        test_pack = pack.iloc[i2:]

        rows = []
        for k in range(self.n_regimes):
            sub = test_pack[test_pack["regime"] == k]
            if len(sub) < 10 or sub["y"].nunique() < 2:
                rows.append({"regime": k, "n_days": len(sub), "auc": float("nan")})
                continue
            X_k = sub.drop(columns=["y", "regime"])
            y_k = sub["y"].astype(int)
            s_k = self.predict(X_k, sub["regime"])
            s_k = s_k.reindex(y_k.index).dropna()
            y_k = y_k.reindex(s_k.index)
            rows.append({
                "regime": k,
                "n_days": len(sub),
                "auc": float(roc_auc_score(y_k, s_k)) if y_k.nunique() > 1 else float("nan"),
            })
        return pd.DataFrame(rows)