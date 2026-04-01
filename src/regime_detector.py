"""
regime_detector.py
------------------
Volatility-regime segmentation using a simple but theoretically clean
approach: CUSUM-based change-point detection on EWMA volatility, followed
by k-means clustering of rolling volatility statistics into n_regimes states.

Why not HMM?
  - HMM requires hmmlearn, adds a dependency, and its EM fitting can be
    sensitive to initialisation on financial data.
  - The CUSUM + k-means approach is fully reproducible, requires only numpy
    and scikit-learn, and is easier to explain in a paper methods section.

Strict temporal causality:
  - The k-means centroids are fitted ONLY on training data.
  - Regime labels for validation and test data are assigned by nearest-centroid
    lookup — no future information leaks.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    """
    Two-stage regime detector:
      Stage 1 — CUSUM change-point detection on log-volatility.
      Stage 2 — k-means clustering of local volatility statistics into
                n_regimes discrete states (0 = calm, ..., n-1 = crisis).

    Parameters
    ----------
    n_regimes   : int   Number of volatility regimes (default 3).
    vol_series  : pd.Series  Full EWMA volatility series.
    train_ratio : float Fraction used for fitting centroids.
    cusum_h     : float CUSUM decision threshold (in std units).
    """

    def __init__(
        self,
        n_regimes: int,
        vol_series: pd.Series,
        train_ratio: float = 0.6,
        cusum_h: float = 4.0,
    ):
        self.n_regimes   = n_regimes
        self.vol_series  = vol_series.astype(float)
        self.train_ratio = train_ratio
        self.cusum_h     = cusum_h

        self._scaler:  StandardScaler | None = None
        self._kmeans:  KMeans         | None = None
        self._fitted:  bool           = False

    # ── helpers ────────────────────────────────────────────────────────────

    def _make_features(self, vol: pd.Series) -> pd.DataFrame:
        """
        Local volatility statistics used for regime assignment.
        All windows look backward only — no look-ahead.
        """
        df = pd.DataFrame(index=vol.index)
        df["vol_level"]  = vol
        df["vol_5d_ma"]  = vol.rolling(5,  min_periods=1).mean()
        df["vol_20d_ma"] = vol.rolling(20, min_periods=1).mean()
        df["vol_ratio"]  = (vol / vol.rolling(60, min_periods=5).mean().replace(0, np.nan))
        df["log_vol"]    = np.log(vol.clip(lower=1e-8))
        return df.ffill().bfill()

    def _cusum(self, series: np.ndarray) -> np.ndarray:
        """
        Two-sided CUSUM on standardised series.
        Returns array of detected change-point indices.
        """
        mu    = series.mean()
        sigma = series.std() + 1e-8
        z     = (series - mu) / sigma

        S_pos = np.zeros(len(z))
        S_neg = np.zeros(len(z))
        cps   = []

        for i in range(1, len(z)):
            S_pos[i] = max(0.0, S_pos[i-1] + z[i] - 0.5)
            S_neg[i] = max(0.0, S_neg[i-1] - z[i] - 0.5)
            if S_pos[i] > self.cusum_h or S_neg[i] > self.cusum_h:
                cps.append(i)
                S_pos[i] = 0.0
                S_neg[i] = 0.0

        return np.array(cps, dtype=int)

    # ── public API ─────────────────────────────────────────────────────────

    def fit(self) -> "RegimeDetector":
        """
        Fit scaler and k-means centroids on training data only.
        """
        n_train = int(len(self.vol_series) * self.train_ratio)
        vol_train = self.vol_series.iloc[:n_train].dropna()

        feat_train = self._make_features(vol_train).dropna()

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(feat_train)

        self._kmeans = KMeans(
            n_clusters=self.n_regimes,
            random_state=42,
            n_init=20,
            max_iter=500,
        )
        self._kmeans.fit(X_scaled)

        # Sort cluster labels so that 0 = lowest mean vol, n-1 = highest
        centers = self._kmeans.cluster_centers_
        # Use first feature (vol_level) to define ordering
        vol_col_idx = 0
        order = np.argsort(centers[:, vol_col_idx])
        self._label_map = {old: new for new, old in enumerate(order)}
        self._fitted = True

        return self

    def predict(self, vol_series: pd.Series) -> pd.Series:
        """
        Assign regime labels to a (possibly new) volatility series.
        Causality preserved: uses centroids fitted on training data only.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        feat = self._make_features(vol_series.astype(float))
        valid_idx = feat.dropna().index

        X_scaled = self._scaler.transform(feat.loc[valid_idx])
        raw_labels = self._kmeans.predict(X_scaled)
        mapped = np.array([self._label_map[l] for l in raw_labels])

        regimes = pd.Series(index=vol_series.index, dtype=float)
        regimes.loc[valid_idx] = mapped.astype(float)
        regimes = regimes.ffill().fillna(0).astype(int)

        return regimes

    def regime_stats(self, vol_series: pd.Series, y: pd.Series) -> pd.DataFrame:
        """
        Summary statistics per regime — useful for the paper's regime table.
        Returns a DataFrame with one row per regime.
        """
        regimes = self.predict(vol_series)
        rows = []
        for r in range(self.n_regimes):
            mask = (regimes == r)
            rows.append({
                "regime":       r,
                "label":        ["Calm", "Transitional", "Crisis"][r] if self.n_regimes == 3 else str(r),
                "n_days":       int(mask.sum()),
                "mean_vol":     float(vol_series[mask].mean()),
                "failure_rate": float(y.reindex(vol_series.index)[mask].mean()),
            })
        return pd.DataFrame(rows)