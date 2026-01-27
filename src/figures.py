# src/figures.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _ensure_datetime(s):
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.to_datetime(s, errors="coerce")


def make_figure4(
    preds_path="artifacts/preds.csv",
    out_path="artifacts/figure4.png",
    gamma=1.0,
):
    df = pd.read_csv(preds_path)

    t_col = _pick_col(df, ["t", "time", "timestamp", "date", "datetime"])
    p_col = _pick_col(df, ["p", "p_t", "prob", "proba", "reliability_prob", "reliability_probability"])
    L_col = _pick_col(df, ["L", "loss", "realized_loss", "realized_losses", "realized"])
    R_col = _pick_col(df, ["R", "reported_risk", "risk", "var", "VaR", "es", "ES"])

    if p_col is None:
        raise FileNotFoundError(f"[figure4] Could not find probability column in {preds_path}. Columns: {list(df.columns)}")

    if t_col is None:
        t = np.arange(len(df))
    else:
        t = _ensure_datetime(df[t_col])
        if t.isna().all():
            t = np.arange(len(df))

    p = pd.to_numeric(df[p_col], errors="coerce").values

    L = None
    if L_col is not None:
        L = pd.to_numeric(df[L_col], errors="coerce").values

    exceed_mask = np.zeros(len(df), dtype=bool)
    if (L is not None) and (R_col is not None):
        R = pd.to_numeric(df[R_col], errors="coerce").values
        R_lag = np.roll(R, 1)
        R_lag[0] = np.nan
        exceed_mask = np.isfinite(L) & np.isfinite(R_lag) & (L > (gamma * R_lag))
    else:
        y_col = _pick_col(df, ["y", "label", "unreliable", "is_unreliable", "target"])
        if y_col is not None:
            ytmp = pd.to_numeric(df[y_col], errors="coerce").values
            exceed_mask = np.isfinite(ytmp) & (ytmp > 0.5)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig = plt.figure(figsize=(12.0, 5.2))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(t, p, linewidth=1.8)
    ax1.set_ylabel("Reliability probability $p_t$")
    ax1.set_ylim(0, 1)

    if L is not None:
        ax2.plot(t, L, linewidth=1.2, alpha=0.9)
        ax2.set_ylabel("Realized loss $L_t$")
        if exceed_mask.any():
            ax2.scatter(np.array(t)[exceed_mask], L[exceed_mask], s=18, c="red", zorder=5)

    ax1.set_title("Figure 4: Reliability score vs realized loss over time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def make_figure5(
    preds_path="artifacts/preds.csv",
    out_path="artifacts/figure5.png",
    gamma=1.0,
    n_bins=10,
):
    df = pd.read_csv(preds_path)

    p_col = _pick_col(df, ["p", "p_t", "prob", "proba", "reliability_prob", "reliability_probability"])
    if p_col is None:
        raise FileNotFoundError(f"[figure5] Could not find probability column in {preds_path}. Columns: {list(df.columns)}")

    p = pd.to_numeric(df[p_col], errors="coerce").values

    y_col = _pick_col(df, ["y", "label", "unreliable", "is_unreliable", "target"])
    if y_col is not None:
        y_raw = pd.to_numeric(df[y_col], errors="coerce").values
        y = (y_raw > 0.5).astype(float)
    else:
        L_col = _pick_col(df, ["L", "loss", "realized_loss", "realized_losses", "realized"])
        R_col = _pick_col(df, ["R", "reported_risk", "risk", "var", "VaR", "es", "ES"])
        if (L_col is None) or (R_col is None):
            raise FileNotFoundError(
                "[figure5] Need either a label column (y/label/unreliable/...) OR both realized loss (L) and risk (R/var/es) in preds.csv."
            )
        L = pd.to_numeric(df[L_col], errors="coerce").values
        R = pd.to_numeric(df[R_col], errors="coerce").values
        R_lag = np.roll(R, 1)
        R_lag[0] = np.nan
        y = (np.isfinite(L) & np.isfinite(R_lag) & (L > (gamma * R_lag))).astype(float)

    mask = np.isfinite(p) & np.isfinite(y)
    p = p[mask]
    y = y[mask]

    # Bin edges and centers (centers make a readable curve even if p is concentrated)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2.0

    bin_id = np.digitize(p, bins) - 1
    bin_id = np.clip(bin_id, 0, n_bins - 1)

    mean_p = np.full(n_bins, np.nan)
    mean_y = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        m = (bin_id == b)
        counts[b] = int(m.sum())
        if counts[b] > 0:
            mean_p[b] = float(np.mean(p[m]))
            mean_y[b] = float(np.mean(y[m]))

    # Print a quick bin summary so you can *see* why it looks empty
    print("[figure5] bin counts:", counts.tolist())
    print("[figure5] p range:", float(np.min(p)), "to", float(np.max(p)))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Two-panel: calibration + probability mass
    fig = plt.figure(figsize=(7.2, 7.2))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2)

    ok = np.isfinite(mean_y)
    # Use centers on x so curve is visible even if mean_p collapses near 0
    ax1.plot(centers[ok], mean_y[ok], marker="o", linewidth=1.8)

    for xp, yp, c in zip(centers[ok], mean_y[ok], counts[ok]):
        ax1.annotate(str(int(c)), (xp, yp), textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Predicted unreliability probability (bin center)")
    ax1.set_ylabel("Empirical unreliability rate")
    ax1.set_title("Figure 5: Calibration (reliability diagram)")

    ax2 = plt.subplot(2, 1, 2)
    ax2.bar(centers, counts / max(1, int(np.sum(counts))), width=(1.0 / n_bins) * 0.9)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Predicted probability bins")
    ax2.set_ylabel("Fraction of samples")
    ax2.set_title("Probability mass across bins")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def make_figure3(
    preds_path="artifacts/preds.csv",
    out_path="artifacts/figure3.png",
    vol_window=20,
    stress_quantile=0.90,
    use_abs_returns=True,
    show=False,
):
    df = pd.read_csv(preds_path)

    t_col = _pick_col(df, ["t", "time", "timestamp", "date", "datetime"])
    L_col = _pick_col(df, ["L", "logret", "ret", "return", "returns"])

    if L_col is None:
        raise FileNotFoundError(
            f"[figure3] Could not find returns column. Need one of: L/logret/ret/return/returns. Columns: {list(df.columns)}"
        )

    # time axis
    if t_col is None:
        t = np.arange(len(df))
        t_is_dt = False
    else:
        t = _ensure_datetime(df[t_col])
        if getattr(t, "isna", lambda: False)().all():
            t = np.arange(len(df))
            t_is_dt = False
        else:
            t_is_dt = True

    r = pd.to_numeric(df[L_col], errors="coerce").values
    if use_abs_returns:
        r_for_vol = np.abs(r)
    else:
        r_for_vol = r

    vol = pd.Series(r_for_vol).rolling(vol_window, min_periods=vol_window).std().values

    finite = np.isfinite(vol)
    if not finite.any():
        raise ValueError("[figure3] Rolling volatility is all NaN/inf. Check returns column and vol_window.")

    thr = float(np.nanquantile(vol[finite], stress_quantile))
    stress = finite & (vol >= thr)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fig = plt.figure(figsize=(12.0, 4.8))
    ax = plt.gca()

    # plot only finite points 
    ax.plot(np.array(t)[finite], vol[finite], linewidth=1.6, label=f"Rolling vol (w={vol_window})")
    ax.axhline(thr, linestyle="--", linewidth=1.2, label=f"Stress threshold (q={stress_quantile:.2f})")

    # shade contiguous stress regions
    if stress.any():
        idx = np.where(stress)[0]
        starts = [idx[0]]
        ends = []
        for i in range(1, len(idx)):
            if idx[i] != idx[i - 1] + 1:
                ends.append(idx[i - 1])
                starts.append(idx[i])
        ends.append(idx[-1])

        for a, b in zip(starts, ends):
            ax.axvspan(np.array(t)[a], np.array(t)[b], alpha=0.15)

    ax.set_ylabel(f"Rolling volatility (window={vol_window})")
    ax.set_title("Figure 3: Regime segmentation / stress windows (top-quantile volatility)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)

    print(f"[figure3] wrote {out_path}")
    print(f"[figure3] stress threshold (q={stress_quantile}): {thr}")
    print(f"[figure3] stress points: {int(stress.sum())} / {len(stress)}")

    if show:
        plt.show()
    else:
        plt.close(fig)
