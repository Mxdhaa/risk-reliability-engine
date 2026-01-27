import matplotlib.pyplot as plt
import pandas as pd

def killer_plot(df: pd.DataFrame, start: str, end: str, tau_high: float, title: str = "Killer Plot"):
    w = df.loc[start:end].copy()
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(w.index, w["close"])
    ax1.set_title(title)
    ax1.set_ylabel("Price")

    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax2.plot(w.index, w["var_base"])
    ax2.set_ylabel("Baseline VaR (loss)")

    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
    ax3.plot(w.index, w["s_score"])
    ax3.set_ylabel("Reliability score (P[failure])")
    ax3.set_ylim(-0.05, 1.05)

    # Shade "detection zone" where failure probability is high (>= tau_high)
    if "s_score" in w.columns:
        mask = (w["s_score"] >= tau_high).fillna(False)
        if mask.any():
            ax3.fill_between(w.index, 0, 1, where=mask, alpha=0.2)
            ax1.fill_between(w.index, w["close"].min(), w["close"].max(), where=mask, alpha=0.08)
            ax2.fill_between(w.index, w["var_base"].min(), w["var_base"].max(), where=mask, alpha=0.08)

    plt.tight_layout()
    plt.show()
