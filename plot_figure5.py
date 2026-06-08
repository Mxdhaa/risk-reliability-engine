import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif']

import matplotlib.gridspec as gridspec
import os

ASSETS = {
    "GSPC":  "S&P 500",
    "FTSE":  "FTSE 100",
    "N225":  "Nikkei 225",
    "GDAXI": "DAX",
    "HSI":   "Hang Seng",
    "EEM":   "EM ETF",
}

# Real values from the pipeline execution (rounded to nearest integer)
ESB_REDUCTIONS = {
    "GSPC": 51, "FTSE": 50, "N225": 56,
    "GDAXI": 49, "HSI": 69, "EEM": 51,
}

fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    "RCRE Reliability Score Across Six Global Equity Markets",
    fontsize=13, fontweight="bold", y=1.01
)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

for idx, (symbol, name) in enumerate(ASSETS.items()):
    row, col = divmod(idx, 2)
    ax = fig.add_subplot(gs[row, col])

    fpath = f"artifacts/preds_{symbol}.csv"
    if not os.path.exists(fpath):
        ax.text(0.5, 0.5, f"No data:\n{fpath}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="red")
        ax.set_title(name)
        continue

    df = pd.read_csv(fpath, parse_dates=["date"])

    # Threshold values
    tau_low  = 0.019
    tau_high = 0.064

    # Reliability score
    ax.plot(df["date"], df["s_rcre"],
            color="#1f77b4", linewidth=0.6, label="RCRE score")

    # Threshold lines
    ax.axhline(tau_low,  color="orange", linestyle="--",
               linewidth=0.8, alpha=0.8)
    ax.axhline(tau_high, color="red",    linestyle="--",
               linewidth=0.8, alpha=0.8)

    # Breach days (vlines capped at 0.3)
    breach_days = df[df["y"] == 1]["date"]
    ax.vlines(breach_days, 0, 0.3,
              color="crimson", linewidth=0.3, alpha=0.3)

    esb_redn = ESB_REDUCTIONS.get(symbol, "?")
    ax.set_title(f"{name}  |  ESB $\\downarrow${esb_redn}\\%",
                 fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("$\\hat{p}_{RCRE}$", fontsize=10)
    ax.tick_params(axis="x", labelsize=9, rotation=30)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Shared legend
handles = [
    plt.Line2D([0], [0], color="#1f77b4", lw=0.6, label="RCRE score"),
    plt.Line2D([0], [0], color="orange",  lw=0.8,
               linestyle="--", label=r"$\tau_{low}=0.019$ (de-risk)"),
    plt.Line2D([0], [0], color="red",     lw=0.8,
               linestyle="--", label=r"$\tau_{high}=0.064$ (halt)"),
    plt.Line2D([0], [0], color="crimson", lw=0.3,
               alpha=0.3, label="Actual breach day"),
]
fig.legend(handles=handles, loc="lower center",
           ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.03))

plt.savefig("artifacts/fig5_multiasset_scores.png",
            dpi=150, bbox_inches="tight")
print("Saved: artifacts/fig5_multiasset_scores.png")