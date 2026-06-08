import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif']
matplotlib.rcParams['font.weight'] = 'normal'
matplotlib.rcParams['axes.labelweight'] = 'normal'
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11

import matplotlib.gridspec as gridspec
import numpy as np

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
    "GSPC": 50, "FTSE": 50, "N225": 56,
    "GDAXI": 49, "HSI": 69, "EEM": 53,
}

gamma    = 0.8
tau_low  = 0.019
tau_high = 0.064
window   = 63

def get_pi(score):
    if score >= tau_high: return 0.0
    if score >= tau_low:  return 0.5
    return 1.0

def rolling_esb(loss, viol, w):
    esb = []
    for i in range(len(loss)):
        if i < w:
            esb.append(np.nan)
        else:
            sl = loss.iloc[i-w:i]
            sv = viol.iloc[i-w:i]
            breaches = sl[sv == 1]
            esb.append(breaches.mean() if len(breaches) > 0 else np.nan)
    return pd.Series(esb, index=loss.index)

fig = plt.figure(figsize=(14, 10))
fig.suptitle(
    "Aperture Effect: ESB Reduction Across Six Global Equity Markets",
    fontsize=15, fontweight="bold", y=1.01
)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

for idx, (symbol, name) in enumerate(ASSETS.items()):
    row, col = divmod(idx, 2)
    ax = fig.add_subplot(gs[row, col])
    ax2 = ax.twinx()

    fpath = f"artifacts/preds_{symbol}.csv"
    df = pd.read_csv(fpath, parse_dates=["date"])

    df["pi_rcre"] = df["s_rcre"].apply(get_pi)
    df["lambda_gated"] = gamma * df["pi_rcre"] * df["var_base"]
    df["lambda_base"]  = gamma * df["var_base"]
    df["viol_base"]    = (df["loss"] > df["lambda_base"]).astype(float)
    df["viol_gated"]   = (df["loss"] > df["lambda_gated"]).astype(float)

    df["esb_base_roll"]  = rolling_esb(df["loss"], df["viol_base"],  window)
    df["esb_gated_roll"] = rolling_esb(df["loss"], df["viol_gated"], window)
    df["exp_roll"]       = df["pi_rcre"].rolling(window).mean()

    df_plot = df.dropna(subset=["esb_base_roll"])

    # ESB lines on primary axis (made bolder)
    ax.plot(df_plot["date"], df_plot["esb_base_roll"],
            color="steelblue", lw=1.8, label="No gating")
    ax.plot(df_plot["date"], df_plot["esb_gated_roll"],
            color="crimson", lw=1.8, linestyle="--", label="RCRE-gated")

    # Exposure shaded on secondary axis (more prominent shading)
    ax2.fill_between(df_plot["date"], df_plot["exp_roll"],
                     alpha=0.28, color="green")
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel("Avg Exposure", fontsize=12, color="green")
    ax2.tick_params(axis="y", labelsize=11, colors="green")

    esb_redn = ESB_REDUCTIONS.get(symbol, "?")
    ax.set_title(f"{name}  |  ESB $\\downarrow${esb_redn}\\%",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("ESB (63d rolling)", fontsize=12)
    ax.tick_params(axis="x", labelsize=11, rotation=30)
    ax.tick_params(axis="y", labelsize=11)
    ax.spines["top"].set_visible(False)

    if idx == 0:
        ax.legend(fontsize=10, loc="upper left")

fig.text(0.5, -0.02,
         "Blue = no gating ESB. Red dashed = RCRE-gated ESB. "
         "Green shading = average exposure.",
         ha="center", fontsize=8, color="gray")

plt.savefig("artifacts/fig6_aperture_all_assets.png",
            dpi=150, bbox_inches="tight")
print("Saved: artifacts/fig6_aperture_all_assets.png")