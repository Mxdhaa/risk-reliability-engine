import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif']
matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'

import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

df = pd.read_csv("artifacts/preds_GSPC.csv", parse_dates=["date"])
regime_map = {0: "Calm (k=0)", 1: "Transitional (k=1)", 2: "Crisis (k=2)"}
df["regime_label"] = df["regime"].map(regime_map)
df_slice = df[df["date"] >= "2019-01-01"].copy().reset_index(drop=True)

ROLL = 63
BAND_COLOR = "#C5CAE9"
jet_colors = [cm.jet(0.1), cm.jet(0.55), cm.jet(0.9)]
regime_keys = [0, 1, 2]
regime_labels = ["Calm (k=0)", "Transitional (k=1)", "Crisis (k=2)"]

fig = plt.figure(figsize=(14, 11), facecolor="white")
fig.suptitle(
    "RCRE: Three-Stage Regime-Conditional Reliability Estimation (S&P 500)",
    fontsize=19, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.6, wspace=0.55)

# ── Panel (a): Stage 1 — scatter, small dots ───────────────────────
ax1 = fig.add_subplot(gs[0, 0:3])

for rk, rl, rc in zip(regime_keys, regime_labels, jet_colors):
    mask = df["regime"] == rk
    subset = df[mask]
    ax1.scatter(subset["date"], subset["var_base"],
                color=rc, s=1.5, alpha=0.6, label=rl, zorder=3)

y_roll = df["var_base"].rolling(ROLL, center=True, min_periods=10)
mean = y_roll.mean()
std  = y_roll.std()
ax1.fill_between(df["date"], mean - std, mean + std,
                 color=BAND_COLOR, alpha=0.5, zorder=1)
ax1.plot(df["date"], mean, color="#5C6BC0", lw=1.2, zorder=2)

ax1.set_title("Stage 1: Regime Detection\n"
              "CUSUM + $k$-means on log-volatility",
              fontsize=18, fontweight="bold", pad=8)
ax1.set_xlabel("(a)", fontsize=21, labelpad=8, fontstyle="italic")
ax1.set_ylabel("Baseline HS-VaR $\\hat{R}(t|t{-}1)$", fontsize=19)
ax1.tick_params(axis="x", labelsize=16, rotation=30)
ax1.tick_params(axis="y", labelsize=16)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.legend(fontsize=12.5, markerscale=3.0,
           framealpha=0.5, loc="upper left")

# ── Panel (b): Stage 2 — Raincloud Plot ────────────────────────────
ax2 = fig.add_subplot(gs[0, 3:6])

from scipy.stats import gaussian_kde

regime_keys   = [2, 1, 0]   # top→bottom: Crisis, Transitional, Calm
regime_labels = ["Crisis (k=2)", "Transitional (k=1)", "Calm (k=0)"]
colors      = [cm.jet(0.9),  cm.jet(0.55), cm.jet(0.1)]   # Crisis, Trans., Calm
dark_colors = [cm.jet(0.95), cm.jet(0.60), cm.jet(0.15)]  # slightly darker for boxes

score_col = "s_rcre"
y_positions = [2, 1, 0]        # vertical centres for each regime
kde_height  = 0.55             # max height of KDE fill
box_halves  = {2: 0.09, 1: 0.04, 0: 0.07}   # Crisis box taller, Transitional narrower
bw_methods  = {2: 0.25, 1: 0.18, 0: 0.18}   # per-regime bandwidth
dot_spread  = 0.10             # vertical jitter band for dots

x_min, x_max = 0.0, 0.12
x_grid = np.linspace(x_min, x_max, 400)

for yc, rk, rl, rc, dc in zip(y_positions, regime_keys,
                                regime_labels, colors, dark_colors):
    scores = df[df["regime"] == rk][score_col].dropna().values
    scores = scores[(scores >= x_min) & (scores <= 0.10)]
    n      = len(scores)
    box_half = box_halves[rk]

    # ── KDE (above centre line) ──────────────────────────────────
    kde     = gaussian_kde(scores, bw_method=bw_methods[rk])
    density = kde(x_grid)
    density = density / density.max() * kde_height   # normalise height

    ax2.fill_between(x_grid, yc, yc + density,
                     color=rc, alpha=0.55, zorder=2)
    ax2.plot(x_grid, yc + density,
             color=rc, lw=1.2, zorder=3)

    # ── Boxplot (below centre line) ──────────────────────────────
    q25, q50, q75 = np.percentile(scores, [25, 50, 75])
    iqr   = q75 - q25
    w_lo  = max(scores.min(), q25 - 1.5 * iqr)
    w_hi  = min(scores.max(), q75 + 1.5 * iqr)
    by    = yc - box_half * 1.4   # vertical centre of box

    # whisker
    ax2.plot([w_lo, w_hi], [by, by],
             color="grey", lw=1.0, zorder=4)
    # IQR box
    rect = plt.Rectangle((q25, by - box_half), q75 - q25, box_half * 2,
                          facecolor=dc, edgecolor="white",
                          linewidth=0.6, zorder=5)
    ax2.add_patch(rect)
    # median line
    ax2.plot([q50, q50], [by - box_half, by + box_half],
             color="white", lw=1.5, zorder=6)

    # ── Jittered dots (below box) ────────────────────────────────
    rng     = np.random.default_rng(42 + rk)
    sample  = rng.choice(scores, size=min(n, 300), replace=False)
    jitter  = rng.uniform(-dot_spread * 0.5, dot_spread * 0.5, len(sample))
    dy      = yc - box_half * 1.4 - box_half * 2.2
    ax2.scatter(sample, dy + jitter,
                color=rc, s=2.5, alpha=0.45, zorder=2)

    # ── Label ────────────────────────────────────────────────────
    ax2.text(x_min - 0.003, yc, f"{rl}\n(n={n:,})",
             va="center", ha="right", fontsize=15.5)

# Threshold lines
ax2.axvline(0.019, color="black",  linestyle="--", lw=1.2, alpha=0.9, label=r"$\tau_{low}=0.019$")
ax2.axvline(0.064, color="crimson", linestyle=":",  lw=1.5, alpha=0.9, label=r"$\tau_{high}=0.064$")

ax2.set_xlim(x_min - 0.007, x_max + 0.005)
ax2.set_ylim(-0.55, 2.85)
ax2.set_yticks([])
ax2.set_ylabel("") # clean left side
ax2.set_title("Stage 2: Regime-Specific Classifiers\n"
              "$\\hat{s}_k(\\varphi(t))=g_k(f_{\\theta_k}(\\varphi(t)))$",
              fontsize=18, fontweight="bold", pad=8)
ax2.set_xlabel("(b)", fontsize=21, labelpad=8, fontstyle="italic")
ax2.tick_params(axis="x", labelsize=16)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.legend(fontsize=12.5, framealpha=0.5, loc="upper right")

# ── Panel (c): Stage 3 — High-quality Gaussian surface (reference-style) ───
ax3 = fig.add_subplot(gs[1, 1:5], projection="3d")
ax3.set_box_aspect((1.4, 1.0, 0.7))

n = len(df_slice)
time_idx = np.linspace(0, 1, n)

# High-resolution grid for smooth surface
n_t = 120   # time resolution
n_r = 80    # regime resolution
time_fine = np.linspace(0, 1, n_t)
regime_fine = np.linspace(0, 1, n_r)
T, R = np.meshgrid(time_fine, regime_fine)
Z = np.zeros_like(T)

sigma_r = 0.22
sigma_t = 0.018
regime_positions = {0: 0.0, 1: 0.5, 2: 1.0}

for i in range(0, n, 2):   # finer subsampling
    t_center = time_idx[i]
    for rk, r_center in regime_positions.items():
        w_col = f"w_{rk}"
        if w_col not in df_slice.columns:
            continue
        height = df_slice[w_col].iloc[i]
        if height < 0.01:
            continue
        Z += height * np.exp(
            -((T - t_center)**2 / (2 * sigma_t**2) +
              (R - r_center)**2 / (2 * sigma_r**2))
        )

# Smooth and normalize
Z = gaussian_filter(Z, sigma=3.5)
Z = Z / Z.max()

# Clip noise floor so base is clean and flat
Z = np.clip(Z, 0.005, None)
Z[Z < 0.015] = 0.0

T_plot = T * (n - 1)
R_plot = R * 2

# ── Draw surface with dense wireframe overlay ──────────────────────
surf = ax3.plot_surface(
    T_plot, R_plot, Z,
    cmap=cm.jet,
    alpha=1.0,
    linewidth=0,
    antialiased=True,
    rstride=1, cstride=1,    # full resolution — no skipping
    shade=True
)

# Overlay wireframe for the mesh-grid look in reference image
ax3.plot_wireframe(
    T_plot, R_plot, Z,
    color="black",
    alpha=0.25,
    linewidth=0.4,
    rstride=1, cstride=3     # dense but not overwhelming
)

# X ticks — dates
n_ticks = 5
tick_pos  = np.linspace(0, n-1, n_ticks, dtype=int)
tick_labs = [df_slice["date"].iloc[i].strftime("%Y-%m") for i in tick_pos]
ax3.set_xticks(np.linspace(0, n-1, n_ticks))
ax3.set_xticklabels(tick_labs, fontsize=15, rotation=20, ha="right", family="serif")

ax3.set_yticks([0, 1, 2])
ax3.set_yticklabels(["Calm\n(k=0)", "Trans.\n(k=1)", "Crisis\n(k=2)"], fontsize=15, family="serif")
ax3.set_zticks([0.0, 0.5, 1.0])
ax3.tick_params(axis="z", labelsize=15)

for label in ax3.get_xticklabels() + ax3.get_yticklabels() + ax3.get_zticklabels():
    label.set_fontfamily("serif")

ax3.set_xlabel("Time",       fontsize=19, labelpad=28, family="serif")
ax3.set_ylabel("Regime $k$", fontsize=19, labelpad=28, family="serif")
ax3.set_zlabel("$w_k(t)$",   fontsize=19, labelpad=8, family="serif")
ax3.set_ylim(2, 0)

# Cleaner pane backgrounds (reference image has white/neutral walls)
ax3.xaxis.pane.fill = False
ax3.yaxis.pane.fill = False
ax3.zaxis.pane.fill = False
ax3.xaxis.pane.set_edgecolor("lightgrey")
ax3.yaxis.pane.set_edgecolor("lightgrey")
ax3.zaxis.pane.set_edgecolor("lightgrey")

ax3.set_title(
    "Stage 3: Soft Gaussian Mixing\n"
    "$\\hat{p}_{RCRE}(t)=\\sum_k w_k(t)\\cdot\\hat{s}_k(\\varphi(t))$",
    fontsize=18, fontweight="bold", pad=12, family="serif"
)
fig.text(0.5, 0.04, "(c)", fontsize=21, fontstyle="italic", ha="center", family="serif")

ax3.view_init(elev=35, azim=-55)

cbar = fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=10, pad=0.12)
cbar.set_label("$w_k(t)$", fontsize=19, family="serif")
cbar.ax.tick_params(labelsize=15.5)
for label in cbar.ax.get_yticklabels():
    label.set_fontfamily("serif")

plt.savefig("artifacts/fig5_rcre_stages.png",
            dpi=150, bbox_inches="tight", facecolor="white")
print("Saved: artifacts/fig5_rcre_stages.png")