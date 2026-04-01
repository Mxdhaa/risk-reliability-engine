import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_IN = "artifacts/table_covid_regimes.csv"
PNG_OUT = "artifacts/table_covid_regimes.png"

df = pd.read_csv(CSV_IN)

# ---- formatting helpers ----
def fmt4(x):
    if pd.isna(x):
        return "—"
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)

# keep dates clean
for c in ["start", "end"]:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")

# format numeric columns
num_cols = [c for c in df.columns if c not in ("regime", "start", "end")]
for c in num_cols:
    df[c] = df[c].apply(fmt4)

# ---- render table as image ----
fig_w = 14
fig_h = 2.6  # increase height so rows don't collide
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.axis("off")

# column widths (sum ~ 1.0)
col_widths = [0.10, 0.12, 0.12, 0.12, 0.12, 0.11, 0.11, 0.12]

tbl = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc="center",
    colLoc="center",
    loc="center",
    colWidths=col_widths,
)

# fonts + spacing
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)

# style header + borders
for (r, c), cell in tbl.get_celld().items():
    cell.set_linewidth(1.2)
    if r == 0:  # header row
        cell.set_text_props(weight="bold")
        cell.set_height(cell.get_height() * 1.15)

plt.savefig(PNG_OUT, dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close()
print(f"Saved: {PNG_OUT}")
