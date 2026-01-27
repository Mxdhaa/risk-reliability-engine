import numpy as np
import pandas as pd

def sharpe_ratio(pnl: pd.Series, ann_factor: float = 252.0) -> float:
    x = pnl.dropna().astype(float)
    if len(x) < 3:
        return float("nan")
    sd = float(x.std(ddof=1))
    if sd == 0.0:
        return float("nan")
    return float((x.mean() / sd) * np.sqrt(ann_factor))

def sortino_ratio(pnl: pd.Series, ann_factor: float = 252.0) -> float:
    x = pnl.dropna().astype(float)
    if len(x) < 3:
        return float("nan")
    downside = x[x < 0]
    dd = float(downside.std(ddof=1)) if len(downside) >= 2 else 0.0
    if dd == 0.0:
        return float("nan")
    return float((x.mean() / dd) * np.sqrt(ann_factor))

def breach_ratio(losses: pd.Series, risk_limit: pd.Series) -> float:
    L = losses.astype(float)
    R = risk_limit.astype(float)
    m = (L > R).dropna()
    return float(m.mean()) if len(m) else float("nan")

def expected_shortfall_of_breaches(losses: pd.Series, risk_limit: pd.Series) -> float:
    L = losses.astype(float)
    R = risk_limit.astype(float)
    mask = (L > R)
    x = L[mask].dropna()
    return float(x.mean()) if len(x) else 0.0

def prr_ratio(returns: pd.Series, losses: pd.Series, pi: pd.Series, alpha: float) -> float:
    r = returns.astype(float).dropna()
    L = losses.astype(float).reindex(r.index)
    pi = pi.astype(float).reindex(r.index).fillna(1.0)

    pnl_base = r
    pnl_gated = pi * r

    tail_thr = float(L.quantile(alpha))
    tail_mask = (L >= tail_thr)

    tail_loss_base = float(L[tail_mask].mean()) if tail_mask.any() else 0.0
    tail_loss_gated = float((pi[tail_mask] * L[tail_mask]).mean()) if tail_mask.any() else 0.0
    risk_reduction = float(tail_loss_base - tail_loss_gated)

    opp_cost = float(pnl_base.mean() - pnl_gated.mean())
    return float(risk_reduction / (opp_cost + 1e-12))

def latex_table(rows: list[dict], caption: str, label: str) -> str:
    cols = list(rows[0].keys())

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{" + "l" * len(cols) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\midrule")

    for row in rows:
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append("NA" if v != v else f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append(" & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)
