import pandas as pd
from metrics import breach_ratio, expected_shortfall_of_breaches, sharpe_ratio, sortino_ratio, prr_ratio

def gate_policy(s: pd.Series, tau_low: float, tau_high: float, phi: float) -> pd.Series:
    # s = probability of FAILURE (unreliable tomorrow)
    # high s => reduce exposure
    pi = pd.Series(index=s.index, dtype=float)
    pi[s >= tau_high] = 0.0
    pi[(s >= tau_low) & (s < tau_high)] = phi
    pi[s < tau_low] = 1.0
    return pi


def compute_metrics_v2(returns: pd.Series, base_risk: pd.Series, pi: pd.Series, alpha: float) -> dict:
    r = returns.astype(float)
    L = (-r).rename("loss")
    pi = pi.astype(float).reindex(r.index).fillna(1.0)

    risk_base = base_risk.astype(float).reindex(r.index)
    risk_gated = risk_base * pi

    pnl_base = r
    pnl_gated = pi * r

    return {
        "breach_ratio_base": breach_ratio(L, risk_base),
        "breach_ratio_gated": breach_ratio(L, risk_gated),
        "esb_base": expected_shortfall_of_breaches(L, risk_base),
        "esb_gated": expected_shortfall_of_breaches(L, risk_gated),
        "sharpe_base": sharpe_ratio(pnl_base),
        "sharpe_gated": sharpe_ratio(pnl_gated),
        "sortino_base": sortino_ratio(pnl_base),
        "sortino_gated": sortino_ratio(pnl_gated),
        "prr": prr_ratio(r, L, pi, alpha),
        "avg_exposure": float(pi.mean()),
    }
