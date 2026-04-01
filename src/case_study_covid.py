# src/case_study_covid.py
# FIX: your preds.csv columns are ['date','p','y','L','R'] not ['st','yt','Lt','Rt'].
# This version auto-maps both schemas and keeps everything else identical.

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_preds(artifacts_dir: str) -> pd.DataFrame:
    p = os.path.join(artifacts_dir, "preds.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p}. Run your pipeline first to create artifacts/preds.csv")
    df = pd.read_csv(p)

    # normalize date column
    if "t" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"t": "date"})
    if "date" not in df.columns:
        raise ValueError(f"preds.csv must contain a 'date' column (or 't'). Found: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either schema:
      A) ['date','st','yt','Lt','Rt']  (paper naming)
      B) ['date','p','y','L','R']      (your artifacts naming)
    Produces canonical columns: ['date','st','yt','Lt','Rt']
    """
    cols = set(df.columns)

    # schema B (your run): p,y,L,R
    if {"p", "y", "L", "R"}.issubset(cols):
        df = df.rename(columns={"p": "st", "y": "yt", "L": "Lt", "R": "Rt"})
        return df

    # schema A already
    if {"st", "yt", "Lt", "Rt"}.issubset(cols):
        return df

    raise ValueError(
        f"preds.csv missing required cols. Expected either "
        f"['date','st','yt','Lt','Rt'] or ['date','p','y','L','R']. Found: {list(df.columns)}"
    )


def _window_slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    return df[(df["date"] >= s) & (df["date"] <= e)].copy()


def _compute_case_cols(df: pd.DataFrame, kappa: float, phi: float, tau_low: float, tau_high: float):
    # Align to next-day outcome (your label does shift(-1))
    df["L_next"] = df["Lt"].shift(-1)
    df["R_t"] = df["Rt"]

    # Baseline next-day breach (aligned to time t decision)
    df["breach_base_next"] = (df["L_next"] > (kappa * df["R_t"])).astype(int)

    # Policy (exact from paper)
    df["pi_t"] = 1.0
    df.loc[df["st"] >= tau_low, "pi_t"] = phi
    df.loc[df["st"] >= tau_high, "pi_t"] = 0.0

    # Gated limit and gated next-day loss stream
    df["limit_gated"] = df["pi_t"] * (kappa * df["R_t"])
    df["L_next_gated"] = df["pi_t"] * df["L_next"]
    df["breach_gated_next"] = (df["L_next_gated"] > df["limit_gated"]).astype(int)

    return df


def _vr_esb(df: pd.DataFrame, breach_col: str, loss_col: str):
    vr = float(df[breach_col].mean()) if len(df) else float("nan")
    breached = df[df[breach_col] == 1]
    esb = float(breached[loss_col].mean()) if len(breached) else float("nan")
    avg_expo = float(df["pi_t"].mean()) if "pi_t" in df.columns and len(df) else float("nan")
    return vr, esb, avg_expo


def _save_table(rows, out_csv: str, out_tex: str):
    tab = pd.DataFrame(rows)
    tab.to_csv(out_csv, index=False)
    tex = tab.to_latex(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x))
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(tex)


def _plot_timeseries(df: pd.DataFrame, shock_start: str, shock_end: str, out_png: str, title: str):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(df["date"], df["st"])
    ax2.plot(df["date"], df["L_next"])

    b = df["breach_base_next"] == 1
    ax2.scatter(df.loc[b, "date"], df.loc[b, "L_next"], s=12)

    s = pd.to_datetime(shock_start)
    e = pd.to_datetime(shock_end)
    ax1.axvspan(s, e, alpha=0.2)

    ax1.set_title(title)
    ax1.set_xlabel("date")
    ax1.set_ylabel("calibrated failure probability s_t")
    ax2.set_ylabel("next-day loss L_{t+1}")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_breach_magnitude(df: pd.DataFrame, out_png: str, title: str):
    base = df.loc[df["breach_base_next"] == 1, "L_next"]
    gated = df.loc[df["breach_gated_next"] == 1, "L_next_gated"]

    fig, ax = plt.subplots()
    ax.hist(base.dropna().values, bins=30, alpha=0.6, label="baseline breach magnitudes")
    ax.hist(gated.dropna().values, bins=30, alpha=0.6, label="gated breach magnitudes")
    ax.set_title(title)
    ax.set_xlabel("loss magnitude")
    ax.set_ylabel("count")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", default="artifacts")

    # COVID case window
    ap.add_argument("--cs_start", default="2019-10-01")
    ap.add_argument("--cs_end", default="2020-08-31")

    # regime splits
    ap.add_argument("--pre_start", default="2019-10-01")
    ap.add_argument("--pre_end", default="2020-02-14")
    ap.add_argument("--shock_start", default="2020-02-18")
    ap.add_argument("--shock_end", default="2020-04-30")
    ap.add_argument("--post_start", default="2020-05-01")
    ap.add_argument("--post_end", default="2020-08-31")

    # params (match paper defaults)
    ap.add_argument("--kappa", type=float, default=0.8189672074)
    ap.add_argument("--phi", type=float, default=0.5)
    ap.add_argument("--tau_low", type=float, default=0.0714285746)
    ap.add_argument("--tau_high", type=float, default=0.1578947306)

    args = ap.parse_args()
    os.makedirs(args.artifacts_dir, exist_ok=True)

    df = _read_preds(args.artifacts_dir)
    df = _normalize_schema(df)

    df = _compute_case_cols(df, args.kappa, args.phi, args.tau_low, args.tau_high)

    # Figure CS1
    cs = _window_slice(df, args.cs_start, args.cs_end)
    _plot_timeseries(
        cs,
        shock_start=args.shock_start,
        shock_end=args.shock_end,
        out_png=os.path.join(args.artifacts_dir, "fig_covid_timeseries.png"),
        title="COVID case study: s_t vs next-day loss with breaches",
    )

    regimes = [
        ("pre", args.pre_start, args.pre_end),
        ("shock", args.shock_start, args.shock_end),
        ("post", args.post_start, args.post_end),
    ]

    rows = []
    for name, s, e in regimes:
        sub = _window_slice(df, s, e).dropna(subset=["L_next", "R_t", "st"])

        vr_b, esb_b, _ = _vr_esb(sub, "breach_base_next", "L_next")
        vr_g, esb_g, avg_expo = _vr_esb(sub, "breach_gated_next", "L_next_gated")

        rows.append(
            dict(
                regime=name,
                start=s,
                end=e,
                vr_baseline=vr_b,
                esb_baseline=esb_b,
                vr_gated=vr_g,
                esb_gated=esb_g,
                avg_exposure=avg_expo,
            )
        )

    _save_table(
        rows,
        out_csv=os.path.join(args.artifacts_dir, "table_covid_regimes.csv"),
        out_tex=os.path.join(args.artifacts_dir, "table_covid_regimes.tex"),
    )

    # Figure CS2 (shock window)
    shock = _window_slice(df, args.shock_start, args.shock_end)
    _plot_breach_magnitude(
        shock,
        out_png=os.path.join(args.artifacts_dir, "fig_covid_breach_magnitudes.png"),
        title="COVID shock: breach magnitude distribution (baseline vs gated)",
    )

    with open(os.path.join(args.artifacts_dir, "covid_case_study_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
