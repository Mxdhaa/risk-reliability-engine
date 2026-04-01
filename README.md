# Risk Reliability Engine

> **Predicting Decision-Time Reliability of Equity Market Risk under Delayed Inputs and Non-Stationary Conditions**

A machine learning system that predicts *when* a Value-at-Risk (VaR) model will fail — before it happens — using only information available at decision time. Validated across 6 global equity markets.

---

## The Problem

Risk desks consume VaR reports with a 1-day reporting lag. During volatility regime shifts, this creates a **decision-time gap**: the reported risk estimate is stale while markets have already moved. Traditional backtesting catches this *after* the fact. This system catches it *before*.

---

## Results

### Cross-Asset Discrimination (ROC-AUC)

| Asset | Vol Threshold | Logistic Reg | XGBoost | RCRE (ours) |
|-------|:---:|:---:|:---:|:---:|
| S&P 500 | 0.508 | 0.755 | 0.646 | **0.669** |
| FTSE 100 | 0.498 | 0.712 | 0.718 | **0.699** |
| Nikkei 225 | 0.496 | 0.650 | 0.614 | **0.567** |
| DAX | 0.497 | 0.736 | 0.648 | **0.631** |
| Hang Seng | 0.542 | 0.644 | 0.567 | **0.533** |
| EM ETF | 0.500 | 0.611 | 0.551 | **0.531** |

### Tail Severity Reduction (Expected Shortfall Breach)

| Policy | ESB | Reduction |
|--------|:---:|:---:|
| No gating (baseline) | 0.0323 | — |
| XGBoost-gated | 0.0137 | **58%** |
| RCRE-gated | 0.0168 | **48%** |

> ESB reduction of **42–63% holds consistently across all 6 markets.**

---

## COVID-19 Stress Test

The reliability monitor was evaluated on the COVID-19 crash (Feb–May 2020) as a **fully held-out** stress period — never seen during training or calibration.

![COVID Figure](artifacts/fig3.png)

The monitor triggered de-risking ahead of both main crash waves, demonstrating early warning capability when the delayed HS-VaR was most stale.

---

## Architecture

```
Market Data (Yahoo Finance)
        │
        ▼
Delayed HS-VaR Baseline (252-day rolling, α=0.99)
        │
        ▼
Lag-Consistent Feature Map (7 features, Fₜ-measurable)
   ├── EWMA volatility + momentum
   ├── Risk-volatility divergence
   ├── Rolling breach count (20-day)
   ├── Excess kurtosis (60-day)
   ├── Instant loss-risk divergence
   └── Absolute loss magnitude
        │
        ▼
┌─────────────────────────────────────┐
│   RCRE: Regime-Conditional          │
│   Reliability Estimator             │
│                                     │
│  1. CUSUM regime detector           │
│     (Calm / Transitional / Crisis)  │
│  2. Per-regime XGBoost classifiers  │
│  3. Soft Gaussian mixing weights    │
│     p̂(t) = Σₖ wₖ(t) · ŝₖ(φₜ)     │
└─────────────────────────────────────┘
        │
        ▼
Isotonic Calibration (validation set only)
        │
        ▼
Three-Level Gating Policy
   ├── ŝₜ < 0.019  → Full exposure (πₜ = 1.0)
   ├── 0.019 ≤ ŝₜ < 0.064 → Half exposure (πₜ = 0.5)
   └── ŝₜ ≥ 0.064  → Halt (πₜ = 0.0)
```

---

## Key Design Principles

**Strict temporal causality** — Features at decision time t use only Fₜ. The baseline risk report uses Fₜ₋₁. No look-ahead leakage anywhere in the pipeline.

**Calibrated probabilities** — Isotonic regression transforms raw classifier scores into economically interpretable failure probabilities, fitted on validation data only.

**Soft regime mixing** — RCRE blends regime-specific classifiers using a Gaussian recency kernel, avoiding discontinuities at regime transition boundaries.

---

## Installation

```bash
git clone https://github.com/Mxdhaa/risk-reliability-engine
cd risk-reliability-engine/src
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

**requirements.txt:**
```
xgboost>=2.0
scikit-learn>=1.3
yfinance>=0.2
pandas>=2.0
numpy>=1.24
arch>=6.0
scipy>=1.10
```

---

## Usage

### Run full pipeline (single asset)
```bash
python run_pipeline.py
```

### Run multi-asset validation
```bash
python multi_asset.py
```

### Run feature ablation
```bash
python ablation.py
```

Output artifacts are written to `src/artifacts/`:
- `preds.csv` — predictions, labels, scores per day
- `fit_metrics.json` — discrimination and calibration metrics
- `multi_asset_results.csv` — cross-asset results table
- `ablation_metrics.json` — feature importance via leave-one-out

---

## Project Structure

```
risk-reliability-engine/
└── src/
    ├── config.py            # All hyperparameters
    ├── data_factory.py      # Data fetching and returns
    ├── risk_models.py       # HS-VaR and EWMA baseline
    ├── features.py          # Lag-consistent feature map
    ├── reliability.py       # XGBoost + isotonic calibration
    ├── rcre.py              # RCRE novel algorithm
    ├── regime_detector.py   # CUSUM + k-means regime detection
    ├── baselines.py         # LR and volatility threshold baselines
    ├── ablation.py          # Leave-one-out feature ablation
    ├── multi_asset.py       # Cross-asset validation
    ├── backtest.py          # Gating policy
    ├── metrics.py           # ESB, Sharpe, Sortino, PRR
    └── run_pipeline.py      # Main entry point
```

---

## Paper

> *Predicting Decision-Time Reliability of Equity Market Risk under Delayed Inputs and Non-Stationary Conditions*
> Anonymous Submission — arXiv preprint coming soon.
