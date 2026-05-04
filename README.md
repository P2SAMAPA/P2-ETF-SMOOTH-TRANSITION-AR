# P2-ETF-SMOOTH-TRANSITION-AR

**Smooth Transition Autoregressive (STAR) and Threshold AR (TAR) models for ETF regime detection.**

## Features

- **SETAR** (Self-Exciting Threshold AR) – abrupt regime switches
- **LSTAR** (Logistic STAR) – smooth transitions between regimes
- **ESTAR** (Exponential STAR) – symmetric smooth transitions
- Walk‑forward backtesting with Sharpe ratio, hit rate, max drawdown
- Linearity diagnostics (LM test, RESET, Tsay test)
- Runs on three ETF universes: FI_COMMODITIES, EQUITY_SECTORS, COMBINED
- Data from `P2SAMAPA/fi-etf-macro-signal-master-data` (2008–2026)

## Installation

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-SMOOTH-TRANSITION-AR.git
cd P2-ETF-SMOOTH-TRANSITION-AR
pip install -r requirements.txt
Usage
Run all steps (recommended via GitHub Actions)
bash
python run_diagnostics.py
python run_training.py --model setar
python run_training.py --model lstar
python run_training.py --model estar
python run_backtest.py
Outputs
models/ – pickled models and JSON summaries

backtest/backtest_summary.csv – performance metrics

diagnostics/linearity_tests.csv – tests for nonlinearity

Configuration
Edit config.yaml to change AR orders, backtest window, tickers.

Papers
Tong (1978) – TAR

Teräsvirta (1994) – STAR framework

van Dijk, Teräsvirta, Franses (2002) – STAR survey

Tsay (1989) – SETAR testing
