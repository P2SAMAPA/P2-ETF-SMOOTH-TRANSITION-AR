import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from huggingface_hub import HfFileSystem
import os

st.set_page_config(page_title="Smooth Transition AR", layout="wide")
st.title("📈 Smooth Transition Autoregressive (STAR) Models")
st.caption("Regime‑based forecasting for ETFs | SETAR / LSTAR / ESTAR")

HF_TOKEN = os.environ.get("HF_TOKEN", None)
OUTPUT_REPO = "P2SAMAPA/p2-etf-smooth-transition-results"

@st.cache_data(ttl=3600)
def load_summary(model_type):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(f"datasets/{OUTPUT_REPO}/{model_type}/summary.json", "r") as f:
            return json.load(f)
    except:
        return None

@st.cache_data(ttl=3600)
def load_backtest():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(f"datasets/{OUTPUT_REPO}/backtest_summary.csv", "r") as f:
            return pd.read_csv(f)
    except:
        return None

@st.cache_data(ttl=3600)
def load_diagnostics():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(f"datasets/{OUTPUT_REPO}/linearity_tests.csv", "r") as f:
            return pd.read_csv(f)
    except:
        return None

st.sidebar.header("Configuration")
model_type = st.sidebar.selectbox("Model type", ["setar", "lstar", "estar"])
universe = st.sidebar.selectbox("Universe", ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"])

# Load data
summary = load_summary(model_type)
backtest = load_backtest()
diag = load_diagnostics()

if summary is None:
    st.warning(f"No summary found for {model_type}. Run training first.")
    st.stop()

# Filter backtest for the selected model type
if backtest is not None:
    bt = backtest[backtest["model_type"] == model_type]
else:
    bt = None

# Build list of tickers from config (hardcoded for display)
tickers_map = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM"]
}
if universe == "COMBINED":
    all_tickers = list(set(tickers_map["FI_COMMODITIES"] + tickers_map["EQUITY_SECTORS"]))
else:
    all_tickers = tickers_map[universe]

# Show diagnostic summary
st.header("Linearity Diagnostics")
if diag is not None:
    diag_sub = diag[diag["ticker"].isin(all_tickers)]
    st.dataframe(diag_sub[["ticker", "lm_pvalue", "reset_pvalue", "tsay_pvalue", "nonlinear"]])
else:
    st.info("Diagnostics not available.")

# Show backtest metrics for the best model per ticker
st.header("Backtest Performance (Walk‑Forward)")
if bt is not None and len(bt) > 0:
    bt_sub = bt[bt["ticker"].isin(all_tickers)]
    bt_sub = bt_sub.sort_values("sharpe", ascending=False)
    st.dataframe(bt_sub[["ticker", "p", "d", "sharpe", "hit_rate", "max_drawdown"]])
else:
    st.info("No backtest results.")

# Recommendation: pick ticker with highest Sharpe ratio (or lowest AIC)
st.header("📌 Trading Recommendation for Next Trading Day")
if summary:
    best_ticker = None
    best_sharpe = -np.inf
    for ticker in all_tickers:
        # Find model for this ticker with best Sharpe (from backtest)
        if bt is not None:
            this_bt = bt[(bt["ticker"] == ticker)]
            if len(this_bt) > 0:
                sharpe = this_bt["sharpe"].max()
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_ticker = ticker
    if best_ticker is None:
        best_ticker = all_tickers[0]

    st.success(f"**Recommended ETF:** {best_ticker}")
    st.caption(f"Based on highest Sharpe ratio ({best_sharpe:.2f}) among {model_type} models.")

    # Show regime transition (if SETAR)
    if model_type == "setar":
        # Look for the model parameters in summary
        for key, val in summary.items():
            if key.startswith(best_ticker):
                st.write("**Threshold (c):**", val["params"]["c"])
                st.write("**Regime proportions:**", val["regime_props"])
                break
    st.markdown("---")
    st.info("This recommendation uses walk‑forward backtesting. Trade at your own risk.")

st.caption(f"Data: {OUTPUT_REPO} | Models trained monthly on 2008‑2026 YTD")
