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

OUTPUT_REPO = "P2SAMAPA/p2-etf-smooth-transition-results"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Helper to list all files in the HF repo recursively (returns full paths)
@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    all_files = []
    try:
        for info in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True):
            if info['type'] == 'file':
                # info['name'] is something like "datasets/P2SAMAPA/.../backtest_summary.csv"
                all_files.append(info['name'])
    except Exception as e:
        st.error(f"Cannot list repo: {e}")
    return all_files

# Helper to load JSON from a full path
@st.cache_data(ttl=3600)
def load_json_from_fullpath(full_path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(full_path, "r") as f:
            return json.load(f)
    except Exception as e:
        return None

# Helper to load CSV from a full path
@st.cache_data(ttl=3600)
def load_csv_from_fullpath(full_path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(full_path, "r") as f:
            return pd.read_csv(f)
    except Exception as e:
        return None

# Sidebar
model_type = st.sidebar.selectbox("Model type", ["setar", "lstar", "estar"])
universe = st.sidebar.selectbox("Universe", ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"])

# List files
files = list_repo_files()
st.subheader("📂 Files in HF repository")
st.write(f"Total files: {len(files)}")
if files:
    for f in sorted(files):
        st.code(f, language="text")
else:
    st.warning("No files found. Run the GitHub workflow again.")

# Find summary.json for selected model type
summary_path = None
for full_path in files:
    if f"models/{model_type}/summary.json" in full_path:
        summary_path = full_path
        break
if summary_path:
    summary = load_json_from_fullpath(summary_path)
else:
    summary = None

# Find backtest CSV (any path containing backtest_summary.csv)
backtest_path = None
for full_path in files:
    if "backtest_summary.csv" in full_path:
        backtest_path = full_path
        break
backtest = load_csv_from_fullpath(backtest_path) if backtest_path else None

# Find diagnostics CSV
diag_path = None
for full_path in files:
    if "linearity_tests.csv" in full_path:
        diag_path = full_path
        break
diag = load_csv_from_fullpath(diag_path) if diag_path else None

# Debug expander
with st.expander("🔍 Debug info (loaded files)"):
    st.write("Summary path:", summary_path)
    st.write("Summary keys:", list(summary.keys())[:5] if summary else "None")
    st.write("Backtest path:", backtest_path)
    st.write("Backtest shape:", backtest.shape if backtest is not None else "None")
    st.write("Diagnostics shape:", diag.shape if diag is not None else "None")

if summary is None:
    st.error(f"No summary found for model type '{model_type}'. Ensure training finished and files are uploaded.")
    st.stop()

# ---- Tickers list ----
tickers_map = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM"]
}
if universe == "COMBINED":
    all_tickers = list(set(tickers_map["FI_COMMODITIES"] + tickers_map["EQUITY_SECTORS"]))
else:
    all_tickers = tickers_map[universe]

# ---- Diagnostics ----
st.header("Linearity Diagnostics")
if diag is not None:
    diag_sub = diag[diag["ticker"].isin(all_tickers)]
    st.dataframe(diag_sub[["ticker", "lm_pvalue", "reset_pvalue", "tsay_pvalue", "nonlinear"]])
else:
    st.info("Diagnostics file not found.")

# ---- Backtest ----
st.header("Backtest Performance (Walk‑Forward)")
if backtest is not None and len(backtest) > 0:
    bt = backtest[backtest["model_type"] == model_type]
    bt_sub = bt[bt["ticker"].isin(all_tickers)]
    bt_sub = bt_sub.sort_values("sharpe", ascending=False)
    st.dataframe(bt_sub[["ticker", "p", "d", "sharpe", "hit_rate", "max_drawdown"]])
else:
    st.info("No backtest results for this model type.")

# ---- Recommendation ----
st.header("📌 Trading Recommendation for Next Trading Day")
if backtest is not None and not backtest.empty:
    bt_model = backtest[backtest["model_type"] == model_type]
    best_ticker = None
    best_sharpe = -np.inf
    for ticker in all_tickers:
        sub = bt_model[bt_model["ticker"] == ticker]
        if len(sub) > 0:
            sharpe = sub["sharpe"].max()
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_ticker = ticker
    if best_ticker:
        st.success(f"**Recommended ETF:** {best_ticker}")
        st.caption(f"Best Sharpe ratio ({best_sharpe:.2f}) among {model_type} models.")
        if model_type == "setar" and summary is not None:
            for key, val in summary.items():
                if key.startswith(best_ticker):
                    st.write("**Estimated threshold (c):**", val.get("params", {}).get("c", "N/A"))
                    break
    else:
        st.warning("No backtest data for this universe.")
else:
    st.warning("Backtest data missing; cannot provide recommendation.")

st.caption(f"Models trained monthly, data 2008–2026 YTD. Results from {OUTPUT_REPO}")
