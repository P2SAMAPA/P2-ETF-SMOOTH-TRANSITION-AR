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

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    all_files = []
    try:
        for info in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True):
            if info['type'] == 'file':
                all_files.append(info['name'])
    except Exception as e:
        st.error(f"Cannot list repo: {e}")
    return all_files

def load_json_from_fullpath(full_path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(full_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"JSON load error {full_path}: {e}")
        return None

def load_csv_from_fullpath(full_path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(full_path, "r") as f:
            content = f.read()
            # Check if empty or just header
            if not content.strip():
                st.warning(f"CSV file {full_path} is empty")
                return None
            # Try reading, show first few lines on error
            try:
                df = pd.read_csv(pd.compat.StringIO(content))
                if df.empty:
                    st.warning(f"CSV file {full_path} has no data rows")
                return df
            except Exception as e:
                st.error(f"CSV parse error {full_path}: {e}")
                st.code(content[:500], language="text")
                return None
    except Exception as e:
        st.error(f"Cannot open {full_path}: {e}")
        return None

# Sidebar
model_type = st.sidebar.selectbox("Model type", ["setar", "lstar", "estar"])
universe = st.sidebar.selectbox("Universe", ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"])

files = list_repo_files()
with st.expander("📂 Files in HF repository"):
    st.write(f"Total files: {len(files)}")
    for f in sorted(files):
        st.code(f, language="text")

# Find summary
summary_path = None
for f in files:
    if f"models/{model_type}/summary.json" in f:
        summary_path = f
        break
summary = load_json_from_fullpath(summary_path) if summary_path else None

# Find backtest
backtest_path = None
for f in files:
    if "backtest_summary.csv" in f:
        backtest_path = f
        break
backtest = load_csv_from_fullpath(backtest_path) if backtest_path else None

# Find diagnostics
diag_path = None
for f in files:
    if "linearity_tests.csv" in f:
        diag_path = f
        break
diag = load_csv_from_fullpath(diag_path) if diag_path else None

with st.expander("🔍 Debug info"):
    st.write("Summary path:", summary_path)
    st.write("Summary keys:", list(summary.keys())[:5] if summary else "None")
    st.write("Backtest path:", backtest_path)
    st.write("Backtest shape:", backtest.shape if backtest is not None else "None")
    st.write("Diagnostics shape:", diag.shape if diag is not None else "None")

if summary is None:
    st.error(f"No valid summary for {model_type}.")
    st.stop()

# Tickers
tickers_map = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM"]
}
all_tickers = list(set(tickers_map["FI_COMMODITIES"] + tickers_map["EQUITY_SECTORS"])) if universe == "COMBINED" else tickers_map[universe]

# Diagnostics
st.header("Linearity Diagnostics")
if diag is not None:
    diag_sub = diag[diag["ticker"].isin(all_tickers)]
    st.dataframe(diag_sub[["ticker", "lm_pvalue", "reset_pvalue", "tsay_pvalue", "nonlinear"]])
else:
    st.info("Diagnostics file not found.")

# Backtest
st.header("Backtest Performance")
if backtest is not None and not backtest.empty:
    bt = backtest[backtest["model_type"] == model_type]
    bt_sub = bt[bt["ticker"].isin(all_tickers)]
    bt_sub = bt_sub.sort_values("sharpe", ascending=False)
    st.dataframe(bt_sub[["ticker", "p", "d", "sharpe", "hit_rate", "max_drawdown"]])
else:
    st.info("No backtest results. The CSV may be empty – rerun training/backtest.")

# Recommendation
st.header("📌 Trading Recommendation")
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
        st.caption(f"Sharpe: {best_sharpe:.2f}")
        if model_type == "setar" and summary:
            for key, val in summary.items():
                if key.startswith(best_ticker):
                    st.write("**Threshold c:**", val.get("params", {}).get("c", "N/A"))
                    break
    else:
        st.warning("No backtest data for this universe.")
else:
    st.warning("Backtest data missing – the CSV is likely empty or corrupt.")

st.caption(f"Data from {OUTPUT_REPO}")
