import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from huggingface_hub import HfFileSystem, HfApi
import os

st.set_page_config(page_title="Smooth Transition AR", layout="wide")
st.title("📈 Smooth Transition Autoregressive (STAR) Models")
st.caption("Regime‑based forecasting for ETFs | SETAR / LSTAR / ESTAR")

# ---------- Configuration ----------
OUTPUT_REPO = "P2SAMAPA/p2-etf-smooth-transition-results"
HF_TOKEN = os.environ.get("HF_TOKEN", None)  # set in Streamlit secrets

# Debug: show token status
if HF_TOKEN:
    st.sidebar.success("HF_TOKEN is set")
else:
    st.sidebar.warning("HF_TOKEN not set – trying without token (public dataset only)")

# ---------- Helper to load JSON ----------
@st.cache_data(ttl=3600)
def load_json(remote_path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(f"datasets/{OUTPUT_REPO}/{remote_path}", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load {remote_path}: {e}")
        return None

@st.cache_data(ttl=3600)
def load_csv(remote_path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(f"datasets/{OUTPUT_REPO}/{remote_path}", "r") as f:
            return pd.read_csv(f)
    except Exception as e:
        st.error(f"Failed to load {remote_path}: {e}")
        return None

# ---------- Sidebar ----------
model_type = st.sidebar.selectbox("Model type", ["setar", "lstar", "estar"])
universe = st.sidebar.selectbox("Universe", ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"])

# ---------- Load data ----------
summary = load_json(f"{model_type}/summary.json")
backtest = load_csv("backtest_summary.csv")
diag = load_csv("linearity_tests.csv")

# Debug expander
with st.expander("🔍 Debug info (show loaded data)"):
    st.write("Summary keys:", list(summary.keys())[:5] if summary else "None")
    st.write("Backtest shape:", backtest.shape if backtest is not None else "None")
    st.write("Diagnostics shape:", diag.shape if diag is not None else "None")

if summary is None:
    st.warning(f"No summary found for {model_type}. Make sure training has finished and files are uploaded to HF.")
    st.stop()

# ---------- Helper to get ticker list from universe ----------
tickers_map = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM"]
}
if universe == "COMBINED":
    all_tickers = list(set(tickers_map["FI_COMMODITIES"] + tickers_map["EQUITY_SECTORS"]))
else:
    all_tickers = tickers_map[universe]

# ---------- Linearity diagnostics ----------
st.header("Linearity Diagnostics")
if diag is not None:
    diag_sub = diag[diag["ticker"].isin(all_tickers)]
    st.dataframe(diag_sub[["ticker", "lm_pvalue", "reset_pvalue", "tsay_pvalue", "nonlinear"]])
else:
    st.info("Diagnostics file not found.")

# ---------- Backtest performance ----------
st.header("Backtest Performance (Walk‑Forward)")
if backtest is not None and len(backtest) > 0:
    bt = backtest[backtest["model_type"] == model_type]
    bt_sub = bt[bt["ticker"].isin(all_tickers)]
    bt_sub = bt_sub.sort_values("sharpe", ascending=False)
    st.dataframe(bt_sub[["ticker", "p", "d", "sharpe", "hit_rate", "max_drawdown"]])
else:
    st.info("No backtest results for this model type.")

# ---------- Recommendation ----------
st.header("📌 Trading Recommendation for Next Trading Day")
if backtest is not None and not backtest.empty:
    bt_model = backtest[backtest["model_type"] == model_type]
    # pick ticker with highest Sharpe ratio among those in universe
    best = None
    best_sharpe = -np.inf
    for ticker in all_tickers:
        sub = bt_model[bt_model["ticker"] == ticker]
        if len(sub) > 0:
            sharpe = sub["sharpe"].max()
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best = ticker
    if best:
        st.success(f"**Recommended ETF:** {best}")
        st.caption(f"Best Sharpe ratio ({best_sharpe:.2f}) among {model_type} models.")
        # Show threshold if SETAR
        if model_type == "setar" and summary is not None:
            for key, val in summary.items():
                if key.startswith(best):
                    st.write("**Estimated threshold (c):**", val.get("params", {}).get("c", "N/A"))
                    break
    else:
        st.warning("No backtest data for this universe.")
else:
    st.warning("Backtest data missing; cannot provide recommendation.")

st.caption(f"Models trained monthly, data 2008–2026 YTD. Results from {OUTPUT_REPO}")
