"""app.py — Smooth Transition AR Dashboard."""

from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="STAR Models · P2Quant",
    layout="wide",
    page_icon="📈",
)

OUTPUT_REPO = "P2SAMAPA/p2-etf-smooth-transition-results"
HF_TOKEN = os.environ.get("HF_TOKEN")

TICKERS_MAP = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY",
        "QQQ",
        "XLK",
        "XLF",
        "XLE",
        "XLV",
        "XLI",
        "XLY",
        "XLP",
        "XLU",
        "GDX",
        "XME",
        "IWF",
        "XSD",
        "XBI",
        "IWM",
    ],
}
TICKERS_MAP["COMBINED"] = sorted(
    set(TICKERS_MAP["FI_COMMODITIES"] + TICKERS_MAP["EQUITY_SECTORS"])
)

MODEL_COLOURS = {"setar": "#1B4F8A", "lstar": "#27AE60", "estar": "#E74C3C"}
MODEL_LABELS = {
    "setar": "SETAR — Self-Exciting Threshold AR",
    "lstar": "LSTAR — Logistic Smooth Transition AR",
    "estar": "ESTAR — Exponential Smooth Transition AR",
}


# ── Data loading ──────────────────────────────────────────────────────────────
def _hf_headers() -> dict:
    return {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def _raw_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{OUTPUT_REPO}/resolve/main/{path}"


@st.cache_data(ttl=3600, show_spinner="Loading results…")
def load_summary(model_type: str) -> dict | None:
    url = _raw_url(f"{model_type}/summary.json")
    try:
        r = requests.get(url, headers=_hf_headers(), timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner="Loading backtest…")
def load_backtest() -> pd.DataFrame | None:
    url = _raw_url("backtest_summary.csv")
    try:
        r = requests.get(url, headers=_hf_headers(), timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        if not r.text.strip():
            return None
        from io import StringIO

        df = pd.read_csv(StringIO(r.text))
        return df if not df.empty else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner="Loading diagnostics…")
def load_diagnostics() -> pd.DataFrame | None:
    url = _raw_url("linearity_tests.csv")
    try:
        r = requests.get(url, headers=_hf_headers(), timeout=30)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        if not r.text.strip():
            return None
        from io import StringIO

        df = pd.read_csv(StringIO(r.text))
        return df if not df.empty else None
    except Exception:
        return None


def fmt_pct(v: float) -> str:
    return f"{v * 100:+.2f}%"


def fmt_sharpe(v: float) -> str:
    return f"{v:.3f}"


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📈 Smooth Transition AR Models")
st.caption(
    "Regime-based forecasting: SETAR · LSTAR · ESTAR | "
    "Walk-forward backtest | Sharpe-ranked ETF recommendations"
)

backtest = load_backtest()
diag = load_diagnostics()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    model_type = st.selectbox(
        "Model",
        ["setar", "lstar", "estar"],
        format_func=lambda x: x.upper(),
    )
    universe = st.selectbox(
        "Universe", ["FI_COMMODITIES", "EQUITY_SECTORS", "COMBINED"]
    )
    st.divider()
    st.markdown(f"**Model:** {MODEL_LABELS[model_type]}")
    st.markdown(f"**Universe:** {len(TICKERS_MAP[universe])} tickers")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

all_tickers = TICKERS_MAP[universe]
summary = load_summary(model_type)

# ── Diagnostics tab ───────────────────────────────────────────────────────────
tab_diag, tab_backtest, tab_recommend, tab_params = st.tabs(
    [
        "🧪 Linearity Tests",
        "📊 Backtest Results",
        "🎯 Recommendation",
        "🔬 Model Parameters",
    ]
)

with tab_diag:
    st.subheader("Linearity Diagnostics")
    st.caption(
        "LM = Lagrange Multiplier test · RESET = Ramsey RESET · Tsay = Tsay's test. "
        "p < 0.05 → significant nonlinearity → STAR model appropriate."
    )
    if diag is not None:
        diag_sub = diag[diag["ticker"].isin(all_tickers)].copy()

        # Colour code p-values
        nl_count = diag_sub["nonlinear"].sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Tickers Tested", len(diag_sub))
        c2.metric("Nonlinear (p<0.05)", int(nl_count))
        c3.metric("Linear", int(len(diag_sub) - nl_count))

        display_cols = [
            "ticker",
            "lm_pvalue",
            "reset_pvalue",
            "tsay_pvalue",
            "nonlinear",
        ]
        st.dataframe(
            diag_sub[display_cols]
            .sort_values("lm_pvalue")
            .style.format(
                {
                    "lm_pvalue": "{:.4f}",
                    "reset_pvalue": "{:.4f}",
                    "tsay_pvalue": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        # P-value bar chart
        fig_diag = go.Figure()
        for col, label in [
            ("lm_pvalue", "LM"),
            ("reset_pvalue", "RESET"),
            ("tsay_pvalue", "Tsay"),
        ]:
            fig_diag.add_trace(
                go.Bar(
                    name=label,
                    x=diag_sub["ticker"],
                    y=diag_sub[col],
                )
            )
        fig_diag.add_hline(
            y=0.05, line_dash="dash", line_color="red", annotation_text="5% threshold"
        )
        fig_diag.update_layout(
            title="Linearity Test p-values by Ticker",
            barmode="group",
            yaxis_title="p-value",
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_diag, use_container_width=True, key="diag_chart")
    else:
        st.info("Run `python run_diagnostics.py` to generate linearity tests.")

with tab_backtest:
    st.subheader(f"Backtest Performance — {model_type.upper()}")
    if backtest is not None:
        bt = backtest[
            (backtest["model_type"] == model_type)
            & (backtest["ticker"].isin(all_tickers))
        ].copy()

        if bt.empty:
            st.info(f"No backtest results for {model_type.upper()} in {universe}.")
        else:
            # Best config per ticker (highest Sharpe)
            bt_best = bt.sort_values("sharpe", ascending=False).drop_duplicates(
                "ticker"
            )
            bt_best = bt_best.sort_values("sharpe", ascending=False)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tickers", len(bt_best))
            m2.metric("Avg Sharpe", f"{bt_best['sharpe'].mean():.3f}")
            m3.metric("Avg Hit Rate", f"{bt_best['hit_rate'].mean():.1%}")
            m4.metric("Avg Max DD", f"{bt_best['max_drawdown'].mean():.1%}")

            # Bar chart — Sharpe by ticker
            colours = [
                "#27AE60" if s > 0.5 else "#F39C12" if s > 0 else "#E74C3C"
                for s in bt_best["sharpe"]
            ]
            fig_bt = go.Figure(
                go.Bar(
                    y=bt_best["ticker"],
                    x=bt_best["sharpe"],
                    orientation="h",
                    marker_color=colours,
                    text=[f"{s:.3f}" for s in bt_best["sharpe"]],
                    textposition="outside",
                )
            )
            fig_bt.add_vline(x=0, line_dash="dot", line_color="gray")
            fig_bt.update_layout(
                title=f"Sharpe Ratio — {model_type.upper()} ({universe})",
                xaxis_title="Sharpe Ratio",
                yaxis=dict(autorange="reversed"),
                height=max(300, len(bt_best) * 30),
                margin=dict(t=40, b=30, l=60, r=80),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_bt, use_container_width=True, key="bt_bar")

            # Full table
            st.dataframe(
                bt_best[
                    [
                        "ticker",
                        "p",
                        "d",
                        "sharpe",
                        "hit_rate",
                        "max_drawdown",
                        "mean_return",
                        "volatility",
                    ]
                ]
                .rename(
                    columns={
                        "sharpe": "Sharpe",
                        "hit_rate": "Hit Rate",
                        "max_drawdown": "Max DD",
                        "mean_return": "Ann. Return",
                        "volatility": "Ann. Vol",
                    }
                )
                .style.format(
                    {
                        "Sharpe": "{:.3f}",
                        "Hit Rate": "{:.1%}",
                        "Max DD": "{:.1%}",
                        "Ann. Return": "{:.1%}",
                        "Ann. Vol": "{:.1%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            # Scatter: Sharpe vs Hit Rate
            fig_scatter = go.Figure(
                go.Scatter(
                    x=bt_best["hit_rate"],
                    y=bt_best["sharpe"],
                    mode="markers+text",
                    text=bt_best["ticker"],
                    textposition="top center",
                    marker=dict(
                        size=10,
                        color=bt_best["sharpe"],
                        colorscale="RdYlGn",
                        colorbar=dict(title="Sharpe"),
                        showscale=True,
                    ),
                )
            )
            fig_scatter.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_scatter.update_layout(
                title="Sharpe vs Hit Rate",
                xaxis_title="Hit Rate",
                yaxis_title="Sharpe",
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_scatter, use_container_width=True, key="bt_scatter")
    else:
        st.info("No backtest results. Run `python run_backtest.py` to generate.")

with tab_recommend:
    st.subheader(f"🎯 Top ETF Recommendation — {model_type.upper()} · {universe}")
    if backtest is not None:
        bt = backtest[
            (backtest["model_type"] == model_type)
            & (backtest["ticker"].isin(all_tickers))
        ]
        bt_best = bt.sort_values("sharpe", ascending=False).drop_duplicates("ticker")

        if not bt_best.empty:
            top = bt_best.iloc[0]
            sharpe_val = float(top["sharpe"])
            signal = (
                "STRONG"
                if sharpe_val > 0.5
                else "MODERATE" if sharpe_val > 0 else "WEAK"
            )
            sig_colour = {
                "STRONG": "#27AE60",
                "MODERATE": "#F39C12",
                "WEAK": "#E74C3C",
            }[signal]

            r_col, m_col = st.columns([2, 3])
            with r_col:
                st.markdown(
                    f"### {top['ticker']} &nbsp; "
                    f'<span style="background:{sig_colour};color:white;padding:3px 12px;'
                    f'border-radius:12px;font-size:14px">{signal}</span>',
                    unsafe_allow_html=True,
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Sharpe", f"{sharpe_val:.3f}")
                c2.metric("Hit Rate", f"{float(top['hit_rate']):.1%}")
                c3.metric("Max DD", f"{float(top['max_drawdown']):.1%}")
                c4.metric("Ann. Return", f"{float(top['mean_return']):.1%}")
                st.caption(
                    f"Best config: p={int(top['p'])}, d={int(top['d'])} | "
                    f"Model: {MODEL_LABELS[model_type]}"
                )

            with m_col:
                # Top 5 ranked
                st.markdown("**Top 5 ETFs by Sharpe**")
                top5 = bt_best.head(5)[["ticker", "sharpe", "hit_rate", "max_drawdown"]]
                st.dataframe(
                    top5.style.format(
                        {
                            "sharpe": "{:.3f}",
                            "hit_rate": "{:.1%}",
                            "max_drawdown": "{:.1%}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.warning(f"No backtest data for {model_type.upper()} in {universe}.")
    else:
        st.info("No backtest data. Run `python run_backtest.py`.")

with tab_params:
    st.subheader(f"Model Parameters — {model_type.upper()}")
    if summary is None:
        st.info(f"No summary for {model_type.upper()}. Run training first.")
    else:
        ticker_filter = st.multiselect(
            "Filter tickers",
            options=all_tickers,
            default=all_tickers[:5],
        )
        rows = []
        for key, meta in summary.items():
            parts = key.split("_")
            if len(parts) < 3:
                continue
            ticker = parts[0]
            if ticker not in ticker_filter:
                continue
            try:
                p = int(parts[1][1:])
                d = int(parts[2][1:])
            except (ValueError, IndexError):
                continue

            params = meta.get("params", {})
            row: dict = {"Ticker": ticker, "p": p, "d": d}

            if model_type == "setar":
                row["Threshold c"] = round(params.get("c", 0), 6)
                row["AIC"] = round(meta.get("aic", 0), 2)
                row["BIC"] = round(meta.get("bic", 0), 2)
                r1 = meta.get("regime_props", {})
                row["Regime 1 %"] = f"{r1.get('regime1_pct', 0):.1%}"
                row["Regime 2 %"] = f"{r1.get('regime2_pct', 0):.1%}"
            else:
                row["Gamma"] = round(params.get("gamma", 0), 4)
                row["Threshold c"] = round(params.get("c", 0), 6)
                if "sigma" in meta:
                    row["Sigma"] = round(meta["sigma"], 6)

            rows.append(row)

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Threshold distribution chart
            thresholds = [r.get("Threshold c") for r in rows if "Threshold c" in r]
            tickers_plot = [r["Ticker"] for r in rows if "Threshold c" in r]
            if thresholds:
                fig_thresh = go.Figure(
                    go.Bar(
                        x=tickers_plot,
                        y=thresholds,
                        marker_color=MODEL_COLOURS[model_type],
                        name="Threshold c",
                    )
                )
                fig_thresh.update_layout(
                    title="Estimated Threshold (c) by Ticker",
                    yaxis_title="Threshold c",
                    height=300,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(
                    fig_thresh, use_container_width=True, key="thresh_chart"
                )
        else:
            st.info("No matching model entries for selected tickers.")

st.divider()
st.caption(f"P2Quant STAR Engine · Data: {OUTPUT_REPO}")
