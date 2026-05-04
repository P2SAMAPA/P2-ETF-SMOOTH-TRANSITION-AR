#!/usr/bin/env python3
"""run_diagnostics.py — Linearity tests for all tickers."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml

from diagnostics import lm_linearity_test, reset_test, tsay_test
from utils import load_data, upload_results


def main() -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    token = os.environ.get("HF_TOKEN")
    returns = load_data(token=token)

    output_dir = Path("diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers: list[str] = []
    for universe_tickers in config["data"]["tickers"].values():
        tickers.extend(universe_tickers)
    tickers = sorted(set(tickers))

    rows: list[dict] = []
    for ticker in tickers:
        if ticker not in returns.columns:
            continue
        y = returns[ticker].dropna().values
        if len(y) < 200:
            continue

        print(f"  {ticker} ...", end=" ", flush=True)
        lm_stat, lm_p = lm_linearity_test(y, p=2, d=1)
        reset_stat, reset_p = reset_test(y, p=2)
        tsay_stat, tsay_p = tsay_test(y, p=2, d=1)
        nonlinear = (lm_p < 0.05) or (reset_p < 0.05) or (tsay_p < 0.05)

        rows.append(
            {
                "ticker": ticker,
                "lm_stat": round(lm_stat, 4),
                "lm_pvalue": round(lm_p, 4),
                "reset_stat": round(reset_stat, 4),
                "reset_pvalue": round(reset_p, 4),
                "tsay_stat": round(tsay_stat, 4),
                "tsay_pvalue": round(tsay_p, 4),
                "nonlinear": nonlinear,
            }
        )
        print(f"nonlinear={nonlinear}  lm_p={lm_p:.3f}")

    df = pd.DataFrame(rows)
    out_csv = output_dir / "linearity_tests.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nDiagnostics: {df['nonlinear'].sum()}/{len(df)} tickers non-linear")

    if token:
        upload_results(out_csv, "linearity_tests.csv", token)


if __name__ == "__main__":
    main()
