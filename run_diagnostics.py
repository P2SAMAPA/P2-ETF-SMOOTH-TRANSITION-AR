#!/usr/bin/env python3
import argparse, yaml, os
from pathlib import Path
import pandas as pd
from diagnostics import lm_linearity_test, reset_test, tsay_test
from utils import load_data, upload_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="diagnostics/")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    returns = load_data()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = []
    for universe in config["data"]["tickers"].values():
        tickers.extend(universe)
    tickers = sorted(set(tickers))

    diag = []
    for ticker in tickers:
        if ticker not in returns.columns:
            continue
        y = returns[ticker].dropna().values
        if len(y) < 200:
            continue
        lm, lm_p = lm_linearity_test(y, p=2, d=1)
        reset, reset_p = reset_test(y, p=2)
        tsay, tsay_p = tsay_test(y, p=2, d=1)
        diag.append({
            "ticker": ticker,
            "lm_stat": lm, "lm_pvalue": lm_p,
            "reset_stat": reset, "reset_pvalue": reset_p,
            "tsay_stat": tsay, "tsay_pvalue": tsay_p,
            "nonlinear": (lm_p < 0.05) or (reset_p < 0.05) or (tsay_p < 0.05)
        })
    df = pd.DataFrame(diag)
    df.to_csv(output_dir / "linearity_tests.csv", index=False)
    print(f"Diagnostics done. {df['nonlinear'].sum()}/{len(df)} tickers non‑linear.")

    token = os.environ.get("HF_TOKEN")
    if token:
        upload_results(output_dir / "linearity_tests.csv", "linearity_tests.csv", token)

if __name__ == "__main__":
    main()
