#!/usr/bin/env python3
import argparse
import pickle
import yaml
import os
import numpy as np
from pathlib import Path
import pandas as pd
from backtest import WalkForwardBacktest
from utils import load_data, upload_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="models/")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="backtest/")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    returns = load_data()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    wf = WalkForwardBacktest(window_size=252, step_size=63)  # fixed, ignore config for now
    all_rows = []

    for model_type in ["setar", "lstar", "estar"]:
        model_dir = Path(args.models) / model_type
        if not model_dir.exists():
            print(f"Skipping {model_type} – no directory")
            continue
        for pkl_file in model_dir.glob("*.pkl"):
            # filename: e.g., IWF_p2_d1.pkl
            parts = pkl_file.stem.split('_')
            if len(parts) < 3:
                print(f"Skipping {pkl_file.name}: malformed name")
                continue
            ticker = parts[0]
            try:
                p = int(parts[1][1:])  # 'p2' -> 2
                d = int(parts[2][1:])  # 'd1' -> 1
            except:
                print(f"Skipping {pkl_file.name}: cannot parse p/d")
                continue

            if ticker not in returns.columns:
                print(f"Skipping {ticker}: not in returns data")
                continue

            y = returns[ticker].dropna().values
            if len(y) < 252 + 63:
                print(f"Skipping {ticker}: series too short ({len(y)})")
                continue

            print(f"Backtesting {model_type} {ticker} p={p} d={d} ...")
            with open(pkl_file, "rb") as f:
                model = pickle.load(f)

            predictions = []
            actuals = []
            T = len(y)
            window = 252
            step = 63
            for start in range(0, T - window - step, step):
                end = start + window
                train = y[start:end]
                # Re‑create a fresh model instance
                new_model = type(model)(p=p, d=d)
                new_model.fit(train)
                for i in range(step):
                    if end + i < T:
                        hist = y[:end + i]
                        pred = new_model.predict(hist)
                        predictions.append(pred)
                        actuals.append(y[end + i])
            if len(predictions) > 0:
                metrics = wf.evaluate(np.array(predictions), np.array(actuals))
                all_rows.append({
                    "model_type": model_type,
                    "ticker": ticker,
                    "p": p,
                    "d": d,
                    "sharpe": metrics["sharpe"],
                    "hit_rate": metrics["hit_rate"],
                    "max_drawdown": metrics["max_drawdown"],
                    "mean_return": metrics["mean_return"],
                    "volatility": metrics["volatility"]
                })
                print(f"  -> Added {len(predictions)} predictions, Sharpe={metrics['sharpe']:.2f}")
            else:
                print(f"  -> No predictions generated")

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(output_dir / "backtest_summary.csv", index=False)
        print(f"Saved {len(df)} rows to backtest_summary.csv")
    else:
        print("No backtest results were generated. Check model files and data.")

    token = os.environ.get("HF_TOKEN")
    if token and output_dir.exists() and (output_dir / "backtest_summary.csv").exists():
        upload_results(output_dir / "backtest_summary.csv", "backtest_summary.csv", token)

if __name__ == "__main__":
    main()
