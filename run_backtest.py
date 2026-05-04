#!/usr/bin/env python3
import argparse
import pickle
import yaml
import os
import numpy as np  # <-- added
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

    wf = WalkForwardBacktest(window_size=config["backtest"]["window_size"], step_size=config["backtest"]["step_size"])
    all_rows = []

    for model_type in ["setar", "lstar", "estar"]:
        model_dir = Path(args.models) / model_type
        if not model_dir.exists():
            continue
        for pkl_file in model_dir.glob("*.pkl"):
            parts = pkl_file.stem.split('_')
            p = int([x for x in parts if x.startswith('p')][0][1:])
            d = int([x for x in parts if x.startswith('d')][0][1:])
            ticker = parts[0]
            if ticker not in returns.columns:
                continue
            y = returns[ticker].dropna().values
            # skip if any inf or nan (could happen with bad data)
            if not np.isfinite(y).all():
                print(f"Skipping {ticker} due to non-finite returns")
                continue
            # reload model class
            with open(pkl_file, "rb") as f:
                model = pickle.load(f)
            # walk‑forward
            predictions = []
            actuals = []
            T = len(y)
            window = config["backtest"]["window_size"]
            step = config["backtest"]["step_size"]
            for start in range(0, T - window - step, step):
                end = start + window
                train = y[start:end]
                new_model = type(model)(p=p, d=d)
                new_model.fit(train)
                for i in range(step):
                    if end + i < len(y):
                        hist = y[:end + i]
                        pred = new_model.predict(hist)
                        predictions.append(pred)
                        actuals.append(y[end + i])
            if predictions:
                metrics = wf.evaluate(np.array(predictions), np.array(actuals))
                all_rows.append({
                    "model_type": model_type, "ticker": ticker, "p": p, "d": d,
                    **metrics
                })
    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "backtest_summary.csv", index=False)
    print(f"Backtest complete. {len(df)} entries.")

    token = os.environ.get("HF_TOKEN")
    if token:
        upload_results(output_dir / "backtest_summary.csv", "backtest_summary.csv", token)

if __name__ == "__main__":
    main()
