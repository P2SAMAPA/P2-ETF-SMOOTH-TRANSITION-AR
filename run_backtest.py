#!/usr/bin/env python3
"""run_backtest.py — Walk-forward backtest using saved model parameters."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from backtest import WalkForwardBacktest
from estar import ESTAR
from lstar import LSTAR
from setar import SETAR
from utils import load_data, upload_results

MODEL_CLASSES = {"setar": SETAR, "lstar": LSTAR, "estar": ESTAR}


def restore_model(model_type: str, p: int, d: int, params: dict):
    """Reconstruct a fitted model from saved parameters (no pkl needed)."""
    cls = MODEL_CLASSES[model_type]
    model = cls(p=p, d=d)

    if model_type == "setar":
        import numpy as np
        from sklearn.linear_model import LinearRegression

        model.c_ = params["c"]
        model.phi1_ = np.array(params["phi1"])
        model.phi2_ = np.array(params["phi2"])
        # Rebuild dummy regressors so predict() works
        model.reg1_ = LinearRegression(fit_intercept=False)
        model.reg1_.coef_ = model.phi1_
        model.reg2_ = LinearRegression(fit_intercept=False)
        model.reg2_.coef_ = model.phi2_

    elif model_type in ("lstar", "estar"):
        model.gamma_ = params["gamma"]
        model.c_ = params["c"]
        model.phi1_ = np.array(params["phi1"])
        model.phi2_ = np.array(params["phi2"])

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="models/")
    parser.add_argument("--output", default="backtest/")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")

    # Download model summaries from HF (training jobs upload there, not to disk)
    from huggingface_hub import hf_hub_download
    from utils import OUTPUT_REPO

    for model_type in ["setar", "lstar", "estar"]:
        local_dir = Path(args.models) / model_type
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            path = hf_hub_download(
                repo_id=OUTPUT_REPO,
                filename=f"{model_type}/summary.json",
                repo_type="dataset",
                token=token,
                local_dir=".",
            )
            import shutil

            shutil.copy(path, local_dir / "summary.json")
            print(f"Downloaded {model_type}/summary.json")
        except Exception as e:
            print(f"Could not download {model_type}/summary.json: {e}")

    returns = load_data(token=token)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    wf = WalkForwardBacktest(window_size=252, step_size=63)
    all_rows: list[dict] = []

    for model_type in ["setar", "lstar", "estar"]:
        summary_path = Path(args.models) / model_type / "summary.json"
        if not summary_path.exists():
            print(f"No summary for {model_type} — skipping")
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        print(f"\n{'='*50}")
        print(f"Backtesting {model_type.upper()} ({len(summary)} models)")
        print(f"{'='*50}")

        for key, meta in summary.items():
            # key format: TICKER_pP_dD
            parts = key.split("_")
            if len(parts) < 3:
                continue
            ticker = parts[0]
            try:
                p = int(parts[1][1:])
                d = int(parts[2][1:])
            except (ValueError, IndexError):
                print(f"  Skipping {key}: cannot parse p/d")
                continue

            if ticker not in returns.columns:
                print(f"  Skipping {ticker}: not in returns")
                continue

            y = returns[ticker].dropna().values
            if len(y) < 252 + 63:
                print(f"  Skipping {ticker}: series too short ({len(y)})")
                continue

            params = meta.get("params", {})

            print(f"  {ticker} p={p} d={d} ...", end=" ", flush=True)
            predictions, actuals = wf.run(
                model_type=model_type, p=p, d=d, params=params, returns=y
            )

            if len(predictions) == 0:
                print("no predictions")
                continue

            metrics = wf.evaluate(predictions, actuals)
            all_rows.append(
                {
                    "model_type": model_type,
                    "ticker": ticker,
                    "p": p,
                    "d": d,
                    "sharpe": round(metrics["sharpe"], 4),
                    "hit_rate": round(metrics["hit_rate"], 4),
                    "max_drawdown": round(metrics["max_drawdown"], 4),
                    "mean_return": round(metrics["mean_return"], 4),
                    "volatility": round(metrics["volatility"], 4),
                }
            )
            print(
                f"Sharpe={metrics['sharpe']:.2f}  "
                f"Hit={metrics['hit_rate']:.1%}  "
                f"MDD={metrics['max_drawdown']:.1%}"
            )

    if all_rows:
        df = pd.DataFrame(all_rows)
        out_csv = output_dir / "backtest_summary.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nSaved {len(df)} rows → {out_csv}")
        if token:
            upload_results(out_csv, "backtest_summary.csv", token)
    else:
        print("\nNo backtest results generated.")


if __name__ == "__main__":
    main()
