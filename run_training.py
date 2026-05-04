#!/usr/bin/env python3
"""run_training.py — Train SETAR / LSTAR / ESTAR models on all tickers."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import yaml

from estar import ESTAR
from lstar import LSTAR
from setar import SETAR
from utils import load_data, upload_results

MODEL_CLASSES = {"setar": SETAR, "lstar": LSTAR, "estar": ESTAR}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["setar", "lstar", "estar"], required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="models/")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    token = os.environ.get("HF_TOKEN")
    returns = load_data(token=token)

    output_dir = Path(args.output) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all tickers across universes
    tickers: list[str] = []
    for universe_tickers in config["data"]["tickers"].values():
        tickers.extend(universe_tickers)
    tickers = sorted(set(tickers))

    model_cfg = config["models"][args.model]
    cls = MODEL_CLASSES[args.model]
    results: dict = {}

    for ticker in tickers:
        if ticker not in returns.columns:
            print(f"  {ticker}: not in data — skipping")
            continue

        y = returns[ticker].dropna().values
        if len(y) < 500:
            print(f"  {ticker}: too short ({len(y)}) — skipping")
            continue

        for p in model_cfg["p"]:
            for d in model_cfg["d"]:
                key = f"{ticker}_p{p}_d{d}"
                print(f"Training {args.model.upper()} {key} ...", end=" ", flush=True)
                try:
                    model = cls(p=p, d=d)
                    model.fit(y)
                except Exception as e:
                    print(f"FAILED: {e}")
                    continue

                # Save pkl
                pkl_path = output_dir / f"{key}.pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump(model, f)

                # Build results dict
                entry: dict = {}
                if args.model == "setar":
                    entry = {
                        "params": {
                            "c": float(model.c_),
                            "phi1": model.phi1_.tolist(),
                            "phi2": model.phi2_.tolist(),
                        },
                        "regime_props": model.regime_props_,
                        "aic": float(model.aic_),
                        "bic": float(model.bic_),
                    }
                    print(
                        f"c={model.c_:.4f}  "
                        f"AIC={model.aic_:.1f}  "
                        f"regime1={model.regime_props_['regime1_pct']:.1%}"
                    )
                elif args.model == "lstar":
                    entry = {
                        "params": {
                            "gamma": float(model.gamma_),
                            "c": float(model.c_),
                            "phi1": model.phi1_.tolist(),
                            "phi2": model.phi2_.tolist(),
                        },
                        "sigma": float(model.sigma_),
                    }
                    print(f"gamma={model.gamma_:.4f}  c={model.c_:.4f}")
                elif args.model == "estar":
                    entry = {
                        "params": {
                            "gamma": float(model.gamma_),
                            "c": float(model.c_),
                            "phi1": model.phi1_.tolist(),
                            "phi2": model.phi2_.tolist(),
                        }
                    }
                    print(f"gamma={model.gamma_:.4f}  c={model.c_:.4f}")

                results[key] = entry

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} models → {summary_path}")

    if token:
        upload_results(summary_path, f"{args.model}/summary.json", token)


if __name__ == "__main__":
    main()
