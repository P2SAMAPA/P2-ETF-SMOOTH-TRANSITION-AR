#!/usr/bin/env python3
import argparse, json, pickle, yaml, os
from pathlib import Path
from setar import SETAR
from lstar import LSTAR
from estar import ESTAR
from utils import load_data, upload_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["setar","lstar","estar"], required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="models/")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    returns = load_data()
    output_dir = Path(args.output) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    model_cfg = config["models"][args.model]
    # build all tickers from universes
    tickers = []
    for universe in config["data"]["tickers"].values():
        tickers.extend(universe)
    tickers = sorted(set(tickers))

    for ticker in tickers:
        if ticker not in returns.columns:
            continue
        y = returns[ticker].dropna().values
        if len(y) < 500:
            continue
        for p in model_cfg["p"]:
            for d in model_cfg["d"]:
                print(f"Training {args.model} on {ticker} p={p} d={d}")
                if args.model == "setar":
                    model = SETAR(p=p, d=d)
                    model.fit(y)
                    key = f"{ticker}_p{p}_d{d}"
                    results[key] = {
                        "params": {"c": float(model.c_), "phi1": model.phi1_.tolist(), "phi2": model.phi2_.tolist()},
                        "regime_props": model.regime_props_,
                        "aic": model.aic_, "bic": model.bic_
                    }
                    with open(output_dir / f"{key}.pkl", "wb") as f:
                        pickle.dump(model, f)
                elif args.model == "lstar":
                    model = LSTAR(p=p, d=d)
                    model.fit(y)
                    key = f"{ticker}_p{p}_d{d}"
                    results[key] = {
                        "params": {"gamma": float(model.gamma_), "c": float(model.c_), "phi1": model.phi1_.tolist(), "phi2": model.phi2_.tolist()},
                        "sigma": float(model.sigma_)
                    }
                    with open(output_dir / f"{key}.pkl", "wb") as f:
                        pickle.dump(model, f)
                elif args.model == "estar":
                    model = ESTAR(p=p, d=d)
                    model.fit(y)
                    key = f"{ticker}_p{p}_d{d}"
                    results[key] = {
                        "params": {"gamma": float(model.gamma_), "c": float(model.c_), "phi1": model.phi1_.tolist(), "phi2": model.phi2_.tolist()}
                    }
                    with open(output_dir / f"{key}.pkl", "wb") as f:
                        pickle.dump(model, f)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    token = os.environ.get("HF_TOKEN")
    if token:
        upload_results(output_dir / "summary.json", f"{args.model}/summary.json", token)

if __name__ == "__main__":
    main()
