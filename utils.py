"""utils.py — Data loading and HuggingFace upload utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

RAW_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RAW_DATA_FILE = "master_data.parquet"
OUTPUT_REPO = "P2SAMAPA/p2-etf-smooth-transition-results"

# Known ETF tickers in the master data
ETF_TICKERS = [
    "TLT",
    "VCIT",
    "LQD",
    "HYG",
    "VNQ",
    "GLD",
    "SLV",
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
]


def load_data(token: str | None = None) -> pd.DataFrame:
    """Load master data and return log-return DataFrame (ETF tickers as columns)."""
    file_path = hf_hub_download(
        repo_id=RAW_DATA_REPO,
        filename=RAW_DATA_FILE,
        repo_type="dataset",
        token=token,
        cache_dir="./hf_cache",
    )
    df = pd.read_parquet(file_path)

    # Normalise index
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    prices = df.set_index("Date")

    # Keep only known ETF ticker columns — avoids macro cols causing dropna to kill rows
    available = [t for t in ETF_TICKERS if t in prices.columns]
    prices = prices[available].ffill()

    log_returns = np.log(prices / prices.shift(1)).dropna()
    print(f"Loaded {len(log_returns)} rows x {len(log_returns.columns)} tickers")
    return log_returns


def upload_results(local_path: Path | str, remote_path: str, token: str) -> None:
    """Upload a single file to the HF results dataset."""
    local_path = Path(local_path)
    api = HfApi(token=token)

    api.create_repo(
        repo_id=OUTPUT_REPO,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=remote_path,
        repo_id=OUTPUT_REPO,
        repo_type="dataset",
        commit_message=f"Update {remote_path}",
    )
    print(f"Uploaded {local_path} -> {OUTPUT_REPO}/{remote_path}")


def upload_folder(local_dir: Path | str, remote_dir: str, token: str) -> None:
    """Upload all files in a directory as a single commit — avoids HF 500 errors."""
    local_dir = Path(local_dir)
    api = HfApi(token=token)

    api.create_repo(
        repo_id=OUTPUT_REPO,
        repo_type="dataset",
        exist_ok=True,
        private=False,
    )
    api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=remote_dir,
        repo_id=OUTPUT_REPO,
        repo_type="dataset",
        commit_message=f"Update {remote_dir}",
    )
    print(f"Uploaded folder {local_dir} -> {OUTPUT_REPO}/{remote_dir}")
