"""utils.py — Data loading and HuggingFace upload utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

RAW_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RAW_DATA_FILE = "master_data.parquet"
OUTPUT_REPO = "P2SAMAPA/p2-etf-smooth-transition-results"


def load_data(token: str | None = None) -> pd.DataFrame:
    """Load master data and return log-return DataFrame (tickers as columns)."""
    file_path = hf_hub_download(
        repo_id=RAW_DATA_REPO,
        filename=RAW_DATA_FILE,
        repo_type="dataset",
        token=token,
        cache_dir="./hf_cache",
    )
    df = pd.read_parquet(file_path)

    # Normalise to a flat DataFrame with Date as index
    # Case 1: DatetimeIndex already set
    if isinstance(df.index, pd.DatetimeIndex):
        df.index.name = "Date"
    # Case 2: Date is a column
    elif "Date" in df.columns:
        df = df.set_index("Date")
    # Case 3: index is named something else but is datetime-like
    else:
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Drop any non-numeric columns (e.g. macro text fields)
    df = df.select_dtypes(include=[np.number])

    # Forward-fill gaps
    prices = df.ffill()

    # Keep only columns where all prices are positive
    good_cols = (prices > 0).all(axis=0)
    prices = prices.loc[:, good_cols]

    # Log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    print(f"Loaded {len(log_returns)} rows × {len(log_returns.columns)} tickers")
    return log_returns


def upload_results(local_path: Path | str, remote_path: str, token: str) -> None:
    """Upload a file to the HF results dataset using reliable HfApi.upload_file."""
    local_path = Path(local_path)
    api = HfApi(token=token)

    # Ensure repo exists
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
    print(f"Uploaded {local_path} → {OUTPUT_REPO}/{remote_path}")
