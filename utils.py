import pandas as pd
import numpy as np
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, HfFileSystem

RAW_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RAW_DATA_FILE = "master_data.parquet"
OUTPUT_REPO = "P2SAMAPA/p2-etf-smooth-transition-results"

def load_data(token=None):
    file_path = hf_hub_download(
        repo_id=RAW_DATA_REPO,
        filename=RAW_DATA_FILE,
        repo_type="dataset",
        token=token
    )
    df = pd.read_parquet(file_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    if 'Ticker' in df.columns:
        df_wide = df.pivot(columns='Ticker', values='Close')
    else:
        df_wide = df
    rets = np.log(df_wide / df_wide.shift(1)).dropna()
    return rets

def upload_results(local_path, remote_path, token):
    fs = HfFileSystem(token=token)
    with fs.open(f"datasets/{OUTPUT_REPO}/{remote_path}", "w") as f:
        if str(local_path).endswith('.json'):
            with open(local_path, 'r') as src:
                f.write(src.read())
        elif str(local_path).endswith('.csv'):
            with open(local_path, 'rb') as src:
                f.write(src.read())
        else:
            with open(local_path, 'rb') as src:
                f.write(src.read())
    print(f"Uploaded {local_path} -> {OUTPUT_REPO}/{remote_path}")
