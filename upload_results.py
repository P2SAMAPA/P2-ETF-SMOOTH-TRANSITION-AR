import os, glob
from pathlib import Path
from utils import upload_results

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set")
        return
    # Upload all JSON and CSV files from models/, backtest/, diagnostics/
    for f in Path("models").glob("**/*.json"):
        upload_results(f, str(f), token)
    for f in Path("models").glob("**/*.pkl"):
        upload_results(f, str(f), token)  # optional
    for f in Path("backtest").glob("*.csv"):
        upload_results(f, f.name, token)
    for f in Path("diagnostics").glob("*.csv"):
        upload_results(f, f.name, token)

if __name__ == "__main__":
    main()
