#!/usr/bin/env python3
import os
from pathlib import Path
from utils import upload_results

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not set. Cannot upload.")
        return

    # Upload model summaries
    for model_type in ["setar", "lstar", "estar"]:
        summary = Path(f"models/{model_type}/summary.json")
        if summary.exists():
            upload_results(summary, f"{model_type}/summary.json", token)

    # Upload backtest CSV
    bt = Path("backtest/backtest_summary.csv")
    if bt.exists():
        upload_results(bt, "backtest_summary.csv", token)

    # Upload diagnostics CSV
    diag = Path("diagnostics/linearity_tests.csv")
    if diag.exists():
        upload_results(diag, "linearity_tests.csv", token)

    print("✅ All results uploaded.")

if __name__ == "__main__":
    main()
