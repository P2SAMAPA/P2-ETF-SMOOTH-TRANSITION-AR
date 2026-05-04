#!/usr/bin/env python3
import os
from pathlib import Path
from huggingface_hub import HfApi, HfFileSystem

OUTPUT_REPO = "P2SAMAPA/p2-etf-smooth-transition-results"

def upload_file(local_path, remote_path, token):
    fs = HfFileSystem(token=token)
    remote_full = f"datasets/{OUTPUT_REPO}/{remote_path}"
    # Determine if binary or text
    local_path = Path(local_path)
    is_text = local_path.suffix in ('.json', '.csv', '.txt', '.yaml')
    mode = 'w' if is_text else 'wb'
    with fs.open(remote_full, mode) as f:
        if is_text:
            with open(local_path, 'r') as src:
                f.write(src.read())
        else:
            with open(local_path, 'rb') as src:
                f.write(src.read())
    print(f"✅ Uploaded {local_path} -> {remote_full}")

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not set. Cannot upload.")
        return

    # Test token validity by listing the repo
    api = HfApi()
    try:
        api.list_files(repo_id=OUTPUT_REPO, repo_type="dataset", token=token)
        print(f"✅ Token is valid, can access {OUTPUT_REPO}")
    except Exception as e:
        print(f"❌ Token failed: {e}")
        return

    # Upload model summaries
    for model_type in ["setar", "lstar", "estar"]:
        summary = Path(f"models/{model_type}/summary.json")
        if summary.exists():
            upload_file(summary, f"{model_type}/summary.json", token)

    # Upload backtest CSV
    bt = Path("backtest/backtest_summary.csv")
    if bt.exists():
        upload_file(bt, "backtest_summary.csv", token)

    # Upload diagnostics CSV
    diag = Path("diagnostics/linearity_tests.csv")
    if diag.exists():
        upload_file(diag, "linearity_tests.csv", token)

    print("✅ All results uploaded.")

if __name__ == "__main__":
    main()
