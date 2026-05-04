#!/usr/bin/env python3
"""upload_results.py — Upload all results to HuggingFace."""

from __future__ import annotations

import os
from pathlib import Path

from utils import upload_results


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set — cannot upload.")
        return

    uploads = [
        (Path("diagnostics/linearity_tests.csv"), "linearity_tests.csv"),
        (Path("backtest/backtest_summary.csv"), "backtest_summary.csv"),
    ]
    for model_type in ["setar", "lstar", "estar"]:
        uploads.append(
            (
                Path(f"models/{model_type}/summary.json"),
                f"{model_type}/summary.json",
            )
        )
        for pkl in Path(f"models/{model_type}").glob("*.pkl"):
            uploads.append((pkl, f"models/{model_type}/{pkl.name}"))

    for local, remote in uploads:
        if local.exists():
            upload_results(local, remote, token)
        else:
            print(f"Skipping {local} — not found")

    print("Upload complete.")


if __name__ == "__main__":
    main()
