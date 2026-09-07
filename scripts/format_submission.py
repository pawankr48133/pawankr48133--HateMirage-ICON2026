#!/usr/bin/env python3
"""
HateMirage ICON 2026 — Codabench Submission Formatter
======================================================
Takes the raw predictions CSV from run_baseline.py and creates
a SINGLE combined ZIP for Codabench (both Task A + Task B together).

Combined submission: predictions.csv with columns (id, target, intent, implication)

Usage:
    python scripts/format_submission.py --predictions outputs/val_predictions.csv
"""

import argparse
import os
import zipfile

import pandas as pd


def format_submission(predictions_path: str, output_dir: str = "outputs") -> None:
    """
    Create a single Codabench-ready ZIP with all predictions combined.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load predictions
    df = pd.read_csv(predictions_path)
    print(f"Loaded {len(df)} predictions from {predictions_path}")
    print(f"Columns: {list(df.columns)}\n")

    # --- Combined submission: all 3 fields in one file ---
    combined_csv = os.path.join(output_dir, "predictions.csv")
    combined_zip = os.path.join(output_dir, "submission.zip")

    combined_df = pd.DataFrame({
        "id": df["Index"],
        "target": df["Pred_Target"].fillna(""),
        "intent": df["Pred_Intent"].fillna(""),
        "implication": df["Pred_Implication"].fillna(""),
    })
    combined_df.to_csv(combined_csv, index=False)

    with zipfile.ZipFile(combined_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(combined_csv, "predictions.csv")  # must be named predictions.csv inside ZIP

    print(f"Combined submission created:")
    print(f"  CSV: {combined_csv}")
    print(f"  ZIP: {combined_zip}")
    print(f"  Rows: {len(combined_df)}")
    print(f"\n  Sample rows:")
    print(combined_df.head(3).to_string(index=False))

    print(f"\n{'='*50}")
    print(f"  Upload this ONE file to Codabench:")
    print(f"  >>> {combined_zip}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format Codabench Submission")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions CSV from run_baseline.py")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for ZIP file")
    args = parser.parse_args()
    format_submission(args.predictions, args.output_dir)
