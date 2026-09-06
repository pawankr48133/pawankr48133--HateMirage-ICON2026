#!/usr/bin/env python3
"""
HateMirage ICON 2026 — Codabench Submission Formatter
======================================================
Takes the raw predictions CSV from run_baseline.py and creates
properly formatted ZIP files for Codabench submission.

Task A submission: predictions.csv with columns (id, target)
Task B submission: predictions.csv with columns (id, intent, implication)

Usage:
    python scripts/format_submission.py --predictions outputs/zero-shot_vanilla_results.csv
"""

import argparse
import os
import zipfile

import pandas as pd


def format_submission(predictions_path: str, output_dir: str = "outputs") -> None:
    """
    Create Codabench-ready ZIP files from raw predictions.

    Args:
        predictions_path: path to CSV from run_baseline.py
        output_dir: where to save the ZIP files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load predictions
    df = pd.read_csv(predictions_path)
    print(f"Loaded {len(df)} predictions from {predictions_path}")
    print(f"Columns: {list(df.columns)}\n")

    # --- Task A: Target Identification ---
    task_a_csv = os.path.join(output_dir, "predictions_task_a.csv")
    task_a_zip = os.path.join(output_dir, "submission_task_a.zip")

    task_a_df = pd.DataFrame({
        "id": df["Index"],
        "target": df["Pred_Target"].fillna(""),
    })
    task_a_df.to_csv(task_a_csv, index=False)

    with zipfile.ZipFile(task_a_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(task_a_csv, "predictions.csv")  # must be named predictions.csv inside ZIP

    print(f"Task A submission created:")
    print(f"  CSV: {task_a_csv}")
    print(f"  ZIP: {task_a_zip}")
    print(f"  Rows: {len(task_a_df)}")
    print(f"  Sample:")
    print(task_a_df.head(3).to_string(index=False))
    print()

    # --- Task B: Intent & Implication Generation ---
    task_b_csv = os.path.join(output_dir, "predictions_task_b.csv")
    task_b_zip = os.path.join(output_dir, "submission_task_b.zip")

    task_b_df = pd.DataFrame({
        "id": df["Index"],
        "intent": df["Pred_Intent"].fillna(""),
        "implication": df["Pred_Implication"].fillna(""),
    })
    task_b_df.to_csv(task_b_csv, index=False)

    with zipfile.ZipFile(task_b_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(task_b_csv, "predictions.csv")  # must be named predictions.csv inside ZIP

    print(f"Task B submission created:")
    print(f"  CSV: {task_b_csv}")
    print(f"  ZIP: {task_b_zip}")
    print(f"  Rows: {len(task_b_df)}")
    print(f"  Sample:")
    print(task_b_df.head(3).to_string(index=False))
    print()

    print(f"{'='*50}")
    print(f"  Ready to submit on Codabench!")
    print(f"  Task A ZIP: {task_a_zip}")
    print(f"  Task B ZIP: {task_b_zip}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format Codabench Submission")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions CSV from run_baseline.py")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for ZIP files")
    args = parser.parse_args()
    format_submission(args.predictions, args.output_dir)
