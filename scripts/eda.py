#!/usr/bin/env python3
"""
HateMirage ICON 2026 — Exploratory Data Analysis
=================================================
Quick EDA on the sample dataset: class balance, comment lengths,
language distribution, and example rows.

Usage:
    python scripts/eda.py --data_path data/sample-data.xlsx
"""

import argparse
import io
import os
import re
import sys
from collections import Counter

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd


def detect_code_mixed(text: str) -> str:
    """Heuristic: if the text contains Devanagari characters, classify as code-mixed."""
    if not isinstance(text, str):
        return "unknown"
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    if devanagari_pattern.search(text):
        return "code-mixed"
    return "english"


def run_eda(data_path: str) -> None:
    # ---- Load data ----
    print(f"\n{'='*60}")
    print(f"  HateMirage EDA — {data_path}")
    print(f"{'='*60}\n")

    df = pd.read_excel(data_path)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}\n")

    # ---- Missing values ----
    print("--- Missing Values ---")
    print(df.isnull().sum().to_string())
    print()

    # ---- Column types ----
    print("--- Column Types ---")
    print(df.dtypes.to_string())
    print()

    # Drop rows with missing annotations for analysis
    required_cols = ["Index", "Comments", "Target", "Intent", "Implication"]
    available_cols = [c for c in required_cols if c in df.columns]
    df_clean = df[available_cols].dropna()
    print(f"Clean rows (no NaN in annotation columns): {len(df_clean)}\n")

    # ---- Target class balance ----
    if "Target" in df_clean.columns:
        print("--- Target Class Balance (Top 20) ---")
        # Targets can be comma-separated; split and count each individually
        all_targets = []
        for t in df_clean["Target"]:
            targets = [x.strip() for x in str(t).split(",")]
            all_targets.extend(targets)
        target_counts = Counter(all_targets)
        for target, count in target_counts.most_common(20):
            bar = "█" * min(count, 50)
            print(f"  {target:30s} {count:4d}  {bar}")
        print(f"  ... Total unique targets: {len(target_counts)}")
        print()

    # ---- Comment length distribution ----
    if "Comments" in df_clean.columns:
        print("--- Comment Length Statistics (characters) ---")
        lengths = df_clean["Comments"].astype(str).str.len()
        print(f"  Min:    {lengths.min()}")
        print(f"  Max:    {lengths.max()}")
        print(f"  Mean:   {lengths.mean():.1f}")
        print(f"  Median: {lengths.median():.1f}")
        print(f"  Std:    {lengths.std():.1f}")
        print()

        print("--- Comment Length Statistics (words) ---")
        word_counts = df_clean["Comments"].astype(str).str.split().str.len()
        print(f"  Min:    {word_counts.min()}")
        print(f"  Max:    {word_counts.max()}")
        print(f"  Mean:   {word_counts.mean():.1f}")
        print(f"  Median: {word_counts.median():.1f}")
        print(f"  Std:    {word_counts.std():.1f}")
        print()

    # ---- Language distribution ----
    if "Comments" in df_clean.columns:
        print("--- Language Distribution (heuristic) ---")
        df_clean = df_clean.copy()
        df_clean["language"] = df_clean["Comments"].apply(detect_code_mixed)
        lang_counts = df_clean["language"].value_counts()
        for lang, count in lang_counts.items():
            pct = count / len(df_clean) * 100
            print(f"  {lang:15s} {count:4d}  ({pct:.1f}%)")
        print()

    # ---- Intent / Implication text length ----
    for field in ["Intent", "Implication"]:
        if field in df_clean.columns:
            print(f"--- {field} Text Length (words) ---")
            wc = df_clean[field].astype(str).str.split().str.len()
            print(f"  Min: {wc.min()}, Max: {wc.max()}, Mean: {wc.mean():.1f}, Median: {wc.median():.1f}")
            print()

    # ---- Example rows ----
    print("--- Sample Rows (first 5) ---")
    for i, row in df_clean.head(5).iterrows():
        print(f"\n  [{row.get('Index', i)}]")
        comment = str(row.get("Comments", ""))
        print(f"  Comment:     {comment[:120]}{'...' if len(comment) > 120 else ''}")
        print(f"  Target:      {row.get('Target', 'N/A')}")
        print(f"  Intent:      {str(row.get('Intent', 'N/A'))[:100]}")
        print(f"  Implication: {str(row.get('Implication', 'N/A'))[:100]}")

    print(f"\n{'='*60}")
    print("  EDA Complete")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HateMirage EDA")
    parser.add_argument("--data_path", type=str, default="data/sample-data.xlsx",
                        help="Path to the dataset Excel file")
    args = parser.parse_args()
    run_eda(args.data_path)
