#!/usr/bin/env python3
"""
HateMirage ICON 2026 — Results Table Generator
================================================
Reads experiment_log.csv and produces formatted comparison tables
in Markdown and LaTeX.

Usage:
    python scripts/generate_results_table.py --log outputs/experiment_log.csv
"""

import argparse
import os
import sys

import pandas as pd


def generate_markdown_table(df: pd.DataFrame) -> str:
    """Generate a markdown comparison table."""
    lines = []

    lines.append("# HateMirage — Experiment Results Comparison\n")
    lines.append("## Task A — Target Identification\n")
    lines.append("| # | Model | Mode | Prompt | SBERT Sim | ROUGE-L | **Task A Score** |")
    lines.append("|---|-------|------|--------|-----------|---------|------------------|")

    # Sort by task_a_score descending
    df_sorted = df.sort_values("task_a_score", ascending=False) if "task_a_score" in df.columns else df

    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        model_short = row.get("model_id", "").split("/")[-1]
        lines.append(
            f"| {i} | {model_short} | {row.get('mode', '')} | {row.get('prompt_variant', '')} "
            f"| {row.get('target_sbert', 'N/A'):.4f} "
            f"| {row.get('target_rouge_l', 'N/A'):.4f} "
            f"| **{row.get('task_a_score', 'N/A'):.4f}** |"
        )

    lines.append("")
    lines.append("## Task B — Intent & Implication Generation\n")
    lines.append(
        "| # | Model | Mode | Prompt "
        "| Intent SBERT | Intent ROUGE-L "
        "| Impl SBERT | Impl ROUGE-L "
        "| **Task B Score** |"
    )
    lines.append(
        "|---|-------|------|--------"
        "|--------------|----------------"
        "|------------|---------------"
        "|------------------|"
    )

    df_sorted_b = df.sort_values("task_b_score", ascending=False) if "task_b_score" in df.columns else df

    for i, (_, row) in enumerate(df_sorted_b.iterrows(), 1):
        model_short = row.get("model_id", "").split("/")[-1]
        lines.append(
            f"| {i} | {model_short} | {row.get('mode', '')} | {row.get('prompt_variant', '')} "
            f"| {row.get('intent_sbert', 'N/A'):.4f} "
            f"| {row.get('intent_rouge_l', 'N/A'):.4f} "
            f"| {row.get('impl_sbert', 'N/A'):.4f} "
            f"| {row.get('impl_rouge_l', 'N/A'):.4f} "
            f"| **{row.get('task_b_score', 'N/A'):.4f}** |"
        )

    lines.append("")
    lines.append(f"*Generated from {len(df)} experiments.*\n")

    return "\n".join(lines)


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate a LaTeX comparison table for the paper."""
    lines = []

    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Comparison of systems on HateMirage Tasks A and B.}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{llll|cc|c|cccc|c}")
    lines.append(r"\toprule")
    lines.append(
        r"\# & Model & Mode & Prompt & \multicolumn{2}{c|}{Task A} & Score A "
        r"& \multicolumn{4}{c|}{Task B} & Score B \\"
    )
    lines.append(
        r" & & & & SBERT & ROUGE-L & & Int-SB & Int-RL & Imp-SB & Imp-RL & \\"
    )
    lines.append(r"\midrule")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        model_short = row.get("model_id", "").split("/")[-1]
        model_short = model_short.replace("_", r"\_")
        lines.append(
            f"{i} & {model_short} & {row.get('mode', '')} & {row.get('prompt_variant', '')} "
            f"& {row.get('target_sbert', 0):.4f} & {row.get('target_rouge_l', 0):.4f} "
            f"& \\textbf{{{row.get('task_a_score', 0):.4f}}} "
            f"& {row.get('intent_sbert', 0):.4f} & {row.get('intent_rouge_l', 0):.4f} "
            f"& {row.get('impl_sbert', 0):.4f} & {row.get('impl_rouge_l', 0):.4f} "
            f"& \\textbf{{{row.get('task_b_score', 0):.4f}}} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate Results Tables")
    parser.add_argument("--log", type=str, default="outputs/experiment_log.csv",
                        help="Path to experiment log CSV")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save generated tables")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"Error: {args.log} not found. Run experiments first.")
        sys.exit(1)

    df = pd.read_csv(args.log)
    print(f"Loaded {len(df)} experiment results from {args.log}\n")

    # Generate Markdown
    md = generate_markdown_table(df)
    md_path = os.path.join(args.output_dir, "results_comparison.md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown table saved to {md_path}")
    print(md)

    # Generate LaTeX
    tex = generate_latex_table(df)
    tex_path = os.path.join(args.output_dir, "results_comparison.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"\nLaTeX table saved to {tex_path}")


if __name__ == "__main__":
    main()
