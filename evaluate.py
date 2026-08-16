#!/usr/bin/env python3
"""
HateMirage ICON 2026 — Competition Evaluation Script
=====================================================
Standalone evaluation matching the exact competition scoring:
  - Sentence-BERT cosine similarity
  - ROUGE-L F1

Scoring rules:
  Task A: both metrics on Target. Final = mean(SBERT, ROUGE-L)
  Task B: both metrics on Intent & Implication, averaged.
          Final = mean(avg_SBERT, avg_ROUGE-L)

Usage:
    python evaluate.py --predictions outputs/results.csv --gold data/sample-data.xlsx
    python evaluate.py --predictions outputs/results.csv --gold data/sample-data.xlsx --task A
    python evaluate.py --predictions outputs/results.csv --gold data/sample-data.xlsx --task B
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util


# =============================================================================
# Metric Computation
# =============================================================================
def compute_sbert_similarity(predictions: list[str], references: list[str],
                             model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> list[float]:
    """Compute per-sample Sentence-BERT cosine similarity."""
    model = SentenceTransformer(model_name)

    pred_embeddings = model.encode(predictions, convert_to_tensor=True, show_progress_bar=False)
    ref_embeddings = model.encode(references, convert_to_tensor=True, show_progress_bar=False)

    # Pairwise cosine similarity (diagonal of the similarity matrix)
    similarities = []
    for i in range(len(predictions)):
        sim = util.cos_sim(pred_embeddings[i], ref_embeddings[i]).item()
        similarities.append(sim)

    return similarities


def compute_rouge_l(predictions: list[str], references: list[str]) -> list[float]:
    """Compute per-sample ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        result = scorer.score(str(ref), str(pred))
        scores.append(result['rougeL'].fmeasure)
    return scores


# =============================================================================
# Main Evaluation
# =============================================================================
def evaluate(predictions_path: str, gold_path: str, task: str = "both",
             sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
             output_path: str = None) -> dict:
    """
    Run evaluation matching competition scoring.

    Args:
        predictions_path: CSV with columns Pred_Target, Pred_Intent, Pred_Implication
        gold_path: xlsx or CSV with columns Target, Intent, Implication
        task: "A" (Target only), "B" (Intent + Implication), or "both"
        sbert_model: Sentence-BERT model for similarity computation
        output_path: optional path to save JSON results

    Returns:
        dict with all scores
    """
    # --- Load predictions ---
    pred_df = pd.read_csv(predictions_path)

    # --- Load gold ---
    if gold_path.endswith(".xlsx"):
        gold_df = pd.read_excel(gold_path)
    else:
        gold_df = pd.read_csv(gold_path)

    # --- Align on Index if available ---
    if "Index" in pred_df.columns and "Index" in gold_df.columns:
        merged = pred_df.merge(gold_df[["Index", "Target", "Intent", "Implication"]],
                               on="Index", how="inner", suffixes=("_pred", "_gold"))
        # Handle column naming: if predictions already have Gold_ prefix
        if "Gold_Target" in pred_df.columns:
            gold_target = pred_df["Gold_Target"].astype(str).tolist()
            gold_intent = pred_df["Gold_Intent"].astype(str).tolist()
            gold_implication = pred_df["Gold_Implication"].astype(str).tolist()
        else:
            gold_target = merged["Target"].astype(str).tolist()
            gold_intent = merged["Intent"].astype(str).tolist()
            gold_implication = merged["Implication"].astype(str).tolist()
    else:
        # Assume same order
        gold_target = gold_df["Target"].astype(str).tolist()
        gold_intent = gold_df["Intent"].astype(str).tolist()
        gold_implication = gold_df["Implication"].astype(str).tolist()

    pred_target = pred_df["Pred_Target"].fillna("").astype(str).tolist()
    pred_intent = pred_df["Pred_Intent"].fillna("").astype(str).tolist()
    pred_implication = pred_df["Pred_Implication"].fillna("").astype(str).tolist()

    n = len(pred_target)
    print(f"\nEvaluating {n} samples...")
    print(f"SBERT model: {sbert_model}\n")

    results = {"n_samples": n}

    # --- Task A: Target ---
    if task in ["A", "both"]:
        print("Computing Task A (Target) metrics...")
        target_sbert = compute_sbert_similarity(pred_target, gold_target, sbert_model)
        target_rouge = compute_rouge_l(pred_target, gold_target)

        avg_target_sbert = float(np.mean(target_sbert))
        avg_target_rouge = float(np.mean(target_rouge))
        task_a_final = (avg_target_sbert + avg_target_rouge) / 2

        results["task_a"] = {
            "target_sbert_sim": avg_target_sbert,
            "target_rouge_l": avg_target_rouge,
            "final_score": task_a_final,
        }

    # --- Task B: Intent + Implication ---
    if task in ["B", "both"]:
        print("Computing Task B (Intent + Implication) metrics...")

        intent_sbert = compute_sbert_similarity(pred_intent, gold_intent, sbert_model)
        intent_rouge = compute_rouge_l(pred_intent, gold_intent)

        impl_sbert = compute_sbert_similarity(pred_implication, gold_implication, sbert_model)
        impl_rouge = compute_rouge_l(pred_implication, gold_implication)

        avg_intent_sbert = float(np.mean(intent_sbert))
        avg_intent_rouge = float(np.mean(intent_rouge))
        avg_impl_sbert = float(np.mean(impl_sbert))
        avg_impl_rouge = float(np.mean(impl_rouge))

        # Average Intent and Implication, then average the two metrics
        avg_sbert = (avg_intent_sbert + avg_impl_sbert) / 2
        avg_rouge = (avg_intent_rouge + avg_impl_rouge) / 2
        task_b_final = (avg_sbert + avg_rouge) / 2

        results["task_b"] = {
            "intent_sbert_sim": avg_intent_sbert,
            "intent_rouge_l": avg_intent_rouge,
            "implication_sbert_sim": avg_impl_sbert,
            "implication_rouge_l": avg_impl_rouge,
            "avg_sbert_sim": avg_sbert,
            "avg_rouge_l": avg_rouge,
            "final_score": task_b_final,
        }

    # --- Print results ---
    print(f"\n{'='*62}")
    if "task_a" in results:
        ta = results["task_a"]
        print(f"  Task A — Target Identification")
        print(f"  {'─'*58}")
        print(f"  {'Field':<20} {'SBERT Sim':>12} {'ROUGE-L':>12} {'Final':>12}")
        print(f"  {'─'*58}")
        print(f"  {'Target':<20} {ta['target_sbert_sim']:>12.4f} {ta['target_rouge_l']:>12.4f} {ta['final_score']:>12.4f}")
        print()

    if "task_b" in results:
        tb = results["task_b"]
        print(f"  Task B — Intent & Implication Generation")
        print(f"  {'─'*58}")
        print(f"  {'Field':<20} {'SBERT Sim':>12} {'ROUGE-L':>12} {'Final':>12}")
        print(f"  {'─'*58}")
        print(f"  {'Intent':<20} {tb['intent_sbert_sim']:>12.4f} {tb['intent_rouge_l']:>12.4f} {'—':>12}")
        print(f"  {'Implication':<20} {tb['implication_sbert_sim']:>12.4f} {tb['implication_rouge_l']:>12.4f} {'—':>12}")
        print(f"  {'─'*58}")
        print(f"  {'Average (Task B)':<20} {tb['avg_sbert_sim']:>12.4f} {tb['avg_rouge_l']:>12.4f} {tb['final_score']:>12.4f}")
        print()

    print(f"{'='*62}\n")

    # --- Save JSON ---
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

    return results


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="HateMirage Evaluation Script")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions CSV")
    parser.add_argument("--gold", type=str, required=True,
                        help="Path to gold labels (xlsx or csv)")
    parser.add_argument("--task", type=str, choices=["A", "B", "both"],
                        default="both", help="Which task to evaluate")
    parser.add_argument("--sbert_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-BERT model for similarity")
    parser.add_argument("--output", type=str, default="outputs/eval_results.json",
                        help="Path to save JSON results")
    args = parser.parse_args()

    evaluate(
        predictions_path=args.predictions,
        gold_path=args.gold,
        task=args.task,
        sbert_model=args.sbert_model,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
