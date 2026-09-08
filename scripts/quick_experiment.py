#!/usr/bin/env python3
"""
HateMirage ICON 2026 — Quick Experiment Comparator
====================================================
Runs strategies 1 (RAG), 2 (Qwen3-8B), 3 (Few-shot) on a 100-row
subset of training data and compares scores.

Usage (on Colab):
    python scripts/quick_experiment.py --config config.yaml
"""

import argparse
import os
import sys
import time

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_subset(config: dict, n: int = 100, seed: int = 42) -> str:
    """Create a 100-row subset from training data for quick testing."""
    train_path = config["data"]["train_path"]
    df = pd.read_excel(train_path)
    df = df[["Index", "Comments", "Target", "Intent", "Implication"]].dropna()

    subset = df.sample(n=min(n, len(df)), random_state=seed)
    subset_path = os.path.join(config["data"]["output_dir"], "subset_100.xlsx")
    os.makedirs(os.path.dirname(subset_path), exist_ok=True)
    subset.to_excel(subset_path, index=False)
    print(f"Created {len(subset)}-row subset: {subset_path}")
    return subset_path


def run_and_evaluate(config: dict, data_path: str, model_id: str,
                     mode: str, prompt: str, use_combined: bool,
                     exp_name: str) -> dict:
    """Run inference + evaluation for one experiment."""
    from scripts.run_baseline import run_inference
    from evaluate import evaluate as run_eval

    output_path = os.path.join(config["data"]["output_dir"], f"{exp_name}_results.csv")

    print(f"\n{'#'*60}")
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  Model: {model_id} | Mode: {mode} | Prompt: {prompt}")
    print(f"{'#'*60}\n")

    start = time.time()

    # Run inference
    run_inference(
        config=config,
        mode=mode,
        prompt_variant=prompt,
        model_id=model_id,
        data_path=data_path,
        output_path=output_path,
        use_combined=use_combined,
        checkpoint_every=50,
    )

    elapsed = time.time() - start

    # Evaluate
    eval_output = os.path.join(config["data"]["output_dir"], f"{exp_name}_eval.json")
    eval_results = run_eval(
        predictions_path=output_path,
        gold_path=data_path,
        task="both",
        sbert_model=config["evaluation"]["sbert_model"],
        output_path=eval_output,
    )

    result = {
        "experiment": exp_name,
        "model": model_id.split("/")[-1],
        "mode": mode,
        "prompt": prompt,
        "time_min": round(elapsed / 60, 1),
    }

    if "task_a" in eval_results:
        result["task_a_sbert"] = round(eval_results["task_a"]["target_sbert_sim"], 4)
        result["task_a_rouge"] = round(eval_results["task_a"]["target_rouge_l"], 4)
        result["task_a_final"] = round(eval_results["task_a"]["final_score"], 4)

    if "task_b" in eval_results:
        result["task_b_sbert"] = round(eval_results["task_b"]["avg_sbert_sim"], 4)
        result["task_b_rouge"] = round(eval_results["task_b"]["avg_rouge_l"], 4)
        result["task_b_final"] = round(eval_results["task_b"]["final_score"], 4)

    return result


def print_comparison(results: list) -> None:
    """Print a nice comparison table."""
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Experiment':<25} {'Task A':>10} {'Task B':>10} {'Time':>8}")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  {r['experiment']:<25} {r.get('task_a_final', 'N/A'):>10} {r.get('task_b_final', 'N/A'):>10} {r['time_min']:>6.1f}m")
    print(f"{'='*80}")

    # Find best
    best_a = max(results, key=lambda x: x.get('task_a_final', 0))
    best_b = max(results, key=lambda x: x.get('task_b_final', 0))
    print(f"\n  BEST Task A: {best_a['experiment']} ({best_a.get('task_a_final', 'N/A')})")
    print(f"  BEST Task B: {best_b['experiment']} ({best_b.get('task_b_final', 'N/A')})")
    print(f"\n  Run the best on full val set for submission!")


def main():
    parser = argparse.ArgumentParser(description="Quick Experiment Comparator")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--subset_size", type=int, default=100,
                        help="Number of rows for quick testing (default: 100)")
    parser.add_argument("--skip", type=str, nargs="*", default=[],
                        help="Experiment names to skip (e.g., --skip E1 E3)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create subset
    subset_path = create_subset(config, n=args.subset_size)

    # Define experiments
    EXPERIMENTS = {
        "E1_phi_zeroshot": {
            "model_id": "microsoft/Phi-3.5-mini-instruct",
            "mode": "zero-shot",
            "prompt": "vanilla",
            "use_combined": True,
        },
        "E2_phi_rag": {
            "model_id": "microsoft/Phi-3.5-mini-instruct",
            "mode": "rag",
            "prompt": "vanilla",
            "use_combined": True,
        },
        "E3_phi_fewshot": {
            "model_id": "microsoft/Phi-3.5-mini-instruct",
            "mode": "zero-shot",
            "prompt": "few_shot",
            "use_combined": True,
        },
        "E4_qwen_zeroshot": {
            "model_id": "Qwen/Qwen3-8B",
            "mode": "zero-shot",
            "prompt": "vanilla",
            "use_combined": True,
        },
        "E5_qwen_rag": {
            "model_id": "Qwen/Qwen3-8B",
            "mode": "rag",
            "prompt": "vanilla",
            "use_combined": True,
        },
        "E6_qwen_rag_fewshot": {
            "model_id": "Qwen/Qwen3-8B",
            "mode": "rag",
            "prompt": "few_shot",
            "use_combined": True,
        },
    }

    results = []
    for exp_name, exp_config in EXPERIMENTS.items():
        if exp_name in args.skip:
            print(f"\nSkipping {exp_name}...")
            continue

        try:
            result = run_and_evaluate(
                config=config,
                data_path=subset_path,
                exp_name=exp_name,
                **exp_config,
            )
            results.append(result)

            # Save intermediate results
            pd.DataFrame(results).to_csv(
                os.path.join(config["data"]["output_dir"], "experiment_comparison.csv"),
                index=False,
            )
        except Exception as e:
            print(f"\n  ERROR in {exp_name}: {e}")
            print("  Skipping to next...\n")
            continue

    if results:
        print_comparison(results)
        # Save final comparison
        comp_path = os.path.join(config["data"]["output_dir"], "experiment_comparison.csv")
        pd.DataFrame(results).to_csv(comp_path, index=False)
        print(f"\nDetailed results saved to {comp_path}")


if __name__ == "__main__":
    main()
