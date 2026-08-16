#!/usr/bin/env python3
"""
HateMirage ICON 2026 — Experiment Runner
==========================================
Automates running multiple experiments across models, modes, and prompt
variants. Results are evaluated and logged to a central CSV.

Usage:
    # Run all experiments defined in the script
    python scripts/run_experiment.py --config config.yaml

    # Run a single experiment
    python scripts/run_experiment.py --config config.yaml \
        --model_id "Qwen/Qwen3-8B" --mode rag --prompt cot
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_baseline import run_inference


def run_single_experiment(config: dict, model_id: str, mode: str,
                          prompt_variant: str, experiment_name: str = None) -> dict:
    """
    Run a single experiment and return evaluation results.

    Returns:
        dict with experiment metadata + eval scores
    """
    exp_name = experiment_name or f"{model_id.split('/')[-1]}_{mode}_{prompt_variant}"
    output_path = os.path.join(config["data"]["output_dir"], f"{exp_name}_results.csv")

    print(f"\n{'#'*60}")
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  Model: {model_id}")
    print(f"  Mode: {mode} | Prompt: {prompt_variant}")
    print(f"{'#'*60}\n")

    start_time = time.time()

    # Run inference
    run_inference(
        config=config,
        mode=mode,
        prompt_variant=prompt_variant,
        model_id=model_id,
        output_path=output_path,
    )

    elapsed = time.time() - start_time

    # Run evaluation
    from evaluate import evaluate as run_eval
    eval_output = os.path.join(config["data"]["output_dir"], f"{exp_name}_eval.json")
    eval_results = run_eval(
        predictions_path=output_path,
        gold_path=config["data"]["train_path"],
        task="both",
        sbert_model=config["evaluation"]["sbert_model"],
        output_path=eval_output,
    )

    # Compile result row
    result = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": exp_name,
        "model_id": model_id,
        "mode": mode,
        "prompt_variant": prompt_variant,
        "inference_time_s": round(elapsed, 1),
        "n_samples": eval_results.get("n_samples", 0),
    }

    # Task A scores
    if "task_a" in eval_results:
        ta = eval_results["task_a"]
        result.update({
            "target_sbert": round(ta["target_sbert_sim"], 4),
            "target_rouge_l": round(ta["target_rouge_l"], 4),
            "task_a_score": round(ta["final_score"], 4),
        })

    # Task B scores
    if "task_b" in eval_results:
        tb = eval_results["task_b"]
        result.update({
            "intent_sbert": round(tb["intent_sbert_sim"], 4),
            "intent_rouge_l": round(tb["intent_rouge_l"], 4),
            "impl_sbert": round(tb["implication_sbert_sim"], 4),
            "impl_rouge_l": round(tb["implication_rouge_l"], 4),
            "task_b_score": round(tb["final_score"], 4),
        })

    return result


def log_experiment(result: dict, log_path: str) -> None:
    """Append experiment result to the CSV log."""
    file_exists = os.path.exists(log_path)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    print(f"Experiment logged to {log_path}")


# =============================================================================
# Experiment Matrix
# =============================================================================
# Default experiment grid (all combinations we want to test)
EXPERIMENT_MATRIX = [
    # Baseline: Phi-3.5-mini across modes and prompts
    {"model_id": "microsoft/Phi-3.5-mini-instruct", "mode": "zero-shot", "prompt": "vanilla"},
    {"model_id": "microsoft/Phi-3.5-mini-instruct", "mode": "zero-shot", "prompt": "few_shot"},
    {"model_id": "microsoft/Phi-3.5-mini-instruct", "mode": "zero-shot", "prompt": "cot"},
    {"model_id": "microsoft/Phi-3.5-mini-instruct", "mode": "rag", "prompt": "vanilla"},
    {"model_id": "microsoft/Phi-3.5-mini-instruct", "mode": "rag", "prompt": "cot"},

    # Qwen3-8B
    {"model_id": "Qwen/Qwen3-8B", "mode": "zero-shot", "prompt": "vanilla"},
    {"model_id": "Qwen/Qwen3-8B", "mode": "zero-shot", "prompt": "cot"},
    {"model_id": "Qwen/Qwen3-8B", "mode": "rag", "prompt": "vanilla"},
    {"model_id": "Qwen/Qwen3-8B", "mode": "rag", "prompt": "cot"},

    # Mistral-7B
    {"model_id": "mistralai/Mistral-7B-Instruct-v0.3", "mode": "zero-shot", "prompt": "vanilla"},
    {"model_id": "mistralai/Mistral-7B-Instruct-v0.3", "mode": "zero-shot", "prompt": "cot"},
    {"model_id": "mistralai/Mistral-7B-Instruct-v0.3", "mode": "rag", "prompt": "vanilla"},
    {"model_id": "mistralai/Mistral-7B-Instruct-v0.3", "mode": "rag", "prompt": "cot"},
]


def run_all_experiments(config: dict) -> None:
    """Run all experiments in the matrix and log results."""
    log_path = config["experiment"]["log_path"]
    print(f"\nRunning {len(EXPERIMENT_MATRIX)} experiments...\n")

    for i, exp in enumerate(EXPERIMENT_MATRIX, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(EXPERIMENT_MATRIX)}] Starting experiment...")
        print(f"{'='*60}")

        try:
            result = run_single_experiment(
                config=config,
                model_id=exp["model_id"],
                mode=exp["mode"],
                prompt_variant=exp["prompt"],
            )
            log_experiment(result, log_path)
        except Exception as e:
            print(f"\n  ERROR in experiment {i}: {e}")
            print("  Skipping to next experiment...\n")
            continue

    print(f"\n{'='*60}")
    print(f"  All experiments complete! Results in: {log_path}")
    print(f"{'='*60}\n")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="HateMirage Experiment Runner")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Run single experiment with this model")
    parser.add_argument("--mode", type=str, choices=["zero-shot", "rag"], default=None)
    parser.add_argument("--prompt", type=str, choices=["vanilla", "few_shot", "cot"],
                        default=None)
    parser.add_argument("--all", action="store_true",
                        help="Run full experiment matrix")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.all:
        run_all_experiments(config)
    elif args.model_id and args.mode and args.prompt:
        result = run_single_experiment(config, args.model_id, args.mode, args.prompt)
        log_experiment(result, config["experiment"]["log_path"])
    else:
        print("Either use --all or specify --model_id, --mode, and --prompt")
        sys.exit(1)


if __name__ == "__main__":
    main()
