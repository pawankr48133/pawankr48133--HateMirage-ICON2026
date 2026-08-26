#!/usr/bin/env python3
"""
HateMirage ICON 2026 — Unified Baseline Runner (v2)
=====================================================
Major improvements over v1:
  1. COMBINED PROMPT: generates Target+Intent+Implication in 1 LLM call (3x faster)
  2. CHECKPOINTING: saves progress every N rows — resume from crash automatically
  3. TIME ESTIMATION: shows ETA so you know if you'll finish before Colab disconnects

Usage:
    # Normal run (combined prompt, with checkpoint every 25 rows)
    python scripts/run_baseline.py --config config.yaml --mode zero-shot

    # Resume after crash (automatic — just re-run the same command)
    python scripts/run_baseline.py --config config.yaml --mode zero-shot

    # Use separate per-field prompts (slower, 3x more LLM calls)
    python scripts/run_baseline.py --config config.yaml --mode zero-shot --no-combined

    # Change checkpoint frequency
    python scripts/run_baseline.py --config config.yaml --mode zero-shot --checkpoint-every 10
"""

import argparse
import json
import os
import re
import sys
import time

import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.prompt_templates import get_prompt, get_combined_prompt


# =============================================================================
# Response Cleaning
# =============================================================================
def clean_response(response: str, field: str) -> str:
    """Extract the generated answer for a field from the full model output."""
    pattern = rf'## {field}:\s*(.+?)(?:\n## |\n\n|$)'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

    if not match:
        parts = re.split(rf'## {field}:', response, flags=re.IGNORECASE)
        if len(parts) > 1:
            text = parts[-1].strip()
        else:
            lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
            return lines[-1] if lines else ""
    else:
        text = match.group(1).strip()

    if field == "Target":
        quoted = re.findall(r'"([^"]+)"', text)
        if quoted:
            return ", ".join(quoted).strip()
        lines = text.splitlines()
        for line in lines:
            line = line.strip().strip("-*").strip()
            if line and re.fullmatch(r'[\w\s,]+', line):
                return re.sub(r'\s*,\s*', ', ', line).strip(", ")
        return text.splitlines()[0].strip() if text else ""

    elif field in ["Intent", "Implication"]:
        sentences = re.findall(r'[^.?!]*[.?!]', text)
        if not sentences:
            return text.splitlines()[0].strip() if text else ""
        cleaned = " ".join(sentences[:2]).strip()
        return re.sub(r'\s+', ' ', cleaned)

    return text


def parse_combined_response(response: str) -> dict:
    """Parse a combined response that contains all 3 fields."""
    result = {"Target": "", "Intent": "", "Implication": ""}

    for field in ["Target", "Intent", "Implication"]:
        # Try pattern: "Field: answer" or "**Field**: answer"
        patterns = [
            rf'\*?\*?{field}\*?\*?\s*:\s*(.+?)(?:\n|$)',
            rf'## {field}:\s*(.+?)(?:\n|$)',
            rf'{field}\s*[-:]\s*(.+?)(?:\n|$)',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                # Clean up
                text = re.sub(r'^\*+\s*', '', text)  # remove leading **
                text = re.sub(r'\s*\*+$', '', text)  # remove trailing **
                text = text.strip('"').strip()
                if text and text.lower() not in ['<your answer>', 'n/a', 'none']:
                    result[field] = text
                    break

    # Post-process Target: keep only first line, comma-separated
    if result["Target"]:
        result["Target"] = result["Target"].split('\n')[0].strip()

    # Post-process Intent/Implication: keep max 2 sentences
    for field in ["Intent", "Implication"]:
        if result[field]:
            sentences = re.findall(r'[^.?!]*[.?!]', result[field])
            if sentences:
                result[field] = " ".join(sentences[:2]).strip()
            else:
                result[field] = result[field].split('\n')[0].strip()

    return result


# =============================================================================
# Checkpointing
# =============================================================================
def get_checkpoint_path(output_path: str) -> str:
    """Get checkpoint file path from output path."""
    base = os.path.splitext(output_path)[0]
    return f"{base}_checkpoint.csv"


def load_checkpoint(output_path: str) -> tuple:
    """Load checkpoint if it exists. Returns (results_list, completed_indices_set)."""
    ckpt_path = get_checkpoint_path(output_path)
    if os.path.exists(ckpt_path):
        df = pd.read_csv(ckpt_path)
        results = df.to_dict('records')
        completed = set(df["Index"].tolist())
        print(f"  CHECKPOINT FOUND: {len(completed)} rows already done. Resuming...")
        return results, completed
    return [], set()


def save_checkpoint(results: list, output_path: str):
    """Save current progress to checkpoint file."""
    ckpt_path = get_checkpoint_path(output_path)
    pd.DataFrame(results).to_csv(ckpt_path, index=False)


# =============================================================================
# RAG: FAISS retrieval
# =============================================================================
def build_faiss_index(config: dict):
    """Build or load FAISS index from RAG reference documents."""
    from langchain.docstore.document import Document
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    faiss_dir = config["rag"]["faiss_index_dir"]
    embedding_model = config["rag"]["embedding_model"]
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    if os.path.exists(os.path.join(faiss_dir, "index.faiss")):
        print(f"  Loading existing FAISS index from {faiss_dir}")
        return FAISS.load_local(faiss_dir, embeddings,
                                allow_dangerous_deserialization=True)

    print("  Building FAISS index from source documents...")
    jsonl_path = config["rag"]["source_docs"]
    documents = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            documents.append(
                Document(
                    page_content=item["text"],
                    metadata={
                        "topic": item.get("topic"),
                        "chunk_id": item.get("chunk_id"),
                        "tags": item.get("tags", []),
                    },
                )
            )
    faiss_db = FAISS.from_documents(documents, embeddings)
    os.makedirs(faiss_dir, exist_ok=True)
    faiss_db.save_local(faiss_dir)
    print(f"  FAISS index saved ({len(documents)} documents)")
    return faiss_db


def search_rag_context(faiss_db, query: str, top_k: int = 5) -> str:
    results = faiss_db.similarity_search(query, k=top_k)
    return " ".join([doc.page_content for doc in results])


# =============================================================================
# Few-shot example selection
# =============================================================================
def select_few_shot_examples(df: pd.DataFrame, n: int = 3, seed: int = 42) -> list:
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    examples = []
    for _, row in sample.iterrows():
        examples.append({
            "comment": str(row["Comments"]),
            "Target": str(row["Target"]),
            "Intent": str(row["Intent"]),
            "Implication": str(row["Implication"]),
        })
    return examples


# =============================================================================
# Main inference loop (with checkpointing + combined prompts)
# =============================================================================
def run_inference(config: dict, mode: str, prompt_variant: str,
                  model_id: str = None, data_path: str = None,
                  output_path: str = None, use_combined: bool = True,
                  checkpoint_every: int = 25) -> pd.DataFrame:
    """
    Run inference with checkpointing and combined prompts.

    Args:
        use_combined: if True, use 1 LLM call per comment (3x faster)
        checkpoint_every: save progress every N rows
    """
    # --- Resolve config ---
    model_id = model_id or config["model"]["id"]
    data_path = data_path or config["data"]["train_path"]
    output_dir = config["data"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(output_dir, f"{mode}_{prompt_variant}_results.csv")

    max_new_tokens = config["model"].get("max_new_tokens", 128)
    if use_combined:
        max_new_tokens = max(max_new_tokens, 256)  # combined needs more tokens
    do_sample = config["model"].get("do_sample", False)

    print(f"\n{'='*60}")
    print(f"  HateMirage Baseline Runner v2")
    print(f"  Mode: {mode} | Prompt: {prompt_variant} | Model: {model_id}")
    print(f"  Combined prompt: {use_combined} | Checkpoint every: {checkpoint_every} rows")
    print(f"{'='*60}\n")

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # --- Load data ---
    print(f"\n  Loading data from {data_path}...")
    df = pd.read_excel(data_path)

    # Handle both labeled (train) and unlabeled (val) data
    has_labels = all(c in df.columns for c in ["Target", "Intent", "Implication"])
    if has_labels:
        df = df[["Index", "Comments", "Target", "Intent", "Implication"]].dropna()
    else:
        df = df[["Index", "Comments"]].dropna()
    print(f"  Loaded {len(df)} rows (labels: {has_labels})\n")

    # --- Load checkpoint ---
    results, completed_indices = load_checkpoint(output_path)
    remaining = len(df) - len(completed_indices)
    if completed_indices:
        print(f"  Remaining: {remaining} rows to process\n")

    if remaining == 0:
        print("  All rows already processed! Loading final results.")
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_path, index=False)
        return result_df

    # --- Load model ---
    print(f"  Loading model: {model_id}...")
    quant = config["model"].get("quantization", "none")
    load_kwargs = {"device_map": "auto"}

    if quant == "4bit-nf4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["quantization_config"] = bnb_config

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    print("  Model loaded.\n")

    # --- RAG setup ---
    faiss_db = None
    if mode == "rag":
        faiss_db = build_faiss_index(config)
        top_k = config["rag"].get("top_k", 5)

    # --- Few-shot examples ---
    few_shot_examples = None
    if prompt_variant == "few_shot" and has_labels:
        num_examples = config["prompt"].get("num_examples", 3)
        few_shot_examples = select_few_shot_examples(df, n=num_examples,
                                                      seed=config["data"].get("seed", 42))
        print(f"  Using {len(few_shot_examples)} few-shot examples\n")

    # --- Generation function ---
    def generate(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(output[0], skip_special_tokens=True).strip()

    # --- Inference loop with checkpointing ---
    start_time = time.time()
    processed_this_session = 0
    times_per_row = []

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Inference")):
        # Skip already completed rows
        if row["Index"] in completed_indices:
            continue

        row_start = time.time()
        comment = str(row["Comments"])

        result = {"Index": row["Index"], "Comments": comment}
        if has_labels:
            result["Gold_Target"] = str(row["Target"])
            result["Gold_Intent"] = str(row["Intent"])
            result["Gold_Implication"] = str(row["Implication"])

        try:
            # Retrieve RAG context
            context = ""
            if mode == "rag" and faiss_db is not None:
                context = search_rag_context(faiss_db, comment, top_k=top_k)

            if use_combined:
                # === COMBINED MODE: 1 LLM call for all 3 fields ===
                prompt_kwargs = {}
                if context:
                    prompt_kwargs["context"] = context
                if few_shot_examples:
                    prompt_kwargs["examples"] = few_shot_examples

                prompt = get_combined_prompt(prompt_variant, comment, **prompt_kwargs)
                raw_response = generate(prompt)
                parsed = parse_combined_response(raw_response)

                result["Pred_Target"] = parsed["Target"]
                result["Pred_Intent"] = parsed["Intent"]
                result["Pred_Implication"] = parsed["Implication"]
            else:
                # === PER-FIELD MODE: 3 LLM calls per comment ===
                for field in ["Target", "Intent", "Implication"]:
                    prompt_kwargs = {}
                    if mode == "rag":
                        prompt_kwargs["context"] = context
                        prompt_kwargs["base_variant"] = prompt_variant
                        actual_variant = "rag"
                    else:
                        actual_variant = prompt_variant
                    if prompt_variant == "few_shot":
                        prompt_kwargs["examples"] = few_shot_examples

                    prompt = get_prompt(actual_variant, comment, field, **prompt_kwargs)
                    raw_response = generate(prompt)
                    result[f"Pred_{field}"] = clean_response(raw_response, field)

        except Exception as e:
            print(f"\n  WARNING: Failed on Index={row['Index']}: {e}")
            result.setdefault("Pred_Target", "")
            result.setdefault("Pred_Intent", "")
            result.setdefault("Pred_Implication", "")

        results.append(result)
        completed_indices.add(row["Index"])
        processed_this_session += 1

        # Track timing
        row_time = time.time() - row_start
        times_per_row.append(row_time)

        # Show ETA every checkpoint
        if processed_this_session % checkpoint_every == 0:
            save_checkpoint(results, output_path)
            avg_time = sum(times_per_row) / len(times_per_row)
            rows_left = remaining - processed_this_session
            eta_seconds = avg_time * rows_left
            eta_min = eta_seconds / 60
            elapsed = time.time() - start_time
            print(f"\n  CHECKPOINT SAVED ({len(results)} total rows)")
            print(f"  Speed: {avg_time:.1f}s/row | Done: {processed_this_session}/{remaining}")
            print(f"  ETA: {eta_min:.0f} min ({eta_seconds/3600:.1f} hr) | Elapsed: {elapsed/60:.0f} min\n")

    # --- Final save ---
    elapsed = time.time() - start_time
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)

    # Clean up checkpoint
    ckpt_path = get_checkpoint_path(output_path)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print(f"\n{'='*60}")
    print(f"  DONE! {len(results)} rows in {elapsed/60:.1f} min")
    if times_per_row:
        print(f"  Avg speed: {sum(times_per_row)/len(times_per_row):.1f}s/row")
    print(f"  Results saved to: {output_path}")
    print(f"{'='*60}\n")

    return result_df


# =============================================================================
# CLI Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="HateMirage Baseline Runner v2")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--mode", type=str, choices=["zero-shot", "rag"],
                        default="zero-shot", help="Inference mode")
    parser.add_argument("--prompt", type=str, choices=["vanilla", "few_shot", "cot"],
                        default=None, help="Prompt variant (overrides config)")
    parser.add_argument("--model_id", type=str, default=None,
                        help="HuggingFace model ID (overrides config)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to dataset xlsx (overrides config)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save results CSV (overrides config)")
    parser.add_argument("--no-combined", action="store_true",
                        help="Use separate per-field prompts (slower, 3x more calls)")
    parser.add_argument("--checkpoint-every", type=int, default=25,
                        help="Save checkpoint every N rows (default: 25)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    prompt_variant = args.prompt or config["prompt"].get("variant", "vanilla")

    run_inference(
        config=config,
        mode=args.mode,
        prompt_variant=prompt_variant,
        model_id=args.model_id,
        data_path=args.data_path,
        output_path=args.output_path,
        use_combined=not args.no_combined,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
