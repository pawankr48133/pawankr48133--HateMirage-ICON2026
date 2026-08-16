#!/usr/bin/env python3
"""
HateMirage ICON 2026 — Unified Baseline Runner
================================================
Runs either Zero-Shot or RAG mode on the dataset using a configurable LLM.

Usage:
    # Zero-shot with Phi-3.5-mini
    python scripts/run_baseline.py --config config.yaml --mode zero-shot

    # RAG mode
    python scripts/run_baseline.py --config config.yaml --mode rag

    # Override model from CLI
    python scripts/run_baseline.py --config config.yaml --mode zero-shot \
        --model_id "Qwen/Qwen3-8B" --output_path outputs/qwen_zs.csv
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

# Add project root to path so we can import from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.prompt_templates import get_prompt


# =============================================================================
# Response Cleaning (from starter kit, improved)
# =============================================================================
def clean_response(response: str, field: str) -> str:
    """Extract the generated answer for a field from the full model output."""
    # Try to find the answer after the field marker
    pattern = rf'## {field}:\s*(.+?)(?:\n## |\n\n|$)'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

    if not match:
        # Fallback: take everything after the last occurrence of the field marker
        parts = re.split(rf'## {field}:', response, flags=re.IGNORECASE)
        if len(parts) > 1:
            text = parts[-1].strip()
        else:
            # Last resort: return the last non-empty line
            lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
            return lines[-1] if lines else ""
    else:
        text = match.group(1).strip()

    if field == "Target":
        # Clean target: extract quoted text or comma-separated words
        quoted = re.findall(r'"([^"]+)"', text)
        if quoted:
            return ", ".join(quoted).strip()
        # Take the first line that looks like target word(s)
        lines = text.splitlines()
        for line in lines:
            line = line.strip().strip("-•").strip()
            if line and re.fullmatch(r'[\w\s,]+', line):
                return re.sub(r'\s*,\s*', ', ', line).strip(", ")
        return text.splitlines()[0].strip() if text else ""

    elif field in ["Intent", "Implication"]:
        # Take at most 2 sentences
        sentences = re.findall(r'[^.?!]*[.?!]', text)
        if not sentences:
            return text.splitlines()[0].strip() if text else ""
        cleaned = " ".join(sentences[:2]).strip()
        return re.sub(r'\s+', ' ', cleaned)

    return text


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

    # Try loading existing index
    if os.path.exists(os.path.join(faiss_dir, "index.faiss")):
        print(f"Loading existing FAISS index from {faiss_dir}")
        return FAISS.load_local(faiss_dir, embeddings,
                                allow_dangerous_deserialization=True)

    # Build from scratch
    print("Building FAISS index from source documents...")
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
    print(f"FAISS index saved to {faiss_dir} ({len(documents)} documents)")
    return faiss_db


def search_rag_context(faiss_db, query: str, top_k: int = 5) -> str:
    """Retrieve top-k documents and concatenate their text."""
    results = faiss_db.similarity_search(query, k=top_k)
    return " ".join([doc.page_content for doc in results])


# =============================================================================
# Few-shot example selection
# =============================================================================
def select_few_shot_examples(df: pd.DataFrame, n: int = 3, seed: int = 42) -> list[dict]:
    """Randomly sample N examples from the dataset for few-shot prompting."""
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
# Main inference loop
# =============================================================================
def run_inference(config: dict, mode: str, prompt_variant: str,
                  model_id: str = None, data_path: str = None,
                  output_path: str = None) -> pd.DataFrame:
    """
    Run inference on the dataset.

    Args:
        config: loaded config.yaml dict
        mode: "zero-shot" or "rag"
        prompt_variant: "vanilla", "few_shot", "cot"
        model_id: override model ID from CLI
        data_path: override data path from CLI
        output_path: override output path from CLI

    Returns:
        DataFrame with predictions
    """
    # --- Resolve config ---
    model_id = model_id or config["model"]["id"]
    data_path = data_path or config["data"]["train_path"]
    output_dir = config["data"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if output_path is None:
        output_path = os.path.join(output_dir, f"{mode}_{prompt_variant}_results.csv")

    max_new_tokens = config["model"].get("max_new_tokens", 128)
    do_sample = config["model"].get("do_sample", False)

    print(f"\n{'='*60}")
    print(f"  HateMirage Baseline Runner")
    print(f"  Mode: {mode} | Prompt: {prompt_variant} | Model: {model_id}")
    print(f"{'='*60}\n")

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print(f"Loading data from {data_path}...")
    df = pd.read_excel(data_path)
    df = df[["Index", "Comments", "Target", "Intent", "Implication"]].dropna()
    print(f"Loaded {len(df)} rows\n")

    # --- Load model ---
    print(f"Loading model: {model_id}...")
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
    print("Model loaded.\n")

    # --- RAG setup ---
    faiss_db = None
    if mode == "rag":
        faiss_db = build_faiss_index(config)
        top_k = config["rag"].get("top_k", 5)

    # --- Few-shot examples ---
    few_shot_examples = None
    if prompt_variant == "few_shot":
        num_examples = config["prompt"].get("num_examples", 3)
        few_shot_examples = select_few_shot_examples(df, n=num_examples,
                                                      seed=config["data"].get("seed", 42))
        print(f"Using {len(few_shot_examples)} few-shot examples\n")

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

    # --- Inference loop ---
    results = []
    start_time = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        comment = str(row["Comments"])
        result = {
            "Index": row["Index"],
            "Comments": comment,
            "Gold_Target": str(row["Target"]),
            "Gold_Intent": str(row["Intent"]),
            "Gold_Implication": str(row["Implication"]),
        }

        # Retrieve context for RAG
        context = ""
        if mode == "rag" and faiss_db is not None:
            context = search_rag_context(faiss_db, comment, top_k=top_k)
            result["RAG_Context"] = context[:500]  # truncate for CSV readability

        for field in ["Target", "Intent", "Implication"]:
            try:
                # Build prompt
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

                # Generate
                raw_response = generate(prompt)
                cleaned = clean_response(raw_response, field)
                result[f"Pred_{field}"] = cleaned

            except Exception as e:
                print(f"\n  WARNING: Failed on Index={row['Index']}, field={field}: {e}")
                result[f"Pred_{field}"] = ""

        results.append(result)

    elapsed = time.time() - start_time
    print(f"\nInference complete: {len(results)} rows in {elapsed:.1f}s "
          f"({elapsed/len(results):.2f}s/row)")

    # --- Save ---
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}\n")

    return result_df


# =============================================================================
# CLI Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="HateMirage Baseline Runner")
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
    )


if __name__ == "__main__":
    main()
