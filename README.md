# HateMirage ICON 2026 — Explainable Faux Hate Detection

Submission for the [HateMirage ICON 2026 Shared Task](https://sai-kartheek-reddy.github.io/HateMirage-ICON2026/): **Explainable Faux Hate Detection and Multi-Dimensional Reasoning**.

- **Task A** — Target Identification: identify who is being targeted in a faux-hate comment.
- **Task B** — Intent & Implication Generation: explain the motive and social consequence.

> **Codabench**: [Competition Page](https://www.codabench.org/competitions/17783/)  
> **Paper**: [HateMirage (LREC 2026)](https://arxiv.org/abs/2603.02684)

---

## Repository Structure

```text
HateMirage-ICON2026/
├── config.yaml                     # Central configuration (models, paths, hyperparams)
├── evaluate.py                     # Competition-matching evaluation script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/
│   └── sample-data.xlsx            # Sample dataset (50 rows)
│
├── source_docs/
│   ├── RAG_Reference_Data.jsonl    # RAG context documents
│   └── fake_claims.txt             # List of debunked fake claims
│
├── scripts/
│   ├── run_baseline.py             # Unified Zero-Shot + RAG inference runner
│   ├── run_experiment.py           # Automated experiment matrix driver
│   ├── prompt_templates.py         # All prompt variants (vanilla, few-shot, CoT)
│   ├── generate_results_table.py   # Results formatting (Markdown + LaTeX)
│   └── eda.py                      # Exploratory data analysis
│
├── Starter-Kit/                    # Original starter kit (preserved for reference)
│
├── faiss_index/                    # Generated FAISS index (auto-created)
│
└── outputs/                        # All predictions, eval results, experiment logs
```

---

## Quick Start (Google Colab)

### 1. Setup

```python
# Clone and install
!git clone https://github.com/YOUR-USERNAME/HateMirage-ICON2026.git
%cd HateMirage-ICON2026
!pip install -r requirements.txt
!python -c "import nltk; nltk.download('punkt')"
```

### 2. Run EDA

```bash
python scripts/eda.py --data_path data/sample-data.xlsx
```

### 3. Run a Baseline (Zero-Shot)

```bash
python scripts/run_baseline.py \
    --config config.yaml \
    --mode zero-shot \
    --prompt vanilla
```

### 4. Run a Baseline (RAG)

```bash
python scripts/run_baseline.py \
    --config config.yaml \
    --mode rag \
    --prompt vanilla
```

### 5. Evaluate Predictions

```bash
python evaluate.py \
    --predictions outputs/zero-shot_vanilla_results.csv \
    --gold data/sample-data.xlsx \
    --task both
```

### 6. Run Full Experiment Matrix

```bash
python scripts/run_experiment.py --config config.yaml --all
```

### 7. Generate Results Table

```bash
python scripts/generate_results_table.py --log outputs/experiment_log.csv
```

---

## Model & Config Choices

All configuration is in [`config.yaml`](config.yaml). Key decisions:

| Setting | Choice | Rationale |
|---------|--------|-----------|
| Quantization | 4-bit NF4 | Fits 8B models on free Colab T4 (15 GB VRAM) |
| Embedding model | `all-mpnet-base-v2` | Starter kit default; strong for semantic retrieval |
| Eval SBERT model | `all-MiniLM-L6-v2` | Lightweight, matches competition scoring |
| Candidate LLMs | Phi-3.5-mini, Qwen3-8B, Mistral-7B | Cover 3B-8B range, all Apache 2.0 / open |
| Prompt variants | vanilla, few-shot, chain-of-thought | Test instruction quality vs. reasoning depth |
| RAG top-k | 5 | Starter kit default; tunable in config |

---

## Evaluation

Scoring matches the competition exactly:
- **Metrics**: Sentence-BERT cosine similarity + ROUGE-L F1
- **Task A**: Both metrics on Target → Final = mean(SBERT, ROUGE-L)
- **Task B**: Both metrics on Intent & Implication, averaged → Final = mean(avg_SBERT, avg_ROUGE-L)

---

## Results

*(Table will be populated after running experiments)*

See [`outputs/results_comparison.md`](outputs/results_comparison.md) for the full comparison.

---

## Known Limitations

1. **Sample data only**: Built against 50-row sample; performance numbers will change with the full 4,530-comment training set.
2. **Colab-only**: Pipeline assumes T4 GPU with 15 GB VRAM. Larger models (>8B) won't fit without further quantization.
3. **No fine-tuning**: All experiments use pretrained models via prompting. QLoRA fine-tuning could improve scores but requires more compute time.
4. **Code-mixed handling**: Hindi-English code-mixed comments may get lower-quality outputs from models not extensively trained on Hindi.

---

## AI Writing Assistance Disclosure

> **[TODO]**: This submission used an AI coding assistant to help write infrastructure code (data loading, evaluation scripts, experiment automation). All scientific decisions (model selection, prompt design, result interpretation) were made by the authors. Per the shared task guidelines, this usage is disclosed here.

---

## Citation

```bibtex
@article{kasu2026hatemirage,
  title={HateMirage: An Explainable Multi-Dimensional Dataset for Decoding Faux Hate and Subtle Online Abuse},
  author={Kasu, Sai Kartheek Reddy and Biradar, Shankar and Saumya, Sunil and Akhtar, Md. Shad},
  journal={arXiv preprint arXiv:2603.02684},
  year={2026}
}
```

> **[TODO]**: Add the shared task overview paper citation once published.

---

## License

This repository is for research purposes as part of the HateMirage ICON 2026 shared task.
