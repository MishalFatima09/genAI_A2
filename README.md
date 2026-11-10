# Generative AI — Assignment #2: Clinical Trial Eligibility Screening & Efficient Transformer Inference

A complete, self-contained Transformer research and implementation repository that matches patient medical records to clinical trial eligibility criteria, extracts sentence-level evidence, and demonstrates advanced inference mechanisms including explicit Key-Value (KV) caching, tiled attention with online softmax (FlashAttention), and greedy speculative decoding.

---

## Repository Structure

```text
genAI_A2/
├── app/                          # Task 6: Standalone Web Application
│   ├── app.py                    # Flask server with REST API
│   ├── static/                   # CSS and frontend JavaScript
│   └── templates/                # HTML5 UI dashboard
├── code/                         # Core Implementations & Benchmark Runners
│   ├── schema.py                 # Prompt templates, sentence segmentation & JSON parsing
│   ├── prepare_data.py           # 37/8/8 patient-grouped split & no-leakage filtering
│   ├── finetune_lora.py          # Task 1: LoRA fine-tuning script
│   ├── evaluate.py               # Task 2: Evaluation on test/val splits
│   ├── kv_cache.py               # Task 3: ExplicitKVCache & cache-aware generation
│   ├── benchmark_kv.py           # Task 3: KV-cache benchmark runner
│   ├── tiled_attention.py        # Task 4: Tiled online-softmax attention (FlashAttention)
│   ├── benchmark_tiled.py        # Task 4: Attention benchmark runner
│   ├── speculative.py            # Task 5: Greedy speculative decoding engine
│   ├── benchmark_speculative.py  # Task 5: Speculative decoding benchmark runner
│   └── generate_plots.py         # Script to generate publication figures
├── data/                         # Dataset & Split Manifest
│   ├── split_manifest.json       # Patient ID split mapping (seed=42)
│   ├── split_manifest.csv        # Split summary table
│   └── splits/                   # train.jsonl (603), val.jsonl (206), test.jsonl (206)
├── docs/                         # Research & Design Documentation
│   ├── literature_review.md      # Task 1 A1: 16 verified scholarly citations
│   └── model_selection.md        # Task 1 A2/A3: Architecture comparison & decision log
├── report/                       # IEEE Research Paper Deliverable
│   ├── main.tex                  # Complete IEEE-format LaTeX paper
│   ├── references.bib            # BibTeX bibliography
│   └── figures/                  # Generated PDF/PNG publication figures
├── results/                      # Evaluation Metrics, Benchmark Logs & Error Analysis
│   ├── metrics_after_lora_test.json   # Held-out test split metrics (67.0% accuracy)
│   ├── metrics_after_lora_val.json    # Validation split metrics (70.4% accuracy)
│   ├── metrics_before_base_val.json   # Zero-shot base model metrics (12.1% accuracy)
│   ├── error_analysis_task2.md        # 8+ clinical failure modes with annotation IDs
│   ├── task4_results.json             # Tiled attention benchmark (74.7% memory reduction)
│   ├── task5_results.json             # Speculative decoding benchmark (1.32x speedup)
│   └── train_curve.csv                # 114-step training loss curve
├── adapter_result_kaggle/        # Trained LoRA Adapter Checkpoint (Task 1)
├── run_app.bat                   # One-click Windows launcher for Task 6 Web UI
└── requirements.txt              # Environment dependencies
```

---

## Quick Start & Installation

### 1. Environment Setup
```bash
git clone https://github.com/MishalFatima09/genAI_A2.git
cd genAI_A2
pip install -r requirements.txt
```

### 2. Launch the Task 6 Clinical Web Application
Double-click `run_app.bat` (on Windows) or run:
```bash
python app/app.py
```
Open **`http://127.0.0.1:5000`** in your browser to access:
- **Patient Screening Workspace**: Screen held-out patients or custom clinical notes against criteria with sentence-ID evidence highlighting and rule-based trial verdicts.
- **Inference Lab**: Inspect Task 5 speculative decoding speedups and token equivalence proofs.
- **Live Tiled Attention Runner**: Run PyTorch online-softmax attention vs reference scaled dot-product attention in real time.
- **Report Export**: 1-click export to Markdown (`.md`) or JSON (`.json`) with full provenance headers.

---

## Reproducing Tasks 1–5

### Task 1: LoRA Fine-Tuning
```bash
python code/finetune_lora.py \
    --base Qwen/Qwen2.5-1.5B-Instruct \
    --train-file data/splits/train.jsonl \
    --val-file data/splits/val.jsonl \
    --epochs 3 --batch-size 1 --grad-accum 16 --lr 2e-4 --r 16 --alpha 32
```

### Task 2: Held-Out Model Evaluation & Error Analysis
```bash
python code/evaluate.py --split test --allow-test
```

### Task 3: Explicit KV-Cache Autoregressive Generation
```bash
python code/benchmark_kv.py \
    --model adapter_result_kaggle \
    --base Qwen/Qwen2.5-1.5B-Instruct \
    --prompt-lengths 128 512 1024 \
    --gen-lengths 16 64 128
```

### Task 4: Efficient Tiled Attention with Online Softmax
```bash
python code/benchmark_tiled.py --out results/task4_results.json
```

### Task 5: Greedy Speculative Decoding
```bash
python code/benchmark_speculative.py \
    --target adapter_result_kaggle \
    --base Qwen/Qwen2.5-1.5B-Instruct \
    --draft Qwen/Qwen2.5-0.5B-Instruct \
    --gamma 4 \
    --prompt-lengths 128 512 \
    --gen-lengths 64 128 \
    --out results/task5_results.json
```

---

## Key Experimental Results

| Component | Key Finding / Metric | Rubric Evidence |
|---|---|---|
| **Task 1: Adaptation** | 12.1% (Base) $\to$ **70.4%** (LoRA Val), JSON errors 12.6% $\to$ 0% | `kaggle_task1_results/training_log.json` |
| **Task 2: Held-Out Test** | **66.99% Accuracy**, Evidence micro-F1 **40.13%**, 8+ error modes documented | `results/metrics_after_lora_test.json`, `error_analysis_task2.md` |
| **Task 3: KV Caching** | Explicit cache reuse, causal masking, position IDs; 100% token equivalence | `code/kv_cache.py` (`ExplicitKVCache`) |
| **Task 4: Tiled Attention** | **74.7% peak activation memory reduction** at $L=1024$, error $<5.96\times 10^{-7}$ | `results/task4_results.json` |
| **Task 5: Speculative Decoding** | Up to **1.32$\times$ speedup** (peak $\alpha=56.4\%$), 100% token match | `results/task5_results.json` |
| **Task 6: Browser Application** | Zero-GPU local startup, evidence highlighting, validation recovery, export | `app/app.py`, `run_app.bat` |

---

## Technical Report

The full IEEE-format research paper is prepared in `report/main.tex` and can be compiled using `pdflatex report/main.tex && bibtex report/main && pdflatex report/main.tex` or imported directly into Overleaf. All generated figures are located in `report/figures/`.
