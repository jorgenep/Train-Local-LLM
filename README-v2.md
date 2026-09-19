# Llama-Code-287M: Pre-Training & Fine-Tuning a Custom Code LLM from Scratch

A lightweight, high-density Llama-style language model (**~287M parameters**) designed and optimized for **Python code generation, completion, instruction following, and debugging**. This repository contains a complete, self-contained pipeline for training a custom BPE tokenizer, zero-RAM binary memory-mapped data preprocessing, Llama model architecture, pre-training, Supervised Fine-Tuning (SFT), benchmark evaluation, and interactive terminal inference across **Intel GPUs (PyTorch XPU)**, **NVIDIA GPUs (CUDA)**, or **CPUs**.

---

## 🌟 Architectural Features

* **Llama-Style Transformer Backbone:** 
  * **24 Layers** and **1024 Hidden Embedding Dimension** (providing deep composition capacity for multi-step code reasoning).
  * **2048 Sequence Context Window** for processing complete Python functions and class scopes.
* **Grouped-Query Attention (GQA):** 16 Query heads and 4 Key/Value heads, significantly reducing KV-cache memory overhead.
* **Modern Sub-Layers:** Rotary Position Embeddings (**RoPE**), **RMSNorm** pre-normalization, and **SwiGLU** feed-forward activation functions without linear bias terms.
* **Tied Input/Output Embeddings:** Shares weights between input token embeddings and output language modeling head (`lm_head.weight = tok_emb.weight`), saving ~16–32 million redundant parameters.
* **Custom Python BPE Tokenizer:** Compact 16,384 (16k) vocabulary trained directly on code, preserving exact indentation, syntax structure, and special control tokens (`<|im_start|>`, `<|im_end|>`).
* **Zero-RAM Data Streaming:** Memory-mapped binary token dataset (`np.memmap`) allowing massive datasets (10B+ tokens) to stream directly from disk into GPU VRAM without filling system RAM.
* **Instruction Supervised Fine-Tuning (SFT):** Fine-tunes model on ChatML prompt-solution pairs with masked cross-entropy loss (loss computed exclusively on response tokens).
* **Process-Isolated Benchmark Evaluation:** Evaluates pass@1 performance on HumanEval and MBPP problems inside an isolated, timed subprocess sandbox.
* **Interactive Terminal CLI (`chat.py`):** Interactive REPL with temperature, top-k, and top-p (nucleus) sampling supporting both raw code continuation and ChatML instruction mode.
* **Multi-Platform Acceleration:** Native support for **Intel Arc GPUs (`torch.xpu`)**, **NVIDIA GPUs (`torch.cuda`)**, and **CPUs** with `bfloat16`/`float16` Automatic Mixed Precision (AMP).

---

## 📁 Repository Structure

```
.
├── model.py            # PyTorch implementation of Llama Code LM (RoPE, RMSNorm, SwiGLU, GQA, Tied Embeddings)
├── train_tokenizer.py  # Custom Byte-Pair Encoding (BPE) tokenizer training script (16k vocabulary)
├── prepare_data.py     # Parallel dataset downloader, BPE pre-tokenizer, and binary memory-mapper
├── train.py            # Pre-training loop with Intel XPU / CUDA auto-detection, AMP, and checkpointing
├── sft_train.py        # Supervised Fine-Tuning (SFT) script with ChatML formatting and masked loss
├── eval_humaneval.py   # HumanEval / MBPP pass@1 benchmark evaluator with process-isolated sandbox
├── chat.py             # Interactive terminal inference CLI (sampling, ChatML mode, device auto-detect)
├── sanity_check.py     # Comprehensive automated 10-step verification test suite
├── requirements.txt    # Python package dependencies
└── README.md           # Complete technical guide and documentation
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

Clone the repository and activate your Python virtual environment:

```bash
git clone https://github.com/jorgenep/Train-Local-LLM.git
cd Train-Local-LLM

python3 -m venv train_env
source train_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### For Intel GPU (XPU) Hardware (e.g., Intel Arc B570 Pro / Data Center GPUs):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

#### For NVIDIA GPU (CUDA) Hardware:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### 2. End-to-End Execution Pipeline

#### Step 1: Train the Custom Python Tokenizer
Trains a 16k vocabulary Byte-Pair Encoding (BPE) tokenizer on your raw code corpus and exports `python_bpe_16k.json`:
```bash
python3 train_tokenizer.py
```

#### Step 2: Pre-Tokenize Data to Memory-Mapped Binary (`train_data.bin`)
Downloads Python code files (such as `HuggingFaceTB/smollm-corpus` `python-edu`), tokenizes text across CPU cores, and outputs a memory-mapped binary array:
```bash
python3 prepare_data.py
```

#### Step 3: Pre-Train the Base Model
Auto-detects GPU hardware (Intel XPU or NVIDIA CUDA), initializes the ~287M Llama architecture, and pre-trains with bfloat16 mixed precision:
```bash
python3 train.py
```

#### Step 4: Supervised Fine-Tuning (SFT)
Fine-tunes the pre-trained base model on instruction-response coding datasets using ChatML formatting and masked loss computation:
```bash
python3 sft_train.py --pretrained_path checkpoints/model_final.pth
```

#### Step 5: Evaluate Benchmark Performance
Measures pass@1 accuracy on HumanEval coding tasks inside a process-isolated sandbox:
```bash
python3 eval_humaneval.py --model_path sft_checkpoints/sft_model_final.pth
```

#### Step 6: Interactive Terminal Inference
Launch the interactive CLI to test code completion or chat with your model:
```bash
# Interactive ChatML Instruction Mode
python3 chat.py --checkpoint sft_checkpoints/sft_model_final.pth --chat_mode

# Raw Code Continuation Mode
python3 chat.py --checkpoint checkpoints/model_final.pth
```

---

## 🧪 Automated System Verification

Run the comprehensive test suite to verify mathematics, attention layers, tokenizers, data mapping, pre-training, fine-tuning, sandboxing, and inference CLI:

```bash
python3 sanity_check.py
```

**Passing Verification Battery:**
* `[1/10]` Rotary Position Embeddings (RoPE) Shapes & Rotation Gradients
* `[2/10]` RMSNorm & SwiGLU Feed-Forward Networks
* `[3/10]` Grouped-Query Attention (GQA) Key/Value Head Grouping
* `[4/10]` Weight Tying Verification (`tok_emb` & `lm_head`)
* `[5/10]` BPE Tokenizer Roundtrip & Whitespace Preservation
* `[6/10]` Binary Dataset Memory-Mapping & Shifted Target Alignment (`x[1:] == y[:-1]`)
* `[7/10]` End-to-End Autoregressive Pre-Training & Checkpoint Output
* `[8/10]` Supervised Fine-Tuning (SFT) & Masked Loss Calculation
* `[9/10]` Code Sandbox & Benchmark Evaluation (HumanEval / MBPP)
* `[10/10]` Interactive CLI Sampling Generator (`chat.py`)

---

## 📊 Hardware & Resource Requirements

| Metric | Requirement / Value |
| :--- | :--- |
| **Target GPU Memory (VRAM)** | **12 GB to 32 GB** (Comfortably fits inside 32GB VRAM with activation headroom) |
| **Static Weight Footprint** | ~575 MB (BF16) / ~1.15 GB (FP32) |
| **Supported Compute Backends** | Intel XPU (`torch.xpu`), NVIDIA CUDA (`torch.cuda`), CPU |
| **Optimizer Memory** | ~3.4 GB (AdamW FP32 states) |

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
