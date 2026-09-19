# Llama-Code-287M: Pre-Training a Custom Code LLM from Scratch

A lightweight, high-density Llama-style language model (**~287M parameters**) designed and optimized for **Python code generation, completion, and debugging**. This repository contains a complete, self-contained pipeline for training a custom BPE tokenizer, binary memory-mapped data preprocessing, model architecture implementation, and mixed-precision pre-training across **Intel GPUs (PyTorch XPU)**, **NVIDIA GPUs (CUDA)**, or **CPUs**.

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
* **Multi-Platform Mixed Precision:** Native acceleration for **Intel Arc GPUs (`torch.xpu`)**, **NVIDIA GPUs (`torch.cuda`)**, and **CPUs** utilizing `bfloat16`/`float16` Automatic Mixed Precision (AMP) and AdamW with Cosine Annealing learning rate decay.

---

## 📁 Repository Structure

```
.
├── model.py            # PyTorch implementation of Llama Code LM (RoPE, RMSNorm, SwiGLU, GQA, Tied Embeddings)
├── train_tokenizer.py  # Custom Byte-Pair Encoding (BPE) tokenizer training script (16k vocabulary)
├── prepare_data.py     # Parallel dataset downloader, BPE pre-tokenizer, and binary memory-mapper
├── train.py            # Pre-training loop with Intel XPU / CUDA auto-detection, AMP, and checkpointing
└── sanity_check.py     # Automated 7-step verification test suite for all components
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
```

#### For Intel GPU (XPU) Hardware (e.g., Intel Arc B570 Pro / Data Center GPUs):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install datasets tiktoken tokenizers numpy
```

#### For NVIDIA GPU (CUDA) Hardware:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install datasets tiktoken tokenizers numpy
```

---

### 2. Step-by-Step Training Pipeline

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

#### Step 3: Launch Pre-Training Loop
Auto-detects your GPU hardware (Intel XPU or NVIDIA CUDA), initializes the ~287M Llama architecture, and begins pre-training with bfloat16 mixed precision and automatic checkpointing:
```bash
python3 train.py
```

---

## 🧪 System Verification

Run the comprehensive automated test suite to verify math operations, attention heads, weight tying, dataset streaming, and hardware training loops:

```bash
python3 sanity_check.py
```

**Passing Test Battery:**
* `[1/7]` Rotary Position Embeddings (RoPE) Shapes & Rotation Gradients
* `[2/7]` RMSNorm & SwiGLU Feed-Forward Networks
* `[3/7]` Grouped-Query Attention (GQA) Key/Value Head Grouping
* `[4/7]` Weight Tying Verification (`tok_emb` & `lm_head`)
* `[5/7]` BPE Tokenizer Roundtrip & Whitespace Preservation
* `[6/7]` Binary Dataset Memory-Mapping & Shifted Target Alignment (`x[1:] == y[:-1]`)
* `[7/7]` End-to-End Autoregressive Pre-Training & Checkpoint Output

---

## 📊 Hardware & Resource Requirements

| Metric | Requirement / Value |
| :--- | :--- |
| **Target GPU Memory (VRAM)** | **12 GB to 32 GB** (Comfortably fits inside 32GB VRAM with activation headroom) |
| **Static Weight Footprint** | ~575 MB (BF16) / ~1.15 GB (FP32) |
| **Supported Compute Backends** | Intel XPU (`torch.xpu`), NVIDIA CUDA (`torch.cuda`), CPU |
| **Optimizer Memory** | ~3.4 GB (AdamW FP32 states) |

---

## 🛣️ Post-Training Roadmap

1. **Instruction Supervised Fine-Tuning (SFT):** Fine-tune the base model on ChatML-formatted prompt/solution datasets for code completion and bug fixing.
2. **Reinforcement Learning (GRPO):** Apply Group Relative Policy Optimization with executable unit-test rewards to improve code correctness.
3. **Evaluation Benchmarks:** Measure pass@1 performance on HumanEval and MBPP.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
