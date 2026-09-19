# LLM VRAM Memory & Architecture Scaling Guide

This guide provides a comprehensive mathematical and practical breakdown for scaling modern decoder-only language models (**Llama-style architecture**) based on available GPU VRAM. It details how hyperparameters affect memory consumption and provides recommended configurations across six hardware VRAM tiers.

---

## 🧮 Mathematical Foundations of VRAM Allocation

Training a Transformer model from scratch requires memory for four distinct components:
1. **Model Weights (Parameters):** Stored in 16-bit precision (`bfloat16` or `float16`) = **2 bytes per parameter**.
2. **Gradients:** Stored in 16-bit precision = **2 bytes per parameter**.
3. **Optimizer States (AdamW):** Stored in 32-bit precision (`FP32` master weights/momentum + variance) = **8 bytes per parameter**.
4. **Activation Memory:** Intermediate layer outputs cached during the forward pass for backpropagation.

### Static Memory Equation (AdamW + Mixed Precision)
$$\text{Static VRAM (Bytes)} = P \times (2_{\text{weights}} + 2_{\text{gradients}} + 8_{\text{optimizer}}) = 12 \times P \text{ bytes}$$

*For example, a **287M parameter model** requires:*
$$\text{Static Memory} = 287,000,000 \times 12 \text{ bytes} \approx 3.44 \text{ GB}$$

---

## ⚙️ Key Hyperparameters Impacting Memory

| Hyperparameter | Parameter Impact | VRAM Impact | Architectural Function |
| :--- | :--- | :--- | :--- |
| **`emb_dim` ($D$)** | **Quadratic ($O(D^2)$)** | High | Controls semantic width per token. FFN layers scale as $\approx 8 D^2$. |
| **`n_layers` ($L$)** | **Linear ($O(L)$)** | Moderate | Controls logical composition and sequential reasoning depth. |
| **`context_length` ($S$)** | **None** (0 params) | **Quadratic ($O(S^2)$)** | Controls maximum sequence window and activation memory. |
| **`vocab_size` ($V$)** | **Linear ($O(V \cdot D)$)** | Moderate | Input/Output token lookup table size (minimized via Tied Embeddings). |
| **`n_kv_heads`** | **Minimal** | High (KV-Cache) | Grouped-Query Attention (GQA) reduces KV projection memory by $H_Q / H_{KV}$. |

---

## 📊 VRAM Tier Configurations & Recommendations

### Tier 1: Up to 8 GB VRAM
* **Target Hardware:** NVIDIA RTX 3060 8GB, RTX 4060 8GB, Tesla T4, Intel Arc A580.
* **Target Model Scale:** **~50M – 120M Parameters**
* **Primary Focus:** Fast prototyping, testing code syntax, small domain models.

```python
# Up to 8 GB VRAM Configuration (~110M Params)
vocab_size = 16384       # Compact BPE vocab
emb_dim = 768            # Standard base width
n_layers = 12            # Standard depth
n_heads = 12             # 64 dim per head
n_kv_heads = 4           # GQA 3:1 ratio
context_length = 1024    # Reduced sequence window
batch_size = 4           # Small micro-batch size
gradient_accumulation = 4 # Effective batch size = 16
```
* **Memory Distribution:** Static Memory $\approx 1.3 \text{ GB}$ | Activations $\approx 3.5 \text{ GB}$ | **Total Peak VRAM $\approx 5.5 - 7.0 \text{ GB}$**

---

### Tier 2: 8 – 16 GB VRAM
* **Target Hardware:** NVIDIA RTX 4070 12GB, RTX 3080 10/12GB, RTX 4060 Ti 16GB, Intel Arc A770 16GB.
* **Target Model Scale:** **~150M – 280M Parameters**
* **Primary Focus:** High-density code generation, basic reasoning capabilities.

```python
# 8 - 16 GB VRAM Configuration (~210M Params)
vocab_size = 16384
emb_dim = 896
n_layers = 18            # Increased depth for reasoning
n_heads = 14
n_kv_heads = 4
context_length = 2048    # Full 2k code context
batch_size = 4
gradient_accumulation = 4
```
* **Memory Distribution:** Static Memory $\approx 2.5 \text{ GB}$ | Activations $\approx 6.0 \text{ GB}$ | **Total Peak VRAM $\approx 9.5 - 12.5 \text{ GB}$**

---

### Tier 3: 16 – 24 GB VRAM
* **Target Hardware:** NVIDIA RTX 3090 24GB, RTX 4090 24GB, NVIDIA A10G 24GB.
* **Target Model Scale:** **~280M – 450M Parameters**
* **Primary Focus:** Balanced width and depth for multi-step logic and code completion.

```python
# 16 - 24 GB VRAM Configuration (~287M Params)
vocab_size = 16384
emb_dim = 1024           # 1k embedding dimension
n_layers = 24            # 24 Transformer blocks
n_heads = 16             # 64 dim per head
n_kv_heads = 4           # GQA 4:1 ratio
context_length = 2048
batch_size = 8
gradient_accumulation = 2
```
* **Memory Distribution:** Static Memory $\approx 3.44 \text{ GB}$ | Activations $\approx 10.5 \text{ GB}$ | **Total Peak VRAM $\approx 15.0 - 19.5 \text{ GB}$**

---

### Tier 4: 24 – 32 GB VRAM
* **Target Hardware:** Intel Arc B570 Pro 32GB, NVIDIA V100 32GB, RTX 6000 Ada 48GB (low batch).
* **Target Model Scale:** **~500M – 800M Parameters**
* **Primary Focus:** Complex instruction-following, docstring synthesis, multi-file code understanding.

```python
# 24 - 32 GB VRAM Configuration (~580M Params)
vocab_size = 32768       # Extended vocabulary
emb_dim = 1280
n_layers = 28
n_heads = 20
n_kv_heads = 4
context_length = 2048
batch_size = 8
gradient_accumulation = 4
```
* **Memory Distribution:** Static Memory $\approx 6.96 \text{ GB}$ | Activations $\approx 16.0 \text{ GB}$ | **Total Peak VRAM $\approx 24.0 - 28.5 \text{ GB}$**

---

### Tier 5: 32 – 64 GB VRAM
* **Target Hardware:** NVIDIA A6000 48GB, A100 40GB/80GB, L40S 48GB, Dual RTX 3090/4090.
* **Target Model Scale:** **~1.0B – 2.2B Parameters**
* **Primary Focus:** High-capacity general code generation and advanced reasoning.

```python
# 32 - 64 GB VRAM Configuration (~1.3B Params)
vocab_size = 32768
emb_dim = 2048
n_layers = 24
n_heads = 32
n_kv_heads = 8
context_length = 4096    # Extended 4k sequence length
batch_size = 8
gradient_accumulation = 4
```
* **Memory Distribution:** Static Memory $\approx 15.6 \text{ GB}$ | Activations $\approx 24.0 \text{ GB}$ | **Total Peak VRAM $\approx 42.0 - 52.0 \text{ GB}$**

---

### Tier 6: 64+ GB VRAM
* **Target Hardware:** NVIDIA A100 80GB, H100 80GB, H200 141GB, Multi-GPU Distributed (FSDP / DeepSpeed ZeRO-3).
* **Target Model Scale:** **~3.0B – 7.0B+ Parameters**
* **Primary Focus:** Frontier-class small language models, full repository understanding.

```python
# 64+ GB VRAM Configuration (~3.8B Params)
vocab_size = 32768
emb_dim = 3072
n_layers = 32
n_heads = 32
n_kv_heads = 8
context_length = 4096
batch_size = 16
gradient_accumulation = 2
```
* **Memory Distribution:** Static Memory $\approx 45.6 \text{ GB}$ | Activations $\approx 28.0 \text{ GB}$ | **Total Peak VRAM $\approx 74.0 \text{ GB}$**

---

## 🛠️ Dynamic Memory Estimator Script

Run this Python snippet inside your project to estimate VRAM requirements before initializing model weights:

```python
def estimate_vram_requirements(
    vocab_size=16384,
    emb_dim=1024,
    n_layers=24,
    n_heads=16,
    n_kv_heads=4,
    context_length=2048,
    batch_size=8
):
    # Parameter estimation (approximate Llama-style)
    embed_params = vocab_size * emb_dim
    
    # Per layer: Attn (q, k, v, o) + FFN (w1, w2, w3) + Norms
    head_dim = emb_dim // n_heads
    q_params = emb_dim * (n_heads * head_dim)
    kv_params = 2 * emb_dim * (n_kv_heads * head_dim)
    o_params = (n_heads * head_dim) * emb_dim
    attn_params = q_params + kv_params + o_params
    
    hidden_dim = int(2 * (4 * emb_dim) / 3)
    hidden_dim = 256 * ((hidden_dim + 255) // 256)
    ffn_params = 3 * (emb_dim * hidden_dim)
    
    layer_params = attn_params + ffn_params
    total_params = embed_params + (n_layers * layer_params)
    
    # Memory calculations
    static_vram_gb = (total_params * 12) / (1024 ** 3)
    
    # Rough estimate for activation memory in mixed precision
    activation_vram_gb = (batch_size * context_length * n_layers * emb_dim * 16) / (1024 ** 3)
    total_vram_gb = static_vram_gb + activation_vram_gb

    print(f"--- Model Memory Footprint Estimate ---")
    print(f"Total Parameters:      {total_params / 1e6:.2f} Million")
    print(f"Static VRAM (AdamW):   {static_vram_gb:.2f} GB")
    print(f"Est. Activation VRAM:  {activation_vram_gb:.2f} GB")
    print(f"Est. Total Peak VRAM:  {total_vram_gb:.2f} GB")

if __name__ == "__main__":
    estimate_vram_requirements(emb_dim=1024, n_layers=24, batch_size=8)
```
