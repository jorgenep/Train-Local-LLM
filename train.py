"""
Main Pre-Training Execution Script (Optimized V3 Final)
Supports Intel XPU (Arc GPUs), NVIDIA CUDA, and CPU fallback.
Implements Mixed Precision, torch.compile graph fusion, Gradient Accumulation,
DataLoader pinning, AdamW with Cosine Decay, and moving average speed / ETA tracking.
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from model import LlamaCodeLM
except ImportError:
    from model import LlamaCodeLM

try:
    from prepare_data import BinaryTokenDataset
except ImportError:
    from prepare_data import BinaryTokenDataset

def get_device_and_type():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu"), "xpu"
    elif torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    else:
        return torch.device("cpu"), "cpu"

def train_model(
    bin_path: str = "train_data.bin",
    vocab_size: int = 16384,
    emb_dim: int = 1024,
    n_layers: int = 24,
    context_length: int = 2048,
    micro_batch_size: int = 4,
    grad_accum_steps: int = 2,
    learning_rate: float = 5e-4,
    max_steps: int = 1000,
    warmup_steps: int = 100,
    save_every: int = 250,
    checkpoint_dir: str = "/workspace/scratch/checkpoints",
    use_compile: bool = True
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device, device_type = get_device_and_type()
    effective_batch_size = micro_batch_size * grad_accum_steps
    tokens_per_step = effective_batch_size * context_length

    print(f"==================================================")
    print(f"Launching Optimized Pre-Training on: {device} ({device_type.upper()})")
    print(f"Micro-batch Size: {micro_batch_size} | Grad Accum: {grad_accum_steps} | Effective Batch Size: {effective_batch_size}")
    print(f"Tokens per Optimizer Step: {tokens_per_step:,}")
    print(f"==================================================")

    # 1. Initialize Dataset & DataLoader with pinned memory for fast GPU streaming
    if not os.path.exists(bin_path):
        print(f"Warning: Binary data file '{bin_path}' not found. Creating mock dataset for execution check...")
        data = np.random.randint(0, vocab_size, size=(context_length * 200,), dtype=np.uint16)
        data.tofile(bin_path)

    dataset = BinaryTokenDataset(bin_path, sequence_length=context_length)
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=(device_type in ["xpu", "cuda"]),
        num_workers=2 if os.cpu_count() > 2 else 0,
        persistent_workers=True if os.cpu_count() > 2 else False
    )

    # 2. Instantiate Model Architecture
    model = LlamaCodeLM(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        n_layers=n_layers,
        context_length=context_length
    ).to(device)

    params = model.count_parameters()
    print(f"Model Initialized: {params['total_parameters']:,} Parameters")

    # Optional torch.compile for kernel fusion
    if use_compile and hasattr(torch, "compile") and device_type != "cpu":
        try:
            print("Applying torch.compile() for fused kernel execution...")
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile skipped: {e}")

    # 3. Optimizer & LR Scheduler
    use_fused = (device_type == "cuda" and hasattr(torch.optim.AdamW, "fused"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        fused=use_fused
    )

    def get_lr(step):
        if step < warmup_steps:
            return learning_rate * (step + 1) / (warmup_steps + 1)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 1e-5 + 0.5 * (learning_rate - 1e-5) * (1.0 + math.cos(math.pi * progress))

    scaler_enabled = (device_type in ["xpu", "cuda"])
    scaler = torch.amp.GradScaler(device=device_type, enabled=scaler_enabled)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    model.train()
    start_time = time.time()
    step = 0
    accum_loss = 0.0

    print("\nStarting Training Loop...")
    data_iter = iter(dataloader)

    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro_step in range(grad_accum_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            dtype = torch.bfloat16 if (device_type in ["xpu", "cuda"] and torch.cuda.is_bf16_supported()) else torch.float16
            with torch.autocast(device_type=device_type, dtype=dtype, enabled=scaler_enabled):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / grad_accum_steps

            if scaler_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_loss += loss.item() * grad_accum_steps

        # Clip Gradients & Optimizer Step
        if scaler_enabled:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        step += 1

        # Update LR
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Progress Logging & Moving Average ETA
        if step % 5 == 0 or step == max_steps or step == 1:
            elapsed = time.time() - start_time
            total_tokens = step * tokens_per_step
            tokens_per_sec = total_tokens / max(1.0, elapsed)
            remaining_steps = max_steps - step
            remaining_seconds = (remaining_steps * tokens_per_step) / max(1.0, tokens_per_sec)
            eta_mins = remaining_seconds / 60.0

            print(f"Step {step:5d}/{max_steps} | Loss: {step_loss:.4f} | LR: {lr:.6f} | Speed: {tokens_per_sec:.0f} tok/s | ETA: {eta_mins:.1f}m")

        # Save Checkpoint
        if step % save_every == 0 or step == max_steps:
            ckpt_path = os.path.join(checkpoint_dir, f"model_step_{step}.pth")
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({
                'step': step,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': step_loss,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    total_time = time.time() - start_time
    print(f"\nPre-Training Completed in {total_time:.1f}s!")
    return model

@torch.no_grad()
def generate_code(model, prompt_ids: torch.Tensor, max_new_tokens: int = 50, temperature: float = 0.8) -> torch.Tensor:
    model.eval()
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    device = next(raw_model.parameters()).device
    idx = prompt_ids.to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -raw_model.context_length:]
        logits = raw_model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        probs = nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

if __name__ == "__main__":
    train_model(
        bin_path="/workspace/scratch/test_train_data.bin",
        vocab_size=256,
        emb_dim=256,
        n_layers=2,
        context_length=64,
        micro_batch_size=2,
        grad_accum_steps=2,
        max_steps=5,
        save_every=5,
        checkpoint_dir="/workspace/scratch/test_ckpts",
        use_compile=False
    )
