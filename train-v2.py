"""
Main Pre-Training Execution Script
Supports Intel XPU (Arc GPUs), NVIDIA CUDA, and CPU fallback.
Implements Mixed Precision, AdamW, Cosine LR Decay, Gradient Clipping, and Checkpointing.
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import LlamaCodeLM
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
    batch_size: int = 4,
    learning_rate: float = 5e-4,
    max_steps: int = 1000,
    warmup_steps: int = 100,
    save_every: int = 250,
    checkpoint_dir: str = "/workspace/scratch/checkpoints"
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device, device_type = get_device_and_type()
    print(f"==================================================")
    print(f"Launching Pre-Training on Compute Device: {device} ({device_type.upper()})")
    print(f"==================================================")

    # 1. Initialize Dataset & DataLoader
    if not os.path.exists(bin_path):
        print(f"Warning: Binary data file '{bin_path}' not found. Using fallback mock dataset.")
        # Create a mock binary dataset for execution verification if needed
        data = np.random.randint(0, vocab_size, size=(context_length * 200,), dtype=np.uint16)
        data.tofile(bin_path)

    dataset = BinaryTokenDataset(bin_path, sequence_length=context_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 2. Instantiate Model
    model = LlamaCodeLM(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        n_layers=n_layers,
        context_length=context_length
    ).to(device)
    
    params = model.count_parameters()
    print(f"Model Initialized: {params['total_parameters']:,} Parameters")

    # 3. Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95)
    )

    # Cosine Annealing with Warmup
    def get_lr(step):
        if step < warmup_steps:
            return learning_rate * (step + 1) / (warmup_steps + 1)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 1e-5 + 0.5 * (learning_rate - 1e-5) * (1.0 + math.cos(math.pi * progress))

    # Mixed Precision Scaler
    scaler_enabled = (device_type in ["xpu", "cuda"])
    scaler = torch.amp.GradScaler(device=device_type, enabled=scaler_enabled)
    
    # Enable matmul precision optimization
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    model.train()
    start_time = time.time()
    step = 0
    
    print("\nStarting Training Loop...")
    data_iter = iter(dataloader)

    while step < max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        
        # Update Learning Rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        dtype = torch.bfloat16 if (device_type in ["xpu", "cuda"] and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=scaler_enabled):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Backward Pass & Gradient Clipping
        if scaler_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        step += 1

        # Progress Logging
        if step % 10 == 0 or step == max_steps:
            elapsed = time.time() - start_time
            tokens_processed = step * batch_size * context_length
            tokens_per_sec = tokens_processed / max(1.0, elapsed)
            print(f"Step {step:5d}/{max_steps} | Loss: {loss.item():.4f} | LR: {lr:.6f} | Speed: {tokens_per_sec:.0f} tok/s")

        # Save Checkpoint
        if step % save_every == 0 or step == max_steps:
            ckpt_path = os.path.join(checkpoint_dir, f"model_step_{step}.pth")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    print("\nPre-Training Completed Successfully!")
    return model

@torch.no_grad()
def generate_code(model, prompt_ids: torch.Tensor, max_new_tokens: int = 50, temperature: float = 0.8) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    idx = prompt_ids.to(device)
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.context_length:]
        logits = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        probs = nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
    return idx

if __name__ == "__main__":
    import numpy as np
    # Quick sanity check training run for 20 steps
    train_model(
        bin_path="/workspace/scratch/test_train_data.bin",
        vocab_size=256,
        emb_dim=256,
        n_layers=2,
        context_length=64,
        batch_size=2,
        max_steps=20,
        save_every=20,
        checkpoint_dir="/workspace/scratch/test_ckpts"
    )
