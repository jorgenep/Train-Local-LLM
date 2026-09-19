"""
Supervised Fine-Tuning (SFT) Script for Llama-Code-287M.
Fine-tunes a pre-trained LlamaCodeLM model on instruction-response pairs using ChatML format
with masked loss computation (loss computed exclusively on assistant response tokens).
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from model import LlamaCodeLM

def get_compute_device():
    """Detects available hardware compute device."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu"), "xpu"
    elif torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    else:
        return torch.device("cpu"), "cpu"

class SFTDataset(Dataset):
    """
    Dataset for ChatML Instruction Fine-Tuning with Masked Cross-Entropy Loss.
    Masks user prompt tokens with label index -100 so gradients flow only through response tokens.
    """
    def __init__(self, data_list, tokenizer, max_length=2048):
        if isinstance(tokenizer, str):
            tokenizer = Tokenizer.from_file(tokenizer)
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for item in data_list:
            user_msg = item.get("instruction", "") or item.get("user", "") or item.get("prompt", "")
            assistant_msg = item.get("response", "") or item.get("solution", "") or item.get("assistant", "") or item.get("output", "")
            
            if not user_msg or not assistant_msg:
                continue

            user_prefix = f"<|im_start|>user\n{user_msg}\n<|im_end|>\n<|im_start|>assistant\n"
            assistant_suffix = f"{assistant_msg}\n<|im_end|>"

            user_tokens = tokenizer.encode(user_prefix).ids
            assistant_tokens = tokenizer.encode(assistant_suffix).ids

            full_tokens = user_tokens + assistant_tokens
            if len(full_tokens) > max_length:
                user_tokens = user_tokens[:max_length // 2]
                assistant_tokens = assistant_tokens[:max_length // 2]
                full_tokens = user_tokens + assistant_tokens

            # Mask user prompt with -100 label
            labels = [-100] * len(user_tokens) + assistant_tokens

            self.samples.append({
                "input_ids": torch.tensor(full_tokens, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample["input_ids"], sample["labels"]

def collate_sft_batch(batch, pad_id=0):
    """Pads input_ids with pad_id and labels with -100 to uniform batch length."""
    max_len = max(inp.size(0) for inp, _ in batch)
    batch_input_ids = []
    batch_labels = []

    for inp, lbl in batch:
        pad_len = max_len - inp.size(0)

        if pad_len > 0:
            inp = torch.cat([inp, torch.full((pad_len,), pad_id, dtype=torch.long)])
            lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])

        batch_input_ids.append(inp)
        batch_labels.append(lbl)

    return torch.stack(batch_input_ids), torch.stack(batch_labels)

def train_sft(
    pretrained_path=None,
    sft_samples=None,
    tokenizer_path="python_bpe_16k.json",
    data_path="sft_data.json",
    checkpoint_dir="sft_checkpoints",
    vocab_size=16384,
    emb_dim=1024,
    n_layers=24,
    n_heads=16,
    n_kv_heads=4,
    context_length=2048,
    epochs=1,
    batch_size=2,
    lr=2e-5
):
    """Main Supervised Fine-Tuning routine."""
    device, device_type = get_compute_device()
    print(f"==================================================")
    print(f"Launching SFT Training on Compute Device: {device} ({device_type.upper()})")
    print(f"==================================================")

    if isinstance(tokenizer_path, str) and os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        tokenizer = tokenizer_path

    pad_id = tokenizer.token_to_id("<pad>") if hasattr(tokenizer, "token_to_id") and tokenizer.token_to_id("<pad>") is not None else 0

    if sft_samples is not None:
        data_list = sft_samples
    elif os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
    else:
        data_list = [
            {"instruction": "Write a Python function to check if a number is prime.",
             "response": "def is_prime(n):\n    return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))"},
            {"instruction": "Fix bug: def add(a,b): return a-b",
             "response": "def add(a,b):\n    return a+b"}
        ] * 4

    dataset = SFTDataset(data_list, tokenizer, max_length=context_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda b: collate_sft_batch(b, pad_id=pad_id)
    )

    actual_vocab_size = tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") else vocab_size
    state_dict = None

    if pretrained_path and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location=device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        
        # Infer dimensions if present in state_dict
        if "tok_emb.weight" in state_dict:
            actual_vocab_size, emb_dim = state_dict["tok_emb.weight"].shape
        if "layers.0.attn.k_proj.weight" in state_dict:
            kv_out_dim = state_dict["layers.0.attn.k_proj.weight"].shape[0]
            q_out_dim = state_dict["layers.0.attn.q_proj.weight"].shape[0]
            head_dim = emb_dim // n_heads if n_heads > 0 else 32
            if head_dim > 0:
                n_kv_heads = kv_out_dim // head_dim

    model = LlamaCodeLM(
        vocab_size=actual_vocab_size,
        emb_dim=emb_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        context_length=context_length
    ).to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    step = 0

    for epoch in range(epochs):
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            step += 1
            input_ids, labels = input_ids.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1),
                ignore_index=-100
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            print(f"SFT Step {step:2d} | Loss: {loss.item():.4f}")

    sft_checkpoint_path = os.path.join(checkpoint_dir, "sft_model_final.pth")
    torch.save(model.state_dict(), sft_checkpoint_path)
    print(f"SFT Completed! Saved checkpoint to: {sft_checkpoint_path}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for Llama Code Model")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default="python_bpe_16k.json")
    parser.add_argument("--data_path", type=str, default="sft_data.json")
    parser.add_argument("--checkpoint_dir", type=str, default="sft_checkpoints")
    args = parser.parse_args()

    train_sft(
        pretrained_path=args.pretrained_path,
        tokenizer_path=args.tokenizer_path,
        data_path=args.data_path,
        checkpoint_dir=args.checkpoint_dir
    )
