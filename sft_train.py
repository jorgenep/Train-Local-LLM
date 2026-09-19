"""
Supervised Fine-Tuning (SFT) Script for Llama Code Model (V3 Final)
Fine-tunes a pre-trained LlamaCodeLM model on instruction-response pairs (ChatML format)
with masked loss computation (loss computed only on assistant response tokens).
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

try:
    from model_final import LlamaCodeLM
except ImportError:
    from model import LlamaCodeLM

def get_compute_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu"), "xpu"
    elif torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    else:
        return torch.device("cpu"), "cpu"

class SFTDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for item in data_list:
            user_msg = item.get("instruction", "") or item.get("user", "")
            assistant_msg = item.get("response", "") or item.get("assistant", "")

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

            labels = [-100] * len(user_tokens) + assistant_tokens

            self.samples.append({
                "input_ids": torch.tensor(full_tokens, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_sft_batch(batch, pad_id=0):
    max_len = max(sample["input_ids"].size(0) for sample in batch)
    batch_input_ids = []
    batch_labels = []

    for sample in batch:
        inp = sample["input_ids"]
        lbl = sample["labels"]
        pad_len = max_len - inp.size(0)

        if pad_len > 0:
            inp = torch.cat([inp, torch.full((pad_len,), pad_id, dtype=torch.long)])
            lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])

        batch_input_ids.append(inp)
        batch_labels.append(lbl)

    return torch.stack(batch_input_ids), torch.stack(batch_labels)

def train_sft(
    model_path=None,
    tokenizer_path="python_bpe_16k.json",
    data_path="sft_data.json",
    output_dir="sft_checkpoints",
    epochs=1,
    batch_size=2,
    lr=2e-5,
    max_length=64,
    vocab_size=None,
    emb_dim=None,
    n_layers=None
):
    device, device_type = get_compute_device()
    print(f"==================================================")
    print(f"Launching SFT Training on Compute Device: {device} ({device_type.upper()})")
    print(f"==================================================")

    if not os.path.exists(tokenizer_path):
        print(f"Warning: Tokenizer file not found at {tokenizer_path}, checking fallback...")

    tokenizer = None
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)

    pad_id = tokenizer.token_to_id("<pad>") if (tokenizer and tokenizer.token_to_id("<pad>") is not None) else 0

    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data_list = json.load(f)
    else:
        data_list = [
            {"instruction": "Write a Python function to check if a number is prime.",
             "response": "def is_prime(n):\n    return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))"},
            {"instruction": "Fix bug: def add(a,b): return a-b",
             "response": "def add(a,b):\n    return a+b"}
        ] * 4

    if tokenizer is None:
        print("Creating fallback dataset with synthetic integer tokens...")
        class DummySFTDataset(Dataset):
            def __len__(self): return len(data_list)
            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randint(0, vocab_size or 256, (max_length,)),
                    "labels": torch.randint(0, vocab_size or 256, (max_length,))
                }
        dataset = DummySFTDataset()
    else:
        dataset = SFTDataset(data_list, tokenizer, max_length=max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=(lambda b: collate_sft_batch(b, pad_id=pad_id)) if tokenizer else None
    )

    actual_vocab = vocab_size or (tokenizer.get_vocab_size() if tokenizer else 256)
    actual_emb = emb_dim or 1024
    actual_layers = n_layers or 24

    model = LlamaCodeLM(
        vocab_size=actual_vocab,
        emb_dim=actual_emb,
        n_layers=actual_layers,
        context_length=max_length
    ).to(device)

    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"Notice: Loading state dict with strict=False due to shape differences: {e}")
            model.load_state_dict(state_dict, strict=False)

    os.makedirs(output_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    step = 0

    for epoch in range(epochs):
        for batch_idx, batch_data in enumerate(dataloader):
            if isinstance(batch_data, (tuple, list)):
                input_ids, labels = batch_data
            else:
                input_ids, labels = batch_data["input_ids"], batch_data["labels"]

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

    sft_checkpoint_path = os.path.join(output_dir, "sft_model_final.pth")
    torch.save(model.state_dict(), sft_checkpoint_path)
    print(f"SFT Completed! Saved checkpoint to: {sft_checkpoint_path}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for Llama Code Model")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default="python_bpe_16k.json")
    parser.add_argument("--data_path", type=str, default="sft_data.json")
    parser.add_argument("--output_dir", type=str, default="sft_checkpoints")
    args = parser.parse_args()

    train_sft(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
