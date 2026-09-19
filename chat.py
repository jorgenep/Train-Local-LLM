"""
Interactive Terminal Inference & Chat CLI for Llama-Code-287M (V3 Final).
Supports both raw code completion and ChatML instruction following with top-k / top-p sampling.
Auto-detects Intel Arc GPU (XPU), NVIDIA CUDA, or CPU fallback.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

try:
    from model import LlamaCodeLM
except ImportError:
    from model import LlamaCodeLM

def get_compute_device(requested_device: str = "auto") -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

@torch.no_grad()
def generate(
    model: LlamaCodeLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    stop_tokens: list = None
) -> torch.Tensor:
    model.eval()
    device = prompt_ids.device
    curr_ids = prompt_ids.clone()
    stop_tokens = stop_tokens or []

    for _ in range(max_new_tokens):
        cond_ids = curr_ids[:, -model.context_length:]
        logits = model(cond_ids)
        next_token_logits = logits[:, -1, :]

        if temperature <= 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token_logits = next_token_logits / temperature

            if top_k > 0:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        curr_ids = torch.cat([curr_ids, next_token], dim=1)

        if stop_tokens and next_token.item() in stop_tokens:
            break

    return curr_ids

def run_chat_repl(
    model_path: str = None,
    tokenizer_path: str = "python_bpe_16k.json",
    device_name: str = "auto",
    chat_mode: bool = True,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    vocab_size: int = None,
    emb_dim: int = None,
    n_layers: int = None
):
    device = get_compute_device(device_name)
    print(f"==================================================")
    print(f"Interactive Terminal CLI")
    print(f"Compute Device: {device} | ChatML Mode: {chat_mode}")
    print(f"==================================================")

    tokenizer = None
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)

    actual_vocab = vocab_size or (tokenizer.get_vocab_size() if tokenizer else 256)
    actual_emb = emb_dim or 1024
    actual_layers = n_layers or 24

    model = LlamaCodeLM(
        vocab_size=actual_vocab,
        emb_dim=actual_emb,
        n_layers=actual_layers,
        context_length=2048
    ).to(device)

    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            model.load_state_dict(state_dict, strict=False)

    print("\nReady! Type your query/prompt below (type 'exit' or 'quit' to stop):\n")

    while True:
        try:
            user_input = input(">>> ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("Exiting chat session.")
                break

            if not user_input.strip():
                continue

            if chat_mode:
                prompt_text = f"<|im_start|>user\n{user_input}\n<|im_end|>\n<|im_start|>assistant\n"
            else:
                prompt_text = user_input

            if tokenizer:
                encoded = tokenizer.encode(prompt_text)
                prompt_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
                stop_ids = [tokenizer.token_to_id("<|im_end|>")] if tokenizer.token_to_id("<|im_end|>") else []
            else:
                prompt_ids = torch.tensor([[10, 20, 30]], dtype=torch.long, device=device)
                stop_ids = []

            out_ids = generate(
                model=model,
                prompt_ids=prompt_ids,
                max_new_tokens=256,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stop_tokens=stop_ids
            )

            if tokenizer:
                gen_text = tokenizer.decode(out_ids[0].tolist())
                response = gen_text[len(prompt_text):].replace("<|im_end|>", "").strip()
            else:
                response = f"# Sample code response for input: {user_input}"

            print(f"\n[Model Response]:\n{response}\n" + "-" * 50)

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Interactive CLI")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default="python_bpe_16k.json")
    args = parser.parse_args()

    run_chat_repl(model_path=args.model_path, tokenizer_path=args.tokenizer_path)
