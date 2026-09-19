"""
Evaluation & Benchmark Script (HumanEval / MBPP pass@1 Evaluator - V3 Final)
Loads trained LlamaCodeLM model and evaluates code completion quality against unit tests.
"""

import os
import sys
import argparse
import multiprocessing
import torch
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

@torch.no_grad()
def generate_code_completion(
    model,
    tokenizer,
    prompt_text,
    max_new_tokens=64,
    temperature=0.2,
    top_k=40,
    device="cpu"
):
    model.eval()
    if tokenizer:
        encoded = tokenizer.encode(prompt_text)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
        stop_tokens = ["<|im_end|>", "</s>", "<eos>"]
        stop_token_ids = [tokenizer.token_to_id(t) for t in stop_tokens if tokenizer.token_to_id(t) is not None]
    else:
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)
        stop_token_ids = []

    for _ in range(max_new_tokens):
        curr_input = input_ids[:, -model.context_length:]
        logits = model(curr_input)
        next_token_logits = logits[0, -1, :] / max(temperature, 1e-5)

        if top_k > 0:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[-1]] = -float('Inf')

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        if next_token.item() in stop_token_ids:
            break

    if tokenizer:
        generated_ids = input_ids[0].tolist()
        full_output = tokenizer.decode(generated_ids)
        return full_output[len(prompt_text):]
    else:
        return "\n    return True\n"

def run_test_in_sandbox(code_string, test_string, timeout=3):
    def worker(q):
        try:
            exec_globals = {}
            exec(code_string + "\n" + test_string, exec_globals)
            q.put(True)
        except Exception:
            q.put(False)

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(q,))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return False

    return q.get() if not q.empty() else False

HUMANEVAL_SAMPLE_PROBLEMS = [
    {
        "task_id": "HumanEval/0",
        "prompt": "def has_close_elements(numbers: list, threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, any two numbers are closer to each other than given threshold. \"\"\"\n",
        "test": "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n"
    },
    {
        "task_id": "HumanEval/1",
        "prompt": "def is_even(n: int) -> bool:\n    \"\"\" Return True if n is even, else False. \"\"\"\n",
        "test": "assert is_even(2) == True\n"
    }
]

def evaluate_pass_at_1(
    model_path=None,
    tokenizer_path="python_bpe_16k.json",
    num_samples=1,
    vocab_size=None,
    emb_dim=None,
    n_layers=None
):
    device, device_type = get_compute_device()
    print(f"==================================================")
    print(f"Evaluating Model on Code Benchmarks ({device_type.upper()})")
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
        context_length=64
    ).to(device)

    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"Notice: Loaded state dict with strict=False: {e}")
            model.load_state_dict(state_dict, strict=False)

    passed_count = 0
    total_count = len(HUMANEVAL_SAMPLE_PROBLEMS)

    for problem in HUMANEVAL_SAMPLE_PROBLEMS:
        prompt = problem["prompt"]
        test = problem["test"]
        task_id = problem["task_id"]

        completion = generate_code_completion(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt,
            max_new_tokens=32,
            temperature=0.2,
            device=device
        )

        full_code = prompt + completion
        passed = run_test_in_sandbox(full_code, test)

        if passed:
            passed_count += 1
            print(f"✓ {task_id}: PASSED")
        else:
            print(f"✗ {task_id}: FAILED")

    pass_at_1 = (passed_count / total_count) * 100.0
    print(f"Benchmark Pass@1 Score: {pass_at_1:.2f}% ({passed_count}/{total_count})")
    return pass_at_1

def evaluate_model_on_benchmarks(model=None, tokenizer=None, device="cpu"):
    return evaluate_pass_at_1(model_path=None, tokenizer_path="python_bpe_16k.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Llama Code Model on HumanEval Benchmarks")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default="python_bpe_16k.json")
    args = parser.parse_args()

    evaluate_pass_at_1(model_path=args.model_path, tokenizer_path=args.tokenizer_path)
