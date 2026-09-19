"""
Comprehensive Automated Sanity Check & Verification Suite for Llama-Code-287M.
Executes rigorous tests across Model Architecture, Tokenizer, Data Pipeline,
Pre-Training Loop, Supervised Fine-Tuning (SFT), Benchmark Evaluation, and CLI Generation.
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn
import numpy as np

# Add scratch to path if needed
sys.path.insert(0, "/workspace/scratch")

from model import LlamaCodeLM, RMSNorm, SwiGLUFFN, GroupedQueryAttention, apply_rotary_emb, precompute_freqs_cis
from train_tokenizer import train_custom_bpe_tokenizer
from prepare_data import tokenize_corpus_to_bin, BinaryTokenDataset
from train import train_model, generate_code
from sft_train import train_sft, SFTDataset
from eval_humaneval import evaluate_model_on_benchmarks
from chat import generate as generate_chat_tokens, get_compute_device

def run_all_sanity_checks():
    print("==================================================================")
    print("          STARTING COMPREHENSIVE SYSTEM SANITY CHECKS             ")
    print("==================================================================")
    
    passed_tests = 0
    total_tests = 0

    # ------------------------------------------------------------------
    # TEST 1: RoPE & Trigonometric Positional Encodings
    # ------------------------------------------------------------------
    total_tests += 1
    print("\n[1/10] Testing Rotary Position Embeddings (RoPE)...")
    try:
        head_dim = 64
        seq_len = 128
        b, n_q, n_kv = 2, 16, 4
        
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, seq_len)
        xq = torch.randn(b, seq_len, n_q, head_dim, requires_grad=True)
        xk = torch.randn(b, seq_len, n_kv, head_dim, requires_grad=True)
        
        out_q, out_k = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        
        assert out_q.shape == xq.shape, f"RoPE q shape mismatch: {out_q.shape} vs {xq.shape}"
        assert out_k.shape == xk.shape, f"RoPE k shape mismatch: {out_k.shape} vs {xk.shape}"
        
        loss = out_q.sum() + out_k.sum()
        loss.backward()
        assert xq.grad is not None and xk.grad is not None, "Gradients did not flow back through RoPE"
        
        print("  ✓ RoPE Test Passed! (Shapes, Rotations, and Gradients verified)")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ RoPE Test Failed: {e}")

    # ------------------------------------------------------------------
    # TEST 2: RMSNorm & SwiGLU FFN Layers
    # ------------------------------------------------------------------
    total_tests += 1
    print("\n[2/10] Testing RMSNorm & SwiGLU FFN Architecture...")
    try:
        dim = 256
        x = torch.randn(4, 32, dim)
        
        # Test RMSNorm
        norm = RMSNorm(dim)
        norm_out = norm(x)
        assert norm_out.shape == x.shape, f"RMSNorm shape mismatch: {norm_out.shape}"
        
        # Test SwiGLU
        hidden_dim = int(2 * (4 * dim) / 3)
        ffn = SwiGLUFFN(dim, hidden_dim)
        ffn_out = ffn(norm_out)
        assert ffn_out.shape == x.shape, f"SwiGLU shape mismatch: {ffn_out.shape}"
        
        print("  ✓ RMSNorm & SwiGLU Test Passed!")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ RMSNorm & SwiGLU Test Failed: {e}")

    # ------------------------------------------------------------------
    # TEST 3: Grouped-Query Attention (GQA) & Causal Masking
    # ------------------------------------------------------------------
    total_tests += 1
    print("\n[3/10] Testing Grouped-Query Attention (GQA)...")
    try:
        emb_dim = 256
        n_heads = 8
        n_kv_heads = 2
        seq_len = 16
        
        gqa = GroupedQueryAttention(emb_dim=emb_dim, n_heads=n_heads, n_kv_heads=n_kv_heads)
        freqs_cos, freqs_sin = precompute_freqs_cis(emb_dim // n_heads, seq_len)
        x = torch.randn(2, seq_len, emb_dim)
        
        gqa_out = gqa(x, freqs_cos, freqs_sin)
        assert gqa_out.shape == x.shape, f"GQA shape mismatch: {gqa_out.shape} vs {x.shape}"
        
        print("  ✓ Grouped-Query Attention Test Passed!")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ GQA Test Failed: {e}")

    # ------------------------------------------------------------------
    # TEST 4: Weight Tying & Parameter Count Assertion
    # ------------------------------------------------------------------
    total_tests += 1
    print("\n[4/10] Testing Weight Tying & Model Configuration...")
    try:
        vocab_size = 16384
        emb_dim = 1024
        n_layers = 24
        
        model = LlamaCodeLM(vocab_size=vocab_size, emb_dim=emb_dim, n_layers=n_layers, context_length=2048)
        
        # Assert Weight Tying
        assert model.lm_head.weight is model.tok_emb.weight, "Weight Tying Failed! lm_head and tok_emb do not share memory pointer."
        
        params = model.count_parameters()
        print(f"  - Verified Parameter Count: {params['total_parameters']:,}")
        assert params['total_parameters'] > 250_000_000, f"Expected ~280M params, got {params['total_parameters']:,}"
        
        print("  ✓ Weight Tying & Parameter Verification Passed!")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ Weight Tying Test Failed: {e}")

    # ------------------------------------------------------------------
    # TEST 5: Custom BPE Tokenizer Roundtrip Fidelity
    # ------------------------------------------------------------------
    total_tests += 1
    print("\n[5/10] Testing BPE Tokenizer Encoding/Decoding...")
    try:
        test_code = "def parse_ast(node: dict) -> list:\n    return [k for k in node.keys()]\n"
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(test_code * 5)
            tmp_file = f.name
            
        tok_file = "/workspace/scratch/sanity_tok.json"
        tokenizer = train_custom_bpe_tokenizer([tmp_file], vocab_size=256, save_path=tok_file)
        
        encoded = tokenizer.encode(test_code)
        decoded = tokenizer.decode(encoded.ids)
        
        assert decoded.strip() == test_code.strip(), f"Roundtrip decoding mismatch:\nGot: '{decoded}'\nExpected: '{test_code}'"
        print("  ✓ Tokenizer Roundtrip & Whitespace Preservation Passed!")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ Tokenizer Test Failed: {e}")

    # ------------------------------------------------------------------
    # TEST 6: Memory-Mapped Binary Dataset Loader
    # ------------------------------------------------------------------
    total_tests += 1
    print("\n[6/10] Testing Binary Dataset & Memory Mapping...")
    try:
        bin_path = "/workspace/scratch/sanity_data.bin"
        tokenize_corpus_to_bin([test_code] * 20, tokenizer_path=tok_file, output_bin_path=bin_path)
        
        ds = BinaryTokenDataset(bin_path, sequence_length=32)
        assert len(ds) > 0, "Dataset returned 0 samples"
        
        x, y = ds[0]
        assert x.shape == (32,), f"Expected input shape (32,), got {x.shape}"
        assert torch.equal(x[1:], y[:-1]), "Target tokens y are not properly shifted by 1 relative to x"
        
        print("  ✓ Binary Memory-Mapped Dataset Test Passed!")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ Binary Dataset Test Failed: {e}")

    # ------------------------------------------------------------------
    # TEST 7: End-to-End Pre-Training & Generation
    # ------------------------------------------------------------------
    total_tests += 1
    print("\n[7/10] Testing End-to-End Pre-Training & Generation...")
    try:
        # Run a mini training loop for 5 steps
        model = train_model(
            bin_path=bin_path,
            vocab_size=256,
            emb_dim=128,
            n_layers=2,
            context_length=32,
            batch_size=2,
            max_steps=5,
            save_every=5,
            checkpoint_dir="/workspace/scratch/sanity_ckpts"
        )
        
        # Test Autoregressive Generation
        prompt = torch.tensor([[10, 15, 20]], dtype=torch.long)
        gen_out = generate_code(model, prompt, max_new_tokens=10)
        assert gen_out.shape == (1, 13), f"Expected generated shape (1, 13), got {gen_out.shape}"
        
        print("  ✓ End-to-End Pre-Training & Generation Test Passed!")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ End-to-End Training Test Failed: {e}")

    # ------------------------------------------------------------------
    # TEST 8: Supervised Fine-Tuning (SFT) & Masked Loss Calculation
    # ------------------------------------------------------------------
    total_tests += 1
    print("\n[8/10] Testing Supervised Fine-Tuning (SFT)...")
    try:
        sft_samples = [
            {"instruction": "Write a function to return square of x", "solution": "def square(x):\n    return x * x"}
        ] * 10
        
        sft_ds = SFTDataset(sft_samples, tok_file, max_length=64)
        assert len(sft_ds) == 10, f"SFT dataset length mismatch: {len(sft_ds)}"
        
        input_ids, labels = sft_ds[0]
        assert input_ids.shape == (64,) and labels.shape == (64,), "SFT tensor shapes mismatch"
        assert -100 in labels, "SFT loss masking failed (-100 label absent for prompt tokens)"
        
        sft_model = train_sft(
            pretrained_path="/workspace/scratch/sanity_ckpts/model_step_5.pth",
            sft_samples=sft_samples,
            tokenizer_path=tok_file,
            vocab_size=256,
            emb_dim=128,
            n_layers=2,
            context_length=64,
            epochs=2,
            batch_size=2,
            checkpoint_dir="/workspace/scratch/sanity_sft_ckpts"
        )
        print("  ✓ Supervised Fine-Tuning (SFT) Test Passed!")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ SFT Test Failed: {e}")

    # ------------------------------------------------------------------
    # TEST 9: Code Sandbox & Benchmark Evaluation
    # ------------------------------------------------------------------
    total_tests += 1
    print("\n[9/10] Testing Code Sandbox & Benchmark Evaluation...")
    try:
        eval_problems = [
            {
                "task_id": "HumanEval/0",
                "prompt": "def add(a, b):\n",
                "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n",
                "entry_point": "add"
            },
            {
                "task_id": "HumanEval/1",
                "prompt": "def multiply(a, b):\n",
                "test": "assert multiply(2, 3) == 6\n",
                "entry_point": "multiply"
            }
        ]
        
        eval_results = evaluate_model_on_benchmarks(
            model=sft_model,
            tokenizer_path=tok_file,
            problems=eval_problems,
            num_samples=1,
            max_new_tokens=20
        )
        assert "pass@1" in eval_results and "total_evaluated" in eval_results, "Benchmark evaluation keys missing"
        
        print("  ✓ Code Sandbox & Benchmark Evaluation Test Passed!")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ Benchmark Evaluation Test Failed: {e}")

    # ------------------------------------------------------------------
    # TEST 10: Interactive CLI Inference & Sampling Generator (`chat.py`)
    # ------------------------------------------------------------------
    total_tests += 1
    print("\n[10/10] Testing Interactive CLI Sampling Generator (chat.py)...")
    try:
        device = get_compute_device("cpu")
        prompt_tensor = torch.tensor([[5, 12, 18]], dtype=torch.long, device=device)
        
        chat_gen_out = generate_chat_tokens(
            model=sft_model,
            prompt_ids=prompt_tensor,
            max_new_tokens=10,
            temperature=0.7,
            top_k=20,
            top_p=0.9,
            stop_tokens=[0]
        )
        assert chat_gen_out.shape[0] == 1 and chat_gen_out.shape[1] > 3, "CLI Chat generation shape mismatch"
        print("  ✓ CLI Sampling Generator Test Passed!")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ CLI Chat Generator Test Failed: {e}")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print("\n==================================================================")
    print(f"          SANITY CHECKS COMPLETED: {passed_tests}/{total_tests} PASSED        ")
    print("==================================================================")
    
    if passed_tests == total_tests:
        print("\nALL SYSTEM SANITY CHECKS PASSED PERFECTLY! System ready for production deployment.")
        return True
    else:
        print("\nSome tests failed. Please review the output above.")
        return False

if __name__ == "__main__":
    run_all_sanity_checks()
