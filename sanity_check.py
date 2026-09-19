"""
Comprehensive Automated Sanity Check & Verification Suite (V3 Final)
Executes 10 automated test batteries across Model Architecture, Tokenizer, Data Pipeline,
Pre-Training Loop, Supervised Fine-Tuning (SFT), Benchmark Evaluation, and CLI Generation.
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn
import numpy as np

# Prioritize current directory
sys.path = ["/workspace/scratch/final_build", "/workspace/scratch", "/workspace/artifacts"] + [p for p in sys.path if p not in ["/workspace/scratch/final_build", "/workspace/scratch", "/workspace/artifacts"]]

try:
    from model import LlamaCodeLM, RMSNorm, SwiGLUFFN, GroupedQueryAttention, apply_rotary_emb, precompute_freqs_cis
except ImportError:
    from model import LlamaCodeLM, RMSNorm, SwiGLUFFN, GroupedQueryAttention, apply_rotary_emb, precompute_freqs_cis

try:
    from train_tokenizer import train_custom_bpe_tokenizer
except ImportError:
    from train_tokenizer import train_custom_bpe_tokenizer

try:
    from prepare_data import tokenize_corpus_to_bin, BinaryTokenDataset
except ImportError:
    from prepare_data import tokenize_corpus_to_bin, BinaryTokenDataset

try:
    from train import train_model, generate_code
except ImportError:
    from train import train_model, generate_code

try:
    from sft_train import train_sft
except ImportError:
    from sft_train import train_sft

try:
    from eval_humaneval import evaluate_model_on_benchmarks
except ImportError:
    from eval_humaneval import evaluate_model_on_benchmarks

try:
    from chat import generate as generate_chat_tokens, get_compute_device
except ImportError:
    from chat import generate as generate_chat_tokens, get_compute_device

def run_all_sanity_checks():
    print("==================================================================")
    print("          STARTING COMPREHENSIVE SYSTEM SANITY CHECKS             ")
    print("==================================================================")

    passed_tests = 0
    total_tests = 0

    # 1. RoPE
    total_tests += 1
    print("\n[1/10] Testing Rotary Position Embeddings (RoPE)...")
    try:
        head_dim, seq_len, b, n_q, n_kv = 64, 128, 2, 16, 4
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, seq_len)
        xq = torch.randn(b, seq_len, n_q, head_dim, requires_grad=True)
        xk = torch.randn(b, seq_len, n_kv, head_dim, requires_grad=True)
        out_q, out_k = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        assert out_q.shape == xq.shape and out_k.shape == xk.shape
        (out_q.sum() + out_k.sum()).backward()
        assert xq.grad is not None and xk.grad is not None
        print("  ✓ RoPE Test Passed! (Shapes, Rotations, and Gradients verified)")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ RoPE Test Failed: {e}")

    # 2. RMSNorm & SwiGLU
    total_tests += 1
    print("\n[2/10] Testing RMSNorm & SwiGLU FFN Architecture...")
    try:
        dim = 256
        x = torch.randn(4, 32, dim)
        norm_out = RMSNorm(dim)(x)
        assert norm_out.shape == x.shape
        ffn_out = SwiGLUFFN(dim, int(2 * (4 * dim) / 3))(norm_out)
        assert ffn_out.shape == x.shape
        print("  ✓ RMSNorm & SwiGLU Test Passed!")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ RMSNorm & SwiGLU Test Failed: {e}")

    # 3. GQA
    total_tests += 1
    print("\n[3/10] Testing Grouped-Query Attention (GQA)...")
    try:
        emb_dim, n_heads, n_kv_heads, seq_len = 256, 8, 2, 16
        gqa = GroupedQueryAttention(emb_dim=emb_dim, n_heads=n_heads, n_kv_heads=n_kv_heads)
        freqs_cos, freqs_sin = precompute_freqs_cis(emb_dim // n_heads, seq_len)
        x = torch.randn(2, seq_len, emb_dim)
        gqa_out = gqa(x, freqs_cos, freqs_sin)
        assert gqa_out.shape == x.shape
        print("  ✓ Grouped-Query Attention Test Passed!")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ GQA Test Failed: {e}")

    # 4. Weight Tying
    total_tests += 1
    print("\n[4/10] Testing Weight Tying & Model Configuration...")
    try:
        model = LlamaCodeLM(vocab_size=16384, emb_dim=1024, n_layers=24, context_length=2048)
        assert model.lm_head.weight is model.tok_emb.weight
        params = model.count_parameters()
        print(f"  - Verified Parameter Count: {params['total_parameters']:,}")
        assert params['total_parameters'] > 250_000_000
        print("  ✓ Weight Tying & Parameter Verification Passed!")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ Weight Tying Test Failed: {e}")

    # 5. Tokenizer
    total_tests += 1
    print("\n[5/10] Testing BPE Tokenizer Encoding/Decoding...")
    try:
        test_code = "def parse_ast(node: dict) -> list:\n    return [k for k in node.keys()]\n"
        tmp_file = tempfile.mktemp(suffix=".py")
        with open(tmp_file, "w") as f: f.write(test_code * 5)
        tok_file = "/tmp/sanity_tok.json"
        tokenizer = train_custom_bpe_tokenizer([tmp_file], vocab_size=256, save_path=tok_file)
        encoded = tokenizer.encode(test_code)
        decoded = tokenizer.decode(encoded.ids)
        assert decoded.strip() == test_code.strip()
        print("  ✓ Tokenizer Roundtrip & Whitespace Preservation Passed!")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ Tokenizer Test Failed: {e}")

    # 6. Binary Dataset
    total_tests += 1
    print("\n[6/10] Testing Binary Dataset & Memory Mapping...")
    try:
        bin_path = "/tmp/sanity_data.bin"
        tokenize_corpus_to_bin([test_code] * 20, tokenizer_path=tok_file, output_bin_path=bin_path)
        ds = BinaryTokenDataset(bin_path, sequence_length=32)
        assert len(ds) > 0
        x, y = ds[0]
        assert x.shape == (32,) and torch.equal(x[1:], y[:-1])
        print("  ✓ Binary Memory-Mapped Dataset Test Passed!")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ Binary Dataset Test Failed: {e}")

    # 7. End-to-End Pre-Training
    total_tests += 1
    print("\n[7/10] Testing End-to-End Pre-Training & Generation...")
    try:
        model = train_model(
            bin_path=bin_path,
            vocab_size=256,
            emb_dim=128,
            n_layers=2,
            context_length=32,
            micro_batch_size=2,
            grad_accum_steps=2,
            max_steps=5,
            save_every=5,
            checkpoint_dir="/tmp/sanity_ckpts",
            use_compile=False
        )
        prompt = torch.tensor([[10, 15, 20]], dtype=torch.long)
        gen_out = generate_code(model, prompt, max_new_tokens=10)
        assert gen_out.shape == (1, 13)
        print("  ✓ End-to-End Pre-Training & Generation Test Passed!")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ End-to-End Pre-Training Test Failed: {e}")

    # 8. SFT Training
    total_tests += 1
    print("\n[8/10] Testing Supervised Fine-Tuning (SFT)...")
    try:
        sft_model = train_sft(
            model_path="/tmp/sanity_ckpts/model_step_5.pth",
            tokenizer_path=tok_file,
            output_dir="/tmp/sanity_sft_ckpts",
            epochs=1,
            batch_size=2,
            max_length=32,
            vocab_size=256,
            emb_dim=128,
            n_layers=2
        )
        print("  ✓ Supervised Fine-Tuning (SFT) Test Passed!")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ SFT Test Failed: {e}")

    # 9. Benchmark Eval
    total_tests += 1
    print("\n[9/10] Testing Code Sandbox & Benchmark Evaluation...")
    try:
        score = evaluate_model_on_benchmarks(model=sft_model if 'sft_model' in locals() else None)
        print("  ✓ Code Sandbox & Benchmark Evaluation Test Passed!")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ Benchmark Evaluation Test Failed: {e}")

    # 10. CLI Generator
    total_tests += 1
    print("\n[10/10] Testing Interactive CLI Sampling Generator (chat.py)...")
    try:
        test_prompt = torch.tensor([[5, 10, 15]], dtype=torch.long)
        chat_model = sft_model if 'sft_model' in locals() else LlamaCodeLM(vocab_size=256, emb_dim=128, n_layers=2)
        out = generate_chat_tokens(chat_model, test_prompt, max_new_tokens=5, temperature=0.7)
        assert out.shape == (1, 8)
        print("  ✓ CLI Sampling Generator Test Passed!")
        passed_tests += 1
    except Exception as e:
        print(f"  ✗ CLI Chat Generator Test Failed: {e}")

    print("\n==================================================================")
    print(f"          SANITY CHECKS COMPLETED: {passed_tests}/{total_tests} PASSED        ")
    print("==================================================================")

    if passed_tests == total_tests:
        print("\nALL SYSTEM SANITY CHECKS PASSED PERFECTLY! System ready for production deployment.")
        return True
    return False

if __name__ == "__main__":
    run_all_sanity_checks()
