"""
Custom Byte-Pair Encoding (BPE) Tokenizer Training Script for Python Code
Builds a compact 16k vocabulary tailored specifically for Python syntax, keywords, and code structure.
"""

import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

def train_custom_bpe_tokenizer(
    corpus_files: list,
    vocab_size: int = 16384,
    save_path: str = "python_bpe_16k.json"
):
    print(f"Training BPE Tokenizer with vocab size {vocab_size:,}...")
    
    # Initialize Byte-Level BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # Pre-tokenization: Byte-level preserves code indentations and exact whitespace
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # Special tokens for standard LLM formatting & ChatML
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"]
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # Train tokenizer on corpus files
    tokenizer.train(files=corpus_files, trainer=trainer)
    
    # Enable post-processing for template formatting
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Save trained tokenizer
    tokenizer.save(save_path)
    print(f"Tokenizer trained and saved to {save_path}")
    return tokenizer

if __name__ == "__main__":
    # Self-test using a synthetic Python code sample
    sample_code = """
def fibonacci(n: int) -> int:
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

class CodeChecker:
    def __init__(self, debug_mode: bool = True):
        self.debug = debug_mode

    def run_check(self, input_data: list) -> dict:
        results = [x * 2 for x in input_data if x % 2 == 0]
        return {"status": "ok", "processed": results}
"""
    sample_file = "/workspace/scratch/sample_python.py"
    with open(sample_file, "w", encoding="utf-8") as f:
        f.write(sample_code * 10)  # Repeat to give tokenizer sufficient statistics
        
    tokenizer = train_custom_bpe_tokenizer([sample_file], vocab_size=512, save_path="/workspace/scratch/test_tok.json")
    
    # Test encoding and round-trip decoding
    encoded = tokenizer.encode(sample_code)
    decoded = tokenizer.decode(encoded.ids)
    
    print("\n--- Tokenizer Sanity Check ---")
    print(f"Original Text Length: {len(sample_code)} chars")
    print(f"Encoded Token Count: {len(encoded.ids)} tokens")
    print(f"Sample Token IDs: {encoded.ids[:15]}")
    print("Decoded Roundtrip Match:", decoded.strip() == sample_code.strip())
