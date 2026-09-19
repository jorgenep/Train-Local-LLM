"""
Dataset Preparation & Disk-Backed Memory-Mapped DataLoader
Pre-tokenizes raw Python code datasets into uint16 binary files (.bin)
and provides an efficient PyTorch Dataset backed by np.memmap.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

def tokenize_corpus_to_bin(
    text_samples: list,
    tokenizer_path: str = "python_bpe_16k.json",
    output_bin_path: str = "train_data.bin"
):
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_token_id = tokenizer.token_to_id("</s>") or 3
    
    all_token_ids = []
    print("Tokenizing code samples...")
    for sample in text_samples:
        if sample and isinstance(sample, str):
            encoded = tokenizer.encode(sample)
            ids = encoded.ids
            ids.append(eos_token_id)
            all_token_ids.extend(ids)
            
    print(f"Total tokens generated: {len(all_token_ids):,}")
    all_tokens_np = np.array(all_token_ids, dtype=np.uint16)
    all_tokens_np.tofile(output_bin_path)
    print(f"Binary token file saved to {output_bin_path} ({os.path.getsize(output_bin_path) / (1024*1024):.2f} MB)")
    return output_bin_path

class BinaryTokenDataset(Dataset):
    """
    Disk-backed PyTorch Dataset utilizing np.memmap.
    Provides sliding context windows of length `sequence_length` without loading whole files into RAM.
    """
    def __init__(self, bin_path: str, sequence_length: int = 2048):
        self.seq_len = sequence_length
        self.bin_path = bin_path
        assert os.path.exists(bin_path), f"Binary dataset file {bin_path} does not exist!"
        
        # Memory-map binary file directly from disk
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.num_samples = (len(self.data) - 1) // self.seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        # Input tokens x, Target shifted by 1 token y
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:self.seq_len])
        y = torch.from_numpy(chunk[1 : self.seq_len + 1])
        return x, y

if __name__ == "__main__":
    # Self-test data preparation and BinaryTokenDataset
    sample_texts = [
        "def add(a, b):\n    return a + b\n",
        "class Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n",
        "import torch\nx = torch.randn(2, 2)\nprint(x.mean())\n"
    ] * 200  # Expand samples for testing sequence length
    
    # Train temporary mini tokenizer for data test
    from train_tokenizer import train_custom_bpe_tokenizer
    tmp_code_file = "/workspace/scratch/tmp_dataset_prep.txt"
    with open(tmp_code_file, "w") as f:
        f.write("\n".join(sample_texts))
        
    tok_path = "/workspace/scratch/tmp_prep_tok.json"
    train_custom_bpe_tokenizer([tmp_code_file], vocab_size=256, save_path=tok_path)
    
    bin_path = "/workspace/scratch/test_train_data.bin"
    tokenize_corpus_to_bin(sample_texts, tokenizer_path=tok_path, output_bin_path=bin_path)
    
    # Test DataLoader
    dataset = BinaryTokenDataset(bin_path, sequence_length=64)
    print(f"Dataset Length (samples): {len(dataset)}")
    if len(dataset) > 0:
        x, y = dataset[0]
        print("Sample x shape:", x.shape, "Sample y shape:", y.shape)
        print("Shifted label target match (x[1:] == y[:-1]):", torch.equal(x[1:], y[:-1]))
