"""
Master Pipeline Orchestrator (run_all.py)
Executes the end-to-end Llama-Code-287M training and evaluation pipeline sequentially:
  1. train_tokenizer.py  -> Trains custom 16k BPE tokenizer on Python code
  2. prepare_data.py     -> Pre-tokenizes corpus into zero-RAM memory-mapped binary array
  3. train.py            -> Launches base model pre-training with torch.compile & mixed precision
  4. sft_train.py        -> Fine-tunes model on ChatML instruction/solution pairs
  5. eval_humaneval.py   -> Evaluates pass@1 accuracy on HumanEval coding tasks in a sandbox
"""

import os
import sys
import time
import subprocess
from datetime import datetime

PIPELINE_STAGES = [
    {
        "name": "1. Train Custom BPE Tokenizer",
        "script": "train_tokenizer.py",
        "args": [],
        "description": "Trains a 16,384 vocabulary Byte-Pair Encoding tokenizer on Python code."
    },
    {
        "name": "2. Prepare Binary Token Dataset",
        "script": "prepare_data.py",
        "args": [],
        "description": "Pre-tokenizes Python code files into a zero-RAM memory-mapped train_data.bin file."
    },
    {
        "name": "3. Base Model Pre-Training",
        "script": "train.py",
        "args": [],
        "description": "Pre-trains the ~287M Llama model (24 layers, GQA, RoPE, RMSNorm, SwiGLU) with AMP."
    },
    {
        "name": "4. Supervised Fine-Tuning (SFT)",
        "script": "sft_train.py",
        "args": [],
        "description": "Fine-tunes base model on ChatML instruction pairs with masked loss."
    },
    {
        "name": "5. HumanEval Benchmark Evaluation",
        "script": "eval_humaneval.py",
        "args": [],
        "description": "Evaluates code generation pass@1 performance in an isolated Python sandbox."
    }
]

def format_duration(seconds: float) -> str:
    """Formats time duration into human-readable hours, minutes, and seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def run_pipeline():
    pipeline_start = time.time()
    start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 70)
    print("      LLAMA-CODE-287M END-TO-END PIPELINE ORCHESTRATOR      ")
    print("=" * 70)
    print(f"Start Time: {start_timestamp}")
    print(f"Python Executable: {sys.executable}")
    print(f"Working Directory: {os.getcwd()}")
    print("=" * 70)

    completed_stages = []

    for idx, stage in enumerate(PIPELINE_STAGES, 1):
        stage_name = stage["name"]
        script = stage["script"]
        args = stage["args"]
        description = stage["description"]

        print(f"\n[{idx}/{len(PIPELINE_STAGES)}] STARTING STAGE: {stage_name}")
        print(f"Description: {description}")
        print(f"Command: {sys.executable} {script} {' '.join(args)}")
        print("-" * 70)

        if not os.path.exists(script):
            print(f"❌ ERROR: Script file '{script}' not found in directory!")
            print("Aborting pipeline execution.")
            sys.exit(1)

        stage_start = time.time()
        cmd = [sys.executable, script] + args

        try:
            # Execute subprocess streaming stdout/stderr directly to terminal
            result = subprocess.run(cmd, check=True)
            stage_duration = time.time() - stage_start
            completed_stages.append((stage_name, stage_duration, "SUCCESS"))
            print("-" * 70)
            print(f"✓ STAGE COMPLETED: {stage_name} (Duration: {format_duration(stage_duration)})")

        except subprocess.CalledProcessError as e:
            stage_duration = time.time() - stage_start
            completed_stages.append((stage_name, stage_duration, f"FAILED (Exit Code {e.returncode})"))
            print("\n" + "=" * 70)
            print(f"❌ PIPELINE FAILURE AT STAGE: {stage_name}")
            print(f"Script '{script}' exited with error code {e.returncode}.")
            print("=" * 70)
            sys.exit(e.returncode)

    pipeline_duration = time.time() - pipeline_start
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 70)
    print("                  PIPELINE SUMMARY REPORT                   ")
    print("=" * 70)
    print(f"Start Time: {start_timestamp}")
    print(f"End Time:   {end_timestamp}")
    print(f"Total Duration: {format_duration(pipeline_duration)}")
    print("-" * 70)
    for name, dur, status in completed_stages:
        print(f"  • {name:<35} | Status: {status:<10} | Time: {format_duration(dur)}")
    print("=" * 70)
    print("🎉 ALL STAGES COMPLETED SUCCESSFULLY! Model training & evaluation complete.")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
