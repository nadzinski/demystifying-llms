# EDUCATIONAL USE ONLY: not for company work or company data. See NOTICE.md.
"""Setup check: download the model, run it once, and confirm everything works."""

import os
import sys
import time

import torch

from chat import generate_tokens
from notice import print_notice
from sampling import sample_token
from tokenizer import ChatTokenizer
from weights import choose_device, download_files, load_model


def main():
    print_notice(sys.stdout.isatty() and "NO_COLOR" not in os.environ)
    device = choose_device()
    print(f"Using {device}. Downloading Qwen3-0.6B if needed (about 1.5 GB, first time only)…", flush=True)
    directory = download_files()
    tokenizer = ChatTokenizer(directory)
    model = load_model(directory, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Loaded {parameter_count:,} numbers into the model.")

    prompt = "The capital of France is"
    stats = {"model_seconds": 0.0, "tokens": 0}
    started = time.perf_counter()
    greedy = lambda logits: sample_token(logits, temperature=0)
    output = tokenizer.decode(list(generate_tokens(model, tokenizer, tokenizer.encode(prompt), greedy, 5, stats)))
    print(f"\n  {prompt}{output}\n")
    if "Paris" not in output:
        raise SystemExit("✗ The model ran but gave an unexpected answer. Ask Nadia!")
    print(f"✓ Ready for the workshop ({time.perf_counter() - started:.1f}s for {stats['tokens']} tokens).")
    print("  Try it now:  uv run chat.py")
    print("  Reminder: educational use only, not for company work or company data (see NOTICE.md).")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
