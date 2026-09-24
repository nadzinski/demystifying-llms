# EDUCATIONAL USE ONLY: not for company work or company data. See NOTICE.md.
import argparse
import os
import re
import sys
import time
from functools import partial

import torch

from model import CONFIG
from notice import print_notice
from sampling import sample_token
from stepper import Stepper
from tokenizer import ChatTokenizer
from weights import choose_device, download_files, load_model


def generate_tokens(model, tokenizer, prompt, choose, max_new_tokens=256, stats=None):
    """
    The whole of "generation": run the model, pick a token, append it, repeat.

    prompt is a list of token IDs. choose maps the model's scores to a token ID.
    If given, stats records how long the model itself took (not the sampling or printing).
    """
    device = next(model.parameters()).device
    remaining = CONFIG["max_seq_len"] - len(prompt)
    if remaining <= 0:
        raise ValueError("The conversation is too long. Use /clear to start again.")

    tokens = torch.tensor(prompt, dtype=torch.long, device=device)
    for _ in range(min(max_new_tokens, remaining)):
        started = time.perf_counter()
        # One score (a "logit") for every token in the vocabulary.
        logits = model(tokens)[-1, :tokenizer.vocabulary_size].float().cpu()
        if stats is not None:
            stats["model_seconds"] += time.perf_counter() - started
            stats["tokens"] += 1

        token_id = choose(logits)
        if token_id in tokenizer.stop_ids:
            return
        yield token_id
        # With no KV cache, the next pass processes the entire conversation again.
        tokens = torch.cat((tokens, tokens.new_tensor([token_id])))


def main():
    parser = argparse.ArgumentParser(description="Chat with our Qwen3-0.6B implementation.")
    parser.add_argument("--device", choices=("auto", "mps", "cuda", "cpu"), default="auto")
    parser.add_argument("--think", action="store_true", help="show the model's thinking before its answer")
    parser.add_argument("--step", action="store_true", help="generate one token at a time and choose them yourself")
    parser.add_argument("--raw", action="store_true", help="no chat: the model just continues whatever text you type")
    parser.add_argument("--temperature", type=float, help="default: 0.7, or 0.6 with --think; 0 selects the most likely token")
    parser.add_argument("--top-k", type=int, default=20, help="0 disables the top-k filter")
    parser.add_argument("--top-p", type=float, help="default: 0.8, or 0.95 with --think")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.temperature is None:
        args.temperature = 0.6 if args.think else 0.7
    if args.top_p is None:
        args.top_p = 0.95 if args.think else 0.8
    if not args.temperature >= 0 or args.top_k < 0 or not 0 < args.top_p <= 1 or args.max_new_tokens < 1:
        parser.error("Use temperature >= 0, top-k >= 0, 0 < top-p <= 1, and max-new-tokens >= 1.")
    if args.raw and args.think:
        parser.error("--think only applies to chat, not --raw.")

    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    try:
        device = choose_device(args.device)
    except ValueError as error:
        parser.error(str(error))

    color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    print_notice(color)
    print(f"Loading Qwen3-0.6B on {device}…")
    directory = download_files()
    tokenizer = ChatTokenizer(directory)
    model = load_model(directory, device)
    if args.raw:
        print("Ready (raw mode). Type the start of some text and the model will continue it.")
    else:
        print("Ready.")
    print("/prompt shows exactly what the model sees · /clear starts over · /quit exits\n")

    user_color, model_color, reset = ("\033[36m", "\033[32m", "\033[0m") if color else ("", "", "")
    thinking_color = "\033[2;37m" if color else ""
    dim, bold, highlight = ("\033[2m", "\033[1m", "\033[1;33m") if color else ("", "", "")

    if args.step:
        stepper = Stepper(tokenizer, args.temperature, args.top_k, args.top_p, (dim, bold, reset))
        choose = stepper
    else:
        stepper = None
        choose = partial(sample_token, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)

    system = {"role": "system", "content": "You are a helpful assistant."}
    history = [system]
    last_context = []

    try:
        while True:
            text = input(f"{user_color}You: ")
            print(reset, end="", flush=True)
            if text.strip() in ("/quit", "/exit"):
                break
            if text.strip() == "/clear":
                history = [system]
                last_context = []
                print("Conversation cleared.\n")
                continue
            if text.strip() == "/prompt":
                show_prompt(tokenizer, last_context, highlight, dim, reset)
                continue
            if not text.strip():
                continue

            if args.raw:
                messages = None
                prompt = tokenizer.encode(text)
            else:
                messages = history + [{"role": "user", "content": text}]
                prompt = tokenizer.encode_chat(messages, think=args.think)

            if stepper:
                stepper.new_reply()
            stats = {"model_seconds": 0.0, "tokens": 0}
            generated = []

            def record(token_ids):
                for token_id in token_ids:
                    generated.append(token_id)
                    yield token_id

            token_ids = record(generate_tokens(model, tokenizer, prompt, choose, args.max_new_tokens, stats))
            try:
                if stepper:
                    # Step mode prints as it goes; afterwards show the whole reply cleanly.
                    for _ in token_ids:
                        pass
                    print(reset)
                    pieces = print_reply(tokenizer.stream_decode(generated), args.raw, model_color, thinking_color, reset)
                else:
                    pieces = print_reply(tokenizer.stream_decode(token_ids), args.raw, model_color, thinking_color, reset)
            except ValueError as error:
                print(f"{reset}\n{error}\n")
                continue
            finally:
                print(reset, end="", flush=True)

            last_context = prompt + generated
            if stats["tokens"]:
                speed = stats["tokens"] / stats["model_seconds"]
                print(f"\n{dim}[{stats['tokens']} tokens · {speed:.1f} tokens/sec · "
                      f"context is now {len(last_context)} tokens]{reset}")
            print()
            if not args.raw:
                history = messages + [{"role": "assistant", "content": "".join(pieces)}]
    except (EOFError, KeyboardInterrupt):
        print(f"{reset}\nBye.")


def print_reply(pieces_with_thinking, raw, model_color, thinking_color, reset):
    """Print the reply as it streams in; return the non-thinking text pieces."""
    pieces = []
    previous_thinking = None
    for piece, thinking in pieces_with_thinking:
        if thinking != previous_thinking:
            if previous_thinking is not None:
                print()
            label = "LLM (thinking)" if thinking else ("LLM (continuing)" if raw else "LLM")
            text_color = thinking_color if thinking else model_color
            print(f"{reset}{text_color}{label}: ", end="", flush=True)
            previous_thinking = thinking
        if not thinking:
            pieces.append(piece)
        print(piece, end="", flush=True)
    return pieces


def show_prompt(tokenizer, context, highlight, dim, reset):
    """Print the last context the model saw, special tokens and all."""
    if not context:
        print(f"{dim}Nothing sent to the model yet. Say something first.{reset}\n")
        return
    text = tokenizer.decode_everything(context)
    text = re.sub(r"(<\|[a-z_]+\|>|</?think>)", f"{highlight}\\1{reset}", text)
    print(f"{dim}──── everything the model saw last time: {len(context)} tokens, one long string ────{reset}")
    print(text)
    print(f"{dim}──── end ────{reset}\n")


if __name__ == "__main__":
    main()
