# Demystifying LLMs

Run a real LLM (Qwen3-0.6B) on your laptop, using a few hundred lines of readable Python
plus ~750 million numbers downloaded from the internet. Then open it up and change how it behaves.

This is the companion repo for my *Demystifying LLMs* workshop.

> [!IMPORTANT]
> **Educational use only.** This runs an open-weight model (Qwen3, from Alibaba Cloud). Please don't use
> this model or this code for any company work, or with any company code or data. See [NOTICE.md](NOTICE.md).

## Setup (do this before the workshop)

Paste this into a terminal on your Mac:

```sh
git clone https://github.com/nadzinski/demystifying-llms.git && cd demystifying-llms && ./setup.sh
```

That clones this repo and runs [`setup.sh`](setup.sh), which installs [uv](https://docs.astral.sh/uv/)
(a Python package manager) if you don't already have it, installs PyTorch, downloads the model
(~1.5 GB, once), and runs it. It's safe to re-run.
You're ready when you see:

```
✓ Ready for the workshop
```

## What's in here

| File | What it is |
|---|---|
| `model.py` | **The whole LLM.** Every mathematical step from token IDs to next-token scores. |
| `sampling.py` | Turning scores into probabilities and picking one token: the only random part. |
| `chat.py` | The loop: run the model, pick a token, append it, repeat. Plus the chat CLI. |
| `stepper.py` | Step mode: watch (and choose) each token. |
| `tokenizer.py` | Text ↔ token IDs, and the chat format. |
| `notice.py`, `NOTICE.md` | The educational-use-only notice. |
| `setup.sh` | One-time setup: installs uv if needed, then runs `check.py`. |
| `weights.py` | Downloads the numbers from Hugging Face and loads them into `model.py`. |
| `model_data/` | The downloaded numbers (`model.safetensors`, 1.5 GB). Not in git. |

## Using it

```sh
uv run chat.py                      # chat
uv run chat.py --step               # one token at a time: see the candidates, pick your own
uv run chat.py --raw                # no chat, just "continue this text"
uv run chat.py --temperature 0      # always pick the most likely token (deterministic)
uv run chat.py --temperature 1.5    # more random
uv run chat.py --think              # let the model "think" out loud first
```

Inside the chat, `/prompt` shows exactly what the model sees, `/clear` starts over, `/quit` exits.

After each reply you'll see how many tokens per second the model generated. Watch that number as
a conversation gets longer…

## Exercises

See [EXERCISES.md](EXERCISES.md).

## Troubleshooting

- **`uv: command not found`** (right after setup installed it): open a new terminal window.
- **A dialog asks to install "command line developer tools"**: that's macOS installing `git`. Click
  Install, wait for it to finish, then paste the setup command again.
- **Errors mentioning MPS or Metal**: add `--device cpu` (slower, but works).
- **The download is stuck or failed**: re-run `./setup.sh`; it picks up where it left off.
  At the workshop, grab the model from Nadia's USB stick and copy the `Qwen3-0.6B` folder into `model_data/`.
- **Already cloned before the workshop?** Run `git pull` to get the latest.

## Want to go further?

| If you have… | Try |
|---|---|
| An hour or so | [3Blue1Brown, Deep Learning chapters 5–7](https://www.3blue1brown.com/lessons/gpt): transformers and attention, beautifully visualized. Start with chapter 5. |
| An evening or two | [Andrej Karpathy, *Deep Dive into LLMs like ChatGPT*](https://www.youtube.com/watch?v=7xTGNNLPyMI) (3.5 h): how models are trained, fine-tuned, and turned into assistants: the parts we skipped. |
| A few weekends | [Sebastian Raschka, *Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) ([code](https://github.com/rasbt/LLMs-from-scratch)): the book I learned from. |
| Curiosity about *why* it says what it says | [Anthropic, *Tracing the thoughts of a large language model*](https://www.anthropic.com/research/tracing-thoughts-language-model): mechanistic interpretability, i.e. looking inside a model to see how it reasons. |
| Right now | [The Annotated LLM](https://nadiawadzinski.com/annotated-llm/): an interactive, zoomable diagram of this exact model with its code alongside. |

## Licenses

The code in this repo is MIT licensed ([LICENSE](LICENSE)). The Qwen3-0.6B model weights and
tokenizer are downloaded from [Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B) and are
© Alibaba Cloud under the Apache License 2.0 ([Qwen3-LICENSE.txt](Qwen3-LICENSE.txt)).
