# EDUCATIONAL USE ONLY: not for company work or company data. See NOTICE.md.
"""
Step mode: generate one token at a time and see what the model was choosing between.

For each step we show the model's top candidates with two numbers:
  - "model": the probability the LLM itself gives each token
  - "sampled": the probability after temperature, top-k and top-p have been applied,
    which is what the random pick actually uses

Then you can accept the pick, choose a different token yourself, or let it run.
"""

import unicodedata

import torch

from sampling import next_token_distribution

SHOW_CANDIDATES = 5
SHOW_CONTEXT_CHARS = 70


class Stepper:
    def __init__(self, tokenizer, temperature, top_k, top_p, style):
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.dim, self.bold, self.reset = style
        self.new_reply()

    def new_reply(self):
        self.chosen = []
        self.autopilot = False
        self.printed = ""

    def __call__(self, logits):
        """Pick the next token, like sample_token, but with a human in the loop."""
        sample_probs, sample_ids = next_token_distribution(logits, self.temperature, self.top_k, self.top_p)
        picked = sample_ids[torch.multinomial(sample_probs, num_samples=1)].item()

        if self.autopilot:
            self.chosen.append(picked)
            self._print_new_text()
            return picked

        model_probs = torch.softmax(logits.float(), dim=-1)
        top_probs, top_ids = model_probs.topk(SHOW_CANDIDATES)
        sampled = dict(zip(sample_ids.tolist(), sample_probs.tolist()))
        candidates = top_ids.tolist()
        if picked not in candidates:
            candidates[-1] = picked

        so_far = self.tokenizer.decode(self.chosen).replace("\n", "↵")
        if len(so_far) > SHOW_CONTEXT_CHARS:
            so_far = "…" + so_far[-SHOW_CONTEXT_CHARS:]
        print(f"\n{self.bold}{so_far}{self.reset}▌" if so_far else f"\n{self.dim}(start of reply){self.reset}")
        print(f"{self.dim}     token              model   sampled{self.reset}")
        for number, token_id in enumerate(candidates, start=1):
            label = pad(self.tokenizer.token_label(token_id), 16)
            model_pct = f"{model_probs[token_id].item():7.1%}"
            sampled_pct = f"{sampled[token_id]:7.1%}" if token_id in sampled else "      —"
            marker = "  ◀ picked" if token_id == picked else ""
            print(f"  {number}. {label} {model_pct}  {sampled_pct}{marker}")

        answer = input(f"{self.dim}Enter = accept · 1-{len(candidates)} = choose · a = autopilot › {self.reset}").strip().lower()
        if answer.isdigit() and 1 <= int(answer) <= len(candidates):
            picked = candidates[int(answer) - 1]
        elif answer == "a":
            self.autopilot = True
            self.chosen.append(picked)
            self.printed = ""
            print(f"\n{self.bold}", end="")
            self._print_new_text()
            return picked
        self.chosen.append(picked)
        return picked

    def _print_new_text(self):
        text = self.tokenizer.decode(self.chosen)
        # Wait if the latest token is only part of a multi-byte character.
        if text.endswith("�"):
            return
        print(text[len(self.printed):], end="", flush=True)
        self.printed = text


def pad(text, width):
    """Pad or cut text to a column width, counting wide characters (like 。) as two."""
    out, used = "", 0
    for char in text:
        size = 2 if unicodedata.east_asian_width(char) in "WF" else 1
        if used + size > width:
            break
        out, used = out + char, used + size
    return out + " " * (width - used)
