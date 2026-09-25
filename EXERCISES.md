# Exercises

Work in pairs if you can: one laptop breaking shouldn't stop anyone.

Reminder: educational use only. Use made-up prompts, never company code or data ([NOTICE.md](NOTICE.md)).

## Part 1: Turn the knobs (no code changes)

Try these in order. Each one takes a minute or two.

**1. How sure is it?**
```sh
uv run chat.py --step
```
Ask *"What is the capital of France?"* and press Enter to accept each token.
- **model** is the probability the LLM itself gives each candidate token.
- **sampled** is the probability after the sampling rules (temperature, top-k, top-p), which is
  what the random pick actually uses.

Now type a number to pick a *different* token and see where the answer goes. You're the sampler now.
(What happens if you pick `<|im_end|>`?)

**2. Turn up the temperature**
```sh
uv run chat.py --temperature 0      # run it twice with the same question
uv run chat.py --temperature 1.5    # and again
```
Ask each one to *"Write a one-line poem about dogs."* Which one gives the same answer every time? Why?
Try `--step` together with `--temperature 1.5`: the **model** column doesn't change, only **sampled** does.

**3. Take away the chat**
```sh
uv run chat.py --raw
```
Type `Once upon a time` or `def fibonacci(n):` or `Dear Hiring Manager,`. There's no assistant
here, just a machine continuing text. Where did the helpful assistant go?

**4. What does the model actually see?**
In a normal chat, have a short conversation and then type `/prompt`. Find your own message.
Find the system prompt. Find the empty `<think></think>` right before the model's reply: the code
put that there, so the model thinks its thinking is already done and goes straight to the answer.
Everything, the whole conversation, is one long string of tokens that gets fed in again for every new token.

While you're at it: keep an eye on the **tokens/sec** after each reply as the conversation gets longer.

**5. (Optional) Let it think**
```sh
uv run chat.py --think
```
Now the code *doesn't* close the thinking section for it, so the model writes its own
`<think>…</think>` first (shown dimmed) and then answers. Ask something short, like
*"Is 91 a prime number?"* Thinking is just more tokens, and this model thinks for a *long* time
(about 400 tokens for that question, a couple of minutes on an M2 Air). It's much nicer once
you've done the KV cache in part 2.

---

## Part 2: Change the machine

Pick one (or more!). Each has a prompt you can paste into your coding agent, and a branch with a
working solution if you'd rather just look: `git diff main solution/<name>`, or
`git checkout solution/<name>` to run it (`git checkout main` to come back).

### 1. Ban a word

**Goal:** make it impossible for the model to say "Paris", then ask it what the capital of France is.

**Hint:** the model outputs a score for every token in the vocabulary (see `generate_tokens` in
`chat.py`). A score of minus infinity means probability zero.

**Agent prompt:**
> In this repo, add a `--ban WORD...` option to chat.py. Find every token in the vocabulary that
> spells one of those words (ignoring case and surrounding spaces) and set their logits to
> -infinity before a token is chosen. Keep the change small and in the style of the existing code.

**Try:** `uv run chat.py --ban Paris`, and ask about the capital of France and the Eiffel Tower. Does it
find a way around the ban, or does it just… make something up? Try it with `--step` too.

**Solution:** `solution/ban-a-word`

### 2. Knock out layers

**Goal:** the model is 28 identical-looking layers in a row (look at the `for` loop in
`Qwen3.forward` in `model.py`). What happens if you skip some?

**Hint:** each layer *adds* its result to a running total (the "residual stream"), so skipping a layer
just means not adding anything.

**Agent prompt:**
> In this repo, add a `--skip-layers` option to chat.py that takes layer numbers like `10-20` or
> `3,5,7`, and make Qwen3.forward in model.py skip those transformer blocks. Keep the change small
> and in the style of the existing code.

**Try:** `--temperature 0` with `--skip-layers 27`, then `14`, then `12,13`, then `8-11`, then `20-27`,
then just `0`. Some layers it barely notices; others it can't live without. Can you find a layer that
changes its *style* but not its answer? Nobody fully knows why each layer matters as much as it does.

**Solution:** `solution/knock-out-layers`

### 3. Add a KV cache (for performance nerds)

**Goal:** you probably noticed it gets slower as the conversation gets longer. That's because, for
every single new token, the model re-reads the *entire* conversation from scratch
(`generate_tokens` in `chat.py`). Fix it.

**Hint:** inside attention, each token produces a *key* and a *value*. For the tokens it has already
seen, those never change. Save them per layer, and on the next step only feed in the one new token.
Watch out: the positional encoding (RoPE) and the causal mask both need to know where the new token
sits in the sequence.

**Agent prompt:**
> In this repo, add a KV cache to the Qwen3 model in model.py and use it in generate_tokens in chat.py,
> so each step after the first only processes the newly generated token. Remember RoPE positions and the
> causal mask need an offset for the new tokens. Keep the code readable and in the style of the existing code.

**Try:** have a few-turns conversation before and after, and compare tokens/sec. On an M2 MacBook Air,
at ~500 tokens of context it goes from about 2 tokens/sec to about 20.

**Solution:** `solution/kv-cache`

---

**Finished early?** Explore [The Annotated LLM](https://nadiawadzinski.com/annotated-llm/), an
interactive diagram of this exact model with the code alongside, or read through `model.py`: the
comments are there for you.
