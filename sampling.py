# EDUCATIONAL USE ONLY: not for company work or company data. See NOTICE.md.
import torch


def next_token_distribution(logits, temperature=0.7, top_k=20, top_p=0.8):
    """
    Turn the model's raw scores (logits) into the probabilities we actually sample from.

    Returns (probabilities, token_ids), most likely first. Only the candidates that
    survive the top-k and top-p filters are included.
    """
    if not temperature >= 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("Use temperature >= 0, top_k >= 0, and 0 < top_p <= 1.")
    if temperature == 0:
        # "Greedy": always take the single most likely token.
        return torch.ones(1), logits.argmax().unsqueeze(0)

    scores, token_ids = torch.sort(logits.float() / temperature, descending=True)
    if top_k:
        scores = scores[:top_k]
        token_ids = token_ids[:top_k]

    probabilities = torch.softmax(scores, dim=-1)

    # Keep the smallest prefix whose probability reaches top_p, including its last token.
    probability_before = probabilities.cumsum(dim=-1) - probabilities
    keep = probability_before < top_p
    probabilities = probabilities[keep]
    token_ids = token_ids[keep]
    return probabilities / probabilities.sum(), token_ids


def sample_token(logits, temperature=0.7, top_k=20, top_p=0.8):
    # This is the only random step in the whole LLM.
    probabilities, token_ids = next_token_distribution(logits, temperature, top_k, top_p)
    choice = torch.multinomial(probabilities, num_samples=1)
    return token_ids[choice].item()
