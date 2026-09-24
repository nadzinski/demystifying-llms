# EDUCATIONAL USE ONLY: not for company work or company data. See NOTICE.md.
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import logging as hf_logging
from safetensors.torch import load_file

from model import CONFIG, Qwen3

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_DIR = Path(__file__).resolve().parent / "model_data" / "Qwen3-0.6B"


def download_files(directory=MODEL_DIR):
    directory = Path(directory)
    # Hide Hugging Face's "unauthenticated requests" notice; no account is needed for this model.
    hf_logging.set_verbosity_error()
    for filename in ("tokenizer.json", "model.safetensors"):
        if not (directory / filename).is_file():
            hf_hub_download(MODEL_ID, filename, revision=REVISION, local_dir=directory, token=False)
    return directory


def choose_device(name="auto"):
    if name == "auto":
        if torch.backends.mps.is_available():
            name = "mps"
        elif torch.cuda.is_available():
            name = "cuda"
        else:
            name = "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Metal is unavailable. Try --device cpu.")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Try --device cpu.")
    return torch.device(name)


def load_model(directory=MODEL_DIR, device="cpu"):
    device = torch.device(device)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        dtype = torch.float16

    weights = load_file(str(Path(directory) / "model.safetensors"))
    state = {
        "token_embedding_layer.weight": weights["model.embed_tokens.weight"],
        "output_norm.weight": weights["model.norm.weight"],
        "output_layer.weight": weights.get("lm_head.weight", weights["model.embed_tokens.weight"]).clone(),
    }

    for layer in range(CONFIG["num_transformers"]):
        source = f"model.layers.{layer}"
        target = f"transformer_blocks.{layer}"
        names = {
            "attention_norm.weight": "input_layernorm.weight",
            "feed_forward_norm.weight": "post_attention_layernorm.weight",
            "attention.q_norm.weight": "self_attn.q_norm.weight",
            "attention.k_norm.weight": "self_attn.k_norm.weight",
            "attention.out_proj.weight": "self_attn.o_proj.weight",
            "feed_forward.gate_proj.weight": "mlp.gate_proj.weight",
            "feed_forward.value_proj.weight": "mlp.up_proj.weight",
            "feed_forward.output_proj.weight": "mlp.down_proj.weight",
        }
        for our_name, checkpoint_name in names.items():
            state[f"{target}.{our_name}"] = weights[f"{source}.{checkpoint_name}"]

        for letter in ("q", "k", "v"):
            heads = CONFIG["num_heads"] if letter == "q" else CONFIG["num_kv_heads"]
            weight = weights[f"{source}.self_attn.{letter}_proj.weight"]
            # Unpack the checkpoint's flat projection into separate head matrices.
            weight = weight.reshape(heads, CONFIG["head_dim"], CONFIG["token_embedding_dim"])
            state[f"{target}.attention.W_{letter}"] = weight.transpose(1, 2).contiguous()

    state = {name: tensor.to(device=device, dtype=dtype) for name, tensor in state.items()}
    # Meta tensors describe shapes without allocating or randomly initializing weights.
    with torch.device("meta"):
        model = Qwen3()
    model.load_state_dict(state, strict=True, assign=True)
    model.requires_grad_(False)
    return model.eval()
