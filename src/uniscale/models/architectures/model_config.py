"""
Model architecture configurations based on Apertus architecture.

Using ApertusConfig and ApertusForCausalLM but scaled to smaller sizes.
Reference: https://huggingface.co/swiss-ai/Apertus-8B-2509
"""

from typing import Dict
from transformers import AutoConfig


# Model size configurations following Apertus architecture principles
# Using GQA (Grouped Query Attention) for efficiency
MODEL_SIZE_CONFIGS: Dict[str, dict] = {
    "50M": {
        "hidden_size": 512,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,  # GQA with 4 queries per KV head
        "intermediate_size": 1376,  # ~2.7x hidden_size
    },
    "100M": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "num_key_value_heads": 3,  # GQA with 4 queries per KV head
        "intermediate_size": 2048,  # ~2.7x hidden_size
    },
    "300M": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,  # GQA with 4 queries per KV head
        "intermediate_size": 2816,  # ~2.7x hidden_size
    },
    "1B": {
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,  # GQA with 4 queries per KV head
        "intermediate_size": 5632,  # ~2.7x hidden_size
    },
    "3B": {
        "hidden_size": 2560,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,  # GQA with 4 queries per KV head
        "intermediate_size": 6912,  # ~2.7x hidden_size
    },
}


def get_apertus_config(model_size: str, vocab_size: int):
    """
    Get Apertus model configuration for a given size.

    Args:
        model_size: Model size (e.g., "50M", "100M", "300M", "1B")
        vocab_size: Vocabulary size from tokenizer

    Returns:
        ApertusConfig object
    """
    if model_size not in MODEL_SIZE_CONFIGS:
        raise ValueError(
            f"Unknown model size: {model_size}. "
            f"Available sizes: {list(MODEL_SIZE_CONFIGS.keys())}"
        )

    # Load base Apertus config to ensure we have all the right defaults
    config = AutoConfig.from_pretrained("swiss-ai/Apertus-8B-2509")

    # Update with our custom sizes
    size_config = MODEL_SIZE_CONFIGS[model_size]
    for key, value in size_config.items():
        setattr(config, key, value)

    # Set vocab size
    config.vocab_size = vocab_size

    return config


def estimate_parameters(config) -> int:
    """
    Estimate the number of parameters in the Apertus model.

    Args:
        config: ApertusConfig object

    Returns:
        Estimated parameter count
    """
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    intermediate_size = config.intermediate_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads

    # Embedding
    embedding_params = vocab_size * hidden_size

    # Transformer layers
    per_layer = 0

    # Self-attention (with GQA)
    kv_hidden = hidden_size * num_kv_heads // num_heads
    attn_params = (
        hidden_size * hidden_size  # Q projection
        + hidden_size * kv_hidden  # K projection
        + hidden_size * kv_hidden  # V projection
        + hidden_size * hidden_size  # O projection
    )

    # Feed-forward (gated MLP: gate_proj, up_proj, down_proj)
    ff_params = 3 * hidden_size * intermediate_size

    # RMS norms (2 per layer)
    norm_params = 2 * hidden_size

    per_layer = attn_params + ff_params + norm_params

    # Total transformer params
    transformer_params = per_layer * num_layers

    # Final norm and output layer
    final_params = hidden_size + vocab_size * hidden_size

    total = embedding_params + transformer_params + final_params

    return total
