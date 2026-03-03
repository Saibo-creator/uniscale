from transformers import AutoConfig


def do_model_param_count(hidden_size: int, intermediate_size: int, num_attention_heads: int, num_key_value_heads: int, num_hidden_layers: int, vocab_size: int, qk_norm: bool = True, include_embeddings: bool = True) -> int:
    # Basic dimensions
    h = hidden_size
    i = intermediate_size
    n_heads = num_attention_heads
    n_kv_heads = num_key_value_heads
    d_head = h // n_heads

    def do_layer_param_count() -> int:
        def do_attn_param_count() -> int:
            # Multi-head / Grouped-query attention
            # W_q: (h, n_heads * d_head)
            # W_k, W_v: (h, n_kv_heads * d_head)
            # W_o: (n_heads * d_head, h)
            q_params = h * (n_heads * d_head)
            k_params = h * (n_kv_heads * d_head)
            v_params = h * (n_kv_heads * d_head)
            o_params = (n_heads * d_head) * h

            # qk_norm: layer norms for queries and keys (if enabled in config)
            # Each norm is d_head parameters per head
            qk_norm_params = 0
            if qk_norm:
                qk_norm_params = 2 * d_head  # q_norm and k_norm, one per head dimension

            # Note: attention_bias is false in config
            return q_params + k_params + v_params + o_params + qk_norm_params

        def do_mlp_param_count() -> int:
            # Apertus MLP only has 2 projections (up_proj, down_proj)
            # Unlike SwiGLU, there is NO gate_proj
            up_proj = h * i
            down_proj = i * h
            return up_proj + down_proj

        # Layer Norms (Input and Post-Attention/Pre-MLP)
        layer_norms = 2 * h

        return do_attn_param_count() + do_mlp_param_count() + layer_norms

    # Total layers
    total_layer_params = num_hidden_layers * do_layer_param_count()

    # Embeddings: vocab_size * hidden_size
    # (tie_word_embeddings is false, so we count input embeddings)
    embeddings = vocab_size * h

    # Final Layer Norm
    final_norm = h

    # Output projection (lm_head)
    lm_head = vocab_size * h

    return (total_layer_params + (embeddings if include_embeddings else 0) + final_norm + lm_head) / 1e9  # Divide by 1 billion, not 1 GiB


def get_30m_config():
    return {
        "hidden_size": 256,
        "intermediate_size": 256*6,  # 1024
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 4,
        "vocab_size": 131072,
        "qk_norm": True,
    }  # 0.37B


def get_100m_config():
    return {
        "hidden_size": 768,
        "intermediate_size": 768*6,  # 4608
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 6,
        "vocab_size": 131072,
        "qk_norm": True,
    }  # 0.152B


def get_300m_config():
    return {
        "hidden_size": 1024,
        "intermediate_size": 1024*6,  # 4096
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 12,
        "vocab_size": 131072,
        "qk_norm": True,
    }  # 0.317B


def get_1b_config():
    return {
        "hidden_size": 2048,
        "intermediate_size": 2048*6,  # take from https://github.com/swiss-ai/pretrain/blob/data_ablation/megatron/data_ablation/submit-apertus-1b-100bt.sh
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 16,
        "vocab_size": 131072,
        "qk_norm": True,
    }  # 1.24B


def get_3b_config():
    return {
        "hidden_size": 3072,
        "intermediate_size": 12288,  # take from https://github.com/swiss-ai/pretrain/blob/data_ablation/megatron/data_ablation/submit-apertus-3b-120bt.sh
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "num_hidden_layers": 28,
        "vocab_size": 131072,
        "qk_norm": True,
    }  # 3.22B


def get_8b_config():
    return {
        "hidden_size": 4096,
        "intermediate_size": int(4096*5.25),  # 5.25 x hidden size, as in the huggingface config
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "vocab_size": 131072,
        "qk_norm": True,
    }  # 8.05B


_SIZE_GETTERS = {
    "30M": get_30m_config,
    "100M": get_100m_config,
    "300M": get_300m_config,
    "1B": get_1b_config,
    "3B": get_3b_config,
    "8B": get_8b_config,
}


def get_apertus_config(model_size: str, vocab_size: int):
    """Load the Apertus base HF config and override with the requested size parameters."""
    if model_size not in _SIZE_GETTERS:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(_SIZE_GETTERS)}")
    params = _SIZE_GETTERS[model_size]()
    params["vocab_size"] = vocab_size
    config = AutoConfig.from_pretrained("swiss-ai/Apertus-8B-2509")
    for k, v in params.items():
        if k != "qk_norm":  # qk_norm is already True in the base config
            setattr(config, k, v)
    return config


def estimate_parameters(config) -> int:
    """Return estimated parameter count as a raw integer (compatible with train_lm.py)."""
    return int(do_model_param_count(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        num_hidden_layers=config.num_hidden_layers,
        vocab_size=config.vocab_size,
        qk_norm=getattr(config, "qk_norm", True),
        include_embeddings=True,
    ) * 1e9)


def estimate_parameters_for_size(model_size: str, vocab_size: int) -> int:
    """
    Return estimated parameter count for a given model size and vocab size,
    without making any network calls (unlike get_apertus_config + estimate_parameters).
    Suitable for use in launcher scripts.
    """
    if model_size not in _SIZE_GETTERS:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(_SIZE_GETTERS)}")
    params = _SIZE_GETTERS[model_size]()
    params["vocab_size"] = vocab_size
    return int(do_model_param_count(**params, include_embeddings=True) * 1e9)


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    config = AutoConfig.from_pretrained("swiss-ai/Apertus-8B-2509")
    print(f"Calculated params (with embeddings): {do_model_param_count(**get_8b_config(), include_embeddings=True):.3f}B")
    print(f"Calculated params (no embeddings): {do_model_param_count(**get_3b_config(), include_embeddings=False):.3f}B")
    print(f"Calculated params (no embeddings): {do_model_param_count(**get_1b_config(), include_embeddings=False):.3f}B")
    print(f"Calculated params (no embeddings): {do_model_param_count(**get_300m_config(), include_embeddings=False):.3f}B")
    print(f"Calculated params (no embeddings): {do_model_param_count(**get_100m_config(), include_embeddings=False):.3f}B")
    print(f"Calculated params (no embeddings): {do_model_param_count(**get_30m_config(), include_embeddings=False):.3f}B")

    exit()

    # Try to load model and verify (comment out if you don't need it)
    import torch
    model = AutoModelForCausalLM.from_pretrained(
        "swiss-ai/Apertus-8B-2509",
        torch_dtype=torch.float16,
        device_map=None,  # Load to CPU first, no meta tensors
        low_cpu_mem_usage=False
    )
    print(f"Actual model params: {model.num_parameters() / 1e9:.3f}B") # 8.05B, matches our calculation!
