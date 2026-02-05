"""
Evaluation metrics for language models.

Key metric: Perplexity per byte (bits-per-byte)
"""

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_perplexity_per_byte(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cuda",
    batch_size: int = 8,
    max_length: int = 2048,
) -> Tuple[float, float, float]:
    """
    Compute perplexity per byte for a model on given texts.

    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of text strings to evaluate
        device: Device to run on
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length

    Returns:
        Tuple of (perplexity_per_byte, bits_per_byte, perplexity_per_token)
    """
    model.eval()
    model.to(device)

    total_log_likelihood = 0.0
    total_bytes = 0
    total_tokens = 0

    with torch.no_grad():
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )

            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            # Count bytes for each text
            batch_bytes = [len(text.encode("utf-8")) for text in batch_texts]
            total_bytes += sum(batch_bytes)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute log likelihood
            # Shift for causal LM: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()

            # Compute per-token loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            )

            # Apply mask and sum
            loss = loss.view(shift_labels.size())
            loss = (loss * shift_mask).sum()

            total_log_likelihood += loss.item()

            # Count actual tokens (excluding padding)
            total_tokens += shift_mask.sum().item()

    # Compute metrics
    # Average negative log likelihood per token
    avg_nll_per_token = total_log_likelihood / total_tokens

    # Perplexity per token
    perplexity_per_token = math.exp(avg_nll_per_token)

    # Perplexity per byte
    # We need to account for the compression ratio (bytes per token)
    avg_bytes_per_token = total_bytes / total_tokens
    perplexity_per_byte = math.exp(avg_nll_per_token / avg_bytes_per_token)

    # Bits per byte (cross-entropy in bits)
    bits_per_byte = avg_nll_per_token / (avg_bytes_per_token * math.log(2))

    return perplexity_per_byte, bits_per_byte, perplexity_per_token


def compute_cross_entropy_per_byte(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cuda",
    batch_size: int = 8,
    max_length: int = 2048,
) -> float:
    """
    Compute cross-entropy per byte (bits per byte).

    This is often more interpretable than perplexity per byte.

    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of text strings to evaluate
        device: Device to run on
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length

    Returns:
        Bits per byte
    """
    _, bits_per_byte, _ = compute_perplexity_per_byte(
        model, tokenizer, texts, device, batch_size, max_length
    )
    return bits_per_byte
