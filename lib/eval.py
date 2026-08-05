from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .data import get_loaders


def _input_device(model: torch.nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def evaluate_wikitext2_ppl(
    model: torch.nn.Module,
    tokenizer,
    seqlen: int,
    batch_size: int = 1,
) -> float:
    """Evaluate exact next-token perplexity on WikiText-2 raw test text."""
    if seqlen < 2 or batch_size < 1:
        raise ValueError("seqlen must be >=2 and batch_size must be positive")
    _, test_encoding = get_loaders(
        "wikitext2",
        nsamples=1,
        seed=0,
        seqlen=seqlen,
        tokenizer=tokenizer,
    )
    token_ids = test_encoding.input_ids
    sample_count = token_ids.numel() // seqlen
    if sample_count < 1:
        raise ValueError("WikiText-2 test set is shorter than one evaluation sequence")

    total_nll = 0.0
    total_tokens = 0
    device = _input_device(model)
    original_use_cache = getattr(model.config, "use_cache", None)
    if original_use_cache is not None:
        model.config.use_cache = False
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, sample_count, batch_size):
                stop = min(start + batch_size, sample_count)
                inputs = token_ids[:, start * seqlen : stop * seqlen].reshape(
                    stop - start, seqlen
                ).to(device)
                logits = model(input_ids=inputs, use_cache=False).logits[:, :-1, :].float()
                labels = inputs[:, 1:].to(logits.device)
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="sum",
                )
                total_nll += float(loss_sum.cpu())
                total_tokens += int(labels.numel())
                print(f"[PPL] sequences {stop}/{sample_count}", flush=True)
    finally:
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(math.exp(total_nll / total_tokens))


def evaluate_scenario_manifest_nll(
    model: torch.nn.Module,
    tokenizer,
    manifest_path: str,
    max_length: int,
) -> dict:
    """Evaluate causal next-token NLL/PPL per scenario in a JSON/JSONL manifest."""
    from collections import defaultdict

    from .paper_pruning.scenarios import load_scenario_manifest

    records = load_scenario_manifest(manifest_path, tokenizer, max_length=max_length, limit=None)
    totals = defaultdict(float)
    counts = defaultdict(int)
    device = _input_device(model)
    original_use_cache = getattr(model.config, "use_cache", None)
    if original_use_cache is not None:
        model.config.use_cache = False
    try:
        model.eval()
        with torch.inference_mode():
            for record in records:
                inputs = record["input_ids"].to(device)
                logits = model(input_ids=inputs, use_cache=False).logits[:, :-1, :].float()
                labels = inputs[:, 1:].to(logits.device)
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="sum",
                )
                key = str(record.get("scenario_id", "manifest"))
                totals[key] += float(loss_sum.cpu())
                counts[key] += int(labels.numel())
    finally:
        if original_use_cache is not None:
            model.config.use_cache = original_use_cache
    per_scenario = {
        key: {
            "tokens": counts[key],
            "mean_nll": totals[key] / counts[key],
            "ppl": math.exp(totals[key] / counts[key]),
        }
        for key in sorted(totals)
    }
    total_nll = sum(totals.values())
    total_tokens = sum(counts.values())
    return {
        "overall": {
            "tokens": total_tokens,
            "mean_nll": total_nll / total_tokens,
            "ppl": math.exp(total_nll / total_tokens),
        },
        "per_scenario": per_scenario,
    }
