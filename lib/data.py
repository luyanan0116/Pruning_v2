from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_wikitext_files(root: str | None) -> tuple[Path, Path] | None:
    if not root:
        return None
    base = Path(root)
    candidates = [base, base / "wikitext-2-raw"]
    for candidate in candidates:
        train = candidate / "wiki.train.raw"
        test = candidate / "wiki.test.raw"
        if train.exists() and test.exists():
            return train, test
    raise FileNotFoundError(
        f"WikiText-2 path {base} must contain wiki.train.raw and wiki.test.raw "
        "directly or inside wikitext-2-raw/"
    )


def get_wikitext2(nsamples: int, seed: int, seqlen: int, tokenizer):
    from datasets import load_dataset
    local = _resolve_wikitext_files(os.environ.get("WIKITEXT2_PATH"))
    if local is None:
        train_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        train_file, test_file = local
        train_data = load_dataset("text", data_files={"train": str(train_file)}, split="train")
        test_data = load_dataset("text", data_files={"test": str(test_file)}, split="test")

    train_encoding = tokenizer("\n\n".join(train_data["text"]), return_tensors="pt")
    test_encoding = tokenizer("\n\n".join(test_data["text"]), return_tensors="pt")
    if train_encoding.input_ids.shape[1] < seqlen:
        raise ValueError("WikiText-2 train text is shorter than the requested sequence length")

    set_seed(seed)
    train_loader = []
    max_start = train_encoding.input_ids.shape[1] - seqlen
    for _ in range(nsamples):
        start = random.randint(0, max_start)
        inputs = train_encoding.input_ids[:, start : start + seqlen]
        targets = inputs.clone()
        targets[:, :-1] = -100
        train_loader.append((inputs, targets))
    return train_loader, test_encoding


def _load_local_c4(root: Path):
    from datasets import load_dataset, load_from_disk
    if (root / "dataset_dict.json").exists() or (root / "state.json").exists():
        loaded = load_from_disk(str(root))
        if hasattr(loaded, "keys") and "train" in loaded:
            return loaded["train"], loaded.get("validation", loaded["train"])
        return loaded, loaded
    json_train = root / "en" / "c4-train.00000-of-01024.json.gz"
    json_valid = root / "en" / "c4-validation.00000-of-00008.json.gz"
    if json_train.exists() and json_valid.exists():
        train = load_dataset("json", data_files=str(json_train), split="train")
        valid = load_dataset("json", data_files=str(json_valid), split="train")
        return train, valid
    raise FileNotFoundError(
        f"C4 path {root} is neither a datasets save_to_disk directory nor a supported en/*.json.gz layout"
    )


def get_c4(nsamples: int, seed: int, seqlen: int, tokenizer):
    from datasets import load_dataset
    local = os.environ.get("C4_PATH")
    if local:
        train_data, validation_data = _load_local_c4(Path(local))
    else:
        train_data = load_dataset("allenai/c4", "en", split="train", streaming=True)
        validation_data = load_dataset("allenai/c4", "en", split="validation", streaming=True)

    set_seed(seed)
    train_loader = []
    if local:
        while len(train_loader) < nsamples:
            row = train_data[random.randint(0, len(train_data) - 1)]
            encoded = tokenizer(row["text"], return_tensors="pt")
            if encoded.input_ids.shape[1] < seqlen:
                continue
            start = random.randint(0, encoded.input_ids.shape[1] - seqlen)
            inputs = encoded.input_ids[:, start : start + seqlen]
            targets = inputs.clone()
            targets[:, :-1] = -100
            train_loader.append((inputs, targets))
        validation_text = " ".join(validation_data[:1100]["text"])
    else:
        for row in train_data:
            encoded = tokenizer(row["text"], return_tensors="pt")
            if encoded.input_ids.shape[1] < seqlen:
                continue
            start = random.randint(0, encoded.input_ids.shape[1] - seqlen)
            inputs = encoded.input_ids[:, start : start + seqlen]
            targets = inputs.clone()
            targets[:, :-1] = -100
            train_loader.append((inputs, targets))
            if len(train_loader) >= nsamples:
                break
        validation_rows = []
        for index, row in enumerate(validation_data):
            validation_rows.append(row["text"])
            if index >= 1099:
                break
        validation_text = " ".join(validation_rows)

    validation_ids = tokenizer(validation_text, return_tensors="pt").input_ids[:, : 256 * seqlen]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    return train_loader, TokenizerWrapper(validation_ids)


def get_loaders(name: str, nsamples: int = 128, seed: int = 0, seqlen: int = 2048, tokenizer=None):
    normalized = name.lower()
    if "wikitext2" in normalized:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in normalized:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    raise ValueError(f"unknown dataset name: {name}")
