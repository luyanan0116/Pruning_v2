import os
import random

import numpy as np
import torch
from datasets import load_dataset, load_from_disk


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_wikitext_raw_files(local_wiki_path: str):
    roots = [local_wiki_path, os.path.join(local_wiki_path, "wikitext-2-raw")]
    for root in roots:
        train = os.path.join(root, "wiki.train.raw")
        valid = os.path.join(root, "wiki.valid.raw")
        test = os.path.join(root, "wiki.test.raw")
        if os.path.exists(train) and os.path.exists(test):
            return train, valid if os.path.exists(valid) else None, test
    raise FileNotFoundError(
        "Cannot find WikiText-2 raw files. Expected wiki.train.raw/wiki.valid.raw/wiki.test.raw "
        f"under {local_wiki_path} or {os.path.join(local_wiki_path, 'wikitext-2-raw')}"
    )


def get_wikitext2(nsamples, seed, seqlen, tokenizer, eval_split="test"):
    local_wiki_path = os.environ.get("WIKITEXT2_PATH", "/root/dw2/Lya/Pruning/dataset_wikitext-raw")
    print(f"Loading local WikiText-2 from: {local_wiki_path}")
    train_file, valid_file, test_file = _resolve_wikitext_raw_files(local_wiki_path)
    if eval_split not in {"validation", "test"}:
        raise ValueError("eval_split must be validation or test")
    if eval_split == "validation":
        if valid_file is None:
            raise FileNotFoundError("wiki.valid.raw is required for validation-only tuning")
        eval_file = valid_file
    else:
        eval_file = test_file

    traindata = load_dataset("text", data_files={"train": train_file}, split="train")
    evaldata = load_dataset("text", data_files={"eval": eval_file}, split="eval")
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    evalenc = tokenizer("\n\n".join(evaldata["text"]), return_tensors="pt")

    set_seed(seed)
    trainloader = []
    for _ in range(int(nsamples)):
        max_start = trainenc.input_ids.shape[1] - seqlen
        if max_start < 0:
            raise ValueError("WikiText train tokens are shorter than requested seqlen")
        i = random.randint(0, max_start)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, evalenc


def get_c4(nsamples, seed, seqlen, tokenizer):
    local_c4_path = os.environ.get("C4_PATH", "/root/dw2/Lya/Pruning/dataset_c4")
    print(f"Loading local C4 from: {local_c4_path}")

    if os.path.exists(os.path.join(local_c4_path, "dataset_dict.json")) or os.path.exists(os.path.join(local_c4_path, "state.json")):
        dataset = load_from_disk(local_c4_path)
        traindata = dataset["train"]
        valdata = dataset["validation"]
    elif os.path.exists(os.path.join(local_c4_path, "en")):
        traindata = load_dataset(
            "json",
            data_files={"train": os.path.join(local_c4_path, "en/c4-train.00000-of-01024.json.gz")},
            split="train",
        )
        valdata = load_dataset(
            "json",
            data_files={"validation": os.path.join(local_c4_path, "en/c4-validation.00000-of-00008.json.gz")},
            split="validation",
        )
    else:
        traindata = load_dataset(local_c4_path, split="train")
        valdata = load_dataset(local_c4_path, split="validation")

    set_seed(seed)
    trainloader = []
    for _ in range(int(nsamples)):
        while True:
            doc_idx = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[doc_idx]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break
        max_start = trainenc.input_ids.shape[1] - seqlen - 1
        start = random.randint(0, max_start)
        end = start + seqlen
        inp = trainenc.input_ids[:, start:end]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    return trainloader, TokenizerWrapper(valenc)


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    lowered = name.lower()
    if "wikitext2" in lowered:
        split = "validation" if "validation" in lowered or "valid" in lowered else "test"
        return get_wikitext2(nsamples, seed, seqlen, tokenizer, eval_split=split)
    if "c4" in lowered:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    raise ValueError(f"Unknown dataset name: {name}")
