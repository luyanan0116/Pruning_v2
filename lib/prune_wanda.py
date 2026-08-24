from __future__ import annotations

import torch

from .data import get_loaders
from .paper_pruning.wanda_mask import sequential_wanda_prune_


def _wanda_calibration_loader(args, tokenizer, model):
    seqlen = int(args.wanda_calib_seqlen or model.seqlen)
    if seqlen != int(model.seqlen):
        print(
            f"WARNING: Wanda calibration seqlen={seqlen} differs from evaluation/model seqlen={model.seqlen}; "
            "for an official-style baseline use both at 4096.",
            flush=True,
        )
    loader, _ = get_loaders(
        "c4",
        nsamples=int(args.wanda_nsamples),
        seed=int(args.seed),
        seqlen=seqlen,
        tokenizer=tokenizer,
    )
    return loader, seqlen


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0")):
    del device
    dataloader, seqlen = _wanda_calibration_loader(args, tokenizer, model)
    return sequential_wanda_prune_(
        model,
        dataloader,
        nsamples=int(args.wanda_nsamples),
        seqlen=seqlen,
        sparsity=float(args.sparsity_ratio),
        budget=None,
        storage_mode=args.wanda_activation_storage,
        prune_order=args.prune_order,
    )
