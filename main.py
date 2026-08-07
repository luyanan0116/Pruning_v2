import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from importlib.metadata import version
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.eval import eval_ppl, eval_zero_shot
from lib.prune import (
    check_sparsity,
    prune_ablate,
    prune_magnitude,
    prune_sparsegpt,
    prune_wanda,
)
from lib.prune_paper_strict import PAPER_METHODS, prune_paper


def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.seqlen = model.config.max_position_embeddings
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Wanda-free paper pipeline; Wanda is retained only as an explicit baseline."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--sparsity_ratio", type=float, default=0.15)
    parser.add_argument(
        "--sparsity_type", default="unstructured",
        choices=["unstructured", "4:8", "2:4"],
    )
    parser.add_argument(
        "--prune_method",
        required=True,
        choices=[
            "magnitude", "wanda", "sparsegpt",
            "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter",
            "paper_mi", "paper_mi_gb", "paper_mi_gb_lcb", "paper_full", "lcb",
        ],
    )
    parser.add_argument("--use_variant", action="store_true")

    # Strict paper path.
    parser.add_argument("--lcb_lambda", type=float, default=1.0)
    parser.add_argument("--n_samples_lcb", type=int, default=20)
    parser.add_argument("--paper_prune_targets", default="mlp,attention")
    parser.add_argument("--paper_apply_mode", choices=["shrink", "zero"], default="shrink")
    parser.add_argument("--paper_budget_metric", choices=["params", "uniform"], default="params")
    parser.add_argument("--paper_min_keep_per_block", type=int, default=1)

    parser.add_argument("--paper_score_nsamples", type=int, default=32)
    parser.add_argument("--paper_calib_seqlen", type=int, default=512)
    parser.add_argument("--paper_response_length", type=int, default=32)
    parser.add_argument("--paper_calib_dataset", choices=["c4", "wikitext2"], default="c4")
    parser.add_argument("--paper_scenario_ratios", default="0.5,0.75,1.0")
    parser.add_argument("--paper_event_bins", type=int, default=3)

    parser.add_argument("--paper_num_bins", type=int, default=16)
    parser.add_argument("--paper_num_bands", type=int, default=4)
    parser.add_argument("--paper_mi_neighbors", type=int, default=3)
    parser.add_argument("--paper_fast_small_mi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--paper_probe_units", type=int, default=64)
    parser.add_argument("--paper_kde_bandwidth_scale", type=float, default=1.0)

    parser.add_argument("--paper_purity_thresholds", default="0.65,0.75,0.85")
    parser.add_argument("--paper_min_ball_size", type=int, default=8)
    parser.add_argument("--paper_max_balls", type=int, default=64)
    parser.add_argument("--paper_max_ball_depth", type=int, default=5)
    parser.add_argument("--paper_gb_localization", choices=["unit_local", "layer_shared"], default="unit_local")
    parser.add_argument("--paper_gb_workers", type=int, default=16)
    parser.add_argument("--paper_gb_chunk_size", type=int, default=64)
    parser.add_argument("--paper_kde_scope", choices=["all", "probe", "none"], default="probe")
    parser.add_argument("--paper_lcb_workers", type=int, default=4)
    parser.add_argument("--paper_min_purity_gain", type=float, default=0.0)
    parser.add_argument("--paper_min_radius_reduction", type=float, default=0.0)
    parser.add_argument("--paper_compactness_ratio", type=float, default=0.55)
    parser.add_argument("--paper_min_event_classes", type=int, default=2)

    parser.add_argument("--paper_sample_fraction", type=float, default=0.8)
    parser.add_argument("--paper_scenario_fraction", type=float, default=1.0)
    parser.add_argument("--paper_band_coverage_ratio", type=float, default=0.90)
    parser.add_argument("--paper_coverage_alpha", type=float, default=0.25)
    parser.add_argument("--paper_greedy_batches", type=int, default=512)

    parser.add_argument("--paper_cache_dir", default="paper_response_cache_strict")
    parser.add_argument("--paper_report_dir", default="paper_structured_report")
    parser.add_argument("--paper_overwrite_cache", action="store_true")
    parser.add_argument("--paper_overwrite_scores", action="store_true")

    parser.add_argument("--cache_dir", default="llm_weights")
    parser.add_argument("--c4_path", default=None)
    parser.add_argument("--wikitext2_path", default=None)
    parser.add_argument("--save", default=None)
    parser.add_argument("--save_model", default=None)
    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    if args.c4_path:
        os.environ["C4_PATH"] = args.c4_path
    if args.wikitext2_path:
        os.environ["WIKITEXT2_PATH"] = args.wikitext2_path

    if args.prune_method in PAPER_METHODS and not 0 < args.sparsity_ratio < 1:
        parser.error("strict paper structural sparsity_ratio must be in (0,1)")

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        if args.prune_method in PAPER_METHODS:
            parser.error("strict paper path does not use N:M masks")
        if args.sparsity_ratio != 0.5:
            parser.error("N:M baseline sparsity requires sparsity_ratio=0.5")
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    print("torch", version("torch"))
    print("transformers", version("transformers"))
    print("accelerate", version("accelerate"))
    print("loading", args.model)
    model = get_llm(args.model, args.cache_dir)
    model.seqlen = min(int(args.seqlen), int(model.config.max_position_embeddings))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)

    paper_result = None
    if args.prune_method == "wanda":
        prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "magnitude":
        prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method == "sparsegpt":
        prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif "ablate" in args.prune_method:
        prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
    elif args.prune_method in PAPER_METHODS:
        paper_result = prune_paper(
            args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m
        )

    if paper_result is not None:
        actual_sparsity = float(paper_result["stats"]["actual_candidate_cost_prune_ratio"])
        print(
            "structural pruning:",
            json.dumps(paper_result["stats"], ensure_ascii=False, indent=2),
        )
    else:
        actual_sparsity = float(check_sparsity(model))
        print(f"weight sparsity sanity check: {actual_sparsity:.6f}")

    # PP/PPL stage: evaluate the actually pruned in-memory model.
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        with (out / f"log_{args.prune_method}.txt").open("w", encoding="utf-8") as f:
            print("method\tactual_sparsity\tppl_test", file=f)
            print(f"{args.prune_method}\t{actual_sparsity:.8f}\t{ppl_test:.8f}", file=f)

    if args.eval_zero_shot:
        task_list = [
            "boolq", "rte", "hellaswag", "winogrande",
            "arc_easy", "arc_challenge", "openbookqa",
        ]
        print(eval_zero_shot(args.model, model, tokenizer, task_list, 0, False))

    if args.save_model:
        out = Path(args.save_model)
        out.mkdir(parents=True, exist_ok=True)
        if paper_result is not None and args.paper_apply_mode == "shrink":
            # Per-layer dimensions can differ, so ordinary HF reconstruction from
            # a single global config is not guaranteed. Save the exact state plus
            # the structure manifest used to construct the in-memory model.
            torch.save(model.state_dict(), out / "pytorch_model_structured_state.bin")
            tokenizer.save_pretrained(out)
            manifest_src = Path(args.paper_report_dir) / "structural_pruning_manifest.json"
            if manifest_src.exists():
                (out / "structural_pruning_manifest.json").write_bytes(manifest_src.read_bytes())
            (out / "SAVE_NOTE.txt").write_text(
                "This model uses per-layer physical structural shrinkage. "
                "Use the saved structural_pruning_manifest.json to reconstruct layer shapes "
                "before loading pytorch_model_structured_state.bin.\n",
                encoding="utf-8",
            )
        else:
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)


if __name__ == "__main__":
    main()
