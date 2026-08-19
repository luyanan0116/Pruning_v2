import argparse
import os
from importlib.metadata import version

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.eval import eval_ppl, eval_zero_shot
from lib.prune_paper import PAPER_METHODS, prune_paper
from lib.sparsity import check_transformer_weight_sparsity


print("torch", version("torch"))
print("transformers", version("transformers"))
print("accelerate", version("accelerate"))
print("# of gpus:", torch.cuda.device_count())


def get_llm(model_name: str, cache_dir: str = "llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.seqlen = model.config.max_position_embeddings
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-only v8: frequency-domain mutual information, granular-ball "
            "multi-granularity estimation, repeated LCB, and coverage-aware weight budget."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache_dir", default="llm_weights")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--sparsity_ratio", type=float, default=0.50,
                        help="v8 is designed for exact 50% zeroed transformer projection weights")
    parser.add_argument("--prune_method", required=True, choices=sorted(PAPER_METHODS))

    parser.add_argument("--paper_prune_targets", default="mlp,attention")
    parser.add_argument("--paper_weight_allocation", choices=["paper_nonuniform", "uniform"],
                        default="paper_nonuniform",
                        help="paper_nonuniform allocates weight budgets from paper scores; uniform is a score-free 50% reference")
    parser.add_argument("--paper_weight_min_unit_sparsity", type=float, default=0.35,
                        help="lower bound on the fraction of weights zeroed inside every scored unit")
    parser.add_argument("--paper_weight_max_unit_sparsity", type=float, default=0.65,
                        help="upper bound below 1.0; prevents complete head/channel removal")
    parser.add_argument("--paper_weight_mask_chunk", type=int, default=262144,
                        help="index chunk size for deterministic uniform within-unit zeroing")

    parser.add_argument("--lcb_lambda", type=float, default=0.5)
    parser.add_argument("--paper_lcb_repeats", type=int, default=10)
    parser.add_argument("--paper_lcb_sample_fraction", type=float, default=0.80)
    parser.add_argument("--paper_lcb_scenario_fraction", type=float, default=0.67)

    parser.add_argument("--paper_score_nsamples", type=int, default=64)
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
    parser.add_argument("--paper_min_purity_gain", type=float, default=0.0)
    parser.add_argument("--paper_min_radius_reduction", type=float, default=0.0)
    parser.add_argument("--paper_compactness_ratio", type=float, default=0.55)
    parser.add_argument("--paper_min_event_classes", type=int, default=2)
    parser.add_argument("--paper_min_event_count_per_ball", type=int, default=2)
    parser.add_argument("--paper_gb_fusion_mode",
                        choices=["equal", "inverse_sqrt_dispersion", "inverse_dispersion"],
                        default="inverse_sqrt_dispersion")
    parser.add_argument("--paper_gb_fusion_max_ratio", type=float, default=5.0)

    parser.add_argument("--paper_band_coverage_ratio", type=float, default=0.90)
    parser.add_argument("--paper_band_coverage_ratios", default=None)
    parser.add_argument("--paper_coverage_alpha", type=float, default=0.10)
    parser.add_argument("--paper_greedy_batches", type=int, default=64)

    parser.add_argument("--paper_cache_dir", default="paper_response_cache")
    parser.add_argument("--paper_report_dir", default="paper_report")
    parser.add_argument("--paper_overwrite_cache", action="store_true")
    parser.add_argument("--paper_overwrite_scores", action="store_true")
    parser.add_argument("--paper_plot_layers", default="first,middle,last")

    parser.add_argument("--c4_path", default=None)
    parser.add_argument("--wikitext2_path", default=None)
    parser.add_argument("--save", default=None)
    parser.add_argument("--save_model", default=None)
    parser.add_argument("--eval_zero_shot", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args) -> None:
    if abs(args.sparsity_ratio - 0.50) > 1e-12:
        parser.error("v8 intentionally fixes the target to exactly --sparsity_ratio 0.50")
    if not 0 <= args.paper_weight_min_unit_sparsity < args.paper_weight_max_unit_sparsity < 1:
        parser.error("paper weight sparsity bounds must satisfy 0 <= min < max < 1")
    if not args.paper_weight_min_unit_sparsity <= 0.50 <= args.paper_weight_max_unit_sparsity:
        parser.error("per-unit sparsity bounds must contain the global 0.50 target")
    if args.paper_weight_mask_chunk < 1024:
        parser.error("--paper_weight_mask_chunk must be >= 1024")
    if args.paper_lcb_repeats < 1:
        parser.error("--paper_lcb_repeats must be >= 1")
    if not 0 < args.paper_lcb_sample_fraction <= 1:
        parser.error("--paper_lcb_sample_fraction must be in (0,1]")
    if not 0 < args.paper_lcb_scenario_fraction <= 1:
        parser.error("--paper_lcb_scenario_fraction must be in (0,1]")
    if args.lcb_lambda < 0:
        parser.error("--lcb_lambda must be non-negative")
    if args.paper_response_length < 8 or args.paper_calib_seqlen < args.paper_response_length:
        parser.error("paper response length must be >=8 and <= calibration sequence length")
    if args.paper_min_event_count_per_ball < 1:
        parser.error("--paper_min_event_count_per_ball must be >= 1")
    if args.paper_gb_fusion_max_ratio < 1:
        parser.error("--paper_gb_fusion_max_ratio must be >= 1")
    if not 0 < args.paper_band_coverage_ratio <= 1:
        parser.error("--paper_band_coverage_ratio must be in (0,1]")
    if args.paper_coverage_alpha < 0:
        parser.error("--paper_coverage_alpha must be non-negative")
    if args.paper_band_coverage_ratios is not None:
        values = [float(v.strip()) for v in args.paper_band_coverage_ratios.split(",") if v.strip()]
        if len(values) != args.paper_num_bands:
            parser.error("--paper_band_coverage_ratios length must equal --paper_num_bands")
        if any(not 0 < value <= 1 for value in values):
            parser.error("each band coverage ratio must be in (0,1]")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    if args.c4_path:
        os.environ["C4_PATH"] = args.c4_path
    if args.wikitext2_path:
        os.environ["WIKITEXT2_PATH"] = args.wikitext2_path

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.seqlen = min(int(args.seqlen), int(model.config.max_position_embeddings))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("use device", device)

    print("paper-only weight pruning starts")
    prune_paper(args, model, tokenizer, device)

    print("*" * 30)
    sparsity_ratio = check_transformer_weight_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.9f}")
    print("*" * 30)
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        path = os.path.join(args.save, f"log_{args.prune_method}.txt")
        with open(path, "w", encoding="utf-8") as handle:
            print("method\tactual_sparsity\tppl_test", file=handle)
            print(f"{args.prune_method}\t{sparsity_ratio:.9f}\t{ppl_test:.9f}", file=handle)

    if args.eval_zero_shot:
        task_list = ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
        results = eval_zero_shot(args.model, model, tokenizer, task_list, 0, False)
        print("zero-shot evaluation results")
        print(results)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)


if __name__ == "__main__":
    main()
