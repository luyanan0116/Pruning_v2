import argparse
import os
from pathlib import Path
from importlib.metadata import version

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.eval import eval_ppl, eval_zero_shot
from lib.prune_paper import PAPER_METHODS, prune_paper
from lib.prune_wanda import prune_wanda
from lib.sparsity import check_transformer_weight_sparsity


print("torch", version("torch"))
print("transformers", version("transformers"))
print("accelerate", version("accelerate"))
print("# of gpus:", torch.cuda.device_count())

ALL_METHODS = {"dense", "wanda", *PAPER_METHODS}


def _resolve_local_model_dir(model_path: str) -> str:
    """Resolve and validate a *local* Transformers-format model directory.

    V8.2-local intentionally never treats --model as a Hugging Face repo id.
    This prevents accidental network access and makes experiments reproducible
    on an offline server.
    """
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Local model directory does not exist: {path}. "
            "--model must point to a local Llama checkpoint directory, not a Hugging Face repo id."
        )

    config = path / "config.json"
    if not config.is_file():
        meta_original = (path / "params.json").is_file() and any(path.glob("consolidated.*.pth"))
        if meta_original:
            raise RuntimeError(
                f"Detected an original Meta Llama checkpoint at {path} "
                "(params.json + consolidated.*.pth), not a Transformers-format checkpoint. "
                "This code requires the local model to be converted once to Hugging Face/Transformers format."
            )
        raise FileNotFoundError(f"Missing config.json in local model directory: {path}")

    weight_files = (
        list(path.glob("*.safetensors"))
        + list(path.glob("pytorch_model*.bin"))
        + list(path.glob("model*.bin"))
    )
    if not weight_files:
        raise FileNotFoundError(
            f"No local model weight files found in {path}; expected *.safetensors or pytorch_model*.bin."
        )
    return str(path)


def get_llm(model_path: str, cache_dir: str | None = None):
    # Force all Hugging Face/Transformers components into offline mode.
    # from_pretrained is used only as the Transformers *local checkpoint reader*.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    local_dir = _resolve_local_model_dir(model_path)
    print(f"[local-only] loading model files from: {local_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        local_files_only=True,
    )
    model.seqlen = model.config.max_position_embeddings
    return model


def get_local_tokenizer(model_path: str):
    local_dir = _resolve_local_model_dir(model_path)
    print(f"[local-only] loading tokenizer files from: {local_dir}")
    return AutoTokenizer.from_pretrained(
        local_dir,
        use_fast=False,
        local_files_only=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "V8.2: paper MI/GB/LCB allocates a bounded nonuniform weight quota; "
            "Wanda activation-aware element metrics choose the actual zero positions; "
            "global transformer-projection sparsity is exactly 50%."
        )
    )
    parser.add_argument("--model", required=True, help="LOCAL Transformers-format model directory; repo IDs are rejected")
    parser.add_argument("--cache_dir", default="llm_weights")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seqlen", type=int, default=4096,
                        help="WikiText evaluation context; use 4096 for official-style Llama-2 comparison")
    parser.add_argument("--sparsity_ratio", type=float, default=0.50)
    parser.add_argument("--prune_method", required=True, choices=sorted(ALL_METHODS))

    # Official-style Wanda local masker / baseline.
    parser.add_argument("--wanda_nsamples", type=int, default=128)
    parser.add_argument("--wanda_calib_seqlen", type=int, default=4096)
    parser.add_argument("--wanda_activation_storage", choices=["auto", "cuda", "cpu"], default="auto",
                        help="where to store sequential 128xseqlen hidden-state buffers")
    parser.add_argument(
        "--prune_order", choices=["forward", "reverse", "joint"], default="forward",
        help=("inter-layer mask chronology: forward=V8.2 prune/propagate; "
              "reverse=freeze all dense stats then last->first; "
              "joint=freeze all dense stats before applying any masks")
    )

    # Paper contribution -> bounded unit quota.
    parser.add_argument("--paper_prune_targets", default="mlp,attention")
    parser.add_argument("--paper_weight_allocation", choices=["paper_nonuniform", "uniform"],
                        default="paper_nonuniform",
                        help="uniform bypasses paper scoring and is equivalent to the Wanda baseline")
    parser.add_argument("--paper_weight_min_unit_sparsity", type=float, default=0.45)
    parser.add_argument("--paper_weight_max_unit_sparsity", type=float, default=0.55)
    parser.add_argument("--paper_budget_temperature", type=float, default=1.0,
                        help="smaller => stronger 45%-55% score contrast; start from 1.0")
    parser.add_argument("--paper_weight_mask_chunk", type=int, default=262144,
                        help="legacy V8 compatibility knob; V8.2 Wanda masks do not use it")

    parser.add_argument("--lcb_lambda", type=float, default=0.5)
    parser.add_argument("--paper_lcb_repeats", type=int, default=20)
    parser.add_argument("--paper_lcb_sample_fraction", type=float, default=0.80)
    parser.add_argument("--paper_lcb_scenario_fraction", type=float, default=2 / 3)

    parser.add_argument("--paper_score_nsamples", type=int, default=128)
    parser.add_argument("--paper_calib_seqlen", type=int, default=1024)
    parser.add_argument("--paper_response_length", type=int, default=128)
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
    parser.add_argument(
        "--paper_use_band_gradient", action=argparse.BooleanOptionalAction, default=True,
        help=("enable/disable the Full-method frequency-band coverage gradient/refinement; "
              "--no-paper_use_band_gradient forces coverage alpha to 0")
    )
    parser.add_argument("--paper_greedy_batches", type=int, default=8,
                        help="V8.2 Full coverage-projection refinement iterations")

    parser.add_argument("--paper_cache_dir", default="paper_response_cache")
    parser.add_argument("--paper_report_dir", default="paper_report")
    parser.add_argument("--paper_overwrite_cache", action="store_true")
    parser.add_argument("--paper_overwrite_scores", action="store_true")
    parser.add_argument("--paper_plot_layers", default="first,middle,last")

    parser.add_argument("--eval_wikitext_split", choices=["validation", "test", "both"], default="validation",
                        help="tune on validation; use test only for final reporting")
    parser.add_argument("--c4_path", default=None)
    parser.add_argument("--wikitext2_path", default=None)
    parser.add_argument("--save", default=None)
    parser.add_argument("--save_model", default=None)
    parser.add_argument("--eval_zero_shot", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args) -> None:
    if args.prune_method != "dense" and abs(args.sparsity_ratio - 0.50) > 1e-12:
        parser.error("V8.2 pruning methods intentionally fix transformer projection sparsity to 0.50")
    if not 0 <= args.paper_weight_min_unit_sparsity < args.paper_weight_max_unit_sparsity < 1:
        parser.error("paper weight sparsity bounds must satisfy 0 <= min < max < 1")
    if not args.paper_weight_min_unit_sparsity <= 0.50 <= args.paper_weight_max_unit_sparsity:
        parser.error("per-unit sparsity bounds must contain global 0.50")
    if args.paper_budget_temperature <= 0:
        parser.error("--paper_budget_temperature must be positive")
    if args.wanda_nsamples < 1 or args.wanda_calib_seqlen < 8:
        parser.error("Wanda calibration requires positive nsamples and seqlen >= 8")
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

    print(f"loading LOCAL llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.seqlen = min(int(args.seqlen), int(model.config.max_position_embeddings))
    if args.wanda_calib_seqlen > int(model.config.max_position_embeddings):
        parser.error("--wanda_calib_seqlen exceeds model max_position_embeddings")
    model.eval()
    tokenizer = get_local_tokenizer(args.model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("use device", device)

    if args.prune_method == "dense":
        print("dense baseline: no pruning")
    elif args.prune_method == "wanda":
        print("official-style Wanda 50% baseline starts")
        prune_wanda(args, model, tokenizer, device)
    else:
        print("V8.2 paper-guided activation-aware weight pruning starts")
        prune_paper(args, model, tokenizer, device)

    print("*" * 30)
    sparsity_ratio = check_transformer_weight_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.9f}")
    print("*" * 30)

    ppl_validation = float("nan")
    ppl_test = float("nan")
    if args.eval_wikitext_split in {"validation", "both"}:
        ppl_validation = eval_ppl(args, model, tokenizer, device, split="validation")
        print(f"WikiText-2 validation perplexity {ppl_validation}")
    if args.eval_wikitext_split in {"test", "both"}:
        ppl_test = eval_ppl(args, model, tokenizer, device, split="test")
        print(f"WikiText-2 test perplexity {ppl_test}")

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        path = os.path.join(args.save, f"log_{args.prune_method}.txt")
        with open(path, "w", encoding="utf-8") as handle:
            print("method\tactual_sparsity\tppl_validation\tppl_test\tseqlen", file=handle)
            print(
                f"{args.prune_method}\t{sparsity_ratio:.9f}\t{ppl_validation:.9f}\t{ppl_test:.9f}\t{model.seqlen}",
                file=handle,
            )

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
