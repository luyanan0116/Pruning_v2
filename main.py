from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lib.eval import evaluate_scenario_manifest_nll, evaluate_wikitext2_ppl
from lib.prune_paper import METHODS, prune_paper


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Frequency-domain mutual-information structural pruning: task-gradient "
            "responses, adaptive bands, granular-ball localization, dual-source "
            "repeated estimation, LCB ranking and deployment-budget configuration."
        )
    )
    parser.add_argument("--model", required=True, help="Hugging Face model name or local directory")
    parser.add_argument("--cache_dir", default="llm_weights")
    parser.add_argument("--device_map", choices=["auto", "cpu", "none"], default="auto")
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seqlen", type=int, default=2048, help="evaluation and deployment context length")

    parser.add_argument("--prune_method", choices=METHODS, default="paper_mi_gb_lcb")
    parser.add_argument("--prune_ratio", type=float, default=0.15)
    parser.add_argument("--paper_prune_targets", default="mlp,attention")
    parser.add_argument(
        "--paper_mask_style",
        choices=["structured_zero", "structured_surgery"],
        default="structured_zero",
        help="zero complete units without changing shapes, or physically shrink complete units",
    )
    parser.add_argument(
        "--paper_require_uniform_surgery",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Calibration samples and sample/scenario construction.
    parser.add_argument("--paper_calib_dataset", choices=["c4", "wikitext2"], default="c4")
    parser.add_argument("--paper_scenario_manifest", default=None)
    parser.add_argument("--paper_score_nsamples", type=int, default=128)
    parser.add_argument("--paper_calib_seqlen", type=int, default=2048)
    parser.add_argument("--paper_scenario_ratios", default="0.5,0.75,1.0")
    parser.add_argument("--paper_scenario_crops", default="prefix,center,suffix")
    parser.add_argument("--paper_event_bins", type=int, default=3)

    # Frequency-domain contribution spectrum.
    parser.add_argument("--paper_num_bins", type=int, default=16)
    parser.add_argument("--paper_num_bands", type=int, default=4)
    parser.add_argument("--paper_mi_neighbors", type=int, default=3)
    parser.add_argument("--paper_probe_units", type=int, default=64)
    parser.add_argument("--paper_band_weights", default=None)
    parser.add_argument("--paper_kde_bandwidth_scale", type=float, default=1.0)
    parser.add_argument(
        "--paper_max_merge_loss",
        type=float,
        default=None,
        help="optional stopping threshold for adjacent-bin information loss",
    )

    # Granular-ball localization and adaptive fusion.
    parser.add_argument("--paper_purity_thresholds", default="0.65,0.75,0.85")
    parser.add_argument("--paper_granularity_weights", default=None)
    parser.add_argument(
        "--paper_granularity_weight_mode",
        choices=["equal", "manual", "repeat_variance"],
        default="repeat_variance",
        help=(
            "equal/manual give strict MI->MI+GB->MI+GB+LCB component ablations; "
            "repeat_variance reproduces adaptive granularity fusion"
        ),
    )
    parser.add_argument("--paper_min_ball_size", type=int, default=8)
    parser.add_argument("--paper_max_balls", type=int, default=64)
    parser.add_argument("--paper_max_ball_depth", type=int, default=5)
    parser.add_argument("--paper_min_purity_gain", type=float, default=0.0)
    parser.add_argument("--paper_min_radius_reduction", type=float, default=0.0)
    parser.add_argument("--paper_compactness_ratio", type=float, default=0.55)
    parser.add_argument("--paper_min_event_classes", type=int, default=2)
    parser.add_argument("--paper_gb_workers", type=int, default=1)
    parser.add_argument("--paper_gb_chunk_size", type=int, default=64)
    parser.add_argument(
        "--paper_kde_scope",
        choices=["all", "probe", "none"],
        default="probe",
        help="auxiliary estimator diagnostics; primary scores always use the k-neighbour estimator",
    )

    # Sample-scenario dual-source uncertainty and LCB.
    parser.add_argument("--paper_lcb_repeats", type=int, default=20)
    parser.add_argument("--paper_sample_fraction", type=float, default=0.8)
    parser.add_argument("--paper_scenario_fraction", type=float, default=1.0)
    parser.add_argument("--paper_lcb_lambda", type=float, default=1.0)
    parser.add_argument("--paper_lcb_workers", type=int, default=1)

    # Frequency coverage and deployment budgets.
    parser.add_argument("--paper_band_coverage_ratio", type=float, default=0.90)
    parser.add_argument("--paper_coverage_alpha", type=float, default=0.25)
    parser.add_argument("--paper_fill_budget", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--paper_budget_metrics",
        default="params,flops,memory",
        help="comma-separated subset of params,flops,memory,kv_cache,latency",
    )
    parser.add_argument(
        "--paper_budget_keep_ratios",
        default=None,
        help="one keep ratio per active budget; defaults to 1-prune_ratio",
    )
    parser.add_argument(
        "--paper_budget_metric_weights",
        default=None,
        help="weights for the normalized scalar cost in the marginal-gain denominator",
    )
    parser.add_argument("--paper_latency_profile", default=None)
    parser.add_argument("--paper_dtype_bytes", type=int, default=2)
    parser.add_argument("--paper_require_coverage", action=argparse.BooleanOptionalAction, default=True)

    # Cache, reports and closed-loop validation.
    parser.add_argument("--paper_cache_dir", default="paper_response_cache")
    parser.add_argument("--paper_report_dir", default="paper_report")
    parser.add_argument("--paper_plot_layers", default="first,middle,last")
    parser.add_argument("--paper_overwrite_cache", action="store_true")
    parser.add_argument("--paper_overwrite_scores", action="store_true")
    parser.add_argument("--paper_score_only", action="store_true")
    parser.add_argument("--paper_selection_file", default=None)
    parser.add_argument(
        "--paper_post_prune_validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="recompute contribution spectra after shape-preserving pruning",
    )
    parser.add_argument("--paper_post_cache_dir", default="paper_response_cache_post")
    parser.add_argument("--paper_post_report_dir", default="paper_report_post")
    parser.add_argument("--paper_overwrite_post", action="store_true")

    parser.add_argument("--c4_path", default=None)
    parser.add_argument("--wikitext2_path", default=None)
    parser.add_argument("--eval_before", action="store_true")
    parser.add_argument("--eval_after", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_scenario_manifest", default=None)
    parser.add_argument("--output_dir", default="paper_run")
    parser.add_argument("--save_model", default=None)
    return parser


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dtype(name: str):
    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs: dict[str, Any] = {
        "cache_dir": args.cache_dir,
        "torch_dtype": _dtype(args.torch_dtype),
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map == "auto":
        kwargs["device_map"] = "auto" if torch.cuda.is_available() else "cpu"
    elif args.device_map == "cpu":
        kwargs["device_map"] = "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.device_map == "none":
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    model.seqlen = min(int(args.seqlen), int(getattr(model.config, "max_position_embeddings", args.seqlen)))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )
    return model, tokenizer




def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value

def _save_model(model, tokenizer, destination: str, mask_style: str) -> None:
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    if mask_style == "structured_zero":
        model.save_pretrained(target)
    else:
        torch.save(model.state_dict(), target / "structured_state_dict.pt")
        if hasattr(model.config, "to_json_file"):
            model.config.to_json_file(target / "base_config.json")
        summary = getattr(model, "_paper_structural_summary", {})
        (target / "structural_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    tokenizer.save_pretrained(target)


def _validate_args(parser: argparse.ArgumentParser, args) -> None:
    if not 0 <= args.prune_ratio < 1:
        parser.error("--prune_ratio must be in [0,1)")
    if args.paper_score_nsamples < 4:
        parser.error("--paper_score_nsamples must be at least 4")
    if args.paper_calib_seqlen < 8:
        parser.error("calibration length must be at least 8")
    if args.paper_num_bands > args.paper_num_bins:
        parser.error("--paper_num_bands cannot exceed --paper_num_bins")
    if args.paper_calib_seqlen < args.paper_num_bins:
        parser.error("--paper_calib_seqlen must be at least --paper_num_bins")
    if args.paper_selection_file and args.paper_score_only:
        parser.error("--paper_selection_file and --paper_score_only cannot be combined")
    if not args.paper_score_only and not args.paper_selection_file and args.prune_ratio <= 0:
        parser.error("a positive --prune_ratio is required when generating a pruning configuration")
    if args.paper_mask_style == "structured_surgery" and args.paper_post_prune_validate:
        # Unit indices and dimensions change after surgery, so pre/post spectra
        # cannot be compared one-to-one. The pruning function records this skip.
        pass


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    if args.c4_path:
        os.environ["C4_PATH"] = args.c4_path
    if args.wikitext2_path:
        os.environ["WIKITEXT2_PATH"] = args.wikitext2_path

    _set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading model: {args.model}")
    model, tokenizer = _load_model(args)
    args.seqlen = model.seqlen
    args.paper_calib_seqlen = min(args.paper_calib_seqlen, model.seqlen)

    before_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    baseline_ppl = None
    if args.eval_before:
        baseline_ppl = evaluate_wikitext2_ppl(
            model,
            tokenizer,
            seqlen=model.seqlen,
            batch_size=args.eval_batch_size,
        )

    before_scenarios = None
    if args.eval_before and args.eval_scenario_manifest:
        before_scenarios = evaluate_scenario_manifest_nll(
            model, tokenizer, args.eval_scenario_manifest, model.seqlen
        )

    pruning_result = prune_paper(args, model, tokenizer)
    after_parameters = int(sum(parameter.numel() for parameter in model.parameters()))

    after_ppl = None
    if args.eval_after and not args.paper_score_only:
        after_ppl = evaluate_wikitext2_ppl(
            model,
            tokenizer,
            seqlen=model.seqlen,
            batch_size=args.eval_batch_size,
        )

    after_scenarios = None
    if args.eval_after and args.eval_scenario_manifest and not args.paper_score_only:
        after_scenarios = evaluate_scenario_manifest_nll(
            model, tokenizer, args.eval_scenario_manifest, model.seqlen
        )

    if args.save_model and not args.paper_score_only:
        _save_model(model, tokenizer, args.save_model, args.paper_mask_style)

    result = {
        "model": args.model,
        "method": args.prune_method,
        "mask_style": args.paper_mask_style,
        "seed": args.seed,
        "seqlen": model.seqlen,
        "requested_prune_ratio": args.prune_ratio,
        "parameters_before": before_parameters,
        "parameters_after": after_parameters,
        "physical_parameter_reduction_ratio": float(1.0 - after_parameters / max(1, before_parameters)),
        "baseline_ppl": baseline_ppl,
        "pruned_ppl": after_ppl,
        "baseline_scenario_metrics": before_scenarios,
        "pruned_scenario_metrics": after_scenarios,
        "relative_ppl_increase": (
            None if baseline_ppl is None or after_ppl is None else float(after_ppl / baseline_ppl - 1.0)
        ),
        "pruning": _jsonable(pruning_result),
    }
    (output_dir / "run_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "model", "method", "mask_style", "seed", "seqlen",
        "requested_prune_ratio", "parameters_before", "parameters_after",
        "physical_parameter_reduction_ratio", "baseline_ppl", "pruned_ppl",
        "relative_ppl_increase",
    ]
    with (output_dir / "run_result.tsv").open("w", encoding="utf-8") as handle:
        handle.write("\t".join(fields) + "\n")
        handle.write("\t".join(str(result[key]) for key in fields) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
