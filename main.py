# main.py
import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from lib.prune_paper import PAPER_METHODS, prune_paper
from lib.eval import eval_ppl, eval_zero_shot

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--seqlen', type=int, default=2048, help='evaluation/context length; Wanda reference uses 2048')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"])
    
    parser.add_argument(
        "--prune_method",
        type=str,
        choices=[
            "magnitude", "wanda", "sparsegpt",
            "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter",
            "search", "lcb", "paper_mi", "paper_mi_gb", "paper_mi_gb_lcb", "paper_full",
        ],
    )

    # Paper-aligned ablation: Eq. (1)-(10) in the uploaded proposal.
    parser.add_argument("--lcb_lambda", type=float, default=1.0, help="LCB risk coefficient lambda")
    parser.add_argument("--paper_lcb_repeats", type=int, default=10,
                        help="repeated sample/scenario bootstrap estimates used by the real LCB")
    parser.add_argument("--paper_lcb_sample_fraction", type=float, default=0.8,
                        help="fraction of base calibration samples drawn per LCB repeat")
    parser.add_argument("--paper_lcb_scenario_fraction", type=float, default=0.67,
                        help="fraction of scenario realizations drawn per selected base sample")
    parser.add_argument("--prune_per_layer", type=int, default=0,
                        help="exact MLP channels removed per layer; 0 uses mlp_sparsity_ratio/sparsity_ratio")
    parser.add_argument("--attention_prune_per_layer", type=int, default=0,
                        help="exact attention heads removed per layer; 0 uses attention_sparsity_ratio/sparsity_ratio")
    parser.add_argument("--mlp_sparsity_ratio", type=float, default=None,
                        help="structured MLP-channel sparsity; default uses sparsity_ratio")
    parser.add_argument("--attention_sparsity_ratio", type=float, default=None,
                        help="structured attention-head sparsity; default uses sparsity_ratio")
    parser.add_argument("--paper_prune_targets", type=str, default="mlp,attention",
                        help="comma-separated targets: mlp,attention")
    parser.add_argument(
        "--paper_mask_style",
        type=str,
        default="structured_unit",
        choices=["wanda_weight", "unit_budget_weight", "structured_unit"],
        help=(
            "structured_unit removes complete MLP channels/heads; "
            "wanda_weight is the legacy weak-guidance weight baseline; "
            "unit_budget_weight uses unit scores to assign different per-unit weight sparsities"
        ),
    )
    parser.add_argument("--paper_wanda_score_floor", type=float, default=0.05,
                        help="minimum rank factor used to protect high/low contribution scales")
    parser.add_argument("--paper_wanda_row_spread", type=float, default=0.01,
                        help="how strongly paper scores redistribute row sparsity around the target")
    parser.add_argument("--paper_wanda_temperature", type=float, default=1.0,
                        help="rank-allocation temperature for Wanda-style row budgets")
    parser.add_argument("--paper_wanda_chunk_rows", type=int, default=256,
                        help="row chunk size used while building per-weight masks")
    parser.add_argument("--paper_wanda_guidance_strength", type=float, default=0.01,
                        help="gentle near-one MI/GB/LCB multiplier; 0 reproduces Wanda allocation")
    parser.add_argument("--paper_unit_min_sparsity", type=float, default=0.30,
                        help="minimum per-unit weight sparsity for unit_budget_weight; high-score units approach this")
    parser.add_argument("--paper_unit_max_sparsity", type=float, default=0.70,
                        help="maximum per-unit weight sparsity for unit_budget_weight; low-score units approach this")
    parser.add_argument("--paper_wanda_nsamples", type=int, default=128,
                        help="sequential Wanda calibration sample count")
    parser.add_argument("--paper_wanda_seqlen", type=int, default=2048,
                        help="sequential Wanda calibration sequence length")
    parser.add_argument("--paper_wanda_sequential", action=argparse.BooleanOptionalAction, default=True,
                        help="recompute activation scales layer by layer after prior layers are pruned")
    parser.add_argument("--paper_prune_step", type=int, default=10,
                        help="export selected channel indices in batches of this size")
    parser.add_argument("--paper_score_nsamples", type=int, default=32,
                        help="base calibration samples for gradient-response scoring")
    parser.add_argument("--paper_calib_seqlen", type=int, default=512)
    parser.add_argument("--paper_response_length", type=int, default=32,
                        help="sequence positions retained after adaptive response pooling")
    parser.add_argument("--paper_calib_dataset", type=str, default="c4", choices=["c4", "wikitext2"])
    parser.add_argument("--paper_scenario_ratios", type=str, default="0.5,0.75,1.0",
                        help="context-length scenarios used for double-source uncertainty")
    parser.add_argument("--paper_event_bins", type=int, default=3,
                        help="quantile bins of position-derived next-token NLL used as unified task event Y")
    parser.add_argument("--paper_num_bins", type=int, default=16,
                        help="number of fine DCT frequency bins")
    parser.add_argument("--paper_num_bands", type=int, default=4,
                        help="target bands after task-driven adjacent merging")
    parser.add_argument("--paper_mi_neighbors", type=int, default=3)
    parser.add_argument("--paper_fast_small_mi", action=argparse.BooleanOptionalAction, default=True,
                        help="use an exact Numba-compiled kNN path for the many small ball-level MI problems")
    parser.add_argument("--paper_probe_units", type=int, default=64,
                        help="representative units for task-driven adjacent frequency-bin merging")
    parser.add_argument("--paper_kde_bandwidth_scale", type=float, default=1.0)
    parser.add_argument("--paper_purity_thresholds", type=str, default="0.65,0.75,0.85")
    parser.add_argument("--paper_min_ball_size", type=int, default=8)
    parser.add_argument("--paper_max_balls", type=int, default=64)
    parser.add_argument("--paper_max_ball_depth", type=int, default=5)
    parser.add_argument("--paper_gb_localization", type=str, default="unit_local",
                        choices=["unit_local", "layer_shared"],
                        help="unit_local follows the paper literally; layer_shared is a faster diagnostic approximation")
    parser.add_argument("--paper_gb_workers", type=int, default=16,
                        help="total CPU worker budget for strict unit-local granular balls")
    parser.add_argument("--paper_gb_chunk_size", type=int, default=64,
                        help="units handled per worker task; reduces thread scheduling overhead")
    parser.add_argument("--paper_kde_scope", type=str, default="probe",
                        choices=["all", "probe", "none"],
                        help="auxiliary KDE diagnostics: all units, probe units only, or disabled; never changes kNN pruning scores")
    parser.add_argument("--paper_min_purity_gain", type=float, default=0.0)
    parser.add_argument("--paper_min_radius_reduction", type=float, default=0.0)
    parser.add_argument("--paper_compactness_ratio", type=float, default=0.55)
    parser.add_argument("--paper_min_event_classes", type=int, default=2)
    parser.add_argument("--paper_min_event_count_per_ball", type=int, default=2,
                        help="minimum observations per represented event class in every accepted child ball")
    parser.add_argument("--paper_gb_fusion_mode", type=str, default="inverse_sqrt_dispersion",
                        choices=["equal", "inverse_sqrt_dispersion", "inverse_dispersion"],
                        help="multi-granularity fusion rule; inverse-sqrt is the robust default")
    parser.add_argument("--paper_gb_fusion_max_ratio", type=float, default=5.0,
                        help="maximum raw fusion-weight ratio between granularities")
    parser.add_argument("--paper_band_coverage_ratio", type=float, default=0.90,
                        help="uniform fallback gamma when per-band ratios are not supplied")
    parser.add_argument("--paper_band_coverage_ratios", type=str, default=None,
                        help="per-band gamma_b targets, e.g. 0.85,0.70,0.55 for low/mid/high")
    parser.add_argument("--paper_band_selection_weights", type=str, default=None,
                        help="band-only ranking weights, e.g. 1.0,0.7,0.4 for low/mid/high")
    parser.add_argument(
        "--paper_final_keep_strategy",
        type=str,
        default="paper_hybrid",
        choices=["paper_hybrid", "lcb_only", "band_only"],
        help=(
            "applies to paper_full only: paper_hybrid uses LCB plus frequency under-coverage; "
            "lcb_only disables coverage; band_only is a diagnostic frequency-only selector. "
            "paper_mi, paper_mi_gb and paper_mi_gb_lcb are always clean score-only ablations"
        ),
    )
    parser.add_argument("--paper_coverage_alpha", type=float, default=0.25)
    parser.add_argument("--paper_global_budget", action=argparse.BooleanOptionalAction, default=True,
                        help="allocate each unit type's total pruning budget jointly across all layers")
    parser.add_argument("--paper_global_layer_spread", type=float, default=0.10,
                        help="for weight-mask modes, max layer sparsity deviation around the global target")
    parser.add_argument("--paper_greedy_batches", type=int, default=64)
    parser.add_argument("--paper_cache_dir", type=str, default="paper_response_cache")
    parser.add_argument("--paper_report_dir", type=str, default="paper_ablation_report")
    parser.add_argument("--paper_plot_layers", type=str, default="first,middle,last")
    parser.add_argument("--paper_overwrite_cache", action="store_true")
    parser.add_argument("--paper_overwrite_scores", action="store_true")
    
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument("--c4_path", type=str, default=None, help="local C4 path for this run")
    parser.add_argument("--wikitext2_path", type=str, default=None, help="local WikiText-2 raw path for this run")
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()
    # Run-local paths: accepted as ordinary CLI parameters and never persisted
    # in shell/profile configuration. lib.data reads them from this process only.
    if args.c4_path:
        os.environ["C4_PATH"] = args.c4_path
    if args.wikitext2_path:
        os.environ["WIKITEXT2_PATH"] = args.wikitext2_path
    if args.sparsity_ratio != 0 and args.prune_method is None:
        parser.error("--prune_method is required when --sparsity_ratio is non-zero")
    if args.prune_method in PAPER_METHODS:
        targets = {part.strip().lower() for part in args.paper_prune_targets.split(",") if part.strip()}
        aliases = {"attn": "attention", "head": "attention", "heads": "attention", "ffn": "mlp"}
        targets = {aliases.get(value, value) for value in targets}
        if not targets or not targets.issubset({"mlp", "attention"}):
            parser.error("--paper_prune_targets must contain mlp and/or attention")
        mlp_ratio = args.sparsity_ratio if args.mlp_sparsity_ratio is None else args.mlp_sparsity_ratio
        attn_ratio = args.sparsity_ratio if args.attention_sparsity_ratio is None else args.attention_sparsity_ratio
        if args.paper_band_coverage_ratios is not None:
            try:
                coverage_values = [float(v.strip()) for v in args.paper_band_coverage_ratios.split(",") if v.strip()]
            except ValueError as exc:
                parser.error(f"invalid --paper_band_coverage_ratios: {exc}")
            if len(coverage_values) != args.paper_num_bands:
                parser.error("--paper_band_coverage_ratios length must equal --paper_num_bands")
            if any(not 0 < value <= 1 for value in coverage_values):
                parser.error("each --paper_band_coverage_ratios value must be in (0,1]")
        if args.paper_band_selection_weights is not None:
            try:
                band_weights = [float(v.strip()) for v in args.paper_band_selection_weights.split(",") if v.strip()]
            except ValueError as exc:
                parser.error(f"invalid --paper_band_selection_weights: {exc}")
            if len(band_weights) != args.paper_num_bands:
                parser.error("--paper_band_selection_weights length must equal --paper_num_bands")
            if any(value < 0 for value in band_weights) or not any(value > 0 for value in band_weights):
                parser.error("band selection weights must be non-negative and not all zero")
        if not 0 <= args.paper_global_layer_spread < 1:
            parser.error("--paper_global_layer_spread must be in [0,1)")
        if args.paper_lcb_repeats < 1:
            parser.error("--paper_lcb_repeats must be >= 1")
        if not 0 < args.paper_lcb_sample_fraction <= 1:
            parser.error("--paper_lcb_sample_fraction must be in (0,1]")
        if not 0 < args.paper_lcb_scenario_fraction <= 1:
            parser.error("--paper_lcb_scenario_fraction must be in (0,1]")
        if args.paper_min_event_count_per_ball < 1:
            parser.error("--paper_min_event_count_per_ball must be >= 1")
        if args.paper_gb_fusion_max_ratio < 1:
            parser.error("--paper_gb_fusion_max_ratio must be >= 1")
        if args.paper_mask_style in {"wanda_weight", "unit_budget_weight"}:
            if args.prune_per_layer > 0 or args.attention_prune_per_layer > 0:
                parser.error(
                    "--prune_per_layer and --attention_prune_per_layer are only valid with "
                    "--paper_mask_style structured_unit"
                )
            if "mlp" in targets and not 0 < mlp_ratio < 1:
                parser.error("Wanda-style MLP weight pruning requires a ratio in (0,1)")
            if "attention" in targets and not 0 < attn_ratio < 1:
                parser.error("Wanda-style attention weight pruning requires a ratio in (0,1)")
            if not 0 < args.paper_wanda_score_floor <= 1:
                parser.error("--paper_wanda_score_floor must be in (0,1]")
            if not 0 <= args.paper_wanda_row_spread < 1:
                parser.error("--paper_wanda_row_spread must be in [0,1)")
            if args.paper_wanda_temperature <= 0 or args.paper_wanda_chunk_rows <= 0:
                parser.error("Wanda temperature and chunk rows must be positive")
            if args.paper_wanda_guidance_strength < 0:
                parser.error("--paper_wanda_guidance_strength must be non-negative")
            if args.paper_mask_style == "unit_budget_weight":
                if not 0 <= args.paper_unit_min_sparsity < args.paper_unit_max_sparsity < 1:
                    parser.error("--paper_unit_min_sparsity/--paper_unit_max_sparsity must satisfy 0 <= min < max < 1")
                for label, value in (("MLP", mlp_ratio), ("attention", attn_ratio)):
                    if label.lower() == "mlp" and "mlp" not in targets:
                        continue
                    if label == "attention" and "attention" not in targets:
                        continue
                    if not args.paper_unit_min_sparsity <= value <= args.paper_unit_max_sparsity:
                        parser.error(
                            f"{label} target sparsity {value} must lie within "
                            "--paper_unit_min_sparsity/--paper_unit_max_sparsity"
                        )
            if args.paper_wanda_nsamples < 4 or args.paper_wanda_seqlen < 8:
                parser.error("sequential weight calibration settings are too small")
        else:
            if "mlp" in targets and args.prune_per_layer <= 0 and not 0 < mlp_ratio < 1:
                parser.error("MLP pruning requires --prune_per_layer > 0 or a ratio in (0,1)")
            if "attention" in targets and args.attention_prune_per_layer <= 0 and not 0 < attn_ratio < 1:
                parser.error("attention pruning requires --attention_prune_per_layer > 0 or a ratio in (0,1)")

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.seqlen = min(int(args.seqlen), int(model.config.max_position_embeddings))
    if args.paper_wanda_seqlen > model.seqlen:
        print(f"clamping paper_wanda_seqlen {args.paper_wanda_seqlen} to model seqlen {model.seqlen}")
        args.paper_wanda_seqlen = model.seqlen
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if "30b" in args.model or "65b" in args.model:
        mapped = model.hf_device_map["lm_head"]
        if isinstance(mapped, int):
            device = torch.device(f"cuda:{mapped}")
        elif mapped in {"cpu", "disk"}:
            device = torch.device("cpu")
        else:
            device = torch.device(mapped)
    print("use device ", device)

    should_prune = args.sparsity_ratio != 0 or args.prune_method in PAPER_METHODS
    if should_prune:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method in PAPER_METHODS:
            prune_paper(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if args.save:
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
        with open(save_filepath, "w") as f:
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            print(f"{args.prune_method}\t{sparsity_ratio:.8f}\t{ppl_test:.8f}", file=f, flush=True)

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()