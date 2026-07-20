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
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"])
    
    parser.add_argument(
        "--prune_method",
        type=str,
        choices=[
            "magnitude", "wanda", "sparsegpt",
            "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter",
            "search", "lcb", "paper_mi", "paper_mi_gb", "paper_mi_gb_lcb",
        ],
    )

    # Paper-aligned ablation: Eq. (1)-(10) in the uploaded proposal.
    parser.add_argument("--lcb_lambda", type=float, default=1.0, help="LCB risk coefficient lambda")
    parser.add_argument("--n_samples_lcb", type=int, default=10, help="number of repeated LCB estimates")
    parser.add_argument("--prune_per_layer", type=int, default=0,
                        help="exact MLP channels removed per layer; 0 uses sparsity_ratio")
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
                        help="quantile bins of per-sequence causal-LM loss used as task event Y")
    parser.add_argument("--paper_num_bins", type=int, default=16,
                        help="number of fine DCT frequency bins")
    parser.add_argument("--paper_num_bands", type=int, default=4,
                        help="target bands after task-driven adjacent merging")
    parser.add_argument("--paper_mi_neighbors", type=int, default=3)
    parser.add_argument("--paper_purity_thresholds", type=str, default="0.65,0.75,0.85")
    parser.add_argument("--paper_min_ball_size", type=int, default=8)
    parser.add_argument("--paper_max_balls", type=int, default=64)
    parser.add_argument("--paper_min_purity_gain", type=float, default=0.01)
    parser.add_argument("--paper_min_radius_reduction", type=float, default=0.05)
    parser.add_argument("--paper_compactness_ratio", type=float, default=0.55)
    parser.add_argument("--paper_min_event_classes", type=int, default=2)
    parser.add_argument("--paper_sample_fraction", type=float, default=0.8)
    parser.add_argument("--paper_cache_dir", type=str, default="paper_response_cache")
    parser.add_argument("--paper_report_dir", type=str, default="paper_ablation_report")
    parser.add_argument("--paper_plot_layers", type=str, default="first,middle,last")
    parser.add_argument("--paper_overwrite_cache", action="store_true")
    parser.add_argument("--paper_overwrite_scores", action="store_true")
    
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()
    if args.sparsity_ratio != 0 and args.prune_method is None:
        parser.error("--prune_method is required when --sparsity_ratio is non-zero")
    if args.prune_method in PAPER_METHODS and args.prune_per_layer <= 0 and not 0 < args.sparsity_ratio < 1:
        parser.error("paper methods require --prune_per_layer > 0 or --sparsity_ratio in (0,1)")

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
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
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