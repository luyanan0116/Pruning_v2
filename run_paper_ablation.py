from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


PAPER_METHODS = ("paper_mi", "paper_mi_gb", "paper_mi_gb_lcb", "paper_full")
BASELINES = ("dense", "wanda")


def _stream(command: list[str], log_path: Path, env: dict) -> None:
    print("\n$ " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run V8.2 Dense/Wanda baselines plus paper_mi -> MI+GB -> MI+GB+LCB -> Full "
            "on fresh model copies. Tune on WikiText validation; reserve test for final reporting."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache_dir", default="llm_weights")
    parser.add_argument("--output_dir", default="results/paper_v82_weight50")
    parser.add_argument("--response_cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparsity_ratio", type=float, default=0.50)
    parser.add_argument("--wikitext2_path", default=None)
    parser.add_argument("--c4_path", default=None)
    parser.add_argument("--eval_wikitext_split", choices=["validation", "test", "both"], default="validation")
    parser.add_argument("--include_baselines", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite_scores", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    args, passthrough = parser.parse_known_args()

    if abs(args.sparsity_ratio - 0.50) > 1e-12:
        parser.error("V8.2 ablation fixes sparsity_ratio to 0.50")

    root = Path(args.output_dir)
    logs_dir = root / "logs"
    console_dir = root / "console"
    report_dir = root / "shared_score_report"
    local_cache = root / "response_cache"
    response_cache = Path(args.response_cache_dir) if args.response_cache_dir else local_cache
    for directory in (logs_dir, console_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if args.response_cache_dir is None:
        local_cache.mkdir(parents=True, exist_ok=True)
    if args.overwrite_scores and report_dir.exists():
        shutil.rmtree(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.wikitext2_path:
        env["WIKITEXT2_PATH"] = args.wikitext2_path
    if args.c4_path:
        env["C4_PATH"] = args.c4_path

    methods = ((*BASELINES,) if args.include_baselines else tuple()) + PAPER_METHODS
    main_py = Path(__file__).resolve().parent / "main.py"
    first_paper = True
    for method in methods:
        command = [
            sys.executable, str(main_py),
            "--model", args.model,
            "--cache_dir", args.cache_dir,
            "--seed", str(args.seed),
            "--sparsity_ratio", "0.50",
            "--prune_method", method,
            "--eval_wikitext_split", args.eval_wikitext_split,
            "--paper_cache_dir", str(response_cache),
            "--paper_report_dir", str(report_dir),
            "--save", str(logs_dir),
        ]
        if args.c4_path:
            command.extend(["--c4_path", args.c4_path])
        if args.wikitext2_path:
            command.extend(["--wikitext2_path", args.wikitext2_path])
        if method in PAPER_METHODS and first_paper:
            if args.overwrite_cache:
                command.append("--paper_overwrite_cache")
            if args.overwrite_scores:
                command.append("--paper_overwrite_scores")
            first_paper = False
        command.extend(passthrough)
        _stream(command, console_dir / f"{method}.txt", env)

    rows = []
    for method in methods:
        log_path = logs_dir / f"log_{method}.txt"
        with log_path.open("r", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle, delimiter="\t"))
        rows.append({
            "method": method,
            "actual_weight_sparsity": float(row["actual_sparsity"]),
            "ppl_validation": float(row["ppl_validation"]),
            "ppl_test": float(row["ppl_test"]),
            "seqlen": int(row["seqlen"]),
        })

    summary_path = root / "ablation_ppl_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "method", "actual_weight_sparsity", "ppl_validation", "ppl_test", "seqlen"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print("\nAblation summary")
    for row in rows:
        print(
            f"  {row['method']:<24} val={row['ppl_validation']:.6f}, "
            f"test={row['ppl_test']:.6f}, sparsity={row['actual_weight_sparsity']:.9f}, "
            f"seqlen={row['seqlen']}"
        )
    print(f"saved: {summary_path}")

    # Quota overlap is meaningful only among the four paper methods.
    budgets = {}
    costs = None
    for method in PAPER_METHODS:
        path = report_dir / f"weight_budget_{method}_counts.npz"
        with np.load(path) as archive:
            budgets[method] = np.asarray(archive["prune_counts"], dtype=np.int64)
            current_costs = np.asarray(archive["costs"], dtype=np.int64)
        if costs is None:
            costs = current_costs
        elif not np.array_equal(costs, current_costs):
            raise RuntimeError("weight-budget layouts differ across ablations")

    overlap_rows = []
    for left_pos, left in enumerate(PAPER_METHODS):
        for right in PAPER_METHODS[left_pos + 1:]:
            intersection = int(np.minimum(budgets[left], budgets[right]).sum())
            union = int(np.maximum(budgets[left], budgets[right]).sum())
            overlap_rows.append({
                "left": left,
                "right": right,
                "intersection_quota_weights": intersection,
                "union_quota_weights": union,
                "jaccard": intersection / float(max(1, union)),
                "symmetric_difference_weights": union - intersection,
            })
    overlap_path = report_dir / "weight_quota_overlap.csv"
    with overlap_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "left", "right", "intersection_quota_weights", "union_quota_weights",
            "jaccard", "symmetric_difference_weights",
        ])
        writer.writeheader()
        writer.writerows(overlap_rows)
    print(f"saved: {overlap_path}")


if __name__ == "__main__":
    main()
