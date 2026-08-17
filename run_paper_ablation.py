from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


METHODS = ("paper_mi", "paper_mi_gb", "paper_mi_gb_lcb", "paper_full")


def _stream_command(command, log_path: Path, env: dict) -> None:
    print("\n$ " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run clean MI -> GB -> LCB -> Full ablation on fresh model copies."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache_dir", default="llm_weights")
    parser.add_argument("--output_dir", default="paper_ablation_runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparsity_ratio", type=float, default=0.15)
    parser.add_argument("--prune_per_layer", type=int, default=0)
    parser.add_argument("--wikitext2_path", default=None)
    parser.add_argument("--c4_path", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args, passthrough = parser.parse_known_args()

    if args.prune_per_layer <= 0 and not 0 < args.sparsity_ratio < 1:
        parser.error("use a positive --prune_per_layer or a --sparsity_ratio in (0,1)")

    root = Path(args.output_dir)
    logs_dir = root / "logs"
    console_dir = root / "console"
    response_cache = root / "response_cache"
    report_dir = root / "shared_score_report"
    for directory in (logs_dir, console_dir, response_cache, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.wikitext2_path:
        env["WIKITEXT2_PATH"] = args.wikitext2_path
    if args.c4_path:
        env["C4_PATH"] = args.c4_path

    main_py = Path(__file__).resolve().parent / "main.py"
    for index, method in enumerate(METHODS):
        command = [
            sys.executable,
            str(main_py),
            "--model", args.model,
            "--cache_dir", args.cache_dir,
            "--seed", str(args.seed),
            "--sparsity_type", "unstructured",
            "--sparsity_ratio", str(args.sparsity_ratio),
            "--prune_per_layer", str(args.prune_per_layer),
            "--prune_method", method,
            "--paper_cache_dir", str(response_cache),
            "--paper_report_dir", str(report_dir),
            "--save", str(logs_dir),
        ]
        if args.c4_path:
            command.extend(["--c4_path", args.c4_path])
        if args.wikitext2_path:
            command.extend(["--wikitext2_path", args.wikitext2_path])
        if args.overwrite and index == 0:
            command.extend(["--paper_overwrite_cache", "--paper_overwrite_scores"])
        command.extend(passthrough)
        _stream_command(command, console_dir / f"{method}.txt", env)

    rows = []
    for method in METHODS:
        log_path = logs_dir / f"log_{method}.txt"
        with log_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            row = next(reader)
        rows.append(
            {
                "method": method,
                "actual_weight_sparsity": float(row["actual_sparsity"]),
                "ppl": float(row["ppl_test"]),
            }
        )

    summary_path = root / "ablation_ppl_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "actual_weight_sparsity", "ppl"])
        writer.writeheader()
        writer.writerows(rows)

    ppls = [row["ppl"] for row in rows]
    print("\nAblation summary")
    for row in rows:
        print(f"  {row['method']:<24} PPL={row['ppl']:.8f}, weight sparsity={row['actual_weight_sparsity']:.8f}")
    print(f"saved: {summary_path}")
    if len(set(round(value, 8) for value in ppls)) == 1:
        print("WARNING: all PPL values are still identical; inspect shared_score_report/mask_overlap.csv.")
    if not all(left > right for left, right in zip(ppls, ppls[1:])):
        print("NOTE: PPL is not strictly descending. GB/LCB primarily target stability; the code reports real measurements and does not force a curve.")


if __name__ == "__main__":
    main()
