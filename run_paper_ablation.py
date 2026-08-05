from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path


METHODS = ("paper_mi", "paper_mi_gb", "paper_mi_gb_lcb")
METHOD_LABELS = {
    "paper_mi": "MI",
    "paper_mi_gb": "MI+GB",
    "paper_mi_gb_lcb": "MI+GB+LCB",
}
EXPECTED_STRICT_STAGES = {
    "paper_mi": {"mutual_information"},
    "paper_mi_gb": {
        "mutual_information",
        "granular_ball",
        "multi_granularity_fusion",
    },
    "paper_mi_gb_lcb": {
        "mutual_information",
        "granular_ball",
        "multi_granularity_fusion",
        "dual_source_repeated_estimation",
        "lcb",
    },
}


def _run(command: list[str], log_path: Path, env: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command), flush=True)
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
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _validate_stage_manifest(path: Path, method: str, strict: bool) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    layers = payload.get("layers", [])
    if not layers:
        raise RuntimeError(f"empty stage manifest: {path}")
    stage_sets = {tuple(item["executed_stages"]) for item in layers}
    if len(stage_sets) != 1:
        raise RuntimeError(f"inconsistent stages inside {path}: {stage_sets}")
    stages = list(next(iter(stage_sets)))
    if strict:
        actual = set(stages)
        expected = EXPECTED_STRICT_STAGES[method]
        if method == "paper_mi_gb_lcb":
            # The fusion-mode marker is intentionally additional.
            fusion_markers = {
                "equal_granularity_fusion",
                "manual_granularity_fusion",
                "repeat_variance_granularity_fusion",
            }
            actual = actual - fusion_markers
        if actual != expected:
            raise RuntimeError(
                f"strict ablation contamination for {method}: expected {sorted(expected)}, "
                f"got {sorted(actual)}"
            )
    return stages


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strict incremental PPL ablation: MI; MI+granular ball; "
            "MI+granular ball+LCB. All runs share responses and evaluation settings."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache_dir", default="llm_weights")
    parser.add_argument("--output_dir", default="paper_ablation_strict")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prune_ratio", type=float, default=0.15)
    parser.add_argument(
        "--mask_style",
        choices=["structured_zero", "structured_surgery"],
        default="structured_zero",
    )
    parser.add_argument("--c4_path", default=None)
    parser.add_argument("--wikitext2_path", default=None)
    parser.add_argument("--scenario_manifest", default=None)
    parser.add_argument(
        "--paper_granularity_weight_mode",
        choices=["equal", "manual", "repeat_variance"],
        default="equal",
        help=(
            "equal is recommended for strict component ablation. repeat_variance "
            "makes MI+GB run repeated estimation to adapt fusion weights."
        ),
    )
    parser.add_argument("--paper_granularity_weights", default=None)
    parser.add_argument("--baseline_tolerance", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    args, passthrough = parser.parse_known_args()

    if args.paper_granularity_weight_mode == "manual" and not args.paper_granularity_weights:
        parser.error("manual fusion requires --paper_granularity_weights")
    strict = args.paper_granularity_weight_mode in {"equal", "manual"}

    root = Path(args.output_dir)
    package_root = Path(__file__).resolve().parent
    main_py = package_root / "main.py"
    env = os.environ.copy()
    if args.c4_path:
        env["C4_PATH"] = args.c4_path
    if args.wikitext2_path:
        env["WIKITEXT2_PATH"] = args.wikitext2_path

    rows = []
    for index, method in enumerate(METHODS):
        run_dir = root / method
        report_dir = run_dir / "score_report"
        result_path = run_dir / "run_result.json"
        if not result_path.exists() or args.overwrite:
            command = [
                sys.executable,
                str(main_py),
                "--model", args.model,
                "--cache_dir", args.cache_dir,
                "--seed", str(args.seed),
                "--prune_method", method,
                "--prune_ratio", str(args.prune_ratio),
                "--paper_mask_style", args.mask_style,
                "--paper_cache_dir", str(root / "shared_response_cache"),
                "--paper_report_dir", str(report_dir),
                "--paper_granularity_weight_mode", args.paper_granularity_weight_mode,
                "--eval_before",
                "--eval_after",
                "--no-paper_post_prune_validate",
                "--output_dir", str(run_dir),
            ]
            if args.paper_granularity_weights:
                command += ["--paper_granularity_weights", args.paper_granularity_weights]
            if args.scenario_manifest:
                command += ["--paper_scenario_manifest", args.scenario_manifest]
            if args.c4_path:
                command += ["--c4_path", args.c4_path]
            if args.wikitext2_path:
                command += ["--wikitext2_path", args.wikitext2_path]
            if args.overwrite:
                command += ["--paper_overwrite_scores"]
                if index == 0:
                    command += ["--paper_overwrite_cache"]
            command += passthrough
            _run(command, run_dir / "console.txt", env)

        result = json.loads(result_path.read_text(encoding="utf-8"))
        stages = _validate_stage_manifest(
            report_dir / "executed_stage_manifest.json",
            method,
            strict=strict,
        )
        baseline = float(result["baseline_ppl"])
        pruned = float(result["pruned_ppl"])
        rows.append(
            {
                "method": method,
                "components": METHOD_LABELS[method],
                "executed_stages": "+".join(stages),
                "baseline_ppl": baseline,
                "pruned_ppl": pruned,
                "absolute_ppl_increase": pruned - baseline,
                "relative_ppl_increase": float(result["relative_ppl_increase"]),
            }
        )

    baselines = [row["baseline_ppl"] for row in rows]
    baseline_spread = max(baselines) - min(baselines)
    if not math.isfinite(baseline_spread) or baseline_spread > args.baseline_tolerance:
        raise RuntimeError(
            f"baseline PPL mismatch across fresh-model runs: spread={baseline_spread}; "
            "the three runs are not directly comparable"
        )

    mi_ppl = rows[0]["pruned_ppl"]
    previous = None
    for row in rows:
        row["ppl_delta_vs_mi"] = row["pruned_ppl"] - mi_ppl
        row["ppl_delta_vs_previous_stage"] = (
            0.0 if previous is None else row["pruned_ppl"] - previous
        )
        previous = row["pruned_ppl"]

    summary = root / "ablation_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    protocol = {
        "protocol": "strict_incremental" if strict else "adaptive_fusion",
        "model": args.model,
        "seed": args.seed,
        "prune_ratio": args.prune_ratio,
        "mask_style": args.mask_style,
        "granularity_weight_mode": args.paper_granularity_weight_mode,
        "granularity_weights": args.paper_granularity_weights,
        "shared_response_cache": str(root / "shared_response_cache"),
        "baseline_ppl_spread": baseline_spread,
        "methods": rows,
    }
    (root / "ablation_summary.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
