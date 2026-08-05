from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def _floats(raw: str) -> list[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def _ints(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


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


def _scenario_worst_relative(result: dict) -> float | None:
    before = result.get("baseline_scenario_metrics")
    after = result.get("pruned_scenario_metrics")
    if not before or not after:
        return None
    changes = []
    for key, left in before["per_scenario"].items():
        if key in after["per_scenario"]:
            changes.append(after["per_scenario"][key]["ppl"] / left["ppl"] - 1.0)
    return max(changes) if changes else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep pruning ratios, random seeds and context lengths; report the "
            "maximum stable ratio and the adjacent critical interval."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", default="paper_validation")
    parser.add_argument("--method", choices=["paper_mi", "paper_mi_gb", "paper_mi_gb_lcb"], default="paper_mi_gb_lcb")
    parser.add_argument("--mask_style", choices=["structured_zero", "structured_surgery"], default="structured_zero")
    parser.add_argument("--prune_ratios", default="0.05,0.10,0.15,0.20")
    parser.add_argument("--seqlens", default="512,1024,2048")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--relative_ppl_limit", type=float, default=0.05)
    parser.add_argument("--cache_dir", default="llm_weights")
    parser.add_argument("--c4_path", default=None)
    parser.add_argument("--wikitext2_path", default=None)
    parser.add_argument("--scenario_manifest", default=None, help="paired sample/scenario calibration manifest")
    parser.add_argument("--eval_scenario_manifest", default=None)
    parser.add_argument("--overwrite_scores", action="store_true")
    args, score_passthrough = parser.parse_known_args()

    ratios = sorted(_floats(args.prune_ratios))
    seqlens = _ints(args.seqlens)
    seeds = _ints(args.seeds)
    root = Path(args.output_dir)
    package_root = Path(__file__).resolve().parent
    main_py = package_root / "main.py"
    config_py = package_root / "generate_global_config.py"
    env = os.environ.copy()
    if args.c4_path:
        env["C4_PATH"] = args.c4_path
    if args.wikitext2_path:
        env["WIKITEXT2_PATH"] = args.wikitext2_path

    rows = []
    for seed in seeds:
        score_root = root / f"seed_{seed}" / "scores"
        score_csv = score_root / "report" / "all_contribution_scores.csv"
        if not score_csv.exists() or args.overwrite_scores:
            command = [
                sys.executable,
                str(main_py),
                "--model", args.model,
                "--cache_dir", args.cache_dir,
                "--seed", str(seed),
                "--prune_method", args.method,
                "--prune_ratio", "0",
                "--paper_score_only",
                "--paper_cache_dir", str(score_root / "response_cache"),
                "--paper_report_dir", str(score_root / "report"),
                "--output_dir", str(score_root / "run"),
                "--no-eval_after",
            ]
            if args.scenario_manifest:
                command += ["--paper_scenario_manifest", args.scenario_manifest]
            if args.c4_path:
                command += ["--c4_path", args.c4_path]
            if args.wikitext2_path:
                command += ["--wikitext2_path", args.wikitext2_path]
            if args.overwrite_scores:
                command += ["--paper_overwrite_cache", "--paper_overwrite_scores"]
            command += score_passthrough
            _run(command, score_root / "score_console.txt", env)

        for seqlen in seqlens:
            for ratio in ratios:
                run_root = root / f"seed_{seed}" / f"seqlen_{seqlen}" / f"ratio_{ratio:.4f}"
                selection_path = run_root / "selection.json"
                if not selection_path.exists():
                    command = [
                        sys.executable,
                        str(config_py),
                        "--report_dir", str(score_root / "report"),
                        "--output", str(selection_path),
                        "--model", args.model,
                        "--cache_dir", args.cache_dir,
                        "--method", args.method,
                        "--prune_ratio", str(ratio),
                        "--seqlen", str(seqlen),
                    ]
                    _run(command, run_root / "selection_console.txt", env)

                result_path = run_root / "run" / "run_result.json"
                if not result_path.exists():
                    command = [
                        sys.executable,
                        str(main_py),
                        "--model", args.model,
                        "--cache_dir", args.cache_dir,
                        "--seed", str(seed),
                        "--seqlen", str(seqlen),
                        "--prune_method", args.method,
                        "--prune_ratio", str(ratio),
                        "--paper_mask_style", args.mask_style,
                        "--paper_selection_file", str(selection_path),
                        "--paper_report_dir", str(run_root / "application_report"),
                        "--eval_before",
                        "--eval_after",
                        "--no-paper_post_prune_validate",
                        "--output_dir", str(run_root / "run"),
                    ]
                    if args.eval_scenario_manifest:
                        command += ["--eval_scenario_manifest", args.eval_scenario_manifest]
                    if args.wikitext2_path:
                        command += ["--wikitext2_path", args.wikitext2_path]
                    _run(command, run_root / "run_console.txt", env)

                result = json.loads(result_path.read_text(encoding="utf-8"))
                scenario_worst = _scenario_worst_relative(result)
                ppl_change = result["relative_ppl_increase"]
                combined_worst = max(
                    value for value in (ppl_change, scenario_worst) if value is not None
                )
                rows.append(
                    {
                        "seed": seed,
                        "seqlen": seqlen,
                        "prune_ratio": ratio,
                        "baseline_ppl": result["baseline_ppl"],
                        "pruned_ppl": result["pruned_ppl"],
                        "relative_ppl_increase": ppl_change,
                        "worst_scenario_relative_ppl_increase": scenario_worst,
                        "combined_worst_relative_increase": combined_worst,
                        "stable": combined_worst <= args.relative_ppl_limit,
                    }
                )

    curve_path = root / "validation_curve.csv"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    with curve_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ratio_summary = []
    for ratio in ratios:
        subset = [row for row in rows if row["prune_ratio"] == ratio]
        worst = max(float(row["combined_worst_relative_increase"]) for row in subset)
        ratio_summary.append(
            {
                "prune_ratio": ratio,
                "worst_relative_increase": worst,
                "stable": worst <= args.relative_ppl_limit,
            }
        )
    stable_ratios = [row["prune_ratio"] for row in ratio_summary if row["stable"]]
    maximum_stable = max(stable_ratios) if stable_ratios else None
    first_unstable_above = min(
        (
            row["prune_ratio"]
            for row in ratio_summary
            if not row["stable"] and (maximum_stable is None or row["prune_ratio"] > maximum_stable)
        ),
        default=None,
    )
    summary = {
        "relative_performance_limit": args.relative_ppl_limit,
        "maximum_stable_pruning_ratio": maximum_stable,
        "critical_interval": [maximum_stable, first_unstable_above],
        "seeds": seeds,
        "context_lengths": seqlens,
        "ratio_summary": ratio_summary,
        "curve_file": str(curve_path),
    }
    (root / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
