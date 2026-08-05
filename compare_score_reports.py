from __future__ import annotations

import argparse
from pathlib import Path

from lib.paper_pruning.stability import compare_score_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare contribution spectra and rankings before/after pruning.")
    parser.add_argument("--before_report", required=True)
    parser.add_argument("--after_report", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--top_fraction", type=float, default=0.10)
    args = parser.parse_args()
    before = Path(args.before_report) / "all_contribution_scores.csv"
    after = Path(args.after_report) / "all_contribution_scores.csv"
    output = Path(args.output) if args.output else Path(args.after_report) / "pre_post_spectrum_stability.csv"
    path = compare_score_reports(before, after, output, args.top_fraction)
    print(path)


if __name__ == "__main__":
    main()
