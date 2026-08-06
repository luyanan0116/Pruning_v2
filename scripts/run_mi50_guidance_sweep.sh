#!/usr/bin/env bash
set -euo pipefail

TARGET_PPL="${TARGET_PPL:-6.48}"
STRENGTHS=(0.00025 0.0005 0.001 0.002 0.005)
ROOT="${ROOT:-results/mi50_guidance_sweep}"
mkdir -p "$ROOT"

for strength in "${STRENGTHS[@]}"; do
  tag=$(printf '%s' "$strength" | tr '.' 'p')
  echo "=== paper_mi50 guidance=$strength ==="
  GUIDANCE="$strength" OUT="$ROOT/g_$tag" bash scripts/run_mi50_lowppl.sh
 done

python - "$ROOT" "$TARGET_PPL" <<'PY'
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
target = float(sys.argv[2])
rows = []
for log in sorted(root.glob('g_*/log_paper_mi50.txt')):
    lines = [line.strip().split('\t') for line in log.read_text().splitlines() if line.strip()]
    if len(lines) < 2:
        continue
    method, sparsity, ppl = lines[-1]
    rows.append({
        'run': log.parent.name,
        'method': method,
        'actual_sparsity': float(sparsity),
        'ppl': float(ppl),
        'distance_to_target': abs(float(ppl) - target),
    })
rows.sort(key=lambda row: row['distance_to_target'])
out = root / 'mi50_guidance_sweep_summary.csv'
with out.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['run','method','actual_sparsity','ppl','distance_to_target'])
    writer.writeheader()
    writer.writerows(rows)
print(f'wrote {out}')
if rows:
    print('best non-zero MI guidance:', rows[0])
PY
