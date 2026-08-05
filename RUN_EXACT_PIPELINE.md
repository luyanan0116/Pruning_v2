# 推荐运行顺序

## 1. 准备本地数据

```bash
export C4_PATH=/data/c4
export WIKITEXT2_PATH=/data/wikitext-2-raw
```

## 2. 先只生成贡献谱和稳健排序

```bash
python main.py \
  --model /models/llama-2-7b \
  --prune_ratio 0 \
  --paper_score_only \
  --paper_scenario_manifest paired_scenarios.jsonl \
  --paper_cache_dir outputs/cache \
  --paper_report_dir outputs/scores \
  --output_dir outputs/score_run \
  --no-eval_after
```

检查：

- `all_contribution_scores.csv`
- `frequency_band_boundaries.csv`
- `granular_ball_summary.csv`
- `mi_estimator_consistency.csv`
- `bootstrap_ranking_stability.csv`

## 3. 根据部署预算生成结构配置

```bash
python generate_global_config.py \
  --report_dir outputs/scores \
  --model /models/llama-2-7b \
  --method paper_mi_gb_lcb \
  --prune_ratio 0.15 \
  --budget_metrics params,flops,memory,kv_cache \
  --coverage_ratio 0.90 \
  --output outputs/selection.json
```

## 4. 在新模型实例上执行结构配置并评测

```bash
python main.py \
  --model /models/llama-2-7b \
  --prune_method paper_mi_gb_lcb \
  --prune_ratio 0.15 \
  --paper_selection_file outputs/selection.json \
  --paper_mask_style structured_zero \
  --paper_report_dir outputs/application \
  --eval_before \
  --eval_after \
  --output_dir outputs/evaluation
```

## 5. 完整剪前/剪后贡献谱闭环

不传 `--paper_selection_file`，让 `main.py` 在一次运行中完成评分、配置生成、结构删除和剪后重估：

```bash
bash scripts/run_exact_full_pipeline.sh
```

## 6. 扫描最大稳定剪枝率

```bash
bash scripts/run_validation_sweep.sh
```
