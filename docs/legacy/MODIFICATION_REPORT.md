# 论文对齐剪枝改造说明

## 已完成

1. 新增三条真正独立的消融路径：
   - `paper_mi`
   - `paper_mi_gb`
   - `paper_mi_gb_lcb`
2. 将评分信号改为 MLP 中间通道的任务损失梯度响应：`z * dL/dz`。
3. 实现样本内标准化、DCT、频率能量桶、任务相关性驱动的相邻频带合并。
4. 实现 kNN 主估计器与高斯 KDE 辅助估计器的互信息融合。
5. 粒球按层构建，并根据事件纯度与几何紧致性递归二分。
6. 粒球采用嵌套多粒度层级，高纯度阈值在前一粒度上继续分裂。
7. LCB 使用事件与场景联合分层 bootstrap，每次重新合并频带、重新构建粒球。
8. 导出每个通道的 MI、粒球贡献、LCB 均值、标准差与最终 LCB。
9. 导出每层按 10 个索引分批的剪枝计划。
10. MLP 通道剪枝同步置零 `gate_proj` 行、`up_proj` 行和 `down_proj` 列。
11. 新增一键三组消融脚本 `run_paper_ablation.py`，每组重新加载原模型。
12. 新增掩码重合诊断，避免三种方法实际使用同一套索引而不自知。

## 主要新增文件

```text
lib/prune_paper.py
lib/paper_pruning/config.py
lib/paper_pruning/collector.py
lib/paper_pruning/mi.py
lib/paper_pruning/granular_ball.py
lib/paper_pruning/resampling.py
lib/paper_pruning/pipeline.py
lib/paper_pruning/selection.py
lib/paper_pruning/apply.py
lib/paper_pruning/reporting.py
run_paper_ablation.py
tests/test_paper_pruning.py
```

## 验证结果

```text
PYTHONPATH=. pytest -q tests/test_paper_pruning.py
5 passed
```

未在当前环境中运行真实 Llama 权重与本地 C4/WikiText 数据，因此真实 PPL 需要在你的 A100 环境中复测。代码不会人为调整分数来强制 PPL 单调下降；如果结果仍接近，应检查 `mask_overlap.csv`、实际剪枝比例和 PPL 输出精度。
