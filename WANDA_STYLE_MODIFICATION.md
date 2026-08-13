# v4 修改摘要

## 关键纠正

官方 Wanda 的 `--sparsity_ratio 0.5` 是 50% **权重稀疏度**，并不是删除 50% 的完整 MLP 通道和注意力头。Wanda 在每个 Transformer block 内递归处理所有 `nn.Linear`，因此 Attention 与 MLP 都会被剪，但剪的是矩阵元素。

## 本版本新增

- 新参数 `--paper_mask_style wanda_weight|structured_unit`，默认 `wanda_weight`。
- 对 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` 全部应用权重级掩码。
- 缓存每层七个线性模块输入的均方激活，用于 Wanda 度量 `|W|*sqrt(E[x^2])`。
- 使用 MI、粒球或 LCB 单元分数重新分配相同总权重预算，使三组消融得到不同掩码。
- 保存 `unit_scores.npz` 和每个方法的 `weight_mask_summary_*.csv`。
- 保留 `structured_unit` 以复现实验性完整通道/头置零，但不推荐在 50% 下作为 Wanda 对比。

## 测试

```text
8 passed
```
