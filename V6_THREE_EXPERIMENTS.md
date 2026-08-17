> **v7 注意：** 本文件记录的是旧版实现。当前行为以 `V7_GLOBAL_BOOTSTRAP_ABLATION.md` 为准：已恢复真实双源 bootstrap/LCB，并采用 MI→GB→LCB→Full 四级干净消融与跨层全局预算。

# v6 三组对比实验

## 结论先行

原始 v6 的 `--paper_mask_style` 默认值是 `wanda_weight`，因此 `--sparsity_ratio 0.5` 默认表示 q/k/v/o、gate/up/down 权重矩阵约 50% 的权重元素置零，而不是删除 50% 完整通道/注意力头。`structured_unit` 才表示完整结构单元剪枝。

本修改版把默认值改成 `structured_unit`，同时保留 `wanda_weight` 仅用于实验1的旧版权重级基线。

## 实验1：50%权重 vs 50%完整结构单元

运行：`scripts/exp1_weight50_vs_unit50.sh`。

- `weight50`：旧版权重级基线；它包含既有的权重幅值/激活尺度路径，因此不是“纯互信息”实验。
- `unit50`：按同一 MI+粒球+LCB+频带配置选择结果，删除 50% 完整 FFN 中间通道和 Attention heads。

只给一个通道级 MI 分数时，无法在“纯 MI”前提下区分该通道内部不同权重；所以不能把 weight50 解释成纯 MI 的 weight-level 对照。

## 实验2：低/中/高频带平衡

运行：`scripts/exp2_low_mid_high_balance.sh`。

将最终频带数设为 3（自适应合并后仍按频率顺序对应低/中/高），比较：

- uniform: gamma=(0.70,0.70,0.70)
- low_heavy: gamma=(0.85,0.70,0.55)

这里的 gamma 是“频带贡献覆盖率”，不是“低频通道数占85%”。同一个结构单元可以同时贡献多个频带。

## 实验3：最终保留依据

运行：`scripts/exp3_lcb_vs_band_keep.sh`。

- `lcb_only`：只按标量 LCB 排序保留。
- `band_only`：不使用标量 LCB，按低/中/高频带贡献及覆盖目标保留。
- `paper_hybrid`：LCB + 欠覆盖频带补偿；这是最接近论文公式(11)-(13)的完整配置生成方式。

完整方法中，频带覆盖使用重复估计得到的稳健频带均值；标量稳定性惩罚由 LCB 单独承担，避免对方差重复惩罚。
