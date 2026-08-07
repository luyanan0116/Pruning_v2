# v6 Wanda 代码路径精确清单与 v7 处理决定

结论先写在前面：**Wanda 可以保留，但只能作为外部基线/消融基线；不应再进入论文主流程的打分、预算分配或 mask 生成。**
`sample.pdf` 的预实验本来就是把 Wanda 与 Grad_norm 当作稳定性对照，而论文主方法是“梯度响应→频域互信息→粒球→重复估计→LCB→覆盖/预算→结构单元剪枝”。

| 文件名 | 函数/代码段 | 行为 | Wanda/非Wanda | v7 处理 |
|---|---|---|---|---|
| `main.py` | `prune_method == "wanda"` | 调普通 Wanda 基线 | Wanda 基线 | **保留**，仅显式 `--prune_method wanda` 才进入 |
| `main.py` | `paper_mask_style=wanda_weight` 与全部 `paper_wanda_*` 参数 | 让论文 MI/GB/LCB 分数去引导 Wanda 权重级 mask | Wanda 混入论文路径 | **删除** |
| `lib/prune.py` | `prune_wanda()` | `|W|*sqrt(E[x^2])` 的标准 Wanda 权重级剪枝 | Wanda 基线 | **保留** |
| `lib/prune.py` | `prune_ablate()` 中 `ablate_wanda_*` | Wanda mask + 二阶/迭代消融 | Wanda 基线/消融 | **保留**，但不属于论文主方法 |
| `lib/ablate.py` | `get_wanda_mask()` | 构造 Wanda mask | Wanda | **保留**，仅供基线消融 |
| `lib/ablate.py` | `fasterprune()` 内 `"wanda"` 分支 | 用 Wanda metric 生成局部 mask | Wanda | **保留**，仅供基线消融 |
| `lib/layerwrapper.py` | `WrappedGPT.scaler_row` | 累计输入激活二范数，供 Wanda metric 使用 | Wanda 基础设施 | **保留**，因为普通 Wanda 仍需要 |
| `lib/paper_pruning/wanda_weight.py` | `centered_rank_factor()` | 用论文分数乘到 Wanda metric 上 | Wanda+论文混合 | **从 active paper path 删除**；原版归档到 `legacy/v6_wanda_paper/` |
| `lib/paper_pruning/wanda_weight.py` | `allocate_row_prune_counts()` | 按论文分数重新分配 Wanda 每行权重稀疏预算 | Wanda+论文混合 | **删除/归档** |
| `lib/paper_pruning/wanda_weight.py` | `apply_guided_wanda_module_()` | `|W|*sqrt(E[x^2])` 后按行/列置零 | Wanda+论文混合 | **删除/归档** |
| `lib/paper_pruning/wanda_weight.py` | `apply_paper_wanda_weight_masks_()` | 对 q/k/v/o、gate/up/down 做论文引导的 Wanda 权重剪枝 | Wanda+论文混合 | **删除/归档** |
| `lib/paper_pruning/wanda_sequential.py` | `capture_first_layer_inputs()` | Wanda 顺序校准输入捕获 | Wanda | **删除/归档** |
| `lib/paper_pruning/wanda_sequential.py` | `apply_sequential_paper_wanda_masks_()` | 逐层重算激活后做 Wanda 权重 mask | Wanda+论文混合 | **删除/归档** |
| `lib/paper_pruning/collector.py` | `LINEAR_MODULES` + `activation_sums` + `register_activation_hook()` | 缓存七个线性层输入均方，仅为 Wanda `sqrt(E[x^2])` 服务 | Wanda 辅助路径 | **改写**：严格 collector 只收集任务损失梯度响应和 NLL 事件 |
| `lib/prune_paper.py` | Wanda imports/config/validation/apply 分支 | 论文分数最后落到 Wanda 权重 mask | Wanda+论文混合 | **整体改写**：现在只生成全局 K 并做完整结构单元剪枝 |
| `lib/paper_pruning/budget.py` | `final_priority` / “for Wanda guidance” | 把 Eq.(11)-(13) 的结果转成连续优先级给 Wanda mask | Wanda 接口 | **改写**：直接输出 Eq.(11)-(13) 的最终保留集合 K |
| `lib/lcb_utils.py` | `compute_lcb_weight_metric()` | 最终仍是 `abs(W)*sqrt(scaler_row)*lcb_factor` | Wanda+旧近似 | **删除/归档**；且内部用 FFT 近似 DCT、随机子集替代粒球，不应进入论文实现 |
| `scripts/run_wanda_baseline.sh` | 脚本本体 | 普通 Wanda 50% 基线 | Wanda 基线 | **保留** |
| `scripts/run_fast_paper_ablation.sh` | `paper_mask_style wanda_weight` / `paper_wanda_*` | 论文实验最终跑 Wanda 权重 mask | Wanda+论文混合 | **改写**：改为完整 Head/GQA bundle + FFN Channel 结构剪枝 |
| `scripts/run_target65_ablation.sh` | 全套 Wanda 低 PPL 参数 | 以 PPL≈6.5 为 Wanda 权重稀疏目标 | Wanda 专用 | **移出 active**，归档 |
| `tests/test_paper_pruning.py` | Wanda mask 相关测试 | 验证论文分数对 Wanda mask 的影响 | Wanda+论文混合 | **移出 active**；v7 新增 Eq.(11)-(13) 与完整结构剪枝测试 |
| `README/PAPER_ALIGNMENT_REPORT/RUN_TARGET_65` | Wanda 作为论文落地路径的描述 | 混淆“权重稀疏”和“结构单元剪枝” | Wanda+论文混合 | **改写/归档** |

## v7 严格论文路径

`梯度响应 → 样本内标准化 → DCT → 细桶频带能量 → 任务驱动相邻频带合并 → MI → 粒球局部化 → 多粒度融合 → 样本-场景双源重复估计 → LCB → Eq.(11) 频带硬覆盖 → Eq.(12) 欠覆盖权重 → Eq.(13) 预算边际增益 → K → 完整 Head/GQA bundle 与 FFN Channel 剪枝 → 结构剪枝率 → PPL`

关键改动：

1. `collector_strict.py` 不再收集任何 Wanda 激活尺度。
2. Attention 在 GQA 模型中把“一个 KV head + 与其共享的所有 query heads”作为一个可执行结构单元；MHA 时就是普通单 head。
3. Eq.(11)-(13) 从“每层固定 keep_count”改为**跨层、跨 Head/FFN 的全局 K**。
4. Eq.(11) 是硬约束：预算下若达不到频带覆盖，直接报 infeasible，不再只打印 coverage。
5. Eq.(11) 的覆盖度使用稳健逐频带贡献；目标函数使用 LCB。
6. `paper_apply_mode=shrink` 物理缩小 q/k/v/o 与 gate/up/down 的张量维度；`zero` 仅用于兼容性消融。
7. 输出三种剪枝率：结构单元比例、候选资源 cost 比例、整模型真实参数量比例。
8. PPL 在实际剪枝后的内存模型上直接评测。
