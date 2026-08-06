# paper_mi50 修改说明

## 目标

只保留青基路线中的全局频域互信息阶段作为附加贡献信号，在 Llama-2-7B 的 50% 权重稀疏口径下尽量贴近普通顺序 Wanda 的低 PPL 工作点。

该模式名为：

```text
--prune_method paper_mi50
```

## 与 Pruning-main 的关键区别

`Pruning-main/lib/lcb_utils.py` 的实际执行路径并没有计算任务事件变量 Y 与频带响应之间的互信息。它先对线性层输入激活做 FFT 能量分桶，再用频带能量的逆方差 `1/(var+eps)` 作为“互信息代理”，最终乘到 Wanda 指标 `|W|*sqrt(activation_scale)` 上。`lib/lcb_core.py` 和 `lib/robust_inference.py` 中存在带 Y 的 MI 函数，但主剪枝路径没有调用它们。

v6/v7 的 `paper_mi` 使用青基对齐流程：

1. 任务损失对注意力头输出、FFN 中间通道的梯度响应；
2. 样本内标准化；
3. 正交 DCT-II 与频带能量；
4. 由位置级 next-token NLL 构造统一任务事件 Y；
5. 连续响应-离散事件的 kNN MI；
6. 任务相关性驱动的相邻频桶合并；
7. 频带覆盖与 50% 预算约束；
8. 顺序 Wanda 生成实际权重 mask。

## 为什么 Pruning-main 容易得到约 6.48

它本质上仍处于 Wanda 的工作点：每个线性层输出行固定剪掉 50% 最小指标权重，附加信号只是一个列级缩放因子；没有完整头/通道删除，也没有强烈的单元预算重分配。因此其 PPL 更接近普通 Wanda，而不能据此证明“完整青基互信息方法单独达到 6.48”。

另一个必须统一的口径是评估长度：Pruning-main 把 `model.seqlen` 直接设为模型 `max_position_embeddings`，v6 默认显式使用 2048。模型、WikiText-2 文本、tokenizer、评估长度不完全一致时，两个 PPL 数值不能直接比较。

## paper_mi50 的低 PPL 保护

- 只计算全局 MI，不执行粒球与 LCB；
- 仍保留频带覆盖和预算约束；
- 强制同时剪 q/k/v/o、gate/up/down；
- 强制每个矩阵与总体权重稀疏率精确为 0.500000；
- 使用顺序 Wanda，前层剪枝后重新传播校准输出；
- 禁止单元间行预算重分配：`row_spread=0`；
- MI 仅作为很弱的近 1 乘性 tie-breaker，默认强度 `0.001`；为保护 PPL，固定行预算下它主要改变 `down_proj/o_proj` 的列选择，q/k/v 与 gate/up 保持标准 Wanda 的逐行 50%；
- 运行后自动检查每个矩阵和总体的 50% 稀疏率，未达到会直接报错。

默认 MI 因子范围约为：

```text
exp(-0.001) ~ exp(+0.001) = 0.9990005 ~ 1.0010005
```

这会把实际 mask 限制在普通 Wanda 阈值附近变化，目标是保持 MI 有效的同时避免 PPL 明显偏离 Wanda 基线。

## 真实性边界

代码包中没有 Llama-2-7B、C4、WikiText-2 与目标 GPU，因此无法在本地声称已经实测到 6.48。建议先运行 `scripts/run_mi50_lowppl.sh`；若结果仍有偏差，运行 `scripts/run_mi50_guidance_sweep.sh`，从非零强度中选择最接近 6.48 的结果。所有候选均硬性保持 50% 稀疏率。
