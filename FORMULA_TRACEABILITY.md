# 两份研究方案公式—代码逐项对齐表

本表按两份材料的共同主线“贡献定义—稳健排序—配置生成—闭环验证”核对。生产路径对不同场景的原生序列长度直接执行样本内标准化与 DCT-II，不在 DCT 前做池化、插值或长度重采样。

## 一、任务梯度响应

| 研究方案公式 | 代码位置 | 实现 |
|---|---|---|
| 注意力头 `G_{l,h}=∂L/∂O_{l,h}`，沿头维取二范数 | `lib/paper_pruning/collector.py` | 在 `o_proj` 输入注册梯度钩子，按 `[T,H,d_h]` 重排后对 `d_h` 取 L2，得到每个头沿位置的一维响应 |
| 前馈通道 `g_{l,j}(t)=|∂L/∂A_l(t,j)|` | `collector.py` | 在 `down_proj` 输入注册梯度钩子并取绝对值，得到每个中间通道沿位置的一维响应 |
| 样本内标准化 `(g-μ)/(σ+ε)` | `collector.py::_torch_proposal_dct`；固定长度测试路径为 `mi.py::standardize_responses` | 对每个样本、每个结构单元仅沿序列位置维计算均值和标准差 |

## 二、频域贡献谱

| 研究方案公式 | 代码位置 | 实现 |
|---|---|---|
| DCT-II 余弦求和 | 生产路径 `collector.py::_torch_proposal_dct`；测试路径 `mi.py::proposal_dct` | 使用 FFT 恒等式或 `scipy.fft.dct/2`，均与材料中未归一化余弦求和逐项一致 |
| `e_u(k)=|ĝ_u(k)|²` | `collector.py::_torch_fine_energy`；测试路径 `mi.py::dct_frequency_energy` | 对 DCT 系数逐点平方 |
| `z_u(q)=Σ_{k∈K_q}e_u(k)` | 同上 | 将原生长度频率轴拆为连续细桶并求和；缓存的是公式所需充分统计量 `[样本,单元,细桶]` |
| 位置级 NLL 与分位事件 | `collector.py::_quantile_events` | 保存完整 `position_losses.npy`、`position_events.npy` 与每条样本的偏移；进入样本级互信息的事件口径见文末说明 |
| 探针聚合 `z̄(q)=|U_probe|^{-1}Σ_u z_u(q)` | `mi.py::aggregate_probe_energy` | 对代表性结构单元的细桶能量求平均 |
| `r(q)=I(z̄(q);Y)` | `mi.py::_range_relevance` | 使用主 k 近邻互信息估计器 |
| 合并损失 `Δ(i)=r(b_i)+r(b_{i+1})-r(b_i∪b_{i+1})` | `mi.py::adaptive_merge_adjacent_bins` | 每轮选择合并损失最小的相邻桶，直到目标频带数或达到停止阈值 |
| `Z_u^(b)=Σ_{q∈b}z_u(q)` | `mi.py::aggregate_bands` | 对同一连续频带内的细桶能量求和 |
| `I_b(u)=I(Z_u^(b);Y)` | `mi.py::calculate_band_mi` | 主估计器为熵分解 k 近邻 MI；辅助估计器为 KDE |
| `S(u)=Σ_b π_b I_b(u)` | `mi.py::weighted_total_mi` | 频带权重归一化后加权求和 |

## 三、互信息估计器

| 研究方案公式 | 代码位置 | 实现 |
|---|---|---|
| `I(Z;Y)=H(Z)-Σ_c p(c)H(Z|Y=c)` | `mi.py::estimate_knn_mi_matrix` | 全局微分熵减去事件条件微分熵 |
| Kozachenko–Leonenko 近邻熵 | `mi.py::_kl_entropy_1d_matrix` | `ψ(n)-ψ(k)+log(V_1)+mean(log ε_i)`，标量响应时 `V_1=2` |
| KDE 条件密度与条件熵 | `mi.py::estimate_kde_mi_matrix` | 高斯核；粒球局部估计时带宽与粒球半径成比例 |

## 四、粒球多粒度局部化

| 研究方案公式 | 代码位置 | 实现 |
|---|---|---|
| 频带响应向量 `v_u(x)=[Z_u^(1),...,Z_u^(B)]ᵀ` | `granular_ball.py::unit_localization_features` | 直接使用原始频带响应向量；不再额外做跨样本标准化 |
| 粒球中心 `μ_g` 与半径 `R_g=max||v-μ_g||₂` | `granular_ball.py::make_ball` | 样本均值与到中心的最大欧氏距离 |
| 事件纯度 | `granular_ball.py::event_purity` | 球内占比最大的事件类别 |
| 样本数、事件纯度与几何紧致性联合分裂 | `build_multigranularity_hierarchy`、`split_ball` | 确定性二均值分裂，并检查最小样本、纯度、半径缩减、深度与类别支持 |
| 球内局部 MI | `_local_mi_for_partition` | 对每个结构单元、每个频带在球内估计连续—离散互信息 |
| `I_{b,m}=Σ_g p(g)I_b^(g)`，`p(g)=|S_g|/N` | `_local_mi_for_partition` | 按原始样本占比汇聚局部互信息 |
| 多粒度融合 `I_b^rob=Σ_m ρ_m I_{b,m}` | `multi_granularity_level_mi`、`fuse_granularity_levels` | 先保留各粒度估计，再按重复估计得到的自适应权重融合 |
| `ρ_m` 由估计方差或排序稳定性确定 | `granularity_weights_from_repeat_variance` | 采用材料明确允许的“估计方差”口径：对重复估计的单元—频带贡献求经验方差，使用逆方差归一化权重；也可显式给定权重 |

## 五、样本—场景双源重复估计与 LCB

| 研究方案公式 | 代码位置 | 实现 |
|---|---|---|
| 样本侧与场景侧双源抽样 | `resampling.py::dual_source_bootstrap_indices` | 第一阶段有放回抽取基础样本；第二阶段仅从该基础样本实际存在的场景实现中有放回抽取，保留配对依赖 |
| 每次重复重建频带、粒球和局部 MI | `pipeline.py::score_repeat` | 每个重复估计重新做任务相关频带合并、逐单元粒球划分与局部 MI |
| 多粒度方差权重 | `pipeline.py::score_layer_from_fine_energy` | 先汇总所有重复的逐粒度估计，再计算 `ρ_m` 并融合每个重复结果 |
| 均值与样本标准差 | 同上 | `mean` 与 `std(ddof=1)` |
| `LCB(u)=S̄(u)-λσ(u)` | 同上 | 同时输出总贡献 LCB 与逐频带稳健贡献 |

## 六、频带覆盖和部署预算下的配置生成

| 研究方案公式 | 代码位置 | 实现 |
|---|---|---|
| `C_b(K)=Σ_{u∈K}I_b^rob(u)` | `global_budget.py::coverage_aware_multi_budget_indices` | 动态累计保留集合的各频带稳健贡献 |
| `C_b(K)≥γ_b C_b^U` | 同上 | 每个频带独立检查最低覆盖比例 |
| `Σ cost(u)≤C` | 同上 | 参数量、FLOPs、显存、KV-cache、实测时延分别作为硬约束 |
| 欠覆盖权重 `w_b(K)=max(0,γ_bC_b^U-C_b(K))` | 同上 | 每次加入结构单元后更新 |
| 边际增益 `LCB/cost + αΣ_b w_b I_b^rob/cost` | 同上 | 严格逐单元贪心；懒惰堆只减少重复计算，不改变全量重算的下一步选择 |

材料给出的是标量部署代价。代码在多个预算同时启用时使用无量纲扩展 `Σ_j a_j c_{u,j}/C_j` 作为边际增益分母，同时仍逐项强制满足每一个资源预算；只启用一个指标时严格退化为材料公式。

## 七、完整结构化执行与闭环验证

| 研究方案流程 | 代码位置 | 实现 |
|---|---|---|
| 删除完整前馈通道 | `apply.py::zero_mlp_channels_` | 同时处理 gate/up 的对应行与 down 的对应列 |
| 删除完整注意力头 | `apply.py::zero_attention_heads_` | 同时处理 q/k/v 的对应行与 o 的对应列；对键值头共享结构拒绝不精确的单头删除 |
| 物理结构缩形 | `materialize.py::materialize_structured_units_` | 真实缩小耦合线性矩阵并报告参数变化 |
| 剪前/剪后贡献谱与排序稳定性 | `prune_paper.py::_run_post_prune_validation`、`stability.py` | Kendall、Spearman、Top-k Jaccard 与贡献变化 |
| 最大稳定剪枝率、性能突降区间与适用范围 | `run_paper_validation.py` | 多种子、多上下文长度、可选多任务/语言/输入场景评测 |

## 八、两份材料没有给出唯一公式的事件对齐点

详细材料先定义位置级事件 `Y'_t`，而 DCT 后的频带能量 `Z_u^(b)(x)` 是每个样本—场景一条向量；原文没有规定二者如何一一配对。生产代码完整保存位置级 NLL 与位置级事件用于诊断，同时以每个样本—场景的平均位置 NLL做全局分位离散，形成进入互信息估计的样本级事件 `Y`。这样不会把同一频带向量按 token 数重复复制而产生伪样本。该口径写入缓存元数据，便于复验。

## 三阶段消融公式映射

| 消融方法 | 使用的公式阶段 | 明确不执行的阶段 |
|---|---|---|
| `paper_mi` | 梯度响应、标准化、DCT、频带能量、频带互信息与总贡献 | 粒球、重复估计、LCB |
| `paper_mi_gb` | `paper_mi` + 粒球构造、粒内互信息、样本占比汇聚、多粒度融合 | 双源重复估计、均值/标准差、LCB |
| `paper_mi_gb_lcb` | `paper_mi_gb` + 双源重复估计、经验均值/标准差、LCB | 无 |

严格消融采用固定的 `ρ_m`，使第二组到第三组的唯一新增评分步骤为重复估计和 LCB。完整方法可使用 `repeat_variance` 恢复方差自适应 `ρ_m`。
