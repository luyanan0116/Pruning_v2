# V8.3 FAST 六组实验

此版本保留 `forward / reverse / joint × band on/off` 六组入口，并针对重复计算做缓存。

## 默认：balanced

默认 `FAST_PROFILE=balanced`：

- Wanda calibration: `32 × 1024`（原版 `128 × 4096`）
- Paper gradient samples: `32`（原版 `128`）
- Paper calibration length: `512`（原版 `1024`）
- Scenario ratios: `0.5,1.0`（原版三组）
- LCB repeats: `8`（原版 `20`）
- WikiText evaluation context: `2048`（原版 `4096`）
- 关闭 KDE 辅助诊断和绘图；KDE 在该实现中不参与主 pruning score，因此关闭它本身不会改变主 score。

因此 balanced 适合先完成六组对比，但其数值不能直接当作原 V8.3 full-calibration 的最终论文结果。

## full_cache

`FAST_PROFILE=full_cache` 保留原 V8.3 的 calibration 数量/长度，只启用共享缓存、关闭诊断 KDE/绘图。结果定义更接近原 V8.3，但第一组仍然可能较慢。

## 三类共享缓存

同一 profile 下六组自动共享：

1. `shared_response_cache`: task-gradient response
2. `shared_score_cache`: MI / granular-ball / LCB evidence
3. `shared_dense_wanda_stats`: reverse/joint 的 dense Wanda activation statistics

`forward` 不能复用 dense stats，因为前层剪枝会改变后层 activation；这是有意保留的实验定义。

## nohup 一次跑完

```bash
chmod +x run_all_v83_fast.sh
mkdir -p logs
nohup env FAST_PROFILE=balanced bash run_all_v83_fast.sh > logs/run_all_v83_fast.log 2>&1 < /dev/null &
```

查看总进度：

```bash
tail -f logs/run_all_v83_fast.log
```

若要用原 calibration 尺度但保留缓存优化：

```bash
nohup env FAST_PROFILE=full_cache bash run_all_v83_fast.sh > logs/run_all_v83_full_cache.log 2>&1 < /dev/null &
```
