# V8.2 Change Log

## P0: final mask quality

- Added `lib/paper_pruning/wanda_mask.py`.
- Added sequential layerwise activation statistics matching Wanda's `scaler_row` definition.
- Added per-output Wanda 50% baseline.
- Paper variants use Wanda element metrics instead of deterministic uniform positions.

## P0: budget bias

- Removed `score / cost` from the active nonuniform allocator.
- Added robust score normalization + `tanh` score-to-sparsity mapping.
- Added bisection shift + integer residual correction for exact global 50%.
- Changed default per-unit sparsity bounds from 35%-65% to 45%-55%.

## P1: transposed unit ownership

For matrices where structural ownership is column-oriented (`down_proj`, `o_proj`), V8.2 first applies a standard Wanda per-output 50% anchor mask. Remaining unit quota is absorbed by `gate/up` and `q/k/v` respectively. This prevents activation statistics from cancelling out under pure column-wise ranking.

## P1: fair evaluation

- Default model/evaluation sequence length: 4096.
- Added explicit `dense` and `wanda` methods.
- Added WikiText-2 validation/test split control.
- Default ablation tuning uses validation.

## P1: stronger paper statistics

- score samples 64 -> 128
- paper calibration length 512 -> 1024
- response length 32 -> 128
- LCB repeats 10 -> 20
- scenario fraction 0.67 -> 2/3
- fixed scenario bootstrap count to rounded/clamped selection, so 3 scenarios at 2/3 selects 2 rather than `ceil(2.01)=3`.

## Verification

Current local unit test result in the build environment: `12 passed`.
The build environment does not contain the `transformers` package, so a full Llama-2-7B GPU run could not be executed here; server-side Dense/Wanda baselines should be run first.
