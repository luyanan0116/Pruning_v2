# V8.2 — Paper-guided 50% Weight Pruning + Wanda Element Mask

This package is a **performance-oriented adaptation** of the V8 paper pipeline. It no longer claims that the final weight mask is purely paper-derived.

The paper-side contribution pipeline is retained:

1. task-loss gradient response;
2. frequency decomposition and frequency-band MI contribution;
3. granular-ball multi-granularity local MI;
4. repeated sample/scenario estimation and LCB;
5. optional frequency-coverage refinement for `paper_full`.

The final pruning stage is redesigned for the user's hard constraint:

- pruning object = **individual weights**, not complete heads/channels;
- targeted transformer projections = q/k/v/o + gate/up/down;
- global targeted-weight sparsity = **exactly 50%**;
- every paper unit is constrained to **45%-55%** sparsity by default;
- paper score decides the unit quota;
- Wanda `|W| * sqrt(input-feature energy)` decides which elements are zeroed;
- masking is sequential layer-by-layer: collect current-layer activations -> prune -> propagate sparse output to the next layer.

## Why V8.2 changed V8

V8 divided paper score by unit weight cost and then selected zero locations uniformly inside the chosen unit. For Llama-2-7B, an attention head owns far more weights than one FFN channel, so `score/cost` systematically suppresses attention-head utility. Uniform within-unit zeroing also ignores weight/activation importance.

V8.2 removes both behaviors:

- **no `score / cost` ranking**;
- **no deterministic-uniform mask in the main path**.

A bounded constrained projection maps contribution score directly to a unit sparsity fraction. A scalar shift is solved by bisection so the weighted sum of all unit prune counts is exactly 50%.

## Baseline first

Use the same model, tokenizer and WikiText sequence length for Dense and Wanda:

```bash
MODEL_PATH=meta-llama/Llama-2-7b-hf \
C4_PATH=/path/to/c4 \
WIKITEXT2_PATH=/path/to/wikitext-2-raw \
bash scripts/run_v82_dense_wanda_baseline.sh
```

The Wanda baseline uses:

- C4 calibration;
- `nsamples=128`;
- `seed=0`;
- `seqlen=4096`;
- per-output-row 50% unstructured pruning;
- tokenizer `use_fast=False`.

## Four paper ablations + baselines

```bash
MODEL_PATH=meta-llama/Llama-2-7b-hf \
C4_PATH=/path/to/c4 \
WIKITEXT2_PATH=/path/to/wikitext-2-raw \
bash scripts/run_v8_paper_only_weight50.sh
```

This runs fresh model copies for:

- `dense`
- `wanda`
- `paper_mi`
- `paper_mi_gb`
- `paper_mi_gb_lcb`
- `paper_full`

Default tuning evaluation is **WikiText-2 validation**, not test. Use test only after parameters are fixed:

```bash
python run_paper_ablation.py ... --eval_wikitext_split both
```

## V8.2 defaults

Paper statistics:

```text
paper_score_nsamples       128
paper_calib_seqlen         1024
paper_response_length      128
paper_lcb_repeats          20
paper_lcb_sample_fraction  0.80
paper_lcb_scenario_fraction 2/3
```

Mask statistics / fair Wanda comparison:

```text
wanda_nsamples             128
wanda_calib_seqlen         4096
seqlen                     4096
```

Budget:

```text
paper_weight_min_unit_sparsity 0.45
paper_weight_max_unit_sparsity 0.55
paper_budget_temperature       1.0
sparsity_ratio                 0.50
```

## Recommended tuning order

Do not tune all MI/GB/LCB hyperparameters at once. First verify:

1. Dense PPL is correct under the 4096 evaluation path.
2. `wanda` is close to your standalone Wanda reproduction.
3. `paper_mi` does not regress badly versus Wanda.
4. Add GB, then LCB, then Full and compare **validation PPL**.

If nonuniform allocation hurts, first weaken the quota contrast rather than changing MI:

```text
A. 0.475 - 0.525
B. 0.45  - 0.55   (default)
C. 0.425 - 0.575  only after A/B are stable
```

The target PPL around 6.5 is an experimental objective, **not a guaranteed outcome**. The code is designed to remove the two most damaging V8 mechanisms and make the comparison with Wanda controlled and diagnosable.

## GPU memory

Official-style 128 x 4096 sequential calibration stores large hidden-state buffers. V8.2 supports:

```bash
--wanda_activation_storage auto   # default
--wanda_activation_storage cuda
--wanda_activation_storage cpu
```

`auto` uses CUDA only when sufficient free memory is detected; otherwise it stores calibration hidden states on CPU and transfers one sample at a time.

## Tests

```bash
pytest -q
```

V8.2 includes tests for:

- exact 50% budget;
- 45%-55% per-unit quota bounds;
- score direction changing quota;
- no complete unit removal;
- exact sequential Wanda 50% mask;
- exact paper-unit quota after activation-aware masking.


## V8.2 local-only model loading

This build intentionally rejects Hugging Face repository IDs. `--model` must point to a local
Transformers-format checkpoint directory (for example `/root/dw2/Lya/models/Llama-2-7b`).
The loader uses `local_files_only=True` and the run scripts export `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`. The local directory must contain
`config.json`, tokenizer files, and `.safetensors` or `pytorch_model*.bin` weights. An original
Meta checkpoint containing `params.json` + `consolidated.*.pth` must first be converted to
Transformers format.
