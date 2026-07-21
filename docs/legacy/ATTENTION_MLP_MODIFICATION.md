# MLP + Attention Structured Pruning Modification

## What changed

1. Added attention-head task-gradient response collection at the input of `self_attn.o_proj`.
2. Added MI, granular-ball and LCB scoring for attention heads at every layer.
3. Added structured attention masks across q/k/v rows and o-projection columns.
4. Kept the existing structured MLP-channel mask.
5. Added separate ratios and exact-count parameters for MLP and attention.
6. Added `unit_type` to CSV reports and separate attention/MLP plots.
7. Versioned the response cache so old MLP-only caches cannot be reused accidentally.
8. Added per-observation progress printing.

## New arguments

- `--paper_prune_targets mlp,attention`
- `--mlp_sparsity_ratio 0.50`
- `--attention_sparsity_ratio 0.50`
- `--attention_prune_per_layer 0`

When the module-specific ratios are omitted, both use `--sparsity_ratio`.

## Expected sparsity for Llama-2-7B

With 50% MLP channels and 50% attention heads zeroed, every q/k/v/o and gate/up/down projection receives a 50% structured zero mask. Therefore `check_sparsity()` should report approximately 0.50 for each transformer layer, apart from any tiny pre-existing numerical-zero difference.

## Tests

```text
PYTHONPATH=. pytest -q
5 passed
```
