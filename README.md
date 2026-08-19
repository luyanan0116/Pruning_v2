# v8 Paper-only 50% Weight Pruning

This version implements the proposal pipeline without any external weight-pruning metric:

1. task-loss gradient response for attention heads and FFN intermediate channels;
2. per-sample standardization, DCT, fine frequency buckets, task-driven adjacent-band merging;
3. frequency-band mutual-information contribution spectrum;
4. granular-ball local MI and multi-granularity fusion;
5. sample/scenario repeated estimation and LCB ranking;
6. Full method: LCB + frequency under-coverage + parameter budget marginal gain;
7. map paper-unit contribution to a global **weight** budget and set exactly 50% of the targeted transformer projection weights to zero.

## Important adaptation

The proposals formulate the final decision over complete heads/channels. This v8 intentionally does **not** delete complete heads or FFN channels. Instead, each paper unit receives a partial weight-pruning budget. The default bounds are 35%-65% sparsity per unit, so no scored unit is fully zeroed. Within each unit, exact zero positions are deterministic and uniformly distributed from a fixed seed. No weight magnitude, input-activation importance, Hessian, or other external weight score is used.

For Llama-2-7B, the targeted weights are the seven transformer projections per block: q/k/v/o and gate/up/down. Embeddings, normalization parameters, biases, and the LM head are not part of the 50% target.

## Reusing the v7 response cache

A v7 `response_cache/` is compatible. v8 reads only gradient responses, event labels, sample/scenario identifiers and NLL information; any extra old files in that cache are ignored.

Example:

```bash
RESPONSE_CACHE_DIR=results/paper_v7_clean_s050/response_cache \
bash scripts/run_v8_paper_only_weight50.sh
```

## Four clean ablations

- `paper_mi`: frequency-domain MI only;
- `paper_mi_gb`: MI + granular-ball multi-granularity local estimation;
- `paper_mi_gb_lcb`: previous stage + repeated sample/scenario LCB;
- `paper_full`: LCB + frequency coverage + global parameter budget.

All four runs use fresh model copies and exactly 50% target projection-weight sparsity.
