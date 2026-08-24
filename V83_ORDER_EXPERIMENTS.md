# V8.3 Layer-Order Experiments

This extension keeps the V8.2 paper-guided quota + Wanda element-mask design and adds a layer chronology axis.

- `forward`: original V8.2 behavior. For layer 1 -> N: collect Wanda input statistics, prune, then propagate the pruned output to the next layer.
- `reverse`: first collect/freeze Wanda statistics for all layers on the unpruned model; then apply masks from layer N -> 1.
- `joint`: first collect/freeze Wanda statistics for all layers; only after all layers are calculated are masks applied. The implementation applies the already-decided local masks layer-by-layer to avoid storing multi-billion-element boolean masks.

## Frequency-band gradient switch

`--paper_use_band_gradient` enables the `paper_full` frequency-band coverage refinement using `--paper_coverage_alpha` (default experiment value `0.10`).

`--no-paper_use_band_gradient` sets the effective coverage-refinement strength to zero while retaining the same MI/GB/LCB contribution pipeline and the same Wanda masker. This isolates the effect of the final frequency-band refinement without changing the rest of V8.2.

## Six scripts

```bash
bash scripts/run_v83_forward_band_on.sh
bash scripts/run_v83_forward_band_off.sh
bash scripts/run_v83_reverse_band_on.sh
bash scripts/run_v83_reverse_band_off.sh
bash scripts/run_v83_joint_band_on.sh
bash scripts/run_v83_joint_band_off.sh
```

Server paths can be overridden exactly like the V8.2 scripts:

```bash
MODEL_PATH=/path/to/model \
C4_PATH=/path/to/c4 \
WIKITEXT2_PATH=/path/to/wikitext \
bash scripts/run_v83_forward_band_on.sh
```

Results default to `results/v83_order/<order>_band_<on|off>/`. The response cache is shared at `results/v83_order/shared_response_cache` because the paper gradient-response collection does not depend on Wanda pruning order.

### Expected reverse vs joint relationship

For Wanda's metric `|W| * sqrt(input-feature energy)`, a later layer's pruning cannot change an earlier layer's input activations. Both `reverse` and `joint` therefore use identical dense-model frozen activation statistics and should produce the same final masks under deterministic kernels. They remain separate modes so this equivalence can be checked experimentally and reported explicitly.
