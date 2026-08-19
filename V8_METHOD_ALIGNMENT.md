# V8.2 Method Alignment Note

## Paper-derived part

V8.2 preserves the research pipeline used to obtain a robust structural-unit contribution signal:

`task gradient response -> frequency MI -> granular-ball local estimation -> repeated LCB -> optional frequency coverage`

The four ablations remain:

- `paper_mi`
- `paper_mi_gb`
- `paper_mi_gb_lcb`
- `paper_full`

## Deliberate engineering adaptation

The paper/proposal ultimately discusses pruning structural units such as attention heads and FFN channels. V8.2 instead enforces a user-required **weight-level 50% sparsity**. Therefore the final mask cannot be a literal implementation of the structural pruning stage.

V8.2 interprets the paper score as a **budget allocator**:

- high contribution -> lower weight sparsity for that unit;
- low contribution -> higher weight sparsity for that unit;
- default range is 45%-55%;
- a constrained projection enforces exactly 50% globally.

Within that quota, V8.2 uses Wanda's weight/activation metric to choose exact zero locations. This external element-wise metric is intentional and exists to preserve language-model quality.

## Removed V8 behavior

The primary V8 path no longer uses:

- `paper_score / unit_weight_cost` as a per-weight utility;
- deterministic/uniform pseudo-random zero positions inside a unit.

The old deterministic helper functions remain in `weight_budget.py` only for legacy unit tests/backward compatibility and are not called by the V8.2 paper pruning path.

## Full method coverage adaptation

A literal 90% retained band coverage is infeasible when every unit is only 45%-55% retained/pruned. V8.2 maps the requested coverage ratio onto feasible keep-fraction headroom. With the default 50% global target and 45% minimum unit sparsity, a 0.90 coverage ratio corresponds to a desired per-band retained fraction of approximately 0.545.
