# v6.3 score-driven unit-budget weight pruning

New mask style:

```bash
--paper_mask_style unit_budget_weight
--sparsity_ratio 0.5
--paper_unit_min_sparsity 0.30
--paper_unit_max_sparsity 0.70
```

The final unit score (LCB-only, band-only, or paper-hybrid priority) now controls
the actual amount of weight pruning assigned to each Attention head / FFN channel.

For 50% total sparsity:
- highest-priority units approach 30% local weight sparsity;
- middle-priority units are around 50%;
- lowest-priority units approach 70%;
- every linear module preserves an exact ~50% total zero-weight budget (integer rounding only).

The within-unit weight choice still uses `abs(weight) * sqrt(input_activation_scale)`.
That metric no longer determines how much budget a unit receives; it only chooses
which individual weights are removed inside the unit's assigned budget.

For row-oriented modules (q/k/v, gate/up), the budget is applied to rows belonging
to each head/channel. For column-oriented modules (o_proj, down_proj), the budget
is applied to the corresponding head/channel columns as well, so the unit score
affects all seven linear projections rather than only q/k/v/gate/up.
