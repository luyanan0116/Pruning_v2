# V8 method alignment

The implemented statistical path is unchanged from the proposal: gradient task response -> DCT frequency representation -> task-event mutual information -> granular-ball local estimation -> multi-granularity fusion -> repeated sample/scenario estimates -> LCB -> coverage-aware budget.

The sole deliberate extension is the final execution granularity. Instead of deleting an entire scored head/channel, the unit's paper score controls how many of its constituent projection weights are retained. The global retained-weight budget is exactly 50% of targeted projection parameters. Per-unit lower/upper sparsity bounds prevent complete unit deletion. Within-unit positions are selected by a deterministic uniform permutation, so no extra importance signal is introduced.
