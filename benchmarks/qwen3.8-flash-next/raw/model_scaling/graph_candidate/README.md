# Adaptive depth-four graph candidate

The completed frozen candidate adds eligible B7/B8, Q5 target graphs. analysis/stage_comparison/comparison.json compares it directly with the prior d2c85e serving stage. Ancestor-validation reports are retained only for audit. The prior baseline archive remains unchanged.

analysis/homogeneous.summary.json contains separate same-prompt finite-wave controls; they are not added to the ordinary serving rates. Command counters include warmups, and standard C6/C8 counters are combined. Raw responses, smokes, process/memory snapshots and commands remain under run/.

scripts/prior_checkpoint_cache_release.json records release of clean cache from the completed different Qwen3.5 checkpoint before model loading. It did not change checkpoint contents and is outside timing.

Tokenizer copies and other large artifacts remain external with hashes. Model weights and executables are not copied or rehashed here. The run's original manifests preserve original layouts; SHA256SUMS.json covers every archived file except itself.
