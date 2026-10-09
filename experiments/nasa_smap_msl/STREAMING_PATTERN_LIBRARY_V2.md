# Streaming Patternly v2 — generator-level admission

This is an experimental rewrite of the incremental Patternly library. It is **not** an evaluated NASA preemption detector.

The old v1 rule compared each current **data window** against past *data-window exemplars* using native LSmash. A small exemplar distance threshold such as 0.02 can make almost every window a new pattern; it does not establish that the inferred stochastic generators differ.

The v2 rule has two separate layers:

1. **Boundary/screening evidence:** actual compiled native LSmash compares an incoming observed window to its immediately previous window, and to frozen library exemplar windows. An above-threshold predecessor difference starts a candidate investigation. `--screen-threshold 0` disables the exemplar prefilter, *not* the switching boundary. A candidate already pending confirmation is reexamined on the next window even if the first boundary signal has subsided.
2. **Generator novelty:** real `zedsuite.genesess.GenESeSS` infers a new candidate PFSA for the current window. The program reads the *native* `%PITILDE` and `%CONNX` matrices (rejecting malformed/ambiguous models). Two PFSAs are compared by the exact stationary output word distributions of length 1 ... `--generator-depth` using average Jensen–Shannon divergence **in bits**. Unlike a comparison between PFSA state rows, this is invariant to the arbitrary renaming of latent states. It is not LSmash between generators, and it is not the entropy-rate KL divergence.

### Automatic null calibration

For each **admitted** native generator `G_i`, simulate `--generator-null-replicates` sequences of the same window length from that very PFSA; independently infer a real GenESeSS model from each sample and compute the distance back to `G_i`. The model's novelty cutoff is the larger of:

- `--min-generator-js`, a conservative nonzero minimum separation (default 0.03 bits)
- The `--generator-null-quantile` quantile (default 0.95) of that model's self-resampling generator distances.

A new PFSA is **eligible only if it exceeds every existing model's calibrated cutoff**. It is then provisionally held until `--confirmations` consecutive candidate windows (default 2) infer mutually similar PFSAs; similarity is determined by `--confirmation-js` (default 0.05 bits). Pending patterns are *not* part of the library, and are not counted as discovered. If the candidate duplicates any existing model, it is assigned to that model and the pending candidate is discarded.

This is an exploratory parametric-bootstrap heuristic. The empirical 0.95 quantile of only 16 null replicates is imprecise, dependent on model assumptions, and **not** a validated false-discovery guarantee under repeated streaming tests. If inference fails for too many bootstrap replicates, the run stops and records that failure: no surrogate generators are substituted. Only one recurrent stationary class per native generator is currently supported.

### Primary command — NASA P-1

From the repo root, with `.venv-genesess` active:

```bash
python experiments/nasa_smap_msl/streaming_pattern_library_v2.py \
  --input results/nasa_concatenated_research_view/P-1.npz \
  --input-kind continuous --fit-length 500 \
  --window 512 --stride 512 \
  --switch-threshold 0.02 --screen-threshold 0 \
  --epsilon 0.2 \
  --generator-depth 4 \
  --generator-null-replicates 16 \
  --generator-null-quantile 0.95 \
  --min-generator-js 0.03 \
  --confirmations 2 --confirmation-js 0.05 \
  --seed 47 \
  --out results/nasa_pattern_library/P-1-v2
```

**Do not overwrite the v1 result directory.** Native inference and self-bootstrap can be slow. For stronger self-null estimates use `--generator-null-replicates 32` or 64; 16 is an initial runtime compromise. Never tune generator JS thresholds to maximize overlap with NASA anomaly annotations. The fit prefix determines quantization; the annotated events are not used for calibration. A recording boundary in the `.npz` terminates adjacency and clears provisional candidates.

### Review outputs

- `windows.csv`: observed LSmash distances, statuses (`candidate_pending`, `generator_duplicate`, `new_pattern`, `matched_switch`, etc.), vector of candidate-to-library generator divergences, and all applicable calibrated cutoffs.
- `library.json`: patterns, native model locations, occupancy/transition estimates, exact generator JS matrix, and the per-generator bootstrap distributions.
- `generator_js.csv`: growing **generator-to-generator** process-level divergence matrix. This matrix is not a window LSmash matrix.
- `library_lsmash.csv`: most recent library **exemplar-window** distances as a separate object.
- `matrices/generator_at_<time>.csv`: immutable snapshot after each newly admitted generator.
- `models/`: admitted native PFSAs; `candidates/`: inferred candidates and self-bootstrap PFSAs for audit.

Visualize using the existing plot script (some panels continue to display exemplar LSmash instead of the new generator JS matrix):

```bash
python experiments/nasa_smap_msl/plot_streaming_pattern_library.py \
  --result results/nasa_pattern_library/P-1-v2 \
  --input results/nasa_concatenated_research_view/P-1.npz
```

### Controlled tuning strategy

Keep the first run with window/stride 512, epsilon 0.2 and generator-depth 4. Inspect rejected duplicates and `generator_null_details` before adjusting thresholds. A reasonable scientific sensitivity analysis changes `--epsilon` across 0.10, 0.20, 0.30 and changes block depth between 3 and 5, using completely different output directories. Prefer stable recovered generators and transition structure across adjacent parameter settings rather than a specific target number of patterns. Lowering the JS floor below the self-null cutoff has no effect. Increasing the threshold or confirmation count makes admission more conservative.

For statistical validity, later test synthetic known-generator switching with labels **reserved for evaluation**, then real NASA telemetry, with false-switch and detection-delay curves. Native integration cannot be validated in an environment without compiled `lsmash` and `zedsuite`.