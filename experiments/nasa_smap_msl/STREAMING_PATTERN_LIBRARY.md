# Streaming Patternly — native PFSA library and switching graph (prototype v0.1)

This is a **new implementation** of Patternly's growing-PFSA-library idea. It does not modify the legacy `patternly/detection.py`. It is an exploratory online regime-discovery model, **not** a validated NASA anomaly-prediction system.

The script `streaming_pattern_library.py` ingests a symbol stream or a continuous univariate series; a continuous series is quantized using the **first `--fit-length` observations only**, and no scoring is performed before that fitting prefix ends. At each completed window, it performs one genuine compiled LSmash comparison of the current window to all previously discovered fixed exemplars, optionally including the previous window. It infers a genuine GenESeSS PFSA whenever it admits a new pattern. No custom PFSA substitute or surrogate native distance exists in the application.

## Mathematical scheme

Let the finite-alphabet stream be `x[0], ..., x[t]`. Let `W_t` be a completed window of length `L`, evaluated at time `t` (its **last** observation). Assume the currently discovered patterns are `P_0, ..., P_(K-1)`. Each pattern stores an immutable example sequence `E_i`, its original GenESeSS inferred generator `G_i`, its discovery time, and its count.

At each window end, native LSmash produces a *fresh* symmetric sequence-distance matrix on `[E_0, ..., E_(K-1), W_t, W_previous]` (omit the previous window across dataset joins). The first `K × K` principal submatrix is the **current library distance matrix**, `D_t(i,j)`. The distances `d_t(i) = D(E_i,W_t)` assign a candidate library match `i* = argmin_i d_t(i)`. The score `b_t = D(W_previous,W_t)` provides *switching evidence*.

The rules are intentionally simple and visible:

- **Seed:** On the first scored window, infer `G_0` using actual GenESeSS; admit pattern `P_0` only if inference succeeded.
- **Novelty admission:** If `min_i d_t(i) > tau_new`, and the preceding window is absent (new segment) or `b_t >= tau_switch`, run native GenESeSS on `W_t` and admit a new immutable pattern.
- **Existing-pattern switch:** If `i* != previous_pattern`, switch to `i*` only when a predecessor is available and `b_t >= tau_switch`. Otherwise keep the previous assignment. The first window of a new segment can match any existing pattern.
- **Unresolved novelty:** If the window is far from all exemplars but there is insufficient boundary evidence, retain a prior label (if available) and mark the row **unresolved** rather than fabricate a new PFSA.
- **Inference failure:** A failed GenESeSS call produces a marked unmatched window (ID `-1`). It **never** adds a fictitious generator. A fatal LSmash process error stops the run, retaining partial outputs.

The two thresholds are in *native LSmash distance units* and are user-selected, not calibrated in this version. They are **not** statistical significance thresholds. LSmash is used here to compare *observed exemplar sequences*, not to compute a generator-to-generator metric. Triangle inequality is not assumed. A new window can change default native projector behavior; the recorded library-matrix drift diagnostic measures that sensitivity.

At each assigned window, the estimator maintains occupancy counts `N_i` and within-segment transition counts `C_ij`. After `N` assigned windows and with symmetric Dirichlet pseudocount `alpha`, the reported probabilities are

```text
P(pattern i)            = (N_i + alpha) / (N + alpha*K)
P(next=j | current=i)   = (C_ij + alpha) / (sum_j C_ij + alpha*K)
```

Thus `P(pattern i)` is an estimate of the frequency of **window assignments**, not a claim to have identified the unique latent switching process. With `stride < window`, adjacent windows overlap, inflating apparent persistence. Use `--stride` equal to `--window` initially to make transition edges physically interpretable. No transition crosses an unverified recording join. The original NASA anomaly annotations are **not read** by this script.

## Native prerequisites

Use the project's functioning **Python 3.9** GenESeSS environment (`zedsuite==0.0.7`) and the **experimental native LSmash** branch `experiment/genesess-projectors` (pinned earlier in the NASA work to commit `66960a98a012f63face06ce70145b7ba3a386f7c`). Required imports are `lsmash`, `zedsuite.genesess.GenESeSS`, `numpy`, and `pandas`. The `--seed` option seeds only synthetic demo generation; it does not guarantee determinism of native C++ LSmash projectors. The native methods run in individual child processes to isolate known C++ memory crashes. The script is unable to run genuine inference without those native packages.

A check in your working environment:

```bash
python -c 'import lsmash; from zedsuite.genesess import GenESeSS; print("native engines available")'
```

## Run on the NASA concatenated P-1 channel

From your Patternly repository directory, after placing the Python file in `experiments/nasa_smap_msl/`:

```bash
python experiments/nasa_smap_msl/streaming_pattern_library.py \
  --input results/nasa_concatenated_research_view/P-1.npz \
  --npz-key value --input-kind continuous --fit-length 1500 \
  --window 512 --stride 512 \
  --novelty-threshold 0.12 --switch-threshold 0.08 \
  --epsilon 0.1 \
  --out results/nasa_pattern_library/P-1
```

The thresholds `0.12` and `0.08` are **illustrative, uncalibrated starting values**, not empirically validated choices. Audit their sensitivity with actual score distributions. Set `--window 212 --stride 212` for smaller windows, but GenESeSS models inferred from very short windows can be unstable. Larger windows reduce temporal resolution. Overlapping windows are supported (e.g. `--stride 5`) but will make the transition matrix mostly self-transitions.

To try a synthetic, labeled-for-validation-only three-regime binary stream (these hidden regime boundaries are not used for decisions):

```bash
python experiments/nasa_smap_msl/streaming_pattern_library.py \
  --demo --window 512 --stride 512 \
  --novelty-threshold 0.12 --switch-threshold 0.08 \
  --out results/nasa_pattern_library/demo
```

The synthetic generator provides a simple smoke-test signal but **does not prove** that LSmash separates those regimes or that the chosen thresholds work. Inspect the output.

## Output schema

| File | Contents |
|---|---|
| `configuration.json` | Exact input, quantizer, thresholds, native backend provenance and run settings |
| `windows.csv` | Causal window timestamps, assigned ID, native nearest/exemplar distances, predecessor LSmash boundary evidence, library size, and *running* occupancy / outgoing-transition posteriors |
| `library.json` | Discovered PFSA metadata, model file paths, final occurrence probabilities, transition counts and transition probabilities, and latest library-distance matrix |
| `edges.csv` | Directed weighted transition graph including self-edges; weights are Dirichlet-smoothed |
| `library_lsmash.csv` | Final **observed-exemplar** pairwise LSmash matrix |
| `matrices/at_<end>.csv` | Snapshot of the library distance matrix each time a new generator is admitted |
| `models/pattern_XXXX.pfsa` | Genuine GenESeSS-inferred source PFSA files |
| `exemplars.npy` | Immutable representative quantized windows of admitted patterns |
| `_native_work/` | Child-worker temporary inputs and results, retained for debugging |

## Scope and limitations

1. **No likelihood-based generator routing yet.** Native GenESeSS is used to infer and archive each pattern's actual PFSA; assignments and boundary decisions use compiled native LSmash of exemplar windows. This distinction is essential and will be a natural next extension, using `zedsuite.zutil.Llk` with per-model score calibration.
2. **No calibrated false-switch rate.** Novelty and switch thresholds are fixed user inputs. A clean historical calibration period, causal thresholds, and event-level false alarms must be added before serious anomaly claims.
3. **Library snapshots are not guaranteed to be stable across calls.** Native LSmash's default projector ensemble may vary with the input sequence set or RNG, even for unchanged exemplars. `library_matrix_drift` in `windows.csv` is a diagnostic, not a correction. Perform identical-input repeatability and causal-prefix tests with the **actual native engine** before using the resulting graph scientifically.
4. **One fixed exemplar per generator.** No retraining, merging, splitting, prototype updates, model selection, or retirement is implemented. Each PFSA inference is made only from the window being admitted; sparse or constant windows may fail.
5. **No online prediction is claimed.** This model discovers and describes switching patterns from completed windows. To test *preemption*, independently compare score/alarm timestamps with future-onset labels without using those labels in fitting, quantization, thresholds, or discovery.
6. **Process cost.** A fresh `(K+1 or K+2) × (K+1 or K+2)` native LSmash matrix is recomputed each scored window in an isolated process. This is correct for a first auditable prototype, but not yet optimized for large streams or thousands of patterns.

## Unit tests

```bash
python -m pip install pytest numpy
cd /path/to/this/folder
pytest -q test_streaming_pattern_library.py
```

The included tests inject an explicitly labeled **fake backend only into the Python orchestration** to verify state transitions and prefix dependence; they never represent scientific performance or native-engine equivalence. The application never selects that fake backend.

### Next experiments

Run the native binary on short P-1, T-1 and F-7 streams; verify identical-input matrix repeatability, actual PFSA creation, occupancy and switching graphs, and boundary alignment with held-out anomaly labels. Then add native `Llk` calibration for window-to-generator assignment, hysteresis/persistence admission, and a causal false-positive-controlled boundary detector.