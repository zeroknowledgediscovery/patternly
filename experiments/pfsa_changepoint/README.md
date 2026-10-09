# PFSA Change-Point Benchmark (October 2026)

Reproducible comparison between conventional change-point inference, two data-adaptive de Bruijn PFSAs, and the original Patternly streaming implementation.

## Source construction

For each seed, a stream of 60,000 or 180,000 bits switches from a depth-10 PFSA A to depth-10 PFSA B at fraction 0.53 of its length without a history reset. Source pair logits are matched so that A and B have identical stationary frequencies of blocks of length <=10 and identical entropy rates. Difference strength is 0.17 in conditional logit units.

## Methods

- `benchmark.py`: unknown-PFSA depth-10 maximum-likelihood change scan using the entire stream, and self-trained PFSAs estimated independently from the first and last 20% (guaranteed pure by the permitted change-point interval), followed by log-likelihood CUSUM. It evaluates symbolwise and binned windows of 250, 500, 1000, 2000, 4000, and 8000 symbols.
- `legacy_probe.py`: actual `patternly.detection.StreamingDetection` with KMeans/LSmash/GenESeSS from the 2022 distribution, run on a 60k pilot with 4000-symbol windows and two requested regimes. Errors (including dependency or execution failures) are recorded as failures, not silently replaced.
- [Separate local audit bundle](https://github.com/zeroknowledgediscovery/patternly/tree/experiment/pfsa-changepoint-2026/experiments/pfsa_changepoint/): full modern surrogates for window clustering and second-level sequence-of-regimes detection were tested independently in this ChatGPT session. Those are **not** historical Patternly and should not be labeled as such.

## Run

```bash
python -m pip install numpy scipy numba pandas scikit-learn
python experiments/pfsa_changepoint/benchmark.py --seeds 5
```

GitHub Actions installs the historical Patternly separately and tests `legacy_probe.py` under a wall-clock budget. Results will appear under `experiments/pfsa_changepoint/results/`.

## Evaluation caveats

The correctly specified depth-10 likelihood scanner has access to the entire sequence but no source probabilities. The adaptive two-model approach uses the first and last 20% to fit two generators; this assumes a change occurs in the central 60%. Neither method is an information-theoretically optimal detector. The original Patternly implementation is an older research prototype and may fail on modern Python environments or on long-context PFSAs.

Failure of window clustering does not imply failure of Patternly's PFSA modeling layer, and failure of the historical package to install is **not** evidence of scientific inadequacy.


## Completed results

Five seeds at each of N=60k and N=180k, contrast delta=0.17:

| method | N=60k median error | N=180k median error |
|---|---:|---:|
| Correct-order Markov likelihood scan | 210 | 310 |
| Two tail-trained order-10 PFSA contrast | 4,672 | 420 |
| Original Patternly likelihood on 500-symbol windows | 15,200 | 24,100 |
| Original Patternly likelihood on 1000-symbol windows | 14,300 (4/5 valid) | 37,400 |

The historical Patternly sweep was executed with Python 3.9 / zedsuite 0.0.7 binary wheel. Of 50 combinations, 42 yielded a valid estimate, three crashed in native code, and five (N=60k, window=8000) had too few complete windows for the tail-based orientation and segmentation used here. Other window sizes and full success/failure counts are recorded in `results/legacy_isolated_summary.json`. These outcomes are preliminary five-seed synthetic benchmarks, not a general ranking of stochastic changepoint algorithms.

GitHub Actions: `patternly-legacy-isolated.yml` (historical library), `patternly-pfsa-cp-2026.yml` (modern baseline). No changes were made to Patternly's historical main branch.
