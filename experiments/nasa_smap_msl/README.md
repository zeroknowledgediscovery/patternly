# Real NASA SMAP/MSL 12-channel pilot

The dataset is already under `datasets/nasa_smap_msl/` on branch `data/nasa-smap-msl-telemetry`. No external download is needed.

## Install and run (Python 3.10+)

```bash
git fetch origin
git switch data/nasa-smap-msl-telemetry
python3 -m venv .venv-nasa
source .venv-nasa/bin/activate
python -m pip install -r experiments/nasa_smap_msl/requirements.txt

# Smoke test on 3 channels
python experiments/nasa_smap_msl/pilot.py \
    --channels P-1,E-1,F-7 \
    --out results/nasa_smoke \
    --fail-on-error

# Full 12-channel pilot, modern baselines + adaptive PFSA
python experiments/nasa_smap_msl/pilot.py \
    --out results/nasa_smap_msl_pilot \
    --fail-on-error
```

Default channels (six SMAP and six MSL): `P-1,E-1,E-10,G-7,P-4,T-1,F-7,C-1,T-13,T-8,S-2,M-1`.

## Methods

- `zscore`: train-normalized absolute deviation.
- `cusum`: two-sided normalized CUSUM (resets at start of validation/test).
- `matrix_profile`: nearest-neighbor subsequence shape distances to a 256-window normal reference bank; requires `scikit-learn`. This is an approximate **normal-reference matrix-profile baseline**, not STUMPY's exact self-join implementation.
- `pfsa`: train-only quantized Markov/PFSA generator with context depth selected on normal validation and a complexity penalty, causal rolling negative log-likelihood.
- `lsmash`: **optional**, uses the real compiled C++ `lsmash.from_sequences` four-projector distance matrix and a minimum-distance normal-reference score; not a substitute KL calculation.

Every method is trained on the first 70% of a channel's normal training series; alert threshold uses the last 30% of normal training values at quantile 0.995. The test series and annotation file are never used for model or threshold fitting. The program accesses annotations only after generating detection scores.

The native LSmash extension needs Boost and GSL headers/libraries, plus a C++ toolchain:

```bash
# Fedora:
sudo dnf install -y gcc-c++ boost-devel gsl-devel
# Ubuntu:
# sudo apt-get install -y g++ libboost-all-dev libgsl-dev

python -m pip install "lsmash @ git+https://github.com/zeroknowledgediscovery/lsmash.git"

python experiments/nasa_smap_msl/pilot.py \
    --methods zscore,cusum,matrix_profile,pfsa,lsmash \
    --out results/nasa_with_native_lsmash \
    --fail-on-error
```

The native method is slower than the others. For initial validation, restrict `--channels P-1,E-1,F-7`.

## Outputs

In the selected `--out` directory:

- `summary.csv`: aggregated recall, false alerts and delay.
- `per_channel.csv`: each detector/channel event scores and threshold.
- `plots/*.png`: raw telemetry and chronological detector scores with annotated intervals shaded.
- `scores/*.npz`: full per-symbol scores and fitted thresholds for independent analysis.
- `run.json`: exact channels, hyperparameters, selected PFSA order, timings and exceptions.

False alarms count distinct alert episodes entirely outside annotated anomaly intervals, normalized to 1,000 test observations. Recall counts annotated anomaly events with at least one alert inside the interval. Delays are measured from event start to first alarm *inside* the event. Contextual and point annotations are not conflated with known PFSA change points; this is an anomaly-detection benchmark, not a ground-truth generator-switch benchmark.

The optional original 2022 Patternly package requires a separate Python 3.9 environment (and has demonstrated native crashes on synthetic sources). Do not label the modern `pfsa` approach as original Patternly.

## CI

The `nasa-telemetry-pilot.yml` workflow executes the smoke and complete 12-channel run against the checked-in real telemetry arrays, and uploads all plots/scores/tables as a GitHub Actions artifact. Raw generated results are intentionally not committed to Git.
