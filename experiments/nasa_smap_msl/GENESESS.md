# Native GenESeSS epsilon experiment — NASA SMAP/MSL

This benchmark calls the **actual native zedsuite.GenESeSS** for epsilon
0.05 and 0.10, and the actual zedsuite.zutil.Llk to score new telemetry.
It does not substitute the fixed-order Markov baseline currently named
"pfsa" in the original pilot script.

## Installation

The historical zedsuite 0.0.7 binary is available for **Python 3.9 on
Linux x86-64**, not Python 3.11. Create a separate environment.

    git fetch origin
    git switch data/nasa-smap-msl-telemetry
    git pull --ff-only

    python3.9 -m venv .venv-genesess
    source .venv-genesess/bin/activate
    python -m pip install "pip==24.0"
    python -m pip install "numpy==1.23.5" "scipy==1.10.1" "pandas==1.5.3" "scikit-learn==1.2.2" "matplotlib==3.7.5" dill
    python -m pip install --only-binary=:all: "zedsuite==0.0.7"

If python3.9 is unavailable, use your system package manager or conda to
obtain a Python 3.9 runtime. Avoid installing into the existing Python
3.11 NASA environment.

## Run genuine GenESeSS (both epsilon values)

Three-channel binary smoke test:

    python experiments/nasa_smap_msl/genesess_pilot.py --channels P-1,E-1,F-7 --eps-values 0.05,0.1 --alphabet 2 --out results/nasa_genesess_smoke

Full 12-channel four-level experiment:

    python experiments/nasa_smap_msl/genesess_pilot.py --eps-values 0.05,0.1 --alphabet 4 --window 128 --stride 32 --out results/nasa_genesess_12ch

## Procedure

- Quantization boundaries are derived only from the first 70% of each
  normal training stream; ties may reduce the realized alphabet size.
- Native GenESeSS fits a PFSA to the complete quantized 70% training
  stream with the *explicit requested epsilon*.
- Native Llk scores 128-symbol observation windows with stride 32. Scores
  are assigned to the last index of their windows, and then carried
  forward until the next completed window.
- Thresholds are calibrated from the remaining 30% of normal training
  data, at quantile 0.995. Test labels are used only after scoring.
- Per-channel results report event-hit recall, onset-based recall,
  non-event time spent in alarm, learned state counts, actual epsilon,
  successful trials, and native failures.
- Every channel and epsilon is isolated in a subprocess because
  historical zedsuite can segfault.

These outputs are **not** model likelihood scores at every time step;
they are window-level Llk scores. The detector uses one threshold for
each model. A fixed threshold quantile is not matched false alarm
calibration between methods.

## Outputs

- per_channel.csv and summary.csv: successes, failures and event metrics
- scores/*.npz: full alert scores and fitted threshold
- plots/*.png: annotated channel plots
- models/<channel>/eps_0.05/inferred.pfsa: actual inferred generator
- trials.jsonl and run.json: run configuration, statuses, timings

GitHub Actions workflow: .github/workflows/nasa-genesess-epsilon.yml
runs both pilot configurations and uploads result artifacts.
