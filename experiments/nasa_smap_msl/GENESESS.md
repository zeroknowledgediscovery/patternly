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


## Reproduced results, October 2026

Executed with actual zedsuite 0.0.7 GenESeSS + Llk on 12 NASA
channels in GitHub Actions. Native run:
https://github.com/zeroknowledgediscovery/patternly/actions/runs/37880202728

Both eps=0.05 and eps=0.10 yielded **identical aggregate metrics**
at the selected thresholds, but did not always produce identical
automata.

Quantization | Valid channels | Annotated events in valid channels | Event hits | New-alarm events
:--|--:|--:|--:|--:
Four-level quartiles | 11/12 | 23 | 6 (26.1%) | 5 (21.7%)
Binary median | 11/12 | 23 | 4 (17.4%) | 3 (13.0%)

The one invalid channel in each variant is G-7: native Llk
returned nonfinite scores on normal calibration windows. The code reports
this error explicitly and does not silently substitute another algorithm.
This reduces the event denominator from 26 to 23. Do not compare pooled
recall to the 26-event baseline without accounting for exclusion.

Median inferred state count was **1 state**, for both eps values and
both quantization schemes, among the successful channels. Some channels
produce genuinely nontrivial models: under binary quantization, P-1 has
6 states at 0.05 vs 5 at 0.10, and F-7 has 11 vs 4. However neither
detected annotated events at the current calibration operating point.

These are fixed-threshold pilot outcomes, not a final model comparison
at matched false-alarm rates. In particular, one-state automata on
near-constant channels indicate that quantization and data richness,
not just epsilon, constrain the experiment.

## Automated epsilon sweep for two or more predictive states

Run the native GenESeSS search over 14 epsilons on each of 12 channels with both binary and four-level training-only quantization:

    python experiments/nasa_smap_msl/genesess_epsilon_sweep.py --alphabets 2,4 --min-states 2 --out results/nasa_genesess_epsilon_sweep

The sweep tests epsilon = 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5 and 0.7. Each epsilon uses the genuine zedsuite GenESeSS implementation on the normal training fit segment only, isolated from other runs. A new PFSA is inferred at each epsilon and state count is read from its probability morph matrix. The method picks the smallest eligible state count (>=2), breaking ties toward larger epsilon. It scores validation/test data with native Llk only after selection, preventing test-label leakage.

Results:

- epsilon_grid.csv: all attempted epsilons, actual epsilon, state count, failure reason, training entropy, and model location
- selection_summary.csv: each channel/quantizer success status, selected epsilon and state count
- selected.csv: selected models and their scores/failures
- models/: all successful inferred generators, including ones with one state
- scores/ and plots/: selected models only

If a training channel has only one distinct symbol after quantization, the experiment declares it degenerate rather than manufacturing two PFSA states. If every epsilon fails to yield 2+ states, it reports no eligible model. The selection criterion targets a minimum nontrivial state count, not best anomaly-detection performance.

GitHub Actions: https://github.com/zeroknowledgediscovery/patternly/actions/workflows/nasa-genesess-two-state-sweep.yml


## At-least-two-state inference results (October 2026)

Reproduced successful GitHub runs:
- Grid: https://github.com/zeroknowledgediscovery/patternly/actions/runs/37880764127
- Boundary refinement: https://github.com/zeroknowledgediscovery/patternly/actions/runs/37881153073

The 14-epsilon training-only grid (0.001 to 0.7) produced 9 scored
models with >=2 states out of 24 (12 channels times 2 quantizers).
For most eligible channels a *high* epsilon was selected because the
criterion explicitly minimizes state count and breaks ties toward
larger epsilon. This is not evidence that high epsilon is scientifically
optimal for anomaly detection.

Channel | Binary eps / states | 4-symbol eps / states
:--|:--|:--
P-1 | 0.7 / 2 | 0.7 / 2
T-1 | 0.3 / 2 | 0.5 / 2
F-7 | 0.7 / 2 | 0.7 / 2
C-1 | 0.7 / 2 | 0.002 / 12
M-1 | 0.7 / 2 | no eligible model
G-7 | no eligible model | no eligible model
T-13 | no eligible model | no eligible model

The five remaining channels E-1, E-10, P-4, T-8 and S-2 had
only one distinct symbol in the normal-fit data for either quantizer.
No meaningful multi-state inference is possible with that representation.

A follow-up sweep explored 17 more epsilons between 1e-5 and 0.99
for C-1, G-7, T-13 and M-1. C-1 under four-symbol quantization
yielded a **6-state model at eps=0.0035**, reducing the smallest
nontrivial observed state count from 12. No 2+ state models emerged
for G-7 or T-13 with either quantizer, or M-1 with four symbols.

The native GenESeSS runs succeeded; every selected initial-grid
model was also successfully scored with native Llk. These results
are about state structure, not necessarily superior detection.
Detailed model files, inference error, state counts, model-score
outputs and test-onset metrics are retained in both workflow artifacts.

Epsilon-to-state count is often **nonmonotonic**: e.g. P-1 binary
at 0.001 has 82 states; at 0.002 it collapses to one state; at
0.005 it yields 13. Do not infer epsilon directionality from a
single pair of runs.
