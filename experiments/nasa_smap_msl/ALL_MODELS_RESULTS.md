# NASA SMAP/MSL: all-model GenESeSS evaluation (October 9, 2026)

## Reproducible inputs and exact code

- Source: experiments/nasa_smap_msl/genesess_all_models.py
- Successful GitHub Actions run: https://github.com/zeroknowledgediscovery/patternly/actions/runs/37886699061
- Result artifact: nasa-genesess-all-models-heldout-and-threshold-curves
- Original 14-epsilon inference artifact: run 37880764127
- Boundary refinement artifact: run 37881153073
- Four original baseline model scores: run 37879203458

The generator models are **native zedsuite.GenESeSS** PFSAs, not a surrogate
Markov estimator. The evaluation uses native zedsuite.zutil.Llk scoring;
no model is refitted, and no test label is used for inference, model
selection, or threshold calibration. Each native scoring attempt is
isolated in a Python 3.9 subprocess.

## Complete native-score audit

- 332 trained PFSA model/quantizer/epsilon combinations attempted.
- 270 valid likelihood score series.
- 62 native score failures, all from channel G-7 (31 binary and 31
  four-symbol models): native Llk returned nonfinite validation windows.
- Valid inference is not the same as a scorable predictor; do not treat
  G-7 as an ordinary one-state non-detection.

## Split protocol

- Normal train 0–70%: original native PFSA inference.
- Approximately 70–85%: mean normal held-out Llk score for choosing
  the best predictor, separately for each quantized alphabet.
- Approximately 85–100%: independent quantile threshold calibration.
- Test: retrospective anomaly onset recall, number of false-alarm onsets,
  and proportion of non-event observations under alarm.

Normal calibration quantiles: 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995,
and 0.999. Report observed test false-alarm fraction for every point,
rather than supposing that a 0.995 calibration quantile fixes the
test false alarm rate.

Selection rules (strictly training-only):
- Minimum nontrivial model: smallest states >=2, tie by larger epsilon.
- Best heldout Llk: smallest mean Llk among models with >=2 states.
- More complex smaller epsilon: largest state count among candidates
  with epsilon smaller than the min-nontrivial candidate.

One-state models are evaluated as marginal-only controls but are
excluded from those three named selection rules.

## Selection results at normal threshold quantile 0.995

Over the eligible channels only (NOT all 12 channels / 26 anomalies):

Selection / alphabet | Selected channels | Events with new-onset detection | Mean non-event time in alarm
:--|--:|:--|--:
Min nontrivial, binary | 5 | 3/11 | 0.14%
Best heldout Llk, binary | 5 | 3/11 | 0.54%
More complex, binary | 5 | 4/11 | 5.95%
Min nontrivial, four symbols | 4 | 6/10 | 34.76%
Best heldout Llk, four symbols | 4 | 3/10 | 31.74%
More complex, four symbols | 4 | 3/10 | 29.80%

These are NOT matched-FPR operating points. Four-symbol apparent recall
is accompanied by unacceptable alarm occupancy in several channels:
T-1 alarms for roughly 99% of non-event test data for all three selected
four-symbol models; P-1 four-symbol minimum-two-state model alarms for
24% of non-event test data. Do not treat 6/10 as a detector advantage.

Among the five binary-eligible channels, the min-nontrivial model
detects one event each in C-1, M-1 and T-1, with no non-event alarm
occupancy at the chosen threshold; it detects no event on P-1 and F-7.
Sparse event counts and different calibration schemes preclude firm
generalization claims.

## P-1: focused result and marginal-only negative control

Three annotated anomaly intervals: [2149,2349], [3539,3779],
[4536,4844] (inclusive indices in original test data).

At normal threshold calibration quantile 0.995:

Model / quantization | Inferred states | New-onset detections | Non-event alarm occupancy
:--|--:|:--|--:
Original z-score | NA | 0/3 | 0.27%
Original CUSUM | NA | 0/3 | 0.70%
Original matrix-profile approximation | NA | 0/3 | 0%
Original fixed-order Markov ("pfsa" pilot) | NA | 0/3 | 0.52%
GenESeSS eps=.70, binary | 2 | 0/3 | 0%
GenESeSS eps=.03, binary (best predictive among >=2 states) | 12 | 0/3 | 1.26%
GenESeSS eps=.70, four-symbol (smallest AND best-predictive >=2 states) | 2 | 1/3 | 24.07%
GenESeSS eps=.30, four-symbol | 12 | 1/3 | 2.52%
GenESeSS eps=.10, four-symbol | 3 | **2/3** | **0%**
GenESeSS eps=.001, four-symbol | 1 | **2/3** | 0.22%

The epsilon=.10, three-state option was found by *posthoc examination of
test evaluation across epsilon* and is therefore exploratory and
test-label-selected, not an unbiased prospective model recommendation.
Its two new alarm onsets occur 154 and 140 observations into the first
two anomaly intervals. The one-state control detects these same two
intervals with new alarm onsets 90 and 44 observations after onset,
while generating 0.22% non-event alarm exposure.

This contradicts any claim that the P-1 detector must use temporal
predictive states: a marginal-only generator performs similarly or
better for these events. At the same time, the three-state model's
zero measured non-event alarm exposure is potentially useful if
independently replicated.

P-1 four-symbol eps=.10 has 2/3 event-onset recall and 0% non-event
test occupancy at calibration quantiles 0.95 through 0.999. At lower
quantiles, false exposure rises; see its saved curve. The inference
does not show improved aggregate AUC over baselines (AUC was not
estimated). There are only three annotated P-1 events, and the
repeated epsilon evaluations are not a separate heldout test.

The earlier pilot using full-30%-normal-data threshold calibration
reported 2/3 detections for the eps=.70 P-1 model. This revised,
stricter holdout split detects only 1/3 while alarming for 24.07% of
non-event test observations. The original signal is not robust to
calibration protocol; do not cite it as a successful detector without
this qualification.

## Scientific conclusions and next controls

1. Better held-out native log-likelihood does not reliably imply better
   anomaly-onset detection. P-1's best-predictive four-symbol >=2-state
   model is eps=.70 (2 states), but the strongest posthoc detected
   ε=.10 model (3 states) has a worse normal held-out Llk score.
2. Even genuine multi-state inference does not by itself demonstrate
   a temporal-dependency anomaly signal when a one-state marginal-only
   model detects the same events.
3. The quantizer can collapse an entire channel's normal-fit stream to
   one symbol; epsilon cannot rescue this representation.
4. PFSAs are not yet superior on the 12-channel benchmark at controlled
   false-alarm exposure. The few potentially useful channels require
   independent pre-registered testing and/or a second test split.
5. Test-score labeling must not be used to select epsilon when
   reporting prospective accuracy. Prespecify a selection metric on
   normal heldout data, train/select/calibrate, and evaluate exactly
   once on a new untouched test partition.
6. To test temporal mechanisms, compare against the one-state model
   at matched realized false-alarm fraction and compare conditional
   vs marginal log-score changes, ideally with external commands.

## Artifacts

The successful CI artifact contains:
- all_model_scores.csv: every attempted model, states, normal Llk, failures
- selection_comparison.csv: three predeclared selection rules
- baseline_comparison.csv: original four methods and selections
- curves/*.csv: eight calibration thresholds per successfully scored model
- scores/*.npz: raw native validation/test score arrays
- figures/*.png: epsilon/state/heldout LLK and alarm-recall curves
- run.json: complete configuration and counts
