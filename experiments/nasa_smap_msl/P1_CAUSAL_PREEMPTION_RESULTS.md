# P-1 causal advance-warning audit, no look-ahead

Date: 2026-10-09. Successful native GitHub Actions run:
https://github.com/zeroknowledgediscovery/patternly/actions/runs/37965267604

**Question:** Can we raise an alarm *before* a NASA-annotated anomaly
begins, rather than assign a high distance to a window containing an
already-occurring anomaly?

## NON-NEGOTIABLE: temporal causality

- Every test score is stamped at the **end** of its completed
  212-observation window (5-observation step).
- The 4-symbol quantizer uses P-1's normal training series, not test.
- Fixed LSmash reference bank: 258 212-observation windows from
  first 1500 normal-train observations, stride 5.
- GenESeSS PFSAs are inferred using native zedsuite 0.0.7 from the
  first 1024 points of that normal training prefix, at epsilon
  0.01,0.05,0.1,0.2. Actual state counts: 3,3,3,9.
- Threshold calibration uses 233 complete windows wholly inside
  training observations 1500–2871, independent from the fixed
  reference windows. They are overlapping and correlated.
- Query/test window comparisons are exclusively against
  **fixed pre-test training reference windows**. Nothing about the
  reference population is computed from any test window.
  No future test measurement is needed to score a particular
  completed test window.
- The default LSmash score is the mean actual C++ LSmash distance
  from a query window to each training reference.
- The learned GenESeSS-projector score is the mean actual C++
  llk_distance(S,G) distance from that query window to the
  same training references, using external GenESeSS PFSAs.
- Statistical controls are separately labeled hand-implemented
  four-category frequency JS and sixteen-category bigram JS to
  that **same historical reference bank**. They are not claimed
  to be LSmash, GenESeSS, or LSM.

A *preemptive alarm* is a NEW crossing of the threshold at test
window-end time t strictly within [a-H,a), where a is the NASA
anomaly onset. An alarm that starts at or after a is detection,
not advance warning. An alarm that began before the warning horizon
and persisted does not count as a NEW warning.

Horizon lengths (in observations): 50,100,212,500.
Normal-calibration quantiles: 0.95,0.98,0.99,0.995.
The main reported condition is q=0.99, H=212.

## Confirmed result, q=.99, H=212

Detector | Before onset (3 events) | After onset | Ordinary non-warning, non-anomaly occupancy | Clean alarm onsets per 1000 score times | Pre-event-only AUROC
:--|--:|--:|--:|--:|--:
Native LSmash original default PFSAs | 0/3 | 3/3 | 0.02531 | 1.446 | 0.351
Native LSmash with genuine GenESeSS projectors | 1/3 | 3/3 | 0.03037 | 10.123 | 0.293
Custom marginal frequency JS baseline | 1/3 | 2/3 | 0.03760 | 7.231 | 0.283
Custom first-order bigram frequency JS baseline | 1/3 | 2/3 | 0.04628 | 7.231 | 0.238

Onset by event:

- Event 1 begins at 2149: NONE of the four detectors produces
  a new pre-onset warning in the preceding 212 samples.
- Event 2 begins at 3539: NONE of the four detectors gives
  a new pre-onset warning within 212 observations.
- Event 3 begins at 4536:
  - learned GenESeSS native projector: first new alarm at 4526
    (10 observations of lead time);
  - marginal JS and bigram JS: first new alarm at 4531
    (5 observations of lead time);
  - stock LSmash: no pre-onset alarm.

At H=50 and H=100 the same 1/3 learned-projector preemption
occurs, not additional events.
At H=500, GenESeSS-projector LSmash has 2/3 pre-onset
alarm hits, including an alarm 413 observations before event 1.
That earlier alarm is weak evidence at best: clean false alarm
onsets are common (about 10 per 1000 score time points under
the main condition) and the large 500-observation warning
horizon raises the chance of incidental coincidences.

The 99th-percentile thresholds produced *realized* clean alarm
occupancies between 2.5 and 4.6%, not the nominal 1%.
In the pre-interval classification, AUROCs are BELOW 0.5 for
all four methods at H=212. Thus their score populations do
not show convincing general pre-onset enrichment.

## Scientific interpretation

- Real native LSmash and GenESeSS-assisted distance reliably
  respond DURING the anomalies in this P-1 retrospective case.
- There is only one marginally early alert (10 observations)
  using learned PFSAs, and the simpler JS baselines provide a
  comparable 5-observation warning.
- There is NO evidence here of a reliable, calibrated,
  unique predictive precursor learned by the native models.
- Anomalies need not have a precursor observable in
  this single P-1 sensor. Predicting future onset cannot
  be guaranteed by any method in absence of such a signal.
- Only 3 independent annotated events exist. 1659 sliding
  windows are not 1659 independent cases.
- Test labels enter only evaluation, not distance computation,
  score calibration, or model selection. No posthoc labels
  are used to pick parameters for a claimed model performance.
- Observation indices are not converted to minutes/hours;
  operational lead time would require sampling cadence.
- The former all-TEST-window JS baseline result and its perfect
  AUROC used future observations and is explicitly SUPERSEDED
  for this causal-preemption question. The older frozen
  original-distance experiment remains a separate descriptive
  artifact; it was not an online deployment evaluation.

## Reproduction (real native engines)

Python 3.9, Linux compiled lsmash experimental binding and native
zedsuite==0.0.7 required, as described in
P1_SLIDING_NATIVE_LSMASH.md.

    python experiments/nasa_smap_msl/p1_causal_preemption.py \
      --window 212 --stride 5 \
      --fit-length 1500 --initial-train-length 1024 \
      --epsilons 0.01,0.05,0.1,0.2 \
      --out results/p1_causal_preemption

Outputs include native models, full unmodified native scores,
preemptive_summary.csv, per_event_advance_warnings.csv,
preemptive_causal_scores.png, and complete execution metadata.

For Honeydew: before constructing a preemptive task, identify
multiple independently verified sequences where measurable
pre-event precursors exist, include robust negative controls
and operational false-alarm limits, and make only
strictly-before-onset detection count. Otherwise the
challenge could be scientifically impossible rather than hard.
