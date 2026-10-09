# Pre-registered NASA GenESeSS temporal-dependency replication

Protocol frozen on October 9, 2026, before evaluating any held-out
channel's test annotations.

Question: Does native GenESeSS inference of more than one
predictive state detect independent spacecraft-telemetry anomalies
better than a one-state marginal-only model when both use exactly
the same quantization, windows, and normal-data alarm budget?

## Independent evaluation unit

Enumerate all 82 NASA SMAP/MSL NPZ training-series filenames.
Exclude the 12 channels already evaluated in the pilot:
P-1, E-1, E-10, G-7, P-4, T-1, F-7, C-1, T-13, T-8, S-2, M-1.
Exclude unannotated T-10. This leaves 69 new channels, but not
new recordings of P-1. Correlated spacecraft telemetry means channels
are not necessarily statistically independent. No test labels are
used to select channels. List every channel including failures.

## Models and settings

- Normal training first 70%: learn all parameters.
- Four-symbol training-quantile discretization; collapse tied edges.
- Native zedsuite 0.0.7 GenESeSS eps=0.10 with all training symbols
  passed as one dataframe row, and no epsilon sweep.
- One-state control: same symbolization, same train symbols,
  add-half-smoothed empirical marginal probabilities.
- Native zedsuite.zutil.Llk scores 128-symbol windows, stride 32.
  The marginal control scores identical windows by average negative
  log probability; both scores assigned at window end and carried
  forward causally. Score units need not match for quantile thresholds.
- Remaining 30% normal training is calibration for both independently.
  Primary normal quantile=0.95 (5% nominal alarm occupancy).
  Secondary quantiles=0.99 (1%), 0.98 (2%), 0.90 (10%).
  Calibrated normal alarm occupancy does not guarantee a particular
  realized test false alarm fraction.
- Test measurements are scored BEFORE annotations are loaded.
  Labels must never determine epsilon, model, or threshold.
- Each native GenESeSS trial isolated in a child process.
  Never substitute another detector when native inference fails.
- Require at least 3 distinct completed scoring windows in
  normal calibration, otherwise mark channel insufficient.

## Primary analyses

- Report all eligible channel-by-channel results, including models
  with one state and failures, but interpret temporal structure for
  state count >=2 (and separate exact-3-state subgroup).
- Primary paired event-level new alarm onset recall for a 5% nominal
  normal-data false alarm budget. Report realized non-event alarm
  fraction and new false-alarm onsets/1000 points for both methods.
- Report paired event outcomes: GenESeSS only, marginal only, both,
  neither. Only a new threshold crossing within the event counts.
- Sensitivity at 1%, 2% and 10% nominal calibration occupancy.
- Archive fitted PFSAs, scores, thresholds, per-event outcomes,
  per-channel results, explicit failure reasons and summary plots.

This is independent-channel replication, NOT new P-1 telemetry.
Previous P-1 epsilon=.10 finding is exploratory and was used to fix
the method before these channels are evaluated. Do not describe
fixed-epsilon GenESeSS as a forced three-state model on new channels:
state count is inferred from the normal fit and may differ.
