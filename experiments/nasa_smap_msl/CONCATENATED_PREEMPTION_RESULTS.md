# Seven-channel NASA concatenated telemetry: causal anomaly preemption audit

Verified final native GitHub Actions execution on 2026-10-09:
https://github.com/zeroknowledgediscovery/patternly/actions/runs/37978860869

**Historical experiment remains frozen separately**:
freeze/p1-native-sliding-20261009.
This report describes a NEW causal multi-channel replication, with
no silently substituted GenESeSS/LSmash implementations.

## Method and time causality

Channels P-1, E-13, E-1, E-10, G-7, F-7, T-1.
Each is one concatenated telemetry stream with original train/test seam
explicitly recorded. The first 1500 observations are historical model
reference; observations [1500,2250) are normal calibration; all later
windows, including both the remainder of the original training segment
and the original test segment, are scored at the END of the window.

Windows are 212 points wide with stride 5 and MUST NOT cross the seam.
The two original recorded segments are not verified to be physically
continuous. Annotation offsets are shifted by the seam. No annotation
influences fitting, scoring, parameter selection or thresholds.
Model/threshold updates are *not* enabled in this replication.

Native algorithms are:
1. original compiled LSmash default random PFSA projectors;
2. original C++ LSmash llk_distance(S,G) with actual GenESeSS PFSAs
   fitted using native zedsuite 0.0.7 from first 1024 observations.
   Four preselected epsilon values .01, .05, .10, .20. Uses separate
   experimental pybind API; not a Python reimplementation.

Custom controls are explicitly **not** native PFSA implementations:
3. four-category marginal-histogram Jensen–Shannon against historical
   reference windows;
4. first-order-symbol-pair 4x4 Jensen–Shannon against those same
   historical reference windows.

For tied train-only quartiles, preserve rare-value contrast by using
right-closed edges ONLY when a collapsed quartile equals historical
minimum (instead of silently mapping the entire remainder to a single
symbol). E-1, E-10 and G-7 yielded only 2 realized symbols; this is
reported explicitly. Native PFSA and distance operations run in
isolated subprocesses: a genuine native memory abort is recorded as a
failure, never replaced by a surrogate result.

Train-calibrated alarm thresholds at 95%, 98%, 99%, 99.5%.
Warning horizons 50, 100, 212, 500 observations.
Preemptive event hit requires a NEW threshold crossing strictly
in [onset-H, onset), with an actual prior below-threshold observation
in the same segment. No alarm during any previous anomaly is counted
as a precursor. Clean occupancy excludes both warning horizons
and labeled anomaly intervals. No arbitrary timescale is inferred
from sample indices.

## Main prespecified result q=.99, horizon=212

Channel | Native default early hits | Native with GenESeSS early hits | Marginal JS early hits | Bigram JS early hits | Annotated events
:--|--:|--:|--:|--:|--:
P-1 | 0 | 1 | 1 | 1 | 3
E-13 | 1 | N/A (native abort) | 3 | 3 | 3
E-1 | 1 | 1 | 1 | 1 | 2
E-10 | 1 | N/A (native abort) | 1 | 1 | 2
G-7 | 0 | N/A (invalid PFSA state) | 0 | 0 | 3
F-7 | 1 | 0 | 1 | 0 | 3
T-1 | 0 | 0 | 0 | 0 | 2
**Total** | **4/18** | **2/10** on 4 channels | **7/18** | **6/18** | **18**

Across available channel/method pairs, unweighted mean clean-window
alarm occupancy:
- LSmash default 25.68%
- native GenESeSS projector LSmash 24.92% (only four channels)
- marginal JS 31.48%
- bigram JS 31.45%.

CAUTION: this is NOT operationally credible advance warning.
Some channel-wise clean alarm occupancies are extreme:
- T-1: 92–96% for ALL FOUR methods.
- E-13: 64% marginal JS and 63% bigram JS.
- F-7: 20–26% for most methods.
Hence nominal 99th percentile historical calibration does NOT
guarantee 1% false alarms under future distribution shift.

Example lead-time findings (observations, not clock time):
- P-1 third episode: native GenESeSS projector 95 early, marginal JS
  and bigram JS 100 early.
- E-1 second episode: native GenESeSS projector and marginal JS both
  89 early; default native 44 early.
- E-13 marginal JS triggers before all 3 episodes, but clean-window
  alarm occupancy is ~64%, making those hits difficult to interpret.
- F-7 third episode: native default and marginal JS both 194 early.

No clear and robust unique native-GenESeSS preemption signal has
been established. The claimed "events" are labeled telemetry anomaly
onsets, not verified engineering failures. The 212-point pre-onset
score windows overlap heavily and independently identified episodes
are few. Do not interpret overlapping windows as independent trials.

## Native failures and incomplete methods

- E-13: genuine native GenESeSS worker exited -6 with heap corruption;
  other methods completed normally. NO learned-projector outcome.
- E-10: genuine native GenESeSS worker exited -6 with invalid memory;
  default native and both controls completed. NO learned-projector outcome.
- G-7: the genuine GenESeSS PFSA has an invalid transition destination
  under strict parser validation; no learned-projector distance
  substituted. Default native and both controls completed.

The precise failure messages appear in per-channel
native_genesess_worker.log and method_status.csv.
Do not drop these channels silently or interpret absence as a negative
early-warning event. For fair complete-case method comparisons use
P-1, E-1, F-7, T-1 only, covering 10 annotated events; even there
GenESeSS projectors have no demonstrated advantage.

## Reproduction

From the checked-out patternly repository root, active .venv-genesess
Python 3.9 environment with the genuine compiled native LSmash
experimental learned-PFSA binding installed:

    git switch data/nasa-smap-msl-telemetry
    git pull --ff-only
    bash experiments/nasa_smap_msl/run_concatenated_preemption.sh

Results under results/nasa_concat_preemption:
- primary_q99_h212.csv (primary per-channel full four-method results)
- all_horizons_thresholds.csv (complete 50/100/212/500 x quantiles)
- method_status.csv (actual native validity/failures)
- method_aggregate.csv (aggregate with DIFFERENT method denominators)
- seven_channel_preemption.png (visual)
- <channel>/per_event_advance_warnings.csv
- <channel>/causal_preemption_scores.png
- <channel>/native_default_worker.log and native_genesess_worker.log
- <channel>/metadata.json, past_only_scores.npz.

The entire verified GitHub Actions execution and all outputs are
available in its archive named
nasa-seven-channel-concat-causal-preemption-native-and-JS.

Next scientific step: examine whether actual pre-onset information
exists at a controlled false-alarm level, potentially with additional
physically synchronized channels or longer ESA anomaly data. Do not
construct a Honeydew preemption task demanding impossible foresight.
