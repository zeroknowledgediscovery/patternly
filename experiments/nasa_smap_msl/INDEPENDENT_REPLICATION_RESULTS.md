# Independent NASA GenESeSS vs one-state marginal: preregistered replication

Date: October 9, 2026.
Protocol (frozen before evaluation): INDEPENDENT_REPLICATION_PROTOCOL.md
Implementation: independent_replication.py
Successful native CI:
https://github.com/zeroknowledgediscovery/patternly/actions/runs/37925064973
Artifact: nasa-independent-genesess-vs-marginal

## Design and independence limits

The 12 previously studied NASA SMAP/MSL channels, plus unannotated T-10,
were excluded. We tested all 69 other channels, i.e. other
channel recordings in the SAME published NASA dataset.
This is *independent-channel* replication of a model specification,
NOT additional or independent P-1 telemetry.
Telemetry channels from a shared spacecraft need not be statistically
independent.

Both models use the exact same four-symbol quantization derived from
the first 70% of each channel's normal training series, and the same
128-symbol windows, stride 32. Native GenESeSS epsilon is FIXED
at 0.10. The marginal model uses an empirical add-0.5 one-state
symbol distribution on the exact same training segment.
Normal train remaining 30% calibrates alarm thresholds at the
same quantile. The primary nominal alarm budget is 5% on normal
calibration, not on test. Test labels are read only after all
scores, quantizers and thresholds have been computed.

Primary metric: new alarm onsets within annotated anomaly intervals.
An already active alarm crossing an event boundary does not count
as a new detection. Report test negative-point alarm occupancy and
false-alarm episodes/1000.

## Trial accounting

- 69 channels were attempted.
- 41 produced complete, finite, paired anomaly-score time series.
- 19 degenerated to one unique symbol in normal training.
- 9 native trials failed: A-6 and G-4 due to nonfinite Llk scores,
  A-8 because native inference did not return a valid PFSA,
  T-9 because its normal calibration contained only one completed
  128-symbol window, D-11/D-3/D-4/F-3/G-6 because the native
  subprocess crashed with signal 11 (segmentation fault).
- Of the 41 scorable channels, 18 learned >=2 GenESeSS states and
  6 learned EXACTLY 3 states. The six three-state channels were:
  A-3, D-15, F-1, F-2, P-10 and P-15.

These exclusions matter. The method does not have proven coverage of
all 69 new channels.

## Prespecified operational-threshold comparison

Normal validation quantile 0.95 (intended 5% calibration alarm budget):

State-count stratum | Channels | Annotated events | GenESeSS new-onset recall | Marginal new-onset recall | GenESeSS test non-event alarm fraction | Marginal test non-event alarm fraction
:--|--:|--:|:--|:--|--:|--:
All scorable | 41 | 45 | 20/45 (44.4%) | 21/45 (46.7%) | 26.76% | 27.46%
>=2 states | 18 | 19 | 8/19 (42.1%) | 9/19 (47.4%) | 20.30% | 36.08%
Exactly 3 states | 6 | 6 | 4/6 (66.7%) | 3/6 (50.0%) | 11.51% | 18.22%

Cluster bootstrap by channel, difference in onset recall
(GenESeSS minus marginal), 95% intervals:
- All 41 scorable channels: -2.2 percentage points,
  CI [-14.6,+11.4] pp.
- >=2-state channels: -5.3 percentage points,
  CI [-30.0,+16.7] pp.
- Exactly-3-state channels: +16.7 percentage points,
  CI [-33.3,+66.7] pp.

Paired event discordance, primary budget:
- All channels (45 events): both 16, GenESeSS-only 4,
  marginal-only 5, neither 20.
- >=2-state (19 events): both 6, GenESeSS-only 2,
  marginal-only 3, neither 8.
- Exactly-3-state (6 events): both 2, GenESeSS-only 2,
  marginal-only 1, neither 1.

In the three-state subgroup:
- A-3: both detected; GenESeSS 6.56% non-event alarm occupancy,
  marginal 18.25%.
- D-15: GenESeSS only; GenESeSS 1.22%, marginal 43.53%.
- F-1: GenESeSS only; GenESeSS 20.79%, marginal 0%.
- F-2: marginal only; both 0% non-event alarm occupancy.
- P-10: both detected; both 23.60%.
- P-15: neither detected; GenESeSS 0%, marginal 87.00%.

## Major finding: calibration shift

A 5% nominal normal-train threshold produced alarming during
11-36% of non-anomalous test observations across the key strata.
So calibration quantile is NOT a matched *realized test* FPR.
Native GenESeSS does not show overall statistically convincing
better anomaly-onset recall in these independent channels.
The apparently favorable 4/6 three-state result is too small
and statistically uncertain to establish temporal superiority.

Failure of transfer to the test stream is not the same as a
false inference. Drift in telemetry distributions and
calibration score distributions appears to be a major limitation.

## Secondary descriptive only: labels-known oracle false-alarm ceilings

After completing the prespecified evaluation, a SEPARATE script
exploratory_matched_fpr.py selected thresholds from the distribution
of true NON-ANOMALOUS test scores. This uses test labels. It is NOT
a prospective model comparison and cannot be used to claim
deployable accuracy or validate thresholds. This is a posthoc
diagnostic of sensitivity under a controlled realized FPR ceiling.

Oracle CI: https://github.com/zeroknowledgediscovery/patternly/actions/runs/37925631556
Artifact: nasa-independent-oracle-matched-realized-fpr-DESCRIPTIVE-ONLY

At 5% *maximum* realized non-event alarm fraction (ties make the
achieved rates conservative):

Stratum | GenESeSS recall | Marginal recall | Actual GenESeSS FPR | Actual marginal FPR
:--|:--|:--|--:|--:
All valid | 13/45 (28.9%) | 11/45 (24.4%) | 3.50% | 3.07%
>=2 states | 7/19 (36.8%) | 4/19 (21.1%) | 3.80% | 2.46%
Exactly 3 states | 3/6 (50.0%) | 1/6 (16.7%) | 3.83% | 2.02%

These are both BELOW the specified false-alarm ceiling, but the
achieved FPRs are NOT identical: GenESeSS uses a larger share of
the allowed budget. Channel bootstrap 95% CIs for the oracle
onset-recall differences also include zero:
- >=2: +15.8 percentage points, CI [-5.3,+38.9] pp.
- Exactly 3: +33.3 pp, CI [0,+66.7] pp.

This posthoc result is an exploratory hint that a conditional
generator can sometimes exploit temporal information at a more
restricted false-alarm budget. It is NOT confirmatory, partly
because true test-negative labels were used to set thresholds and
because 6 events give inadequate statistical power.

## Conclusions / next experiment

1. The independent-channel prospective comparison provides NO
   robust evidence of superior temporal anomaly detection by
   fixed epsilon=0.10 GenESeSS over the marginal-only model.
2. The exact-3-state subgroup has a positive but very uncertain
   recall difference, with no evidence of statistical significance.
3. Failures on 28/69 channels must be addressed: quantization
   degeneracy, native segfaults/nonfinite likelihoods, and too-short
   validation windows.
4. Normal-trained thresholds drift badly in deployment, and
   a claimed 5% budget must be validated on a future unannotated
   stream without using future labels for threshold selection.
5. Prior exploratory selection of epsilon from P-1 test labels
   must not be reused as though it were an untouched test.
6. Next: improve robust, train-only calibration under
   nonstationarity and obtain genuinely new P-1 recordings
   or additional prospective spacecraft datasets.

All artifacts preserve per-channel outcomes, model files,
both score arrays, per-budget and per-event paired labels,
and native failure records.
