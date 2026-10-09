# STATUS: SUPERSEDED FOR PREEMPTIVE DETECTION

The original below audit scores windows against **all other test windows**;
it is retrospective and noncausal. That violates the newly required
past-only constraint. Even though its labels were not used to construct
those distances, future measurements entered every transductive
reference population. Its perfect 40-window JS AUROC **must not be used
to judge whether a prospective preemption task is easy**.

The corrected, reference-only, train-calibrated early-warning audit is
implemented in p1_causal_preemption.py and is executed by
.github/workflows/nasa-p1-causal-preemption.yml.
Both genuine native LSmash variants and explicit statistical
JS baselines use only normal training reference windows, and a
new alarm must begin strictly before the annotated anomaly onset
to count as preemptive. Historical results below remain intact
solely as descriptive transductive observations.

---

# P-1 Honeydew difficulty audit: can simple agents solve the anomalies?

Date: 2026-10-09.

Frozen native P-1 experiment branch: `freeze/p1-native-sliding-20261009` at commit `19e0a50144bac2a87e0905ae35d684f4d5263b5d`.
Unchanged native run ID 37945644905 (1659-window original and GenESeSS-projector matrices); native nonoverlap 40-window run ID 37942422102.

Successful audit run: https://github.com/zeroknowledgediscovery/patternly/actions/runs/37964126368
Baseline code: `experiments/nasa_smap_msl/p1_honeydew_feasibility_baselines.py`.
Artifact: `P1-Honeydew-baselines-versus-true-native-LSmash`, with CSV and AUC plots.

## Input and retrospective evaluation

P-1 test: 8505 observations. Train: 2872. Four symbolic values from quartile boundaries calculated exclusively from train. Windows: length 212. The larger comparison uses stride 5, 1659 (highly overlapping) windows, of which 276 overlap any NASA anomaly. The separate nonoverlap comparison uses stride 212, 40 windows, 6 overlapping anomaly labels. Positives are windows with *any* overlap with an annotated interval; this can label an otherwise mostly clean window as positive.

Window anomaly ranking scores evaluated by AUROC, average precision, and precision at K (K = number of positive windows). NASA labels are used ONLY for retrospective evaluation, not baseline construction. No prospective alarm thresholds are evaluated or claimed.

**CRITICAL IMPLEMENTATION DISTINCTION:** Native results were *not recreated* in this script; previously computed C++ LSmash native matrices were loaded unchanged. All baseline methods are explicitly simple Python NumPy/SciPy statistical controls. None are simplified implementations of GenESeSS, LSmash, or LSM.

## Sliding 1659 overlapping windows

Method | AUROC | Average precision | Precision at 276
:--|--:|--:|--:
First-order bigram Jensen-Shannon vs all test windows | 0.827056 | 0.733446 | 0.637681
**Original native LSmash with actual GenESeSS PFSAs** | **0.825490** | **0.693722** | **0.692029**
Marginal histogram Jensen-Shannon vs all test windows | 0.798147 | 0.727267 | 0.634058
First-order bigram Jensen-Shannon vs normal training | 0.781590 | 0.597017 | 0.500000
Marginal histogram Jensen-Shannon vs normal training | 0.778257 | 0.624998 | 0.539855
Original native LSmash default random PFSAs | 0.741423 | 0.632346 | 0.543478
Raw-window absolute mean deviation | 0.353731 | 0.143199 | 0.177536

The bigram-JS agent baseline edges the learned-projector native method on AUROC and average precision. Native GenESeSS-projector LSmash has higher precision at K. This is a descriptive comparison on an overlapping-window, same-test reference distribution, not independent proof that any method dominates.

## Nonoverlapping 40 windows

Method | AUROC | Average precision | Precision at 6
:--|--:|--:|--:
**Marginal histogram JS vs all test windows** | **1.000000** | **1.000000** | **1.000000**
Bigram JS vs all test windows | 0.990196 | 0.948413 | 0.833333
Marginal histogram JS vs normal training | 0.931373 | 0.720370 | 0.666667
Bigram JS vs normal training | 0.906863 | 0.671164 | 0.500000
Original native LSmash default random PFSAs | 0.862745 | 0.762228 | 0.666667

We did NOT compute native GenESeSS-projector LSmash on these 40 nonoverlap windows; do not infer its outcome.

## Implication for Honeydew

Using P-1 anomalies as-is with this problem construction is too easy to show a requirement for learned temporal predictive structure. On the nonoverlap version, 4-symbol marginal Jensen-Shannon against the unlabeled test-window population perfectly separates all six overlap-labeled windows. A capable agent can implement this without any PFSA.

Before attempting to make a Honeydew task:
1. Construct real-derived, synthetic temporal corruptions with the same one-symbol and possibly two-symbol frequencies as genuine reference records, while modifying higher-order organization.
2. Confirm they are statistically identifiable: run original native GenESeSS and LSmash as well as exact-frequency and bigram baselines, on precisely matched held-out windows.
3. Include challenging clean windows and genuine distribution drift. Require a calibration protocol and a scientifically meaningful false-alarm ceiling.
4. Evaluate an Oracle and independent agents without exposing answer identities, types, or an anomaly count; set pass criteria only after the Oracle's stable solvability is demonstrated.

The present exercise is a *task-difficulty audit* rather than a completed Honeydew task.
