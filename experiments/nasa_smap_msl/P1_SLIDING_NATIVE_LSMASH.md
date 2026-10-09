# P-1 sliding native LSmash vs GenESeSS learned PFSA projectors

The two reproducible experiments compute all-pairs distances for 1659
overlapping P-1 test windows (212 observations per window, stride 5).
Both use real compiled C++ LSmash, not a Python distance approximation.

Test 1 invokes the original native LSmash from_sequences method with
its four internally generated default PFSAs.

Test 2 invokes the original native C++ llk_distance(S,G) overload via
a small experimental pybind extension in the lsmash repository at
experiment/genesess-projectors. G contains four *actual GenESeSS*
PFSAs fitted on the same first 1024 P-1 normal training observations,
with epsilon 0.01,0.05,0.1,0.2. The models are native
zedsuite.genesess.GenESeSS version 0.0.7. Their probability morphs
and symbol-conditioned transition maps are *copied without change*
into zbase PFSA text format. This is serialization conversion only;
the native C++ LSmash likelihood algorithm is not rewritten.

The experimental lsmash branch also declares Python>=3.9 to support
the native zedsuite wheel, while the stock package declares >=3.10.

Reproduction from patternly repo, Fedora, Python 3.9:

    sudo dnf install gcc-c++ boost-devel gsl-devel libgomp
    source .venv-genesess/bin/activate
    python -m pip install --force-reinstall --no-deps \
      "lsmash @ git+https://github.com/zeroknowledgediscovery/lsmash.git@experiment/genesess-projectors"

    python experiments/nasa_smap_msl/p1_sliding_lsmash.py \
      --mode default --channel P-1 --window 212 --stride 5 \
      --out results/p1_sliding_default

    python experiments/nasa_smap_msl/p1_sliding_lsmash.py \
      --mode genesess --channel P-1 --window 212 --stride 5 \
      --initial-train-length 1024 --epsilons 0.01,0.05,0.1,0.2 \
      --out results/p1_sliding_genesess

To fit the initial 212-observation window exactly, instead set
--initial-train-length 212; failed native inference or likelihood is
reported as failure, never replaced by an approximation.

Outputs in both directories:
- annotated_heatmap.png: native chronological N-by-N matrix with
  red strips showing windows overlapping P-1 anomaly annotations
- anomaly_distance_profile.png: time-indexed mean distance and
  ground-truth anomaly spans
- distance_matrix.npy and distance_matrix.csv: unchanged native output
- windows.csv: chronological start/end plus annotation-overlap flag
- window_distance_profile.csv: mean distance per window
- metadata.json: measured settings and native provenance
Test 2 also exports all native GenESeSS and zbase PFSA model files.

Verified native GitHub Actions run:
https://github.com/zeroknowledgediscovery/patternly/actions/runs/37945644905

Native default: median offdiagonal 0.0533279, mean of annotation-overlap
windows 0.112428, mean of other windows 0.061446.
Native GenESeSS projection family: median offdiagonal 0.0682524,
mean of annotation-overlap windows 0.141915, others 0.075998.
The four trained models had 3,3,3,9 states respectively.

Scores from distinct projector families have different scales.
Both matrices are retrospective and windows overlap heavily;
these observations are not independent evaluation or calibrated
anomaly detection accuracy. Compare baseline later on *identical*
windows before attributing information specifically to dynamics.
