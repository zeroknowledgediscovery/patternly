#!/usr/bin/env bash
# Full seven-channel NASA historical-prefix preemption replication.
# Run from patternly root with Python 3.9 .venv-genesess activated.
# Requires actual compiled LSmash and native zedsuite GenESeSS.
set -euo pipefail
python - <<'PY'
import lsmash
from zedsuite.genesess import GenESeSS
assert hasattr(lsmash,"from_sequences_with_pfsas"),"Install the actual native experimental projector LSmash binding"
print("NATIVE_ENGINES_OK",lsmash.__file__,flush=True)
PY
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
CONCAT=results/nasa_concatenated_research_view
OUT=results/nasa_concat_preemption
mkdir -p "$OUT"
python experiments/nasa_smap_msl/concatenate_channels.py \
  --channels P-1,E-13,E-1,E-10,G-7,F-7,T-1 \
  --window 212 --stride 5 --out "$CONCAT"
for ch in P-1 E-13 E-1 E-10 G-7 F-7 T-1; do
  echo "=== $ch ==="
  mkdir -p "$OUT/$ch"
  if python experiments/nasa_smap_msl/concatenated_preemption_suite.py \
      --channel "$ch" --concat-root "$CONCAT" \
      --reference-length 1500 --calibration-length 750 \
      --initial-train-length 1024 --window 212 --stride 5 \
      --epsilons 0.01,0.05,0.1,0.2 \
      --out "$OUT/$ch" >"$OUT/$ch/stdout.log" 2>&1; then
    grep -E '^PRIMARY_PREEMPTION|^CHANNEL_STATUS|^NATIVE_METHOD_FAILURE' "$OUT/$ch/stdout.log" || true
  else
    rc=$?
    printf 'Channel %s failed: exit %s\n' "$ch" "$rc" > "$OUT/$ch/ERROR.txt"
    tail -30 "$OUT/$ch/stdout.log"
  fi
done
python experiments/nasa_smap_msl/aggregate_concat_preemption.py --root "$OUT"
echo "COMPLETE $OUT/primary_q99_h212.csv"
echo "COMPLETE $OUT/method_status.csv"
echo "COMPLETE $OUT/all_horizons_thresholds.csv"
echo "COMPLETE $OUT/seven_channel_preemption.png"
