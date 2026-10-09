# NASA SMAP/MSL spacecraft telemetry (real-world benchmark data)

Source: [appleparan/telemanom on Hugging Face](https://huggingface.co/datasets/appleparan/telemanom), originally from [NASA Telemanom](https://github.com/khundman/telemanom).

This data branch holds the **actual downloaded observations**, not simulated sequences.

## Contents

- `values/train/<channel>.npz`: original training telemetry target values, as lossless float64 arrays
- `values/test/<channel>.npz`: original test telemetry target values, as lossless float64 arrays
- `labeled_anomalies.csv`: untouched published test-set anomaly windows
- `manifest.json`: source, per-file counts, ranges, command-feature counts and SHA-256 checksums
- `SHA256SUMS.tsv`: SHA-256 of original Parquet files for independent verification
- `download_validate.py`: rerun complete source download, integrity checks and stream extraction
- `LICENSE.txt`: redistributed dataset license

The full original **164 Parquet files** (including the 24 or 54 auxiliary command columns) are additionally attached to the [successful GitHub Actions import run](https://github.com/zeroknowledgediscovery/patternly/actions/runs/37876933850) as `nasa-smap-msl-original-parquet-82-channels`. That artifact has a limited retention period; this branch keeps the 164 lossless univariate telemetry arrays permanently.

## Read the data

```python
import numpy as np
import pandas as pd

train = np.load("datasets/nasa_smap_msl/values/train/P-1.npz")["value"]
test  = np.load("datasets/nasa_smap_msl/values/test/P-1.npz")["value"]
labels = pd.read_csv("datasets/nasa_smap_msl/labeled_anomalies.csv")
print(train.shape, test.shape)
print(labels[labels.chan_id == "P-1"])
```

Indices are zero-based, stored implicitly as `np.arange(len(series))`. Repeated metadata rows were retained exactly as published. **T-10 has telemetry recordings but no annotation entry**; it must not be treated as a normal/negative test series without additional review. There are 82 distinct channels but only 81 channels with published labels.

For benchmarks, fit transforms and quantizers using training recordings only. Evaluate predictions against labels from `labeled_anomalies.csv` on the test recordings. Intervals denote annotated anomalies; they are not guaranteed to represent a single abrupt change between stationary stochastic laws.

## Reproduce the import

```bash
pip install numpy pandas pyarrow huggingface_hub
python datasets/nasa_smap_msl/download_validate.py \
   --source /tmp/smap_msl_source \
   --output datasets/nasa_smap_msl
```

The downloader fetches only the original Parquet files and metadata, intentionally excluding redundant NumPy files from the source repository.
