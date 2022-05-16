# Examples

+ example 0: 2023 samples
+ example 1: 55 samples
+ example 2: streaming example (single stream, anomaly arises somewhere along the stream)
+ example 3: 55 samples

## Satellite Data

A satellite data set containing particle counts in 19 energy bands (in keV):
``` 60.7, 89.42, 128.99, 185.7, 274.8, 394.62, 577.44, 840.33, 1122.5, 1210.3, 1580.99, 1989.97, 2437.21, 3074.09, 3968.63, 5196.15, 6841.05, 9178.24, and 16692.51.```

For analysis the data stream in each band is
partitioned into sequences of uniform window length.

+ ```Satellite_manually_assign_clusters```: The number of clusters expected to
  be present are assigned a priori. We then generate a PFSA for each cluster
  and assign each window sequence to one of the clusters. If a sequence is
  found that does not map back to one of the established PFSAs, it is labelled
  as anamolous.

+ ```Satellite_continuous_streaming```: No set amount of clusters is
  predefined. Instead, ```patternly``` analyzes the continuous streams and
  contstructs  a PFSA library when it comes across an uknown sequence.

+ ```Satellite_half_assign_half_stream```: This approach is imilar to both
  ```Satellite_manually_assign_clusters``` and
  ```Satellite_continuous_streaming```. We first establish a set number of
  PFSAs to construct for the first half of the data. Then, upon streaming the
  second half of the data, we add new PFSAs to the already established library
  rather than marking these unknown sequences as anamolous.

## Test data and notebooks
