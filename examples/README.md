# Examples

This directory is a repository of example usages of ```patternly```
in self-contained jupyter notebooks.

## Basic Examples
+ example 0: 2023 samples
+ example 1: 55 samples
+ example 2: streaming example (single stream, anomaly arises somewhere along the stream)
+ example 3: 55 samples

## Satellite Data

_Located in ```./data/01_2015_LANL-01A_SOPA_MPA.txt```_

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
  contstructs a new PFSA to add to the library when it comes across an sequence
  that does not map back to any of the already established PFSAs.

+ ```Satellite_half_assign_half_stream```: This approach is similar to both
  ```Satellite_manually_assign_clusters``` and
  ```Satellite_continuous_streaming```. We first establish a set number of
  PFSAs to construct for the first half of the data. Then, upon streaming the
  second half of the data, when we come across sequences that do not map back to any of the
  PFSAs in the library, we construct a new PFSA and add it to the established library
  rather than marking these unknown sequences as anamolous.


## Agitation Data

From a study where patients wore wearable devices that contained accelerometer, heartrate,
temperature, and electrodermal bloodflow sensors. This example demonstrates find a library
of PFSAs of the raw data, then abstracting to find PFSAs of these PFSAs (i.e. finding 
the pattern of patterns).

## Sleep Data

_Located in ```./data/sc4002e0.rec.edf```. The entire dataset can be downloaded
from [physionet](https://physionet.org/content/sleep-edf/1.0.0/)._


## Dealing with .edf files

Many of the EEG datasets from [physionet](https://physionet.org/content/) come in the
European Data Format (EDF). ```Reading_european_data_format_(edf).ipynb``` demonstrates
basic manipulation of .edf files to quickly obtain a pandas ```DataFrame```.

<!-- ## Test data and notebooks -->
