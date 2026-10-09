#!/usr/bin/env python3
"""Concatenate per-channel original NASA train+test target telemetry.

Creates a single research-view array, original train/test boundary marker,
correctly shifted NASA test anomaly annotations, and full-stream plots.

IMPORTANT: NASA has anonymized timestamps; consecutive train/test segments
are NOT proven physically adjacent. The boundary is explicitly recorded.
Never permit an analysis window to bridge that boundary unless true
temporal adjacency has been independently established.

This is data preparation / visualization only, NOT GenESeSS or LSmash,
and it does NOT perform or claim early-warning prediction.
"""
from __future__ import annotations
import argparse
import ast
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pilot import DATA,load_series,event_intervals

DEFAULT_CHANNELS=("P-1","E-13","E-1","E-10","G-7","F-7","T-1")

def shifts_for_channel(channel,test_len,annotations,offset):
    if channel not in annotations.chan_id.values:
        return []
    return [[int(a+offset),int(b+offset)]
            for a,b in event_intervals(channel,test_len,annotations)]

def concat_one(channel,out,annotations,window,stride):
    train,test=load_series(channel)
    n_train,n_test=len(train),len(test)
    joined=np.concatenate([train,test])
    assert len(joined)==n_train+n_test
    # Preserve split offset to avoid treating an uncertain join as a
    # genuine transition in any subsequent LSM/LSmash analysis.
    periods=[[0,n_train],[n_train,len(joined)]]
    intervals=shifts_for_channel(channel,n_test,annotations,n_train)
    np.savez_compressed(out/(channel+".npz"),
        value=joined,segment_boundary=np.asarray([n_train],np.int64),
        anomaly_intervals=np.asarray(intervals,dtype=np.int64).reshape((-1,2)),
        segment_ranges=np.asarray(periods,dtype=np.int64))
    # Every window belongs to a single underlying recorded segment.
    windows=[]
    for segment,(start,end) in enumerate(periods):
        for s in range(start,end-window+1,stride):
            windows.append((s,s+window,segment))
    # No leakage by accidental crossing a potentially noncontiguous seam.
    assert all(not(s<n_train<e) for s,e,_ in windows)
    meta=dict(channel=channel,train_length=n_train,test_length=n_test,
        concatenated_length=len(joined),
        boundary_index=n_train,
        physical_contiguity_confirmed=False,
        inferred_sampling_gap=None,
        anomaly_intervals_concatenated=intervals,
        length_window=window,stride=stride,
        windows_not_crossing_uncertain_boundary=len(windows),
        source="Original real NASA SMAP/MSL target channel values",
        note="The joined array is an index-concatenation for exploration; "
        "the train/test split is not documented as physically contiguous. "
        "Only test-period NASA anomaly annotations are available.")
    (out/(channel+"_metadata.json")).write_text(json.dumps(meta,indent=2)+"\n")
    with (out/(channel+"_windows.csv")).open("w",newline="") as stream:
        wr=csv.writer(stream)
        wr.writerow(["window","start","end_exclusive","source_segment"])
        wr.writerows((i,*row) for i,row in enumerate(windows))

    fig,ax=plt.subplots(figsize=(13.5,4))
    x=np.arange(len(joined),dtype=int)
    ax.plot(x,joined,color="#365b90",lw=.65)
    ax.axvline(n_train,color="#222222",lw=1.35,linestyle="--",
        label="Join — physical continuity unverified")
    for i,(a,b) in enumerate(intervals):
        ax.axvspan(a,b,color="#bd4847",alpha=.25,
            label="Published anomaly interval" if i==0 else None)
    ax.set(xlabel="Index in concatenated research-view series",
           ylabel="Normalized target telemetry value",
           title=(f"NASA {channel}: {len(joined):,} total observations; "
                  f"{len(intervals)} annotated intervals in original test portion"))
    ax.legend(fontsize=8,loc="best")
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(out/(channel+"_concatenated.png"),dpi=175)
    plt.close(fig)
    return meta

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--channels",default=",".join(DEFAULT_CHANNELS))
    p.add_argument("--out",default="results/nasa_concatenated_research_view")
    p.add_argument("--window",type=int,default=212)
    p.add_argument("--stride",type=int,default=5)
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    labels=pd.read_csv(DATA/"labeled_anomalies.csv")
    rows=[]
    for c in (x.strip() for x in args.channels.split(",")):
        if not c:continue
        meta=concat_one(c,out,labels,args.window,args.stride)
        rows.append(meta)
        print("CONCAT_CHANNEL",json.dumps(meta),flush=True)
    (out/"manifest.json").write_text(json.dumps(dict(
        channels=rows,method="concat(original_train_values, original_test_values)",
        boundary_assumption="Unknown temporal gap at join; do not bridge"),
        indent=2)+"\n")
    print("CONCAT_READY",len(rows),str(out.resolve()),flush=True)

if __name__=="__main__":
    main()
