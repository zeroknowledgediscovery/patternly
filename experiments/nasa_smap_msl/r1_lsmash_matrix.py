#!/usr/bin/env python3
"""Native compiled LSmash 40x40 pairwise window distances on NASA R-1.

This script does not implement or approximate LSmash. The only distance
operation is lsmash.from_sequences, using compiled native projectors.

The test trace is split into N exactly nonoverlapping equal-length windows.
Quantization edges are learned exclusively on the normal training recording.
Any trailing test observations smaller than one window are omitted.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pilot import load_series

def make_windows(train, x, requested_n, requested_alphabet):
    if not (2<=requested_alphabet<=8):
        raise ValueError("Alphabet requested must be 2..8")
    if not (2<=requested_n<=200):
        raise ValueError("n-windows must be 2..200")
    edges=np.unique(np.quantile(train,np.arange(1,requested_alphabet)/requested_alphabet))
    k=len(edges)+1
    # Never invent categorical distinctions to overcome tied quantiles.
    if k<2:
        raise ValueError("Quantization degenerate: all quantile edges identical")
    symbols=np.digitize(x,edges).astype(np.uint32)
    window_length=len(x)//requested_n
    if window_length<32:
        raise ValueError(f"Insufficient R-1 samples for {requested_n} windows: window_len={window_length}")
    used_len=requested_n*window_length
    mat=symbols[:used_len].reshape(requested_n,window_length)
    counts=np.bincount(mat.ravel().astype(int),minlength=k)
    return mat, edges, counts, used_len, window_length

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--channel",default="R-1")
    parser.add_argument("--split",choices=["train","test"],default="test")
    parser.add_argument("--n-windows",type=int,default=40)
    parser.add_argument("--alphabet",type=int,default=4)
    parser.add_argument("--out",default="results/nasa_r1_native_lsmash_matrix")
    args=parser.parse_args()
    out=Path(args.out)
    out.mkdir(exist_ok=True,parents=True)

    train,test=load_series(args.channel)
    x=train if args.split=="train" else test
    matrix,edges,counts,used_len,w=make_windows(train,x,args.n_windows,args.alphabet)
    if np.count_nonzero(counts)<2:
        raise RuntimeError("Selected windows contain only one symbol; native distances cannot resolve dynamics")
    import lsmash
    opts=lsmash.LsmashOptions()
    opts.data_type="symbolic"
    opts.sae=False
    seq=[row.astype(int).tolist() for row in matrix]
    t0=time.perf_counter()
    native=np.asarray(lsmash.from_sequences(seq,opts),dtype=np.float64)
    elapsed=time.perf_counter()-t0
    if native.shape!=(args.n_windows,args.n_windows):
        raise RuntimeError(f"Unexpected native LSmash shape {native.shape}")
    if not np.isfinite(native).all():
        raise RuntimeError(f"Nonfinite entries in native LSmash matrix: {np.sum(~np.isfinite(native))}")
    # Keep the returned native matrix UNCHANGED in CSV/NPY.
    np.save(out/"distance_matrix.npy",native)
    np.savetxt(out/"distance_matrix.csv",native,delimiter=",",fmt="%.12g")
    with (out/"windows.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,["window","start_inclusive","end_exclusive","n_samples"])
        wr.writeheader()
        for i in range(args.n_windows):
            wr.writerow(dict(window=i,start_inclusive=i*w,end_exclusive=(i+1)*w,n_samples=w))
    upper=native[np.triu_indices(args.n_windows,k=1)]
    metadata=dict(
        source="NASA SMAP/MSL R-1 dataset via patternly",
        implementation="Native zeroknowledgediscovery/lsmash package: lsmash.from_sequences",
        channel=args.channel,split=args.split,
        train_samples=int(len(train)),test_samples=int(len(test)),
        n_windows=args.n_windows,window_length=w,
        used_samples=used_len,unused_tail_samples=int(len(x)-used_len),
        alphabet_requested=args.alphabet,alphabet_realized=int(len(edges)+1),
        train_quantile_edges=edges.tolist(),symbol_counts=counts.astype(int).tolist(),
        native_options=dict(data_type="symbolic",sae=False),
        cpu_seconds=elapsed,
        diag_min=float(np.diag(native).min()),diag_max=float(np.diag(native).max()),
        symmetric_max_abs_error=float(np.max(np.abs(native-native.T))),
        offdiag_min=float(upper.min()),offdiag_median=float(np.median(upper)),
        offdiag_mean=float(upper.mean()),offdiag_max=float(upper.max()),
        note="No Euclidean, KL, Markov, or other surrogate distances; all values originate from actual compiled lsmash.from_sequences."
    )
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")

    fig,ax=plt.subplots(figsize=(9.5,8.5))
    im=ax.imshow(native,cmap="viridis",interpolation="nearest",
                 origin="upper",aspect="equal")
    ticks=np.arange(0,args.n_windows,5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("Window j (nonoverlapping, chronological)")
    ax.set_ylabel("Window i (nonoverlapping, chronological)")
    ax.set_title(f"Native LSmash pairwise distance — {args.channel} {args.split}\n{args.n_windows} windows × {w} samples • {len(edges)+1} symbols")
    cb=fig.colorbar(im,ax=ax,fraction=.047,pad=.04)
    cb.set_label("Native LSmash distance")
    fig.tight_layout()
    fig.savefig(out/"distance_heatmap.png",dpi=195)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,3.3))
    ax.plot(np.arange(len(x)),x,linewidth=.65,color="#3264aa")
    for p in range(w,used_len,w):
        ax.axvline(p,alpha=.12,linewidth=.5,color="#121212")
    ax.axvline(used_len,color="tab:red",alpha=.8,label="End of included windows")
    ax.set(title=f"{args.channel} {args.split} telemetry: nonoverlapping window boundaries",
        xlabel="Observation index",ylabel="Raw normalized channel value")
    ax.legend(loc="best",fontsize=7)
    fig.tight_layout()
    fig.savefig(out/"telemetry_windows.png",dpi=180)
    plt.close(fig)
    print("LSMASH_MATRIX_SUMMARY",json.dumps(metadata,allow_nan=False),flush=True)
    print("LSMASH_MATRIX_PREVIEW",json.dumps(np.round(native[:5,:5],6).tolist()),flush=True)
    print("LSMASH_MATRIX_WRITTEN",str(out.resolve()),flush=True)

if __name__=="__main__":
    main()
