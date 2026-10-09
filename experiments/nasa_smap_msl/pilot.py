#!/usr/bin/env python3
"""NASA SMAP/MSL 12-channel anomaly detection pilot.

All transformations/models/thresholds use the original training series.
Test annotation labels are loaded ONLY after predictions for evaluation.
Run: python experiments/nasa_smap_msl/pilot.py --out results/nasa_pilot
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/nasa_smap_msl"
DEFAULT_CHANNELS = ("P-1","E-1","E-10","G-7","P-4","T-1",
                    "F-7","C-1","T-13","T-8","S-2","M-1")


def load_series(channel):
    with np.load(DATA/"values"/"train"/(channel+".npz")) as f:
        train=np.asarray(f["value"],dtype=float)
    with np.load(DATA/"values"/"test"/(channel+".npz")) as f:
        test=np.asarray(f["value"],dtype=float)
    if not (np.isfinite(train).all() and np.isfinite(test).all()):
        raise ValueError(f"Nonfinite observations in {channel}")
    if len(train)<80 or len(test)<100:
        raise ValueError(f"Not enough data for {channel}")
    return train,test


def event_intervals(channel,n,annotations):
    # Dataset indices are inclusive endpoints; duplicate annotation rows
    # (notably P-2) are combined without treating them as separate channels.
    rows=annotations.loc[annotations.chan_id==channel]
    if len(rows)==0:
        raise ValueError(f"No ground-truth annotation for {channel}")
    ivals=[]
    for raw in rows.anomaly_sequences:
        for a,b in ast.literal_eval(raw):
            a,b=int(a),int(b)
            if a<0 or b<a or b>=n:
                raise ValueError(f"Bad label for {channel}: {a},{b},{n}")
            ivals.append((a,b+1))
    ivals.sort()
    merged=[]
    for a,b in ivals:
        if merged and a<=merged[-1][1]:
            merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:
            merged.append((a,b))
    return merged


def median_scale(fit):
    med=float(np.median(fit))
    mad=float(1.4826*np.median(np.abs(fit-med)))
    # Keep a fixed train-only scale, even for nearly constant telemetry.
    fallback=float(np.std(fit))
    scale=max(mad,fallback*0.1,1e-8)
    return med,scale


def zscore(fit,x):
    med,scale=median_scale(fit)
    return np.abs(x-med)/scale


def cusum(fit,x):
    med,scale=median_scale(fit)
    d=(x-med)/scale
    score=np.empty(len(x))
    pos=neg=0.0
    for i,z in enumerate(d):
        pos=max(0.0,pos+float(z)-1.0)
        neg=max(0.0,neg-float(z)-1.0)
        score[i]=max(pos,neg)
    return score


def lag_context(symbols,d,k):
    n=len(symbols)
    if d==0: return np.zeros(n,dtype=np.int64)
    hist=np.zeros(n,dtype=np.int64)
    if d>=n: return hist
    m=k**d
    state=0
    for j in range(d):
        state=state*k+int(symbols[j])
    for i in range(d,n):
        hist[i]=state
        state=(state*k+int(symbols[i]))%m
    return hist


def pfsa_probs(symbols,d,k,alpha=0.5):
    ctx=lag_context(symbols,d,k)
    indexes=ctx[d:]*k+symbols[d:]
    counts=np.bincount(indexes,minlength=(k**d)*k).reshape(-1,k)
    return (counts+alpha)/(counts.sum(axis=1,keepdims=True)+alpha*k)


def pfsa_surprise(symbols,d,k,probs):
    ctx=lag_context(symbols,d,k)
    score=-np.log(np.maximum(probs[ctx,np.asarray(symbols,dtype=int)],1e-12))
    if d:score[:d]=np.nan
    return score


def rolling_mean_causal(x,n):
    return pd.Series(x,dtype=float).rolling(n,min_periods=max(1,n//3)).mean().to_numpy()


def fit_pfsa(fit,val,up_to=5,rolling=32):
    # Bin boundaries and model order derived solely from normal training.
    edges=np.unique(np.quantile(fit,[0.25,0.50,0.75]))
    k=len(edges)+1
    f=np.digitize(fit,edges)
    v=np.digitize(val,edges)
    candidates=[]
    for d in range(min(up_to,5)+1):
        if len(f)<=d+10 or len(v)<=d+10:continue
        probs=pfsa_probs(f,d,k)
        surpr=pfsa_surprise(v,d,k,probs)
        nll=float(np.nanmean(surpr))
        # Explicit complexity cost discourages fitting rare contexts.
        complexity=(k-1)*(k**d)
        cost=nll+complexity*np.log(len(f))/(2*len(f))
        candidates.append((cost,d,probs))
    if not candidates:raise ValueError("No eligible PFSA")
    cost,d,probs=min(candidates,key=lambda item:item[0])
    def score(x):
        sym=np.digitize(x,edges)
        return rolling_mean_causal(pfsa_surprise(sym,d,k,probs),rolling)
    return score,{"selected_order":d,"alphabet":k,"selection_cost":cost}


def profile_features(x,window,stride):
    if len(x)<window:return np.array([],dtype=int),np.empty((0,window))
    starts=np.arange(0,len(x)-window+1,stride)
    seq=sliding_window_view(x,window)[starts].copy()
    mu=seq.mean(axis=1,keepdims=True)
    sd=seq.std(axis=1,keepdims=True)
    # z-normalize subsequences: deliberate shape-only baseline.
    norm=(seq-mu)/np.maximum(sd,1e-8)
    return starts+window-1,norm


def sparse_to_causal(full_length,ends,values):
    scores=np.full(full_length,np.nan)
    if len(ends)==0:return scores
    # Attribute each distance at the last observed symbol in its window;
    # forward-fill only, never use future observations.
    indexes=np.searchsorted(ends,np.arange(full_length),side="right")-1
    good=indexes>=0
    scores[good]=np.asarray(values)[indexes[good]]
    return scores


def fit_matrix_profile(fit,window=64,stride=16,reference_cap=256):
    _,bank=profile_features(fit,window,stride)
    if len(bank)<3:raise ValueError("Too few reference windows")
    bank=bank[np.unique(np.linspace(0,len(bank)-1,min(reference_cap,len(bank))).astype(int))]
    nn=NearestNeighbors(n_neighbors=1,algorithm="brute",metric="euclidean")
    nn.fit(bank)
    def score(x):
        ends,features=profile_features(x,window,stride)
        if len(ends)==0:return np.full(len(x),np.nan)
        distances=nn.kneighbors(features,return_distance=True)[0].ravel()/math.sqrt(window)
        return sparse_to_causal(len(x),ends,distances)
    return score,{"reference_windows":len(bank),"profile_window":window}


def fit_native_lsmash(fit,val,test,window=250,reference_cap=8):
    """REAL compiled LSmash distance, no KL or probability-counting surrogate.

    Uses the 4 default native PFSA projectors. Transforms train/val/test
    using one threshold learned on the normal training part.
    """
    import lsmash
    med=float(np.median(fit))
    def sampled(x,max_windows=180):
        arr=(np.asarray(x)>med).astype(np.uint32)
        if len(arr)<window:raise ValueError(f"Need >= {window} observations")
        stride=max(1,window//2)
        starts=np.arange(0,len(arr)-window+1,stride)
        if len(starts)>max_windows:
            starts=starts[np.linspace(0,len(starts)-1,max_windows).astype(int)]
        return starts+window-1,[arr[s:s+window].tolist() for s in starts]
    _,refs=sampled(fit,max_windows=reference_cap)
    v_ends,v=sampled(val)
    t_ends,t=sampled(test)
    opts=lsmash.LsmashOptions()
    opts.data_type="symbolic"
    opts.sae=False
    D=np.asarray(lsmash.from_sequences(refs+v+t,opts),dtype=float)
    if D.shape!=(len(refs)+len(v)+len(t),)*2 or not np.isfinite(D).all():
        raise ValueError("Native LSmash matrix invalid")
    vdistance=D[:len(refs),len(refs):len(refs)+len(v)].min(axis=0)
    tdistance=D[:len(refs),len(refs)+len(v):].min(axis=0)
    return (sparse_to_causal(len(val),v_ends,vdistance),
            sparse_to_causal(len(test),t_ends,tdistance),
            {"native_projectors":"four_default","reference_windows":len(refs),
             "window":window})


def threshold_from_validation(val_scores,quantile=.995):
    finite=np.asarray(val_scores,dtype=float)
    finite=finite[np.isfinite(finite)]
    if len(finite)<20:raise ValueError(f"Too few finite calibration observations: {len(finite)}")
    return float(np.quantile(finite,quantile))


def evaluate(test_scores,threshold,intervals):
    good=np.isfinite(test_scores)
    alerts=(np.asarray(test_scores)>threshold)&good
    mask=np.zeros(len(alerts),dtype=bool)
    detected=0
    delays=[]
    for a,b in intervals:
        mask[a:b]=True
        times=np.flatnonzero(alerts[a:b])
        if len(times):
            detected+=1
            delays.append(int(times[0]))
    # Count distinct false alert episodes outside all annotated intervals.
    false=alerts & (~mask)
    starts=np.flatnonzero(false & ~np.r_[False,false[:-1]])
    valid_outside=np.count_nonzero(good & ~mask)
    return dict(event_recall=detected/len(intervals),
        events_detected=detected,events_total=len(intervals),
        median_delay=float(np.median(delays)) if delays else None,
        false_alert_episodes=len(starts),
        false_alerts_per_1000=1000.0*len(starts)/max(valid_outside,1),
        false_alert_points=int(false.sum()),
        evaluated_points=int(good.sum()),
        alert_fraction=float(alerts.sum()/max(good.sum(),1)))


def plot_channel(channel,test,intervals,series,output):
    n_methods=len(series)
    fig,axes=plt.subplots(n_methods+1,1,figsize=(13,2.0*(n_methods+1)),
                          sharex=True,constrained_layout=True)
    axes[0].plot(test,lw=.65,color="#334155")
    axes[0].set_ylabel("Telemetry")
    axes[0].set_title(f"{channel} — real spacecraft telemetry and train-calibrated detectors")
    for j,(method,(scores,thresh)) in enumerate(series.items(),start=1):
        ax=axes[j]
        ax.plot(scores,lw=.85)
        ax.axhline(thresh,ls="--",lw=1,color="#c2410c")
        ax.set_ylabel(method,fontsize=9)
    for ax in axes:
        for a,b in intervals:ax.axvspan(a,b,color="#dc2626",alpha=.13,lw=0)
        ax.grid(alpha=.18)
    axes[-1].set_xlabel("Test time index (observations)")
    fig.savefig(output,dpi=130)
    plt.close(fig)


def run_channel(channel,methods,out,args,annotations):
    train,test=load_series(channel)
    pivot=int(0.7*len(train))
    fit,val=train[:pivot],train[pivot:]
    if len(val)<50:raise ValueError("Insufficient normal calibration segment")
    # No evaluation annotations accessed until all method scores + thresholds
    # have been computed, preventing label-dependent tuning.
    results={}
    metainfo={}
    for method in methods:
        begin=time.perf_counter()
        try:
            if method=="zscore":
                sv=zscore(fit,val); st=zscore(fit,test)
                extra={}
            elif method=="cusum":
                sv=cusum(fit,val);st=cusum(fit,test)
                extra={"reset_at_split":True}
            elif method=="matrix_profile":
                f,extra=fit_matrix_profile(fit,args.profile_window,args.profile_stride)
                sv=f(val); st=f(test)
            elif method=="pfsa":
                f,extra=fit_pfsa(fit,val,args.max_order,args.pfsa_roll)
                sv=f(val);st=f(test)
            elif method=="lsmash":
                sv,st,extra=fit_native_lsmash(fit,val,test,args.lsmash_window)
            else:raise ValueError("Unknown method "+method)
            threshold=threshold_from_validation(sv,args.cal_quantile)
            results[method]=(np.asarray(st),threshold)
            metainfo[method]={"threshold":threshold,"elapsed_seconds":round(time.perf_counter()-begin,3),**extra}
        except Exception as exc:
            # Retain error in output; never replace a failed detector with another.
            metainfo[method]={"status":"failed","error":repr(exc),"elapsed_seconds":round(time.perf_counter()-begin,3)}
            if args.fail_on_error:raise
    intervals=event_intervals(channel,len(test),annotations)
    summaries=[]
    for method,(score,threshold) in results.items():
        detail=evaluate(score,threshold,intervals)
        summaries.append({"channel":channel,"method":method,"N_train":len(train),
            "N_test":len(test),"spacecraft":str(annotations.loc[annotations.chan_id==channel,"spacecraft"].iloc[0]),
            "calibration_quantile":args.cal_quantile,**detail,**metainfo[method]})
    if results:
        plot_channel(channel,test,intervals,results,out/"plots"/(channel+".png"))
    np.savez_compressed(out/"scores"/(channel+".npz"),
        **{f"score_{k}":v[0] for k,v in results.items()},
        **{f"threshold_{k}":np.asarray([v[1]]) for k,v in results.items()})
    print("CHANNEL",json.dumps({"channel":channel,"events":len(intervals),
         "methods":{r["method"]:{"recall":r["event_recall"],
                  "false_per_1000":r["false_alerts_per_1000"]} for r in summaries},
         "errors":{k:v["error"] for k,v in metainfo.items() if "error" in v}}),flush=True)
    return summaries,metainfo


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--channels",default=",".join(DEFAULT_CHANNELS))
    ap.add_argument("--methods",default="zscore,cusum,matrix_profile,pfsa")
    ap.add_argument("--out",default="results/nasa_smap_msl_pilot")
    ap.add_argument("--cal-quantile",type=float,default=.995)
    ap.add_argument("--profile-window",type=int,default=64)
    ap.add_argument("--profile-stride",type=int,default=16)
    ap.add_argument("--max-order",type=int,default=5)
    ap.add_argument("--pfsa-roll",type=int,default=32)
    ap.add_argument("--lsmash-window",type=int,default=250)
    ap.add_argument("--fail-on-error",action="store_true")
    args=ap.parse_args()
    if not 0.5<args.cal_quantile<1:ap.error("Invalid quantile")
    channels=[c.strip() for c in args.channels.split(",") if c.strip()]
    methods=[m.strip() for m in args.methods.split(",") if m.strip()]
    if not channels or not methods:ap.error("Need channels and methods")
    allowed={"zscore","cusum","matrix_profile","pfsa","lsmash"}
    if any(m not in allowed for m in methods):ap.error(f"Methods must be among {sorted(allowed)}")
    out=Path(args.out); (out/"plots").mkdir(parents=True,exist_ok=True)
    (out/"scores").mkdir(parents=True,exist_ok=True)
    ann=pd.read_csv(DATA/"labeled_anomalies.csv")
    allrows=[];details={}
    for channel in channels:
        rows,meta=run_channel(channel,methods,out,args,ann)
        allrows.extend(rows);details[channel]=meta
    if not allrows:raise RuntimeError("No successful detector output")
    df=pd.DataFrame(allrows)
    df.to_csv(out/"per_channel.csv",index=False)
    aggregate=(df.groupby("method").agg(
        channels=("channel","nunique"),
        mean_event_recall=("event_recall","mean"),
        median_event_recall=("event_recall","median"),
        median_false_alerts_per_1000=("false_alerts_per_1000","median"),
        total_events_detected=("events_detected","sum"),
        total_events=("events_total","sum"),
        median_detection_delay=("median_delay","median")).reset_index())
    aggregate["pooled_event_recall"]=aggregate.total_events_detected/aggregate.total_events
    aggregate.to_csv(out/"summary.csv",index=False)
    metadata=dict(channels=channels,methods=methods,calibration="last 30% of normal train",
       train_fit="first 70% of normal train",test_labels="evaluation only",
       quantile=args.cal_quantile,
       notes="Not a statistically independent cross-method threshold at fixed FPR; validation quantile only.",
       per_channel=details)
    (out/"run.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("SUMMARY",aggregate.to_json(orient="records"),flush=True)
    print("WROTE",str(out.resolve()),flush=True)


if __name__=="__main__":
    main()
