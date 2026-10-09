#!/usr/bin/env python3
"""P-1 online preemption audit — NO FUTURE DATA in features or thresholds.

Real native algorithms:
- original compiled LSmash from_sequences(), internal random PFSAs
- original C++ llk_distance(S,G) with genuinely inferred zedsuite.GenESeSS
  PFSAs, via experiment/genesess-projectors binding.

Labeled CUSTOM statistical baselines, not GenESeSS or LSmash:
- Jensen-Shannon distance of four-symbol histograms to TRAIN reference windows
- Jensen-Shannon distance of bigram counts to SAME TRAIN reference windows

Reference PFSAs, quantizer and calibration threshold use only normal TRAIN
observations. Each TEST window is scored at its LAST sample. At scoring time
the reference bank and all samples in the scored window are available. No
test observations beyond a window end are used in score computations.

Preemptive alert: an ALARM ONSET (threshold crossing) at a time t strictly
before the NASA onset a and within [a-H,a), for prechosen horizons H. An
alarm starting within an anomaly never counts as preemptive.

This is still a retrospective, three-event evaluation, not independent
validation of prospective warning.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score, average_precision_score

from pilot import DATA,load_series,event_intervals
from p1_sliding_lsmash import train_quantization,sliding,learned_projection_models

HORIZONS=(50,100,212,500)
CAL_QUANTILES=(.95,.98,.99,.995)
METHOD_NAMES=("native_LSmash_default","native_LSmash_GenESeSS",
              "CUSTOM_marginal_JS","CUSTOM_bigram_JS")

def to_features(rows,k):
    symbols=np.asarray(rows,dtype=np.int64)
    hist=np.stack([np.bincount(x,minlength=k) for x in symbols]).astype(float)+.5
    hist/=hist.sum(axis=1,keepdims=True)
    pairs=np.stack([np.bincount(k*x[:-1]+x[1:],minlength=k*k)
                    for x in symbols]).astype(float)+.5
    pairs/=pairs.sum(axis=1,keepdims=True)
    return hist,pairs

def train_reference_js(ref,queries,batch=40):
    """Explicit custom statistical JS baseline, never a PFSA approximation."""
    er=-np.sum(ref*np.log2(ref),axis=1)
    eq=-np.sum(queries*np.log2(queries),axis=1)
    score=np.empty(len(queries))
    for start in range(0,len(queries),batch):
        end=min(len(queries),start+batch)
        mix=(queries[start:end,None,:]+ref[None,:,:])/2
        em=-np.sum(mix*np.log2(mix),axis=2)
        dist=np.maximum(em-.5*eq[start:end,None]-.5*er[None,:],0)
        score[start:end]=dist.mean(axis=1)
    return score

def save_csv(path,rows):
    if not rows:
        path.write_text("")
        return
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="") as stream:
        wr=csv.DictWriter(stream,fieldnames=keys)
        wr.writeheader();wr.writerows(rows)

def compute_scores(args,out,train_s,test_s):
    import lsmash
    opts=lsmash.LsmashOptions()
    opts.data_type="symbolic";opts.sae=False
    ref_starts,refs=sliding(train_s[:args.fit_length],args.window,args.stride)
    # All calibration windows lie strictly after reference fit; no overlap.
    val_starts,val=sliding(train_s[args.fit_length:],args.window,args.stride)
    test_starts,test=sliding(test_s,args.window,args.stride)
    if len(refs)<5 or len(val)<50:
        raise RuntimeError("Insufficient separate training and normal calibration windows")
    nref,nval,ntest=len(refs),len(val),len(test)
    all_sequences=refs+val+test
    print("CAUSAL_DATA",json.dumps({
        "n_train_references":nref,"n_train_calibration":nval,"n_test":ntest,
        "fitted_reference_until":args.fit_length-1,
        "first_calibration_window_start":args.fit_length,
        "last_train_index":len(train_s)-1,
        "no_future_test_observations":True}),flush=True)

    native={}
    begin=time.perf_counter()
    D=np.asarray(lsmash.from_sequences(all_sequences,opts),dtype=float)
    if D.shape!=(len(all_sequences),)*2 or not np.isfinite(D).all():
        raise RuntimeError("Unusable full native default LSmash matrix")
    native["native_LSmash_default"]=(
        D[:nref,nref:nref+nval].mean(axis=0),
        D[:nref,nref+nval:].mean(axis=0)[nval:])
    print("NATIVE_DEFAULT_DONE",time.perf_counter()-begin,flush=True)
    del D

    models_dir=out/"native_models"
    models_dir.mkdir(exist_ok=True,parents=True)
    eps=tuple(float(x) for x in args.epsilons.split(","))
    models=learned_projection_models(train_s,models_dir,args.initial_train_length,
                                    eps,4)
    if not hasattr(lsmash,"from_sequences_with_pfsas"):
        raise RuntimeError("Native LSmash learned projector binding is missing. "
                           "No surrogate fallback.")
    begin=time.perf_counter()
    D=np.asarray(lsmash.from_sequences_with_pfsas(
        all_sequences,[m["zbase_file"] for m in models],opts),dtype=float)
    if D.shape!=(len(all_sequences),)*2 or not np.isfinite(D).all():
        raise RuntimeError("Unusable native GenESeSS-projected LSmash matrix")
    native["native_LSmash_GenESeSS"]=(
        D[:nref,nref:nref+nval].mean(axis=0),
        D[:nref,nref+nval:].mean(axis=0)[nval:])
    print("NATIVE_GENESESS_PROJECTIONS_DONE",time.perf_counter()-begin,flush=True)
    del D

    ref_hist,ref_big=to_features(refs,4)
    val_hist,val_big=to_features(val,4)
    test_hist,test_big=to_features(test,4)
    scores={
        **native,
        "CUSTOM_marginal_JS":(
            train_reference_js(ref_hist,val_hist),train_reference_js(ref_hist,test_hist)),
        "CUSTOM_bigram_JS":(
            train_reference_js(ref_big,val_big),train_reference_js(ref_big,test_big)),
    }
    for k,(v,t) in scores.items():
        if not (np.isfinite(v).all() and np.isfinite(t).all()):
            raise RuntimeError("Invalid score for "+k)
    times=(test_starts+args.window-1).astype(int)
    assert np.max(times)<len(test_s)
    arr={"test_end_times":times,
         "cal_end_times":(val_starts+args.fit_length+args.window-1).astype(int)}
    for name,(cal,test_scores) in scores.items():
        arr["cal_"+name]=cal;arr["test_"+name]=test_scores
    np.savez_compressed(out/"causal_scores.npz",**arr)
    return times,scores,models,dict(n_ref=nref,n_cal=nval,n_test=ntest)

def prospective_events(times,score,threshold,events,horizon):
    """Onset-based alarms on time-indexed, fully observed causal windows."""
    alarm=score>threshold
    rising=alarm & ~np.r_[False,alarm[:-1]]
    rise_times=times[rising]
    # A first observed above-threshold alarm at the first test score is
    # not necessarily a true onset. Never credit it as a precursor.
    rise_times=rise_times[rise_times>times[0]]
    covered=np.zeros(len(times),dtype=bool)
    for a,b in events:
        covered|=(times>=a-horizon)&(times<b)
    outside=(~covered)
    fa_times=np.array([t for t in rise_times
                       if not any(a-horizon<=t<b for a,b in events)])
    # Precursor-specific false positives exclude *both* warnings and
    # the entire annotated anomalies, so after-onset alarms are not
    # mislabeled as ordinary false alarms.
    def is_far(t):
        return not any(a-horizon<=t<b for a,b in events)
    false_times=np.array([t for t in rise_times if is_far(t)],dtype=int)
    per_event=[]
    for i,(a,b) in enumerate(events):
        precursors=rise_times[(rise_times>=a-horizon)&(rise_times<a)]
        new_post=rise_times[(rise_times>=a)&(rise_times<b)]
        per_event.append(dict(event=i,start=a,end_exclusive=b,horizon=horizon,
            advance_warning=int(len(precursors)>0),
            earliest_advance_time=(int(precursors.min()) if len(precursors) else ""),
            maximum_lead_samples=(int(a-precursors.min()) if len(precursors) else ""),
            first_post_onset_detection=(int(new_post.min()) if len(new_post) else ""),
            after_onset_detection=int(len(new_post)>0)))
    # Non-warning non-anomaly intervals provide a fair negative exposure.
    valid=np.ones(len(times),dtype=bool)
    for a,b in events:
        valid &= ~((times>=a-horizon)&(times<b))
    exposure=int(valid.sum())
    # Count only alarm onsets in actual negative time, not inside anomalies.
    neg=np.ones(len(times),dtype=bool)
    for a,b in events:
        neg &= ~((times>=a-horizon)&(times<b))
    fpr_events=0
    for t in rise_times:
        if not any(a-horizon<=t<b for a,b in events):
            fpr_events+=1
    # Exclude annotation intervals from false alarms (they are actual events).
    clean=np.ones(len(times),dtype=bool)
    for a,b in events:
        clean &= ~((times>=a-horizon)&(times<b))
        clean &= ~((times>=a)&(times<b))
    clean_n=int(clean.sum())
    clean_alarm_count=int((clean&alarm).sum())
    clean_rises=sum(int(clean[np.searchsorted(times,t)]) for t in rise_times)
    return per_event,dict(
        advance_hits=sum(x["advance_warning"] for x in per_event),
        events_total=len(events),post_onset_hits=sum(x["after_onset_detection"] for x in per_event),
        clean_alarm_occupancy=clean_alarm_count/max(clean_n,1),
        clean_alarm_onsets_per_1000=1000*clean_rises/max(clean_n,1),
        clean_alarm_onsets=int(clean_rises),clean_observations=clean_n)

def time_auroc(times,scores,events,horizon):
    positive=np.zeros(len(times),dtype=bool)
    ignored=np.zeros(len(times),dtype=bool)
    for a,b in events:
        positive|=((times>=a-horizon)&(times<a))
        ignored|=((times>=a)&(times<b))
    retained=~ignored
    if not np.any(positive&retained) or not np.any(~positive&retained):
        return dict(pre_interval_auc=None,pre_interval_AP=None)
    return dict(pre_interval_auc=float(roc_auc_score(positive[retained],scores[retained])),
                pre_interval_AP=float(average_precision_score(positive[retained],scores[retained])))

def plot_scores(times,scores,cal,events,out):
    fig,axes=plt.subplots(len(scores),1,figsize=(14,9),sharex=True)
    for axis,(method,(_,values)) in zip(axes,scores.items()):
        axis.plot(times,values,lw=.75,label=method)
        th=float(np.quantile(cal[method],.99))
        axis.axhline(th,color="tab:red",lw=1,ls="--",label="99th percentile of pre-test normal calibration")
        for i,(a,b) in enumerate(events):
            axis.axvspan(a-212,a,color="#eaa43b",alpha=.2,
                         label="212-step precursor window" if i==0 else None)
            axis.axvspan(a,b,color="#c94343",alpha=.17,
                         label="NASA anomaly" if i==0 else None)
        axis.set_ylabel(method,fontsize=8)
        axis.grid(alpha=.15)
    axes[0].legend(fontsize=7,loc="upper left",ncol=3)
    axes[-1].set_xlabel("Test observation index at END of scored window")
    fig.suptitle("P-1 genuine native LSmash vs explicit statistical baselines: causal scores",fontsize=13)
    fig.tight_layout()
    fig.savefig(out/"preemptive_causal_scores.png",dpi=150)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="results/nasa_p1_causal_preemption")
    ap.add_argument("--window",type=int,default=212)
    ap.add_argument("--stride",type=int,default=5)
    ap.add_argument("--fit-length",type=int,default=1500)
    ap.add_argument("--initial-train-length",type=int,default=1024)
    ap.add_argument("--epsilons",default="0.01,0.05,0.1,0.2")
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    train,test=load_series("P-1")
    if args.initial_train_length>args.fit_length:
        raise ValueError("Learned PFSAs must be inferred entirely within reference-fitting prefix")
    train_s,test_s,edges=train_quantization(train,test,4)
    times,scores,models,sizes=compute_scores(args,out,train_s,test_s)
    # The original NASA labels enter *only after* every score was computed.
    events=event_intervals("P-1",len(test),pd.read_csv(DATA/"labeled_anomalies.csv"))
    rows=[]
    per_event=[]
    for method,(cal,test_sc) in scores.items():
        for q in CAL_QUANTILES:
            threshold=float(np.quantile(cal,q))
            for h in HORIZONS:
                per,metrics=prospective_events(times,test_sc,threshold,events,h)
                auc=time_auroc(times,test_sc,events,h)
                rows.append(dict(method=method,calibration_quantile=q,
                    preemption_horizon=h,threshold=threshold,
                    **metrics,**auc))
                for entry in per:
                    per_event.append(dict(method=method,calibration_quantile=q,**entry))
    save_csv(out/"preemptive_summary.csv",rows)
    save_csv(out/"per_event_advance_warnings.csv",per_event)
    plot_scores(times,scores,{k:v[0] for k,v in scores.items()},events,out)
    meta=dict(channel="P-1",n_test_observations=len(test),n_train_observations=len(train),
        window=args.window,stride=args.stride,fit_length=args.fit_length,
        first_training_sample_of_normal_calibration=args.fit_length,
        learned_projectors_train_length=args.initial_train_length,
        train_quantizer_edges=edges.tolist(),
        q_levels=CAL_QUANTILES,horizons=HORIZONS,events=events,
        score_at_window_end_only=True,reference_population="TRAIN prefix only",
        labels_not_used_for_scoring_or_thresholds=True,
        native_algorithms=[
            "stock C++ LSmash original projectors",
            "original C++ LSmash llk_distance(S,G), native GenESeSS-trained projectors"],
        custom_baselines=["four-symbol marginal histogram JS divergence",
                          "4x4 first-order bigram histogram JS divergence"],
        learned_models=models,**sizes,
        disclaimer="Retrospective audit of three events, not independent proof of early warning")
    (out/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    best=[x for x in rows if x["calibration_quantile"]==.99 and x["preemption_horizon"]==212]
    for row in best:print("PRIMARY_212_PREEMPTION",json.dumps(row,allow_nan=False),flush=True)
    print("PREEMPTIVE_DONE",json.dumps({k:meta[k] for k in [
        "channel","window","stride","fit_length","n_ref","n_cal","n_test","events"]}),flush=True)
if __name__=="__main__":
    main()
