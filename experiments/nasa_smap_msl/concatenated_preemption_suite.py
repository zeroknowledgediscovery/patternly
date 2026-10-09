#!/usr/bin/env python3
"""Seven-channel *concatenated* NASA causal predictive-anomaly replication.

Runs ACTUAL compiled LSmash default projectors and (where possible)
ACTUAL compiled LSmash llk_distance(S,G) with native zedsuite.GenESeSS PFSAs.
Separate, plainly labeled custom marginal and first-order bigram JS controls.

All methods see a single chronological concatenated target channel.
The join between original train and test segments is tracked explicitly.
No window crosses the join; neither calibration nor native model inference
uses observations after its calibration cutoff. Query windows are scored
at their end time, from FIXED past-only reference windows. The complete
distance computation is batched only to accelerate independently scored
windows, with no test-reference fitting, normalization, or pooling.

Caveat: timestamps are anonymized; a train/test join is not independently
verified physical adjacency. Preemption means *before a published label*
and is NOT established hardware failure prognosis.

PRE-REGISTERED PARAMETERS: first 1500 values as reference, next 750 as
calibration, window 212, stride 5, quantiles [.95,.98,.99,.995],
horizons [50,100,212,500]; initial 1024 reference samples for native PFSAs
at epsilon .01,.05,.1,.2. No test-label parameter selection.
"""
from __future__ import annotations
import argparse
import json
import traceback
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score

from pilot import DATA,load_series,event_intervals
from p1_sliding_lsmash import learned_projection_models, sliding
from p1_causal_preemption import to_features,train_reference_js,save_csv,HORIZONS,CAL_QUANTILES

CHANNELS=("P-1","E-13","E-1","E-10","G-7","F-7","T-1")
METHODS=("native_LSmash_default","native_LSmash_GenESeSS",
         "CUSTOM_marginal_JS","CUSTOM_bigram_JS")

def windows_in_segment(symbols,start,end,length,stride):
    if end-start<length:return np.empty(0,dtype=np.int64),[]
    local_starts,rows=sliding(symbols[start:end],length,stride)
    global_starts=local_starts+start
    assert all(s>=start and s+length<=end for s in global_starts)
    return global_starts,rows

def mean_native_to_references(lsmash_module,ref,cal,queries,options,pfsa_files=None):
    """Original C++ native library throughout. No surrogate likelihood."""
    n=len(ref);v=len(cal);q=len(queries)
    sequences=ref+cal+queries
    if pfsa_files is None:
        D=np.asarray(lsmash_module.from_sequences(sequences,options),dtype=float)
    else:
        D=np.asarray(lsmash_module.from_sequences_with_pfsas(
            sequences,[str(x) for x in pfsa_files],options),dtype=float)
    if D.shape!=(n+v+q,n+v+q) or not np.isfinite(D).all():
        raise RuntimeError(f"Native distance output invalid: {D.shape}")
    if np.max(np.abs(D-D.T))>1e-7 or np.max(np.abs(np.diag(D)))>1e-7:
        raise RuntimeError("Native distance symmetry/diagonal validation failed")
    return D[:n,n:n+v].mean(axis=0),D[:n,n+v:].mean(axis=0)

def alarm_onsets(times,scores,threshold,stride,seam):
    """Only new crossings with an observed below-threshold predecessor
    in the SAME contiguous recording. First score of each segment is
    never credited as an alarm onset."""
    active=np.asarray(scores>threshold,dtype=bool)
    same_segment=np.r_[False,
        (np.diff(times)==stride)&
        ((times[1:]<seam)==(times[:-1]<seam))]
    transitions=active&same_segment&~np.r_[False,active[:-1]]
    return active,times[transitions]

def evaluate(times,scores,calibration,events,stride,seam,q,h):
    threshold=float(np.quantile(calibration,q))
    active,risetimes=alarm_onsets(times,scores,threshold,stride,seam)
    events=[(int(a),int(b)) for a,b in events]
    def in_any_anomaly(t):
        return any(a<=t<b for a,b in events)
    per=[]
    for event_id,(a,b) in enumerate(events):
        leads=[int(t) for t in risetimes if a-h<=t<a and not in_any_anomaly(t)]
        during=[int(t) for t in risetimes if a<=t<b]
        per.append(dict(event=event_id,anomaly_onset=a,anomaly_end=b,
            advance_warning=int(bool(leads)),
            earliest_advance_time=min(leads) if leads else "",
            maximum_lead_samples=a-min(leads) if leads else "",
            post_onset_detected=int(bool(during)),
            first_post_onset_alarm=min(during) if during else "",
            warning_horizon=h))
    valid=np.ones(len(times),bool)
    future_pre=np.zeros(len(times),bool)
    for a,b in events:
        valid &= ~((times>=a)&(times<b))
        future_pre |= ((times>=a-h)&(times<a))
    # No precursor samples that lie inside *another* anomaly.
    prec=valid&future_pre
    clean=valid&~future_pre
    alert_onsets_clean=sum(bool(np.any((times==t)&clean)) for t in risetimes)
    clean_n=int(clean.sum())
    auc=float(roc_auc_score(prec[valid],scores[valid])) if prec.any() and clean.any() else None
    ap=float(average_precision_score(prec[valid],scores[valid])) if prec.any() and clean.any() else None
    return dict(
        calibration_quantile=q,preemption_horizon=h,threshold=threshold,
        advance_hits=sum(p["advance_warning"] for p in per),
        post_onset_hits=sum(p["post_onset_detected"] for p in per),
        events_total=len(events),
        clean_alarm_occupancy=float(np.mean(active[clean])) if clean_n else None,
        clean_alarm_onsets=int(alert_onsets_clean),
        clean_alarm_onsets_per_1000=1000*alert_onsets_clean/max(clean_n,1),
        clean_scored_windows=clean_n,
        pre_interval_auc=auc,pre_interval_ap=ap),per

def save_plot(out,channel,times,methods,calibrations,events,seam):
    if not methods:return
    fig,axes=plt.subplots(len(methods),1,figsize=(14,2.45*len(methods)+1),
                          sharex=True,squeeze=False)
    for ax,(name,score) in zip(axes.ravel(),methods.items()):
        ax.plot(times,score,lw=.85)
        ax.axhline(np.quantile(calibrations[name],.99),
                   color="#9f3434",ls="--",lw=1)
        ax.axvline(seam,color="#333333",ls=":",lw=1.2)
        for i,(a,b) in enumerate(events):
            ax.axvspan(a-212,a,color="#d7aa45",alpha=.18)
            ax.axvspan(a,b,color="#ca4848",alpha=.16)
        ax.set_ylabel(name,fontsize=8)
        ax.grid(alpha=.14)
    axes[-1,0].set_xlabel("Observation index in concatenated recording (window END)")
    fig.suptitle(f"{channel} — past-only preemption scores. Orange: preceding 212; red: NASA labels\n"
                 "Dotted line: unverified historical recording join; dashed score line: pre-test calibrated q=99%",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out/"causal_preemption_scores.png",dpi=150,bbox_inches="tight")
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--channel",required=True,choices=CHANNELS)
    p.add_argument("--concat-root",default="")
    p.add_argument("--out",default="")
    p.add_argument("--reference-length",type=int,default=1500)
    p.add_argument("--calibration-length",type=int,default=750)
    p.add_argument("--initial-train-length",type=int,default=1024)
    p.add_argument("--window",type=int,default=212)
    p.add_argument("--stride",type=int,default=5)
    p.add_argument("--epsilons",default="0.01,0.05,0.1,0.2")
    args=p.parse_args()
    if args.initial_train_length>args.reference_length:
        raise ValueError("GenESeSS model inference must be inside fixed past reference")
    cutoff=args.reference_length+args.calibration_length
    out=Path(args.out or f"results/nasa_concat_preemption/{args.channel}")
    out.mkdir(parents=True,exist_ok=True)
    if args.concat_root:
        path=Path(args.concat_root)/(args.channel+".npz")
        with np.load(path) as z:
            full=np.asarray(z["value"],dtype=float)
            seam=int(z["segment_boundary"][0])
            events=[(int(a),int(b)) for a,b in z["anomaly_intervals"]]
            assert np.allclose(full[:seam],load_series(args.channel)[0])
            assert np.allclose(full[seam:],load_series(args.channel)[1])
    else:
        train,test=load_series(args.channel)
        seam=len(train)
        full=np.concatenate([train,test])
        labels=pd.read_csv(DATA/"labeled_anomalies.csv")
        events=[(a+seam,b+seam) for a,b in
                event_intervals(args.channel,len(test),labels)]
    if cutoff>seam:
        raise ValueError(f"Reference+calibration extends beyond historical normal prefix {seam}")
    if not np.isfinite(full).all():raise ValueError("Invalid raw telemetry")
    fit=full[:args.reference_length]
    # Bin boundaries determined only from initial 1500 historical samples,
    # never the rest of the series. Ties are collapsed explicitly.
    edges=np.unique(np.quantile(fit,np.arange(1,4)/4))
    k=len(edges)+1
    if k<2:
        raise RuntimeError(f"DEGENERATE_QUANTIZATION: {args.channel} has one bin in reference")
    # For a collapsed quartile at the reference minimum, numpy's default
    # right=False maps *both* the historical minimum and all later higher
    # values to the SAME code. Choose the side based ONLY on historical
    # reference measurements; this preserves the ability to detect higher
    # values without using future observations or anomaly labels.
    right_closed=bool(k<4 and edges[0]==np.min(fit))
    symbols=np.digitize(full,edges,right=right_closed).astype(np.uint32)
    ref_start,ref=windows_in_segment(
        symbols,0,args.reference_length,args.window,args.stride)
    cal_start,cal=windows_in_segment(
        symbols,args.reference_length,cutoff,args.window,args.stride)
    tail_start,tail=windows_in_segment(
        symbols,cutoff,seam,args.window,args.stride)
    post_start,post=windows_in_segment(
        symbols,seam,len(full),args.window,args.stride)
    starts=np.concatenate([tail_start,post_start])
    queries=tail+post
    times=starts+args.window-1
    nref,ncal,nquery=len(ref),len(cal),len(queries)
    if nref<5 or ncal<50 or nquery<50:raise ValueError(
        f"Insufficient reference={nref}, calibration={ncal}, evaluation={nquery} windows")
    assert np.all(times[:len(tail)]<seam)
    assert np.all(times[len(tail):]>=seam+args.window-1)
    assert not np.any((starts<seam)&(starts+args.window>seam))
    # If native default projector vocabulary inferred from future sequences,
    # that would contaminate its random PFSA family. Require the reference
    # prefix already to contain all observed codes.
    ref_symbols=set(int(v) for row in ref for v in row)
    alphabet_full=set(int(x) for x in symbols)
    methods={}
    calibrations={}
    statuses={}
    model_info=[]
    def add(name,val_scores):
        calibration,test_scores=val_scores
        calibration=np.asarray(calibration,float)
        test_scores=np.asarray(test_scores,float)
        if calibration.shape!=(ncal,) or test_scores.shape!=(nquery,):
            raise RuntimeError(f"{name}: wrong native score shape")
        if not np.isfinite(calibration).all() or not np.isfinite(test_scores).all():
            raise RuntimeError(f"{name}: nonfinite scoring")
        calibrations[name]=calibration;methods[name]=test_scores
        statuses[name]="ok"

    # Each native C++ method runs in its own isolated subprocess.
    # A native free()/malloc abort (seen in GenESeSS on some channels)
    # cannot invalidate the other detectors or erase completed results.
    import subprocess,sys
    np.savez_compressed(out/"native_input.npz",
        ref=np.asarray(ref,dtype=np.uint32),
        cal=np.asarray(cal,dtype=np.uint32),
        test=np.asarray(queries,dtype=np.uint32),
        initial=np.asarray(symbols[:args.initial_train_length],dtype=np.uint32),
        alphabet=np.asarray(k,dtype=int))
    methods_info={}
    worker=Path(__file__).with_name("native_causal_worker.py")
    for mode,name in (("default","native_LSmash_default"),
                      ("genesess","native_LSmash_GenESeSS")):
        if mode=="default" and not alphabet_full.issubset(ref_symbols):
            statuses[name]="SKIPPED: reference vocabulary missing symbols seen later; would leak into random-projector alphabet"
            continue
        output=out/("native_"+mode+"_scores.npz")
        command=[sys.executable,str(worker),"--mode",mode,
                 "--infile",str(out/"native_input.npz"),
                 "--out",str(output),"--epsilons",args.epsilons]
        cp=subprocess.run(command,capture_output=True,text=True)
        (out/("native_"+mode+"_worker.log")).write_text(
            "RETURN_CODE "+str(cp.returncode)+"\n"+cp.stdout+"\n"+cp.stderr)
        if cp.returncode!=0:
            statuses[name]=f"FAILED_NATIVE_WORKER: exit {cp.returncode}, "+(cp.stderr or cp.stdout).strip()[-280:]
            print("NATIVE_METHOD_FAILURE",args.channel,name,statuses[name],flush=True)
            continue
        try:
            with np.load(output) as arrays:
                add(name,(arrays["cal"],arrays["test"]))
            info=json.loads(output.with_suffix(".json").read_text())
            methods_info[name]=info
            if mode=="genesess":model_info=info["models"]
        except Exception as ex:
            statuses[name]="FAILED_NATIVE_OUTPUT_VALIDATION: "+repr(ex)
            methods.pop(name,None);calibrations.pop(name,None)

    for name,dimension in (("CUSTOM_marginal_JS",0),("CUSTOM_bigram_JS",1)):
        try:
            ref_feats=to_features(ref,k)[dimension]
            cal_feats=to_features(cal,k)[dimension]
            query_feats=to_features(queries,k)[dimension]
            add(name,(train_reference_js(ref_feats,cal_feats),
                     train_reference_js(ref_feats,query_feats)))
        except Exception as ex:statuses[name]="FAILED_CUSTOM: "+repr(ex)

    annotations_count=len(events)
    if annotations_count==0:raise RuntimeError("No annotated events for channel")
    rows=[];per_events=[]
    for name,values in methods.items():
        for q in CAL_QUANTILES:
            for h in HORIZONS:
                summary,per=evaluate(times,values,calibrations[name],
                          events,args.stride,seam,q,h)
                rows.append(dict(channel=args.channel,method=name,**summary))
                per_events += [dict(channel=args.channel,method=name,
                     calibration_quantile=q,**p) for p in per]
    save_csv(out/"preemptive_summary.csv",rows)
    save_csv(out/"per_event_advance_warnings.csv",per_events)
    np.savez_compressed(out/"past_only_scores.npz",
         end_times=times,calibration_end_times=cal_start+args.window-1,
         **{"score_"+name:values for name,values in methods.items()},
         **{"cal_"+name:values for name,values in calibrations.items()})
    save_plot(out,args.channel,times,methods,calibrations,events,seam)
    manifest=dict(channel=args.channel,total_len=int(len(full)),seam=seam,
         seam_physical_contiguity_confirmed=False,
         first_reference_end_exclusive=args.reference_length,
         calibration_range=[args.reference_length,cutoff],
         test_scored_after_index=cutoff,
         n_ref=nref,n_cal=ncal,n_scored=nquery,
         n_pre_seam_scored=len(tail),
         n_post_seam_scored=len(post),
         window=args.window,stride=args.stride,
         reference_quantile_edges=edges.tolist(),alphabet_size=k,
         train_only_bin_edge_tie_policy=("right_closed" if right_closed else "numpy_default_left_closed"),
         reference_symbols=sorted(ref_symbols),
         full_symbols=sorted(alphabet_full),events=events,
         reference_policy="fixed past-only prefix, no model update",
         label_policy="NASA annotations accessed only to evaluate scores",
         alarm_onset_policy="new crossings only, same-segment predecessor required",
         methods_status=statuses,native_projector_models=model_info,
         native_worker_provenance=methods_info,
         default_native_pinned="original C++ LSmash method",
         learned_native_pinned="real native GenESeSS PFSAs and C++ llk_distance(S,G)",
         baseline_class="simple custom statistical JS, not LSmash, GenESeSS, or LSM")
    (out/"metadata.json").write_text(json.dumps(manifest,indent=2)+"\n")
    mainrows=[x for x in rows if x["calibration_quantile"]==.99
              and x["preemption_horizon"]==212]
    for x in mainrows:print("PRIMARY_PREEMPTION",json.dumps(x,allow_nan=False),flush=True)
    print("CHANNEL_STATUS",json.dumps({k:manifest[k] for k in (
         "channel","total_len","seam","n_scored","alphabet_size","methods_status")},
         allow_nan=False),flush=True)
    if len(methods)<2:
        raise RuntimeError("Fewer than two detector methods produced valid scores")
if __name__=="__main__":
    main()
