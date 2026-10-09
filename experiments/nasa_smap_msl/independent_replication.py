#!/usr/bin/env python3
"""Pre-registered, outside-pilot NASA GenESeSS vs marginal replication.

All model choices and calibration thresholds use normal training only.
No test labels until both test scores and thresholds have been saved.
Requires Python 3.9, native zedsuite==0.0.7.

  python experiments/nasa_smap_msl/independent_replication.py \
      --out results/nasa_independent_replication
"""
from __future__ import annotations
import argparse
import ast
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from pilot import DATA, DEFAULT_CHANNELS, load_series, event_intervals, sparse_to_causal
from genesess_pilot import quantize, windows
from genesess_epsilon_sweep import check_native_environment

PREFIX="INDEPENDENT_CHANNEL="
BUDGETS=(.05,.01,.02,.10)
SKIP_CHANNELS=set(DEFAULT_CHANNELS)|{"T-10"}

def dumpcsv(path,items):
    if not items:
        path.write_text("")
        return
    keys=sorted(set().union(*(x.keys() for x in items)))
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        w.writerows(items)

def all_channels():
    names=sorted(x.stem for x in (DATA/"values"/"train").glob("*.npz"))
    if len(names)!=82:
        raise ValueError("Expected 82 NASA train channels, got %s"%len(names))
    return [n for n in names if n not in SKIP_CHANNELS]

def score_native(x,path,window,stride):
    from zedsuite.zutil import Llk
    frame,ends=windows(x,window,stride)
    vals=np.asarray(Llk(data=frame,pfsafile=str(path)).run(),dtype=float).ravel()
    if len(vals)!=len(ends):
        raise ValueError("Native Llk length mismatch: %d / %d"%(len(vals),len(ends)))
    if not np.isfinite(vals).all():
        raise ValueError("Native Llk returned %d nonfinite windows"%int((~np.isfinite(vals)).sum()))
    return sparse_to_causal(len(x),ends,vals),len(ends)

def score_marginal(x,p,window,stride):
    frame,ends=windows(x,window,stride)
    values=-np.log(np.maximum(p,1e-15))[frame.to_numpy(dtype=int)].mean(axis=1)
    return sparse_to_causal(len(x),ends,values),len(ends)

def metrics(scores,threshold,events):
    scores=np.asarray(scores,dtype=float)
    valid=np.isfinite(scores)
    alarm=valid & (scores>threshold)
    start=alarm & ~np.r_[False,alarm[:-1]]
    annotation=np.zeros(len(scores),dtype=bool)
    onset_by_event=[]
    hit_by_event=[]
    onset_delays=[]
    for a,b in events:
        annotation[a:b]=True
        onset=bool(start[a:b].any())
        onset_by_event.append(onset)
        hit_by_event.append(bool(alarm[a:b].any()))
        if onset:onset_delays.append(int(np.flatnonzero(start[a:b])[0]))
    neg=~annotation & valid
    neg_n=int(neg.sum())
    if neg_n==0:
        raise ValueError("No observed non-event data")
    return dict(
        event_onsets_detected=sum(onset_by_event),
        event_hits=sum(hit_by_event),
        event_total=len(events),
        new_onset_recall=(sum(onset_by_event)/len(events)),
        event_hit_recall=(sum(hit_by_event)/len(events)),
        false_alarm_points=int(np.sum(neg&alarm)),
        negative_points=neg_n,
        false_alarm_fraction=float(np.sum(neg&alarm)/neg_n),
        false_alarm_onset_count=int(np.sum(neg&start)),
        false_alarm_onsets_per_1000=float(1000*np.sum(neg&start)/neg_n),
        mean_delay=(float(np.mean(onset_delays)) if onset_delays else None),
        event_onset_flags=onset_by_event)

def child(args):
    from zedsuite.genesess import GenESeSS
    train,test=load_series(args.channel)
    pivot=int(.7*len(train))
    fit,val=train[:pivot],train[pivot:]
    (fit_s,val_s,test_s),k,edges=quantize(fit,val,test,4)
    if len(np.unique(fit_s))<2:
        print(PREFIX+json.dumps(dict(channel=args.channel,status="degenerate_training",
             distinct_training_symbols=len(np.unique(fit_s)),states=None)),flush=True)
        return
    if len(val_s)<args.window:
        print(PREFIX+json.dumps(dict(channel=args.channel,status="too_short_calibration",
                                    calibration_samples=len(val_s))),flush=True)
        return
    out=Path(args.out)
    (out/"models").mkdir(parents=True,exist_ok=True)
    (out/"scores").mkdir(parents=True,exist_ok=True)
    model_file=out/"models"/(args.channel+".pfsa")
    model=GenESeSS(data=pd.DataFrame([fit_s.tolist()]),outfile=str(model_file),
                   data_type="symbolic",data_dir="row",force=True,eps=args.eps)
    found=bool(model.run())
    if not found or not model_file.is_file() or not model_file.stat().st_size:
        raise RuntimeError("Native GenESeSS inference failed")
    state_morph=np.asarray(model.probability_morph_matrix)
    if state_morph.ndim!=2 or len(state_morph)<1:
        raise ValueError("Invalid GenESeSS state matrix shape")
    states=int(len(state_morph))
    freq=np.bincount(fit_s,minlength=k).astype(float)
    marginal=(freq+.5)/(freq.sum()+.5*k)

    # Identical completed windows and indices for both detectors.
    normal_native,n_cal_windows=score_native(val_s,model_file,args.window,args.stride)
    normal_marg,n_cal2=score_marginal(val_s,marginal,args.window,args.stride)
    if n_cal_windows!=n_cal2 or n_cal_windows<3:
        raise RuntimeError("Too few or different calibration windows %s %s"%(n_cal_windows,n_cal2))
    # Compute the normal-data thresholds before looking at any test labels.
    cal_native=normal_native[np.isfinite(normal_native)]
    cal_marg=normal_marg[np.isfinite(normal_marg)]
    thresholds={}
    for budget in BUDGETS:
        q=1-budget
        thresholds[str(budget)]=dict(GenESeSS=float(np.quantile(cal_native,q)),
                                   marginal=float(np.quantile(cal_marg,q)))
    test_native,n_test=score_native(test_s,model_file,args.window,args.stride)
    test_marg,n_test2=score_marginal(test_s,marginal,args.window,args.stride)
    if n_test!=n_test2:
        raise RuntimeError("Test window counts differ")
    np.savez_compressed(out/"scores"/(args.channel+".npz"),
                        GenESeSS=test_native,marginal=test_marg,
                        normal_GenESeSS=normal_native,normal_marginal=normal_marg,
                        quantizer_edges=edges,marginal_probabilities=marginal,
                        thresholds=np.asarray([[thresholds[str(b)]["GenESeSS"],
                                                thresholds[str(b)]["marginal"]] for b in BUDGETS]),
                        nominal_false_alarm_budgets=np.asarray(BUDGETS))
    # Only now access annotations. Prior steps cannot depend on test labels.
    labels=pd.read_csv(DATA/"labeled_anomalies.csv")
    intervals=event_intervals(args.channel,len(test),labels)
    per_budget=[]
    per_event=[]
    for budget in BUDGETS:
        result={}
        for name,score in [("GenESeSS",test_native),("marginal",test_marg)]:
            m=metrics(score,thresholds[str(budget)][name],intervals)
            result[name]=m
            per_budget.append(dict(channel=args.channel,method=name,
                budget=budget,threshold=thresholds[str(budget)][name],
                n_states=states,n_distinct_train_symbols=len(np.unique(fit_s)),
                cal_windows=n_cal_windows,cal_samples=len(cal_native),
                **{key:value for key,value in m.items() if key!="event_onset_flags"}))
        for j,(a,b) in enumerate(intervals):
            per_event.append(dict(channel=args.channel,budget=budget,
                event_index=j,start=a,end_exclusive=b,
                states=states,GenESeSS_onset=int(result["GenESeSS"]["event_onset_flags"][j]),
                marginal_onset=int(result["marginal"]["event_onset_flags"][j])))
    dumpcsv(out/"scores"/(args.channel+"_budgets.csv"),per_budget)
    dumpcsv(out/"scores"/(args.channel+"_events.csv"),per_event)
    result=dict(channel=args.channel,status="success",states=states,
                eps_requested=args.eps,eps_used=float(model.epsilon_used),
                train_length=len(train),fit_length=len(fit),test_length=len(test),
                alphabet_requested=4,alphabet_realized=k,
                distinct_training_symbols=len(np.unique(fit_s)),
                cal_windows=n_cal_windows,n_events=len(intervals),
                normal_entropy_bits=float(-np.sum(marginal*np.log2(marginal))),
                marginal_probabilities=marginal.tolist(),
                threshold_primary=thresholds[str(.05)],
                score_path=str((out/"scores"/(args.channel+".npz")).resolve()))
    print(PREFIX+json.dumps(result,allow_nan=False),flush=True)

def bootstrap_results(channel_rows,budget,subset="all"):
    # Bootstrap over channels, rather than treating correlated events
    # within a channel as statistically independent.
    chosen=[r for r in channel_rows if r["budget"]==budget and
            (subset=="all" or (subset=="multi" and r["n_states"]>=2) or
             (subset=="three" and r["n_states"]==3))]
    by={}
    for r in chosen:by.setdefault(r["channel"],{})[r["method"]]=r
    keys=sorted(k for k,x in by.items() if set(x)=={"GenESeSS","marginal"})
    if not keys:return None
    rows=[by[k] for k in keys]
    sums={}
    for name in ("GenESeSS","marginal"):
        ev=sum(x[name]["event_total"] for x in rows)
        tp=sum(x[name]["event_onsets_detected"] for x in rows)
        neg=sum(x[name]["negative_points"] for x in rows)
        fp=sum(x[name]["false_alarm_points"] for x in rows)
        sums[name]=dict(events=ev,onsets=tp,recall=tp/ev,
             false_alarm_fraction=fp/neg,
             false_alarm_episodes=sum(x[name]["false_alarm_onset_count"] for x in rows),
             false_alarm_onsets_per_1000=1000*sum(x[name]["false_alarm_onset_count"] for x in rows)/neg)
    rng=np.random.default_rng(264217)
    diffs=[]
    for _ in range(1000):
        chosen=[rows[i] for i in rng.integers(len(rows),size=len(rows))]
        gnum=sum(x["GenESeSS"]["event_onsets_detected"] for x in chosen)
        mnum=sum(x["marginal"]["event_onsets_detected"] for x in chosen)
        den=sum(x["GenESeSS"]["event_total"] for x in chosen)
        if den:diffs.append((gnum-mnum)/den)
    return dict(budget=budget,stratum=subset,
                channels=len(keys),
                events=sums["GenESeSS"]["events"],
                GenESeSS=sums["GenESeSS"],marginal=sums["marginal"],
                recall_difference=sums["GenESeSS"]["recall"]-sums["marginal"]["recall"],
                bootstrap_95ci=(np.quantile(diffs,[.025,.975]).tolist() if diffs else None))

def parent(args):
    check_native_environment()
    channels=all_channels()
    print("FROZEN_NEW_CHANNELS",json.dumps(channels),flush=True)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    # Frozen manifest before any access to channel test labels.
    manifest=dict(channels=channels,excluded=sorted(SKIP_CHANNELS),
         nominal_false_alarm_budgets=BUDGETS,primary_budget=.05,
         window=args.window,stride=args.stride,eps=args.eps,alphabet=4,
         training_ratio=.7,selection_uses_test_labels=False)
    (out/"protocol_run.json").write_text(json.dumps(manifest,indent=2)+"\n")
    results=[]
    for i,channel in enumerate(channels):
        command=[sys.executable,str(Path(__file__).resolve()),"child",
                 "--channel",channel,"--out",str(out.resolve()),
                 "--eps",str(args.eps),"--window",str(args.window),
                 "--stride",str(args.stride)]
        t0=time.monotonic()
        try:
            p=subprocess.run(command,capture_output=True,text=True,timeout=args.timeout)
            msg=[x[len(PREFIX):] for x in p.stdout.splitlines() if x.startswith(PREFIX)]
            if p.returncode==0 and msg:
                item=json.loads(msg[-1])
            else:
                item=dict(channel=channel,status="native_failed",returncode=p.returncode,
                          stderr_tail=p.stderr[-1600:],stdout_tail=p.stdout[-300:])
        except subprocess.TimeoutExpired:
            item=dict(channel=channel,status="timeout",timeout=args.timeout)
        item["wall_seconds"]=round(time.monotonic()-t0,3)
        results.append(item)
        print("CHANNEL",json.dumps({key:item.get(key) for key in
            ("channel","status","states","eps_used","cal_windows","n_events","wall_seconds")},
            allow_nan=False),flush=True)
        if item["status"]=="native_failed":
            print("CHANNEL_ERROR",channel,item.get("stderr_tail"),flush=True)
        dumpcsv(out/"channel_status.csv",results)
    channel_data=[]
    event_data=[]
    for r in results:
        if r["status"]=="success":
            ch=r["channel"]
            budget=pd.read_csv(out/"scores"/(ch+"_budgets.csv"))
            event=pd.read_csv(out/"scores"/(ch+"_events.csv"))
            channel_data.extend(budget.to_dict("records"))
            event_data.extend(event.to_dict("records"))
    dumpcsv(out/"per_budget.csv",channel_data)
    dumpcsv(out/"per_event.csv",event_data)
    total=[]
    for budget in BUDGETS:
        for stratum in ("all","multi","three"):
            val=bootstrap_results(channel_data,budget,stratum)
            if val:total.append(val)
    (out/"paired_summary.json").write_text(json.dumps(total,indent=2)+"\n")
    for v in total:
        print("PAIRED",json.dumps(v,allow_nan=False),flush=True)
    if event_data:
        pairs=[]
        d=pd.DataFrame(event_data)
        for budget in BUDGETS:
            for stratum in ("all","multi","three"):
                events=d[d.budget==budget]
                if stratum!="all":
                    eligible=[r["channel"] for r in results if r["status"]=="success" and
                              (r["states"]>=2 if stratum=="multi" else r["states"]==3)]
                    events=events[events.channel.isin(eligible)]
                if events.empty:continue
                a=events.GenESeSS_onset.astype(bool)
                b=events.marginal_onset.astype(bool)
                pairs.append(dict(budget=budget,stratum=stratum,n_events=len(events),
                    both=int((a&b).sum()),GenESeSS_only=int((a&~b).sum()),
                    marginal_only=int((~a&b).sum()),neither=int((~a&~b).sum())))
        dumpcsv(out/"paired_event_discordance.csv",pairs)
    print("FINAL_STATUS",json.dumps(dict(attempted=len(channels),
       scored=sum(x["status"]=="success" for x in results),
       multi_state=sum(x.get("states",0)>=2 for x in results if x["status"]=="success"),
       exactly_three=sum(x.get("states",0)==3 for x in results if x["status"]=="success"),
       not_scored=sum(x["status"]!="success" for x in results))),flush=True)

def self_test():
    x=np.array([0,1,0,0,1,1,0,1,0,0,1,1])
    p=np.array([.4,.6])
    s,n=score_marginal(x,p,4,2)
    assert n==5 and np.isnan(s[:3]).all()
    assert abs(s[3] - (-np.mean(np.log(p[x[:4]]))))<1e-10
    events=[(5,7)]
    a=np.array([np.nan,0,0,0,0,1,1,0,0],float)
    m=metrics(a,.5,events)
    assert m["event_onsets_detected"]==1 and m["event_hits"]==1
    print("SELF_TEST_OK")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",nargs="?",default="parent",choices=("parent","child","selftest"))
    ap.add_argument("--channel",default="A-1")
    ap.add_argument("--eps",type=float,default=.1)
    ap.add_argument("--window",type=int,default=128)
    ap.add_argument("--stride",type=int,default=32)
    ap.add_argument("--timeout",type=int,default=45)
    ap.add_argument("--out",default="results/nasa_independent_replication")
    args=ap.parse_args()
    if args.mode=="selftest":self_test()
    elif args.mode=="child":child(args)
    else:parent(args)

if __name__=="__main__":
    main()
