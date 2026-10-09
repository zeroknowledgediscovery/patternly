#!/usr/bin/env python3
"""Training-only GenESeSS epsilon sweep, targeting >=2 inferred PFSA states.

Each native invocation is isolated as a subprocess; no Markov substitutes.
Scoring is performed only for the train-selected model, never while searching
epsilon. Both 2- and 4-symbol training-only quantizers are investigated.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from pilot import DATA, DEFAULT_CHANNELS, load_series, event_intervals, evaluate, plot_channel, sparse_to_causal, threshold_from_validation
from genesess_pilot import quantize, windows
from audit_alerts import audit_score

PREFIX="GENESESS_EPS_CHILD="
DEFAULT_EPS="0.001,0.002,0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.3,0.5,0.7"

def epsilon_id(eps):
    return format(float(eps),".6g").replace(".","p")

def train_and_quantize(channel,alphabet):
    train,test=load_series(channel)
    n=int(.7*len(train))
    fit,val=train[:n],train[n:]
    (fq,vq,tq),real_k,edges=quantize(fit,val,test,alphabet)
    counts=np.bincount(fq,minlength=real_k)
    used=counts[counts>0]
    p=used/used.sum()
    entropy=float(-np.sum(p*np.log2(p)))
    return train,test,fit,val,fq,vq,tq,real_k,edges,entropy,len(used),float(max(p))

def emit(info):
    print(PREFIX+json.dumps(info,allow_nan=False),flush=True)

def infer_child(args):
    from zedsuite.genesess import GenESeSS
    train,test,fit,val,fq,vq,tq,k,edges,entropy,used,majority=train_and_quantize(args.channel,args.alphabet)
    if used<2:
        raise ValueError("Training has one distinct symbol; no nontrivial predictive states identifiable")
    dest=Path(args.out)/"models"/str(args.alphabet)/args.channel/("eps_"+epsilon_id(args.eps))
    dest.mkdir(parents=True,exist_ok=True)
    model=dest/"inferred.pfsa"
    t0=time.monotonic()
    alg=GenESeSS(data=pd.DataFrame([fq.tolist()]),outfile=str(model),
        data_type="symbolic",data_dir="row",force=True,eps=args.eps)
    found=bool(alg.run())
    if not found or not model.is_file() or model.stat().st_size==0:
        raise RuntimeError("GenESeSS did not return a usable model")
    morph=np.asarray(alg.probability_morph_matrix)
    n_states=int(morph.shape[0])
    if n_states<1 or morph.ndim!=2:
        raise ValueError("Invalid morph probability matrix shape")
    actual_eps=float(alg.epsilon_used)
    if not np.isfinite(actual_eps):raise ValueError("Nonfinite epsilon_used")
    info=dict(channel=args.channel,alphabet_requested=args.alphabet,
        alphabet_realized=k,n_distinct_training_symbols=used,
        fit_entropy_bits=entropy,majority_symbol_fraction=majority,
        requested_eps=args.eps,actual_eps=actual_eps,n_states=n_states,
        inference_error=float(alg.inference_error),
        model_path=str(model.resolve()),
        fit_samples=len(fq),status="success",
        inference_seconds=round(time.monotonic()-t0,4))
    if not np.isfinite(info["inference_error"]):
        info["inference_error"]=None
    emit(info)

def score_child(args):
    from zedsuite.zutil import Llk
    train,test,fit,val,fq,vq,tq,k,edges,entropy,used,majority=train_and_quantize(args.channel,args.alphabet)
    model=Path(args.model)
    if not model.is_file():raise FileNotFoundError(model)
    def get_scores(x):
        frame,ends=windows(x,args.window,args.stride)
        ll=np.asarray(Llk(data=frame,pfsafile=str(model)).run(),dtype=float).ravel()
        if len(ll)!=len(ends):
            raise RuntimeError(f"Llk returned {len(ll)} scores instead of {len(ends)}")
        if not np.isfinite(ll).all():
            raise ValueError(f"Nonfinite Llk values: {np.sum(~np.isfinite(ll))}/{len(ll)}")
        return sparse_to_causal(len(x),ends,ll)
    calibration=get_scores(vq)
    threshold=threshold_from_validation(calibration,args.quantile)
    scores=get_scores(tq)
    out=Path(args.out)
    scores_dir=out/"scores"/str(args.alphabet)
    scores_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(scores_dir/(args.channel+".npz"),
        score=scores,calibration_score=calibration,
        threshold=np.asarray([threshold]),
        selected_eps=np.asarray([args.eps]))
    annotations=pd.read_csv(DATA/"labeled_anomalies.csv")
    intervals=event_intervals(args.channel,len(test),annotations)
    result=evaluate(scores,threshold,intervals)
    audit=audit_score(scores,threshold,intervals)
    plot_dir=out/"plots"/str(args.alphabet)
    plot_dir.mkdir(parents=True,exist_ok=True)
    plot_channel(args.channel,test,intervals,
        {f"GenESeSS eps {args.eps:g}":(scores,threshold)},
        plot_dir/(args.channel+".png"))
    emit(dict(channel=args.channel,alphabet_requested=args.alphabet,
        requested_eps=args.eps,status="scored",
        threshold=threshold,**result,
        onset_hits=audit["onset_hits"],
        onset_event_recall=audit["onset_event_recall"],
        non_event_alarm_fraction=audit["non_event_alarm_fraction"],
        false_alarm_onsets_per_1000=audit["false_alarm_onsets_per_1000"]))

def subprocess_call(args,mode,channel,alphabet,eps,model=None):
    command=[sys.executable,str(Path(__file__).resolve()),mode,
             "--out",str(Path(args.out).resolve()),
             "--channel",channel,"--alphabet",str(alphabet),"--eps",str(eps),
             "--window",str(args.window),"--stride",str(args.stride),
             "--quantile",str(args.quantile)]
    if model:command+=["--model",str(model)]
    start=time.monotonic()
    try:
        p=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                         text=True,timeout=args.timeout)
        answers=[x[len(PREFIX):] for x in p.stdout.splitlines() if x.startswith(PREFIX)]
        if p.returncode!=0 or not answers:
            result=dict(channel=channel,alphabet_requested=alphabet,
                requested_eps=eps,status="native_failed",
                returncode=p.returncode,stderr_tail=p.stderr[-900:],
                stdout_tail=p.stdout[-300:])
        else:
            result=json.loads(answers[-1])
    except subprocess.TimeoutExpired:
        result=dict(channel=channel,alphabet_requested=alphabet,
            requested_eps=eps,status="timeout",timeout_seconds=args.timeout)
    result["wall_seconds"]=round(time.monotonic()-start,3)
    return result

def csv_write(path,items):
    if not items: return
    keys=sorted(set().union(*(x.keys() for x in items)))
    with path.open("w",newline="") as fd:
        writer=csv.DictWriter(fd,fieldnames=keys)
        writer.writeheader()
        writer.writerows(items)

def parent(args):
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    channels=[c.strip() for c in args.channels.split(",") if c.strip()]
    alphabets=[int(x) for x in args.alphabets.split(",")]
    epsilons=sorted(set(float(x) for x in args.eps_values.split(",")))
    if not channels or not epsilons or any(not(0<e<1) for e in epsilons):
        raise ValueError("Invalid channels or epsilon grid")
    if any(k not in (2,3,4) for k in alphabets):
        raise ValueError("Alphabet must be 2,3 or 4")
    all_grid=[]
    winners=[]
    summary=[]
    for alphabet in alphabets:
        for channel in channels:
            (_,_,_,_,fq,_,_,k,edges,entropy,used,majority)=train_and_quantize(channel,alphabet)
            rows=[]
            if used<2:
                reason=f"Only 1 training symbol (majority={majority:.3f}); epsilon cannot recover predictive states"
                print("DEGENERATE",channel,"alphabet",alphabet,reason,flush=True)
                for eps in epsilons:
                    rows.append(dict(channel=channel,alphabet_requested=alphabet,
                        alphabet_realized=k,n_distinct_training_symbols=used,
                        fit_entropy_bits=entropy,majority_symbol_fraction=majority,
                        requested_eps=eps,status="degenerate_training"))
            else:
                for eps in epsilons:
                    entry=subprocess_call(args,"infer",channel,alphabet,eps)
                    rows.append(entry)
                    print("INFER",json.dumps({key:entry.get(key) for key in
                        ["channel","alphabet_requested","requested_eps","actual_eps",
                         "n_states","status","wall_seconds"]}),flush=True)
            all_grid.extend(rows)
            # Model selection uses the train-fit data only. Choose the *fewest*
            # nontrivial states (>=2), then prefer larger epsilon on ties.
            eligible=[r for r in rows if r.get("status")=="success"
                      and args.min_states<=r.get("n_states",0)<=args.max_states]
            winner=min(eligible,key=lambda r:(r["n_states"],-r["requested_eps"])) if eligible else None
            if winner:
                scored=subprocess_call(args,"score",channel,alphabet,
                                       winner["requested_eps"],winner["model_path"])
                merged={**winner,**{("score_"+key):value for key,value in scored.items()}}
                merged["selection_status"]="model_selected"
                merged["scoring_status"]=scored["status"]
                winners.append(merged)
                print("SELECTED",json.dumps({"channel":channel,"alphabet":alphabet,
                    "eps":winner["requested_eps"],"states":winner["n_states"],
                    "score_status":scored["status"],
                    "onset_event_recall":scored.get("onset_event_recall"),
                    "score_error":scored.get("stderr_tail","")[-300:]}),flush=True)
                status="selected"
            else:
                status="none_found" if used>1 else "degenerate_training"
                print("NO_ELIGIBLE_MODEL",channel,"alphabet",alphabet,
                      "attempts",len(epsilons),"status",status,flush=True)
            summary.append(dict(channel=channel,alphabet_requested=alphabet,
                alphabet_realized=k,fit_entropy_bits=entropy,
                distinct_training_symbols=used,
                majority_symbol_fraction=majority,
                n_epsilon_attempts=len(rows),
                n_successful_inferences=sum(r["status"]=="success" for r in rows),
                n_nontrivial=sum(r.get("status")=="success" and r.get("n_states",0)>=2 for r in rows),
                selection_status=status,
                selected_epsilon=(winner["requested_eps"] if winner else None),
                selected_states=(winner["n_states"] if winner else None),
                scoring_status=(winners[-1]["scoring_status"] if winner else None)))
            # Save incremental checkpoints for interrupted runs.
            csv_write(out/"epsilon_grid.csv",all_grid)
            csv_write(out/"selected.csv",winners)
            csv_write(out/"selection_summary.csv",summary)
    manifest=dict(channels=channels,alphabets=alphabets,epsilons=epsilons,
        selection="minimal states >=min_states, tie larger epsilon; training data only",
        min_states=args.min_states,max_states=args.max_states,
        window=args.window,stride=args.stride,quantile=args.quantile,
        n_selected=len(winners),n_scored=sum(r["scoring_status"]=="scored" for r in winners),
        n_with_no_model=sum(r["selection_status"]!="selected" for r in summary))
    (out/"run.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("SWEEP_SUMMARY",json.dumps(manifest),flush=True)
    if args.require_all and (len(winners)<len(channels)*len(alphabets) or manifest["n_scored"]!=len(winners)):
        raise SystemExit("Not every channel yielded a scored 2+ state generator")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",nargs="?",default="parent",
                        choices=["parent","infer","score"])
    parser.add_argument("--channels",default=",".join(DEFAULT_CHANNELS))
    parser.add_argument("--channel",default="P-1")
    parser.add_argument("--alphabets",default="2,4")
    parser.add_argument("--alphabet",type=int,default=4)
    parser.add_argument("--eps-values",default=DEFAULT_EPS)
    parser.add_argument("--eps",type=float,default=.05)
    parser.add_argument("--min-states",type=int,default=2)
    parser.add_argument("--max-states",type=int,default=500)
    parser.add_argument("--window",type=int,default=128)
    parser.add_argument("--stride",type=int,default=32)
    parser.add_argument("--quantile",type=float,default=.995)
    parser.add_argument("--timeout",type=int,default=20)
    parser.add_argument("--model",default="")
    parser.add_argument("--out",default="results/nasa_genesess_epsilon_sweep")
    parser.add_argument("--require-all",action="store_true")
    args=parser.parse_args()
    if args.min_states<2 or args.max_states<args.min_states:
        parser.error("Require 2 <= min_states <= max_states")
    if args.mode=="infer": infer_child(args)
    elif args.mode=="score":score_child(args)
    else:parent(args)

if __name__=="__main__":
    main()
