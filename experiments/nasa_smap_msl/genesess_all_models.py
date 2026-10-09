#!/usr/bin/env python3
"""Evaluate every native GenESeSS PFSA in the NASA epsilon sweep.

Reuse model bytes and epsilon_grid.csv from a previous training-only sweep.
Do NOT refit generators, tune epsilon against test labels, or replace GenESeSS
with Markov surrogates. Native Llk crashes/nonfinite outputs are recorded
separately by isolating each model in a subprocess.

Val 70-85% train: select best predictive model by native Llk score.
Val 85-100% train: calibrate anomaly thresholds and report quantile curves.
Both phases are from the *provided normal training recording*.
Test data/labels: retrospective evaluation ONLY, never model selection.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pilot import (DATA, DEFAULT_CHANNELS, event_intervals, evaluate,
                   load_series, sparse_to_causal)
from genesess_pilot import quantize, windows
from genesess_epsilon_sweep import epsilon_id, check_native_environment
from audit_alerts import audit_score

PREFIX="ALL_PFSA_SCORE="
QUANTILES=(.5,.8,.9,.95,.98,.99,.995,.999)
MIN_VALID_SELECTION=20
MIN_VALID_CALIBRATION=20

def record(message, item):
    print(message, json.dumps(item,allow_nan=False),flush=True)

def records_to_csv(file, data):
    if not data:
        pd.DataFrame().to_csv(file,index=False)
        return
    keys=sorted(set().union(*(d.keys() for d in data)))
    with file.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

def train_quantized(channel,alphabet):
    train,test=load_series(channel)
    pivot=int(.7*len(train))
    _,val=train[:pivot],train[pivot:]
    q,real_k,edges=quantize(train[:pivot],val,test,alphabet)
    return train,test,q,real_k,edges

def split_validation(scores):
    # First 70% of provided normal training fits GenESeSS; of remaining
    # normal data, allocate equal numbers of FINITE scored indices to
    # heldout predictive selection and threshold calibration.
    valid=np.flatnonzero(np.isfinite(scores))
    mid=len(valid)//2
    if mid<MIN_VALID_SELECTION or len(valid)-mid<MIN_VALID_CALIBRATION:
        raise RuntimeError("Insufficient validation scores for separate selection and calibration")
    return scores[valid[:mid]],scores[valid[mid:]]

def score_windows(x,model,window,stride):
    frames,ends=windows(x,window,stride)
    from zedsuite.zutil import Llk
    z=np.asarray(Llk(data=frames,pfsafile=str(model)).run(),dtype=float).ravel()
    if len(z)!=len(ends):
        raise RuntimeError("Wrong native Llk output length %d versus %d" % (len(z),len(ends)))
    if not np.isfinite(z).all():
        raise RuntimeError("Native Llk returned %d nonfinite windows" % int(np.count_nonzero(~np.isfinite(z))))
    return sparse_to_causal(len(x),ends,z)

def child(args):
    _,test,seq,k,edges=train_quantized(args.channel,args.alphabet)
    _,vq,tq=seq
    model=Path(args.model)
    if not model.is_file() or not model.stat().st_size:
        raise FileNotFoundError(model)
    val_scores=score_windows(vq,model,args.window,args.stride)
    selection,calibration=split_validation(val_scores)
    test_scores=score_windows(tq,model,args.window,args.stride)
    # Thresholds determined from the last half of the normal validation.
    thresholds={str(q):float(np.quantile(calibration,q)) for q in QUANTILES}
    out=Path(args.out)
    name=args.channel+"__k"+str(args.alphabet)+"__e"+epsilon_id(args.eps)
    scores_dir=out/"scores"
    scores_dir.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(scores_dir/(name+".npz"),
        test=test_scores,val=val_scores,thresholds=np.array(list(thresholds.values())),
        threshold_quantiles=np.array(QUANTILES))
    # Labels are first accessed AFTER native scores and thresholds finalized.
    annotations=pd.read_csv(DATA/"labeled_anomalies.csv")
    ivals=event_intervals(args.channel,len(test_scores),annotations)
    curves=[]
    for q in QUANTILES:
        threshold=thresholds[str(q)]
        audit=audit_score(test_scores,threshold,ivals)
        curves.append(dict(channel=args.channel,alphabet=args.alphabet,
            epsilon=args.eps,n_states=args.n_states,quantile=q,threshold=threshold,
            event_hits=audit["event_hits"],onset_hits=audit["onset_hits"],
            events=audit["annotated_events"],
            event_hit_recall=audit["event_recall"],
            onset_event_recall=audit["onset_event_recall"],
            non_event_alarm_fraction=audit["non_event_alarm_fraction"],
            false_alarm_onsets_per_1000=audit["false_alarm_onsets_per_1000"],
            overall_alarm_fraction=audit["overall_alarm_fraction"],
            false_alarm_points=audit["false_alarm_points"],
            negative_points=audit["negative_points"]))
    curdir=out/"curves"
    curdir.mkdir(exist_ok=True,parents=True)
    records_to_csv(curdir/(name+".csv"),curves)
    record(PREFIX,dict(channel=args.channel,alphabet=args.alphabet,
        epsilon=args.eps,n_states=args.n_states,status="success",
        select_heldout_llk=float(np.mean(selection)),
        calibration_llk=float(np.mean(calibration)),
        calibration_n=len(calibration),heldout_n=len(selection),
        score_path=str(scores_dir/(name+".npz")),
        curve_path=str(curdir/(name+".csv")),
        event_count=len(ivals)))

def call_child(args,entry,root):
    # Model path in the old grid was absolute and belongs to CI runner;
    # relative path reconstructed from the stable model-file layout.
    channel=str(entry["channel"])
    alpha=int(entry["alphabet_requested"])
    eps=float(entry["requested_eps"])
    model=Path(root)/"models"/str(alpha)/channel/("eps_"+epsilon_id(eps))/"inferred.pfsa"
    command=[sys.executable,str(Path(__file__).resolve()),"child",
        "--out",str(Path(args.out).resolve()),
        "--channel",channel,"--alphabet",str(alpha),
        "--eps",str(eps),"--n-states",str(int(entry["n_states"])),
        "--model",str(model.resolve()),
        "--window",str(args.window),"--stride",str(args.stride)]
    try:
        p=subprocess.run(command,capture_output=True,text=True,timeout=args.timeout)
        messages=[s[len(PREFIX):] for s in p.stdout.splitlines() if s.startswith(PREFIX)]
        if p.returncode==0 and messages:
            out=json.loads(messages[-1])
        else:
            out=dict(channel=channel,alphabet=alpha,epsilon=eps,
                n_states=int(entry["n_states"]),status="native_failed",
                returncode=p.returncode,stderr_tail=p.stderr[-1400:],
                stdout_tail=p.stdout[-500:])
    except subprocess.TimeoutExpired:
        out=dict(channel=channel,alphabet=alpha,epsilon=eps,
            n_states=int(entry["n_states"]),status="timeout")
    return out

def load_candidates(args):
    roots=[Path(x.strip()) for x in args.model_roots.split(",") if x.strip()]
    rows={}
    for root in roots:
        file=root/"epsilon_grid.csv"
        if not file.is_file():raise FileNotFoundError(file)
        data=pd.read_csv(file)
        good=data[(data.status=="success") & (data.n_states>=1)]
        for _,v in good.iterrows():
            key=(str(v.channel),int(v.alphabet_requested),float(v.requested_eps))
            # Last model root can overwrite duplicates when supplied;
            # no epsilon/state fitted using test-label information.
            rows[key]=(v.to_dict(),root)
    return rows

def curve_for(result):
    path=Path(result["curve_path"])
    return pd.read_csv(path)

def representative_rows(results):
    by={}
    for x in results:
        if x.get("status")=="success":
            by.setdefault((x["channel"],x["alphabet"]),[]).append(x)
    choices=[]
    for (channel,alpha),v in sorted(by.items()):
        # Three prespecified selection rules; ties deterministic. Do not
        # examine event labels here; only state counts, epsilon and the
        # held-out normal Llk (first half of validation).
        nontrivial=[x for x in v if x["n_states"]>=2]
        if not nontrivial:continue
        smallest=min(nontrivial,key=lambda x:(x["n_states"],-x["epsilon"]))
        predictive=min(nontrivial,key=lambda x:(x["select_heldout_llk"],x["n_states"],-x["epsilon"]))
        # Within the region of smaller epsilon than the smallest model,
        # select the *most complex* successfully inferred model.
        complex_opts=[x for x in nontrivial if
                      x["epsilon"] < smallest["epsilon"] and
                      x["n_states"]>smallest["n_states"]]
        complex_model=(max(complex_opts,key=lambda x:(x["n_states"],-x["epsilon"]))
                       if complex_opts else None)
        for name,model in [("min_nontrivial",smallest),
                           ("best_heldout_llk",predictive),
                           ("complex_smaller_epsilon",complex_model)]:
            if model is None:continue
            curve=curve_for(model)
            fixed=curve.iloc[int(np.argmin(np.abs(curve["quantile"].to_numpy()-.995)))]
            choices.append(dict(selection=name,channel=channel,alphabet=alpha,
                epsilon=model["epsilon"],states=model["n_states"],
                heldout_llk=model["select_heldout_llk"],
                calibration_llk=model["calibration_llk"],
                events=int(fixed["events"]),onset_hits=int(fixed["onset_hits"]),
                onset_recall=float(fixed["onset_event_recall"]),
                non_event_alarm_fraction=float(fixed["non_event_alarm_fraction"]),
                onset_false_alarms_per_1000=float(fixed["false_alarm_onsets_per_1000"]),
                curve_path=str(model["curve_path"])))
    return choices

def plots(results,choices,out):
    figdir=out/"figures"
    figdir.mkdir(parents=True,exist_ok=True)
    good=pd.DataFrame([r for r in results if r.get("status")=="success"])
    if good.empty:return
    for (channel,alpha),group in good.groupby(["channel","alphabet"]):
        group=group.sort_values("epsilon")
        fig,axes=plt.subplots(2,1,figsize=(8,6),sharex=True)
        axes[0].plot(group.epsilon,group.n_states,"o-")
        axes[0].set_ylabel("Inferred states")
        axes[1].plot(group.epsilon,group.select_heldout_llk,"o-")
        axes[1].set_ylabel("Selection holdout mean Llk")
        axes[1].set_xlabel("GenESeSS epsilon")
        axes[1].set_xscale("log")
        fig.suptitle("NASA %s / %d-symbol models" %(channel,alpha))
        fig.tight_layout()
        fig.savefig(figdir/("%s_k%d_eps_profile.png"%(channel,alpha)),dpi=130)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,5))
        selection=[r for r in choices if r["channel"]==channel and r["alphabet"]==alpha]
        for item in selection:
            curve=pd.read_csv(item["curve_path"])
            ax.plot(100*curve.non_event_alarm_fraction,curve.onset_event_recall,
                    "o-",label=("%s: eps=%.4g, %d states"%
                                (item["selection"],item["epsilon"],item["states"])))
        ax.set(xlabel="Non-event time under alarm on test (%)",
               ylabel="New-alarm event recall",title="%s %d-symbol: retrospective thresholds"%
                  (channel,alpha))
        ax.set_ylim(-.02,1.02)
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figdir/("%s_k%d_recall_vs_alarm.png"%(channel,alpha)),dpi=130)
        plt.close(fig)

def baseline_compare(args,choices,out):
    base=Path(args.baseline)
    if not (base/"scores/P-1.npz").exists():
        print("NO_BASELINE_ARTIFACT",base,flush=True)
        return
    annotations=pd.read_csv(DATA/"labeled_anomalies.csv")
    rows=[]
    for channel in sorted(set(r["channel"] for r in choices)):
        score_file=base/"scores"/(channel+".npz")
        if not score_file.is_file(): continue
        _,test=load_series(channel)
        ivals=event_intervals(channel,len(test),annotations)
        with np.load(score_file) as z:
            for method in ("matrix_profile","zscore","cusum","pfsa"):
                if "score_"+method not in z:continue
                score=z["score_"+method]
                threshold=float(z["threshold_"+method][0])
                audit=audit_score(score,threshold,ivals)
                rows.append(dict(channel=channel,method=method,alphabet=None,
                    epsilon=None,states=None,events=len(ivals),
                    onset_recall=audit["onset_event_recall"],
                    onset_hits=audit["onset_hits"],
                    non_event_alarm_fraction=audit["non_event_alarm_fraction"],
                    false_alarm_onsets_per_1000=audit["false_alarm_onsets_per_1000"],
                    note="saved original pilot, calibrated at quantile 0.995"))
    for c in choices:
        if c["selection"] in ("min_nontrivial","best_heldout_llk","complex_smaller_epsilon"):
            rows.append(dict(channel=c["channel"],method="GenESeSS "+c["selection"],
                alphabet=c["alphabet"],epsilon=c["epsilon"],states=c["states"],
                events=c["events"],onset_recall=c["onset_recall"],
                onset_hits=c["onset_hits"],
                non_event_alarm_fraction=c["non_event_alarm_fraction"],
                false_alarm_onsets_per_1000=c["onset_false_alarms_per_1000"],
                note="calibrated on independent later-half of normal validation"))
    records_to_csv(out/"baseline_comparison.csv",rows)
    p1=[r for r in rows if r["channel"]=="P-1"]
    record("P1_COMPARISON",p1)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",nargs="?",choices=["parent","child"],default="parent")
    parser.add_argument("--model-roots",default="results/previous_main,results/previous_edge")
    parser.add_argument("--baseline",default="results/previous_baseline")
    parser.add_argument("--channel",default="P-1")
    parser.add_argument("--alphabet",type=int,default=4)
    parser.add_argument("--eps",type=float,default=.1)
    parser.add_argument("--n-states",type=int,default=2)
    parser.add_argument("--model",default="")
    parser.add_argument("--window",type=int,default=128)
    parser.add_argument("--stride",type=int,default=32)
    parser.add_argument("--timeout",type=int,default=20)
    parser.add_argument("--require-all",action="store_true")
    parser.add_argument("--out",default="results/nasa_genesess_all_models")
    args=parser.parse_args()
    if args.mode=="child":
        child(args)
        return
    check_native_environment()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    all_models=load_candidates(args)
    rows=[]
    errors=[]
    for i, (key,(entry,root)) in enumerate(sorted(all_models.items())):
        result=call_child(args,entry,root)
        rows.append(result)
        if result["status"]!="success":errors.append(result)
        if (i+1)%10==0 or result["status"]!="success":
            record("SCORED_PROGRESS",dict(complete=i+1,total=len(all_models),
                errors=len(errors),channel=result["channel"],
                alphabet=result["alphabet"],epsilon=result["epsilon"],
                status=result["status"]))
        records_to_csv(out/"all_model_scores.csv",rows)
    selected=representative_rows(rows)
    records_to_csv(out/"selection_comparison.csv",selected)
    plots(rows,selected,out)
    baseline_compare(args,selected,out)
    summary=dict(n_models=len(all_models),
        n_scored=sum(r["status"]=="success" for r in rows),
        n_failed=len(errors),n_selections=len(selected),
        n_channel_alphabets=len(set((r["channel"],r["alphabet"])
                                    for r in rows if r["status"]=="success")),
        selection_rules=["min_nontrivial","best_heldout_llk","complex_smaller_epsilon"],
        epsilon_selection_uses_test_labels=False,
        caveats=["Different alphabets produce non-comparable raw Llk values",
                 "Test labels used only for retrospective event-recall curves",
                 "Baseline stored pilot is calibrated on full 30% validation, whereas GenESeSS uses later half",
                 "No fixed quantile guarantees matched realized test false-alarm rate"])
    (out/"run.json").write_text(json.dumps(summary,indent=2)+"\n")
    record("ALL_MODEL_SUMMARY",summary)
    if args.require_all and errors:raise SystemExit("Some models had native score failures")
    # Deliberately not require all: failed native likelihood is a meaningful
    # scientific result, not grounds to discard successful comparisons.

if __name__=="__main__":
    main()
