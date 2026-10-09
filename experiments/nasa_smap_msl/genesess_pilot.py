#!/usr/bin/env python3
"""Genuine zedsuite.GenESeSS NASA experiment at epsilon 0.05 and 0.10.

Native inference/scoring are run in separate processes per channel/epsilon,
because the historical Cython zedsuite may segfault on some inputs. Never
replace a failed inference with a Markov surrogate.

Uses a fixed train-only quantizer; fits GenESeSS using only normal train[:70%],
calibrates likelihood score threshold using the normal train[70%:] and scores
the disjoint test recording. Anomaly annotations used ONLY after score output.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

# Re-use exactly the NASA evaluation/plotting routines from modern pilot.
from pilot import (DATA, DEFAULT_CHANNELS, load_series,
                   event_intervals, evaluate, sparse_to_causal,
                   threshold_from_validation, plot_channel)

def quantize(fit, val, test, alphabet=2):
    if alphabet not in (2,3,4): raise ValueError("Alphabet must be 2,3,4")
    edges=np.unique(np.quantile(fit,np.arange(1,alphabet)/alphabet))
    if len(edges)==0: raise ValueError("Empty edges")
    out=[np.digitize(x,edges).astype(np.int32) for x in (fit,val,test)]
    return out,len(edges)+1,edges

def windows(symbols,window,stride):
    if len(symbols)<window:
        raise ValueError(f"Need {window} symbols; got {len(symbols)}")
    starts=np.arange(0,len(symbols)-window+1,stride,dtype=int)
    # Explicit batch, each row a full length-W sequence; no padding.
    # Similar to original Patternly's 'data_dir=row' convention.
    arr=np.stack([symbols[s:s+window] for s in starts],axis=0)
    return pd.DataFrame(arr),starts+window-1

def do_trial(args):
    from zedsuite.genesess import GenESeSS
    from zedsuite.zutil import Llk

    begin=time.monotonic()
    train,test=load_series(args.channel)
    pivot=int(.7*len(train))
    fit,val=train[:pivot],train[pivot:]
    (fit_q,val_q,test_q),k,edges=quantize(fit,val,test,args.alphabet)

    model_dir=Path(args.out)/"models"/args.channel/f"eps_{args.eps:.2f}"
    model_dir.mkdir(parents=True,exist_ok=True)
    model_file=model_dir/"inferred.pfsa"
    # One row of normal training symbols. Direct native GenESeSS call;
    # no Patternly clustering or fixed-order Markov approximation.
    training=pd.DataFrame([fit_q.tolist()])
    alg=GenESeSS(
        data=training,
        outfile=str(model_file),
        data_type="symbolic",
        data_dir="row",
        force=True,
        eps=args.eps,
    )
    found=alg.run()
    if not found or not model_file.is_file() or not model_file.stat().st_size:
        raise RuntimeError("Native GenESeSS returned no usable PFSA")
    matrix=np.asarray(alg.probability_morph_matrix)
    n_states=int(matrix.shape[0]) if matrix.ndim else 0
    actual_eps=float(alg.epsilon_used)
    if not np.isfinite(actual_eps):raise ValueError("GenESeSS epsilon_used not finite")
    if n_states<1:raise ValueError("Empty morph matrix")

    def get_scores(x):
        frame,ends=windows(x,args.window,args.stride)
        ll=np.asarray(Llk(data=frame,pfsafile=str(model_file)).run(),dtype=float).ravel()
        if len(ll)!=len(ends):raise RuntimeError(f"Llk returned {len(ll)} scores, expected {len(ends)}")
        if not np.isfinite(ll).all():
            raise ValueError(f"Nonfinite likelihood scores: {np.count_nonzero(~np.isfinite(ll))}")
        return sparse_to_causal(len(x),ends,ll)

    cal_scores=get_scores(val_q)
    threshold=threshold_from_validation(cal_scores,args.cal_quantile)
    test_scores=get_scores(test_q)

    metadata=dict(
        channel=args.channel,eps_requested=args.eps,eps_used=actual_eps,
        alphabet=k,alphabet_requested=args.alphabet,quantizer_edges=edges.tolist(),
        n_states=n_states,window=args.window,stride=args.stride,
        fit_samples=len(fit_q),calibration_samples=len(val_q),
        test_samples=len(test_q),threshold=threshold,
        model_path=str(model_file),status="success",
        elapsed_seconds=round(time.monotonic()-begin,3))
    # Preserve all test scores for independent post-hoc evaluation.
    out=Path(args.out)
    (out/"scores").mkdir(parents=True,exist_ok=True)
    path=out/"scores"/f"{args.channel}_eps{args.eps:.2f}.npz"
    np.savez_compressed(path,score=test_scores,calibration_score=cal_scores,
                        threshold=np.array([threshold]),eps=np.array([args.eps]))

    # Read annotation data only after model, calibration and test scores
    # are finalized. No test label selection or test FPR tuning.
    annotations=pd.read_csv(DATA/"labeled_anomalies.csv")
    intervals=event_intervals(args.channel,len(test),annotations)
    evaluation=evaluate(test_scores,threshold,intervals)
    plot_channel(args.channel,test,intervals,
                {f"GenESeSS e={args.eps:.2f}":(test_scores,threshold)},
                 out/"plots"/f"{args.channel}_eps{args.eps:.2f}.png")
    from audit_alerts import audit_score
    audit=audit_score(test_scores,threshold,intervals)
    result={**metadata,**evaluation,
            "onset_event_recall":audit["onset_event_recall"],
            "onset_hits":audit["onset_hits"],
            "non_event_alarm_fraction":audit["non_event_alarm_fraction"],
            "false_alarm_onsets_per_1000":audit["false_alarm_onsets_per_1000"]}
    return result

def run_parent(args):
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    (out/"plots").mkdir(exist_ok=True)
    channels=[c.strip() for c in args.channels.split(",") if c.strip()]
    eps_values=[float(v) for v in args.eps_values.split(",")]
    for eps in eps_values:
        if not 0<eps<1:raise ValueError("epsilon must be in (0,1)")
    rows=[]
    for channel in channels:
        for eps in eps_values:
            command=[sys.executable,str(Path(__file__).resolve()),
              "child","--channel",channel,"--eps",str(eps),
              "--alphabet",str(args.alphabet),
              "--window",str(args.window),"--stride",str(args.stride),
              "--cal-quantile",str(args.cal_quantile),"--out",str(out.resolve())]
            t=time.monotonic()
            try:
                p=subprocess.run(command,capture_output=True,text=True,
                                  timeout=args.timeout)
                observed=[line.split("GENESESS_TRIAL=",1)[1]
                          for line in p.stdout.splitlines()
                          if line.startswith("GENESESS_TRIAL=")]
                if p.returncode==0 and observed:
                    row=json.loads(observed[-1])
                else:
                    row=dict(channel=channel,eps_requested=eps,status="failed",
                            returncode=p.returncode,stderr_tail=p.stderr[-1200:],
                            stdout_tail=p.stdout[-1200:])
            except subprocess.TimeoutExpired:
                row=dict(channel=channel,eps_requested=eps,status="timeout",
                         timeout_seconds=args.timeout)
            row["wall_seconds"]=round(time.monotonic()-t,3)
            rows.append(row)
            print("TRIAL",json.dumps(row,allow_nan=False),flush=True)
            with (out/"trials.jsonl").open("a") as f:
                f.write(json.dumps(row,allow_nan=False)+"\n")
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (out/"per_channel.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary=[]
    for eps in eps_values:
        group=[r for r in rows if r["eps_requested"]==eps]
        good=[r for r in group if r["status"]=="success"]
        events=sum(x["events_total"] for x in good)
        summary.append(dict(
          eps=eps,attempted=len(group),success=len(good),
          failed=len(group)-len(good),
          pooled_event_recall=(sum(x["events_detected"] for x in good)/events if events else None),
          pooled_onset_event_recall=(sum(x["onset_hits"] for x in good)/events if events else None),
          median_non_event_alarm_fraction=(float(np.median([x["non_event_alarm_fraction"] for x in good])) if good else None),
          median_states=(float(np.median([x["n_states"] for x in good])) if good else None)))
    pd.DataFrame(summary).to_csv(out/"summary.csv",index=False)
    (out/"run.json").write_text(json.dumps(
      dict(channels=channels,eps_values=eps_values,window=args.window,stride=args.stride,
           alphabet=args.alphabet,threshold_quantile=args.cal_quantile,
           native_engine="zedsuite.GenESeSS + zedsuite.zutil.Llk",
           isolation="One subprocess per channel and epsilon"),indent=2)+"\n")
    print("SUMMARY",json.dumps(summary),flush=True)
    print("WROTE",out.resolve(),flush=True)
    if args.require_all and any(r["status"]!="success" for r in rows):
        raise SystemExit("Some GenESeSS trials failed; see per_channel.csv")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",nargs="?",choices=["parent","child"],default="parent")
    p.add_argument("--channels",default=",".join(DEFAULT_CHANNELS))
    p.add_argument("--eps-values",default="0.05,0.1")
    p.add_argument("--channel",default="P-1")
    p.add_argument("--eps",type=float,default=0.1)
    p.add_argument("--window",type=int,default=128)
    p.add_argument("--stride",type=int,default=32)
    p.add_argument("--alphabet",type=int,default=2)
    p.add_argument("--cal-quantile",type=float,default=.995)
    p.add_argument("--timeout",type=int,default=90)
    p.add_argument("--out",default="results/nasa_genesess")
    p.add_argument("--require-all",action="store_true")
    args=p.parse_args()
    if args.window<32 or args.stride<1 or args.alphabet not in (2,3,4):
        p.error("Invalid window / stride / alphabet")
    if args.mode=="child":
        result=do_trial(args)
        print("GENESESS_TRIAL="+json.dumps(result,allow_nan=False),flush=True)
    else:run_parent(args)

if __name__=="__main__":
    main()
