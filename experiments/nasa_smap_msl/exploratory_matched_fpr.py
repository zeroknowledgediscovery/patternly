#!/usr/bin/env python3
"""RETROSPECTIVE ORACLE equal-test-FPR diagnostic for independent NASA channels.

IMPORTANT: selects thresholds using labeled NON-ANOMALOUS test samples,
so this is NOT a prospective detector result or a valid operating protocol.
It is an explicitly test-label-informed, descriptive control for differing
realized false-alarm exposures under normal-train calibration.
Never use these thresholds to report prospective performance.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pilot import DATA, load_series, event_intervals
from independent_replication import metrics, dumpcsv, bootstrap_results

TARGETS=(.01,.02,.05,.10)

def get_threshold(scores,ivals,target):
    valid=np.isfinite(scores)
    mask=valid.copy()
    for start,end in ivals:mask[start:end]=False
    negatives=scores[mask]
    if len(negatives)<20:raise RuntimeError("Too few test negative points")
    # "higher" picks an observed score at/above the requested quantile.
    # Threshold comparison is strictly greater-than, making the result
    # conservative in cases with many ties.
    return float(np.quantile(negatives,1-target,method="higher"))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--original",default="results/nasa_independent_replication")
    parser.add_argument("--out",default="results/nasa_independent_replication_oracle")
    args=parser.parse_args()
    original=Path(args.original)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    status=pd.read_csv(original/"channel_status.csv")
    labels=pd.read_csv(DATA/"labeled_anomalies.csv")
    rows=[]
    event_rows=[]
    for _,item in status[status.status=="success"].iterrows():
        channel=item.channel
        _,test=load_series(channel)
        ivals=event_intervals(channel,len(test),labels)
        with np.load(original/"scores"/(channel+".npz")) as z:
            for budget in TARGETS:
                flags={}
                for name in ("GenESeSS","marginal"):
                    score=z[name]
                    threshold=get_threshold(score,ivals,budget)
                    value=metrics(score,threshold,ivals)
                    if value["false_alarm_fraction"]>budget+1e-10:
                        raise ValueError("Budget violated %s %s"%(channel,name))
                    flags[name]=value.pop("event_onset_flags")
                    rows.append(dict(channel=channel,n_states=int(item.states),
                         method=name,budget=budget,threshold=threshold,
                         oracle_test_labels_used=True,**value))
                for j,ival in enumerate(ivals):
                    event_rows.append(dict(channel=channel,budget=budget,
                        n_states=int(item.states),event_index=j,
                        GenESeSS_onset=int(flags["GenESeSS"][j]),
                        marginal_onset=int(flags["marginal"][j])))
    dumpcsv(out/"oracle_channel_budget.csv",rows)
    dumpcsv(out/"oracle_per_event.csv",event_rows)
    results=[]
    for budget in TARGETS:
        for stratum in ("all","multi","three"):
            a=bootstrap_results(rows,budget,stratum)
            if a:
                a["test_negative_labels_used_to_calibrate"]=True
                results.append(a)
                print("ORACLE_PAIRED",json.dumps(a),flush=True)
    (out/"oracle_summary.json").write_text(json.dumps(results,indent=2)+"\n")
    meta=dict(
        description="Test-label-conditioned equal-realized-FPR descriptive analysis",
        prospective=False,
        warning="DO NOT treat these test-label-oracle thresholds as deployable or blind results",
        note="Ties yield conservative rates at or below each target; no randomization",
        n_channels=status.status.eq("success").sum(),
        targets=TARGETS)
    (out/"WARN_ORACLE_NOT_PROSPECTIVE.json").write_text(json.dumps(meta,indent=2)+"\n")
    print("ORACLE_DONE",json.dumps(meta,default=int),flush=True)

if __name__=="__main__":main()
