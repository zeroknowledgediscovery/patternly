#!/usr/bin/env python3
"""Aggregate seven-channel causal early-warning replications, visibly retaining
failed native methods / channels rather than silently omitting them."""
from pathlib import Path
import argparse,json,csv
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPECTED=["P-1","E-13","E-1","E-10","G-7","F-7","T-1"]
METHODS=("native_LSmash_default","native_LSmash_GenESeSS",
         "CUSTOM_marginal_JS","CUSTOM_bigram_JS")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="results/nasa_concat_preemption")
    args=p.parse_args()
    base=Path(args.root)
    all_df=[]
    methods=[]
    for ch in EXPECTED:
        folder=base/ch
        meta=folder/"metadata.json"
        if not meta.is_file():
            reason=(folder/"ERROR.txt").read_text()[:800] if (folder/"ERROR.txt").exists() else "Missing complete result"
            for method in METHODS:
                methods.append(dict(channel=ch,method=method,status="FAILED_CHANNEL: "+reason))
            continue
        j=json.loads(meta.read_text())
        for method in METHODS:
            methods.append(dict(channel=ch,method=method,
                                status=j["methods_status"].get(method,"MISSING_METHOD"),
                                alphabet_size=j["alphabet_size"],
                                n_scored=j["n_scored"],events=len(j["events"]),
                                seam=j["seam"]))
        dpath=folder/"preemptive_summary.csv"
        if dpath.is_file() and dpath.stat().st_size>0:
            d=pd.read_csv(dpath)
            all_df.append(d)
    table=pd.DataFrame(methods)
    table.to_csv(base/"method_status.csv",index=False)
    if all_df:
        results=pd.concat(all_df,ignore_index=True)
    else:
        results=pd.DataFrame()
    results.to_csv(base/"all_horizons_thresholds.csv",index=False)
    if results.empty:
        print("NO_METHOD_RESULTS",table.to_string(index=False),flush=True)
        return
    primary=results[np.isclose(results.calibration_quantile,.99)&
                     (results.preemption_horizon==212)].copy()
    primary.to_csv(base/"primary_q99_h212.csv",index=False)
    aggregate=(primary.groupby("method")
       .agg(channels=("channel","nunique"),
            total_events=("events_total","sum"),
            preemptive_events=("advance_hits","sum"),
            post_onset_events=("post_onset_hits","sum"),
            average_clean_occupancy=("clean_alarm_occupancy","mean"),
            average_false_alarm_onsets_per_1000=("clean_alarm_onsets_per_1000","mean"))
       .reset_index())
    aggregate["early_event_recall"]=aggregate.preemptive_events/aggregate.total_events
    aggregate.to_csv(base/"method_aggregate.csv",index=False)
    print("CHANNEL_METHOD_STATUS\n"+table.to_string(index=False),flush=True)
    print("PRIMARY_PER_CHANNEL\n"+primary[[
        "channel","method","advance_hits","events_total","post_onset_hits",
        "clean_alarm_occupancy","clean_alarm_onsets_per_1000"]].to_string(index=False),flush=True)
    print("METHOD_AGGREGATE\n"+aggregate.to_string(index=False),flush=True)
    fig,ax=plt.subplots(figsize=(13,5))
    order=primary.channel.drop_duplicates().tolist()
    x=np.arange(len(order))
    width=.2
    for idx,method in enumerate(METHODS):
        select=primary[primary.method==method].set_index("channel")
        vals=[]
        for ch in order:
            vals.append(select.loc[ch,"advance_hits"]/select.loc[ch,"events_total"]
                        if ch in select.index else np.nan)
        ax.bar(x+(idx-1.5)*width,vals,width,
               label=method.replace("native_LSmash_","LSmash ").replace("CUSTOM_",""))
    ax.set_xticks(x,order)
    ax.set_ylim(0,1.06)
    ax.set(ylabel="Fraction of annotated episodes with new before-onset alarm",
           xlabel="Concatenated NASA channel",
           title="Causal NASA early warnings (99% train-calibration, 212-sample horizon)")
    ax.legend(fontsize=8,ncol=2)
    ax.grid(axis="y",alpha=.15)
    fig.tight_layout()
    fig.savefig(base/"seven_channel_preemption.png",dpi=175)
    plt.close(fig)
    print("AGGREGATE_RESULTS_READY",str(base.resolve()),flush=True)
if __name__=="__main__":
    main()
