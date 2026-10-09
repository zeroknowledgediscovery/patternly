#!/usr/bin/env python3
"""Scientific difficulty audit: is P-1 easy without GenESeSS or LSmash?

Baselines ONLY: raw-signal statistics, 4-symbol marginal Jensen-Shannon,
and first-order transition JS, implemented and transparently labeled here.
No simplified or replacement implementation of GenESeSS/LSmash.

Native LSmash baseline matrices are *loaded unchanged* from two independent
successful CI artifacts and compared against these simpler controls.

The output is a retrospective diagnosis of task difficulty, NOT a prospective
detector or scientific significance test: NASA test labels evaluate rankings
only and 1659 windows overlap heavily.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pilot import DATA, load_series, event_intervals

def js_div(p,q):
    # Explicit, hand-implemented categorical statistical baseline, NOT LSmash.
    # scipy.spatial.distance.jensenshannon returns sqrt(JS); square it.
    return float(jensenshannon(p,q,base=2)**2)

def quantized_windows(train,test,win,stride):
    if len(test)<win: raise RuntimeError("Too few observations")
    edges=np.unique(np.quantile(train,[.25,.5,.75]))
    if len(edges)!=3:
        raise RuntimeError("Expected 4-symbol P-1 train quantiles")
    q_train=np.digitize(train,edges).astype(int)
    q_test=np.digitize(test,edges).astype(int)
    starts=np.arange(0,len(test)-win+1,stride,dtype=int)
    xs=np.stack([test[s:s+win] for s in starts])
    qs=np.stack([q_test[s:s+win] for s in starts])
    return starts,xs,qs,q_train


def mean_js_to_all_test(P,chunk_size=48):
    """Custom pairwise Jensen-Shannon *baseline*, not the native LSmash method.

    Like the mean native LSmash-to-all score, this is transductive:
    it uses all test windows without their labels as a reference ensemble.
    """
    entropy=-np.sum(P*np.log2(P),axis=1)
    output=np.empty(len(P),dtype=float)
    for start in range(0,len(P),chunk_size):
        stop=min(start+chunk_size,len(P))
        avg=(P[start:stop,None,:]+P[None,:,:])/2
        entropy_mix=-np.sum(avg*np.log2(avg),axis=2)
        divergence=entropy_mix-.5*entropy[start:stop,None]-.5*entropy[None,:]
        output[start:stop]=np.mean(np.maximum(divergence,0),axis=1)
    return output

def baselines(train,test,win,stride):
    starts,X,Q,q_train=quantized_windows(train,test,win,stride)
    k=4
    scale=max(np.std(train)*.1,1.4826*np.median(np.abs(train-np.median(train))),1e-8)
    center=np.median(train)
    z=np.abs((X-center)/scale)
    p_ref=np.bincount(q_train,minlength=k).astype(float)+.5
    p_ref/=p_ref.sum()
    # Prior transitions in training; reference-only. The small smoothing
    # constant is a baseline choice, not a hidden native PFSA.
    trans_train=k*q_train[:-1]+q_train[1:]
    t_ref=np.bincount(trans_train,minlength=k*k).astype(float)+.5
    t_ref/=t_ref.sum()
    marg=np.stack([np.bincount(q,minlength=k) for q in Q],axis=0).astype(float)+.5
    marg/=marg.sum(axis=1,keepdims=True)
    tr=np.stack([np.bincount(k*q[:-1]+q[1:],minlength=k*k) for q in Q],axis=0).astype(float)+.5
    tr/=tr.sum(axis=1,keepdims=True)
    result={
        "raw_abs_window_mean_z":np.abs(X.mean(axis=1)-center)/scale,
        "raw_mean_absolute_z":z.mean(axis=1),
        "raw_peak_absolute_z":z.max(axis=1),
        "raw_window_standard_deviation":X.std(axis=1),
        "raw_mean_absolute_step_change":np.abs(np.diff(X,axis=1)).mean(axis=1),
        "marginal_symbol_JS_vs_train":np.array([js_div(p,p_ref) for p in marg]),
        "marginal_symbol_JS_mean_to_test":mean_js_to_all_test(marg),
        "first_order_symbol_JS_vs_train":np.array([js_div(p,t_ref) for p in tr]),
        "first_order_symbol_JS_mean_to_test":mean_js_to_all_test(tr),
    }
    return starts,result

def anomaly_flags(starts,win,events):
    return np.array([any(s<e and s+win>a for a,e in events) for s in starts],dtype=bool)

def summarize(name,method,score,starts,win,events,outrows):
    labels=anomaly_flags(starts,win,events)
    if score.shape!=(len(starts),):raise RuntimeError(f"Bad score array for {method}")
    if not np.isfinite(score).all():raise RuntimeError(f"Invalid score values for {method}")
    positive=int(labels.sum())
    negative=int(len(labels)-positive)
    if not positive or not negative:raise RuntimeError("No both-class evaluation")
    auc=float(roc_auc_score(labels,score))
    ap=float(average_precision_score(labels,score))
    ranking=np.argsort(-score,kind="stable")
    top=ranking[:positive]
    precision=float(labels[top].mean())
    event_coverage=sum(int(np.any((starts[top]<b)&(starts[top]+win>a))) for a,b in events)
    outrows.append(dict(view=name,method=method,n_windows=len(labels),
                        positive_overlap_windows=positive,negative_windows=negative,
                        auc=auc,average_precision=ap,
                        baseline_precision=positive/len(labels),
                        precision_at_num_positives=precision,
                        events_touched_in_top_k=int(event_coverage),
                        k=positive))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--native-root",required=True,help="Unpacked native sliding CI artifact root")
    ap.add_argument("--native-40-root",required=True,help="Unpacked native 40x40 CI artifact root")
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    native=Path(args.native_root); native40=Path(args.native_40_root)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    train,test=load_series("P-1")
    import pandas as pd
    events=event_intervals("P-1",len(test),pd.read_csv(DATA/"labeled_anomalies.csv"))
    rows=[]
    for view,win,stride in (("sliding_212_stride_5",212,5),("nonoverlap_212",212,212)):
        starts,base=baselines(train,test,win,stride)
        if view.startswith("sliding"):
            for key,dirname in (
                ("native_LSmash_default","nasa_p1_sliding_default"),
                ("native_LSmash_GenESeSS_projectors","nasa_p1_sliding_genesess")):
                native_matrix=np.load(native/dirname/"distance_matrix.npy")
                native_meta=json.loads((native/dirname/"metadata.json").read_text())
                assert native_meta["window_samples"]==win and native_meta["stride"]==stride
                assert native_matrix.shape==(len(starts),len(starts))
                base[key]=native_matrix.mean(axis=1)
        else:
            m=np.load(native40/"distance_matrix.npy")
            native_meta=json.loads((native40/"metadata.json").read_text())
            assert native_meta["channel"]=="P-1" and native_meta["window_length"]==win
            assert m.shape==(len(starts),len(starts))
            base["native_LSmash_default"]=m.mean(axis=1)
        for method,sc in base.items():
            summarize(view,method,np.asarray(sc),starts,win,events,rows)

        print("VIEW",view,"n",len(starts),"positive",anomaly_flags(starts,win,events).sum(),flush=True)

    import pandas as pd
    tab=pd.DataFrame(rows).sort_values(["view","auc"],ascending=[True,False])
    tab.to_csv(out/"retrospective_baselines.csv",index=False)
    for view,group in tab.groupby("view"):
        print("RETROSPECTIVE_AUC",view,"\n"+group[["method","auc","average_precision","precision_at_num_positives"]].to_string(index=False),flush=True)

    # Diagnostic plots. No labels used to select any hyperparameters.
    for view in tab.view.unique():
        t=tab[tab.view==view].sort_values("auc",ascending=True)
        fig,ax=plt.subplots(figsize=(10,4.5))
        ax.barh(t["method"],t["auc"])
        ax.set_xlim(0,1);ax.set_xlabel("Retrospective overlap-window AUROC")
        ax.set_title(f"P-1: easy baselines versus original native LSmash ({view})")
        ax.axvline(.5,color="gray",linestyle="--",alpha=.7)
        fig.tight_layout()
        fig.savefig(out/(view+"_auc.png"),dpi=160)
        plt.close(fig)
    summary=dict(
        frozen_code_ref="freeze/p1-native-sliding-20261009",
        source="Real NASA P-1 training/test dataset",
        evaluation="Retrospective ranking using known NASA anomaly intervals, no deployable threshold",
        no_genesess_lsmash_surrogates=True,
        methods_explicitly_classified={
            "native_LSmash_default":"Original compiled C++ LSmash output, from frozen CI",
            "native_LSmash_GenESeSS_projectors":"Original compiled C++ LSmash using GenESeSS fitted PFSAs, from frozen CI",
            "marginal_symbol_JS_vs_train":"Custom categorical Jensen-Shannon statistical baseline",
            "marginal_symbol_JS_mean_to_test":"Custom transductive pairwise JS, directly comparable by reference population to mean native LSmash",
            "first_order_symbol_JS_vs_train":"Custom first-order categorical Jensen-Shannon statistical baseline",
            "first_order_symbol_JS_mean_to_test":"Custom transductive first-order bigram JS baseline",
            "raw_*":"Simple custom raw telemetry statistics, not LSmash or GenESeSS"},
        n_rows=len(rows),anomaly_intervals=events,
        caveats=[
            "Sliding test windows overlap heavily and are not statistically independent",
            "The dataset annotations are public and only three distinct P-1 anomalies exist",
            "This is retrospective ranking, not train-calibrated prospective detection",
            "Mean distance to all TEST windows is transductive and differs from reference-to-train score",
            "No learned GenESeSS or LSmash reimplementation was used for baselines",
        ])
    (out/"README_RESULTS.json").write_text(json.dumps(summary,indent=2)+"\n")
if __name__=="__main__":
    main()
