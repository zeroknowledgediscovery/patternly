#!/usr/bin/env python3
"""True historical Patternly streaming tests across window sizes and seeds."""
import argparse,csv,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from patternly.detection import StreamingDetection
from benchmark import generate

def cp_from_scores(score,L,n):
    csum=np.cumsum(score)
    indices=np.arange(len(csum))
    candidate=(indices+1)*L
    csum[(candidate<int(.2*n))|(candidate>int(.8*n))]=-np.inf
    j=int(np.argmax(csum))
    return int((j+1)*L)

def test(n,seed,w,do_layer2=False):
    np.random.seed(1009+seed)
    bits,cp=generate(n,seed)
    start=time.monotonic()
    answer={"N":n,"seed":seed,"window":w,"true_cp":cp,"n_clusters":2}
    try:
        model=StreamingDetection(window_size=w,window_overlap=0,
            n_clusters=2,reduce_clusters=False,quantize=False,
            eps=.1,verbose=False)
        model.fit(pd.Series(bits.astype(int)))
        model.predict()
        labels=np.asarray(model.closest_match,dtype=int)
        score=np.asarray(model.cluster_llks,dtype=float)
        nw=len(labels)
        if nw<10 or score.shape!=(2,nw):raise ValueError(f"invalid windows/scoring {nw} {score.shape}")
        K=max(2,nw//5)
        # Use the guaranteed pure first/last 20% to identify the A- and B-like models.
        a=int(np.argmin(np.mean(score[:,:K],axis=1)))
        b=int(np.argmin(np.mean(score[:,-K:],axis=1)))
        # If both tails choose the same model, score contrasts cannot be oriented.
        same=(a==b)
        if same:b=1-a
        # llk_B - llk_A estimates the log-likelihood ratio in favour of A.
        cp_lik=cp_from_scores(score[b]-score[a],w,n)
        pref=(np.bincount(labels[:K],minlength=2)+.5)/(K+1)
        suff=(np.bincount(labels[-K:],minlength=2)+.5)/(K+1)
        cp_labels=cp_from_scores(np.log(pref[labels]/suff[labels]),w,n)
        answer.update(status="success",n_windows=nw,same_tail_model=same,
            A_index=a,B_index=b,
            cp_likelihood=cp_lik,err_likelihood=abs(cp_lik-cp),
            cp_labels=cp_labels,err_labels=abs(cp_labels-cp))
        if do_layer2:
            try:
                np.random.seed(1009+seed)
                level2=StreamingDetection(window_size=5,window_overlap=0,
                   n_clusters=2,reduce_clusters=False,quantize=False,eps=.1)
                level2.fit(pd.Series(labels))
                level2.predict()
                lbl2=np.asarray(level2.closest_match,dtype=int)
                if len(lbl2)<4:raise ValueError("too few second-layer blocks")
                Q=max(1,len(lbl2)//5)
                first=(np.bincount(lbl2[:Q],minlength=2)+.5)/(Q+1)
                last=(np.bincount(lbl2[-Q:],minlength=2)+.5)/(Q+1)
                cp_layer=cp_from_scores(np.log(first[lbl2]/last[lbl2]),w*5,n)
                answer.update(layer2_status="success",cp_layer2=cp_layer,
                              err_layer2=abs(cp_layer-cp))
            except Exception as e:
                answer.update(layer2_status="failure",layer2_error=repr(e))
    except Exception as e:
        answer.update(status="failed",error=repr(e))
    answer["elapsed_seconds"]=round(time.monotonic()-start,2)
    return answer

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,default=5)
    p.add_argument("--lengths",default="60000,180000")
    p.add_argument("--windows",default="500,1000,2000,4000,8000")
    p.add_argument("--output",default="experiments/pfsa_changepoint/results")
    args=p.parse_args()
    out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    rows=[]
    for n in map(int,args.lengths.split(",")):
        for i in range(args.seeds):
            seed=12554+3*i
            for w in map(int,args.windows.split(",")):
                r=test(n,seed,w,do_layer2=(w<=2000))
                rows.append(r);print("LEGACY_CASE",json.dumps(r),flush=True)
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with (out/"legacy_sweep.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    sums=[]
    for n in map(int,args.lengths.split(",")):
        for w in map(int,args.windows.split(",")):
            good=[r for r in rows if r["N"]==n and r["window"]==w and r["status"]=="success"]
            rec={"N":n,"window":w,"n_success":len(good),"n_total":args.seeds}
            for key in ("err_likelihood","err_labels","err_layer2"):
                vals=[r[key] for r in good if key in r]
                rec[key+"_median"]=float(np.median(vals)) if vals else None
            sums.append(rec);print("LEGACY_SUMMARY",json.dumps(rec),flush=True)
    (out/"legacy_sweep_summary.json").write_text(json.dumps(sums,indent=2)+"\n")
if __name__=="__main__":main()
