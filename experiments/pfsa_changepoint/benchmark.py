#!/usr/bin/env python3
"""PFSA change point benchmark, standalone reproducible MLE/adaptive tests."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.special import expit, xlogy
from numba import njit

@njit
def sample(pa,pb,n,cp,u,initial,d):
    seq=np.empty(n,dtype=np.uint8)
    for k in range(d):seq[k]=(initial>>(d-k-1))&1
    state=initial
    mask=(1<<d)-1
    for t in range(d,n):
        b=int(u[t]<(pa[state] if t<cp else pb[state]))
        seq[t]=b
        state=((state<<1)&mask)|b
    return seq

@njit
def contexts(x,d):
    ctx=np.empty(len(x)-d,dtype=np.int32)
    st=0
    for k in range(d):st=(st<<1)|int(x[k])
    mask=(1<<d)-1
    for t in range(d,len(x)):
        ctx[t-d]=st
        st=((st<<1)&mask)|int(x[t])
    return ctx

def source_pair(seed,d=10,delta=.17):
    rng=np.random.default_rng(seed)
    m=1<<(d-1)
    signs=rng.choice(np.array([-1.,1.]),size=m)
    bg=signs*1.35
    q=np.r_[np.ones(m//2),-np.ones(m-m//2)]
    rng.shuffle(q)
    pert=signs*q
    a=np.r_[bg+delta*pert,-bg-delta*pert]
    b=np.r_[bg-delta*pert,-bg+delta*pert]
    return expit(a),expit(b)

def generate(n,seed,delta=.17,d=10):
    pa,pb=source_pair(seed,d,delta)
    cp=int(n*.53)
    u=np.random.default_rng(seed+100000).random(n)
    return sample(pa,pb,n,cp,u,seed%(1<<d),d),cp

def models(x,d,alpha=.5):
    ctx=contexts(x,d)
    n=np.bincount(ctx,minlength=1<<d)
    o=np.bincount(ctx,weights=x[d:],minlength=1<<d)
    return (o+alpha)/(n+2*alpha)

def cp_scan(x,d,step=100):
    h=contexts(x,d)
    k=np.arange(len(h))//step
    nbin=int(k[-1])+1
    z=k*(1<<d)+h
    total=np.bincount(z,minlength=nbin*(1<<d)).reshape(nbin,-1)
    ones=np.bincount(z,weights=x[d:],minlength=nbin*(1<<d)).reshape(nbin,-1)
    nt=total.sum(axis=0); no=ones.sum(axis=0)
    def loglik(o,t):
        return xlogy(o,o/np.maximum(t,1))+xlogy(t-o,(t-o)/np.maximum(t,1))
    ct=np.cumsum(total,axis=0); co=np.cumsum(ones,axis=0)
    ll=loglik(co,ct).sum(axis=1)+loglik(no-co,nt-ct).sum(axis=1)
    candidates=d+np.minimum((np.arange(nbin)+1)*step,len(x)-d)
    ll[(candidates<.2*len(x))|(candidates>.8*len(x))]=-np.inf
    return int(candidates[np.argmax(ll)])

def cp_adaptive(x,d,window=1):
    n=len(x);q=int(.2*n)
    a=models(x[:q],d);b=models(x[n-q:],d)
    h=contexts(x,d); y=x[d:]
    diff=y*np.log(a[h]/b[h])+(1-y)*np.log((1-a[h])/(1-b[h]))
    cs=np.cumsum(diff)
    ix=np.arange(len(cs))
    if window>1:ix=ix[window-1::window]
    ix=ix[(ix>=.2*n-d)&(ix<=.8*n-d)]
    return int(ix[np.argmax(cs[ix])]+d+1)

def bic_depth(x,max_d=11):
    scores=[]
    for d in range(max_d+1):
        if d==0:
            o=int(x.sum());p=(o+.5)/(len(x)+1)
            ll=o*np.log(p)+(len(x)-o)*np.log(1-p)
        else:
            h=contexts(x,d);n=np.bincount(h,minlength=1<<d)
            o=np.bincount(h,weights=x[d:],minlength=1<<d)
            p=(o+.5)/(n+1)
            ll=np.sum(xlogy(o,p)+xlogy(n-o,1-p))
        scores.append((-2*ll+(1<<d)*np.log(len(x)-d),d))
    return int(min(scores)[1])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",default="60000,180000")
    ap.add_argument("--seeds",type=int,default=5)
    ap.add_argument("--out",default="experiments/pfsa_changepoint/results")
    args=ap.parse_args()
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n in map(int,args.n.split(",")):
        for i in range(args.seeds):
            seed=12554+3*i
            x,cp=generate(n,seed)
            q=int(.2*n)
            order=bic_depth(np.r_[x[:q],x[n-q:]])
            methods={
                "MLE_depth10":cp_scan(x,10),
                "adaptive_depth10":cp_adaptive(x,10),
                "adaptive_BIC":cp_adaptive(x,order),
            }
            for w in (250,500,1000,2000,4000,8000):
                methods[f"adaptive_depth10_window{w}"]=cp_adaptive(x,10,w)
            for name,found in methods.items():
                rows.append(dict(N=n,seed=seed,method=name,true_cp=cp,estimated_cp=found,
                                 abs_error=abs(found-cp),selected_BIC=order))
            print("CASE",json.dumps({"N":n,"seed":seed,"BIC":order,
                       "errors":{k:abs(v-cp) for k,v in methods.items() if k in ("MLE_depth10","adaptive_depth10","adaptive_BIC")}}),flush=True)
    with (root/"results.csv").open("w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    out=[]
    for n in map(int,args.n.split(",")):
        for name in sorted(set(x["method"] for x in rows)):
            vals=[r["abs_error"] for r in rows if r["N"]==n and r["method"]==name]
            item=dict(N=n,method=name,median_abs_error=float(np.median(vals)),n=len(vals))
            print("SUMMARY",json.dumps(item),flush=True);out.append(item)
    (root/"summary.json").write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__":main()
