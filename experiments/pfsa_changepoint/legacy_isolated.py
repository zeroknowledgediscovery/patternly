#!/usr/bin/env python3
"""Crash-isolated historical Patternly evaluation. Each native trial is separate."""
import argparse, csv, json, subprocess, sys, time
from pathlib import Path
import numpy as np

def child(n,seed,window):
    from legacy_sweep import test
    result=test(n,seed,window,do_layer2=(window<=2000))
    print("TRIAL_JSON="+json.dumps(result,sort_keys=True),flush=True)

def parent(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for n in map(int,args.lengths.split(",")):
        for i in range(args.seeds):
            seed=12554+3*i
            for w in map(int,args.windows.split(",")):
                cmd=[sys.executable,__file__,"child",str(n),str(seed),str(w)]
                start=time.monotonic()
                try:
                    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                     encoding="utf-8",timeout=45)
                    marker=[line.split("TRIAL_JSON=",1)[-1] for line in p.stdout.splitlines()
                            if line.startswith("TRIAL_JSON=")]
                    if p.returncode==0 and marker:
                        row=json.loads(marker[-1])
                    else:
                        row={"N":n,"seed":seed,"window":w,"true_cp":int(.53*n),
                             "status":"native_failure","returncode":p.returncode,
                             "stderr_tail":p.stderr[-350:],"stdout_tail":p.stdout[-350:]}
                except subprocess.TimeoutExpired:
                    row={"N":n,"seed":seed,"window":w,"true_cp":int(.53*n),
                         "status":"timeout"}
                row["wall_seconds"]=round(time.monotonic()-start,2)
                rows.append(row)
                print("TRIAL",json.dumps(row),flush=True)
    keys=sorted(set().union(*(x.keys() for x in rows)))
    with (root/"legacy_isolated.csv").open("w",newline="") as fp:
        wr=csv.DictWriter(fp,fieldnames=keys);wr.writeheader();wr.writerows(rows)
    summary=[]
    for n in map(int,args.lengths.split(",")):
        for w in map(int,args.windows.split(",")):
            relevant=[r for r in rows if r["N"]==n and r["window"]==w]
            ok=[r for r in relevant if r["status"]=="success"]
            rec={"N":n,"window":w,"n_success":len(ok),"n_total":len(relevant),
                 "n_crashed":sum(r["status"]=="native_failure" for r in relevant),
                 "n_failed":sum(r["status"]=="failed" for r in relevant),
                 "n_timeouts":sum(r["status"]=="timeout" for r in relevant)}
            for key in ("err_likelihood","err_labels","err_layer2"):
                values=[r[key] for r in ok if key in r]
                rec[key+"_median"]=float(np.median(values)) if values else None
            summary.append(rec)
            print("SUMMARY",json.dumps(rec),flush=True)
    (root/"legacy_isolated_summary.json").write_text(json.dumps(summary,indent=2)+"\n")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=["child","parent"])
    ap.add_argument("params",nargs="*")
    ap.add_argument("--seeds",type=int,default=5)
    ap.add_argument("--lengths",default="60000,180000")
    ap.add_argument("--windows",default="500,1000,2000,4000,8000")
    ap.add_argument("--output",default="experiments/pfsa_changepoint/results")
    a=ap.parse_args()
    if a.mode=="child":
        if len(a.params)!=3:raise SystemExit("child needs N seed window")
        child(*map(int,a.params))
    else:
        parent(a)
