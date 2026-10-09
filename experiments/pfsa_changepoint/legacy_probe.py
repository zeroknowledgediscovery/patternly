#!/usr/bin/env python3
"""Historical Patternly API smoke test; no silent substitution of algorithms."""
import json, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark import generate

DIR=Path("experiments/pfsa_changepoint/results")
DIR.mkdir(parents=True,exist_ok=True)
out={"method":"original_patternly_StreamingDetection",
     "implementation":"patternly.detection.StreamingDetection",
     "N":60000,"seed":12554,"window":4000,"n_clusters":2}
t=time.monotonic()
try:
    from patternly.detection import StreamingDetection
    seq,cp=generate(60000,12554)
    classifier=StreamingDetection(window_size=4000,window_overlap=0,
          n_clusters=2,reduce_clusters=False,quantize=False,
          eps=.1,verbose=False)
    classifier.fit(pd.Series(seq.astype(int)))
    classifier.predict()
    labels=np.asarray(classifier.closest_match,dtype=int)
    n=len(labels)
    assert n>5,("Too few windows",n)
    first=np.bincount(labels[:max(1,n//5)],minlength=2)
    second=np.bincount(labels[-max(1,n//5):],minlength=2)
    A=int(np.argmax(first));B=int(np.argmax(second))
    if A==B:
        raise RuntimeError("Original Patternly collapsed to same cluster on both sides")
    prob_a=(np.bincount(labels[:max(2,n//5)],minlength=2)+.5)/(max(2,n//5)+1)
    prob_b=(np.bincount(labels[-max(2,n//5):],minlength=2)+.5)/(max(2,n//5)+1)
    partial=np.cumsum(np.log(prob_a[labels]/prob_b[labels]))
    indices=np.arange(n)
    partial[(indices<int(.2*n))|(indices>int(.8*n))]=-np.inf
    estimated=int((np.argmax(partial)+1)*4000)
    out.update(status="success",estimated_cp=estimated,true_cp=cp,
               abs_error=abs(estimated-cp),n_windows=n,
               cluster_labels=labels.tolist())
except Exception as exc:
    out.update(status="failed",exception=repr(exc),
               traceback=traceback.format_exc()[-2500:])
out["elapsed_seconds"]=round(time.monotonic()-t,3)
(DIR/"legacy_patternly.json").write_text(json.dumps(out,indent=2)+"\n")
print("LEGACY",json.dumps(out),flush=True)
