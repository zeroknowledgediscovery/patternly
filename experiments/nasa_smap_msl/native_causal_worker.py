#!/usr/bin/env python3
"""Isolated ORIGINAL native PFSA distance worker.

C extensions in zedsuite/LSmash occasionally raise fatal malloc errors,
so run each native method in a *separate process*; one failure can no
longer erase the other native or custom-JS benchmark results.

No surrogate algorithms. All distances here use compiled native LSmash.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from p1_sliding_lsmash import learned_projection_models

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",required=True,choices=["default","genesess"])
    p.add_argument("--infile",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--epsilons",default="0.01,0.05,0.1,0.2")
    args=p.parse_args()
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with np.load(args.infile) as z:
        ref=z["ref"].astype(int).tolist()
        cal=z["cal"].astype(int).tolist()
        test=z["test"].astype(int).tolist()
        initial=z["initial"].astype(np.uint32)
        alphabet=int(z["alphabet"])
    import lsmash
    opt=lsmash.LsmashOptions()
    opt.data_type="symbolic";opt.sae=False
    models=[]
    if args.mode=="genesess":
        if not hasattr(lsmash,"from_sequences_with_pfsas"):
            raise RuntimeError("Missing compiled native learned PFSA projector API")
        eps=tuple(float(x) for x in args.epsilons.split(","))
        if len(eps)!=4:raise ValueError("Expected 4 epsilon levels")
        model_dir=out.parent/"native_models"
        model_dir.mkdir(exist_ok=True)
        models=learned_projection_models(initial,model_dir,len(initial),eps,alphabet)
        files=[m["zbase_file"] for m in models]
    t0=time.perf_counter()
    seq=ref+cal+test
    if args.mode=="genesess":
        matrix=np.asarray(lsmash.from_sequences_with_pfsas(seq,files,opt),dtype=float)
    else:
        matrix=np.asarray(lsmash.from_sequences(seq,opt),dtype=float)
    n,v,q=len(ref),len(cal),len(test)
    if matrix.shape!=(n+v+q,n+v+q) or not np.isfinite(matrix).all():
        raise RuntimeError(f"Nonfinite/wrong shape native result {matrix.shape}; "
                           f"nonfinite count {int((~np.isfinite(matrix)).sum())}")
    if np.max(np.abs(matrix-matrix.T))>1e-7 or np.max(np.abs(np.diag(matrix)))>1e-7:
        raise RuntimeError("Native matrix violates zero diagonal/symmetry")
    val=matrix[:n,n:n+v].mean(axis=0)
    query=matrix[:n,n+v:].mean(axis=0)
    np.savez_compressed(out,cal=val,test=query)
    (out.with_suffix(".json")).write_text(json.dumps(dict(
         mode=args.mode,models=models,n_ref=n,n_cal=v,n_test=q,
         elapsed_native_seconds=time.perf_counter()-t0,
         implementation="actual C++ LSmash and optionally actual native GenESeSS"),indent=2)+"\n")
    print("VERIFIED_NATIVE_WORKER",args.mode,"models",len(models),flush=True)

if __name__=="__main__":
    main()
