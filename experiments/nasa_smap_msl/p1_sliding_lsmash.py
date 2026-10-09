#!/usr/bin/env python3
"""P-1 sliding-window N×N *native* LSmash matrices with anomaly annotation.

Modes:
  default: original compiled LSmash, 4 native random PFSA projectors.
  genesess: original compiled LSmash llk_distance with actual native
    zedsuite.GenESeSS-trained projection PFSAs. Requires the *opt-in*
    native pybind extension from lsmash experiment/genesess-projectors.

We DO NOT implement a distance metric in Python. Reformatting the
native GenESeSS PFSA data into native zbase's "$" and "@" text records
is representation conversion only; all probabilities/edges are copied.
Every failure is fatal, not replaced with a surrogate.

Default: P-1 test, 212-observation windows shifted by 5 observations,
training-derived quartile bins, ~1659 overlapping windows.
"""
from __future__ import annotations
import argparse
import ast
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from pilot import DATA, load_series, event_intervals

def train_quantization(train, test, alphabet):
    q=np.arange(1,alphabet)/alphabet
    edges=np.unique(np.quantile(train,q))
    if len(edges)<1:
        raise RuntimeError("Training quantizer has just one category")
    return np.digitize(train,edges).astype(np.uint32), np.digitize(test,edges).astype(np.uint32),edges

def sliding(series,win,stride):
    if win<32 or stride<1 or len(series)<win:
        raise ValueError("Invalid window length or stride")
    # A view of original samples, no re-sampling or overlapping labels.
    starts=np.arange(0,len(series)-win+1,stride,dtype=np.int64)
    seq=[series[i:i+win].astype(int).tolist() for i in starts]
    return starts,seq

def extract_genesess_native_model(path):
    lines=Path(path).read_text().splitlines()
    def section(title,marker,next_header):
        hits=[i for i,s in enumerate(lines) if s.startswith(title)]
        if len(hits)!=1:
            raise ValueError(f"Missing/ambiguous {title} in native GenESeSS PFSA")
        count=int(lines[hits[0]].split("size(")[1].split(")")[0])
        start=hits[0]+1
        if lines[start].strip()!=marker:
            raise ValueError(f"Unexpected GenESeSS {title} format")
        body=lines[start+1:start+1+count]
        if len(body)!=count:
            raise ValueError(f"Incomplete {title} rows")
        return body
    morph_lines=section("%PITILDE:","#PITILDE","%CONNX:")
    conn_lines=section("%CONNX:","#CONNX",None)
    probs=[[float(c) for c in s.split()] for s in morph_lines]
    edges=[[int(c) for c in s.split()] for s in conn_lines]
    n=len(probs)
    if len(edges)!=n or n<1:
        raise ValueError("Inconsistent GenESeSS machine state count")
    k=len(probs[0])
    for i,(p,row) in enumerate(zip(probs,edges)):
        if len(p)!=k or len(row)!=k or not all(np.isfinite(p)) or any(x<0 for x in p):
            raise ValueError(f"Invalid row {i} of GenESeSS PFSA")
        if abs(sum(p)-1)>2e-4:
            raise ValueError(f"Invalid row {i} probability sum")
        if any((j<0 or j>=n) for j in row):
            raise ValueError(f"Invalid destination state {i}")
    return probs,edges

def write_zbase_native_file(genesess_file,zbase_file,expected_alphabet):
    """Convert ONLY serialization format, never approximate learned PFSA."""
    probabilities,connections=extract_genesess_native_model(genesess_file)
    if len(probabilities[0])!=expected_alphabet:
        raise ValueError(f"Native PFSA alphabet {len(probabilities[0])} != expected {expected_alphabet}")
    # For the original C++ PFSA(string) constructor, '$' is morph and
    # '@' is symbol-conditioned next-state; 's 0' suppresses random
    # initial state and 'len 0' avoids generating any synthetic data.
    rows=[]
    for p in probabilities:
        rows.append("$ "+" ".join(format(float(x),".17g") for x in p))
    for c in connections:
        rows.append("@ "+" ".join(str(x) for x in c))
    rows.extend(["s 0","len 0"])
    Path(zbase_file).write_text("\n".join(rows)+"\n")
    return dict(n_states=len(probabilities),alphabet=len(probabilities[0]),
                input_file=str(genesess_file),zbase_file=str(zbase_file))

def learned_projection_models(train_symbols,output,training_length,epsilons,expected_alphabet):
    from zedsuite.genesess import GenESeSS
    import pandas as pd
    if len(train_symbols)<training_length:
        raise ValueError("Not enough normal training observations for initial GenESeSS segment")
    initial=train_symbols[:training_length].astype(int)
    models=[]
    for eps in epsilons:
        name=("eps_"+format(eps,".6g").replace(".","p"))
        raw=output/(name+".genesess.pfsa")
        zbase=output/(name+".zbase.pfsa")
        native=GenESeSS(data=pd.DataFrame([initial.tolist()]),outfile=str(raw),
                        data_type="symbolic",data_dir="row",force=True,eps=eps)
        if not native.run() or not raw.is_file() or raw.stat().st_size==0:
            raise RuntimeError(f"Native GenESeSS inference failed at epsilon {eps}")
        parsed=write_zbase_native_file(raw,zbase,expected_alphabet)
        observed=int(np.asarray(native.probability_morph_matrix).shape[0])
        if parsed["n_states"]!=observed:
            raise RuntimeError("Native GenESeSS state count does not match saved file")
        models.append({**parsed,"requested_eps":eps,
            "actual_eps":float(native.epsilon_used),
            "n_symbols_initial":len(initial),
            "native_algorithm":"zedsuite.genesess.GenESeSS"})
        print("NATIVE_GENESESS_TRAINED",json.dumps(models[-1]),flush=True)
    return models

def anomaly_mask(starts,win,intervals):
    ends=starts+win
    return np.array([any(a<e and b>s for a,b in intervals)
                     for s,e in zip(starts,ends)],dtype=bool)

def validate_and_save_matrix(D,output,starts,win,meta):
    n=len(starts)
    if D.shape!=(n,n):
        raise RuntimeError(f"Native LSmash shape {D.shape}, expected {(n,n)}")
    if not np.isfinite(D).all():
        raise RuntimeError(f"Native LSmash returned {int((~np.isfinite(D)).sum())} nonfinite elements")
    if np.max(np.abs(D-D.T))>1e-8:
        raise RuntimeError("Native LSmash matrix is not symmetric")
    if np.max(np.abs(np.diag(D)))>1e-8:
        raise RuntimeError("Native LSmash nonzero diagonal when sae=False")
    np.save(output/"distance_matrix.npy",D)
    # For ~1659 windows, format compactly to avoid huge CSV size.
    np.savetxt(output/"distance_matrix.csv",D,fmt="%.10g",delimiter=",")
    intervals=meta["anomaly_intervals"]
    marked=anomaly_mask(starts,win,intervals)
    with (output/"windows.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=[
            "window","start","end_exclusive","overlaps_annotation"])
        writer.writeheader()
        for i,s in enumerate(starts):
            writer.writerow(dict(window=i,start=int(s),end_exclusive=int(s+win),
                                 overlaps_annotation=int(marked[i])))
    means=D.mean(axis=1)
    with (output/"window_distance_profile.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=[
            "window","start","end_exclusive","mean_lsmash_distance","overlaps_annotation"])
        writer.writeheader()
        for i,s in enumerate(starts):
            writer.writerow(dict(window=i,start=int(s),end_exclusive=int(s+win),
                mean_lsmash_distance=float(means[i]),overlaps_annotation=int(marked[i])))
    maskoff=~np.eye(n,dtype=bool)
    meta.update(dict(
        matrix_shape=[n,n],num_anomaly_overlap_windows=int(marked.sum()),
        offdiagonal_min=float(D[maskoff].min()),
        offdiagonal_median=float(np.median(D[maskoff])),
        offdiagonal_max=float(D[maskoff].max()),
        mean_distance_overlap=float(means[marked].mean()) if marked.any() else None,
        mean_distance_other=float(means[~marked].mean()) if (~marked).any() else None))
    (output/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    return marked,means

def annotated_plots(D,starts,win,marked,means,meta,out):
    # Matrix includes actual pairwise native values in *chronological*
    # order, with independently annotated window strips on the top/left.
    from matplotlib.gridspec import GridSpec
    n=len(starts)
    fig=plt.figure(figsize=(11.5,10.5))
    gs=GridSpec(2,2,figure=fig,width_ratios=[0.24,9.5],
        height_ratios=[0.24,9.5],wspace=.04,hspace=.04)
    ax=fig.add_subplot(gs[1,1])
    upper=fig.add_subplot(gs[0,1],sharex=ax)
    left=fig.add_subplot(gs[1,0],sharey=ax)
    vmin=float(np.min(D));vmax=float(np.max(D))
    hm=ax.imshow(D,cmap="viridis",origin="upper",interpolation="nearest",
                 vmin=vmin,vmax=vmax,aspect="auto")
    upper.imshow(marked.astype(float)[None,:],cmap="Reds",vmin=0,vmax=1,
                 origin="upper",interpolation="nearest",aspect="auto")
    left.imshow(marked.astype(float)[:,None],cmap="Reds",vmin=0,vmax=1,
                origin="upper",interpolation="nearest",aspect="auto")
    upper.axis("off");left.axis("off")
    step=max(1,round(n/12/50)*50)
    ticks=np.arange(0,n,step,dtype=int)
    ax.set_xticks(ticks);ax.set_yticks(ticks)
    ax.set_xlabel("Sliding window index j (chronological, step 5)")
    ax.set_ylabel("Sliding window index i")
    ax.set_title(f"P-1 {meta['mode']}: native LSmash distances\n"
         f"{n} overlapping windows × {win} observations; "
         "red strips = NASA anomaly overlap",fontsize=12,pad=14)
    bar=fig.colorbar(hm,ax=ax,fraction=.045,pad=.023)
    bar.set_label("Native LSmash distance")
    fig.savefig(out/"annotated_heatmap.png",dpi=175,bbox_inches="tight")
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(13,4.5))
    ax.plot(starts+win//2,means,lw=.8,color="#3266a8")
    for j,(a,b) in enumerate(meta["anomaly_intervals"]):
        ax.axvspan(a,b,color="#d94e44",alpha=.17,
                   label="NASA anomaly" if j==0 else None)
    if marked.any():
        ax.scatter(starts[marked]+win//2,means[marked],s=4,
                   color="#b8433a",alpha=.45,zorder=3)
    ax.set(title=f"P-1 {meta['mode']} — mean native distance by sliding window",
           xlabel="Test observation index (window center)",
           ylabel="Mean distance to all windows")
    ax.grid(alpha=.17)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"anomaly_distance_profile.png",dpi=180)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--channel",default="P-1")
    ap.add_argument("--mode",choices=["default","genesess"],required=True)
    ap.add_argument("--split",choices=["test"],default="test")
    ap.add_argument("--window",type=int,default=212)
    ap.add_argument("--stride",type=int,default=5)
    ap.add_argument("--alphabet",type=int,default=4)
    ap.add_argument("--initial-train-length",type=int,default=1024,
                    help="first N normal TRAIN points to train GenESeSS projectors")
    ap.add_argument("--epsilons",default="0.01,0.05,0.1,0.2")
    ap.add_argument("--max-windows",type=int,default=2400)
    ap.add_argument("--out",default="")
    args=ap.parse_args()
    if args.channel!="P-1":raise ValueError("This matched experiment is frozen for P-1")
    out=Path(args.out or f"results/nasa_p1_sliding_lsmash_{args.mode}")
    out.mkdir(parents=True,exist_ok=True)
    train,test=load_series(args.channel)
    train_s,test_s,edges=train_quantization(train,test,args.alphabet)
    starts,seq=sliding(test_s,args.window,args.stride)
    n=len(starts)
    if n>args.max_windows:
        raise ValueError(f"Requested {n} windows; use --max-windows explicitly if intended")
    print("INPUT",json.dumps(dict(mode=args.mode,windows=n,win=args.window,
          stride=args.stride,symbols=len(edges)+1,
          n_pairwise=n*n)),flush=True)

    import lsmash
    opts=lsmash.LsmashOptions()
    opts.data_type="symbolic"
    opts.sae=False
    model_info=[]
    model_kernel=None
    if args.mode=="default":
        t0=time.perf_counter()
        D=np.asarray(lsmash.from_sequences(seq,opts),dtype=float)
        kernel="unaltered stock native lsmash.from_sequences"
    else:
        if not hasattr(lsmash,"from_sequences_with_pfsas"):
            raise RuntimeError("Missing native learned-projector binding. "
              "Install lsmash experiment/genesess-projectors branch. "
              "Do NOT use a custom Python distance as fallback.")
        eps=tuple(float(s) for s in args.epsilons.split(","))
        if any(not(0<e<1) for e in eps) or len(set(eps))!=len(eps):
            raise ValueError("Invalid or duplicated epsilons")
        models_dir=out/"native_models"
        models_dir.mkdir(parents=True,exist_ok=True)
        model_info=learned_projection_models(train_s,models_dir,
                  args.initial_train_length,eps,len(edges)+1)
        files=[i["zbase_file"] for i in model_info]
        t0=time.perf_counter()
        D=np.asarray(lsmash.from_sequences_with_pfsas(seq,files,opts),
                     dtype=float)
        kernel="original C++ llk_distance(S,G) with G from native GenESeSS"
    elapsed=time.perf_counter()-t0
    import pandas as pd
    ann=pd.read_csv(DATA/"labeled_anomalies.csv")
    ivals=event_intervals(args.channel,len(test),ann)
    meta=dict(channel=args.channel,split=args.split,mode=args.mode,
        source="NASA SMAP/MSL P-1 test telemetry",
        python_lsmash_module=str(lsmash.__file__),native_lsmash_core=kernel,
        actual_genesess_projectors=(args.mode=="genesess"),
        no_surrogate_distances=True,
        n_train_samples=len(train),n_test_samples=len(test),
        window_samples=args.window,stride=args.stride,n_windows=n,
        n_pairwise_distances=int(n*n),
        training_quantile_edges=edges.tolist(),
        n_symbols=len(edges)+1,
        n_initial_train_samples=(args.initial_train_length if args.mode=="genesess" else None),
        projection_models=model_info,
        native_distance_seconds=elapsed,
        anomaly_intervals=ivals)
    marked,means=validate_and_save_matrix(D,out,starts,args.window,meta)
    annotated_plots(D,starts,args.window,marked,means,meta,out)
    print("LSMASH_SLIDING_RESULT",json.dumps(meta,allow_nan=False),flush=True)
    print("FILES",str(out.resolve()),flush=True)

if __name__=="__main__":
    main()
