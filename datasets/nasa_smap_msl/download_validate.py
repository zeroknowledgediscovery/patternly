#!/usr/bin/env python3
"""Download/validate NASA SMAP-MSL real telemetry and prepare exact univariate streams.

Source: https://huggingface.co/datasets/appleparan/telemanom
Outputs in dataset root: values/{train,test}/{channel}.npz,
labeled_anomalies.csv, manifest.json, SHA256SUMS.tsv, LICENSE.txt.

Retain original 24/54 auxiliary command variables in the downloaded Parquet
source; those files are uploaded by CI separately, not committed to git.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

REPO = "appleparan/telemanom"

def sha256(path):
    dig = hashlib.sha256()
    with Path(path).open("rb") as inp:
        for buf in iter(lambda: inp.read(1024 * 1024), b""):
            dig.update(buf)
    return dig.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",default="/tmp/patternly_nasa_smap_msl")
    p.add_argument("--output",default="datasets/nasa_smap_msl")
    args=p.parse_args()
    src=Path(args.source).resolve()
    dst=Path(args.output).resolve()
    dst.mkdir(parents=True,exist_ok=True)
    # Limit to unique dataset files, avoid the duplicate original .npy trees.
    print("Downloading NASA SMAP/MSL Parquet train/test and annotation metadata",
          flush=True)
    snapshot_download(
        repo_id=REPO, repo_type="dataset", revision="main",
        local_dir=str(src),
        allow_patterns=["data/train/*.parquet","data/test/*.parquet",
                        "labeled_anomalies.csv","LICENSE.txt"],
        max_workers=8
    )
    lpath=src/"labeled_anomalies.csv"
    assert lpath.is_file(), "Missing labeled_anomalies.csv"
    shutil.copy2(lpath,dst/"labeled_anomalies.csv")
    if (src/"LICENSE.txt").exists():shutil.copy2(src/"LICENSE.txt",dst/"LICENSE.txt")
    labels=pd.read_csv(lpath)
    assert set(["chan_id","spacecraft","anomaly_sequences","num_values"]).issubset(labels.columns)
    expected=set(labels.chan_id)
    assert len(expected)>=80, (len(expected),sorted(expected))

    manifest={"source":f"https://huggingface.co/datasets/{REPO}",
       "format":"per-channel compressed NumPy value series with full original Parquet retained in CI artifact",
       "metadata_notes":"Anomaly intervals are test-only and interpreted as original index coordinates.",
       "source_channels":len(expected),
       "annotation_rows":len(labels),
       "channels":[]}
    sha_rows=[]
    for split in ("train","test"):
        files={x.stem:x for x in sorted((src/"data"/split).glob("*.parquet"))}
        print("Downloaded",split,"parquets",len(files),flush=True)
        assert expected.issubset(set(files)),dict(missing=sorted(expected-set(files)))
        if split=="train":
            manifest["source_channels"]=len(files)
            manifest["annotated_channels"]=len(expected)
            manifest["unlabeled_channels"]=sorted(set(files)-expected)
            print("Channels without published anomaly labels:",manifest["unlabeled_channels"],flush=True)
        for ch,path in files.items():
            df=pd.read_parquet(path)
            assert "value" in df.columns and "timestep" in df.columns,(ch,df.columns)
            df=df.sort_values("timestep")
            ts=df["timestep"].to_numpy()
            vals=df["value"].to_numpy(dtype=np.float64)
            assert len(vals)>0 and np.isfinite(vals).all(),(ch,split)
            assert np.array_equal(ts,np.arange(len(ts))), (ch,split,"timestep discontinuity",ts[:10])
            outfile=dst/"values"/split/f"{ch}.npz"
            outfile.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(outfile,value=vals)
            with np.load(outfile) as out:
                assert np.array_equal(vals,out["value"])
            if split=="test" and ch in expected:
                desired=set(labels.loc[labels.chan_id==ch,"num_values"].astype(int))
                assert desired=={len(vals)},(ch,"test count mismatch",desired,len(vals))
            manifest["channels"].append({
                "channel":ch,"split":split,"n_timesteps":len(vals),
                "n_aux_command_columns":len(df.columns)-2,
                "data_path":str(outfile.relative_to(dst)),
                "data_sha256":sha256(outfile),
                "original_parquet_sha256":sha256(path),
                "original_parquet_bytes":path.stat().st_size,
                "value_min":float(vals.min()),"value_max":float(vals.max())
            })
            sha_rows.append((sha256(path),f"data/{split}/{ch}.parquet"))
    manifest["n_channel_files"]=len(manifest["channels"])
    manifest["n_test_samples"]=sum(x["n_timesteps"] for x in manifest["channels"] if x["split"]=="test")
    manifest["n_train_samples"]=sum(x["n_timesteps"] for x in manifest["channels"] if x["split"]=="train")
    assert manifest["n_channel_files"]==2*manifest["source_channels"]
    (dst/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (dst/"SHA256SUMS.tsv").write_text("sha256\toriginal_path\n"+"\n".join(f"{s}\t{p}" for s,p in sha_rows)+"\n")
    print("DATASET_VALIDATION",json.dumps({
      "channels":manifest["source_channels"],
      "train_files":len(expected),"test_files":len(expected),
      "train_samples":manifest["n_train_samples"],
      "test_samples":manifest["n_test_samples"],
      "annotation_rows":manifest["annotation_rows"],
      "compressed_bytes":sum(f.stat().st_size for f in (dst/"values").rglob("*.npz")),
      "raw_parquet_bytes":sum(p.stat().st_size for p in src.glob("data/*/*.parquet"))
      }),flush=True)

if __name__=="__main__":main()
