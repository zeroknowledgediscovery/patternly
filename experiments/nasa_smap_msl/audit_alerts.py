#!/usr/bin/env python3
"""Audit saved NASA SMAP/MSL detector alerts with onset-aware event metrics.

No model fitting and no threshold tuning. Uses existing per-channel scores.
    python experiments/nasa_smap_msl/audit_alerts.py --results results/nasa_smap_msl_pilot
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = ROOT / "datasets/nasa_smap_msl/labeled_anomalies.csv"


def intervals_for(channel, annotations, n):
    """Parse inclusive Telemanom intervals and merge any overlapping labels."""
    import ast
    part = annotations[annotations.chan_id == channel]
    if part.empty:
        raise ValueError("No published labels for channel "+channel)
    values = []
    for seq in part.anomaly_sequences:
        values.extend((int(a), int(b)+1) for a,b in ast.literal_eval(seq))
    for a,b in values:
        if not (0 <= a < b <= n):
            raise ValueError(f"{channel} label {a}:{b} outside {n} observations")
    merged = []
    for a,b in sorted(values):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a,b))
    return merged


def audit_score(values, threshold, intervals):
    values=np.asarray(values, dtype=float)
    valid=np.isfinite(values)
    alerts=(values > threshold) & valid
    onset=alerts & ~np.r_[False,alerts[:-1]]
    labeled=np.zeros(len(values),dtype=bool)
    event_hits=0
    onset_hits=0
    prior_alarm_at_onset=0
    delays=[]
    for start,end in intervals:
        labeled[start:end]=True
        in_event=alerts[start:end]
        event_hits+=bool(in_event.any())
        new_onsets=np.flatnonzero(onset[start:end])
        onset_hits+=bool(len(new_onsets))
        if len(new_onsets): delays.append(int(new_onsets[0]))
        prior_alarm_at_onset+=bool(alerts[start] and not onset[start])
    negative=valid & ~labeled
    n_negative=int(negative.sum())
    fp_points=int(np.sum(negative & alerts))
    fp_episodes=int(np.sum(negative & onset))
    return {
        "annotated_events":len(intervals),
        "event_hits":event_hits,
        "onset_hits":onset_hits,
        "prior_alarm_at_event_start":prior_alarm_at_onset,
        "negative_points":n_negative,
        "false_alarm_points":fp_points,
        "false_alarm_episodes":fp_episodes,
        "valid_points":int(valid.sum()),
        "active_alarm_points":int(alerts.sum()),
        "event_recall":event_hits/len(intervals),
        "onset_event_recall":onset_hits/len(intervals),
        "non_event_alarm_fraction":fp_points/max(1,n_negative),
        "overall_alarm_fraction":float(alerts.sum()/max(1,valid.sum())),
        "false_alarm_onsets_per_1000":1000*fp_episodes/max(1,n_negative),
        "median_new_alarm_delay":float(np.median(delays)) if delays else None,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",default="results/nasa_smap_msl_pilot")
    args=p.parse_args()
    results=Path(args.results)
    csv_path=results/"per_channel.csv"
    if not csv_path.exists():
        p.error(f"Missing {csv_path}: run pilot.py first")
    info=pd.read_csv(csv_path)
    annotations=pd.read_csv(ANNOTATIONS)
    rows=[]
    for channel in sorted(info.channel.unique()):
        saved=results/"scores"/f"{channel}.npz"
        if not saved.exists():
            raise FileNotFoundError(saved)
        with np.load(saved) as payload:
            score_methods={s[len("score_"):] for s in payload.files if s.startswith("score_")}
            listed=set(info.loc[info.channel == channel, "method"])
            if score_methods != listed:
                raise ValueError(f"{channel} result mismatch {score_methods} vs {listed}")
            for method in sorted(score_methods):
                scores=payload[f"score_{method}"]
                threshold=float(payload[f"threshold_{method}"][0])
                ivals=intervals_for(channel,annotations,len(scores))
                vals=audit_score(scores,threshold,ivals)
                rows.append(dict(channel=channel,method=method,threshold=threshold,**vals))
    df=pd.DataFrame(rows)
    df.to_csv(results/"alert_audit_per_channel.csv",index=False)
    summary=[]
    for method, group in df.groupby("method"):
        events=int(group.annotated_events.sum())
        negatives=int(group.negative_points.sum())
        valid=int(group.valid_points.sum())
        entry={
            "method":method,
            "channels":len(group),
            "events":events,
            "event_hit_recall":float(group.event_hits.sum()/events),
            "new_alarm_onset_recall":float(group.onset_hits.sum()/events),
            "prior_alarm_at_onset_events":int(group.prior_alarm_at_event_start.sum()),
            "false_alarm_fraction_non_event":float(group.false_alarm_points.sum()/max(1,negatives)),
            "fraction_test_time_in_alarm":float(group.active_alarm_points.sum()/max(1,valid)),
            "median_channel_alarm_fraction":float(group.overall_alarm_fraction.median()),
            "false_alarm_onsets_per_1000":float(1000*group.false_alarm_episodes.sum()/max(1,negatives)),
            "n_false_onset_episodes":int(group.false_alarm_episodes.sum()),
            "n_false_alarm_points":int(group.false_alarm_points.sum()),
        }
        summary.append(entry)
    summary_df=pd.DataFrame(summary)
    summary_df.to_csv(results/"alert_audit_summary.csv",index=False)
    print("AUDIT_SUMMARY", summary_df.to_json(orient="records"),flush=True)
    print("AUDIT_OUT",str(results.resolve()),flush=True)


if __name__=="__main__":
    main()
