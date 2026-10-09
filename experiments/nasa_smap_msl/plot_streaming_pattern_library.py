#!/usr/bin/env python3
"""Visualize saved native streaming Patternly results (no recomputation)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np


def read_windows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No window records in {path}")
    return rows


def floats(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) if r.get(key, "") not in ("", "None", "null") else np.nan
                     for r in rows], dtype=float)


def load_raw(path: Path | None):
    if path is None or not path.is_file():
        return None, [], []
    if path.suffix == ".npz":
        with np.load(path) as z:
            raw = np.asarray(z["value"], dtype=float).reshape(-1)
            boundaries = (np.asarray(z["segment_boundary"], dtype=int).reshape(-1).tolist()
                          if "segment_boundary" in z else [])
            events = (np.asarray(z["anomaly_intervals"], dtype=int).reshape((-1, 2)).tolist()
                      if "anomaly_intervals" in z else [])
        return raw, boundaries, events
    if path.suffix == ".npy":
        return np.load(path).reshape(-1), [], []
    return None, [], []


def decorate(ax, boundaries, events, legend=False):
    for idx, (left, right) in enumerate(events):
        ax.axvspan(left, right, color="tab:red", alpha=0.12,
                   label="Annotated event" if legend and idx == 0 else None)
    for idx, b in enumerate(boundaries):
        ax.axvline(b, color="0.45", ls=":", lw=1.1,
                   label="Unverified recording join" if legend and idx == 0 else None)


def summary_plot(out: Path, rows: list[dict], library: dict, config: dict,
                 raw, boundaries, events):
    ends = floats(rows, "end")
    assigned = floats(rows, "assigned")
    size = floats(rows, "library_size")
    nearest = floats(rows, "nearest_distance")
    preceding = floats(rows, "predecessor_lsmash")
    k = len(library["patterns"])
    fig, axes = plt.subplots(5 if raw is not None else 4, 1, figsize=(14, 12),
                             sharex=True, constrained_layout=True)
    a = 0
    if raw is not None:
        axes[0].plot(np.arange(raw.size), raw, color="0.25", lw=0.65)
        axes[0].set_ylabel("Raw telemetry")
        decorate(axes[0], boundaries, events, legend=True)
        axes[0].legend(loc="upper right", fontsize=8)
        a = 1

    ax = axes[a]
    ax.step(ends, assigned, where="post", lw=1.5, marker=".", ms=4)
    for r in rows:
        if r["status"] == "new_pattern":
            ax.axvline(int(r["end"]), color="tab:orange", lw=1, alpha=0.75)
        elif r["status"] == "matched_switch":
            ax.axvline(int(r["end"]), color="tab:purple", lw=1, alpha=0.75)
    ax.set_ylabel("Assigned generator")
    ax.set_yticks(np.arange(k))
    ax.set_ylim(-0.5, max(k - 0.5, 0.5))
    decorate(ax, boundaries, events)

    ax = axes[a + 1]
    ax.plot(ends, nearest, ".-", ms=3.5, lw=0.9, label="Nearest-library LSmash")
    ax.plot(ends, preceding, ".-", ms=3.5, lw=0.9, label="Predecessor-window LSmash")
    ax.axhline(float(config["novelty_threshold"]), color="tab:blue",
               ls="--", lw=1, alpha=0.8, label="Novelty threshold")
    ax.axhline(float(config["switch_threshold"]), color="tab:orange",
               ls="--", lw=1, alpha=0.8, label="Boundary threshold")
    for r in rows:
        if r.get("crossing_supported", "").lower() == "true":
            ax.axvline(int(r["end"]), color="tab:red", alpha=0.3, lw=0.8)
    ax.set_ylabel("Native LSmash distance")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    decorate(ax, boundaries, events)

    ax = axes[a + 2]
    ax.step(ends, size, where="post", lw=1.5)
    ax.set_ylim(0, max(k + 0.8, 1.8))
    ax.set_ylabel("Library size")
    decorate(ax, boundaries, events)

    ax = axes[a + 3]
    probs = [json.loads(r["occurrence_posterior"]) for r in rows]
    for i in range(k):
        values = [p[i] if i < len(p) else np.nan for p in probs]
        ax.plot(ends, values, lw=1.2, label=f"G{i}")
    ax.set_ylabel("Window occurrence\nprobability")
    ax.set_ylim(-0.02, 1.02)
    if k <= 12:
        ax.legend(loc="upper right", ncol=min(6, k), fontsize=8)
    decorate(ax, boundaries, events)
    ax.set_xlabel("Observation index (completed window end)")

    fig.suptitle(f"Streaming Patternly | {len(rows)} windows, {k} generators\n"
                 "Anomaly labels shown only for evaluation; not used in inference",
                 fontsize=13)
    dest = out / "streaming_dashboard.png"
    fig.savefig(dest, dpi=170)
    plt.close(fig)
    return dest


def matrix_plots(out: Path, library: dict):
    d = np.asarray(library["library_lsmash"], dtype=float)
    p = np.asarray(library["transition_probabilities"], dtype=float)
    counts = np.asarray(library["transition_counts"], dtype=int)
    k = d.shape[0]
    if d.shape != (k, k) or p.shape != (k, k) or counts.shape != (k, k):
        raise ValueError("Library and transition matrices have incompatible dimensions")
    fig, axes = plt.subplots(1, 3, figsize=(max(12, 2.7 * k), 4.3),
                             constrained_layout=True)
    configs = ((d, "Library exemplar LSmash", ".3f"),
               (counts, "Observed transition counts", "d"),
               (p, "Smoothed transition probabilities", ".3f"))
    for ax, (mat, title, fmt) in zip(axes, configs):
        im = ax.imshow(mat, cmap="viridis", aspect="equal", vmin=0)
        fig.colorbar(im, ax=ax, fraction=0.047, pad=0.03)
        ax.set_title(title)
        ax.set_xticks(range(k), labels=[f"G{i}" for i in range(k)])
        ax.set_yticks(range(k), labels=[f"G{i}" for i in range(k)])
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")
        if k <= 12:
            for i in range(k):
                for j in range(k):
                    val = int(mat[i, j]) if fmt == "d" else float(mat[i, j])
                    ax.text(j, i, format(val, fmt), ha="center", va="center",
                            fontsize=8, color="white")
    dest = out / "library_matrices.png"
    fig.savefig(dest, dpi=170)
    plt.close(fig)
    return dest


def network_plot(out: Path, library: dict):
    patterns = library["patterns"]
    n = len(patterns)
    counts = np.asarray(library["transition_counts"], dtype=int)
    probabilities = np.asarray(library["transition_probabilities"], dtype=float)
    pts = np.array([[np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)]
                    for i in range(n)]) if n > 1 else np.array([[0.0, 0.0]])
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, pattern in enumerate(patterns):
        x, y = pts[i]
        radius = 0.13 + 0.09 * float(pattern["occurrence_probability"]) ** 0.5
        node = Circle((x, y), radius, facecolor=f"C{i % 10}",
                      edgecolor="black", lw=1.0, alpha=0.77, zorder=3)
        ax.add_patch(node)
        ax.text(x, y, f"G{i}\np={pattern['occurrence_probability']:.2f}",
                ha="center", va="center", fontsize=9, zorder=4)

    for i in range(n):
        for j in range(n):
            c = int(counts[i, j])
            if c == 0:
                continue  # Smoothed but unobserved edges are not plotted.
            x, y = pts[i]
            tx, ty = pts[j]
            if i == j:
                start, end = (x - .12, y + .13), (x + .12, y + .13)
                rad = -1.9
                lx, ly = x, y + .50
            else:
                start = (x + .17 * (tx - x) / np.hypot(tx - x, ty - y),
                         y + .17 * (ty - y) / np.hypot(tx - x, ty - y))
                end = (tx - .17 * (tx - x) / np.hypot(tx - x, ty - y),
                       ty - .17 * (ty - y) / np.hypot(tx - x, ty - y))
                rad = 0.15 if counts[j, i] else 0.0
                lx, ly = (x + tx) / 2, (y + ty) / 2
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                         connectionstyle=f"arc3,rad={rad}", mutation_scale=14,
                         color="0.3", lw=1 + np.log1p(c) * 0.6, zorder=1))
            ax.text(lx, ly, f"{c} ({probabilities[i, j]:.2f})",
                    fontsize=8, ha="center", va="center")
    ax.set_aspect("equal")
    ax.set_xlim(-1.8 if n > 1 else -1.0, 1.8 if n > 1 else 1.0)
    ax.set_ylim(-1.65 if n > 1 else -0.85, 1.65 if n > 1 else 1.0)
    ax.axis("off")
    ax.set_title("Empirical directed pattern transitions\n"
                 "Edge: observed count (Dirichlet-smoothed probability)")
    dest = out / "transition_graph.png"
    fig.savefig(dest, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="Original NASA .npz for raw waveform and annotation overlays")
    args = parser.parse_args()
    out = args.result
    library = json.loads((out / "library.json").read_text())
    config = json.loads((out / "configuration.json").read_text())
    rows = read_windows(out / "windows.csv")
    source = args.input
    if source is None and config.get("input"):
        source = Path(config["input"])
    raw, boundaries, events = load_raw(source)
    generated = [
        summary_plot(out, rows, library, config, raw, boundaries, events),
        matrix_plots(out, library),
        network_plot(out, library),
    ]
    print("VISUALIZATION", json.dumps({
        "windows": len(rows), "patterns": len(library["patterns"]),
        "plots": [str(p) for p in generated],
        "note": "Only one pattern was discovered: transitions cannot reveal switching."
                if len(library["patterns"]) == 1 else "Inspect boundaries and transitions.",
    }, indent=2))


if __name__ == "__main__":
    main()
