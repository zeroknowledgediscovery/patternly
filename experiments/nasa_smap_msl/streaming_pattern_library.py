#!/usr/bin/env python3
"""Causal online Patternly-style PFSA library with native LSmash switching evidence.

Native requirements: compiled `lsmash` and `zedsuite.genesess.GenESeSS`.
No algorithmic surrogate is used by the command-line application.

LSmash comparisons are BETWEEN OBSERVED EXEMPLAR WINDOWS, not an assertion that
LSmash calculates a metric between inferred PFSAs. PFSA models are persisted
separately; their native likelihoods are not used in this prototype's routing.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Optional, Protocol

import numpy as np


class NativeError(RuntimeError):
    pass


class Backend(Protocol):
    def distances(self, rows: list[np.ndarray]) -> np.ndarray: ...
    def infer(self, row: np.ndarray, model_file: Path, eps: float) -> dict: ...


class NativeBackend:
    """Run actual compiled native engines in isolated subprocesses.

    Isolation matters: GenESeSS and native LSmash have produced fatal malloc
    aborts on individual NASA telemetry channels. Never silently substitute.
    """
    def __init__(self, workdir: Path, timeout: int = 180):
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.script = Path(__file__).resolve()

    def _run(self, mode: str, payload: Path, result: Path, *, eps: Optional[float] = None) -> None:
        cmd = [sys.executable, str(self.script), "--_native-worker", mode,
               "--_payload", str(payload), "--_result", str(result)]
        if eps is not None:
            cmd += ["--_eps", str(eps)]
        if result.exists():
            result.unlink()
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True,
                                  timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise NativeError(f"{mode} native worker timeout after {self.timeout}s") from exc
        if proc.returncode != 0 or not result.exists():
            msg = (proc.stderr + "\n" + proc.stdout)[-2500:].strip()
            raise NativeError(f"{mode} native worker exit={proc.returncode}: {msg}")

    def distances(self, rows: list[np.ndarray]) -> np.ndarray:
        if not rows:
            raise ValueError("No sequences to compare")
        widths = {len(row) for row in rows}
        if len(widths) != 1:
            raise ValueError("LSmash input sequences must have identical length")
        payload = self.workdir / "distance_input.npz"
        result = self.workdir / "distance_output.npz"
        np.savez(payload, rows=np.asarray(rows, dtype=np.uint32))
        self._run("distance", payload, result)
        with np.load(result) as d:
            D = np.asarray(d["matrix"], dtype=float)
        return check_distance(D, len(rows))

    def infer(self, row: np.ndarray, model_file: Path, eps: float) -> dict:
        model_file.parent.mkdir(parents=True, exist_ok=True)
        payload = self.workdir / "infer_input.npz"
        result = self.workdir / "infer_output.json"
        np.savez(payload, sequence=np.asarray(row, dtype=np.uint32))
        # The model destination is explicitly supplied to the worker.
        cmd = [sys.executable, str(self.script), "--_native-worker", "infer",
               "--_payload", str(payload), "--_result", str(result),
               "--_model-file", str(model_file), "--_eps", str(eps)]
        if result.exists():
            result.unlink()
        if model_file.exists():
            model_file.unlink()
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True,
                                  timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise NativeError(f"infer native worker timeout after {self.timeout}s") from exc
        if proc.returncode or not result.exists() or not model_file.exists():
            msg = (proc.stderr + "\n" + proc.stdout)[-2500:].strip()
            raise NativeError(f"infer native worker exit={proc.returncode}: {msg}")
        return json.loads(result.read_text())


def native_worker(args: argparse.Namespace) -> None:
    """Native execution intentionally limited to this subprocess entrypoint."""
    if args._native_worker == "distance":
        import lsmash
        with np.load(args._payload) as z:
            rows = z["rows"].astype(int).tolist()
        opts = lsmash.LsmashOptions()
        opts.data_type = "symbolic"
        opts.sae = False
        D = check_distance(np.asarray(lsmash.from_sequences(rows, opts), dtype=float), len(rows))
        np.savez(args._result, matrix=D)
    else:
        import pandas as pd
        from zedsuite.genesess import GenESeSS
        with np.load(args._payload) as z:
            row = z["sequence"].astype(int).tolist()
        model_path = Path(args._model_file)
        generator = GenESeSS(data=pd.DataFrame([row]), outfile=str(model_path),
                             data_type="symbolic", data_dir="row", force=True,
                             eps=args._eps)
        if not generator.run() or not model_path.is_file() or model_path.stat().st_size == 0:
            raise NativeError("Genuine native GenESeSS failed to infer a PFSA")
        states = int(np.asarray(generator.probability_morph_matrix).shape[0])
        info = {"engine": "zedsuite.genesess.GenESeSS", "states": states,
                "epsilon_requested": args._eps,
                "epsilon_used": float(generator.epsilon_used),
                "model_file": str(model_path)}
        Path(args._result).write_text(json.dumps(info, indent=2) + "\n")


def check_distance(D: np.ndarray, n: int) -> np.ndarray:
    if D.shape != (n, n) or not np.isfinite(D).all():
        raise NativeError(f"Invalid native LSmash distance matrix {D.shape}, expected {(n, n)}")
    if not np.allclose(D, D.T, atol=1e-7, rtol=0):
        raise NativeError("Native LSmash matrix is not symmetric")
    if not np.allclose(np.diag(D), 0, atol=1e-7, rtol=0):
        raise NativeError("Native LSmash matrix has nonzero diagonal")
    if D.min() < -1e-7:
        raise NativeError("Native LSmash has negative distance entries")
    return D


@dataclass
class Pattern:
    id: int
    exemplar: np.ndarray
    discovered_at: int
    model: dict
    count: int = 0


class OnlinePatternLibrary:
    """Frozen pattern exemplars; observed-window LSmash routing; native PFSA archive.

    All reads and state changes at a call to `observe()` depend only on the
    current window and earlier observations. Distances are freshly evaluated
    for current exemplars, current window, and preceding window. The native
    library can change projection randomness from call to call; snapshots and
    matrix-drift diagnostics expose this rather than asserting invariance.
    """
    def __init__(self, backend: Backend, output: Path, *, eps: float,
                 novelty_threshold: float, switch_threshold: float,
                 alpha: float = 0.5):
        if novelty_threshold < 0 or switch_threshold < 0 or alpha <= 0:
            raise ValueError("thresholds must be nonnegative and Dirichlet alpha positive")
        self.backend = backend
        self.out = output
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "models").mkdir(exist_ok=True)
        (self.out / "matrices").mkdir(exist_ok=True)
        self.eps = eps
        self.novelty_threshold = novelty_threshold
        self.switch_threshold = switch_threshold
        self.alpha = alpha
        self.patterns: list[Pattern] = []
        self.counts = np.zeros(0, dtype=np.int64)
        self.transitions = np.zeros((0, 0), dtype=np.int64)
        self.library_matrix = np.empty((0, 0), dtype=float)
        self.last_window: Optional[np.ndarray] = None
        self.last_pattern: Optional[int] = None
        self.last_segment: Optional[int] = None
        self.records: list[dict] = []

    def _admit(self, window: np.ndarray, end: int) -> int:
        ident = len(self.patterns)
        model_file = self.out / "models" / f"pattern_{ident:04d}.pfsa"
        # Atomic admission: do not add a pattern unless native inference succeeds.
        metadata = self.backend.infer(window, model_file, self.eps)
        self.patterns.append(Pattern(ident, window.copy(), end, metadata))
        self.counts = np.pad(self.counts, (0, 1))
        self.transitions = np.pad(self.transitions, ((0, 1), (0, 1)))
        return ident

    def observe(self, window: np.ndarray, *, start: int, segment: int) -> dict:
        window = np.asarray(window, dtype=np.uint32)
        end = start + len(window) - 1
        same_segment = self.last_segment == segment
        prev = self.last_window if same_segment else None
        if not same_segment:
            self.last_pattern = None
            self.last_window = None
        earlier_count = len(self.patterns)
        distances: list[float] = []
        predecessor_distance = None
        best = None
        distance_drift = None
        D = None
        if earlier_count:
            rows = [p.exemplar for p in self.patterns] + [window]
            if prev is not None:
                rows.append(prev)
            D = self.backend.distances(rows)
            distances = [float(x) for x in D[:earlier_count, earlier_count]]
            best = int(np.argmin(distances))
            if prev is not None:
                predecessor_distance = float(D[earlier_count, earlier_count + 1])
            new_matrix = D[:earlier_count, :earlier_count].copy()
            if self.library_matrix.shape == new_matrix.shape:
                distance_drift = float(np.max(np.abs(new_matrix - self.library_matrix)))
            self.library_matrix = new_matrix
        boundary_evidence = (predecessor_distance is not None
                             and predecessor_distance >= self.switch_threshold)
        allow_admission = prev is None or boundary_evidence
        status = "matched"
        nearest_distance = min(distances) if distances else None
        old_label = self.last_pattern
        try:
            if not self.patterns or (nearest_distance > self.novelty_threshold and allow_admission):
                chosen = self._admit(window, end)
                status = "new_pattern"
                if D is None:
                    self.library_matrix = np.zeros((1, 1), dtype=float)
                else:
                    # Reuse pairwise values from one coherent native call.
                    indices = list(range(earlier_count)) + [earlier_count]
                    self.library_matrix = D[np.ix_(indices, indices)].copy()
                np.savetxt(self.out / "matrices" / f"at_{end:09d}.csv",
                           self.library_matrix, delimiter=",", fmt="%.12g")
            elif nearest_distance > self.novelty_threshold:
                chosen = old_label if old_label is not None else best
                status = "unresolved_novelty_no_boundary"
            elif old_label is not None and best != old_label and not boundary_evidence:
                chosen = old_label
                status = "held_no_boundary"
            else:
                chosen = best
                status = "matched" if chosen == old_label or old_label is None else "matched_switch"
        except (Exception,) as exc:
            # Native generator failed; never create a fictitious model.
            chosen = -1
            status = "native_inference_failed"
            failure = repr(exc)
        else:
            failure = ""
        if chosen is not None and chosen >= 0:
            self.counts[chosen] += 1
            self.patterns[chosen].count += 1
            if old_label is not None and same_segment and old_label >= 0:
                self.transitions[old_label, chosen] += 1
        snapshot = self.result()
        record = dict(start=start, end=end, segment=segment, assigned=chosen,
                      status=status, closest=best, nearest_distance=nearest_distance,
                      predecessor_lsmash=predecessor_distance,
                      predecessor_available=(prev is not None),
                      crossing_supported=bool(boundary_evidence),
                      library_size=len(self.patterns), library_matrix_drift=distance_drift,
                      exemplar_distances=distances,
                      occurrence_posterior=[p["occurrence_probability"] for p in snapshot["patterns"]],
                      outgoing_transition_posterior=(snapshot["transition_probabilities"][chosen]
                                                     if chosen is not None and chosen >= 0 else []),
                      error=failure)
        self.records.append(record)
        self.last_window = window.copy()
        self.last_pattern = chosen if chosen is not None and chosen >= 0 else None
        self.last_segment = segment
        return record

    def result(self) -> dict:
        k = len(self.patterns)
        n = int(self.counts.sum())
        alpha = self.alpha
        occupancy = ((self.counts + alpha) / (n + alpha * k)).tolist() if k else []
        transition_probs = ((self.transitions + alpha) /
                            (self.transitions.sum(axis=1, keepdims=True) + alpha * k)).tolist() if k else []
        return dict(patterns=[dict(id=p.id, discovered_at=p.discovered_at,
                                   model=p.model, occurrences=p.count,
                                   occurrence_probability=occupancy[p.id]) for p in self.patterns],
                    transition_counts=self.transitions.tolist(),
                    transition_probabilities=transition_probs,
                    library_lsmash=self.library_matrix.tolist(),
                    total_assigned_windows=n,
                    alpha=alpha,
                    probability_definition="Dirichlet-smoothed window occupancy and one-step Markov transitions")

    def save(self) -> None:
        summary = self.result()
        (self.out / "library.json").write_text(json.dumps(summary, indent=2) + "\n")
        if self.patterns:
            np.save(self.out / "exemplars.npy", np.stack([p.exemplar for p in self.patterns]))
            np.savetxt(self.out / "library_lsmash.csv", self.library_matrix,
                       delimiter=",", fmt="%.12g")
        with (self.out / "windows.csv").open("w", newline="") as f:
            fields = ["start", "end", "segment", "assigned", "status", "closest",
                      "nearest_distance", "predecessor_lsmash", "predecessor_available",
                      "crossing_supported",
                      "library_size", "library_matrix_drift", "exemplar_distances",
                      "occurrence_posterior", "outgoing_transition_posterior", "error"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for rec in self.records:
                writer.writerow({**rec, "exemplar_distances": json.dumps(rec["exemplar_distances"]),
                                 "occurrence_posterior": json.dumps(rec["occurrence_posterior"]),
                                 "outgoing_transition_posterior": json.dumps(rec["outgoing_transition_posterior"])})
        with (self.out / "edges.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "target", "count", "probability"])
            p = summary["transition_probabilities"]
            for i in range(len(self.patterns)):
                for j in range(len(self.patterns)):
                    writer.writerow([i, j, int(self.transitions[i, j]), p[i][j]])


def prepare_input(path: Optional[Path], demo: bool, input_kind: str,
                  fit_length: int, alphabet: int, key: str, column: Optional[str],
                  seed: int) -> tuple[np.ndarray, list[int], dict]:
    if demo:
        rng = np.random.default_rng(seed)
        # Piecewise context-dependent binary stream; known boundaries are
        # for inspecting behavior, NEVER consumed by detector itself.
        regimes = [(0.82, 0.18), (0.18, 0.82), (0.82, 0.18)]
        out = []
        for p00, p11 in regimes:
            bits = [int(rng.integers(2)), int(rng.integers(2))]
            for _ in range(2046):
                prob = p11 if bits[-2] == bits[-1] else p00
                bits.append(int(rng.random() < prob))
            out.extend(bits)
        raw = np.asarray(out, dtype=np.uint32)
        return raw, [], dict(kind="demo_symbols", boundaries_for_validation_only=[2048, 4096],
                             segment_boundaries=[])
    if path is None:
        raise ValueError("Supply --input or --demo")
    if path.suffix == ".npz":
        with np.load(path) as data:
            raw = np.asarray(data[key]).reshape(-1)
            boundaries = [int(x) for x in np.atleast_1d(data["segment_boundary"])] if "segment_boundary" in data else []
    elif path.suffix == ".npy":
        raw = np.load(path).reshape(-1)
        boundaries = []
    elif path.suffix == ".csv":
        import pandas as pd
        frame = pd.read_csv(path)
        raw = frame[column].to_numpy() if column else frame.iloc[:, 0].to_numpy()
        boundaries = []
    else:
        raw = np.loadtxt(path, dtype=float).reshape(-1)
        boundaries = []
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite input values")
    if input_kind == "symbols":
        if np.any(raw != np.floor(raw)) or raw.min() < 0 or raw.max() >= alphabet:
            raise ValueError("Symbol stream must contain integers from 0 to --alphabet-1")
        symbols = raw.astype(np.uint32)
        metadata = {"kind": "symbols", "alphabet": alphabet}
    else:
        if fit_length < 4 or fit_length > len(raw):
            raise ValueError("Continuous data require 4 <= --fit-length <= stream length")
        # Fixed past-only quantizer. Collapsed quantiles are intentional.
        edges = np.unique(np.quantile(raw[:fit_length], [0.25, 0.5, 0.75]))
        right = bool(len(edges) < 3 and len(edges) > 0 and edges[0] == np.min(raw[:fit_length]))
        symbols = np.digitize(raw, edges, right=right).astype(np.uint32)
        metadata = {"kind": "continuous", "fit_length": fit_length,
                    "quantizer_edges": edges.tolist(), "right_closed": right,
                    "alphabet": len(edges) + 1}
    if any(x <= 0 or x >= len(symbols) for x in boundaries):
        raise ValueError("Invalid segment boundary")
    return symbols, sorted(set(boundaries)), dict(metadata, segment_boundaries=sorted(set(boundaries)))


def stream_windows(symbols: np.ndarray, window: int, stride: int, boundaries: list[int],
                   min_start: int = 0):
    if window < 2 or stride < 1:
        raise ValueError("window>=2 and stride>=1 are required")
    stops = [0] + boundaries + [len(symbols)]
    for segment, (a, b) in enumerate(zip(stops, stops[1:])):
        # Prefix used to fit a continuous quantizer must not be used in
        # purported online scores before the quantizer-fit cutoff.
        for start in range(max(a, min_start), b - window + 1, stride):
            yield start, segment, symbols[start:start + window]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--input-kind", choices=["symbols", "continuous"], default="continuous")
    parser.add_argument("--npz-key", default="value")
    parser.add_argument("--column")
    parser.add_argument("--fit-length", type=int, default=1500)
    parser.add_argument("--alphabet", type=int, default=4)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--novelty-threshold", type=float)
    parser.add_argument("--switch-threshold", type=float)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out", type=Path, default=Path("results/streaming_pattern_library"))
    parser.add_argument("--_native-worker", choices=["distance", "infer"], help=argparse.SUPPRESS)
    parser.add_argument("--_payload", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_result", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_model-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_eps", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._native_worker:
        native_worker(args)
        return
    if args.novelty_threshold is None or args.switch_threshold is None:
        parser.error("Both --novelty-threshold and --switch-threshold must be explicitly supplied")
    symbols, boundaries, metadata = prepare_input(args.input, args.demo, args.input_kind,
                                                    args.fit_length, args.alphabet,
                                                    args.npz_key, args.column, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "configuration.json").write_text(json.dumps(dict(
        input=str(args.input) if args.input else "synthetic demonstration",
        samples=len(symbols), window=args.window, stride=args.stride,
        novelty_threshold=args.novelty_threshold, switch_threshold=args.switch_threshold,
        epsilon=args.epsilon, dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed, provenance=metadata,
        engine="compiled native lsmash and zedsuite GenESeSS; no surrogate"), indent=2) + "\n")
    backend = NativeBackend(args.out / "_native_work", timeout=args.timeout)
    library = OnlinePatternLibrary(backend, args.out, eps=args.epsilon,
                                   novelty_threshold=args.novelty_threshold,
                                   switch_threshold=args.switch_threshold,
                                   alpha=args.dirichlet_alpha)
    min_start = args.fit_length if (metadata["kind"] == "continuous") else 0
    for start, segment, row in stream_windows(symbols, args.window, args.stride, boundaries,
                                              min_start=min_start):
        try:
            result = library.observe(row, start=start, segment=segment)
        except NativeError as exc:
            library.save()
            raise NativeError(f"LSmash failed at window starting {start}; partial results saved: {exc}") from exc
        if result["status"] != "matched":
            print("PATTERN_EVENT", json.dumps({k: result[k] for k in
                  ["start", "end", "status", "assigned", "library_size", "nearest_distance",
                   "predecessor_lsmash", "error"]}), flush=True)
    library.save()
    print("COMPLETE", json.dumps({"windows": len(library.records),
                                  "patterns": len(library.patterns),
                                  "result": str(args.out / "library.json")}), flush=True)


if __name__ == "__main__":
    main()