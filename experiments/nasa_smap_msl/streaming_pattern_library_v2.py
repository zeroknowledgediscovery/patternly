#!/usr/bin/env python3
"""Streaming PFSA pattern library V2: true inferred-generator distinctness.

Reuses native, subprocess-isolated GenESeSS and LSmash from streaming_pattern_library.
LSmash decides whether a switching candidate deserves examination. Admission compares
EXACT finite-block output distributions of the inferred PFSAs (not their training
windows), using Jensen-Shannon divergence. A parametric GenESeSS self-bootstrap
calibrates a model-specific threshold. No future stream data or anomaly labels.

Requires installed native lsmash and zedsuite for actual CLI runs. Synthetic
orchestration tests inject a labelled fake native backend, never for CLI.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Protocol
import numpy as np

from streaming_pattern_library import (NativeBackend, NativeError, Pattern,
                                       check_distance, prepare_input, stream_windows,
                                       OnlinePatternLibrary)


def load_generator(path: Path, alphabet: int | None = None) -> dict:
    """Parse the *actual* GenESeSS model, retaining only real positive transitions.

    Missing destinations are harmless ONLY when their emission has zero probability.
    Such entries are set to zero in this probability-only representation; no
    positive-probability transition is invented or modified.
    """
    lines = path.read_text().splitlines()
    def section(title, marker):
        headers = [i for i, s in enumerate(lines) if s.startswith(title)]
        if len(headers) != 1:
            raise NativeError(f'Missing/ambiguous {title} in native GenESeSS PFSA: {path}')
        head = headers[0]
        try:
            count = int(lines[head].split('size(')[1].split(')')[0])
        except (IndexError, ValueError) as e:
            raise NativeError(f'Invalid PFSA header: {path}') from e
        if head + 1 >= len(lines) or lines[head+1].strip() != marker:
            raise NativeError(f'Invalid PFSA section marker: {path}')
        body = lines[head+2:head+2+count]
        if len(body) != count:
            raise NativeError(f'Incomplete native PFSA section: {path}')
        return body
    probs = [[float(v) for v in line.split()] for line in section('%PITILDE:', '#PITILDE')]
    edges = [[int(v) for v in line.split()] for line in section('%CONNX:', '#CONNX')]
    p = np.asarray(probs, dtype=float)
    nxt = np.asarray(edges, dtype=int)
    if p.ndim != 2 or nxt.shape != p.shape or p.shape[0] < 1:
        raise NativeError(f"Malformed native model at {path}")
    if alphabet is not None and p.shape[1] != alphabet:
        raise NativeError(f"Native model alphabet {p.shape[1]} != expected {alphabet}: {path}")
    if not np.isfinite(p).all() or np.any(p < 0):
        raise NativeError(f"Invalid emission probabilities: {path}")
    if not np.allclose(p.sum(axis=1), 1, atol=2e-4, rtol=0):
        raise NativeError(f"Native rows do not sum to 1: {path}")
    # Native text serialization can round morph rows slightly; preserve the
    # inferred probabilities up to print precision while enforcing stochastic rows.
    p = p / p.sum(axis=1, keepdims=True)
    bad = ((nxt < 0) | (nxt >= p.shape[0])) & (p > 1e-12)
    if np.any(bad):
        raise NativeError(f"Native model contains nonzero-probability invalid transitions: {path}")
    # Zero-mass transitions never contribute to probabilities; their destination is immaterial.
    nxt = np.where((nxt < 0) | (nxt >= p.shape[0]), 0, nxt)
    return {'p': p, 'next': nxt}


def stationary(g: dict) -> np.ndarray:
    """Stationary probability distribution for a single recurrent class.

    Reject ambiguous multi-recurrent-class models, rather than selecting an
    arbitrary state mixture that could change the claimed generator distance.
    """
    p, nxt = g['p'], g['next']
    n, k = p.shape
    T = np.zeros((n, n), dtype=float)
    for i in range(n):
        for a in range(k):
            T[i, nxt[i, a]] += p[i, a]
    A = np.vstack([T.T - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1
    pi, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < n or pi.min() < -1e-7 or np.linalg.norm(pi @ T - pi, ord=1) > 1e-7:
        raise NativeError('Generator has ambiguous/nonstationary state distribution')
    pi = np.maximum(pi, 0.)
    return pi / pi.sum()


def block_laws(g: dict, depth: int) -> list[np.ndarray]:
    """Exact stationary word probability distribution at lengths 1,...,depth."""
    if depth < 1:
        raise ValueError('depth must be positive')
    p, nxt = g['p'], g['next']
    n, k = p.shape
    vecs = stationary(g)[None, :]
    result = []
    for step in range(1, depth + 1):
        expanded = np.zeros((len(vecs) * k, n), dtype=float)
        for a in range(k):
            for s in range(n):
                expanded[a::k, nxt[s, a]] += vecs[:, s] * p[s, a]
        vecs = expanded
        law = vecs.sum(axis=1)
        if not np.isclose(law.sum(), 1, atol=1e-7):
            raise NativeError('Finite-block probabilities fail normalization')
        result.append(law)
    return result


def js_bits(u: np.ndarray, v: np.ndarray) -> float:
    u, v = np.asarray(u, float), np.asarray(v, float)
    if u.shape != v.shape:
        raise ValueError('Incompatible block probabilities')
    m = .5 * (u + v)
    a = u > 0
    b = v > 0
    return float(.5 * (np.sum(u[a] * np.log2(u[a] / m[a])) +
                         np.sum(v[b] * np.log2(v[b] / m[b]))))


def generator_distance(a: dict, b: dict, depth: int = 4) -> float:
    """Average JS divergence in bits of *exact* block laws, [0,1].

    It is a process-level divergence, NOT a C++ LSmash model distance and
    NOT necessarily an asymptotic entropy-rate divergence.
    """
    if a['p'].shape[1] != b['p'].shape[1]:
        raise NativeError('Generator alphabets differ')
    return float(np.mean([js_bits(u, v) for u, v in zip(block_laws(a, depth),
                                                        block_laws(b, depth))]))


def simulate_generator(g: dict, length: int, rng: np.random.Generator) -> np.ndarray:
    """Draw exactly from the real inferred unifilar PFSA; no fitted surrogate."""
    p, nxt = g['p'], g['next']
    state = int(rng.choice(len(p), p=stationary(g)))
    out = np.empty(length, dtype=np.uint32)
    for t in range(length):
        sym = int(rng.choice(p.shape[1], p=p[state]))
        out[t] = sym
        state = int(nxt[state, sym])
    return out


class GeneratorBackend(NativeBackend):
    def infer(self, row: np.ndarray, model_file: Path, eps: float) -> dict:
        info = super().infer(row, model_file, eps)
        native = load_generator(model_file, alphabet=self.alphabet)
        # Matrices loaded from the actual native PFSA, not re-fitted.
        stationary(native)
        info['alphabet'] = int(native['p'].shape[1])
        return info


@dataclass
class Pending:
    model: dict
    generator: dict
    window: np.ndarray
    first_end: int
    confirmations: int
    segment: int


class GeneratorLibrary(OnlinePatternLibrary):
    def __init__(self, backend, output: Path, *, eps: float, screen_threshold: float,
                 switch_threshold: float, alphabet: int, depth: int,
                 null_replicates: int, null_quantile: float, min_generator_js: float,
                 confirmations: int, confirmation_js: float, seed: int,
                 alpha: float = .5):
        super().__init__(backend, output, eps=eps, novelty_threshold=screen_threshold,
                         switch_threshold=switch_threshold, alpha=alpha)
        if not (2 <= null_replicates and 0 < null_quantile < 1 and
                0 <= min_generator_js <= 1 and confirmations >= 1 and
                0 <= confirmation_js <= 1 and 1 <= depth <= 9 and 0 <= screen_threshold):
            raise ValueError('Invalid calibration or generator comparison parameters')
        self.alphabet = alphabet
        self.depth = depth
        self.null_replicates = null_replicates
        self.null_quantile = null_quantile
        self.min_generator_js = min_generator_js
        self.confirmations = confirmations
        self.confirmation_js = confirmation_js
        self.rng = np.random.default_rng(seed)
        self.generators: list[dict] = []
        self.generator_matrix = np.zeros((0, 0), float)
        self.null_thresholds: list[float] = []
        self.null_details: list[dict] = []
        self.pending: Pending | None = None
        self.candidate_id = 0
        (self.out / 'candidates').mkdir(exist_ok=True)

    def infer_candidate(self, window: np.ndarray) -> tuple[dict, dict]:
        self.candidate_id += 1
        path = self.out / 'candidates' / f'candidate_{self.candidate_id:06d}.pfsa'
        meta = self.backend.infer(window, path, self.eps)
        g = load_generator(path, self.alphabet) if isinstance(self.backend, NativeBackend) else self.backend.load_model(path)
        block_laws(g, self.depth)
        return meta, g

    def calibrate(self, g: dict, window_len: int) -> tuple[float, dict]:
        distances, errors = [], []
        for r in range(self.null_replicates):
            boot = simulate_generator(g, window_len, self.rng)
            try:
                meta, inferred = self.infer_candidate(boot)
                distances.append(generator_distance(g, inferred, self.depth))
            except Exception as exc:
                errors.append(f'replicate={r}: {exc!r}')
        # The null is not trustworthy if inference routinely fails.
        min_valid = max(2, math.ceil(.8 * self.null_replicates))
        if len(distances) < min_valid:
            raise NativeError(f'Insufficient successful native self-bootstrap replicates '
                              f'({len(distances)}/{self.null_replicates}), examples: {errors[:3]}')
        threshold = max(self.min_generator_js, float(np.quantile(distances, self.null_quantile)))
        return threshold, dict(null_draws=distances, errors=errors,
                               threshold=threshold, valid=len(distances),
                               attempted=self.null_replicates)

    def admit(self, window: np.ndarray, end: int, meta: dict, generator: dict):
        # Calibrate BEFORE mutation; failed null does not create a library entry.
        threshold, null_info = self.calibrate(generator, len(window))
        ident = len(self.patterns)
        source = Path(meta['model_file'])
        dest = self.out / 'models' / f'pattern_{ident:04d}.pfsa'
        # Test backend models also become durable independent files.
        import shutil
        shutil.copy2(source, dest)
        meta = {**meta, 'model_file': str(dest)}
        old = len(self.generators)
        new_matrix = np.zeros((old + 1, old + 1), float)
        if old:
            new_matrix[:old, :old] = self.generator_matrix
            for j, prior in enumerate(self.generators):
                d = generator_distance(prior, generator, self.depth)
                new_matrix[old, j] = new_matrix[j, old] = d
        self.patterns.append(Pattern(ident, window.copy(), end, meta))
        self.counts = np.pad(self.counts, (0, 1))
        self.transitions = np.pad(self.transitions, ((0, 1), (0, 1)))
        self.generators.append(generator)
        self.generator_matrix = new_matrix
        self.null_thresholds.append(threshold)
        self.null_details.append(null_info)
        np.savetxt(self.out / 'matrices' / f'generator_at_{end:09d}.csv',
                   self.generator_matrix, delimiter=',', fmt='%.12g')
        return ident

    def observe(self, window: np.ndarray, *, start: int, segment: int) -> dict:
        window = np.asarray(window, dtype=np.uint32)
        end = start + len(window) - 1
        same_seg = segment == self.last_segment
        if not same_seg:
            self.pending = None
        prev = self.last_window if same_seg else None
        old_label = self.last_pattern if same_seg else None
        size = len(self.patterns)
        D = None
        ds = []
        predecessor = None
        closest = None
        if size:
            rows = [p.exemplar for p in self.patterns] + [window]
            if prev is not None:
                rows.append(prev)
            D = self.backend.distances(rows)
            ds = D[:size, size].tolist()
            closest = int(np.argmin(ds))
            predecessor = float(D[size, size + 1]) if prev is not None else None
            self.library_matrix = D[:size, :size].copy()
        boundary = predecessor is not None and predecessor >= self.switch_threshold
        may_infer = (not size or (self.pending is not None and same_seg) or
                     ((prev is None or boundary) and min(ds) > self.novelty_threshold))
        status = 'matched'
        chosen = closest
        gen_distances = []
        gen_thresholds = []
        error = ''
        new_meta = None
        try:
            if not size:
                new_meta, g = self.infer_candidate(window)
                chosen = self.admit(window, end, new_meta, g)
                self.library_matrix = np.zeros((1, 1), float)
                status = 'new_pattern'
            elif may_infer:
                new_meta, g = self.infer_candidate(window)
                gen_distances = [generator_distance(g, old, self.depth) for old in self.generators]
                gen_thresholds = self.null_thresholds.copy()
                nearest_gen = int(np.argmin(gen_distances))
                # Null threshold: candidate must be *more* distant than within-generator
                # inference variability from EVERY known generator.
                is_distinct = all(d > t for d, t in zip(gen_distances, gen_thresholds))
                if not is_distinct:
                    chosen = nearest_gen
                    status = 'generator_duplicate'
                    self.pending = None
                else:
                    if self.pending is not None and self.pending.segment == segment:
                        pending_distance = generator_distance(g, self.pending.generator, self.depth)
                        confirms = pending_distance <= self.confirmation_js
                    else:
                        confirms = False
                    if confirms:
                        self.pending.confirmations += 1
                        self.pending.window = window.copy()
                        self.pending.generator = g
                        self.pending.model = new_meta
                    else:
                        self.pending = Pending(new_meta, g, window.copy(), end, 1, segment)
                    if self.pending.confirmations >= self.confirmations:
                        chosen = self.admit(window, end, new_meta, g)
                        # The new exemplar is the just-scored current window.
                        if D is not None:
                            inds = list(range(size)) + [size]
                            self.library_matrix = D[np.ix_(inds, inds)].copy()
                        status = 'new_pattern'
                        self.pending = None
                    else:
                        chosen = old_label if old_label is not None else closest
                        status = 'candidate_pending'
            elif size:
                if closest != old_label and old_label is not None and not boundary:
                    chosen = old_label
                    status = 'held_no_boundary'
                elif closest != old_label and old_label is not None:
                    status = 'matched_switch'
                elif min(ds) > self.novelty_threshold:
                    status = 'unresolved_novelty_no_boundary'
                # Pending confirmation requires consecutive qualifying windows.
                self.pending = None
        except Exception as exc:
            status = 'native_inference_failed'
            error = repr(exc)
            chosen = -1
            self.pending = None
        if chosen is not None and chosen >= 0:
            self.counts[chosen] += 1
            self.patterns[chosen].count += 1
            if old_label is not None and old_label >= 0 and same_seg:
                self.transitions[old_label, chosen] += 1
        snap = self.result()
        record = dict(start=start, end=end, segment=segment, assigned=chosen,
                      status=status, closest=closest,
                      nearest_distance=min(ds) if ds else None,
                      predecessor_lsmash=predecessor, crossing_supported=bool(boundary),
                      library_size=len(self.patterns), exemplar_distances=ds,
                      generator_distances=gen_distances,
                      generator_thresholds=gen_thresholds,
                      pending_confirmations=self.pending.confirmations if self.pending else 0,
                      occurrence_posterior=[p['occurrence_probability'] for p in snap['patterns']],
                      error=error)
        self.records.append(record)
        self.last_window = window.copy()
        self.last_pattern = chosen if chosen is not None and chosen >= 0 else None
        self.last_segment = segment
        return record

    def result(self) -> dict:
        output = super().result()
        output.update(generator_js=self.generator_matrix.tolist(),
                      generator_depth=self.depth,
                      generator_null_thresholds=self.null_thresholds,
                      generator_null_details=self.null_details,
                      admission='Exact finite-block JS of native inferred GenESeSS PFSAs; '
                                'parametric self-bootstrap per library PFSA; '
                                'candidate confirmation; LSmash only switching prefilter')
        return output

    def save(self):
        result = self.result()
        (self.out / 'library.json').write_text(json.dumps(result, indent=2) + '\n')
        if self.patterns:
            np.save(self.out / 'exemplars.npy', np.stack([p.exemplar for p in self.patterns]))
            np.savetxt(self.out / 'library_lsmash.csv', self.library_matrix, delimiter=',')
            np.savetxt(self.out / 'generator_js.csv', self.generator_matrix, delimiter=',')
        fields = ['start', 'end', 'segment', 'assigned', 'status', 'closest',
                  'nearest_distance', 'predecessor_lsmash', 'crossing_supported',
                  'library_size', 'exemplar_distances', 'generator_distances',
                  'generator_thresholds', 'pending_confirmations',
                  'occurrence_posterior', 'error']
        with (self.out / 'windows.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in self.records:
                w.writerow({k: json.dumps(row[k]) if isinstance(row[k], list)
                            else row.get(k) for k in fields})
        with (self.out / 'edges.csv').open('w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['source', 'target', 'count', 'probability'])
            for i in range(len(self.patterns)):
                for j in range(len(self.patterns)):
                    w.writerow([i, j, int(self.transitions[i, j]),
                                result['transition_probabilities'][i][j]])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--input-kind', choices=['continuous', 'symbols'], default='continuous')
    p.add_argument('--npz-key', default='value')
    p.add_argument('--column')
    p.add_argument('--fit-length', type=int, default=500)
    p.add_argument('--alphabet', type=int, default=4)
    p.add_argument('--window', type=int, default=512)
    p.add_argument('--stride', type=int, default=512)
    p.add_argument('--screen-threshold', type=float, default=0.0,
                   help='LSmash exemplar distance prefilter; 0 means test every switching candidate')
    p.add_argument('--switch-threshold', type=float, default=0.02)
    p.add_argument('--epsilon', type=float, default=.2)
    p.add_argument('--generator-depth', type=int, default=4)
    p.add_argument('--generator-null-replicates', type=int, default=16)
    p.add_argument('--generator-null-quantile', type=float, default=.95)
    p.add_argument('--min-generator-js', type=float, default=.03)
    p.add_argument('--confirmations', type=int, default=2)
    p.add_argument('--confirmation-js', type=float, default=.05)
    p.add_argument('--dirichlet-alpha', type=float, default=.5)
    p.add_argument('--seed', type=int, default=47)
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.input_kind == 'symbols' and args.alphabet > 4:
        if args.alphabet ** args.generator_depth > 65536:
            p.error('Generator depth too large for alphabet: max 65536 words')
    symbols, boundaries, metadata = prepare_input(args.input, False, args.input_kind,
                                                    args.fit_length, args.alphabet,
                                                    args.npz_key, args.column, args.seed)
    alphabet = metadata['alphabet']
    if alphabet ** args.generator_depth > 65536:
        p.error('Generator depth too large for alphabet: max 65536 words')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'configuration.json').write_text(json.dumps(dict(
        input=str(args.input), samples=len(symbols), window=args.window,
        stride=args.stride, switch_threshold=args.switch_threshold,
        screen_threshold=args.screen_threshold, novelty_threshold=args.screen_threshold,
        epsilon=args.epsilon, generator_depth=args.generator_depth,
        generator_null_replicates=args.generator_null_replicates,
        generator_null_quantile=args.generator_null_quantile,
        min_generator_js=args.min_generator_js,
        confirmations=args.confirmations, confirmation_js=args.confirmation_js,
        dirichlet_alpha=args.dirichlet_alpha, provenance=metadata,
        validation='native numerical integration NOT run in the development sandbox'), indent=2) + '\n')
    backend = GeneratorBackend(args.out / '_native_work', timeout=args.timeout)
    backend.alphabet = alphabet
    lib = GeneratorLibrary(backend, args.out, eps=args.epsilon,
        screen_threshold=args.screen_threshold, switch_threshold=args.switch_threshold,
        alphabet=alphabet, depth=args.generator_depth,
        null_replicates=args.generator_null_replicates,
        null_quantile=args.generator_null_quantile,
        min_generator_js=args.min_generator_js, confirmations=args.confirmations,
        confirmation_js=args.confirmation_js, seed=args.seed,
        alpha=args.dirichlet_alpha)
    min_start = args.fit_length if metadata['kind'] == 'continuous' else 0
    for start, segment, window in stream_windows(symbols, args.window, args.stride,
                                                  boundaries, min_start=min_start):
        rec = lib.observe(window, start=start, segment=segment)
        if rec['status'] != 'matched':
            print('PATTERN_EVENT', json.dumps({k: rec[k] for k in [
                'start', 'end', 'status', 'assigned', 'library_size',
                'nearest_distance', 'predecessor_lsmash',
                'generator_distances', 'generator_thresholds',
                'pending_confirmations', 'error']}), flush=True)
        if rec['status'] == 'native_inference_failed':
            lib.save()
            raise NativeError(f'Native inference/calibration failed at start={start}: {rec["error"]}')
    lib.save()
    print('COMPLETE', json.dumps(dict(windows=len(lib.records),
           patterns=len(lib.patterns), result=str(args.out / 'library.json'))), flush=True)


if __name__ == '__main__':
    main()