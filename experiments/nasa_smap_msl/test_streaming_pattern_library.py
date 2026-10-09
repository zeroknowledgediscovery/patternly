"""Orchestration tests ONLY: FakeBackend is not used by the user-facing CLI.

These tests do not validate compiled LSmash numerics or GenESeSS inference.
"""
from pathlib import Path
import json

import numpy as np
import pytest

from streaming_pattern_library import (
    OnlinePatternLibrary, check_distance, stream_windows, NativeError,
)


class FakeBackend:
    def __init__(self, fail_symbol=None):
        self.fail_symbol = fail_symbol
        self.inferred = []

    def distances(self, rows):
        means = np.array([np.mean(x) for x in rows])
        return np.abs(means[:, None] - means[None, :])

    def infer(self, row, model_file, eps):
        if self.fail_symbol is not None and int(row[0]) == self.fail_symbol:
            raise NativeError("injected native failure")
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_text("FAKE MODEL: TEST ONLY")
        self.inferred.append(list(map(int, row)))
        return {"engine": "FAKE TEST BACKEND - NOT A GENESSESS RESULT", "model_file": str(model_file)}


def build(tmp_path, values, boundaries=()):
    library = OnlinePatternLibrary(FakeBackend(), tmp_path, eps=0.1,
                                   novelty_threshold=0.3, switch_threshold=0.3)
    for start, seg, row in stream_windows(np.asarray(values, dtype=np.uint32),
                                          window=4, stride=4, boundaries=list(boundaries)):
        library.observe(row, start=start, segment=seg)
    library.save()
    return library


def test_library_grows_and_transition_graph(tmp_path):
    lib = build(tmp_path, [0] * 8 + [1] * 8 + [0] * 4)
    assert [r["assigned"] for r in lib.records] == [0, 0, 1, 1, 0]
    assert [r["status"] for r in lib.records] == ["new_pattern", "matched", "new_pattern", "matched", "matched_switch"]
    assert lib.counts.tolist() == [3, 2]
    assert lib.transitions.tolist() == [[1, 1], [1, 1]]
    assert np.allclose(lib.library_matrix, [[0, 1], [1, 0]])
    assert [p["occurrence_probability"] for p in lib.result()["patterns"]] == pytest.approx([3.5 / 6, 2.5 / 6])
    assert np.allclose(lib.result()["transition_probabilities"], [[0.5, 0.5], [0.5, 0.5]])
    assert (tmp_path / "library.json").exists()
    assert (tmp_path / "edges.csv").exists()
    assert len(list((tmp_path / "matrices").glob("*.csv"))) == 2


def test_no_transition_across_unknown_seam(tmp_path):
    lib = build(tmp_path, [0] * 12 + [1] * 8, boundaries=(12,))
    assert lib.transitions.tolist() == [[2, 0], [0, 1]]
    assert lib.records[3]["predecessor_lsmash"] is None
    assert lib.records[3]["status"] == "new_pattern"


def test_prefix_invariance_with_fixed_backend(tmp_path):
    stream = [0] * 8 + [1] * 8 + [0] * 4
    partial = build(tmp_path / "partial", stream[:16])
    full = build(tmp_path / "full", stream)
    for a, b in zip(partial.records, full.records[:len(partial.records)]):
        assert a == b
    assert np.array_equal(partial.transitions, [[1, 1], [0, 1]])


def test_failed_native_inference_is_not_library_entry(tmp_path):
    lib = OnlinePatternLibrary(FakeBackend(fail_symbol=1), tmp_path,
                               eps=0.1, novelty_threshold=0.3,
                               switch_threshold=0.3)
    for i, symbol in enumerate([0, 1, 0]):
        lib.observe(np.full(4, symbol, dtype=np.uint32), start=4 * i, segment=0)
    assert len(lib.patterns) == 1
    assert lib.records[1]["status"] == "native_inference_failed"
    assert lib.records[1]["assigned"] == -1
    assert lib.transitions.tolist() == [[0]]
    assert lib.counts.tolist() == [2]


def test_overlaps_avoided_at_seam_and_prefix_calibration():
    windows = list(stream_windows(np.arange(20, dtype=np.uint32), 4, 2, [11], min_start=5))
    assert windows[0][0] == 5
    assert all(not (start < 11 < start + 4) for start, _, _ in windows)
    assert windows[0][1] == 0
    assert any(segment == 1 for _, segment, _ in windows)


def test_invalid_matrix_rejected():
    with pytest.raises(NativeError):
        check_distance(np.ones((2, 3)), 2)
    with pytest.raises(NativeError):
        check_distance(np.array([[0, 1], [2, 0.0]]), 2)