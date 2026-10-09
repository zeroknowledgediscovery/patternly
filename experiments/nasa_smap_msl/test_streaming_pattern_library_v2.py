"""Algorithmic/orchestration tests. FakeBackend is NEVER used by CLI."""
from pathlib import Path
import numpy as np
import pytest
from streaming_pattern_library_v2 import (generator_distance, block_laws, stationary,
                                           simulate_generator, GeneratorLibrary,
                                           load_generator)


def bern(p):
    return {'p': np.array([[p, 1-p]]), 'next': np.array([[0,0]])}


class FakeBackend:
    """TEST ONLY: trivial deterministic backend for inference + scoring orchestration."""
    def distances(self, rows):
        means = np.array([np.mean(r) for r in rows], dtype=float)
        return np.abs(means[:,None] - means[None,:])

    def infer(self, row, model_file, eps):
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.write_text('FAKE FOR TESTS')
        return {'model_file': str(model_file), 'engine': 'FAKE TEST ONLY'}

    def load_model(self, path):
        # Infer from the candidate generation input persisted by this fake backend.
        return self._models[str(path)]

    def with_capture(self, row, model_file, eps):
        self.infer(row, model_file, eps)


class FakeWithCapture(FakeBackend):
    def __init__(self):
        self._models = {}

    def infer(self, row, model_file, eps):
        info = super().infer(row, model_file, eps)
        p = np.clip(1 - np.mean(row), 0.05, 0.95)
        self._models[str(model_file)] = bern(p)
        return info


def test_exact_js_zero_and_symmetric():
    a=bern(.2); b=bern(.8)
    assert generator_distance(a,a,depth=5) == pytest.approx(0,abs=1e-14)
    assert generator_distance(a,b,depth=5) == pytest.approx(generator_distance(b,a,depth=5))
    assert .1 < generator_distance(a,b,depth=5) <= 1
    assert np.isclose(block_laws(a,4)[-1].sum(),1)


def test_stationary_and_draw():
    a=bern(.75)
    assert np.allclose(stationary(a),[1])
    x=simulate_generator(a,20000,np.random.default_rng(47))
    assert .72 < np.mean(x == 0) < .78


def test_state_relabel_invariance():
    g={'p':np.array([[.9,.1],[.2,.8]]), 'next': np.array([[0,1],[0,1]])}
    h={'p':g['p'][::-1].copy(), 'next':1-g['next'][::-1]}
    assert generator_distance(g,h,5) == pytest.approx(0,abs=1e-12)


def test_model_file_missing_zero_mass_edge(tmp_path):
    path=tmp_path/'model.pfsa'
    path.write_text('%PITILDE: size(1)\n#PITILDE\n1 0\n%CONNX: size(1)\n#CONNX\n0 -1\n')
    g=load_generator(path,2)
    assert np.array_equal(g['next'],[[0,0]])
    assert np.allclose(g['p'],[[1,0]])


def test_model_file_rejects_positive_missing_edge(tmp_path):
    path=tmp_path/'model.pfsa'
    path.write_text('%PITILDE: size(1)\n#PITILDE\n.9 .1\n%CONNX: size(1)\n#CONNX\n0 -1\n')
    with pytest.raises(Exception):
        load_generator(path,2)


def test_pending_confirms_no_spurious_new_models(tmp_path):
    backend=FakeWithCapture()
    lib=GeneratorLibrary(backend,tmp_path,eps=.1,screen_threshold=0,
                         switch_threshold=.2,alphabet=2,depth=3,
                         null_replicates=8,null_quantile=.9,min_generator_js=.02,
                         confirmations=2,confirmation_js=.1,seed=47)
    assigned=[]
    for i,v in enumerate([0,0,1,1,1,0]):
        r=lib.observe(np.full(64,v,np.uint32),start=i*64,segment=0)
        assigned.append(r['assigned'])
    lib.save()
    assert len(lib.patterns)==2
    assert lib.records[2]['status']=='candidate_pending'
    assert lib.records[3]['status']=='new_pattern'
    assert lib.records[4]['status'] in ('matched','held_no_boundary')
    assert lib.records[5]['status']=='matched_switch'
    assert lib.generator_matrix.shape==(2,2)
    assert lib.generator_matrix[0,1] > lib.null_thresholds[0]
    assert (tmp_path/'generator_js.csv').is_file()
    assert (tmp_path/'windows.csv').is_file()


def test_no_transition_at_segment_seam(tmp_path):
    lib=GeneratorLibrary(FakeWithCapture(),tmp_path,eps=.2,screen_threshold=0,
                         switch_threshold=.2,alphabet=2,depth=2,
                         null_replicates=6,null_quantile=.9,min_generator_js=.01,
                         confirmations=2,confirmation_js=.2,seed=47)
    lib.observe(np.zeros(64,dtype=np.uint32),start=0,segment=0)
    lib.observe(np.ones(64,dtype=np.uint32),start=64,segment=1)
    assert lib.transitions.sum()==0
    assert lib.records[-1]['predecessor_lsmash'] is None