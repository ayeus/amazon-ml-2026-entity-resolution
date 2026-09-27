import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from decide import f05_entity, macro_f05, select_expected_f05


def test_worked_example_from_statement():
    assert abs(f05_entity({"a", "b", "c"}, {"a", "c"}) - 0.714) < 1e-3   # 2/3 precision, recall 1


def test_empty_cases():
    assert f05_entity(set(), set()) == 1.0
    assert f05_entity({"a"}, set()) == 0.0
    assert f05_entity(set(), {"a"}) == 0.0


def test_perfect_and_disjoint():
    assert f05_entity({"a", "b"}, {"a", "b"}) == 1.0
    assert f05_entity({"x"}, {"a"}) == 0.0


def test_vectorised_matches_scalar():
    cases = [({"a", "b", "c"}, {"a", "c"}), (set(), set()), ({"a"}, {"a", "b", "c"}), ({"a"}, set())]
    tp = [len(p & t) for p, t in cases]; n = [len(p) for p, t in cases]; nt = [len(t) for p, t in cases]
    m, f = macro_f05(tp, n, nt)
    assert np.allclose(f, [f05_entity(p, t) for p, t in cases])


def test_expected_f05_selects_confident_and_empties_singletons():
    D = pd.DataFrame({"i1": [0, 0, 0, 1, 1], "y": [1, 1, 0, 0, 0]})
    p = np.array([0.99, 0.95, 0.05, 0.02, 0.01])
    sel = select_expected_f05(D, p)
    assert sel.tolist() == [True, True, False, False, False]
