import os, sys, zlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
from make_dataset import is_val
from competition import _channel
from cheap_filter import select
from make_outputs import lists_by_s1, write_atomic


def test_split_is_by_s1_and_disjoint():
    ids = np.array([f"S1-{i}" for i in range(20000)], dtype=object)
    v = is_val(ids)
    assert 0.18 < v.mean() < 0.22                 # ~20% validation
    assert not (set(ids[v]) & set(ids[~v]))       # no S1 in both train and validation
    assert (is_val(ids) == v).all()               # deterministic


def test_cheap_filter_keeps_at_least_min_keep_per_s1():
    i1 = np.array([0, 0, 0, 0, 1, 1, 2])
    p = np.array([0.9, 0.001, 0.0005, 0.0001, 0.0002, 0.0001, 0.0])
    k = select(p, i1, tau=0.5, min_keep=2)
    assert k.tolist() == [True, True, False, False, True, True, True]   # every S1 with candidates keeps something


def test_competition_channel_is_label_free_and_correct():
    key = np.array([1, 1, 1, 2, 3, 3]); v = np.array([.9, .5, .2, .7, .4, .8], np.float32)
    bo, rk, sz = _channel(key, v)
    assert rk.tolist() == [1, 2, 3, 1, 2, 1] and sz.tolist() == [3, 3, 3, 1, 2, 2]
    assert abs(bo[0] - 0.5) < 1e-6 and abs(bo[1] - 0.9) < 1e-6 and bo[3] == 0


def test_output_lists_one_row_per_s1_and_empty_for_none(tmp_path):
    out = lists_by_s1(4, np.array([0, 0, 2]), np.array(["S2-1", "S3-2", "S2-9"], dtype=object))
    assert out == ["S2-1,S3-2", "", "S2-9", ""]
    p = tmp_path / "x.tsv"
    write_atomic(str(p), ["source1_entity_id\tmatched_entity_ids", "S1-1\tS2-1,S3-2", "S1-2\t"])
    lines = p.read_text().split("\n")
    assert lines[0] == "source1_entity_id\tmatched_entity_ids" and lines[2] == "S1-2\t"
