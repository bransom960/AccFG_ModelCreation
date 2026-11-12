import sys
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parents[1] / 'molecule-fg-data'
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import build_pattern_count_dictionary as stage2  # noqa: E402


def test_compress_counts_patterns_and_keeps_cids():
    fg_names = ['a', 'b', 'c']
    chunks = [
        (fg_names, ['11', '12'], np.array([[0, 1, 0], [1, 0, 0]], dtype=np.uint8)),
        (fg_names, ['13', '14'], np.array([[0, 1, 0], [0, 0, 0]], dtype=np.uint8)),
    ]

    names, patterns, molecules = stage2.compress(iter(chunks))

    assert names == fg_names
    # pattern_index follows sorted pattern strings: '000' < '010' < '100'
    assert patterns['pattern'].tolist() == ['000', '010', '100']
    assert patterns['count'].tolist() == [1, 2, 1]
    assert molecules.set_index('cid')['pattern_index'].to_dict() == {'11': 1, '12': 2, '13': 1, '14': 0}


def test_fg_presence_csv_reader(tmp_path):
    path = tmp_path / 'fg_presence.csv'
    path.write_text('cid,Molecule,a,b,pattern_index\n7,CCO,1,0,1\n8,CCN,0,1,0\n')

    chunks = list(stage2.iter_fg_presence_csv(path))

    assert len(chunks) == 1
    names, cids, X = chunks[0]
    assert names == ['a', 'b']
    assert cids == ['7', '8']
    assert X.tolist() == [[1, 0], [0, 1]]
