import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from accfg import AccFG
from patterns import cluster_pattern_counts, fg_presence_rows, pattern_count_dictionary


def test_fg_presence_rows_add_pattern_column():
    afg = AccFG(print_load_info=False, lite=True)
    rows = fg_presence_rows(afg, ['CCO', 'CCN'], canonical=True)

    assert len(rows) == 2
    assert 'pattern_index' in rows[0]
    assert rows[0]['Molecule'] == 'CCO'
    assert rows[0]['hydroxy'] == 1
    assert rows[1]['amine'] == 1
    assert isinstance(rows[0]['pattern_index'], int)


def test_pattern_count_dictionary_counts_unique_patterns():
    afg = AccFG(print_load_info=False, lite=True)
    counts = pattern_count_dictionary(afg, ['CCO', 'CCN', 'CCO'])

    assert sum(counts.values()) == 3
    assert len(counts) == 2
    assert all(isinstance(k, str) for k in counts)
    assert all(set(k) <= {'0', '1'} for k in counts)
    assert all(isinstance(v, int) for v in counts.values())


def test_cluster_pattern_counts_returns_k_clusters():
    df = pd.DataFrame(
        {
            'pattern_index': [0, 1, 2],
            'pattern': ['1100', '1110', '0011'],
            'count': [10, 8, 6],
        }
    )

    clustered = cluster_pattern_counts(df, k=2)

    assert 'cluster_id' in clustered.columns
    assert set(clustered['cluster_id'].unique()) == {0, 1}
    assert len(clustered) == 3
    assert clustered['count'].sum() == 24
