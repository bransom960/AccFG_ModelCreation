import sys
from pathlib import Path

import pandas as pd

MODULE_DIR = Path(__file__).resolve().parents[1] / 'molecule-fg-data'
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import label_clusters  # noqa: E402


def test_pattern_columns_keep_leading_zeros(tmp_path):
    pattern = '0' * 500 + '1' + '0' * 33
    pd.DataFrame([{
        'pattern_index': 0,
        'pattern': pattern,
        'count': 3,
        'cluster_id': 0,
        'cluster_primary': 0,
        'cluster_memberships': '0',
        'cluster_representative': pattern,
        'member_cids': '7',
    }]).to_csv(tmp_path / 'pattern_clusters.csv', index=False)

    rows = label_clusters.load_cluster_rows(tmp_path / 'pattern_clusters.csv')

    assert rows['pattern'].iloc[0] == pattern
    assert rows['cluster_representative'].iloc[0] == pattern
    assert rows['member_cids'].iloc[0] == '7'
