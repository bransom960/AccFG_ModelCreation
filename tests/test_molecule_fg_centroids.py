import sys
from pathlib import Path

import pandas as pd
import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / 'molecule-fg-data'
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import label_clusters  # noqa: E402

# Cluster 1's group, after load_cluster_rows expands memberships. The first row is a
# pattern whose primary cluster is 0, so its cluster_representative is cluster 0's centroid.
GROUP_1 = pd.DataFrame([
    {'pattern_index': 0, 'cluster_id': 1, 'cluster_primary': 0, 'cluster_representative': '1100'},
    {'pattern_index': 5, 'cluster_id': 1, 'cluster_primary': 1, 'cluster_representative': '0011'},
])


def test_uses_centroid_file_when_present():
    assert label_clusters.cluster_centroid(1, GROUP_1, {0: '1100', 1: '0111'}) == '0111'


def test_falls_back_to_a_row_whose_primary_is_the_cluster():
    assert label_clusters.cluster_centroid(1, GROUP_1, {}) == '0011'


def test_secondary_only_cluster_without_centroid_file_is_an_error():
    secondary_only = GROUP_1.iloc[:1]
    with pytest.raises(ValueError):
        label_clusters.cluster_centroid(1, secondary_only, {})


def test_load_centroids_keeps_leading_zeros(tmp_path):
    path = tmp_path / 'cluster_centroids.csv'
    pd.DataFrame({'cluster_id': [0], 'mixing_weight': [1.0], 'centroid': ['0010']}).to_csv(path, index=False)
    assert label_clusters.load_centroids(path) == {0: '0010'}
