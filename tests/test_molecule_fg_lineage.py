import importlib.util
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / 'molecule-fg-data' / 'assign_clusters_to_models.py'
SPEC = importlib.util.spec_from_file_location('assign_clusters_to_models', MODULE_PATH)
assign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assign)

PATTERN_CLUSTERS = pd.DataFrame([
    {'pattern_index': 0, 'cluster_id': 0, 'cluster_primary': 0, 'cluster_memberships': '0', 'member_cids': '1,2'},
    {'pattern_index': 1, 'cluster_id': 0, 'cluster_primary': 0, 'cluster_memberships': '0,1', 'member_cids': '3'},
    {'pattern_index': 2, 'cluster_id': 1, 'cluster_primary': 1, 'cluster_memberships': '1', 'member_cids': '4'},
    # cluster 2 has no primary members at all
    {'pattern_index': 3, 'cluster_id': 1, 'cluster_primary': 1, 'cluster_memberships': '1,2', 'member_cids': '5,6'},
])


def test_lineage_includes_secondary_members():
    assignments = pd.DataFrame({'model_name': ['A', 'B'], 'cluster_id': [1, 2]})

    lineage = assign.build_lineage_map(assignments, PATTERN_CLUSTERS)

    a = lineage[lineage['model_name'] == 'A']
    assert sorted(a['pattern_index']) == [1, 2, 3]  # pattern 1 joined cluster 1 as a secondary
    assert a.set_index('pattern_index').loc[1, 'is_primary'] == False  # noqa: E712
    b = lineage[lineage['model_name'] == 'B']
    assert list(b['pattern_index']) == [3]           # cluster 2 is not dropped
    assert list(b['member_cids']) == ['5,6']


def test_every_listed_molecule_count_matches_cluster_membership():
    assignments = pd.DataFrame({'model_name': ['A', 'A'], 'cluster_id': [0, 1]})

    lineage = assign.build_lineage_map(assignments, PATTERN_CLUSTERS)

    per_cluster = lineage.groupby('cluster_id')['member_cids'].apply(
        lambda s: sum(len(c.split(',')) for c in s))
    assert per_cluster.to_dict() == {0: 3, 1: 4}
