import importlib.util
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / 'molecule-fg-data' / 'assign_clusters_to_models.py'
SPEC = importlib.util.spec_from_file_location('assign_clusters_to_models', MODULE_PATH)
assign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assign)


def test_unique_molecules_per_model():
    # Pattern 1 (cid 3) is in clusters 0 and 1; model A has both clusters, so cid 3 counts once.
    lineage = pd.DataFrame([
        {'model_name': 'A', 'cluster_id': 0, 'pattern_index': 0, 'is_primary': True, 'member_cids': '1,2'},
        {'model_name': 'A', 'cluster_id': 0, 'pattern_index': 1, 'is_primary': True, 'member_cids': '3'},
        {'model_name': 'A', 'cluster_id': 1, 'pattern_index': 1, 'is_primary': False, 'member_cids': '3'},
        {'model_name': 'A', 'cluster_id': 1, 'pattern_index': 2, 'is_primary': True, 'member_cids': '4'},
        {'model_name': 'B', 'cluster_id': 1, 'pattern_index': 1, 'is_primary': False, 'member_cids': '3'},
        {'model_name': 'B', 'cluster_id': 1, 'pattern_index': 2, 'is_primary': True, 'member_cids': '4'},
    ])
    specs = pd.DataFrame([{'model_name': 'A', 'target_coverage': 0.9},
                          {'model_name': 'B', 'target_coverage': 0.5},
                          {'model_name': 'C', 'target_coverage': 0.1}])

    counts = assign.model_molecule_counts(lineage, specs, total_molecules=4).set_index('model_name')

    assert counts.loc['A', 'n_unique_molecules'] == 4
    assert counts.loc['A', 'n_exclusive_molecules'] == 2   # cids 1, 2
    assert counts.loc['A', 'n_shared_molecules'] == 2      # cids 3, 4 are also in B
    assert counts.loc['B', 'n_unique_molecules'] == 2
    assert counts.loc['B', 'n_exclusive_molecules'] == 0
    assert counts.loc['C', 'n_unique_molecules'] == 0      # a model with no cluster is listed
    assert counts.loc['ALL MODELS', 'n_unique_molecules'] == 4
    assert counts.loc['ALL MODELS', 'coverage'] == 1.0


def test_no_cids_means_no_table():
    lineage = pd.DataFrame({'model_name': ['A'], 'cluster_id': [0]})
    specs = pd.DataFrame([{'model_name': 'A', 'target_coverage': 1.0}])
    assert assign.model_molecule_counts(lineage, specs, total_molecules=1) is None
