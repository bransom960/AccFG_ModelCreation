import importlib.util
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / 'molecule-fg-data' / 'assign_clusters_to_models.py'
SPEC = importlib.util.spec_from_file_location('assign_clusters_to_models', MODULE_PATH)
assign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assign)

CLUSTERS = pd.DataFrame([
    {'cluster_id': 0, 'cluster_weight': 90, 'centroid_fgs': 'amine', 'centroid_n_fgs': 1},
    {'cluster_id': 1, 'cluster_weight': 10, 'centroid_fgs': 'ether', 'centroid_n_fgs': 1},
])
SPECS = pd.DataFrame([
    {'model_name': 'Lookup', 'target_coverage': 0.05},
    {'model_name': 'Main', 'target_coverage': 0.50},
])


def test_coarse_clusters_are_flagged_with_a_rerun_message():
    assignments, achieved = assign.assign_models(CLUSTERS, SPECS)
    report = assign.build_report(assignments, achieved, SPECS, total_weight=100, tolerance=0.02)

    assert not report['within_tolerance'].all()
    message = assign.too_coarse_message(report, CLUSTERS, 100, 0.02)
    assert '--max-components' in message
    assert 'cluster 0 holds 90.0%' in message
    for model in report.loc[~report['within_tolerance'], 'model_name']:
        assert model in message


def test_targets_met_within_tolerance():
    specs = pd.DataFrame([
        {'model_name': 'A', 'target_coverage': 0.89},  # 90% is within 2 points
        {'model_name': 'B', 'target_coverage': 0.11},  # 10% is within 2 points
    ])
    assignments, achieved = assign.assign_models(CLUSTERS, specs)
    report = assign.build_report(assignments, achieved, specs, total_weight=100, tolerance=0.02)

    assert report['within_tolerance'].all()


def test_exit_status_constant():
    assert assign.EXIT_NEEDS_MORE_CLUSTERS == 3
