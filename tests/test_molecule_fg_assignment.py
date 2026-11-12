import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / 'molecule-fg-data' / 'assign_clusters_to_models.py'
SPEC = importlib.util.spec_from_file_location('assign_clusters_to_models', MODULE_PATH)
assign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assign)


def test_whole_clusters_can_hit_targets_exactly():
    atoms = [(0,), (1,), (2,), (3,)]
    counts = np.array([40.0, 30.0, 20.0, 10.0])
    targets = np.array([0.5, 0.6])  # 40+10 and 30+20+10 (or 40+20)

    chosen, optimal, _ = assign.solve_assignment(atoms, counts, 4, targets)

    assert optimal
    assert chosen.any(axis=0).all()  # every cluster in at least one model
    np.testing.assert_allclose(assign.coverage(chosen, atoms, counts), targets)


def test_shared_molecules_are_counted_once():
    atoms = [(0,), (0, 1), (1,)]
    counts = np.array([10.0, 5.0, 10.0])

    assert assign.coverage(np.array([[True, True]]), atoms, counts)[0] == 1.0
    assert assign.coverage(np.array([[True, False]]), atoms, counts)[0] == 15 / 25


def test_every_cluster_is_assigned_even_when_it_overshoots():
    atoms = [(0,), (1,)]
    counts = np.array([90.0, 10.0])
    targets = np.array([0.05, 0.5])

    chosen, _, _ = assign.solve_assignment(atoms, counts, 2, targets)

    assert chosen.any(axis=0).all()
    assert np.abs(assign.coverage(chosen, atoms, counts) - targets).max() > 0.02


def test_build_atoms_groups_patterns_by_membership_set():
    pattern_clusters = pd.DataFrame([
        {'count': 3, 'cluster_id': 0, 'cluster_memberships': '0,1'},
        {'count': 2, 'cluster_id': 1, 'cluster_memberships': '1,0'},
        {'count': 4, 'cluster_id': 1, 'cluster_memberships': '1'},
    ])

    atoms, counts = assign.build_atoms(pattern_clusters)

    assert atoms == [(0, 1), (1,)]
    assert counts.tolist() == [5.0, 4.0]


def test_report_lists_models_that_received_no_cluster():
    cluster_summary = pd.DataFrame([
        {'cluster_id': 0, 'cluster_weight': 100, 'centroid_fgs': 'amine', 'centroid_n_fgs': 1},
    ])
    model_specs = pd.DataFrame([
        {'model_name': 'Big', 'target_coverage': 1.0},
        {'model_name': 'Tiny', 'target_coverage': 0.01},
    ])

    assignments, achieved = assign.assign_models(cluster_summary, model_specs)
    report = assign.build_report(assignments, achieved, model_specs, total_weight=100)

    assert list(report['model_name']) == ['Big', 'Tiny']
    assert report.set_index('model_name').loc['Tiny', 'clusters_used'] == 0
