import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'molecule-fg-data' / 'assign_clusters_to_models.py'
SPEC = importlib.util.spec_from_file_location('assign_clusters_to_models', MODULE_PATH)
assign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assign)

# Two clusters, 60% and 40% of molecules. Targets: A 100%, B 40%.
ATOMS = [(0,), (1,)]
COUNTS = np.array([60.0, 40.0])
TARGETS = np.array([1.0, 0.4])


def test_without_the_rule_b_shares_everything():
    chosen, optimal, _ = assign.solve_assignment(ATOMS, COUNTS, 2, TARGETS)
    assert optimal
    np.testing.assert_allclose(assign.coverage(chosen, ATOMS, COUNTS), [1.0, 0.4])
    assert assign.exclusive_coverage(chosen, ATOMS, COUNTS)[1] == 0.0  # B has nothing of its own


def test_every_model_gets_its_exclusive_share():
    chosen, optimal, _ = assign.solve_assignment(ATOMS, COUNTS, 2, TARGETS,
                                                 min_exclusive=np.array([0.1, 0.1]))
    assert optimal
    alone = assign.exclusive_coverage(chosen, ATOMS, COUNTS)
    assert (alone >= 0.1 - 1e-9).all()
    assert chosen.any(axis=0).all()  # every cluster is still assigned


def test_shared_atoms_are_not_exclusive():
    atoms = [(0,), (0, 1), (1,)]
    counts = np.array([10.0, 5.0, 10.0])
    both = np.array([[True, False], [False, True]])  # A has cluster 0, B has cluster 1
    np.testing.assert_allclose(assign.exclusive_coverage(both, atoms, counts), [10 / 25, 10 / 25])


def test_impossible_rule_raises_no_assignment():
    cluster_summary = pd.DataFrame([
        {'cluster_id': 0, 'cluster_weight': 100, 'centroid_fgs': 'amine', 'centroid_n_fgs': 1},
    ])
    specs = pd.DataFrame([{'model_name': 'A', 'target_coverage': 0.5},
                          {'model_name': 'B', 'target_coverage': 0.5}])
    # One cluster cannot be exclusive to two models.
    with pytest.raises(assign.NoAssignment) as err:
        assign.assign_models(cluster_summary, specs, min_exclusive=np.array([0.1, 0.1]))
    assert err.value.infeasible
