import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parents[1] / 'molecule-fg-data'
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import bernoulli_mixture_clustering as bmc  # noqa: E402


def _two_group_patterns():
    return pd.DataFrame({
        'pattern_index': [0, 1],
        'pattern': ['110000', '000111'],
        'count': [30, 70],
    })


def test_every_surviving_component_has_a_member():
    out, means, weights = bmc.cluster_pattern_counts_overlapping(
        _two_group_patterns(), max_components=6, seed=0, verbose=False, prune_threshold=0.05)

    assert 1 <= len(weights) <= 6
    assert np.isclose(weights.sum(), 1.0)
    used = {int(c) for m in out['cluster_memberships'] for c in str(m).split(',')}
    assert used == set(range(len(weights)))  # every surviving component has a member


def test_refit_mixture_renormalises_and_sorts():
    # 10 rows, so the 1/n floor on the prune threshold (0.1) keeps both groups (25% / 75%).
    X = np.array([[1, 1, 0]] * 5 + [[0, 0, 1]] * 5, dtype=float)
    sample_weights = np.array([1.0] * 5 + [3.0] * 5)
    weights, means, resp = bmc.refit_mixture(
        X, sample_weights, weights=[0.2, 0.2], means=[[0.9, 0.9, 0.1], [0.1, 0.1, 0.9]])

    assert len(weights) == 2
    assert np.isclose(weights.sum(), 1.0)
    assert weights[0] > weights[1]  # sorted, largest first
    assert resp.shape == (10, 2)
