import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from accfg import AccFG

MODULE_DIR = Path(__file__).resolve().parents[1] / 'molecule-fg-data'
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from patterns import fg_presence_rows

MODULE_PATH = Path(__file__).resolve().parents[1] / 'molecule-fg-data' / 'assign_clusters_to_models.py'
SPEC = importlib.util.spec_from_file_location('assign_clusters_to_models', MODULE_PATH)
ASSIGN_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSIGN_MODULE)

LABEL_MODULE_PATH = Path(__file__).resolve().parents[1] / 'molecule-fg-data' / 'label_clusters.py'
LABEL_SPEC = importlib.util.spec_from_file_location('label_clusters', LABEL_MODULE_PATH)
LABEL_MODULE = importlib.util.module_from_spec(LABEL_SPEC)
LABEL_SPEC.loader.exec_module(LABEL_MODULE)

CLUSTER_MODULE_PATH = Path(__file__).resolve().parents[1] / 'molecule-fg-data' / 'bernoulli_mixture_clustering.py'
CLUSTER_SPEC = importlib.util.spec_from_file_location('bernoulli_mixture_clustering', CLUSTER_MODULE_PATH)
CLUSTER_MODULE = importlib.util.module_from_spec(CLUSTER_SPEC)
CLUSTER_SPEC.loader.exec_module(CLUSTER_MODULE)

def test_lite():
    afg = AccFG(print_load_info=False, lite=True)
    smi = 'CCO'
    fgs = afg.run(smi, show_atoms=True, show_graph=False)
    assert {'hydroxy': [(2,)]} == fgs
    
def test_full_0():
    afg = AccFG(print_load_info=False, lite=False)
    smi = 'CCO'
    fgs = afg.run(smi, show_atoms=True, show_graph=False)
    assert {'primary hydroxyl': [(2,)]} == fgs

def test_full_1():
    afg = AccFG(print_load_info=False, lite=False)
    smi = 'O=C(O)C1=CCS[C@@H]2CC(=O)N12'
    fgs = afg.run(smi, show_atoms=True, show_graph=False)
    assert {'alkene': [(3, 4)],
            'azetidin-2-one': [(10, 9, 8, 7, 11)],
            'carboxylic acid': [(1, 0, 2)],
            'dialkyl thioether': [(6,)]} == fgs

def test_show_atoms_false():
    afg = AccFG(print_load_info=False, lite=True)
    smi = 'OCCCCO'
    fgs = afg.run(smi, show_atoms=False, show_graph=False)
    assert ['hydroxy'] == fgs
    
def test_stress_run_time():
    afg = AccFG(print_load_info=False, lite=True)
    smi = ['O=C(O)C1=CCS[C@@H]2CC(=O)N12']*100
    for s in smi:
        fgs = afg.run(s, show_atoms=True, show_graph=False)
    assert True
    
def test_user_defined_fgs():
    my_fgs_dict = {'Cephem': 'O=C(O)C1=CCS[C@@H]2CC(=O)N12', 'Thioguanine': 'Nc1nc(=S)c2[nH]cnc2[nH]1'}
    my_afg = AccFG(user_defined_fgs=my_fgs_dict,print_load_info=False)
    cephalosporin_C = 'CC(=O)OCC1=C(N2[C@@H]([C@@H](C2=O)NC(=O)CCC[C@H](C(=O)O)N)SC1)C(=O)O'
    fgs = my_afg.run(cephalosporin_C, show_atoms=True, show_graph=False)
    assert {'primary aliphatic amine': [(21,)], 
            'carboxylic acid': [(22, 23, 24)], 
            'carboxylic ester': [(1, 2, 3)], 
            'secondary amide': [(15, 16, 14)], 
            'Cephem': [(8, 7, 9, 6, 5, 27, 26, 25, 13, 11, 12, 10)]} == fgs

def test_exclude_fgs():
    afg = AccFG(print_load_info=False, lite=True, exclude_fgs=['hydroxy'])
    smi = 'CCO'
    fgs = afg.run(smi, show_atoms=True, show_graph=False)

    assert afg.search_fg_name('hydroxy') is False
    assert {} == fgs

def test_exclude_rings():
    afg = AccFG(print_load_info=False, exclude_fgs=['rings'])

    assert {} == afg.dict_fg_heterocycle
    assert afg.search_fg_name('benzene') is False
    assert afg.search_fg_name('pyridine') is False
    assert [] == afg.run('c1ccccc1', show_atoms=False, show_graph=False)
    assert 'pyridine' not in afg.run('c1ccncc1', show_atoms=False, show_graph=False)

def test_search_fg_name():
    afg = AccFG(print_load_info=False, lite=True)

    assert '[OX2H]' == afg.search_fg_name('hydroxy')
    assert afg.search_fg_name('not in dict') is False

def test_search_fg_smiles():
    afg = AccFG(print_load_info=False)

    assert 'benzene' == afg.search_fg_smiles('c1ccccc1')
    assert afg.search_fg_smiles('CC') is False


def test_fg_presence_dataframe():
    afg = AccFG(print_load_info=False, lite=True)
    df = afg.fg_presence_dataframe(['CCO', 'CCN'])

    assert 'Molecule' in df.columns
    assert 'hydroxy' in df.columns
    assert 'amine' in df.columns
    assert df.loc[df['Molecule'] == 'CCO', 'hydroxy'].item() == 1
    assert df.loc[df['Molecule'] == 'CCO', 'amine'].item() == 0
    assert df.loc[df['Molecule'] == 'CCN', 'amine'].item() == 1
    assert df.loc[df['Molecule'] == 'CCN', 'hydroxy'].item() == 0


def test_fg_presence_dataframe_append(tmp_path):
    afg = AccFG(print_load_info=False, lite=True)
    csv_path = tmp_path / 'fg_presence.csv'

    afg.fg_presence_dataframe(['CCO'], output_csv=str(csv_path), append_to_csv=False)
    afg.fg_presence_dataframe(['CCO', 'CCN'], output_csv=str(csv_path), append_to_csv=True)

    df = pd.read_csv(csv_path)
    assert list(df['Molecule']) == ['CCO', 'CCN']
    assert df.loc[df['Molecule'] == 'CCO', 'hydroxy'].item() == 1
    assert df.loc[df['Molecule'] == 'CCN', 'amine'].item() == 1


def test_fg_presence_dataframe_append_empty_file(tmp_path):
    afg = AccFG(print_load_info=False, lite=True)
    csv_path = tmp_path / 'fg_presence.csv'
    csv_path.write_text('')

    result = afg.update_fg_presence_csv(['CCO'], str(csv_path))

    assert list(result['Molecule']) == ['CCO']
    assert result.loc[result['Molecule'] == 'CCO', 'hydroxy'].item() == 1
    assert pd.read_csv(csv_path).loc[0, 'Molecule'] == 'CCO'


def test_fg_presence_graph_mentions_all_detected_groups():
    afg = AccFG(print_load_info=False, lite=False)
    smi = 'CCN(CC)CCC1=C2C3=CC=CC=C3CN2C4=C1C=C(C=C4)Br'
    vec = afg.fg_presence_vector(smi, canonical=True)

    assert vec['tertiary aliphatic amine'] == 1
    assert vec['Aryl bromide'] == 1
    assert vec['benzene'] == 1
    assert vec['1H-indole'] == 1
    assert vec['1H-pyrrole'] == 1
    assert vec['hetero N basic no H'] == 1


def test_assign_models_uses_every_cluster():
    cluster_summary = pd.DataFrame([
        {'cluster_id': 1, 'cluster_weight': 30, 'centroid_fgs': 'amide,benzene', 'centroid_n_fgs': 2},
        {'cluster_id': 2, 'cluster_weight': 30, 'centroid_fgs': 'alcohol', 'centroid_n_fgs': 1},
        {'cluster_id': 3, 'cluster_weight': 40, 'centroid_fgs': 'ether,amine,ketone,aromatic,halide', 'centroid_n_fgs': 9},
    ])
    model_specs = pd.DataFrame([
        {'model_name': 'Small', 'target_coverage': 0.4, 'min_fgs': 1, 'max_fgs': 2},
        {'model_name': 'Large', 'target_coverage': 0.4, 'min_fgs': 3, 'max_fgs': 5},
    ])

    assignments, achieved = ASSIGN_MODULE.assign_models(cluster_summary, model_specs)

    assert set(cluster_summary['cluster_id']) == set(assignments['cluster_id'])

    # Whole clusters only: a model's coverage is the sum of its clusters' full weights.
    weights = cluster_summary.set_index('cluster_id')['cluster_weight']
    total_weight = int(weights.sum())
    for model_name in model_specs['model_name']:
        ids = assignments.loc[assignments['model_name'] == model_name, 'cluster_id']
        assert abs(achieved[model_name] - weights.loc[ids].sum() / total_weight) < 1e-9


def test_cluster_summary_uses_unique_pattern_counts_once_per_cluster():
    group = pd.DataFrame([
        {'pattern_index': 0, 'count': 10},
        {'pattern_index': 0, 'count': 10},
        {'pattern_index': 1, 'count': 5},
    ])

    assert LABEL_MODULE.cluster_pattern_weight(group) == 15


def test_fg_presence_rows_keep_cid_for_pattern_join():
    afg = AccFG(print_load_info=False, lite=True)
    rows = fg_presence_rows(afg, ['CCO', 'CCN'], canonical=True, cid_values=[101, 202])

    assert rows[0]['cid'] == 101
    assert rows[1]['cid'] == 202
    assert {'pattern_index', 'Molecule', 'cid'} <= set(rows[0].keys())


def test_label_clusters_uses_canonical_cluster_file_only():
    assert LABEL_MODULE.DEFAULT_INPUT.name == 'pattern_clusters.csv'


def test_weighted_bernoulli_em_uses_pattern_counts():
    X = np.array([
        [1, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=np.int8)
    sample_weights = np.array([2.0, 1.0, 1.0], dtype=float)

    weights, means, resp = CLUSTER_MODULE.bernoulli_mixture_em(
        X,
        n_components=2,
        n_iter=50,
        tol=1e-6,
        seed=0,
        sample_weights=sample_weights,
        verbose=False,
    )

    assert 1 <= len(weights) <= 2
    assert np.allclose(weights.sum(), 1.0)
    assert np.all(np.isfinite(means))
    assert resp.shape[0] == 3
    assert np.all(resp >= 0)


def test_assign_models_counts_shared_molecules_once():
    # 100 unique molecules; 10 of them sit in both clusters, so the weights sum to 110.
    cluster_summary = pd.DataFrame([
        {'cluster_id': 0, 'cluster_weight': 70, 'centroid_fgs': 'amine,benzene', 'centroid_n_fgs': 2},
        {'cluster_id': 1, 'cluster_weight': 40, 'centroid_fgs': 'amide,ketone', 'centroid_n_fgs': 2},
    ])
    pattern_clusters = pd.DataFrame([
        {'pattern_index': 0, 'count': 60, 'cluster_id': 0, 'cluster_memberships': '0'},
        {'pattern_index': 1, 'count': 10, 'cluster_id': 0, 'cluster_memberships': '0,1'},
        {'pattern_index': 2, 'count': 30, 'cluster_id': 1, 'cluster_memberships': '1'},
    ])
    model_specs = pd.DataFrame([
        {'model_name': 'All', 'target_coverage': 1.0},
    ])

    assignments, achieved = ASSIGN_MODULE.assign_models(cluster_summary, model_specs, pattern_clusters)
    report = ASSIGN_MODULE.build_report(assignments, achieved, model_specs, total_weight=100)

    assert abs(achieved['All'] - 1.0) < 1e-9
    assert report['n_molecules'].iloc[0] == 100
    assert report['total_cluster_weight'].iloc[0] == 110


def test_patterns_to_matrix_accepts_numeric_matrix_without_string_conversion():
    X = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int8)

    assert np.array_equal(CLUSTER_MODULE.patterns_to_matrix(X), X)
