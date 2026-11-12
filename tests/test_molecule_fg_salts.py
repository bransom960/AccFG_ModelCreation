import json
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1] / 'molecule-fg-data'
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import build_sample_fg_dataset as stage1  # noqa: E402


def test_salts_and_mixtures_are_dropped_and_listed(tmp_path):
    path = tmp_path / 'smiles.json'
    path.write_text(json.dumps([
        {'cid': 1, 'smiles': 'CCO'},
        {'cid': 2, 'smiles': 'CC(=O)[O-].[Na+]'},
        {'cid': 3, 'smiles': 'CCN'},
    ]))

    records, _ = stage1.load_smiles_records(path)

    assert records == [(1, 'CCO'), (3, 'CCN')]
    assert stage1.load_smiles_list(path) == ['CCO', 'CCN']
    assert stage1.load_rejected_records(path) == [(2, 'CC(=O)[O-].[Na+]', 'salt_or_mixture')]


def test_aromatic_smiles_is_not_a_mixture():
    assert not stage1.is_salt_or_mixture('c1ccccc1')
    assert stage1.is_salt_or_mixture('Cl.c1ccncc1')
