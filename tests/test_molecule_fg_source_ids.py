import json
import sys
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / 'molecule-fg-data'
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import build_sample_fg_dataset as stage1  # noqa: E402


def test_source_ids_are_kept(tmp_path):
    path = tmp_path / 'smiles.json'
    path.write_text(json.dumps([
        {'cid': '176000001', 'smiles': 'CCO'},
        {'cid': 176000002, 'smiles': ' '},     # empty SMILES is skipped
        {'cid': 176000003, 'smiles': 'CCN'},
    ]))

    records, ids_are_positions = stage1.load_smiles_records(path)

    assert records == [(176000001, 'CCO'), (176000003, 'CCN')]
    assert ids_are_positions is False
    assert stage1.load_smiles_list(path) == ['CCO', 'CCN']


def test_plain_smiles_list_is_flagged_as_positional(tmp_path):
    path = tmp_path / 'smiles.json'
    path.write_text(json.dumps(['CCO', '', 'CCN']))

    records, ids_are_positions = stage1.load_smiles_records(path)

    assert records == [(0, 'CCO'), (2, 'CCN')]
    assert ids_are_positions is True


def test_object_without_cid_is_an_error(tmp_path):
    path = tmp_path / 'smiles.json'
    path.write_text(json.dumps([{'smiles': 'CCO'}]))

    with pytest.raises(ValueError):
        stage1.load_smiles_records(path)
