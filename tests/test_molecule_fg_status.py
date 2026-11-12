import sys
from pathlib import Path

import pandas as pd

MODULE_DIR = Path(__file__).resolve().parents[1] / 'molecule-fg-data'
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import build_pattern_count_dictionary as stage2  # noqa: E402


def test_every_input_molecule_gets_a_status(tmp_path):
    patterns = pd.DataFrame({'pattern_index': [0, 1], 'pattern': ['000', '010'], 'count': [1, 2]})
    molecules = pd.DataFrame({'cid': ['1', '2', '3'], 'pattern_index': [1, 0, 1]})
    fg_dir = tmp_path / 'fg'
    (fg_dir / '_rejects').mkdir(parents=True)
    (fg_dir / '_rejects' / 'part.txt').write_text(
        '4\tCC.Cl\tsalt_or_mixture\t\n5\tC(\tunparseable\t\n')

    status = stage2.molecule_status(patterns, molecules, stage2.read_stage1_rejects(fg_dir))

    assert status.set_index('cid')['status'].to_dict() == {
        '1': 'clustered',
        '2': 'rejected_no_fg',
        '3': 'clustered',
        '4': 'rejected_salt_or_mixture',
        '5': 'rejected_unparseable',
    }
