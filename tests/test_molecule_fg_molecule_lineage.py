import json
import sys
from pathlib import Path

import pandas as pd

MODULE_DIR = Path(__file__).resolve().parents[1] / 'molecule-fg-data'
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import molecule_lineage as lineage  # noqa: E402

FG_NAMES = ['alkene', 'hydroxy', 'ether']


def write_stage_outputs(out: Path):
    """Stages 1-5 for 5 molecules: 101 and 102 share a pattern in clusters 0 and 1, 103 is
    in cluster 1 only, 104 has no FG, 105 was rejected by stage 1 as a salt."""
    pd.DataFrame({'cid': ['101', '102', '103', '104'],
                  'Molecule': ['C=CO', 'C=CCO', 'COC', 'C'],
                  'alkene': [1, 1, 0, 0], 'hydroxy': [1, 1, 0, 0], 'ether': [0, 0, 1, 0],
                  }).to_csv(out / 'fg_presence.csv', index=False)
    pd.DataFrame({'cid': ['105'], 'Molecule': ['[Na+].[Cl-]'], 'reason': ['salt_or_mixture']}
                 ).to_csv(out / 'rejected_molecules.csv', index=False)
    (out / 'fg_columns.json').write_text(json.dumps(FG_NAMES))
    pd.DataFrame({'pattern_index': [0, 1, 2], 'pattern': ['000', '001', '110'],
                  'count': [1, 1, 2]}).to_csv(out / 'pubchem_like_pattern_counts.csv', index=False)
    pd.DataFrame({'cid': ['101', '102', '103', '104'], 'pattern_index': [2, 2, 1, 0]}
                 ).to_csv(out / 'molecule_patterns.csv', index=False)
    pd.DataFrame({'cid': ['101', '102', '103', '104', '105'],
                  'status': ['clustered'] * 3 + ['rejected_no_fg', 'rejected_salt_or_mixture'],
                  'detail': [''] * 5}).to_csv(out / 'molecule_status.csv', index=False)
    pd.DataFrame([
        {'pattern_index': 1, 'cluster_id': 1, 'cluster_primary': 1, 'cluster_memberships': '1'},
        {'pattern_index': 2, 'cluster_id': 1, 'cluster_primary': 1, 'cluster_memberships': '1,0'},
    ]).to_csv(out / 'pattern_clusters.csv', index=False)
    pd.DataFrame({'model_name': ['Small', 'Large', 'Large'], 'cluster_id': [0, 0, 1]}
                 ).to_csv(out / 'model_assignments.csv', index=False)


def test_one_row_per_molecule(tmp_path, monkeypatch):
    for name in ('FG_PRESENCE', 'STAGE1_REJECTS', 'PATTERN_COUNTS', 'MOLECULE_PATTERNS',
                 'MOLECULE_STATUS', 'FG_COLUMNS', 'PATTERN_CLUSTERS', 'MODEL_ASSIGNMENTS'):
        monkeypatch.setattr(lineage, name, tmp_path / getattr(lineage, name).name)
    write_stage_outputs(tmp_path)
    out = tmp_path / 'molecule_lineage.csv'

    assert lineage.main(['-o', str(out)]) == 0

    df = pd.read_csv(out, dtype={'cid': str}).set_index('cid')
    assert list(df.columns) == lineage.COLUMNS[1:]
    assert sorted(df.index) == ['101', '102', '103', '104', '105']

    row = df.loc['101']
    assert row['smiles'] == 'C=CO'
    assert row['fgs'] == 'alkene;hydroxy' and row['n_fgs'] == 2
    assert row['primary_cluster'] == 1
    assert row['clusters'] == '1,0'               # primary first, then the secondary
    assert row['models'] == 'Small;Large'         # reached through both clusters, once each

    assert df.loc['103', 'models'] == 'Large'
    assert df.loc['104', 'status'] == 'rejected_no_fg'
    assert pd.isna(df.loc['104', 'clusters']) and pd.isna(df.loc['104', 'models'])
    assert df.loc['105', 'smiles'] == '[Na+].[Cl-]'
    assert pd.isna(df.loc['105', 'pattern_index'])
