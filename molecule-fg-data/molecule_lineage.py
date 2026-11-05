"""After stage 5: one row per input molecule, from its SMILES to its model(s).

Joins what stages 1-5 wrote, by cid, into csv_outputs/molecule_lineage.csv:

  cid             the source molecule ID (the PubChem CID for a run from the .jsonl files)
  smiles          the SMILES AccFG was run on
  status          clustered, rejected_no_fg, rejected_salt_or_mixture, rejected_unparseable
                  or rejected_accfg_error (from molecule_status.csv)
  pattern_index   its FG pattern (stage 2)
  n_fgs, fgs      how many and which functional groups it has, ';'-separated
  primary_cluster its most probable cluster (stage 3)
  clusters        every cluster it belongs to, primary first, ','-separated
  models          every model given any of those clusters (stage 5), ';'-separated

Rejected molecules keep their cid, SMILES and status; the other columns are empty.

The SMILES come from stage 1's output, the same place stage 2 read the FGs from:

    python3 molecule-fg-data/molecule_lineage.py --fg-dir /data/fg      # fg_matrix.py Parquet
    python3 molecule-fg-data/molecule_lineage.py                        # csv_outputs/fg_presence.csv

Add --parquet to also write molecule_lineage.parquet (smaller and faster to load).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'molecule-fg-data'
OUTPUT_DIR = DATA_DIR / 'csv_outputs'

FG_PRESENCE = OUTPUT_DIR / 'fg_presence.csv'
STAGE1_REJECTS = OUTPUT_DIR / 'rejected_molecules.csv'
PATTERN_COUNTS = OUTPUT_DIR / 'pubchem_like_pattern_counts.csv'
MOLECULE_PATTERNS = OUTPUT_DIR / 'molecule_patterns.csv'
MOLECULE_STATUS = OUTPUT_DIR / 'molecule_status.csv'
FG_COLUMNS = OUTPUT_DIR / 'fg_columns.json'
PATTERN_CLUSTERS = OUTPUT_DIR / 'pattern_clusters.csv'
MODEL_ASSIGNMENTS = OUTPUT_DIR / 'model_assignments.csv'
LINEAGE = OUTPUT_DIR / 'molecule_lineage.csv'

COLUMNS = ['cid', 'smiles', 'status', 'pattern_index', 'n_fgs', 'fgs',
           'primary_cluster', 'clusters', 'models']


def read_smiles(fg_dir: Path | None) -> pd.DataFrame:
    """cid -> SMILES for every molecule stage 1 saw, kept and rejected."""
    if fg_dir is not None:
        import pyarrow.dataset as ds
        kept = ds.dataset(fg_dir, format='parquet').to_table(columns=['cid', 'Molecule']).to_pandas()
        rows = []
        # fg_matrix.py: one `cid<TAB>smiles<TAB>code<TAB>detail` line per rejected molecule
        for path in sorted((fg_dir / '_rejects').glob('*.txt')):
            for line in path.read_text(encoding='utf-8').splitlines():
                parts = line.split('\t', 3)
                if len(parts) >= 2:
                    rows.append((parts[0], parts[1]))
        rejected = pd.DataFrame(rows, columns=['cid', 'Molecule'])
    else:
        kept = pd.read_csv(FG_PRESENCE, usecols=['cid', 'Molecule'], dtype=str)
        rejected = (pd.read_csv(STAGE1_REJECTS, usecols=['cid', 'Molecule'], dtype=str)
                    if STAGE1_REJECTS.exists() else pd.DataFrame(columns=['cid', 'Molecule']))
    both = pd.concat([kept, rejected], ignore_index=True)
    both['cid'] = both['cid'].astype(str)
    return (both.drop_duplicates('cid')
            .rename(columns={'Molecule': 'smiles'}))


def per_pattern(fg_names: list[str]) -> pd.DataFrame:
    """pattern_index -> n_fgs, fgs, primary_cluster, clusters, models.

    Worked out once per pattern (thousands) rather than per molecule (hundreds of
    thousands): every molecule with the same pattern has the same FGs, clusters and models.
    """
    patterns = pd.read_csv(PATTERN_COUNTS, usecols=['pattern_index', 'pattern'],
                           dtype={'pattern': str})
    bits = patterns['pattern'].str.zfill(len(fg_names))  # in case a reader dropped leading 0s
    fgs = [[fg_names[i] for i, b in enumerate(p) if b == '1'] for p in bits]
    out = pd.DataFrame({'pattern_index': patterns['pattern_index'],
                        'n_fgs': [len(f) for f in fgs],
                        'fgs': [';'.join(f) for f in fgs]})

    clusters = pd.read_csv(PATTERN_CLUSTERS, dtype={'cluster_memberships': str},
                           usecols=lambda c: c in {'pattern_index', 'cluster_id',
                                                   'cluster_primary', 'cluster_memberships'})
    primary_col = 'cluster_primary' if 'cluster_primary' in clusters.columns else 'cluster_id'
    clusters = clusters.drop_duplicates('pattern_index')
    member_of = []
    for memberships, primary in zip(clusters.get('cluster_memberships', clusters[primary_col]),
                                    clusters[primary_col]):
        ids = ([int(x) for x in str(memberships).split(',') if x.strip()]
               if pd.notna(memberships) and str(memberships).strip() else [int(primary)])
        member_of.append(ids)
    clusters = pd.DataFrame({'pattern_index': clusters['pattern_index'],
                             'primary_cluster': clusters[primary_col].astype(int),
                             'member_of': member_of})

    assignments = pd.read_csv(MODEL_ASSIGNMENTS, usecols=['model_name', 'cluster_id'])
    models_of = assignments.groupby('cluster_id')['model_name'].agg(list).to_dict()
    model_order = list(dict.fromkeys(assignments['model_name']))  # model_specs.csv order

    def models(ids: list[int]) -> str:
        got = {m for c in ids for m in models_of.get(c, [])}
        return ';'.join(m for m in model_order if m in got)

    clusters['clusters'] = [','.join(map(str, ids)) for ids in clusters['member_of']]
    clusters['models'] = [models(ids) for ids in clusters['member_of']]
    return out.merge(clusters.drop(columns='member_of'), on='pattern_index', how='left')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fg-dir', type=Path, default=None,
                        help='stage1/fg_matrix.py Parquet output (default: csv_outputs/fg_presence.csv)')
    parser.add_argument('-o', '--output', type=Path, default=LINEAGE)
    parser.add_argument('--parquet', action='store_true',
                        help='also write the table as .parquet next to the CSV')
    args = parser.parse_args(argv)

    needed = [MOLECULE_PATTERNS, MOLECULE_STATUS, PATTERN_COUNTS, FG_COLUMNS,
              PATTERN_CLUSTERS, MODEL_ASSIGNMENTS]
    if args.fg_dir is None:
        needed.append(FG_PRESENCE)
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        raise SystemExit(f'not found in {OUTPUT_DIR}: {", ".join(missing)} -- run stages 1-5 first')

    fg_names = json.loads(FG_COLUMNS.read_text())
    status = pd.read_csv(MOLECULE_STATUS, usecols=['cid', 'status'], dtype=str)
    mol_patterns = pd.read_csv(MOLECULE_PATTERNS, dtype={'cid': str})
    smiles = read_smiles(args.fg_dir)

    table = (status.drop_duplicates('cid')
             .merge(smiles, on='cid', how='left')
             .merge(mol_patterns, on='cid', how='left')
             .merge(per_pattern(fg_names), on='pattern_index', how='left'))
    # Only clustered molecules have a cluster; a rejected_no_fg molecule keeps its
    # (all-zero) pattern but gets no cluster or model, whatever stage 3 did with it.
    not_clustered = table['status'] != 'clustered'
    table.loc[not_clustered, ['primary_cluster', 'clusters', 'models']] = pd.NA
    for col in ('pattern_index', 'n_fgs', 'primary_cluster'):
        table[col] = table[col].astype('Int64')
    table = table[COLUMNS]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    if args.parquet:
        table.to_parquet(args.output.with_suffix('.parquet'), index=False)

    clustered = table[~not_clustered]
    print(f'Wrote {len(table):,} molecules to {args.output}')
    print('  ' + ', '.join(f'{s} {n:,}' for s, n in table['status'].value_counts().items()))
    problems = {
        'no SMILES': int(table['smiles'].isna().sum()),
        'clustered but no cluster': int(clustered['clusters'].isna().sum()),
        'clustered but no model': int((clustered['models'].fillna('') == '').sum()),
    }
    for what, n in problems.items():
        if n:
            print(f'  WARNING: {n:,} molecules with {what}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
