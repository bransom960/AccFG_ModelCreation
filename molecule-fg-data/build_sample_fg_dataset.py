from pathlib import Path
import json
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from accfg import AccFG
from accfg.spreadsheet import fg_presence_vector
from patterns import fg_pattern_string, pattern_count_dictionary

ROOT = PROJECT_ROOT
DATA_DIR = ROOT / 'molecule-fg-data'
OUTPUT_DIR = DATA_DIR / 'csv_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

SMILES_JSON = DATA_DIR / 'smiles.json'
SAMPLE_DATASET = DATA_DIR / 'pubchem_like_sample_120.csv'
FG_OUTPUT = OUTPUT_DIR / 'fg_presence.csv'
REJECTED_OUTPUT = OUTPUT_DIR / 'rejected_molecules.csv'


def _json_items(payload, dataset_path: Path) -> list:
    """The list of molecule entries in smiles.json: the top-level list, or the list under a
    'smiles' or 'molecules' key."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ('smiles', 'molecules'):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f'Unsupported JSON structure in {dataset_path}')


def is_salt_or_mixture(smiles: str) -> bool:
    """A '.' in SMILES only ever separates disconnected components: a salt or a mixture."""
    return '.' in smiles


def _read_records(dataset_path: Path) -> tuple[list, bool]:
    """Return ([(cid, smiles), ...], ids_are_positions) for every molecule in the file.

    Source IDs are kept when the input carries them: smiles.json entries that are objects
    with 'cid' and 'smiles' keys, or a CSV with a 'cid' column. For a plain list of SMILES
    strings (or a CSV without 'cid') the only ID available is the entry's position in the
    file, and ids_are_positions is True.
    """
    if dataset_path.suffix.lower() == '.json':
        items = _json_items(json.loads(dataset_path.read_text()), dataset_path)
        records, positional = [], False
        for idx, item in enumerate(items):
            if isinstance(item, dict):
                if 'cid' not in item:
                    raise ValueError(f'{dataset_path}: entry {idx} has no "cid" key')
                cid, smiles = item['cid'], str(item.get('smiles', '')).strip()
            else:
                cid, smiles, positional = idx, str(item).strip(), True
            if smiles:
                records.append((int(cid), smiles))
        return records, positional

    dataset = pd.read_csv(dataset_path)
    if 'cid' in dataset.columns:
        return [(int(row['cid']), str(row['smiles']).strip()) for _, row in dataset.iterrows()
                if str(row['smiles']).strip()], False
    return [(idx, str(row['smiles']).strip()) for idx, row in dataset.iterrows()
            if str(row['smiles']).strip()], True


def load_smiles_records(dataset_path: Path) -> tuple[list, bool]:
    """Like _read_records, without salts and mixtures. Later stages read stage 1's output,
    so a molecule dropped here is dropped everywhere."""
    records, ids_are_positions = _read_records(dataset_path)
    return [r for r in records if not is_salt_or_mixture(r[1])], ids_are_positions


def load_rejected_records(dataset_path: Path) -> list:
    """[(cid, smiles, reason), ...] for the molecules load_smiles_records drops."""
    records, _ = _read_records(dataset_path)
    return [(cid, smiles, 'salt_or_mixture') for cid, smiles in records
            if is_salt_or_mixture(smiles)]


def load_smiles_list(dataset_path: Path):
    """The SMILES in load_smiles_records order."""
    return [smiles for _, smiles in load_smiles_records(dataset_path)[0]]


def main():
    source_path = SMILES_JSON if SMILES_JSON.exists() else SAMPLE_DATASET
    afg = AccFG(print_load_info=False, lite=False)
    smiles_records, ids_are_positions = load_smiles_records(source_path)
    if ids_are_positions:
        print(f'WARNING: {source_path.name} has no molecule IDs, so the cid column holds each '
              f"molecule's position in the file, not a source (e.g. PubChem) CID. Give each "
              f'entry as {{"cid": ..., "smiles": ...}} to keep the source IDs.')
    smiles_list = [smiles for _, smiles in smiles_records]
    rejected = pd.DataFrame(load_rejected_records(source_path), columns=['cid', 'Molecule', 'reason'])
    rejected.to_csv(REJECTED_OUTPUT, index=False)
    print(f'Dropped {len(rejected)} salts and mixtures (SMILES containing "."); listed in '
          f'{REJECTED_OUTPUT}')

    patterns = pattern_count_dictionary(afg, smiles_list, canonical=True)
    pattern_map = {pattern: idx for idx, pattern in enumerate(patterns.keys())}
    rows = []
    for cid, smiles in smiles_records:
        vector = fg_presence_vector(afg, smiles, canonical=True)
        row = {'cid': int(cid), 'Molecule': smiles}
        row.update({fg_name: int(vector.get(fg_name, 0)) for fg_name in afg.dict_fgs.keys()})
        row['pattern_index'] = pattern_map[fg_pattern_string(afg, smiles, canonical=True)]
        rows.append(row)

    columns = ['cid', 'Molecule'] + list(afg.dict_fgs.keys()) + ['pattern_index']
    df = pd.DataFrame(rows, columns=columns).fillna(0)
    df.to_csv(FG_OUTPUT, index=False)

    print(f'Wrote {len(df)} rows to {FG_OUTPUT}')
    print(f'Columns: {len(df.columns)}')
    print(df.head(3).to_string(index=False))


if __name__ == '__main__':
    main()
