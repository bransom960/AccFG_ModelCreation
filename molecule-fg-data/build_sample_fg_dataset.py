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


def load_smiles_records(dataset_path: Path):
    if dataset_path.suffix.lower() == '.json':
        payload = json.loads(dataset_path.read_text())
        if isinstance(payload, list):
            return [(idx, str(s).strip()) for idx, s in enumerate(payload) if str(s).strip()]
        if isinstance(payload, dict):
            if 'smiles' in payload and isinstance(payload['smiles'], list):
                records = payload['smiles']
                return [(idx, str(s).strip()) for idx, s in enumerate(records) if str(s).strip()]
            if 'molecules' in payload and isinstance(payload['molecules'], list):
                records = payload['molecules']
                return [(idx, str(s).strip()) for idx, s in enumerate(records) if str(s).strip()]
        raise ValueError(f'Unsupported JSON structure in {dataset_path}')

    dataset = pd.read_csv(dataset_path)
    if 'cid' in dataset.columns:
        return [(int(row['cid']), str(row['smiles']).strip()) for _, row in dataset.iterrows() if str(row['smiles']).strip()]
    return [(idx, str(row['smiles']).strip()) for idx, row in dataset.iterrows() if str(row['smiles']).strip()]


def load_smiles_list(dataset_path: Path):
    return [smiles for _, smiles in load_smiles_records(dataset_path)]


def main():
    source_path = SMILES_JSON if SMILES_JSON.exists() else SAMPLE_DATASET
    afg = AccFG(print_load_info=False, lite=False)
    smiles_records = load_smiles_records(source_path)
    smiles_list = [smiles for _, smiles in smiles_records]

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
