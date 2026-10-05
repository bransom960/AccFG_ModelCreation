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
SMILES_JSON = ROOT / 'molecule-fg data' / 'smiles.json'
SAMPLE_DATASET = ROOT / 'molecule-fg data' / 'pubchem_like_sample_120.csv'
FG_OUTPUT = ROOT / 'molecule-fg data' / 'fg_presence.csv'


def load_smiles_list(dataset_path: Path):
    if dataset_path.suffix.lower() == '.json':
        payload = json.loads(dataset_path.read_text())
        if isinstance(payload, list):
            return [str(s).strip() for s in payload if str(s).strip()]
        if isinstance(payload, dict):
            if 'smiles' in payload and isinstance(payload['smiles'], list):
                return [str(s).strip() for s in payload['smiles'] if str(s).strip()]
            if 'molecules' in payload and isinstance(payload['molecules'], list):
                return [str(s).strip() for s in payload['molecules'] if str(s).strip()]
        raise ValueError(f'Unsupported JSON structure in {dataset_path}')

    dataset = pd.read_csv(dataset_path)
    return dataset['smiles'].astype(str).tolist()


def main():
    source_path = SMILES_JSON if SMILES_JSON.exists() else SAMPLE_DATASET
    afg = AccFG(print_load_info=False, lite=False)
    smiles_list = load_smiles_list(source_path)

    patterns = pattern_count_dictionary(afg, smiles_list, canonical=True)
    pattern_map = {pattern: idx for idx, pattern in enumerate(patterns.keys())}
    rows = []
    for smiles in smiles_list:
        vector = fg_presence_vector(afg, smiles, canonical=True)
        row = {'Molecule': smiles}
        row.update({fg_name: int(vector.get(fg_name, 0)) for fg_name in afg.dict_fgs.keys()})
        row['pattern_index'] = pattern_map[fg_pattern_string(afg, smiles, canonical=True)]
        rows.append(row)

    columns = ['Molecule'] + list(afg.dict_fgs.keys()) + ['pattern_index']
    df = pd.DataFrame(rows, columns=columns).fillna(0)
    df.to_csv(FG_OUTPUT, index=False)

    print(f'Wrote {len(df)} rows to {FG_OUTPUT}')
    print(f'Columns: {len(df.columns)}')
    print(df.head(3).to_string(index=False))


if __name__ == '__main__':
    main()
