from pathlib import Path
import json
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from accfg import AccFG
from patterns import pattern_count_dataframe

ROOT = PROJECT_ROOT
DATA_DIR = ROOT / 'molecule-fg-data'
OUTPUT_DIR = DATA_DIR / 'csv_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

SMILES_JSON = DATA_DIR / 'smiles.json'
SAMPLE_DATASET = DATA_DIR / 'pubchem_like_sample_120.csv'
PATTERN_OUTPUT = OUTPUT_DIR / 'pubchem_like_pattern_counts.csv'


def load_smiles_list(dataset_path: Path):
    if dataset_path.suffix.lower() == '.json':
        payload = json.loads(dataset_path.read_text())
        if isinstance(payload, list):
            return [str(s).strip() for s in payload if str(s).strip()]
        if isinstance(payload, dict) and 'smiles' in payload and isinstance(payload['smiles'], list):
            return [str(s).strip() for s in payload['smiles'] if str(s).strip()]
        raise ValueError(f'Unsupported JSON structure in {dataset_path}')

    dataset = pd.read_csv(dataset_path)
    return dataset['smiles'].astype(str).tolist()


def main():
    source_path = SMILES_JSON if SMILES_JSON.exists() else SAMPLE_DATASET
    afg = AccFG(print_load_info=False, lite=False)
    smiles_list = load_smiles_list(source_path)

    counts_df = pattern_count_dataframe(afg, smiles_list, canonical=True)
    counts_df.to_csv(PATTERN_OUTPUT, index=False)

    print(f'Wrote {len(counts_df)} unique patterns to {PATTERN_OUTPUT}')
    print(counts_df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
