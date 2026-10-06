"""Stage 2: compress stage 1's FG presence matrix into unique FG patterns.

Reads the molecule-by-FG matrix that stage 1 already computed -- AccFG is not run again --
and writes to csv_outputs/:

  pubchem_like_pattern_counts.csv   pattern_index, pattern, count   one row per unique pattern
  molecule_patterns.csv             cid, pattern_index              one row per molecule
  fg_columns.json                   FG names, in pattern-bit order

Input, one of:
  * csv_outputs/fg_presence.csv from build_sample_fg_dataset.py (the default), or
  * --fg-dir: the Parquet output directory of stage1/fg_matrix.py (columns cid, Molecule,
    one 0/1 column per FG, plus _columns.json), for large runs.

A pattern is one '0'/'1' character per FG in fg_columns.json order. pattern_index numbers
the distinct patterns in sorted-string order, the same rule the earlier AccFG-based stage 2
used. Stages 3 and 4 read these files instead of re-running AccFG on smiles.json, so every
stage works from the same molecules and the same pattern_index.

    python3 molecule-fg-data/build_pattern_count_dictionary.py
    python3 molecule-fg-data/build_pattern_count_dictionary.py --fg-dir /data/fg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'molecule-fg-data'
OUTPUT_DIR = DATA_DIR / 'csv_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

FG_PRESENCE = OUTPUT_DIR / 'fg_presence.csv'
PATTERN_OUTPUT = OUTPUT_DIR / 'pubchem_like_pattern_counts.csv'
MOLECULE_PATTERNS = OUTPUT_DIR / 'molecule_patterns.csv'
FG_COLUMNS = OUTPUT_DIR / 'fg_columns.json'

KEY_COLUMNS = ('cid', 'Molecule', 'pattern_index')
_ZERO = np.uint8(ord('0'))


def iter_fg_presence_csv(path: Path, chunksize: int = 50000):
    """Yield (fg_names, cids, X uint8) chunks from stage 1's fg_presence.csv."""
    header = list(pd.read_csv(path, nrows=0).columns)
    if 'cid' not in header:
        raise ValueError(f'{path} has no cid column; re-run build_sample_fg_dataset.py')
    fg_names = [c for c in header if c not in KEY_COLUMNS]
    dtypes = {name: np.uint8 for name in fg_names}
    dtypes['cid'] = str
    for chunk in pd.read_csv(path, usecols=['cid'] + fg_names, dtype=dtypes, chunksize=chunksize):
        yield fg_names, chunk['cid'].tolist(), chunk[fg_names].to_numpy(dtype=np.uint8)


def iter_fg_matrix_parquet(fg_dir: Path, batch_rows: int = 65536):
    """Yield (fg_names, cids, X uint8) batches from stage1/fg_matrix.py's Parquet output."""
    import pyarrow.dataset as ds

    fg_names = json.loads((fg_dir / '_columns.json').read_text())
    dataset = ds.dataset(fg_dir, format='parquet')
    if 'cid' not in dataset.schema.names:
        raise ValueError(f'{fg_dir} has no cid column; it was written by an older fg_matrix.py')
    for batch in dataset.scanner(columns=['cid'] + fg_names, batch_size=batch_rows).to_batches():
        if batch.num_rows == 0:
            continue
        X = np.stack([batch.column(i).to_numpy(zero_copy_only=False)
                      for i in range(1, batch.num_columns)], axis=1).astype(np.uint8)
        yield fg_names, [str(c) for c in batch.column(0).to_pylist()], X


def compress(chunks) -> tuple[list, pd.DataFrame, pd.DataFrame]:
    """Return (fg_names, pattern counts, molecule -> pattern_index) from FG matrix chunks."""
    fg_names = None
    key_to_tmp: dict = {}
    tmp_patterns: list = []
    tmp_counts: list = []
    cids: list = []
    tmp_idx: list = []
    for names, chunk_cids, X in chunks:
        if fg_names is None:
            fg_names = names
        elif names != fg_names:
            raise ValueError('FG columns differ between input chunks')
        rows = (X + _ZERO).astype(np.uint8)
        idx = np.empty(len(rows), dtype=np.int64)
        for i, row in enumerate(rows):
            key = row.tobytes()
            j = key_to_tmp.get(key)
            if j is None:
                j = key_to_tmp[key] = len(tmp_patterns)
                tmp_patterns.append(key.decode('ascii'))
                tmp_counts.append(0)
            tmp_counts[j] += 1
            idx[i] = j
        cids.extend(chunk_cids)
        tmp_idx.append(idx)
    if fg_names is None:
        raise ValueError('stage 1 output holds no molecules')

    order = sorted(range(len(tmp_patterns)), key=tmp_patterns.__getitem__)
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))
    patterns = pd.DataFrame({
        'pattern_index': np.arange(len(order)),
        'pattern': [tmp_patterns[j] for j in order],
        'count': np.asarray(tmp_counts, dtype=np.int64)[order],
    })
    molecules = pd.DataFrame({'cid': cids, 'pattern_index': remap[np.concatenate(tmp_idx)]})
    return fg_names, patterns, molecules


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fg-dir', type=Path, default=None,
                        help='read stage1/fg_matrix.py Parquet output instead of fg_presence.csv')
    args = parser.parse_args(argv)

    if args.fg_dir is not None:
        source, chunks = args.fg_dir, iter_fg_matrix_parquet(args.fg_dir)
    else:
        source, chunks = FG_PRESENCE, iter_fg_presence_csv(FG_PRESENCE)
    fg_names, counts_df, molecules = compress(chunks)

    counts_df.to_csv(PATTERN_OUTPUT, index=False)
    molecules.to_csv(MOLECULE_PATTERNS, index=False)
    FG_COLUMNS.write_text(json.dumps(fg_names, indent=1))

    print(f'Read {len(molecules)} molecules and {len(fg_names)} FG columns from {source}')
    print(f'Wrote {len(counts_df)} unique patterns to {PATTERN_OUTPUT}')
    print(f'Wrote molecule -> pattern map to {MOLECULE_PATTERNS}')
    print(counts_df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
