from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

try:
    from accfg.main import AccFG
    from accfg.spreadsheet import fg_presence_vector, canonical_smiles
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from accfg.main import AccFG
    from accfg.spreadsheet import fg_presence_vector, canonical_smiles


def fg_pattern_string(afg: AccFG, smiles: str, canonical: bool = True) -> str:
    """Return the full FG pattern as a binary string of 0/1 values."""
    if canonical:
        smiles = canonical_smiles(smiles)

    vector = fg_presence_vector(afg, smiles, canonical=False)
    return ''.join('1' if vector.get(fg_name, 0) else '0' for fg_name in afg.dict_fgs.keys())


def fg_pattern_bitmask(afg: AccFG, smiles: str, canonical: bool = True) -> int:
    """Compatibility helper: convert the binary string pattern to an integer bitmask."""
    return int(fg_pattern_string(afg, smiles, canonical=canonical), 2)


def fg_presence_rows(afg: AccFG, smiles_list: Iterable[str], canonical: bool = True, cid_values: Iterable[int] | None = None) -> List[dict]:
    """Return one row per molecule with the FG 0/1 vector, pattern index, and exact cid."""
    pattern_counts = pattern_count_dictionary(afg, smiles_list, canonical=canonical)
    pattern_index = {pattern: idx for idx, pattern in enumerate(pattern_counts.keys())}

    rows = []
    cid_source = list(cid_values) if cid_values is not None else list(range(len(list(smiles_list))))
    for idx, smiles in enumerate(smiles_list):
        canon_smi = canonical_smiles(smiles) if canonical else smiles
        vector = fg_presence_vector(afg, canon_smi, canonical=False)
        pattern = fg_pattern_string(afg, canon_smi, canonical=False)
        cid = cid_source[idx] if idx < len(cid_source) else idx
        row = {'cid': cid, 'Molecule': canon_smi}
        row.update(vector)
        row['pattern_index'] = pattern_index[pattern]
        rows.append(row)
    return rows


def pattern_count_dictionary(afg: AccFG, smiles_list: Iterable[str], canonical: bool = True) -> dict:
    """Count how many molecules share each FG pattern string."""
    counts = {}
    for smiles in smiles_list:
        pattern = fg_pattern_string(afg, smiles, canonical=canonical)
        counts[pattern] = counts.get(pattern, 0) + 1
    return dict(sorted(counts.items()))


def pattern_count_dataframe(afg: AccFG, smiles_list: Iterable[str], canonical: bool = True) -> pd.DataFrame:
    counts = pattern_count_dictionary(afg, smiles_list, canonical=canonical)
    rows = []
    for idx, (pattern, count) in enumerate(counts.items()):
        rows.append({'pattern_index': idx, 'pattern': pattern, 'count': int(count)})
    df = pd.DataFrame(rows)
    return df.sort_values('pattern_index').reset_index(drop=True)


def _pattern_vector(pattern: str) -> np.ndarray:
    pattern = str(pattern).strip()
    if not pattern:
        return np.zeros(0, dtype=np.int8)
    return np.fromiter((int(ch) for ch in pattern), dtype=np.int8)


def _hamming_distance(pattern_a: str, pattern_b: str) -> int:
    a = str(pattern_a)
    b = str(pattern_b)
    if len(a) != len(b):
        max_len = max(len(a), len(b))
        a = a.ljust(max_len, '0')
        b = b.ljust(max_len, '0')
    return sum(1 for x, y in zip(a, b) if x != y)


def _weighted_majority_mode(patterns: list[str], weights: np.ndarray) -> str:
    if not patterns:
        return ''
    length = len(patterns[0])
    mode_bits = []
    for idx in range(length):
        weighted_sum = sum(weights[i] for i, pattern in enumerate(patterns) if pattern[idx] == '1')
        total_weight = sum(weights)
        mode_bits.append('1' if total_weight > 0 and weighted_sum >= (total_weight / 2.0) else '0')
    return ''.join(mode_bits)


def cluster_pattern_counts(pattern_df: pd.DataFrame, k: int, max_iters: int = 25, seed: int = 0) -> pd.DataFrame:
    """Cluster binary FG patterns using weighted K-modes with Hamming distance.

    Patterns that are all zeros are removed because they cover nothing and are not useful for
    model-domain clustering. Each pattern is weighted by its molecule count so frequent FG
    bundles influence cluster centroids and assignments.
    """
    if 'pattern' not in pattern_df.columns:
        raise ValueError("pattern_df must contain a 'pattern' column")
    if 'count' not in pattern_df.columns:
        raise ValueError("pattern_df must contain a 'count' column")

    df = pattern_df.copy().reset_index(drop=True)
    df = df[df['pattern'].astype(str).str.strip().apply(lambda p: any(ch == '1' for ch in p))].copy()

    if len(df) == 0:
        return df.copy().assign(cluster_id=pd.Series(dtype='int64'), cluster_weight=pd.Series(dtype='float64'))

    n_patterns = len(df)
    k = max(1, min(int(k), n_patterns))

    patterns = df['pattern'].astype(str).tolist()
    counts = df['count'].astype(float).to_numpy()

    order = np.argsort(-counts)
    patterns = [patterns[i] for i in order]
    counts = counts[order]
    df = df.iloc[order].reset_index(drop=True)

    center_indices = list(range(min(k, n_patterns)))
    centers = [patterns[idx] for idx in center_indices]

    for _ in range(max_iters):
        assignments = []
        for pattern in patterns:
            dists = [_hamming_distance(pattern, center) for center in centers]
            assignments.append(int(np.argmin(dists)))

        new_centers = []
        for cluster_id in range(k):
            member_idx = [idx for idx, assignment in enumerate(assignments) if assignment == cluster_id]
            if not member_idx:
                new_centers.append(centers[cluster_id])
                continue
            member_patterns = [patterns[idx] for idx in member_idx]
            member_weights = counts[member_idx]
            new_centers.append(_weighted_majority_mode(member_patterns, member_weights))

        if new_centers == centers:
            break
        centers = new_centers

    final_assignments = []
    for pattern in patterns:
        dists = [_hamming_distance(pattern, center) for center in centers]
        final_assignments.append(int(np.argmin(dists)))

    out = df.copy()
    out['cluster_id'] = final_assignments
    out['cluster_representative'] = out['cluster_id'].map({cid: centers[cid] for cid in range(len(centers))})
    cluster_weights = {}
    for cid in range(len(centers)):
        cluster_weights[cid] = int(counts[np.array(final_assignments) == cid].sum())
    out['cluster_weight'] = out['cluster_id'].map(cluster_weights)
    return out.sort_values(['cluster_id', 'pattern_index']).reset_index(drop=True)


def save_pattern_count_csv(afg: AccFG, smiles_list: Iterable[str], output_csv: str, canonical: bool = True):
    df = pattern_count_dataframe(afg, smiles_list, canonical=canonical)
    df.to_csv(output_csv, index=False)
    return df
