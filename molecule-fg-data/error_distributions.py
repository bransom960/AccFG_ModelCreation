"""Build the molecule-by-model ground-truth error table.

Every model has an in-domain and an out-of-domain normal distribution:

- in-domain:     mean = avg_error_in,                    std = std_error_in
- out-of-domain: mean = avg_error_in + 3 * std_error_in, std = std_error_out (defaults to std_error_in)

Works per unique FG pattern: fg_presence.csv is streamed twice, once to collect the unique
FG vectors and once to write one error per (molecule, model), so the input never has to
fit in memory and nearest-neighbour distances are computed once per pattern.

- in-domain molecules sample from the in-domain distribution
- out-of-domain molecules find their nearest in-domain FG vector (Hamming distance) and
  sample from a normal whose mean is shifted down from the out-of-domain mean by their
  FG similarity
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = DEFAULT_ROOT / 'molecule-fg-data'
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / 'csv_outputs'

# Columns of fg_presence.csv that are not FG flags.
METADATA_COLUMNS = {'cid', 'Molecule', 'molecule', 'smiles', 'SMILES', 'pattern', 'pattern_index'}

OUT_OF_DOMAIN_SHIFT_STDS = 3.0


def load_model_specs(model_specs_path: str | Path, default_std: float) -> pd.DataFrame:
    specs = pd.read_csv(model_specs_path)
    if not {'model_name', 'avg_error_in'}.issubset(specs.columns):
        raise ValueError('model_specs.csv must include model_name and avg_error_in columns')
    if specs['model_name'].duplicated().any():
        raise ValueError('model_specs.csv has duplicate model_name rows')
    specs = specs.copy()
    specs['model_name'] = specs['model_name'].astype(str).str.strip()
    if 'std_error_in' not in specs.columns:
        specs['std_error_in'] = default_std
    specs['std_error_in'] = specs['std_error_in'].fillna(default_std)
    if 'std_error_out' not in specs.columns:
        specs['std_error_out'] = specs['std_error_in']
    specs['std_error_out'] = specs['std_error_out'].fillna(specs['std_error_in'])
    if specs['avg_error_in'].isna().any() or (specs[['std_error_in', 'std_error_out']] <= 0).any().any():
        raise ValueError('model_specs.csv needs an avg_error_in for every model and positive std values')
    specs['avg_error_out'] = specs['avg_error_in'] + OUT_OF_DOMAIN_SHIFT_STDS * specs['std_error_in']
    return specs


def load_patterns(fg_presence_path: str | Path, chunksize: int):
    """Read the unique FG vectors and their molecule counts from fg_presence.csv.

    Streams the file so it never has to fit in memory. A pattern is identified by
    pattern_index; FG values > 0 count as present.
    """
    header = pd.read_csv(fg_presence_path, nrows=0).columns
    if not {'cid', 'pattern_index'}.issubset(header):
        raise ValueError('fg_presence.csv must include cid and pattern_index columns')
    fg_columns = [col for col in header if col not in METADATA_COLUMNS]
    if not fg_columns:
        raise ValueError('fg_presence.csv has no FG columns')

    bits_by_index: dict[int, np.ndarray] = {}
    counts_by_index: dict[int, int] = {}
    reader = pd.read_csv(
        fg_presence_path,
        usecols=['pattern_index'] + fg_columns,
        dtype={col: np.float32 for col in fg_columns},
        chunksize=chunksize,
    )
    for chunk in reader:
        indices, counts = np.unique(chunk['pattern_index'].to_numpy(dtype=np.int64), return_counts=True)
        for pattern_index, count in zip(indices.tolist(), counts.tolist()):
            counts_by_index[pattern_index] = counts_by_index.get(pattern_index, 0) + count

        first_rows = chunk.drop_duplicates('pattern_index')
        first_bits = (first_rows[fg_columns].to_numpy() > 0).astype(np.uint8)
        for pattern_index, bits in zip(first_rows['pattern_index'].astype(np.int64).tolist(), first_bits):
            known = bits_by_index.get(pattern_index)
            if known is None:
                bits_by_index[pattern_index] = bits
            elif not np.array_equal(known, bits):
                raise ValueError(f'pattern_index {pattern_index} appears with two different FG vectors')

    pattern_ids = np.array(sorted(bits_by_index), dtype=np.int64)
    bits = np.stack([bits_by_index[i] for i in pattern_ids.tolist()])
    counts = np.array([counts_by_index[i] for i in pattern_ids.tolist()], dtype=np.int64)
    return pattern_ids, bits, counts, fg_columns


def load_domains(pattern_model_map_path: str | Path, pattern_clusters: pd.DataFrame, model_names: list[str]):
    """Return {model_name: set of in-domain pattern_index}."""
    pattern_model_map = pd.read_csv(pattern_model_map_path)
    if 'model_name' not in pattern_model_map.columns:
        raise ValueError('pattern_cluster_model_map.csv must include a model_name column')
    pattern_model_map['model_name'] = pattern_model_map['model_name'].astype(str).str.strip()

    if 'pattern_index' not in pattern_model_map.columns:
        # Cluster-level map: expand every cluster to all of its member patterns.
        if 'cluster_id' not in pattern_model_map.columns or not {'cluster_id', 'pattern_index'}.issubset(pattern_clusters.columns):
            raise ValueError('pattern_cluster_model_map.csv needs pattern_index, or cluster_id plus pattern_clusters.csv')
        pattern_model_map = pattern_model_map.merge(pattern_clusters[['cluster_id', 'pattern_index']], on='cluster_id')

    domains = {
        model_name: set(group['pattern_index'].astype(np.int64).tolist())
        for model_name, group in pattern_model_map.groupby('model_name')
    }
    unknown = sorted(set(domains) - set(model_names))
    if unknown:
        print(f'WARNING: models in the pattern-model map but not in model_specs.csv (ignored): {unknown}')
    for model_name in model_names:
        if not domains.get(model_name):
            print(f'WARNING: model {model_name!r} has no in-domain patterns; every molecule is out-of-domain for it')
    return {model_name: domains.get(model_name, set()) for model_name in model_names}


def nearest_in_domain(bits: np.ndarray, in_mask: np.ndarray, block_out: int = 1024, block_in: int = 16384):
    """Minimum Hamming distance from every out-of-domain pattern to the in-domain patterns.

    Uses d(a, b) = |a| + |b| - 2 a.b on float32 blocks (exact for 0/1 vectors this size).
    Returns (distance, nearest pattern position) arrays over all patterns; in-domain
    patterns get distance 0 and themselves, and every pattern gets inf / -1 when the
    model has no in-domain patterns.
    """
    n_patterns = len(bits)
    distance = np.full(n_patterns, np.inf)
    nearest = np.full(n_patterns, -1, dtype=np.int64)
    ref_idx = np.flatnonzero(in_mask)
    distance[ref_idx] = 0.0
    nearest[ref_idx] = ref_idx
    out_idx = np.flatnonzero(~in_mask)
    if ref_idx.size == 0 or out_idx.size == 0:
        return distance, nearest

    ref = bits[ref_idx].astype(np.float32)
    ref_norm = ref.sum(axis=1)
    for start in range(0, out_idx.size, block_out):
        rows = out_idx[start:start + block_out]
        x = bits[rows].astype(np.float32)
        x_norm = x.sum(axis=1)
        best = np.full(rows.size, np.inf)
        best_pos = np.full(rows.size, -1, dtype=np.int64)
        for ref_start in range(0, ref_idx.size, block_in):
            ref_block = ref[ref_start:ref_start + block_in]
            d = x_norm[:, None] + ref_norm[None, ref_start:ref_start + block_in] - 2.0 * (x @ ref_block.T)
            j = d.argmin(axis=1)
            dj = d[np.arange(rows.size), j]
            better = dj < best
            best[better] = dj[better]
            best_pos[better] = ref_idx[ref_start + j[better]]
        distance[rows] = np.rint(best)
        nearest[rows] = best_pos
    return distance, nearest


def build_pattern_samples(
    specs: pd.DataFrame,
    domains: dict[str, set],
    pattern_ids: np.ndarray,
    bits: np.ndarray,
    counts: np.ndarray,
    cluster_of_pattern: dict[int, int],
) -> pd.DataFrame:
    """The normal each (pattern, model) pair samples its molecules' errors from."""
    n_fgs = bits.sum(axis=1)
    frames = []
    for spec in specs.itertuples(index=False):
        avg_in, std_in = spec.avg_error_in, spec.std_error_in
        avg_out, std_out = spec.avg_error_out, spec.std_error_out

        in_mask = np.isin(pattern_ids, np.fromiter(domains[spec.model_name], dtype=np.int64))
        distance, nearest = nearest_in_domain(bits, in_mask)

        similarity = np.clip(1.0 - distance / bits.shape[1], 0.0, 1.0)
        # Same as the per-row version: no in-domain reference, or similarity 0, counts as 1.0.
        similarity[~np.isfinite(distance) | (similarity == 0.0)] = 1.0
        nearest_pattern = np.where(nearest >= 0, pattern_ids[np.maximum(nearest, 0)], -1)

        frame = pd.DataFrame({
            'pattern_index': pattern_ids,
            'model_name': spec.model_name,
            'domain_flag': np.where(in_mask, 'in', 'out'),
            'n_molecules': counts,
            'n_fgs': n_fgs,
            'nearest_distance': np.where(in_mask, np.nan, distance),
            'nearest_reference_pattern': np.where(in_mask, -1, nearest_pattern),
            'fg_similarity_rank': np.where(in_mask, np.nan, similarity),
            'error_mean': np.where(in_mask, avg_in, avg_out),
            'error_std': np.where(in_mask, std_in, std_out),
            'sample_mean': np.where(in_mask, avg_in, avg_out - similarity * 2.0 * std_out),
            'sample_std': np.where(in_mask, std_in, std_out * 0.75),
        })
        frame['nearest_reference_cluster'] = frame['nearest_reference_pattern'].map(cluster_of_pattern)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_model_error_table(
    fg_presence_path: str | Path = DEFAULT_OUTPUT_DIR / 'fg_presence.csv',
    model_specs_path: str | Path = DEFAULT_OUTPUT_DIR / 'model_specs.csv',
    pattern_clusters_path: str | Path = DEFAULT_OUTPUT_DIR / 'pattern_clusters.csv',
    pattern_model_map_path: str | Path = DEFAULT_OUTPUT_DIR / 'pattern_cluster_model_map.csv',
    output_path: str | Path = DEFAULT_OUTPUT_DIR / 'molecule_model_error_table.csv',
    pattern_output_path: str | Path | None = DEFAULT_OUTPUT_DIR / 'pattern_model_error_table.csv',
    seed: int = 0,
    default_std: float = 0.05,
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """Write one sampled error per (molecule, model) and return the per-pattern table."""
    specs = load_model_specs(model_specs_path, default_std)
    model_names = specs['model_name'].tolist()
    pattern_clusters = pd.read_csv(pattern_clusters_path) if Path(pattern_clusters_path).exists() else pd.DataFrame()
    domains = load_domains(pattern_model_map_path, pattern_clusters, model_names)

    cluster_of_pattern = {}
    if {'pattern_index', 'cluster_id'}.issubset(pattern_clusters.columns):
        first = pattern_clusters.drop_duplicates('pattern_index')
        cluster_of_pattern = dict(zip(first['pattern_index'].astype(np.int64), first['cluster_id'].astype(np.int64)))

    pattern_ids, bits, counts, fg_columns = load_patterns(fg_presence_path, chunksize)
    print(f'{counts.sum()} molecules, {len(pattern_ids)} unique FG patterns, {len(fg_columns)} FGs')

    molecule_rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(len(specs))]

    patterns = build_pattern_samples(specs, domains, pattern_ids, bits, counts, cluster_of_pattern)
    if pattern_output_path is not None:
        Path(pattern_output_path).parent.mkdir(parents=True, exist_ok=True)
        patterns.to_csv(pattern_output_path, index=False)

    # Per model: arrays aligned with pattern_ids, for fast lookup while streaming molecules.
    per_model = {
        model_name: group.set_index('pattern_index').loc[pattern_ids]
        for model_name, group in patterns.groupby('model_name', sort=False)
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        'cid', 'pattern_index', 'model_name', 'domain_flag', 'nearest_distance',
        'nearest_reference_cluster', 'fg_similarity_rank', 'error_mean', 'error_std', 'sampled_error',
    ]
    pd.DataFrame(columns=columns).to_csv(output_path, index=False)

    reader = pd.read_csv(fg_presence_path, usecols=['cid', 'pattern_index'], dtype={'cid': str}, chunksize=chunksize)
    for chunk in reader:
        positions = np.searchsorted(pattern_ids, chunk['pattern_index'].to_numpy(dtype=np.int64))
        for model_name, rng in zip(model_names, molecule_rngs):
            p = per_model[model_name]
            error = (p['sample_mean'].to_numpy()[positions]
                     + p['sample_std'].to_numpy()[positions] * rng.standard_normal(len(chunk)))
            pd.DataFrame({
                'cid': chunk['cid'].to_numpy(),
                'pattern_index': chunk['pattern_index'].to_numpy(),
                'model_name': model_name,
                'domain_flag': p['domain_flag'].to_numpy()[positions],
                'nearest_distance': p['nearest_distance'].to_numpy()[positions],
                'nearest_reference_cluster': p['nearest_reference_cluster'].to_numpy()[positions],
                'fg_similarity_rank': p['fg_similarity_rank'].to_numpy()[positions],
                'error_mean': p['error_mean'].to_numpy()[positions],
                'error_std': p['error_std'].to_numpy()[positions],
                'sampled_error': error,
            }, columns=columns).to_csv(output_path, mode='a', header=False, index=False)
    return patterns


def main():
    parser = argparse.ArgumentParser(description='Build a molecule-model error table from FG patterns and model coverage rules.')
    parser.add_argument('--fg-presence', type=str, default=str(DEFAULT_OUTPUT_DIR / 'fg_presence.csv'))
    parser.add_argument('--model-specs', type=str, default=str(DEFAULT_OUTPUT_DIR / 'model_specs.csv'))
    parser.add_argument('--pattern-clusters', type=str, default=str(DEFAULT_OUTPUT_DIR / 'pattern_clusters.csv'))
    parser.add_argument('--pattern-model-map', type=str, default=str(DEFAULT_OUTPUT_DIR / 'pattern_cluster_model_map.csv'))
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT_DIR / 'molecule_model_error_table.csv'))
    parser.add_argument('--pattern-output', type=str, default=str(DEFAULT_OUTPUT_DIR / 'pattern_model_error_table.csv'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--std', type=float, default=0.05, help='std used when model_specs.csv has no std_error_in')
    parser.add_argument('--chunksize', type=int, default=50_000)
    args = parser.parse_args()

    build_model_error_table(
        fg_presence_path=args.fg_presence,
        model_specs_path=args.model_specs,
        pattern_clusters_path=args.pattern_clusters,
        pattern_model_map_path=args.pattern_model_map,
        output_path=args.output,
        pattern_output_path=args.pattern_output,
        seed=args.seed,
        default_std=args.std,
        chunksize=args.chunksize,
    )
    print(f'Wrote molecule-model errors to {args.output}')
    print(f'Wrote pattern-model errors to {args.pattern_output}')


if __name__ == '__main__':
    main()
