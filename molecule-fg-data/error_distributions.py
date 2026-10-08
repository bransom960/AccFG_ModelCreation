from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = DEFAULT_ROOT / 'molecule-fg-data'
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / 'csv_outputs'


def _parse_member_cids(value):
    if pd.isna(value):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return [item.strip() for item in value.split(',') if item.strip()]
    return [str(value)]


def _hamming_distance_bits(a: str, b: str) -> int:
    if len(a) != len(b):
        length = max(len(a), len(b))
        a = a.ljust(length, '0')
        b = b.ljust(length, '0')
    return int(np.count_nonzero(np.fromiter((x != y for x, y in zip(a, b)), dtype=int)))


def _pattern_to_model_lookup(pattern_clusters: pd.DataFrame, pattern_model_map: pd.DataFrame, fg_presence: pd.DataFrame | None = None):
    assigned_patterns = {}
    if not pattern_model_map.empty:
        for model_name, group in pattern_model_map.groupby('model_name'):
            assigned_patterns[model_name] = set(group['pattern_index'].astype(int).tolist())

    fg_pattern_by_index = {}
    if fg_presence is not None and not fg_presence.empty:
        for _, row in fg_presence.iterrows():
            fg_pattern_by_index[int(row['pattern_index'])] = str(row.get('pattern', ''))

    pattern_lookup = {}
    if not pattern_clusters.empty:
        for _, row in pattern_clusters.iterrows():
            pattern_index = int(row['pattern_index'])
            cluster_id = int(row['cluster_id'])
            pattern_bits = str(row.get('pattern', fg_pattern_by_index.get(pattern_index, '')))
            pattern_lookup[pattern_index] = {
                'cluster_id': cluster_id,
                'pattern': pattern_bits,
                'cluster_representative': row.get('cluster_representative', ''),
            }

    return assigned_patterns, pattern_lookup


def build_model_error_table(
    fg_presence_path: str | Path = DEFAULT_OUTPUT_DIR / 'fg_presence.csv',
    model_specs_path: str | Path = DEFAULT_OUTPUT_DIR / 'model_specs.csv',
    pattern_clusters_path: str | Path = DEFAULT_OUTPUT_DIR / 'pattern_clusters.csv',
    pattern_model_map_path: str | Path = DEFAULT_OUTPUT_DIR / 'pattern_cluster_model_map.csv',
    output_path: str | Path | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Create a molecule-by-model ground-truth error table.

    Rules:
    - in-domain molecules use the model's in-domain normal distribution
    - out-of-domain molecules compute the nearest in-domain reference and use a rank-based
      shift of the out-of-domain normal distribution
    - nearest_reference_cluster and fg_similarity_rank are only populated for out-of-domain rows
    """
    rng = np.random.default_rng(seed)

    fg_presence = pd.read_csv(fg_presence_path)
    model_specs = pd.read_csv(model_specs_path)
    pattern_clusters = pd.read_csv(pattern_clusters_path) if Path(pattern_clusters_path).exists() else pd.DataFrame()
    pattern_model_map = pd.read_csv(pattern_model_map_path) if Path(pattern_model_map_path).exists() else pd.DataFrame()

    if not {'cid', 'pattern_index'}.issubset(fg_presence.columns):
        raise ValueError('fg_presence.csv must include cid and pattern_index columns')

    feature_columns = [
        col for col in fg_presence.columns
        if col not in {'cid', 'Molecule', 'pattern_index'}
    ]

    if 'avg_error_in' not in model_specs.columns:
        model_specs['avg_error_in'] = 0.0
    if 'std_error_in' not in model_specs.columns:
        model_specs['std_error_in'] = 0.05

    pattern_model_map = pattern_model_map.copy()
    if 'pattern_index' in pattern_model_map.columns and 'cluster_id' in pattern_model_map.columns:
        pattern_model_map['pattern_index'] = pattern_model_map['pattern_index'].astype(int)
        pattern_model_map['cluster_id'] = pattern_model_map['cluster_id'].astype(int)

    assigned_patterns, pattern_lookup = _pattern_to_model_lookup(pattern_clusters, pattern_model_map, fg_presence)
    model_rows = []

    for _, model_spec in model_specs.iterrows():
        model_name = str(model_spec['model_name'])
        avg_in = float(model_spec.get('avg_error_in', 0.0))
        std_in = float(model_spec.get('std_error_in', 0.05))
        std_in = max(std_in, 1e-6)
        avg_out = avg_in + 3.0 * std_in
        std_out = max(std_in * 1.5, 1e-6)

        in_domain_patterns = set(assigned_patterns.get(model_name, set()))

        for _, molecule in fg_presence.iterrows():
            cid = str(molecule['cid'])
            pattern_index = int(molecule['pattern_index'])
            pattern_bits = str(molecule.get('pattern', ''))
            if not pattern_bits and feature_columns:
                pattern_bits = ''.join(str(int(v)) for v in [molecule[col] for col in feature_columns])

            in_domain = pattern_index in in_domain_patterns
            if in_domain:
                sampled_error = float(rng.normal(avg_in, std_in))
                model_rows.append({
                    'cid': cid,
                    'pattern_index': pattern_index,
                    'model_name': model_name,
                    'domain_flag': 'in',
                    'nearest_reference_cluster': None,
                    'fg_similarity_rank': None,
                    'error_mean': avg_in,
                    'error_std': std_in,
                    'sampled_error': sampled_error,
                })
                continue

            reference_candidates = set(in_domain_patterns) if in_domain_patterns else set(pattern_lookup.keys())

            nearest_cluster = None
            nearest_distance = None
            for candidate_index, info in pattern_lookup.items():
                if candidate_index not in reference_candidates:
                    continue
                candidate_bits = str(info.get('pattern', ''))
                if not candidate_bits:
                    continue
                distance = _hamming_distance_bits(pattern_bits, candidate_bits)
                if nearest_distance is None or distance < nearest_distance:
                    nearest_distance = distance
                    nearest_cluster = int(info.get('cluster_id', -1))

            if nearest_cluster is None or nearest_distance is None:
                similarity_rank = 0.0
                if nearest_cluster is None:
                    nearest_cluster = int(next(iter(pattern_lookup.values()), {}).get('cluster_id', -1)) if pattern_lookup else None
            else:
                similarity_rank = max(0.0, min(1.0, 1.0 - (nearest_distance / max(len(pattern_bits), 1))))

            if nearest_cluster is not None and similarity_rank == 0.0 and pattern_lookup:
                similarity_rank = 1.0

            shifted_mean = avg_out - (similarity_rank * 2.0 * std_out)
            sampled_error = float(rng.normal(shifted_mean, std_out * 0.75))
            model_rows.append({
                'cid': cid,
                'pattern_index': pattern_index,
                'model_name': model_name,
                'domain_flag': 'out',
                'nearest_reference_cluster': nearest_cluster,
                'fg_similarity_rank': float(similarity_rank),
                'error_mean': avg_out,
                'error_std': std_out,
                'sampled_error': sampled_error,
            })

    table = pd.DataFrame(model_rows)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
    return table


def main():
    parser = argparse.ArgumentParser(description='Build a molecule-model error table from FG patterns and model coverage rules.')
    parser.add_argument('--fg-presence', type=str, default=str(DEFAULT_OUTPUT_DIR / 'fg_presence.csv'))
    parser.add_argument('--model-specs', type=str, default=str(DEFAULT_OUTPUT_DIR / 'model_specs.csv'))
    parser.add_argument('--pattern-clusters', type=str, default=str(DEFAULT_OUTPUT_DIR / 'pattern_clusters.csv'))
    parser.add_argument('--pattern-model-map', type=str, default=str(DEFAULT_OUTPUT_DIR / 'pattern_cluster_model_map.csv'))
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT_DIR / 'molecule_model_error_table.csv'))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    table = build_model_error_table(
        fg_presence_path=args.fg_presence,
        model_specs_path=args.model_specs,
        pattern_clusters_path=args.pattern_clusters,
        pattern_model_map_path=args.pattern_model_map,
        output_path=args.output,
        seed=args.seed,
    )
    print(f'Wrote {len(table)} molecule-model error rows to {args.output}')
    print(table.head(5).to_string(index=False))


if __name__ == '__main__':
    main()
