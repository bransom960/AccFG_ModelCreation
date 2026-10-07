"""Stage 3: overlapping Bernoulli-mixture clusters of the unique FG patterns.

Reads stage 2's outputs (pubchem_like_pattern_counts.csv, molecule_patterns.csv and
fg_columns.json); AccFG is not run again.
"""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from bernoulli_mixture_clustering import cluster_pattern_counts_overlapping, describe_clusters

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'molecule-fg-data'
OUTPUT_DIR = DATA_DIR / 'csv_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
PRUNE_THRESHOLD = 1e-3  # default for --prune-threshold; see bernoulli_mixture_em

PATTERN_OUTPUT = OUTPUT_DIR / 'pubchem_like_pattern_counts.csv'
MOLECULE_PATTERNS = OUTPUT_DIR / 'molecule_patterns.csv'
FG_COLUMNS = OUTPUT_DIR / 'fg_columns.json'
CLUSTER_OUTPUT = OUTPUT_DIR / 'pattern_clusters.csv'
CENTROID_OUTPUT = OUTPUT_DIR / 'cluster_centroids.csv'


def member_cids_by_pattern(molecule_patterns: pd.DataFrame) -> pd.Series:
    """pattern_index -> comma-separated cids of the molecules with that pattern."""
    return (molecule_patterns.groupby('pattern_index', sort=False)['cid']
            .agg(lambda s: ','.join(sorted(pd.unique(s.astype(str)), key=lambda c: (len(c), c))))
            .rename('member_cids'))


def main(max_components: int = 12, tau: float = 0.3, top_n: int | None = 2, seed: int = 0):
    for path in (PATTERN_OUTPUT, MOLECULE_PATTERNS, FG_COLUMNS):
        if not path.exists():
            raise SystemExit(f'{path} not found: run build_pattern_count_dictionary.py first')
    pattern_df = pd.read_csv(PATTERN_OUTPUT, dtype={'pattern': str})
    fg_names = json.loads(FG_COLUMNS.read_text())

    clustered, means, weights = cluster_pattern_counts_overlapping(
        pattern_df,
        max_components=max_components,
        tau=tau,
        top_n=top_n,
        seed=seed,
        verbose=True,
        prune_threshold=PRUNE_THRESHOLD,
    )

    canonical_df = clustered.copy().sort_values(['cluster_id', 'pattern_index']).reset_index(drop=True)
    molecule_patterns = pd.read_csv(MOLECULE_PATTERNS, dtype={'cid': str})
    canonical_df['member_cids'] = canonical_df['pattern_index'].map(
        member_cids_by_pattern(molecule_patterns))

    canonical_df.to_csv(CLUSTER_OUTPUT, index=False)

    # Each cluster's own centroid, keyed by cluster_id. label_clusters.py reads it from here:
    # the per-row cluster_representative column holds the centroid of the row's PRIMARY
    # cluster only, so it cannot label a cluster that a pattern joined as a secondary.
    centroids = pd.DataFrame({
        'cluster_id': list(range(len(weights))),
        'mixing_weight': [round(float(w), 6) for w in weights],
        'centroid': [''.join('1' if p >= 0.5 else '0' for p in m) for m in means],
    })
    centroids.to_csv(CENTROID_OUTPUT, index=False)

    primary_clusters = sorted(clustered['cluster_id'].unique())
    for cid in primary_clusters:
        rep = clustered[clustered['cluster_id'] == cid]['cluster_representative'].iloc[0]
        fgs = [fg_names[i] for i, bit in enumerate(rep) if bit == '1']
        weight = clustered[clustered['cluster_id'] == cid]['cluster_weight'].sum()
        print(f'Cluster {cid} (weight={int(weight)}): {fgs[:20]}')

    describe_clusters(means, weights, fg_names, top_k=8)

    print(f'Clustered {len(pattern_df)} unique patterns from {PATTERN_OUTPUT}')
    print(f'Wrote {len(canonical_df)} clustered pattern rows to {CLUSTER_OUTPUT}')
    print(f'Wrote {len(centroids)} cluster centroids to {CENTROID_OUTPUT}')
    print(canonical_df.head(10).to_string(index=False))


if __name__ == '__main__':
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description='Automatic-K overlapping clustering of binary FG patterns using a Bernoulli mixture.')
    parser.add_argument('--max-components', type=int, default=12, help='Upper bound on K; weak components are pruned.')
    parser.add_argument('--tau', type=float, default=0.3, help='Posterior threshold for extra membership assignments.')
    parser.add_argument('--top-n', type=int, default=2, help='Maximum number of clusters a pattern can join.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for deterministic initialization.')
    parser.add_argument('--k', type=int, default=None, help='Deprecated alias for max-components.')
    parser.add_argument('--prune-threshold', type=float, default=PRUNE_THRESHOLD,
                        help='Drop components whose mixing weight ends at or below this fraction of '
                             'molecules (default 0.001), then re-fit the rest. Lower it if a larger '
                             '--max-components does not produce more clusters.')
    args = parser.parse_args()
    PRUNE_THRESHOLD = args.prune_threshold

    if args.k is not None:
        max_components = args.k
    else:
        max_components = args.max_components

    main(max_components=max_components, tau=args.tau, top_n=args.top_n, seed=args.seed)
