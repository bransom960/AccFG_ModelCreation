from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'molecule-fg data'
OUTPUT_DIR = DATA_DIR / 'csv_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)

CLUSTER_SUMMARY = OUTPUT_DIR / 'cluster_summary.csv'
MODEL_SPECS = OUTPUT_DIR / 'model_specs.csv'
MODEL_ASSIGNMENTS = OUTPUT_DIR / 'model_assignments.csv'
MODEL_REPORT = OUTPUT_DIR / 'model_coverage_report.csv'

REUSE_DECAY = 0.5
OVERLAP_TOLERANCE = 0.5


def assign_models(cluster_summary: pd.DataFrame,
                  model_specs: pd.DataFrame,
                  reuse_decay: float = REUSE_DECAY,
                  overlap_tolerance: float = OVERLAP_TOLERANCE) -> pd.DataFrame:
    """
    Assign clusters to models based on target coverage and rules.
    Returns a long-form DataFrame: one row per (model, cluster).
    """
    total_weight = int(cluster_summary['cluster_weight'].sum())
    cluster_use_count = {cid: 0 for cid in cluster_summary['cluster_id']}
    assignments = []

    for _, spec in model_specs.iterrows():
        model = spec['model_name']
        target = float(spec['target_coverage'])
        min_fgs = int(spec.get('min_fgs', 1))
        max_fgs = int(spec.get('max_fgs', 999))

        candidates = cluster_summary[
            (cluster_summary['centroid_n_fgs'] >= min_fgs) &
            (cluster_summary['centroid_n_fgs'] <= max_fgs)
        ].copy()

        reuse_counts = candidates['cluster_id'].map(cluster_use_count).astype(int)
        candidates = candidates.copy()
        candidates['effective_weight'] = (
            candidates['cluster_weight'].astype(float) * (reuse_decay ** reuse_counts.to_numpy())
        )

        candidates = candidates.sort_values(
            ['effective_weight', 'centroid_n_fgs'],
            ascending=[False, False],
        )

        target_weight = target * total_weight
        cumulative = 0.0
        reused_weight = 0.0

        for _, row in candidates.iterrows():
            if cumulative >= target_weight:
                break

            cid = int(row['cluster_id'])
            w = float(row['effective_weight'])
            was_reused = cluster_use_count[cid] > 0

            if was_reused and reused_weight + w > overlap_tolerance * target_weight:
                continue

            assignments.append({
                'model_name': model,
                'cluster_id': cid,
                'cluster_weight': int(row['cluster_weight']),
                'effective_weight': round(w, 2),
                'centroid_fgs': row['centroid_fgs'],
                'centroid_n_fgs': int(row['centroid_n_fgs']),
                'was_reused': was_reused,
                'target_coverage': target,
            })

            cumulative += w
            if was_reused:
                reused_weight += w
            cluster_use_count[cid] += 1

    return pd.DataFrame(assignments)


def build_report(assignments: pd.DataFrame, total_weight: int) -> pd.DataFrame:
    """Aggregate per-model coverage, FG set, and overlap stats."""
    rows = []
    for model, group in assignments.groupby('model_name'):
        total_effective = group['effective_weight'].sum()
        reused = group[group['was_reused']]
        fgs = sorted({
            fg
            for fgs in group['centroid_fgs']
            for fg in str(fgs).split(',')
            if fg
        })

        rows.append({
            'model_name': model,
            'clusters_used': len(group),
            'clusters_reused': len(reused),
            'total_cluster_weight': int(group['cluster_weight'].sum()),
            'effective_coverage': round(total_effective / total_weight, 4),
            'target_coverage': group['target_coverage'].iloc[0],
            'coverage_gap': round(
                group['target_coverage'].iloc[0] - total_effective / total_weight,
                4,
            ),
            'fgs_used': ','.join(fgs),
            'n_fgs_used': len(fgs),
        })

    return pd.DataFrame(rows)


def main():
    cluster_summary = pd.read_csv(CLUSTER_SUMMARY)
    model_specs = pd.read_csv(MODEL_SPECS)
    total_weight = int(cluster_summary['cluster_weight'].sum())

    print(f'Total molecule weight across all clusters: {total_weight}')
    print(f'Models to fit: {len(model_specs)}')
    print()

    assignments = assign_models(cluster_summary, model_specs)
    assignments.to_csv(MODEL_ASSIGNMENTS, index=False)

    report = build_report(assignments, total_weight)
    report.to_csv(MODEL_REPORT, index=False)

    print(f'Wrote {len(assignments)} assignments to {MODEL_ASSIGNMENTS}')
    print(f'Wrote {len(report)} model reports to {MODEL_REPORT}')
    print()
    print(report.to_string(index=False))


if __name__ == '__main__':
    main()
