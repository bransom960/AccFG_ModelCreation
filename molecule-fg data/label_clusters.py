from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from accfg import AccFG

ROOT = PROJECT_ROOT
DEFAULT_INPUT = ROOT / 'molecule-fg data' / 'pattern_clusters_overlapping.csv'
FALLBACK_INPUT = ROOT / 'molecule-fg data' / 'pattern_clusters.csv'
CLUSTER_LABELED = ROOT / 'molecule-fg data' / 'pattern_clusters_labeled.csv'
CLUSTER_SUMMARY = ROOT / 'molecule-fg data' / 'cluster_summary.csv'


def decode_pattern(pattern: str, fg_names: list[str]) -> list[str]:
    """Return FG names where the bit is 1."""
    return [fg_names[i] for i, bit in enumerate(pattern) if bit == '1']


def weighted_common_fgs(member_patterns, member_counts, fg_names, threshold=0.5):
    """FGs present in >= threshold weighted fraction of cluster members."""
    if not member_patterns:
        return []

    total = sum(member_counts)
    if total == 0:
        return []

    weighted_presence = [0.0] * len(fg_names)
    for pat, cnt in zip(member_patterns, member_counts):
        for i, bit in enumerate(pat):
            if bit == '1':
                weighted_presence[i] += cnt

    return [
        fg_names[i]
        for i in range(len(fg_names))
        if weighted_presence[i] / total >= threshold
    ]


def parse_cluster_memberships(value):
    if pd.isna(value):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return [int(part.strip()) for part in value.split(',') if part.strip()]
    return [int(value)]


def load_cluster_rows(cluster_input: Path) -> pd.DataFrame:
    df = pd.read_csv(cluster_input)

    if 'cluster_memberships' in df.columns:
        expanded_rows = []
        for _, row in df.iterrows():
            cluster_ids = parse_cluster_memberships(row.get('cluster_memberships'))
            if not cluster_ids:
                if 'cluster_id' in df.columns and pd.notna(row.get('cluster_id')):
                    cluster_ids = [int(row['cluster_id'])]
            for cid in cluster_ids:
                new_row = row.to_dict()
                new_row['cluster_id'] = int(cid)
                expanded_rows.append(new_row)
        return pd.DataFrame(expanded_rows)

    return df


def main():
    cluster_input = DEFAULT_INPUT if DEFAULT_INPUT.exists() else FALLBACK_INPUT
    afg = AccFG(print_load_info=False, lite=False)
    fg_names = list(afg.dict_fgs.keys())

    df = load_cluster_rows(cluster_input)

    rows = []
    summary_rows = []
    for cid, group in df.groupby('cluster_id'):
        centroid = group['cluster_representative'].iloc[0]
        centroid_fgs = decode_pattern(centroid, fg_names)

        member_patterns = group['pattern'].tolist()
        member_counts = group['count'].tolist()
        union_fgs = sorted({
            fg
            for pat in member_patterns
            for fg in decode_pattern(pat, fg_names)
        })
        common_fgs = weighted_common_fgs(
            member_patterns,
            member_counts,
            fg_names,
            threshold=0.5,
        )

        group = group.copy()
        group['cluster_representative_fgs'] = ','.join(centroid_fgs)
        group['cluster_representative_n_fgs'] = len(centroid_fgs)
        group['cluster_member_fgs_union'] = ','.join(union_fgs)
        group['cluster_member_fgs_common'] = ','.join(common_fgs)
        rows.append(group)

        summary_rows.append({
            'cluster_id': cid,
            'cluster_weight': int(group['count'].sum()),
            'n_patterns': len(group),
            'centroid_fgs': ','.join(centroid_fgs),
            'centroid_n_fgs': len(centroid_fgs),
            'union_fgs': ','.join(union_fgs),
            'union_n_fgs': len(union_fgs),
            'common_fgs': ','.join(common_fgs),
            'common_n_fgs': len(common_fgs),
        })

    labeled = pd.concat(rows, ignore_index=True)
    labeled.to_csv(CLUSTER_LABELED, index=False)

    summary = pd.DataFrame(summary_rows).sort_values('cluster_id').reset_index(drop=True)
    summary.to_csv(CLUSTER_SUMMARY, index=False)

    print(f'Wrote {len(labeled)} labeled rows to {CLUSTER_LABELED}')
    print(f'Wrote {len(summary)} cluster summaries to {CLUSTER_SUMMARY}')
    print()
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
