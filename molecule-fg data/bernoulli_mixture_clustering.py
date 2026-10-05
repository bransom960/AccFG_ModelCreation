from __future__ import annotations

import numpy as np
import pandas as pd


def bernoulli_mixture_em(
    X: np.ndarray,
    n_components: int,
    n_iter: int = 200,
    tol: float = 1e-5,
    seed: int = 0,
    prune_threshold: float = 1e-2,
    verbose: bool = False,
):
    """Fit a Bernoulli mixture to binary data X (n_samples, n_features)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    K = max(1, int(n_components))

    if n == 0:
        return np.array([], dtype=float), np.zeros((0, d), dtype=float), np.zeros((0, K), dtype=float)

    weights = np.ones(K, dtype=float) / K
    means = rng.uniform(0.25, 0.75, size=(K, d))
    log_likelihood_old = -np.inf

    for it in range(n_iter):
        log_resp = np.zeros((n, K), dtype=float)
        for k in range(K):
            log_p = (
                X * np.log(means[k] + 1e-12)
                + (1 - X) * np.log(1 - means[k] + 1e-12)
            ).sum(axis=1)
            log_resp[:, k] = np.log(weights[k] + 1e-12) + log_p

        log_resp -= log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp)
        resp /= resp.sum(axis=1, keepdims=True)

        Nk = resp.sum(axis=0) + 1e-12
        weights = Nk / n
        means = (resp.T @ X) / Nk[:, None]

        log_likelihood = 0.0
        for k in range(K):
            log_p = (
                X * np.log(means[k] + 1e-12)
                + (1 - X) * np.log(1 - means[k] + 1e-12)
            ).sum(axis=1)
            log_likelihood += (resp[:, k] * (np.log(weights[k] + 1e-12) + log_p)).sum()

        if verbose and (it % 10 == 0 or it == n_iter - 1):
            print(f'  iter {it:3d}  log-likelihood = {log_likelihood:.2f}')

        if abs(log_likelihood - log_likelihood_old) < tol:
            break
        log_likelihood_old = log_likelihood

    keep = weights > prune_threshold
    if not np.any(keep):
        keep[np.argmax(weights)] = True

    weights = weights[keep]
    means = means[keep]
    resp = resp[:, keep]

    order = np.argsort(-weights)
    weights = weights[order]
    means = means[order]
    resp = resp[:, order]

    return weights, means, resp


def patterns_to_matrix(patterns: list[str]) -> np.ndarray:
    """Convert a list of binary strings to a 0/1 numpy matrix."""
    if not patterns:
        return np.zeros((0, 0), dtype=np.int8)
    return np.array([[int(ch) for ch in pattern] for pattern in patterns], dtype=np.int8)


def assign_overlapping(resp: np.ndarray, tau: float = 0.3, top_n: int | None = 2) -> list[list[int]]:
    """Return overlapping cluster assignments for each pattern row."""
    memberships: list[list[int]] = []
    for row in resp:
        order = np.argsort(-row)
        picked = [int(order[0])]
        for cluster_id in order[1:]:
            cluster_idx = int(cluster_id)
            if top_n is not None and len(picked) >= top_n:
                break
            if row[cluster_idx] >= tau:
                picked.append(cluster_idx)
        memberships.append(sorted(set(picked)))
    return memberships


def cluster_pattern_counts_overlapping(
    pattern_df: pd.DataFrame,
    max_components: int = 12,
    tau: float = 0.3,
    top_n: int | None = 2,
    seed: int = 0,
    verbose: bool = True,
):
    """Cluster binary FG patterns with automatic K and overlapping membership."""
    if 'pattern' not in pattern_df.columns:
        raise ValueError("pattern_df must contain a 'pattern' column")
    if 'count' not in pattern_df.columns:
        raise ValueError("pattern_df must contain a 'count' column")

    df = pattern_df.copy().reset_index(drop=True)
    df = df[df['pattern'].astype(str).str.strip().apply(lambda p: any(ch == '1' for ch in p))].copy()

    if df.empty:
        empty = df.copy()
        empty['cluster_id'] = pd.Series([], dtype='int64')
        empty['cluster_primary'] = pd.Series([], dtype='int64')
        empty['cluster_representative'] = pd.Series([], dtype='object')
        empty['cluster_memberships'] = pd.Series([], dtype='object')
        empty['cluster_probabilities'] = pd.Series([], dtype='object')
        empty['cluster_weight'] = pd.Series([], dtype='int64')
        return empty, np.zeros((0, 0), dtype=float), np.zeros((0,), dtype=float)

    patterns = df['pattern'].astype(str).tolist()
    counts = df['count'].astype(int).to_numpy()

    expanded_indices = np.repeat(np.arange(len(patterns)), counts)
    X = patterns_to_matrix([patterns[i] for i in expanded_indices])
    if X.size == 0:
        X = np.zeros((0, len(patterns[0])), dtype=np.int8)

    weights, means, resp_fit = bernoulli_mixture_em(
        X,
        n_components=max(1, int(max_components)),
        seed=seed,
        verbose=verbose,
    )

    K = len(weights)
    if verbose:
        print(f'Discovered K = {K}')
        print(f'Weights: {np.round(weights, 3).tolist()}')

    X_patterns = patterns_to_matrix(patterns)
    pattern_level_resp = np.zeros((len(patterns), K), dtype=float)
    for k in range(K):
        log_p = (
            X_patterns * np.log(means[k] + 1e-12)
            + (1 - X_patterns) * np.log(1 - means[k] + 1e-12)
        ).sum(axis=1)
        pattern_level_resp[:, k] = np.log(weights[k] + 1e-12) + log_p

    pattern_level_resp -= pattern_level_resp.max(axis=1, keepdims=True)
    pattern_level_resp = np.exp(pattern_level_resp)
    pattern_level_resp /= pattern_level_resp.sum(axis=1, keepdims=True)

    memberships = assign_overlapping(pattern_level_resp, tau=tau, top_n=top_n)
    primaries = [m[0] for m in memberships]
    centroids = {k: ''.join('1' if prob >= 0.5 else '0' for prob in means[k]) for k in range(K)}

    out = df.copy()
    out['cluster_id'] = primaries
    out['cluster_primary'] = primaries
    out['cluster_memberships'] = [','.join(map(str, m)) for m in memberships]
    out['cluster_representative'] = [centroids[cid] for cid in primaries]
    out['cluster_probabilities'] = [','.join(f'{p:.3f}' for p in row) for row in pattern_level_resp]
    out['cluster_weight'] = out['count']

    for k in range(K):
        out[f'prob_cluster_{k}'] = pattern_level_resp[:, k].round(4)

    out = out.sort_values(['cluster_id', 'pattern_index']).reset_index(drop=True)
    return out, means, weights


def describe_clusters(means: np.ndarray, weights: np.ndarray, fg_names: list[str], top_k: int = 8):
    print('\n=== Discovered clusters ===')
    for cluster_id, (w, probs) in enumerate(zip(weights, means)):
        top_idx = np.argsort(-probs)[:top_k]
        fgs = [fg_names[i] for i in top_idx if probs[i] > 0.3]
        print(f'Cluster {cluster_id}  weight={w:.3f}  representative={fgs[:10]}')
