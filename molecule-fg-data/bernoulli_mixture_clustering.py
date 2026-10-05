from __future__ import annotations

import numpy as np
import pandas as pd


def bernoulli_mixture_em(
    X: np.ndarray,
    n_components: int,
    n_iter: int = 200,
    tol: float = 1e-5,
    seed: int = 0,
    prune_threshold: float = 1e-3,
    verbose: bool = False,
    sample_weights: np.ndarray | None = None,
    n_restarts: int = 3,
):
    """Fit a Bernoulli mixture to binary data X (n_samples, n_features).

    If sample_weights are provided, each row is treated as a weighted observation rather
    than expanded into repeated rows. This preserves the exact mixture objective for
    unique pattern data while avoiding the expensive molecule-level expansion.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    K = max(1, int(n_components))

    if n == 0:
        return np.array([], dtype=float), np.zeros((0, d), dtype=float), np.zeros((0, K), dtype=float)

    if sample_weights is None:
        sample_weights = np.ones(n, dtype=float)
    else:
        sample_weights = np.asarray(sample_weights, dtype=float)
        if sample_weights.shape[0] != n:
            raise ValueError(f'sample_weights length {sample_weights.shape[0]} does not match X rows {n}')

    total_weight = float(sample_weights.sum())
    if total_weight <= 0:
        raise ValueError('sample_weights must sum to a positive value')
    sample_weights = sample_weights / total_weight

    best_result = None
    best_ll = -np.inf

    for restart in range(max(1, int(n_restarts))):
        rng = np.random.default_rng(seed + restart)
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

            log_resp_shift = log_resp - log_resp.max(axis=1, keepdims=True)
            resp = np.exp(log_resp_shift)
            resp /= resp.sum(axis=1, keepdims=True)

            Nk = (resp * sample_weights[:, None]).sum(axis=0) + 1e-12
            weights = Nk / Nk.sum()
            means = (resp.T @ (sample_weights[:, None] * X)) / Nk[:, None]

            marginal = np.log(np.sum(np.exp(log_resp_shift), axis=1)) + log_resp.max(axis=1)
            log_likelihood = float(np.sum(sample_weights * marginal))

            if verbose and (it % 10 == 0 or it == n_iter - 1):
                print(f'  iter {it:3d}  log-likelihood = {log_likelihood:.2f}')

            if np.isfinite(log_likelihood_old):
                rel_change = abs(log_likelihood - log_likelihood_old) / max(1e-12, abs(log_likelihood_old))
                if rel_change < tol:
                    break
            log_likelihood_old = log_likelihood

        keep = weights > max(prune_threshold, 1.0 / max(1, n))
        if not np.any(keep):
            keep[np.argmax(weights)] = True

        candidate_weights = weights[keep]
        if candidate_weights.size > 0:
            candidate_weights = candidate_weights / candidate_weights.sum()
        candidate_means = means[keep]
        candidate_resp = resp[:, keep]
        order = np.argsort(-candidate_weights)
        candidate_weights = candidate_weights[order]
        candidate_means = candidate_means[order]
        candidate_resp = candidate_resp[:, order]

        result = (candidate_weights, candidate_means, candidate_resp)
        if log_likelihood > best_ll:
            best_result = result
            best_ll = log_likelihood

    if best_result is None:
        raise RuntimeError('Bernoulli EM did not produce a valid solution')

    weights, means, resp = best_result
    return weights, means, resp


def patterns_to_matrix(patterns: list[str] | np.ndarray) -> np.ndarray:
    """Convert binary strings or numeric matrices into a 0/1 numpy array.

    If a numeric matrix is already supplied, keep it as-is instead of re-wrapping every
    character into Python ints.
    """
    if isinstance(patterns, np.ndarray):
        arr = np.asarray(patterns, dtype=np.int8)
        if arr.ndim == 1 and arr.size > 0 and isinstance(patterns[0], str):
            return np.fromiter((int(ch) for ch in patterns[0]), dtype=np.int8).reshape(1, -1)
        return arr

    if not patterns:
        return np.zeros((0, 0), dtype=np.int8)
    if isinstance(patterns[0], str):
        rows = [np.fromiter((int(ch) for ch in pattern), dtype=np.int8) for pattern in patterns]
        if not rows:
            return np.zeros((0, 0), dtype=np.int8)
        return np.vstack(rows)
    return np.asarray(patterns, dtype=np.int8)


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
    counts = df['count'].astype(float).to_numpy()
    X = patterns_to_matrix(patterns)
    if X.size == 0:
        X = np.zeros((0, len(patterns[0]) if patterns else 0), dtype=np.int8)

    weights, means, resp_fit = bernoulli_mixture_em(
        X,
        n_components=max(1, int(max_components)),
        seed=seed,
        verbose=verbose,
        sample_weights=counts,
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
