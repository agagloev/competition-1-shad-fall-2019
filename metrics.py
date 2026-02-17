"""
NDCG по референсной реализации (bwhite/3726239).
Используется для соответствия метрике лидерборда.
"""

import numpy as np


def dcg_at_k(r, k, method=1):
    """Discountecd cumulative gain.
    method=1: weights [1/log2(2), 1/log2(3), ...]
    """
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError("method must be 0 or 1.")
    return 0.0


def ndcg_at_k(r, k, method=1):
    """Normalized DCG: DCG / IDCG."""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k, method) / dcg_max


def mean_ndcg_at_k(y_true, y_pred, group_id, k=10, method=1):
    """
    Средний NDCG@k по группам (query_id).
    y_true, y_pred — массивы relevance и scores в том же порядке.
    Для каждой группы: сортируем relevance по убыванию y_pred, считаем NDCG.
    """
    group_id = np.asarray(group_id)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    groups = np.unique(group_id)
    scores = []
    for g in groups:
        mask = group_id == g
        rel = y_true[mask]
        pred = y_pred[mask]
        if len(rel) < 2:
            continue
        # Порядок по убыванию предсказания
        order = np.argsort(-pred)
        r_ordered = rel[order]
        n = min(k, len(r_ordered))
        scores.append(ndcg_at_k(r_ordered, n, method=method))

    return np.mean(scores) if scores else 0.0
