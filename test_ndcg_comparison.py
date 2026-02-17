"""
Сравнение NDCG: sklearn vs референсная реализация (bwhite).
Запуск: python test_ndcg_comparison.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


# --- Старая реализация (sklearn) ---
def mean_ndcg_sklearn(y_true, y_pred, group_id, k=10):
    """Средний NDCG@k по группам — sklearn.ndcg_score."""
    groups = pd.Series(group_id).unique()
    scores = []
    for g in groups:
        mask = group_id == g
        y_g = np.asarray(y_true)[mask].reshape(1, -1)
        p_g = np.asarray(y_pred)[mask].reshape(1, -1)
        if y_g.shape[1] > 1:
            scores.append(ndcg_score(y_g, p_g, k=min(k, y_g.shape[1])))
    return np.mean(scores) if scores else 0.0


# --- Новая реализация (bwhite) ---
def mean_ndcg_bwhite(y_true, y_pred, group_id, k=10, method=1):
    """Средний NDCG@k — референсная формула bwhite."""
    from metrics import mean_ndcg_at_k
    return mean_ndcg_at_k(y_true, y_pred, group_id, k=k, method=method)


def run_unit_tests():
    """Проверка на простых примерах из docstring bwhite."""
    from metrics import ndcg_at_k
    print("=== Unit tests (bwhite docstring examples) ===")
    # ndcg_at_k([0], 5, method=1) -> 0
    assert ndcg_at_k([0], 5, method=1) == 0.0
    # ndcg_at_k([1], 5, method=1) -> 1
    assert ndcg_at_k([1], 5, method=1) == 1.0
    print("  ndcg_at_k([0], 5, method=1) =", ndcg_at_k([0], 5, method=1))
    print("  ndcg_at_k([1], 5, method=1) =", ndcg_at_k([1], 5, method=1))
    print("  ndcg_at_k([1,0], 5, method=1) =", ndcg_at_k([1, 0], 5, method=1))
    print("  ndcg_at_k([0,1], 5, method=1) =", ndcg_at_k([0, 1], 5, method=1))
    print("  ndcg_at_k([0,1,1], 5, method=1) =", ndcg_at_k([0, 1, 1], 5, method=1))
    print("  ndcg_at_k([0,1,1,1], 5, method=1) =", ndcg_at_k([0, 1, 1, 1], 5, method=1))
    print()


def run_comparison_synthetic():
    """Синтетические данные: 3 query, разное кол-во документов."""
    print("=== Synthetic data ===")
    # Query 1: 4 docs, perfect ranking
    # Query 2: 3 docs, reverse ranking
    # Query 3: 5 docs, mixed
    group_id = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3])
    y_true = np.array([0.9, 0.5, 0.2, 0.0, 0.8, 0.3, 0.1, 0.6, 0.6, 0.1, 0.0, 0.0])
    y_pred = np.array([0.9, 0.5, 0.2, 0.0, 0.1, 0.3, 0.8, 0.6, 0.1, 0.6, 0.0, 0.0])  # q2 reversed

    sk = mean_ndcg_sklearn(y_true, y_pred, group_id, k=10)
    bw = mean_ndcg_bwhite(y_true, y_pred, group_id, k=10)

    print(f"  sklearn NDCG@10: {sk:.6f}")
    print(f"  bwhite NDCG@10:  {bw:.6f}")
    print(f"  diff:           {bw - sk:+.6f}")
    print()


def run_comparison_real():
    """Реальные данные из precomputed, если есть."""
    from pathlib import Path
    path = Path(__file__).parent / "precomputed" / "train_features.parquet"
    if not path.exists():
        print("=== Real data: precomputed/train_features.parquet not found ===")
        return

    print("=== Real data (train_features.parquet) ===")
    df = pd.read_parquet(path)
    group_id = df["query_id"].values
    y_true = df["relevance"].values

    # Имитация предсказаний: relevance + небольшой шум
    np.random.seed(42)
    y_pred = y_true + np.random.randn(len(y_true)) * 0.1

    for k in [5, 10, 20]:
        sk = mean_ndcg_sklearn(y_true, y_pred, group_id, k=k)
        bw = mean_ndcg_bwhite(y_true, y_pred, group_id, k=k)
        print(f"  NDCG@{k}: sklearn={sk:.4f}, bwhite={bw:.4f}, diff={bw-sk:+.4f}")

    # Perfect ranking (y_pred = y_true)
    y_pred_perfect = y_true.copy()
    sk_p = mean_ndcg_sklearn(y_true, y_pred_perfect, group_id, k=10)
    bw_p = mean_ndcg_bwhite(y_true, y_pred_perfect, group_id, k=10)
    print(f"  Perfect rank NDCG@10: sklearn={sk_p:.4f}, bwhite={bw_p:.4f}")
    print()


def main():
    run_unit_tests()
    run_comparison_synthetic()
    run_comparison_real()
    print("Done.")


if __name__ == "__main__":
    main()
