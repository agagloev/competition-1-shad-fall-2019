"""
Feature selection по важности CatBoost.
Запускать после precompute.py. Сохраняет selected_features.json.
"""

import json
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from features import FEATURE_COLS

DATA_DIR = Path(__file__).parent
FEATURES_DIR = DATA_DIR / "precomputed"
OUTPUT_PATH = DATA_DIR / "selected_features.json"


def _mean_ndcg(y_true, y_pred, group_id):
    from sklearn.metrics import ndcg_score
    groups = pd.Series(group_id).unique()
    scores = []
    for g in groups:
        mask = group_id == g
        y_g = np.asarray(y_true)[mask].reshape(1, -1)
        p_g = y_pred[mask].reshape(1, -1)
        if y_g.shape[1] > 1:
            scores.append(ndcg_score(y_g, p_g, k=min(10, y_g.shape[1])))
    return np.mean(scores) if scores else 0.0


def _train_fold_and_importance(X_tr, y_tr, g_tr, X_val, y_val, g_val, feature_names):
    """Один фолд: обучение и важность фичей."""
    model = cb.CatBoostRanker(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        min_data_in_leaf=20,
        subsample=0.8,
        verbose=0,
        loss_function="YetiRank:permutations=10",
        random_seed=42,
    )
    model.fit(
        X_tr, y_tr, group_id=g_tr,
        eval_set=cb.Pool(X_val, y_val, group_id=g_val),
        early_stopping_rounds=50,
    )
    train_pool = cb.Pool(X_tr, y_tr, group_id=g_tr)
    imp = model.get_feature_importance(train_pool)
    return imp


def select_by_importance_threshold(
    train_df,
    feature_cols,
    min_importance_pct=1.0,
    n_splits=5,
):
    """
    Удаляет фичи с важностью < min_importance_pct от средней.
    Возвращает отфильтрованный список фичей.
    """
    X = train_df[feature_cols]
    y = train_df["relevance"].values
    g = train_df["query_id"].values

    gkf = GroupKFold(n_splits=n_splits)
    all_imps = []

    for tr_idx, val_idx in gkf.split(X, y, g):
        imp = _train_fold_and_importance(
            X.iloc[tr_idx], y[tr_idx], g[tr_idx],
            X.iloc[val_idx], y[val_idx], g[val_idx],
            feature_cols,
        )
        all_imps.append(imp)

    mean_imp = np.mean(all_imps, axis=0)
    imp_dict = dict(zip(feature_cols, mean_imp))
    threshold = np.mean(mean_imp) * (min_importance_pct / 100)

    selected = [f for f in feature_cols if imp_dict[f] >= threshold]
    return selected, imp_dict


def select_by_rfe(
    train_df,
    feature_cols,
    n_splits=5,
    ndcg_tol=0.001,
):
    """
    Recursive Feature Elimination: убираем по одной худшей фиче,
    пока OOF NDCG не упадёт больше чем на ndcg_tol.
    """
    X = train_df[feature_cols]
    y = train_df["relevance"].values
    g = train_df["query_id"].values

    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(train_df))

    def train_and_oof(cols):
        oof = np.zeros(len(train_df))
        all_imps = []
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, g)):
            X_tr = X[cols].iloc[tr_idx]
            X_val = X[cols].iloc[val_idx]
            model = cb.CatBoostRanker(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                min_data_in_leaf=20,
                subsample=0.8,
                verbose=0,
                loss_function="YetiRank:permutations=10",
                random_seed=42 + fold,
            )
            model.fit(
                X_tr, y[tr_idx], group_id=g[tr_idx],
                eval_set=cb.Pool(X_val, y[val_idx], group_id=g[val_idx]),
                early_stopping_rounds=50,
            )
            oof[val_idx] = model.predict(X_val)
            train_pool = cb.Pool(X_tr, y[tr_idx], group_id=g[tr_idx])
            all_imps.append(dict(zip(cols, model.get_feature_importance(train_pool))))
        mean_imp = {c: np.mean([ai[c] for ai in all_imps]) for c in cols}
        return oof, mean_imp

    current_cols = list(feature_cols)
    oof, imp = train_and_oof(current_cols)
    best_ndcg = _mean_ndcg(y, oof, g)
    best_cols = list(current_cols)

    print(f"Все фичи ({len(current_cols)}): OOF NDCG = {best_ndcg:.4f}")

    while len(current_cols) > 5:
        worst = min(current_cols, key=lambda c: imp[c])
        next_cols = [c for c in current_cols if c != worst]
        oof, imp = train_and_oof(next_cols)
        ndcg = _mean_ndcg(y, oof, g)
        print(f"  Убрали '{worst}' ({len(next_cols)} фичей): NDCG = {ndcg:.4f}")

        if ndcg < best_ndcg - ndcg_tol:
            print(f"  Стоп: NDCG упал (tol={ndcg_tol})")
            break
        current_cols = next_cols
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_cols = list(current_cols)

    return best_cols


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["threshold", "rfe"],
        default="threshold",
        help="threshold: по порогу важности; rfe: recursive elimination",
    )
    parser.add_argument(
        "--min-importance",
        type=float,
        default=1.0,
        help="Для threshold: мин. важность в %% от средней (default 1.0)",
    )
    parser.add_argument(
        "--ndcg-tol",
        type=float,
        default=0.001,
        help="Для rfe: допуск падения NDCG (default 0.001)",
    )
    args = parser.parse_args()

    train_path = FEATURES_DIR / "train_features.parquet"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Сначала запустите: python precompute.py\nОжидался файл: {train_path}"
        )

    print(f"Загрузка {train_path}...")
    train_df = pd.read_parquet(train_path)

    available = [c for c in FEATURE_COLS if c in train_df.columns]
    print(f"Доступно фичей: {len(available)}")

    if args.mode == "threshold":
        selected, imp_dict = select_by_importance_threshold(
            train_df, available,
            min_importance_pct=args.min_importance,
        )
        print("\nВажность (низкая → высокая):")
        for f in sorted(imp_dict, key=imp_dict.get):
            mark = "✓" if f in selected else "✗"
            print(f"  {mark} {f}: {imp_dict[f]:.2f}")
    else:
        selected = select_by_rfe(train_df, available, ndcg_tol=args.ndcg_tol)

    print(f"\nВыбрано фичей: {len(selected)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)

    print(f"Сохранено: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
