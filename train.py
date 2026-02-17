"""
Обучение модели ранжирования для Kaggle Competition 1 SHAD Fall 2019
"""

import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, ndcg_score

from features import FEATURE_COLS

# Дефолтные параметры модели (из предыдущей версии)
DEFAULT_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "min_data_in_leaf": 20,
    "subsample": 0.8,
    "early_stopping_rounds": 50,
}


def _mean_ndcg(y_true, y_pred, group_id):
    """Средний NDCG@10 по группам (query_id)"""
    groups = pd.Series(group_id).unique()
    scores = []
    for g in groups:
        mask = group_id == g
        y_g = np.asarray(y_true)[mask].reshape(1, -1)
        p_g = y_pred[mask].reshape(1, -1)
        if y_g.shape[1] > 1:
            scores.append(ndcg_score(y_g, p_g, k=min(10, y_g.shape[1])))
    return np.mean(scores) if scores else 0.0


def train_and_predict(train_df, test_df, feature_cols=None, n_splits=5, **kwargs):
    """
    Обучение CatBoostRanker с GroupKFold, YetiRank.
    feature_cols: список фичей (по умолчанию FEATURE_COLS).
    kwargs переопределяют DEFAULT_PARAMS.
    """
    params = {**DEFAULT_PARAMS, **kwargs}
    cols = feature_cols if feature_cols is not None else FEATURE_COLS
    X_train = train_df[cols]
    y_train = train_df["relevance"].values
    group_id_train = train_df["query_id"].values
    X_test = test_df[cols]

    gkf = GroupKFold(n_splits=n_splits)
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    model_params = {
        "iterations": params["iterations"],
        "learning_rate": params["learning_rate"],
        "depth": params.get("depth", 6),
        "min_data_in_leaf": params.get("min_data_in_leaf", 20),
        "subsample": params.get("subsample", 0.8),
        "verbose": 0,
        "loss_function": "YetiRank:permutations=10",
    }

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, group_id_train)):
        X_tr = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]
        g_tr = group_id_train[tr_idx]
        g_val = group_id_train[val_idx]

        model = cb.CatBoostRanker(
            **model_params,
            random_seed=42 + fold,
        )

        model.fit(
            X_tr, y_tr, group_id=g_tr,
            eval_set=cb.Pool(X_val, y_val, group_id=g_val),
            early_stopping_rounds=params["early_stopping_rounds"],
        )

        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / n_splits

        rmse = np.sqrt(mean_squared_error(y_train[val_idx], oof_preds[val_idx]))
        ndcg = _mean_ndcg(y_train[val_idx], oof_preds[val_idx], g_val)
        print(f"  Fold {fold + 1}: RMSE = {rmse:.4f}, NDCG = {ndcg:.4f}")

    overall_rmse = np.sqrt(mean_squared_error(y_train, oof_preds))
    overall_ndcg = _mean_ndcg(y_train, oof_preds, group_id_train)
    print(f"\nOOF RMSE: {overall_rmse:.4f}, OOF NDCG: {overall_ndcg:.4f}")

    return oof_preds, test_preds
