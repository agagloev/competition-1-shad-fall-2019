"""
Обучение модели ранжирования для Kaggle Competition 1 SHAD Fall 2019
"""

import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

from features import FEATURE_COLS
from metrics import mean_ndcg_at_k

# Дефолтные параметры модели (из предыдущей версии)
DEFAULT_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "min_data_in_leaf": 20,
    "subsample": 0.8,
    "early_stopping_rounds": 50,
}


def _mean_ndcg(y_true, y_pred, group_id, k=10):
    """Средний NDCG@k по референсной формуле (bwhite, method=1)."""
    return mean_ndcg_at_k(y_true, y_pred, group_id, k=k, method=1)


def train_and_predict(train_df, test_df, feature_cols=None, n_splits=5, **kwargs):
    """
    Обучение CatBoostRanker с GroupKFold, LambdaMart.
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
        "loss_function": "LambdaMart",
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
