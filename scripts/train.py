"""
Обучение и предсказание для Kaggle Competition 1 SHAD Fall 2019
Ранжирование организаций по релевантности запросу.
Запуск: python -m scripts.train [--n-folds N]

Требует предпосчитанные фичи: python -m scripts.precompute
"""

import argparse
import warnings

import pandas as pd
warnings.filterwarnings("ignore")

from src.data import save_submission
from src.features import FEATURE_COLS
from src.precompute import load_precomputed_features, has_legacy_precomputed, has_modular_precomputed
from src.train import train_and_predict, DEFAULT_N_SPLITS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help=f"Количество фолдов GroupKFold (default {DEFAULT_N_SPLITS})",
    )
    args = parser.parse_args()

    if has_modular_precomputed():
        print("[1/2] Загрузка precomputed фичей (по группам)...")
        train, test = load_precomputed_features()
    elif has_legacy_precomputed():
        from pathlib import Path
        feat_dir = Path(__file__).resolve().parent.parent / "precomputed"
        train_path = feat_dir / "train_features.parquet"
        test_path = feat_dir / "test_features.parquet"
        print("[1/2] Загрузка precomputed (монолитный формат)...")
        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)
    else:
        raise FileNotFoundError(
            "Нет precomputed фичей. Запустите: python -m scripts.precompute"
        )

    feature_cols = [c for c in FEATURE_COLS if c in train.columns]
    print(f"Используем фичи ({len(feature_cols)})")

    n_splits = args.n_folds if args.n_folds is not None else DEFAULT_N_SPLITS
    print(f"\n[2/2] Обучение CatBoostRanker ({n_splits}-fold GroupKFold, LambdaMart)...")
    _, test_preds = train_and_predict(
        train, test, feature_cols=feature_cols, n_splits=n_splits
    )

    print("\nСохранение submission...")
    submission = save_submission(test, test_preds)
    print("Submission сохранён: submission.csv")
    print(submission.head(10).to_string())


if __name__ == "__main__":
    main()
