"""
Baseline для Kaggle Competition 1 SHAD Fall 2019
Задача: ранжирование организаций по релевантности запросу
"""

import argparse
import json
from pathlib import Path
import warnings

import pandas as pd
warnings.filterwarnings("ignore")

from data import load_all, save_submission
from features import extract_features, build_idf_from_corpus, FEATURE_COLS
from precompute import load_precomputed_features, has_legacy_precomputed, has_modular_precomputed
from train import train_and_predict, DEFAULT_N_SPLITS

DATA_DIR = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precomputed",
        action="store_true",
        help="Загрузить precomputed/precomputed фичи из parquet",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Путь к selected_features.json (или auto — если есть)",
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="Использовать все фичи (без отбора)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help=f"Количество фолдов GroupKFold (default {DEFAULT_N_SPLITS})",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        metavar="N",
        help="CatBoost verbose: каждые N iter печать learn/test. 100 — для диагностики пере/недообучения",
    )
    args = parser.parse_args()

    feature_cols = None
    if not args.all_features and args.features:
        p = Path(args.features) if args.features != "auto" else DATA_DIR / "selected_features.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                feature_cols = json.load(f)
            print(f"Используем {len(feature_cols)} фичей из {p.name}")

    if args.precomputed:
        feat_dir = DATA_DIR / "precomputed"
        if has_modular_precomputed():
            print("[1/3] Загрузка precomputed фичей (по группам)...")
            train, test = load_precomputed_features()
        elif has_legacy_precomputed():
            train_path = feat_dir / "train_features.parquet"
            test_path = feat_dir / "test_features.parquet"
            print("[1/3] Загрузка precomputed (монолитный формат)...")
            train = pd.read_parquet(train_path)
            test = pd.read_parquet(test_path)
        else:
            raise FileNotFoundError(
                "Нет precomputed фичей. Запустите: python precompute.py [--no-bert]"
            )
        if feature_cols is None:
            feature_cols = [c for c in FEATURE_COLS if c in train.columns]
            print(f"Используем все фичи ({len(feature_cols)})")
    else:
        print("[1/5] Загрузка данных...")
        data = load_all()
        train = data["train"]
        test = data["test"]
        all_clicks = data["clicks"]
        all_org = data["org_info"]
        all_rubric = data["rubric_info"]

        print(f"Train: {len(train)} строк, Test: {len(test)} строк")
        print(f"Уникальных query в train: {train['query_id'].nunique()}")

        print("\n[2/5] IDF по корпусу org_name...")
        idf_dict = build_idf_from_corpus(train["org_name"].tolist() + test["org_name"].tolist())

        print("\n[3/5] Извлечение фичей (train)...")
        train = extract_features(train, all_clicks, all_org, all_rubric, idf_dict=idf_dict, use_bert=True)
        print("[4/5] Извлечение фичей (test)...")
        test = extract_features(test, all_clicks, all_org, all_rubric, idf_dict=idf_dict, use_bert=True)
        if feature_cols is None:
            feature_cols = [c for c in FEATURE_COLS if c in train.columns]
            print(f"Используем все фичи ({len(feature_cols)})")

    n_splits = args.n_folds if args.n_folds is not None else DEFAULT_N_SPLITS
    step = "[2/3]" if args.precomputed else "[5/5]"
    print(f"\n{step} Обучение CatBoostRanker ({n_splits}-fold GroupKFold, LambdaMart)...")
    if args.verbose:
        print("  (learn=тренировка, test=валидация. learn>>test=переобучение, оба низкие=недообучение)")
    _, test_preds = train_and_predict(
        train, test, feature_cols=feature_cols, n_splits=n_splits, verbose=args.verbose
    )

    print("\nСохранение submission...")
    submission = save_submission(test, test_preds)
    print("Submission сохранён: submission.csv")
    print(submission.head(10).to_string())


if __name__ == "__main__":
    main()
