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
from train import train_and_predict

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
    args = parser.parse_args()

    feature_cols = None
    if args.features:
        p = Path(args.features) if args.features != "auto" else DATA_DIR / "selected_features.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                feature_cols = json.load(f)
            print(f"Используем {len(feature_cols)} фичей из {p.name}")

    if args.precomputed:
        feat_dir = DATA_DIR / "precomputed"
        train_path = feat_dir / "train_features.parquet"
        test_path = feat_dir / "test_features.parquet"
        if not train_path.exists():
            raise FileNotFoundError(f"Сначала: python precompute.py (нет {train_path})")
        print("[1/3] Загрузка precomputed фичей...")
        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)
        if feature_cols is None:
            feature_cols = [c for c in FEATURE_COLS if c in train.columns]
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
        feature_cols = feature_cols or [c for c in FEATURE_COLS if c in train.columns]

    step = "[2/3]" if args.precomputed else "[5/5]"
    print(f"\n{step} Обучение CatBoostRanker (5-fold GroupKFold, YetiRank)...")
    _, test_preds = train_and_predict(train, test, feature_cols=feature_cols)

    print("\nСохранение submission...")
    submission = save_submission(test, test_preds)
    print("Submission сохранён: submission.csv")
    print(submission.head(10).to_string())


if __name__ == "__main__":
    main()
