"""
Baseline для Kaggle Competition 1 SHAD Fall 2019
Задача: ранжирование организаций по релевантности запросу
"""

import warnings

warnings.filterwarnings("ignore")

from data import load_all, save_submission
from features import extract_features
from train import train_and_predict


def main():
    print("Загрузка данных...")
    data = load_all()
    train = data["train"]
    test = data["test"]
    all_clicks = data["clicks"]
    all_org = data["org_info"]
    all_rubric = data["rubric_info"]

    print(f"Train: {len(train)} строк, Test: {len(test)} строк")
    print(f"Уникальных query в train: {train['query_id'].nunique()}")
    print(f"Relevance: min={train['relevance'].min():.3f}, max={train['relevance'].max():.3f}, mean={train['relevance'].mean():.3f}")

    print("\nИзвлечение фичей...")
    train = extract_features(train, all_clicks, all_org, all_rubric)
    test = extract_features(test, all_clicks, all_org, all_rubric)

    print("\nОбучение CatBoostRanker (5-fold GroupKFold, YetiRank)...")
    _, test_preds = train_and_predict(train, test)

    print("\nСохранение submission...")
    submission = save_submission(test, test_preds)
    print("Submission сохранён: submission.csv")
    print(submission.head(10).to_string())


if __name__ == "__main__":
    main()
