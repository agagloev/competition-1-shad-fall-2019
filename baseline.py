"""
Baseline для Kaggle Competition 1 SHAD Fall 2019
Задача: ранжирование организаций по релевантности запросу
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

from data import load_all, save_submission
from features import extract_features, build_idf_from_corpus
from train import train_and_predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert", action="store_true", help="Добавить BERT-фичи (медленно)")
    args = parser.parse_args()

    print("[1/5] Загрузка данных...")
    data = load_all()
    train = data["train"]
    test = data["test"]
    all_clicks = data["clicks"]
    all_org = data["org_info"]
    all_rubric = data["rubric_info"]

    print(f"Train: {len(train)} строк, Test: {len(test)} строк")
    print(f"Уникальных query в train: {train['query_id'].nunique()}")
    print(f"Relevance: min={train['relevance'].min():.3f}, max={train['relevance'].max():.3f}, mean={train['relevance'].mean():.3f}")

    print("\n[2/5] IDF по корпусу org_name...")
    idf_dict = build_idf_from_corpus(train["org_name"].tolist() + test["org_name"].tolist())

    print("\n[3/5] Извлечение фичей (train)...")
    train = extract_features(train, all_clicks, all_org, all_rubric, idf_dict=idf_dict, use_bert=args.bert)
    print("[4/5] Извлечение фичей (test)...")
    test = extract_features(test, all_clicks, all_org, all_rubric, idf_dict=idf_dict, use_bert=args.bert)

    print("\n[5/5] Обучение CatBoostRanker (5-fold GroupKFold, YetiRank)...")
    _, test_preds = train_and_predict(train, test)

    print("\nСохранение submission...")
    submission = save_submission(test, test_preds)
    print("Submission сохранён: submission.csv")
    print(submission.head(10).to_string())


if __name__ == "__main__":
    main()
