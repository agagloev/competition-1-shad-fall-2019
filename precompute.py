"""
Предрасчёт фичей и сохранение в parquet.
Запускать перед feature_selection и baseline --precomputed.
"""

import pickle
from pathlib import Path

from data import load_all
from features import extract_features, build_idf_from_corpus, FEATURE_COLS

DATA_DIR = Path(__file__).parent
FEATURES_DIR = DATA_DIR / "precomputed"


def main():
    print("Загрузка данных...")
    data = load_all()
    train = data["train"]
    test = data["test"]

    print("IDF по корпусу org_name...")
    idf_dict = build_idf_from_corpus(
        train["org_name"].tolist() + test["org_name"].tolist()
    )

    print("Извлечение фичей (train)...")
    train_fe = extract_features(
        train.copy(),
        data["clicks"],
        data["org_info"],
        data["rubric_info"],
        idf_dict=idf_dict,
        use_bert=True,
    )

    print("Извлечение фичей (test)...")
    test_fe = extract_features(
        test.copy(),
        data["clicks"],
        data["org_info"],
        data["rubric_info"],
        idf_dict=idf_dict,
        use_bert=True,
    )

    FEATURES_DIR.mkdir(exist_ok=True)

    train_cols = ["query_id", "org_id", "relevance"] + FEATURE_COLS
    test_cols = ["query_id", "org_id"] + FEATURE_COLS

    train_fe[train_cols].to_parquet(FEATURES_DIR / "train_features.parquet", index=False)
    test_fe[test_cols].to_parquet(FEATURES_DIR / "test_features.parquet", index=False)

    with open(FEATURES_DIR / "idf_dict.pkl", "wb") as f:
        pickle.dump(idf_dict, f)

    print(f"\nСохранено: {FEATURES_DIR / 'train_features.parquet'}")
    print(f"Сохранено: {FEATURES_DIR / 'test_features.parquet'}")
    print(f"Сохранено: {FEATURES_DIR / 'idf_dict.pkl'}")


if __name__ == "__main__":
    main()
