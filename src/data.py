"""
Загрузка и сохранение данных для Kaggle Competition 1 SHAD Fall 2019
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"


def load_train():
    """Загрузка train.csv"""
    return pd.read_csv(DATA_PATH / "train.csv")


def load_test():
    """Загрузка test.csv"""
    return pd.read_csv(DATA_PATH / "test.csv")


def load_data():
    """Загрузка train и test"""
    return load_train(), load_test()


def load_clicks():
    """Загрузка информации о кликах (org_id -> list of query strings)"""
    with open(DATA_PATH / "train_clicks_information.json") as f:
        train_clicks = json.load(f)
    with open(DATA_PATH / "test_clicks_information.json") as f:
        test_clicks = json.load(f)
    return train_clicks, test_clicks


def load_org_info():
    """Загрузка информации об организациях"""
    with open(DATA_PATH / "train_org_information.json") as f:
        train_org = json.load(f)
    with open(DATA_PATH / "test_org_information.json") as f:
        test_org = json.load(f)
    return train_org, test_org


def load_rubric_info():
    """Загрузка информации о рубриках (категориях)"""
    with open(DATA_PATH / "train_rubric_information.json") as f:
        train_rubric = json.load(f)
    with open(DATA_PATH / "test_rubric_information.json") as f:
        test_rubric = json.load(f)
    return train_rubric, test_rubric


def load_all():
    """Загрузка всех данных, объединённых для train+test"""
    train, test = load_data()
    train_clicks, test_clicks = load_clicks()
    train_org, test_org = load_org_info()
    train_rubric, test_rubric = load_rubric_info()

    all_clicks = {**train_clicks, **test_clicks}
    all_org = {**train_org, **test_org}
    all_rubric = {**train_rubric, **test_rubric}

    return {
        "train": train,
        "test": test,
        "clicks": all_clicks,
        "org_info": all_org,
        "rubric_info": all_rubric,
    }


def save_submission(test_df, predictions, path=None):
    """
    Сохранение submission.csv.
    predictions — массив scores в том же порядке, что и test_df.
    """
    if path is None:
        path = PROJECT_ROOT / "submission.csv"
    submission = (
        test_df.assign(score=predictions)[["query_id", "org_id", "score"]]
        .sort_values(["query_id", "score"], ascending=[True, False])
        [["query_id", "org_id"]]
    )
    submission.to_csv(path, index=False)
    return submission
