"""
Предрасчёт фичей и сохранение в parquet.
Фичи складываются по группам в отдельные файлы — при добавлении новой фичи
пересчитывается только её группа (precompute --only text).

Запускать перед baseline --precomputed.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_all, load_train, load_test
from features import (
    build_idf_from_corpus,
    extract_feature_group,
    FEATURE_COLS,
    FEATURE_GROUPS,
)

DATA_DIR = Path(__file__).parent
FEATURES_DIR = DATA_DIR / "precomputed"

# Группы, которые сохраняются в отдельные parquet (без base и interaction)
SAVEABLE_GROUPS = ["text", "idf", "clicks", "rubric", "org_info", "bert"]


def precompute_groups(groups, use_bert=False):
    """Посчитать и сохранить указанные группы."""
    data = load_all()
    train = data["train"]
    test = data["test"]
    idf_dict = build_idf_from_corpus(
        train["org_name"].tolist() + test["org_name"].tolist()
    )
    if "idf" in groups:
        with open(FEATURES_DIR / "idf_dict.pkl", "wb") as f:
            pickle.dump(idf_dict, f)

    aux = {
        "clicks_dict": data["clicks"],
        "org_info": data["org_info"],
        "rubric_info": data["rubric_info"],
        "idf_dict": idf_dict,
        "use_bert": use_bert,
    }

    for group in groups:
        if group not in SAVEABLE_GROUPS:
            continue
        print(f"[precompute] Группа {group}...")
        train_g = train.copy()
        test_g = test.copy()
        extract_feature_group(group, train_g, **aux)
        extract_feature_group(group, test_g, **aux)

        key_cols = ["query_id", "org_id"]
        feat_cols = [c for c in FEATURE_GROUPS[group] if c in train_g.columns]
        train_g[key_cols + feat_cols].to_parquet(
            FEATURES_DIR / f"{group}_train.parquet", index=False
        )
        test_g[key_cols + feat_cols].to_parquet(
            FEATURES_DIR / f"{group}_test.parquet", index=False
        )
        print(f"  Сохранено {FEATURES_DIR / f'{group}_train.parquet'}")


def load_precomputed_features():
    """
    Загрузить все предпосчитанные фичи, собранные по группам.
    Мержим по query_id, org_id. Interaction считаем на лету.
    """
    train_base = load_train()[["query_id", "org_id", "relevance", "region"]].copy()
    test_base = load_test()[["query_id", "org_id", "region"]].copy()
    train_base["region"] = train_base["region"].astype(int)
    test_base["region"] = test_base["region"].astype(int)

    for group in SAVEABLE_GROUPS:
        tr_path = FEATURES_DIR / f"{group}_train.parquet"
        te_path = FEATURES_DIR / f"{group}_test.parquet"
        if not tr_path.exists() or not te_path.exists():
            raise FileNotFoundError(
                f"Нет предпосчитанных фичей для группы {group}. "
                f"Запустите: python precompute.py [--only {group} ...]"
            )
        tr_g = pd.read_parquet(tr_path)
        te_g = pd.read_parquet(te_path)
        # убираем key cols из второго датафрейма при merge, чтобы не дублировать
        merge_cols = ["query_id", "org_id"]
        feat_cols = [c for c in tr_g.columns if c not in merge_cols]
        train_base = train_base.merge(tr_g[merge_cols + feat_cols], on=merge_cols)
        test_base = test_base.merge(te_g[merge_cols + feat_cols], on=merge_cols)

    # Interaction (зависит от jaccard, click_score, geo_distance)
    geo_close_tr = (train_base["geo_distance"] < 10).astype(float)
    geo_close_te = (test_base["geo_distance"] < 10).astype(float)
    train_base["jaccard_x_click"] = train_base["jaccard"] * train_base["click_score"]
    train_base["jaccard_x_geo_close"] = train_base["jaccard"] * geo_close_tr
    test_base["jaccard_x_click"] = test_base["jaccard"] * test_base["click_score"]
    test_base["jaccard_x_geo_close"] = test_base["jaccard"] * geo_close_te

    # num_clicks_rel (относительная популярность среди кандидатов запроса)
    for base in (train_base, test_base):
        max_clicks = base.groupby("query_id")["num_clicks_raw"].transform("max")
        base["num_clicks_rel"] = np.where(max_clicks > 0, base["num_clicks_raw"] / max_clicks, 0.0)

    return train_base, test_base


def has_modular_precomputed():
    """Проверить, есть ли precomputed по группам."""
    return (FEATURES_DIR / "text_train.parquet").exists()


def has_legacy_precomputed():
    """Проверить, есть ли старый монолитный precomputed."""
    return (FEATURES_DIR / "train_features.parquet").exists()


def main():
    parser = argparse.ArgumentParser(
        description="Предрасчёт фичей по группам. Каждая группа — отдельные файлы."
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        metavar="G1,G2,...",
        help="Пересчитать только эти группы: text, idf, clicks, rubric, org_info, bert",
    )
    parser.add_argument(
        "--no-bert",
        action="store_true",
        help="Не использовать BERT-фичи (быстрее, без сети)",
    )
    args = parser.parse_args()

    FEATURES_DIR.mkdir(exist_ok=True)

    if args.only:
        groups = [g.strip() for g in args.only.split(",") if g.strip()]
        invalid = [g for g in groups if g not in SAVEABLE_GROUPS]
        if invalid:
            raise SystemExit(f"Неизвестные группы: {invalid}. Доступны: {SAVEABLE_GROUPS}")
    else:
        groups = SAVEABLE_GROUPS

    precompute_groups(groups, use_bert=not args.no_bert)
    print("\nГотово. Загрузка: load_precomputed_features() или baseline --precomputed")


if __name__ == "__main__":
    main()
