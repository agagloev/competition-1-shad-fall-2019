# Kaggle Competition 1 SHAD Fall 2019

https://www.kaggle.com/competitions/competition-1-shad-fall-2019/overview
Ранжирование организаций по релевантности поискового запроса. CatBoost Ranker (LambdaMart) + предпосчитанные фичи.

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

1. Скачать данные в `data/` (train.csv, test.csv, *_information.json)
2. Предпосчитать фичи: `python -m scripts.precompute`
3. Обучиться и сохранить submission: `python -m scripts.train`

## Структура

- `src/` — модули (data, features, bert_features, train, metrics, precompute)
- `scripts/` — точки входа (train, precompute)
- `precomputed/` — parquet-фичи (не в git)
