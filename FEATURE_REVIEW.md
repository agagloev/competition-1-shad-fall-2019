# Ревизия фичей (39 шт.)

> Последний пересчёт: `python feature_review.py`

---

## Топ-корреляции с relevance

| Фича | corr | mean | std | Инсайт |
|------|-----:|----:|----:|--------|
| rubric_overlap | +0.41 | 0.50 | 0.42 | **Самый сильный сигнал**: overlap query↔рубрики |
| word_count_query | −0.32 | 2.96 | 2.08 | Короткие запросы чаще релевантны |
| rubric_phrase_match | +0.32 | 0.61 | 0.49 | Фраза рубрики в query — сильный показатель |
| click_score | +0.31 | 0.19 | 0.28 | История кликов очень полезна |
| click_jaccard | +0.30 | 0.22 | 0.26 | |
| query_len | −0.29 | 21.3 | 15.9 | Длинные запросы → ниже relevance |
| geo_in_window | +0.27 | 0.69 | 0.46 | Org в окне поиска — важно |
| click_best_jaccard | +0.24 | 0.31 | 0.26 | |
| bert_cosine | +0.18 | 0.82 | 0.03 | BERT даёт сигнал, но std маленький |
| query_in_org | +0.18 | 0.07 | 0.25 | Редкий, но информативный |
| geo_distance | −0.15 | 318 | 976 | Чем дальше — тем хуже |
| jaccard_x_geo_close | +0.15 | 0.05 | 0.14 | Взаимодействие текст×гео полезно |

---

## Подозрительные фичи

| Фича | unique | zeros% | corr | Рекомендация |
|------|-------:|-------:|-----:|--------------|
| region_code_match | 2 | 96.7% | −0.06 | Почти всегда 0, можно исключить |
| has_work_intervals | 2 | 20.1% | +0.02 | Очень слабая связь с relevance |
| org_in_query | 2 | 89.6% | −0.02 | Отрицательная корр., рассмотреть удаление |
| org_in_query_str | 2 | 56.1% | −0.03 | Слабый сигнал |
| region | — | 0% | +0.04 | Выбросы 543x, лучше как категория (уже так) |

---

## Выбросы (max/median > 20)

| Фича | median | max | ratio | Инсайт |
|------|-------:|----:|------:|--------|
| geo_distance | 7.5 | 17613 | 2336x | log1p или clip(0, 500) смягчит |
| region | 213 | 115707 | 543x | Категориальная — ок |
| window_area | 0.06 | 10.5 | 175x | log1p уже есть |
| rubric_jaccard | 0.005 | 0.1 | 22x | Допустимо |

---

## Главные инсайты

1. **Рубрики — главный сигнал**: rubric_overlap и rubric_phrase_match дают самую сильную связь с relevance. Стоит добавить фичи на основе рубрик (IDF, топ-рубрика).
2. **Клики важны**: click_score, click_jaccard в топе. Имеет смысл относительная популярность (num_clicks / max в группе query).
3. **Короткие запросы лучше ранжируются**: word_count_query и query_len отрицательно коррелируют с relevance.
4. **Гео**: geo_in_window сильнее, чем geo_distance; возможна фича geo_closeness = 1/(1+distance).
5. **BERT**: std=0.03 при mean=0.82 — значения в узком диапазоне, но всё равно дают +0.18 корр.
6. **Кандидаты на удаление**: region_code_match (почти константа), has_work_intervals (слабый сигнал).
7. **geo_distance**: много выбросов; log1p или clip может помочь модели.
8. **relevance**: 52% нулей, mean=0.07 — сильный дисбаланс; метрика NDCG это частично сглаживает.

---

## По типам

### Query-only (только запрос)
| Фича | Описание |
|------|----------|
| `query_len` | Длина query в символах |
| `word_count_query` | Число слов в query |
| `avg_word_len_query` | Средняя длина слова |

### Doc / Org-only (только организация)
| Фича | Описание |
|------|----------|
| `org_len` | Длина org_name в символах |
| `word_count_org` | Число слов в org_name |
| `avg_word_len_org` | Средняя длина слова |
| `num_clicks` | log1p(число кликов по org) |
| `num_rubrics` | Число рубрик у org |
| `num_org_names` | Число альтернативных названий |
| `window_area` | log1p(площадь окна поиска) |

### Context (контекст поиска)
| Фича | Описание |
|------|----------|
| `region` | ID региона (целое) |
| `geo_distance` | Расстояние org до window_center (км) |
| `geo_in_window` | 1 если org внутри окна, иначе 0 |

### Query ↔ Doc (взаимодействие query и org)
| Фича | Описание |
|------|----------|
| `jaccard` | Jaccard(query_tokens, org_tokens) |
| `dice` | Dice coefficient |
| `overlap` | \|intersection\| / max(\|query\|, 1) |
| `cosine_sim` | Косинусная близость множеств |
| `bigram_jaccard` | Jaccard по символьным биграммам |
| `query_in_org` | 1 если query_tokens ⊆ org_tokens |
| `org_in_query` | 1 если org_tokens ⊆ query_tokens |
| `query_in_org_str` | 1 если query подстрока org_name |
| `org_in_query_str` | 1 если длинное слово org в query |
| `matching_words` | Число совпавших слов |
| `idf_overlap` | IDF-взвешенное пересечение (BM25-стиль) |

### Clicks (история кликов)
| Фича | Описание |
|------|----------|
| `click_score` | 1 если точное совпадение, 0.5 если substring, 0 иначе |
| `click_jaccard` | Jaccard(query, объединённые клики org) |
| `click_best_jaccard` | Макс Jaccard(query, отдельный клик org) |
| `has_any_click` | 1 если у org есть клики, иначе 0 |
| `click_match_freq` | log1p(частота совпавшего запроса в кликах org) |
| `click_idf_overlap` | IDF-взвешенный overlap query ↔ клики org |
| `num_clicks_rel` | num_clicks / max_clicks_in_query_group [0,1] |

### Rubric (рубрики организации)
| Фича | Описание |
|------|----------|
| `rubric_jaccard` | Jaccard(query, тексты рубрик) |
| `rubric_overlap` | overlap(query, rubric_texts) |
| `rubric_phrase_match` | 1 если фраза рубрики в query |

### Org info (доп. данные org)
| Фича | Описание |
|------|----------|
| `org_names_best_jaccard` | Max Jaccard(query, альтернативные названия) |
| `address_jaccard` | Jaccard(query, адрес org) |

### BERT (семантика)
| Фича | Описание |
|------|----------|
| `bert_cosine` | Cosine sim эмбеддингов query и org_name |

### Interaction (произведения)
| Фича | Описание |
|------|----------|
| `jaccard_x_click` | jaccard × click_score |
| `jaccard_x_geo_close` | jaccard × (geo_distance < 10 km) |

---

## Идеи новых фичей

### Query ↔ Doc (текст)
- **trigram_jaccard** — Jaccard по триграммам (устойчивее к опечаткам)
- **idf_max_match** — max IDF среди совпавших слов / max IDF query
- **exact_token_match** — 1 если хотя бы один токен query точно в org

### Query ↔ Address
- **address_overlap** — overlap(query, address tokens)
- **query_in_address** — 1 если query подстрока address

### Geo / Context
- **geo_closeness** — 1 / (1 + clip(geo_distance, 0, 500))
- **region_match** — 1 если region из query совпадает с region org (парсить query)
- **query_has_geo** — 1 если в query есть гео-паттерны (область, город, улица)

### Clicks
- **click_count_normalized** — num_clicks / max_clicks_in_query_group (относительная популярность)
- **has_any_click** — 1 если num_clicks > 0

### Rubric
- **rubric_idf_overlap** — IDF-weighted overlap query ↔ rubric texts
- **top_rubric_match** — match с самой «релевантной» рубрикой (по idf)

### Org
- **org_name_length_ratio** — len(org) / len(query)
- **name_variants_overlap** — overlap по всем вариантам названий

### Interactions
- **click_jaccard_x_idf** — click_jaccard × idf_overlap
- **rubric_x_jaccard** — rubric_jaccard × jaccard
