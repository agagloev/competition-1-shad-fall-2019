# Ревизия фичей (36 шт.)

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
| `bert_dot` | Dot product (нормализован в [0,1]) |

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
