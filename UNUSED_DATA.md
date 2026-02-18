# Неиспользуемые данные

## org_info (train_org_information, test_org_information)

| Поле | Используется | Описание |
|------|--------------|----------|
| names | ✓ | Альтернативные названия — org_names_best_jaccard |
| rubrics | ✓ | Связь с rubric_info |
| address.formatted | ✓ | Адрес — address_jaccard |
| address.pos.coordinates | ✓ | Гео — geo_distance, geo_in_window |
| **address.region_code** | ✗ | RU, UA, TR, BY, FR — страна организации |
| **address.geo_id** | ✗ | Числовой ID региона (2, 969, 213...) |
| **work_intervals** | ✗ | Расписание: day, time_minutes_begin, time_minutes_end |

## Query структура (train.csv)

Запросы часто имеют формат: `"тип, страна, регион, город"`
- Пример: "суд, Украина, Днепропетровская область, Днепродзержинский городской совет"
- Первая часть — тип (суд, ресторан, школа)
- Остальное — геолокация

## Идеи новых фичей

1. **region_code_match** — query содержит страну, совпадающую с org's region_code (Украина↔UA, Россия↔RU)
2. **has_work_intervals** — у org есть расписание (0/1)
3. **query_main_term** — первая часть до запятой, jaccard с org_name (основной интент)
