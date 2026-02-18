"""
Счёт фичей для Kaggle Competition 1 SHAD Fall 2019
"""

import re

import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()

# Маппинг region_code -> типичные слова в запросах
REGION_QUERY_WORDS = {
    "RU": {"россия", "рф", "российск"},
    "UA": {"украина", "украинск"},
    "BY": {"белоруссия", "беларусь", "белорусск"},
    "TR": {"турция", "турецк"},
    "FR": {"франция", "французск"},
}


def tokenize(text):
    """Токенизация текста (слова в нижнем регистре)"""
    if pd.isna(text):
        return set()
    text = str(text).lower()
    tokens = re.findall(r"\b\w+\b", text)
    return set(tokens)


def tokenize_list(text):
    """Токенизация с сохранением порядка (список)"""
    if pd.isna(text):
        return []
    text = str(text).lower()
    return re.findall(r"\b\w+\b", text)


def build_idf_from_corpus(corpus, min_df=2):
    """Строит IDF по корпусу (список строк). Возвращает dict: word -> idf."""
    from collections import Counter
    doc_count = Counter()
    n_docs = 0
    for doc in corpus:
        tokens = tokenize(doc)
        for t in tokens:
            doc_count[t] += 1
        n_docs += 1
    idf = {}
    for w, df in doc_count.items():
        if df >= min_df:
            idf[w] = np.log((n_docs + 1) / (df + 1)) + 1
    return idf


def idf_weighted_overlap(query_tokens, org_tokens, idf_dict):
    """IDF-взвешенное пересечение: sum(idf[matching]) / sum(idf[query])"""
    if not query_tokens or not idf_dict:
        return 0.0
    query_idf = sum(idf_dict.get(t, 1.0) for t in query_tokens)
    if query_idf < 1e-9:
        return 0.0
    match_idf = sum(idf_dict.get(t, 1.0) for t in (query_tokens & org_tokens))
    return match_idf / query_idf


def ngram_tokens(text, n=2):
    """Символьные n-граммы"""
    if pd.isna(text) or len(text) < n:
        return set()
    text = str(text).lower()
    return set(text[i:i + n] for i in range(len(text) - n + 1))


def cosine_sim(set1, set2):
    """Косинусная близость (для множеств слов)"""
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    return inter / (len(set1) ** 0.5 * len(set2) ** 0.5)


def jaccard_sim(set1, set2):
    """Коэффициент Жаккара"""
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def dice_coef(set1, set2):
    """Коэффициент Дайса"""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    return 2 * intersection / (len(set1) + len(set2))


def _extract_nested_value(obj):
    """Извлечь текстовое value из вложенных структур JSON"""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        if "value" in obj:
            return _extract_nested_value(obj["value"])
        return ""
    if isinstance(obj, list):
        return " ".join(_extract_nested_value(x) for x in obj)
    return ""


def extract_text_features(row):
    """Базовые текстовые фичи из query и org_name"""
    query = str(row["query"]).lower()
    org_name = str(row["org_name"]).lower()
    query_tokens = tokenize(query)
    org_tokens = tokenize(org_name)

    jaccard = jaccard_sim(query_tokens, org_tokens)
    dice = dice_coef(query_tokens, org_tokens)
    overlap = len(query_tokens & org_tokens) / max(len(query_tokens), 1)
    query_in_org = 1.0 if query_tokens.issubset(org_tokens) else 0.0
    org_in_query = 1.0 if org_tokens.issubset(query_tokens) else 0.0

    query_in_org_str = 1.0 if query in org_name else 0.0
    org_in_query_str = 1.0 if any(t in query for t in org_tokens if len(t) > 2) else 0.0

    # Доп. текстовые
    matching_words = len(query_tokens & org_tokens)
    word_count_query = len(query_tokens)
    word_count_org = len(org_tokens)
    cos_sim = cosine_sim(query_tokens, org_tokens)
    q_bigrams = ngram_tokens(query, 2)
    o_bigrams = ngram_tokens(org_name, 2)
    bigram_jaccard = jaccard_sim(q_bigrams, o_bigrams) if (q_bigrams and o_bigrams) else 0.0
    # Первая часть query до запятой — основной интент (тип, "суд", "ресторан")
    query_main = query.split(",")[0].strip() if "," in query else query
    main_tokens = tokenize(query_main)
    query_main_jaccard = jaccard_sim(main_tokens, org_tokens) if (main_tokens and org_tokens) else 0.0
    avg_word_len_query = np.mean([len(t) for t in query_tokens]) if query_tokens else 0.0
    avg_word_len_org = np.mean([len(t) for t in org_tokens]) if org_tokens else 0.0

    return {
        "jaccard": jaccard,
        "dice": dice,
        "overlap": overlap,
        "query_in_org": query_in_org,
        "org_in_query": org_in_query,
        "query_in_org_str": query_in_org_str,
        "org_in_query_str": org_in_query_str,
        "query_len": len(query),
        "org_len": len(org_name),
        "matching_words": matching_words,
        "word_count_query": word_count_query,
        "word_count_org": word_count_org,
        "cosine_sim": cos_sim,
        "bigram_jaccard": bigram_jaccard,
        "query_main_jaccard": query_main_jaccard,
        "avg_word_len_query": avg_word_len_query,
        "avg_word_len_org": avg_word_len_org,
    }


def build_idf_features(df, idf_dict):
    """IDF-взвешенное совпадение query и org_name (BM25-стиль)"""
    def calc(row):
        qt = tokenize(row["query"])
        ot = tokenize(row["org_name"])
        return idf_weighted_overlap(qt, ot, idf_dict)

    return df.apply(calc, axis=1).tolist()


def build_click_idf(clicks_dict, min_df=2):
    """
    IDF по корпусу кликов. Каждая организация — один документ (объединённый текст всех её кликов).
    Возвращает dict: word -> idf. Редкие слова получают больший вес.
    """
    corpus = []
    for org_id, queries in clicks_dict.items():
        text = " ".join(q.strip().lower() for q in queries)
        corpus.append(text)
    return build_idf_from_corpus(corpus, min_df=min_df)


def build_clicks_features(df, clicks_dict, click_idf_dict=None):
    """
    clicks_dict: org_id -> list of query strings (могут повторяться)
    click_idf_dict: optional, для click_idf_overlap
    Возвращает: click_score, num_clicks_raw, click_jaccard, click_best_jaccard,
                has_any_click, click_match_freq, click_idf_overlap
    """
    from collections import Counter

    def calc_row(row):
        org_id = str(row["org_id"])
        query = str(row["query"]).strip().lower()
        query_tokens = tokenize(query)
        if org_id not in clicks_dict:
            return 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0
        org_queries = [q.strip().lower() for q in clicks_dict[org_id]]
        oq_counts = Counter(org_queries)
        num_clicks_raw = len(org_queries)
        has_any_click = 1.0
        match_freq = 0

        all_click_tokens = tokenize(" ".join(org_queries))
        click_idf = 0.0
        if click_idf_dict and all_click_tokens:
            click_idf = idf_weighted_overlap(query_tokens, all_click_tokens, click_idf_dict)

        if query in oq_counts:
            match_freq = oq_counts[query]
            return 1.0, num_clicks_raw, 1.0, 1.0, has_any_click, float(match_freq), click_idf

        best_oq = None
        for oq in oq_counts:
            if query in oq or oq in query:
                if best_oq is None or oq_counts[oq] > oq_counts.get(best_oq, 0):
                    best_oq = oq
        if best_oq is not None:
            match_freq = oq_counts[best_oq]
            return 0.5, num_clicks_raw, 0.5, 0.5, has_any_click, float(match_freq), click_idf

        click_jaccard = jaccard_sim(query_tokens, all_click_tokens) if all_click_tokens else 0.0
        best_j = 0.0
        for oq in oq_counts:
            best_j = max(best_j, jaccard_sim(query_tokens, tokenize(oq)))
        return 0.0, num_clicks_raw, click_jaccard, best_j, has_any_click, 0.0, click_idf

    result = df.progress_apply(calc_row, axis=1)
    return (
        [r[0] for r in result],
        [r[1] for r in result],
        [r[2] for r in result],
        [r[3] for r in result],
        [r[4] for r in result],
        [np.log1p(r[5]) for r in result],
        [r[6] for r in result],
    )


def _get_rubric_texts(rubric_info, rubric_ids):
    """Собрать все тексты (keywords, phrases) для списка rubric_ids"""
    texts = []
    for rid in rubric_ids:
        rid_str = str(rid)
        if rid_str not in rubric_info:
            continue
        r = rubric_info[rid_str]
        for key in ["keywords", "phrases", "descriptions", "names"]:
            if key not in r:
                continue
            for item in r[key]:
                val = _extract_nested_value(item)
                if val:
                    texts.append(str(val).lower())
    return " ".join(texts)


def _get_org_texts(org_info, org_id):
    """Собрать все названия и адрес организации"""
    org_id_str = str(org_id)
    if org_id_str not in org_info:
        return [], ""
    o = org_info[org_id_str]
    names = []
    for n in o.get("names", []):
        v = _extract_nested_value(n)
        if v:
            names.append(v.lower())
    addr = ""
    if "address" in o and "formatted" in o["address"]:
        fmt = o["address"]["formatted"]
        if isinstance(fmt, dict) and "value" in fmt:
            addr = str(fmt["value"]).lower()
        elif isinstance(fmt, str):
            addr = fmt.lower()
    return names, addr


def _parse_coords(coord_str):
    """Парсинг 'lon,lat' в float tuple"""
    if pd.isna(coord_str):
        return None, None
    parts = str(coord_str).strip().split(",")
    if len(parts) >= 2:
        try:
            return float(parts[0].strip()), float(parts[1].strip())
        except (ValueError, TypeError):
            pass
    return None, None


def _get_rubric_phrases(rubric_info, rubric_ids):
    """Список отдельных фраз/ключевых слов рубрик"""
    phrases = []
    for rid in rubric_ids:
        rid_str = str(rid)
        if rid_str not in rubric_info:
            continue
        r = rubric_info[rid_str]
        for key in ["keywords", "phrases", "names"]:
            if key not in r:
                continue
            for item in r[key]:
                val = _extract_nested_value(item)
                if val and len(str(val).strip()) > 1:
                    phrases.append(str(val).strip().lower())
    return phrases


def build_rubric_features(df, org_info, rubric_info):
    """Совпадение запроса с текстами рубрик + phrase match"""
    def calc(row):
        org_id = str(row["org_id"])
        query = str(row["query"]).lower()
        query_tokens = tokenize(query)
        if org_id not in org_info:
            return 0.0, 0.0, 0, 0.0
        o = org_info[org_id]
        rubric_ids = o.get("rubrics", [])
        num_rubrics = len(rubric_ids)
        rubric_text = _get_rubric_texts(rubric_info, rubric_ids)
        phrases = _get_rubric_phrases(rubric_info, rubric_ids)
        phrase_match = 1.0 if any(p in query or query in p for p in phrases if len(p) > 2) else 0.0
        if not rubric_text:
            return 0.0, 0.0, num_rubrics, phrase_match
        rubric_tokens = tokenize(rubric_text)
        jaccard = jaccard_sim(query_tokens, rubric_tokens)
        overlap = len(query_tokens & rubric_tokens) / max(len(query_tokens), 1)
        return jaccard, overlap, num_rubrics, phrase_match

    result = df.progress_apply(calc, axis=1)
    return [r[0] for r in result], [r[1] for r in result], [r[2] for r in result], [r[3] for r in result]


def _haversine_km(lon1, lat1, lon2, lat2):
    """Расстояние в км между двумя точками (приближённо)"""
    R = 6371
    lat1, lat2, lon1, lon2 = map(np.radians, [lat1, lat2, lon1, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_org_info_features(df, org_info):
    """Названия, адрес, гео, window, region_code_match, has_work_intervals"""
    def calc(row):
        org_id = str(row["org_id"])
        query_tokens = tokenize(row["query"])
        names, addr = _get_org_texts(org_info, org_id)
        best_jaccard = 0.0
        for n in names:
            best_jaccard = max(best_jaccard, jaccard_sim(query_tokens, tokenize(n)))
        addr_jaccard = jaccard_sim(query_tokens, tokenize(addr)) if addr else 0.0

        # region_code_match: query содержит страну, совпадающую с org
        region_code_match = 0.0
        has_work_intervals = 0.0
        if org_id in org_info:
            o = org_info[org_id]
            rc = o.get("address", {}).get("region_code", "")
            if rc and rc in REGION_QUERY_WORDS:
                region_words = REGION_QUERY_WORDS[rc]
                if any(t == w or t.startswith(w) for t in query_tokens for w in region_words):
                    region_code_match = 1.0
            wi = o.get("work_intervals", [])
            has_work_intervals = 1.0 if wi else 0.0

        lon_w, lat_w = _parse_coords(row.get("window_center"))
        dw_lon, dw_lat = _parse_coords(row.get("window_size"))
        window_area = dw_lon * dw_lat if (dw_lon and dw_lat) else 0.001

        dist = 999.0
        geo_in_window = 0.0
        if org_id in org_info and "address" in org_info[org_id]:
            pos = org_info[org_id]["address"].get("pos", {})
            coords = pos.get("coordinates", [])
            if len(coords) >= 2 and lon_w is not None and lat_w is not None:
                lon_o, lat_o = float(coords[0]), float(coords[1])
                dist = _haversine_km(lon_w, lat_w, lon_o, lat_o)
                window_radius = max(dw_lon or 0.1, dw_lat or 0.1) * 111  # градусы -> км
                geo_in_window = 1.0 if dist < window_radius else 0.0
        return best_jaccard, addr_jaccard, dist, len(names), window_area, geo_in_window, region_code_match, has_work_intervals

    result = df.apply(calc, axis=1)
    return (
        [r[0] for r in result], [r[1] for r in result], [r[2] for r in result],
        [r[3] for r in result], [r[4] for r in result], [r[5] for r in result],
        [r[6] for r in result], [r[7] for r in result],
    )


FEATURE_COLS = [
    "jaccard", "dice", "overlap", "query_in_org", "org_in_query",
    "query_in_org_str", "org_in_query_str", "query_len", "org_len",
    "matching_words", "word_count_query", "word_count_org",     "cosine_sim", "bigram_jaccard", "query_main_jaccard",
    "avg_word_len_query", "avg_word_len_org",
    "idf_overlap",
    "click_score", "num_clicks", "click_jaccard", "click_best_jaccard",
    "has_any_click", "click_match_freq", "click_idf_overlap", "num_clicks_rel",
    "region",
    "rubric_jaccard", "rubric_overlap", "num_rubrics", "rubric_phrase_match",
    "org_names_best_jaccard", "address_jaccard", "geo_distance", "num_org_names",
    "window_area", "geo_in_window", "region_code_match", "has_work_intervals",
    "jaccard_x_click", "jaccard_x_geo_close",
    "bert_cosine", "bert_click_cosine",
]

# Группы фичей для инкрементального precompute (--only text обновит только text_*.parquet)
FEATURE_GROUPS = {
    "text": [
        "jaccard", "dice", "overlap", "query_in_org", "org_in_query",
        "query_in_org_str", "org_in_query_str", "query_len", "org_len",
        "matching_words", "word_count_query", "word_count_org", "cosine_sim", "bigram_jaccard",
        "query_main_jaccard", "avg_word_len_query", "avg_word_len_org",
    ],
    "idf": ["idf_overlap"],
    "clicks": [
        "click_score", "num_clicks", "click_jaccard", "click_best_jaccard",
        "has_any_click", "click_match_freq", "click_idf_overlap",
        "num_clicks_raw",  # для num_clicks_rel при загрузке
    ],
    "rubric": ["rubric_jaccard", "rubric_overlap", "num_rubrics", "rubric_phrase_match"],
    "org_info": [
        "org_names_best_jaccard", "address_jaccard", "geo_distance", "num_org_names",
        "window_area", "geo_in_window", "region_code_match", "has_work_intervals",
    ],
    "base": ["region"],  # из raw data
    "interaction": ["jaccard_x_click", "jaccard_x_geo_close"],  # считаются при загрузке
    "bert": ["bert_cosine", "bert_click_cosine"],
}


def extract_features(
    df,
    clicks_dict=None,
    org_info=None,
    rubric_info=None,
    idf_dict=None,
    use_bert=False,
):
    """Извлечение всех фичей. idf_dict — из build_idf_from_corpus(train["org_name"])."""
    # Текстовые фичи (query vs org_name)
    text_feats = df.apply(extract_text_features, axis=1)
    text_feat_names = [
        "jaccard", "dice", "overlap", "query_in_org", "org_in_query",
        "query_in_org_str", "org_in_query_str", "query_len", "org_len",
        "matching_words", "word_count_query", "word_count_org", "cosine_sim", "bigram_jaccard",
        "query_main_jaccard", "avg_word_len_query", "avg_word_len_org",
    ]
    for name in text_feat_names:
        df[name] = [f[name] for f in text_feats]

    # IDF-взвешенная фича (BM25-стиль)
    if idf_dict is not None:
        df["idf_overlap"] = build_idf_features(df, idf_dict)
    else:
        df["idf_overlap"] = 0.0

    # Clicks фичи
    if clicks_dict is not None:
        print("[features] Clicks фичи...")
        click_idf_dict = build_click_idf(clicks_dict)
        c0, c1, c2, c3, c4, c5, c6 = build_clicks_features(df, clicks_dict, click_idf_dict)
        df["click_score"] = c0
        df["num_clicks_raw"] = c1
        df["num_clicks"] = np.log1p(c1)
        df["click_jaccard"] = c2
        df["click_best_jaccard"] = c3
        df["has_any_click"] = c4
        df["click_match_freq"] = c5
        df["click_idf_overlap"] = c6
        max_clicks = df.groupby("query_id")["num_clicks_raw"].transform("max")
        df["num_clicks_rel"] = np.where(max_clicks > 0, df["num_clicks_raw"] / max_clicks, 0.0)
    else:
        df["click_score"] = 0.0
        df["num_clicks"] = 0.0
        df["click_jaccard"] = 0.0
        df["click_best_jaccard"] = 0.0
        df["has_any_click"] = 0.0
        df["click_match_freq"] = 0.0
        df["click_idf_overlap"] = 0.0
        df["num_clicks_rel"] = 0.0

    # Rubric фичи
    if org_info is not None and rubric_info is not None:
        print("[features] Rubric фичи...")
        r0, r1, r2, r3 = build_rubric_features(df, org_info, rubric_info)
        df["rubric_jaccard"] = r0
        df["rubric_overlap"] = r1
        df["num_rubrics"] = r2
        df["rubric_phrase_match"] = r3
    else:
        df["rubric_jaccard"] = 0.0
        df["rubric_overlap"] = 0.0
        df["num_rubrics"] = 0
        df["rubric_phrase_match"] = 0.0

    # Org info фичи
    if org_info is not None:
        o0, o1, o2, o3, o4, o5, o6, o7 = build_org_info_features(df, org_info)
        df["org_names_best_jaccard"] = o0
        df["address_jaccard"] = o1
        df["geo_distance"] = o2
        df["num_org_names"] = o3
        df["window_area"] = np.log1p(o4)
        df["geo_in_window"] = o5
        df["region_code_match"] = o6
        df["has_work_intervals"] = o7
    else:
        df["org_names_best_jaccard"] = 0.0
        df["address_jaccard"] = 0.0
        df["geo_distance"] = 999.0
        df["num_org_names"] = 0
        df["window_area"] = 0.0
        df["geo_in_window"] = 0.0
        df["region_code_match"] = 0.0
        df["has_work_intervals"] = 0.0

    df["region"] = df["region"].astype(int)

    # Interaction фичи
    geo_close = (df["geo_distance"] < 10).astype(float)
    df["jaccard_x_click"] = df["jaccard"] * df["click_score"]
    df["jaccard_x_geo_close"] = df["jaccard"] * geo_close

    # BERT фичи (опционально, долго)
    if use_bert:
        print("[features] BERT фичи (загрузка модели + encode)...")
        from .bert_features import build_bert_features
        if clicks_dict is not None:
            bc, bcc = build_bert_features(df, clicks_dict=clicks_dict)
            df["bert_cosine"] = bc
            df["bert_click_cosine"] = bcc
        else:
            df["bert_cosine"] = build_bert_features(df)
            df["bert_click_cosine"] = 0.0
    else:
        df["bert_cosine"] = 0.0
        df["bert_click_cosine"] = 0.0

    return df


def extract_feature_group(group, df, clicks_dict=None, org_info=None, rubric_info=None, idf_dict=None, use_bert=False):
    """
    Извлечь только одну группу фичей. Для инкрементального precompute.
    Возвращает df с добавленными колонками этой группы.
    """
    if group == "text":
        text_feats = df.apply(extract_text_features, axis=1)
        for name in FEATURE_GROUPS["text"]:
            df[name] = [f[name] for f in text_feats]
    elif group == "idf":
        if idf_dict is not None:
            df["idf_overlap"] = build_idf_features(df, idf_dict)
        else:
            df["idf_overlap"] = 0.0
    elif group == "clicks":
        if clicks_dict is not None:
            click_idf_dict = build_click_idf(clicks_dict)
            c0, c1, c2, c3, c4, c5, c6 = build_clicks_features(df, clicks_dict, click_idf_dict)
            df["click_score"] = c0
            df["num_clicks_raw"] = c1
            df["num_clicks"] = np.log1p(c1)
            df["click_jaccard"] = c2
            df["click_best_jaccard"] = c3
            df["has_any_click"] = c4
            df["click_match_freq"] = c5
            df["click_idf_overlap"] = c6
            max_clicks = df.groupby("query_id")["num_clicks_raw"].transform("max")
            df["num_clicks_rel"] = np.where(max_clicks > 0, df["num_clicks_raw"] / max_clicks, 0.0)
        else:
            for c in FEATURE_GROUPS["clicks"]:
                df[c] = 0.0
            df["num_clicks_rel"] = 0.0
    elif group == "rubric":
        if org_info is not None and rubric_info is not None:
            r0, r1, r2, r3 = build_rubric_features(df, org_info, rubric_info)
            df["rubric_jaccard"] = r0
            df["rubric_overlap"] = r1
            df["num_rubrics"] = r2
            df["rubric_phrase_match"] = r3
        else:
            df["rubric_jaccard"] = 0.0
            df["rubric_overlap"] = 0.0
            df["num_rubrics"] = 0
            df["rubric_phrase_match"] = 0.0
    elif group == "org_info":
        if org_info is not None:
            o0, o1, o2, o3, o4, o5, o6, o7 = build_org_info_features(df, org_info)
            df["org_names_best_jaccard"] = o0
            df["address_jaccard"] = o1
            df["geo_distance"] = o2
            df["num_org_names"] = o3
            df["window_area"] = np.log1p(o4)
            df["geo_in_window"] = o5
            df["region_code_match"] = o6
            df["has_work_intervals"] = o7
        else:
            for c in FEATURE_GROUPS["org_info"]:
                df[c] = 0.0 if c != "geo_distance" else 999.0
    elif group == "base":
        df["region"] = df["region"].astype(int)
    elif group == "interaction":
        geo_close = (df["geo_distance"] < 10).astype(float)
        df["jaccard_x_click"] = df["jaccard"] * df["click_score"]
        df["jaccard_x_geo_close"] = df["jaccard"] * geo_close
    elif group == "bert":
        if use_bert:
            from .bert_features import build_bert_features
            if clicks_dict is not None:
                bc, bcc = build_bert_features(df, clicks_dict=clicks_dict)
                df["bert_cosine"] = bc
                df["bert_click_cosine"] = bcc
            else:
                df["bert_cosine"] = build_bert_features(df)
                df["bert_click_cosine"] = 0.0
        else:
            df["bert_cosine"] = 0.0
            df["bert_click_cosine"] = 0.0
    else:
        raise ValueError(f"Unknown group: {group}")
    return df
