"""
Счёт фичей для Kaggle Competition 1 SHAD Fall 2019
"""

import re

import numpy as np
import pandas as pd


def tokenize(text):
    """Токенизация текста (слова в нижнем регистре)"""
    if pd.isna(text):
        return set()
    text = str(text).lower()
    tokens = re.findall(r"\b\w+\b", text)
    return set(tokens)


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
    }


def build_clicks_features(df, clicks_dict):
    """
    clicks_dict: org_id -> list of query strings
    Фичи: click_score (совпадение), num_clicks (популярность организации)
    """
    def calc_row(row):
        org_id = str(row["org_id"])
        query = str(row["query"]).strip().lower()
        if org_id not in clicks_dict:
            return 0.0, 0
        org_queries = [q.strip().lower() for q in clicks_dict[org_id]]
        num_clicks = len(org_queries)
        if query in org_queries:
            return 1.0, num_clicks
        for oq in org_queries:
            if query in oq or oq in query:
                return 0.5, num_clicks
        return 0.0, num_clicks

    result = df.apply(calc_row, axis=1)
    return [r[0] for r in result], [r[1] for r in result]


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


def build_rubric_features(df, org_info, rubric_info):
    """Совпадение запроса с текстами рубрик организации"""
    def calc(row):
        org_id = str(row["org_id"])
        query_tokens = tokenize(row["query"])
        if org_id not in org_info:
            return 0.0, 0.0, 0
        o = org_info[org_id]
        rubric_ids = o.get("rubrics", [])
        num_rubrics = len(rubric_ids)
        rubric_text = _get_rubric_texts(rubric_info, rubric_ids)
        if not rubric_text:
            return 0.0, 0.0, num_rubrics
        rubric_tokens = tokenize(rubric_text)
        jaccard = jaccard_sim(query_tokens, rubric_tokens)
        overlap = len(query_tokens & rubric_tokens) / max(len(query_tokens), 1)
        return jaccard, overlap, num_rubrics

    result = df.apply(calc, axis=1)
    return list(zip(*result))


def build_org_info_features(df, org_info):
    """Лучшее совпадение с вариантами названий, совпадение с адресом, гео-расстояние"""
    def calc(row):
        org_id = str(row["org_id"])
        query_tokens = tokenize(row["query"])
        names, addr = _get_org_texts(org_info, org_id)
        best_jaccard = 0.0
        for n in names:
            best_jaccard = max(best_jaccard, jaccard_sim(query_tokens, tokenize(n)))
        addr_jaccard = jaccard_sim(query_tokens, tokenize(addr)) if addr else 0.0

        lon_w, lat_w = _parse_coords(row.get("window_center"))
        if org_id in org_info and "address" in org_info[org_id]:
            pos = org_info[org_id]["address"].get("pos", {})
            coords = pos.get("coordinates", [])
            if len(coords) >= 2 and lon_w is not None and lat_w is not None:
                lon_o, lat_o = float(coords[0]), float(coords[1])
                dist = ((lon_w - lon_o) ** 2 + (lat_w - lat_o) ** 2) ** 0.5
                return best_jaccard, addr_jaccard, dist, len(names)
        return best_jaccard, addr_jaccard, 999.0, len(names)

    result = df.apply(calc, axis=1)
    return list(zip(*result))


FEATURE_COLS = [
    "jaccard", "dice", "overlap", "query_in_org", "org_in_query",
    "query_in_org_str", "org_in_query_str", "query_len", "org_len",
    "click_score", "num_clicks", "region",
    "rubric_jaccard", "rubric_overlap", "num_rubrics",
    "org_names_best_jaccard", "address_jaccard", "geo_distance", "num_org_names",
]


def extract_features(df, clicks_dict=None, org_info=None, rubric_info=None):
    """Извлечение всех фичей"""
    # Текстовые фичи (query vs org_name)
    text_feats = df.apply(extract_text_features, axis=1)
    feat_names = [
        "jaccard", "dice", "overlap", "query_in_org", "org_in_query",
        "query_in_org_str", "org_in_query_str", "query_len", "org_len"
    ]
    for name in feat_names:
        df[name] = [f[name] for f in text_feats]

    # Clicks фичи
    if clicks_dict is not None:
        click_scores, num_clicks = build_clicks_features(df, clicks_dict)
        df["click_score"] = click_scores
        df["num_clicks"] = np.log1p(num_clicks)
    else:
        df["click_score"] = 0.0
        df["num_clicks"] = 0.0

    # Rubric фичи
    if org_info is not None and rubric_info is not None:
        jaccard_r, overlap_r, num_rubrics = build_rubric_features(df, org_info, rubric_info)
        df["rubric_jaccard"] = jaccard_r
        df["rubric_overlap"] = overlap_r
        df["num_rubrics"] = num_rubrics
    else:
        df["rubric_jaccard"] = 0.0
        df["rubric_overlap"] = 0.0
        df["num_rubrics"] = 0

    # Org info фичи
    if org_info is not None:
        best_jaccard, addr_jaccard, geo_dist, num_names = build_org_info_features(df, org_info)
        df["org_names_best_jaccard"] = best_jaccard
        df["address_jaccard"] = addr_jaccard
        df["geo_distance"] = geo_dist
        df["num_org_names"] = num_names
    else:
        df["org_names_best_jaccard"] = 0.0
        df["address_jaccard"] = 0.0
        df["geo_distance"] = 999.0
        df["num_org_names"] = 0

    df["region"] = df["region"].astype(int)
    return df
