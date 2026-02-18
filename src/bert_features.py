"""
BERT-эмбеддинги для фичей query vs org_name и query vs клики.
Использует intfloat/multilingual-e5-base (query/passage prefixes для retrieval).
"""

import numpy as np
import pandas as pd


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусная близость между двумя векторами"""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_bert_features(
    df: pd.DataFrame,
    model_name: str = "intfloat/multilingual-e5-base",
    batch_size: int = 64,
    clicks_dict=None,
):
    """
    Строит BERT-фичи: bert_cosine (query vs org_name) и опционально bert_click_cosine.
    bert_click_cosine = max(cosine(query, click_i)) по всем клик-запросам org.
    E5 использует "query: " / "passage: " префиксы.
    Возвращает bert_cosine, или (bert_cosine, bert_click_cosine) если clicks_dict задан.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")

    # Query vs org_name
    queries_raw = df["query"].fillna("").astype(str).tolist()
    org_names_raw = df["org_name"].fillna("").astype(str).tolist()
    queries = ["query: " + (q[:500] if len(q) > 500 else q) for q in queries_raw]
    org_names = ["passage: " + (o[:500] if len(o) > 500 else o) for o in org_names_raw]
    q_embs = model.encode(queries, batch_size=batch_size, show_progress_bar=True)
    o_embs = model.encode(org_names, batch_size=batch_size, show_progress_bar=True)
    bert_cosine = [_cosine_sim(q, o) for q, o in zip(q_embs, o_embs)]

    if clicks_dict is None:
        return bert_cosine

    # bert_click_cosine = max similarity между текущим запросом и каждым клик-запросом org
    unique_queries = list(df["query"].fillna("").astype(str).str.strip().str.lower().unique())
    if unique_queries:
        q_prefixed = ["query: " + (q[:500] if len(q) > 500 else q) for q in unique_queries]
        q_click_embs = model.encode(q_prefixed, batch_size=batch_size, show_progress_bar=True)
        query_to_emb = dict(zip(unique_queries, q_click_embs))
    else:
        query_to_emb = {}

    org_ids_with_clicks = [str(oid) for oid in df["org_id"].unique() if str(oid) in clicks_dict]
    org_to_click_embs = {}
    if org_ids_with_clicks:
        all_texts = []
        org_ranges = {}
        for oid in org_ids_with_clicks:
            unique_clicks = list(set(t.strip().lower() for t in clicks_dict[oid]))
            start = len(all_texts)
            prefixed = ["passage: " + (t[:500] if len(t) > 500 else t) for t in unique_clicks]
            all_texts.extend(prefixed)
            org_ranges[oid] = (start, len(prefixed))
        if all_texts:
            all_embs = model.encode(all_texts, batch_size=batch_size, show_progress_bar=True)
            for oid, (start, count) in org_ranges.items():
                org_to_click_embs[oid] = all_embs[start : start + count]

    dim = model.get_sentence_embedding_dimension()
    zero_emb = np.zeros(dim, dtype=np.float32)
    q_strs = df["query"].fillna("").astype(str).str.strip().str.lower().tolist()
    oids = df["org_id"].astype(str).tolist()
    result = []
    for q_str, oid in zip(q_strs, oids):
        if oid not in org_to_click_embs:
            result.append(0.0)
        else:
            q_emb = query_to_emb.get(q_str, zero_emb)
            max_sim = max(_cosine_sim(q_emb, ce) for ce in org_to_click_embs[oid])
            result.append(float(max_sim))
    return bert_cosine, result
