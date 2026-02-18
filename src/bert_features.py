"""
BERT-эмбеддинги для фичей query vs org_name и query vs клики.
Использует intfloat/multilingual-e5-base (query/passage prefixes для retrieval).
"""

import numpy as np
import pandas as pd


def _extract_nested_value(obj):
    """Извлечь текстовое value из вложенных структур org_info JSON."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        if "value" in obj:
            return _extract_nested_value(obj["value"])
        return ""
    if isinstance(obj, list):
        return " ".join(_extract_nested_value(x) for x in obj)
    return ""


def _get_org_names(org_info, org_id):
    """Список всех названий организации из org_info."""
    oid = str(org_id)
    if oid not in org_info:
        return []
    names = []
    for n in org_info[oid].get("names", []):
        v = _extract_nested_value(n)
        if v and str(v).strip():
            names.append(str(v).strip().lower())
    return names


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
    org_info=None,
):
    """
    Строит BERT-фичи:
    - bert_cosine: query vs org_name (из df)
    - bert_click_cosine: max(cosine(query, click_i)) по клик-запросам
    - bert_click_min: min(cosine(query, click_i)) по клик-запросам
    - bert_names_max_sim, bert_names_min_sim: max/min cosine query vs все названия org (из org_info)
    E5 использует "query: " / "passage: " префиксы.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # T4/GPU: batch 256 даёт прирост; FP16 — ещё ~1.5–2x
    eff_batch = 256 if device == "cuda" else batch_size
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model = model.half()

    # Query vs org_name; normalize_embeddings: cosine = dot product
    queries_raw = df["query"].fillna("").astype(str).tolist()
    org_names_raw = df["org_name"].fillna("").astype(str).tolist()
    queries = ["query: " + (q[:500] if len(q) > 500 else q) for q in queries_raw]
    org_names = ["passage: " + (o[:500] if len(o) > 500 else o) for o in org_names_raw]

    use_tensor = device == "cuda"
    q_embs = model.encode(
        queries, batch_size=eff_batch, show_progress_bar=True,
        normalize_embeddings=True, convert_to_tensor=use_tensor,
    )
    o_embs = model.encode(
        org_names, batch_size=eff_batch, show_progress_bar=True,
        normalize_embeddings=True, convert_to_tensor=use_tensor,
    )
    if use_tensor:
        # Batch cosine на GPU (normalized -> dot = cosine)
        bert_cosine = (q_embs * o_embs).sum(dim=1).cpu().numpy().tolist()
        q_embs = q_embs.cpu().numpy()  # для секции names
    else:
        bert_cosine = [_cosine_sim(q, o) for q, o in zip(q_embs, o_embs)]

    if clicks_dict is None:
        click_max_list = [0.0] * len(df)
        click_min_list = [0.0] * len(df)
    else:
        unique_queries = list(df["query"].fillna("").astype(str).str.strip().str.lower().unique())
        if unique_queries:
            q_prefixed = ["query: " + (q[:500] if len(q) > 500 else q) for q in unique_queries]
            q_click_embs = model.encode(
                q_prefixed, batch_size=eff_batch, show_progress_bar=True,
                normalize_embeddings=True,
            )
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
                all_embs = model.encode(
                    all_texts, batch_size=eff_batch, show_progress_bar=True,
                    normalize_embeddings=True,
                )
                for oid, (start, count) in org_ranges.items():
                    org_to_click_embs[oid] = all_embs[start : start + count]

        dim = model.get_sentence_embedding_dimension()
        zero_emb = np.zeros(dim, dtype=np.float32)
        q_strs = df["query"].fillna("").astype(str).str.strip().str.lower().tolist()
        oids = df["org_id"].astype(str).tolist()
        click_max_list = []
        click_min_list = []
        for q_str, oid in zip(q_strs, oids):
            if oid not in org_to_click_embs:
                click_max_list.append(0.0)
                click_min_list.append(0.0)
            else:
                q_emb = query_to_emb.get(q_str, zero_emb)
                sims = [_cosine_sim(q_emb, ce) for ce in org_to_click_embs[oid]]
                click_max_list.append(float(max(sims)))
                click_min_list.append(float(min(sims)))

    # bert_names_max_sim, bert_names_min_sim: query vs все названия org из org_info
    n = len(df)
    names_max_list = [0.0] * n
    names_min_list = [0.0] * n
    if org_info is not None:
        org_ids = df["org_id"].unique()
        org_to_names = {str(oid): _get_org_names(org_info, oid) for oid in org_ids}
        unique_names = list({n for names in org_to_names.values() for n in names if n})
        name_to_emb = {}
        if unique_names:
            prefixed_names = ["passage: " + (nm[:500] if len(nm) > 500 else nm) for nm in unique_names]
            name_embs = model.encode(
                prefixed_names, batch_size=eff_batch, show_progress_bar=True,
                normalize_embeddings=True,
            )
            name_to_emb = dict(zip(unique_names, name_embs))

        q_embs_list = q_embs  # уже закодировали выше
        oids_list = df["org_id"].astype(str).tolist()
        for i in range(n):
            names = org_to_names.get(oids_list[i], [])
            if not names:
                continue
            name_embeddings = [name_to_emb[n] for n in names if n in name_to_emb]
            if not name_embeddings:
                continue
            sims = [_cosine_sim(q_embs_list[i], ne) for ne in name_embeddings]
            names_max_list[i] = float(max(sims))
            names_min_list[i] = float(min(sims))

    return bert_cosine, click_max_list, click_min_list, names_max_list, names_min_list
