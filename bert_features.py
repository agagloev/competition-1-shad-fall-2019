"""
BERT-эмбеддинги для фичей query vs org_name.
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
):
    """
    Строит BERT-фичу: cosine sim между эмбеддингами query и org_name.
    E5 использует "query: " / "passage: " префиксы для retrieval.
    Возвращает список bert_cosine (значения в [-1, 1], обычно в [0.7, 0.95]).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")

    queries_raw = df["query"].fillna("").astype(str).tolist()
    org_names_raw = df["org_name"].fillna("").astype(str).tolist()

    queries = ["query: " + (q[:500] if len(q) > 500 else q) for q in queries_raw]
    org_names = ["passage: " + (o[:500] if len(o) > 500 else o) for o in org_names_raw]

    q_embs = model.encode(queries, batch_size=batch_size, show_progress_bar=True)
    o_embs = model.encode(org_names, batch_size=batch_size, show_progress_bar=True)

    bert_cosine = [_cosine_sim(q, o) for q, o in zip(q_embs, o_embs)]
    return bert_cosine
